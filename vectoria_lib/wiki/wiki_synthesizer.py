import logging
from datetime import datetime

from vectoria_lib.core.config_schema import WikiConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME
from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache
from vectoria_lib.wiki.wiki_storage import WikiStorage

logger = logging.getLogger("wiki")


class WikiSynthesizer:
  def __init__(
    self,
    storage: WikiStorage,
    emb_cache: WikiEmbeddingCache,
    config: WikiConfig,
  ):
    self._storage = storage
    self._emb_cache = emb_cache
    self._config = config

  def synthesize(self, llm, retriever, pending: list[dict], max_per_doc: int | None = None) -> list[str]:
    if not pending:
      return []

    max_per_doc = max_per_doc or self._config.max_topics_per_doc

    by_source: dict[str, list[str]] = {}
    for p in pending:
      by_source.setdefault(p.get("source", ""), []).append(p["snippet"])

    all_topics: list[tuple[str, str]] = []
    for source, snippets in by_source.items():
      step = max(1, len(snippets) // 50)
      doc_snippets = snippets[::step][:50]
      snippets_text = "\n".join(f"- {s}" for s in doc_snippets)

      topic_prompt = self._build_topic_prompt(snippets_text, max_per_doc)
      try:
        resp = llm.invoke(topic_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        topics = [t.strip().lstrip("-•*0123456789. ") for t in raw.splitlines() if t.strip()]
        topics = [t for t in topics if len(t) >= 3][:max_per_doc]
        all_topics.extend((t, source) for t in topics)
        logger.info("Doc '%s' → %d topics: %s", source, len(topics), topics)
      except Exception as exc:
        logger.warning("Topic discovery failed for '%s': %s", source, exc)

    if not all_topics:
      logger.warning("LLM returned no topics from any document")
      return []

    canonical_topics = self._dedupe_topics(all_topics)

    if self._config.max_topics_per_batch and len(canonical_topics) > self._config.max_topics_per_batch:
      dropped = len(canonical_topics) - self._config.max_topics_per_batch
      canonical_topics = canonical_topics[: self._config.max_topics_per_batch]
      logger.info("Wiki: capped batch to %d topics (%d dropped)", self._config.max_topics_per_batch, dropped)

    created: list[str] = []
    sources = self._storage.load_sources()
    for topic, source in canonical_topics:
      try:
        wrote = self._synthesize_page({"name": topic, "source": source, "snippet": ""}, llm, retriever)
        if not wrote:
          continue
        created.append(topic)
        stem = self._storage.safe_stem(topic)
        prev = sources.get(stem, "")
        if source and source not in prev.split(", "):
          sources[stem] = f"{prev}, {source}".lstrip(", ")
      except Exception as exc:
        logger.warning("Wiki synthesis failed for '%s': %s", topic, exc)

    if sources:
      self._storage.save_sources(sources)

    if created:
      embed = WikiEmbeddingCache.get_embedder()
      if embed is not None:
        stems = [self._storage.safe_stem(t) for t in created]
        cache = self._emb_cache.load()
        cache.update(zip(stems, embed.embed_documents(stems), strict=True))
        self._emb_cache.save(cache)

    return created

  def _dedupe_topics(self, topics: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not self._config.language_normalization or not topics:
      return topics

    embed = WikiEmbeddingCache.get_embedder()
    if embed is None:
      logger.debug("Embedder unavailable; skipping cross-language dedup")
      return topics

    existing_pages = self._storage.list_pages()
    canonical_names: list[str] = list(existing_pages)
    canonical_embs: list[list[float]] = self._emb_cache.load_or_compute(embed, existing_pages)

    topic_names = [t for t, _ in topics]
    topic_embs = embed.embed_documents(topic_names) if topic_names else []

    output: list[tuple[str, str]] = []
    for (topic, source), emb in zip(topics, topic_embs, strict=True):
      best_i, best_sim = WikiEmbeddingCache.best_match(emb, canonical_embs)
      if best_i is not None and best_sim >= self._config.similarity_threshold:
        canonical = canonical_names[best_i]
        if canonical != topic:
          logger.info(
            "Topic dedup: '%s' (source=%s) → merged into '%s' (sim=%.3f)",
            topic, source, canonical, best_sim,
          )
        output.append((canonical, source))
      else:
        output.append((topic, source))
        canonical_names.append(topic)
        canonical_embs.append(emb)
    return output

  def _synthesize_page(self, concept: dict, llm, retriever) -> bool:
    try:
      chunks = retriever.invoke(concept["name"])
    except Exception:
      chunks = []

    chunk_count = max(1, self._config.synthesis_context_chunks)
    context_chunks = chunks[:chunk_count] if chunks else []

    if not context_chunks and not concept.get("snippet"):
      logger.warning("Wiki: skip '%s' — no retrievable context (quality gate)", concept["name"])
      return False

    context = (
      "\n\n---\n\n".join(c.page_content for c in context_chunks)
      if context_chunks
      else concept.get("snippet", "")
    )

    is_italian = self._config.canonical_topic_language != "en"
    prompt = (
      self._italian_synthesis_prompt(concept["name"], context)
      if is_italian
      else self._english_synthesis_prompt(concept["name"], context)
    )

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    content = self._strip_leading_title(content, concept["name"])

    chunk_sources = sorted(
      {c.metadata.get(DOC_FILE_NAME, "") for c in context_chunks if getattr(c, "metadata", None)}
      | ({concept.get("source", "")} if concept.get("source") else set())
    )
    chunk_sources = [s for s in chunk_sources if s]

    page_text = self._render_page(concept["name"], content, chunk_sources)
    self._storage.write_page(concept["name"], page_text)
    logger.debug("Wiki page written: %s", self._storage.page_path(concept["name"]).name)
    return True

  def _build_topic_prompt(self, snippets_text: str, max_per_doc: int) -> str:
    if self._config.language_normalization and self._config.canonical_topic_language == "en":
      return (
        "Analyze the following content extracted from a document and identify "
        "ONLY its truly CENTRAL topics — the theses that the whole document "
        "is built around. Ignore minor details, passing examples, and "
        "anecdotal citations.\n\n"
        f"{snippets_text}\n\n"
        f"List AT MOST {max_per_doc} topics. Return FEWER lines (even 1 or 0) "
        "if the material does not contain that many central topics: do NOT pad "
        "the list with secondary themes just to reach the maximum.\n\n"
        "RULES:\n"
        "- Each topic: SHORT NOUN PHRASE IN ENGLISH (1–4 words), canonical "
        "  scientific/technical term, even if the source is in another language.\n"
        "- One topic per line. No numbering, no symbols, no trailing punctuation, "
        "  no quotes, no preamble."
      )
    return (
      "Analizza il seguente contenuto estratto da un documento e identifica SOLO "
      "gli argomenti VERAMENTE CENTRALI — le tesi attorno a cui ruota l'intero "
      "documento. Ignora dettagli marginali, esempi di passaggio, citazioni "
      "episodiche e temi accessori.\n\n"
      f"{snippets_text}\n\n"
      f"Elenca AL MASSIMO {max_per_doc} argomenti. Restituisci MENO righe "
      "(anche 1 o 0) se il materiale non contiene così tanti argomenti centrali: "
      "NON riempire la lista con temi secondari pur di arrivare al massimo.\n\n"
      "REGOLE STRINGENTI:\n"
      "- Sintagma nominale in ITALIANO (1–4 parole) per ciascun argomento.\n"
      "- Usa il TERMINE TECNICO CANONICO italiano (es. 'Mitocondri' non 'organelli "
      "  che producono energia'; 'Apprendimento per rinforzo' non 'metodi di "
      "  addestramento'). Se il termine canonico non esiste in italiano, usa il "
      "  prestito stabilito (es. 'Machine learning', 'Transformer').\n"
      "- Maiuscola solo all'inizio. Niente articoli iniziali ('La gravità' → 'Gravità').\n"
      "- Niente verbi, niente frasi descrittive ('Come funziona X' è SBAGLIATO).\n"
      "- Argomenti distinti, di pari livello di astrazione.\n"
      "- Un argomento per riga. Niente numerazione, trattini, punteggiatura finale, "
      "  virgolette o preambolo."
    )

  def _italian_synthesis_prompt(self, topic: str, context: str) -> str:
    return (
      f"Scrivi una pagina enciclopedica completa sull'argomento: **{topic}**.\n\n"
      f"=== Materiale di riferimento (estratto dai documenti) ===\n{context}\n"
      f"=== fine materiale ===\n\n"
      "Struttura OBBLIGATORIA in Markdown (rispetta esattamente i titoli):\n\n"
      "## Definizione\n"
      "Un paragrafo di 3–5 frasi che definisce l'argomento in modo preciso, "
      "tecnico e autocontenuto. Specifica il dominio di appartenenza.\n\n"
      "## Caratteristiche principali\n"
      "Lista di 4–7 punti, ciascuno una frase intera che esprime un aspetto "
      "rilevante (proprietà, meccanismo, regola, parametro chiave).\n\n"
      "## Approfondimento\n"
      "Uno o due paragrafi che descrivono il funzionamento, la classificazione, "
      "le varianti o il contesto storico/teorico. Vai oltre la definizione: "
      "spiega il PERCHÉ e il COME, non solo il COSA.\n\n"
      "## Esempi e applicazioni\n"
      "2–4 esempi concreti citati nel materiale o riconducibili allo stesso "
      "ambito. Se non ci sono esempi pertinenti, scrivi 'Non disponibili nei "
      "documenti analizzati' e ometti la sezione.\n\n"
      "## Concetti correlati\n"
      "Lista di 3–6 concetti collegati, ognuno espresso come [[wikilink]] "
      f"con il nome canonico italiano (es. [[Apprendimento supervisionato]]).\n\n"
      "REGOLE:\n"
      f"- Lunghezza massima: ~{self._config.max_page_words} parole.\n"
      "- Tono enciclopedico, terza persona, niente formule colloquiali.\n"
      "- Usa SOLO informazioni desumibili dal materiale di riferimento. Non "
      "  inventare dati specifici, numeri, autori o date.\n"
      "- Niente preambolo né titolo H1 (verrà aggiunto automaticamente).\n"
      "- Niente frasi del tipo 'in questo documento', 'l'autore afferma'."
    )

  def _english_synthesis_prompt(self, topic: str, context: str) -> str:
    return (
      f"Write a comprehensive encyclopedic wiki page about: **{topic}**.\n\n"
      f"=== Reference material (extracted from the documents) ===\n{context}\n"
      f"=== end of material ===\n\n"
      "Required Markdown structure (use exactly these headings):\n\n"
      "## Definition\n"
      "3–5 sentences defining the topic precisely, technically, self-contained.\n\n"
      "## Key characteristics\n"
      "4–7 bullets, each a full sentence about a relevant aspect.\n\n"
      "## In depth\n"
      "1–2 paragraphs describing how it works, its variants or theoretical "
      "background. Explain the WHY and HOW, not just the WHAT.\n\n"
      "## Examples and applications\n"
      "2–4 concrete examples grounded in the material. Skip the section if "
      "none are available.\n\n"
      "## Related concepts\n"
      "3–6 [[wikilinks]] to canonical-named related concepts.\n\n"
      "RULES:\n"
      f"- Max length: ~{self._config.max_page_words} words.\n"
      "- Encyclopedic tone, third person.\n"
      "- Use ONLY information supported by the reference material.\n"
      "- No preamble, no H1 title (added automatically)."
    )

  def _render_page(self, topic: str, body: str, sources: list[str]) -> str:
    today = datetime.now().date().isoformat()
    fm_sources = ", ".join(f'"{s}"' for s in sources) if sources else ""
    frontmatter = (
      "---\n"
      f"title: {topic}\n"
      f"created: {today}\n"
      f"sources: [{fm_sources}]\n"
      f"language: {self._config.canonical_topic_language}\n"
      "tags: [wiki, vectoria]\n"
      "---\n\n"
    )
    sources_section = ""
    if sources:
      bullets = "\n".join(f"- {s}" for s in sources)
      sources_section = f"\n\n## Fonti\n{bullets}\n"
    return f"{frontmatter}# {topic}\n\n{body.strip()}{sources_section}\n"

  @staticmethod
  def _strip_leading_title(content: str, topic: str) -> str:
    lines = content.lstrip().splitlines()
    if not lines:
      return content
    first = lines[0].strip().lstrip("#").strip().lower()
    if first == topic.lower() or first.startswith(topic.lower() + " "):
      return "\n".join(lines[1:]).lstrip()
    return content
