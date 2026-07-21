import base64
import logging
from io import BytesIO

logger = logging.getLogger("api")


def normalize_images(content: list) -> list:
  """Re-encode image_url blocks as PNG so vLLM/PIL can always open them.
  Raises ValueError with a user-friendly message if the format is unsupported."""
  try:
    import pillow_heif
    pillow_heif.register_heif_opener()
  except ImportError:
    pass

  from PIL import Image

  result = []
  for part in content:
    if not isinstance(part, dict) or part.get("type") != "image_url":
      result.append(part)
      continue
    url = (part.get("image_url") or {}).get("url", "")
    if not url.startswith("data:"):
      result.append(part)
      continue
    try:
      mime = url.split(";")[0].split(":")[1] if ":" in url else ""
      _, b64data = url.split(",", 1)
      raw = base64.b64decode(b64data)

      if mime == "image/svg+xml" or raw.lstrip()[:5] in (b"<svg ", b"<?xml"):
        raise ValueError(
          "Le immagini SVG non sono supportate. "
          "Converti l'immagine in PNG o JPEG prima di inviarla."
        )

      img = Image.open(BytesIO(raw))
      buf = BytesIO()
      img.save(buf, format="PNG")
      new_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
      result.append({**part, "image_url": {**(part.get("image_url") or {}), "url": new_url}})
    except ValueError:
      raise
    except Exception as exc:
      logger.error("Image normalization failed: %s", exc)
      raise ValueError(
        "Formato immagine non supportato. "
        "Converti l'immagine in PNG o JPEG prima di inviarla. "
        f"(dettaglio: {exc})"
      )
  return result
