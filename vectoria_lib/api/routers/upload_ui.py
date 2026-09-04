#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api")

router = APIRouter(tags=["upload-ui"])

_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Vectoria — Carica Documenti</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f0f0f;
      color: #e5e5e5;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .card {
      background: #1a1a1a;
      border: 1px solid #2e2e2e;
      border-radius: 12px;
      padding: 40px;
      width: 100%;
      max-width: 560px;
    }

    h1 {
      font-size: 1.4rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: #fff;
    }

    .subtitle {
      font-size: 0.875rem;
      color: #888;
      margin-bottom: 28px;
    }

    .tenant-row {
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      align-items: center;
    }

    .tenant-row label {
      font-size: 0.85rem;
      color: #aaa;
      white-space: nowrap;
    }

    .tenant-row input {
      flex: 1;
      background: #111;
      border: 1px solid #333;
      border-radius: 6px;
      color: #e5e5e5;
      padding: 8px 12px;
      font-size: 0.875rem;
      outline: none;
    }

    .tenant-row input:focus { border-color: #555; }

    .dropzone {
      border: 2px dashed #333;
      border-radius: 10px;
      padding: 48px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 20px;
    }

    .dropzone.over {
      border-color: #6366f1;
      background: #1e1e30;
    }

    .dropzone .icon { font-size: 2.5rem; margin-bottom: 12px; }

    .dropzone p { font-size: 0.9rem; color: #aaa; }
    .dropzone p span { color: #6366f1; text-decoration: underline; cursor: pointer; }

    #file-input { display: none; }

    .file-list {
      list-style: none;
      margin-bottom: 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .file-list li {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 0.85rem;
    }

    .file-list li .name { color: #ccc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 360px; }
    .file-list li .remove { color: #666; cursor: pointer; font-size: 1rem; flex-shrink: 0; }
    .file-list li .remove:hover { color: #e55; }

    .btn {
      width: 100%;
      padding: 12px;
      background: #6366f1;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    .btn:hover { background: #4f51d8; }
    .btn:disabled { background: #333; color: #666; cursor: not-allowed; }

    .btn-cancel {
      width: 100%;
      margin-top: 10px;
      padding: 10px;
      background: transparent;
      color: #e07070;
      border: 1px solid #5a2d2d;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
      display: none;
    }

    .btn-cancel:hover { background: #2e1a1a; }

    .status {
      margin-top: 20px;
      padding: 14px;
      border-radius: 8px;
      font-size: 0.875rem;
      display: none;
    }

    .status.info    { background: #1e2a3a; border: 1px solid #2d4a6a; color: #7ab4e8; display: block; }
    .status.success { background: #1a2e1a; border: 1px solid #2d5a2d; color: #6dbf6d; display: block; }
    .status.error   { background: #2e1a1a; border: 1px solid #5a2d2d; color: #e07070; display: block; }

    .progress {
      margin-top: 10px;
      display: none;
    }

    .progress-track {
      height: 6px;
      background: #2e2e2e;
      border-radius: 3px;
      overflow: hidden;
    }

    .progress-bar {
      height: 100%;
      background: #6366f1;
      width: 0%;
      transition: width 0.2s;
      border-radius: 3px;
    }

    .progress-bar.indeterminate {
      width: 40%;
      animation: indeterminate 1.2s infinite ease-in-out;
    }

    .progress-label {
      font-size: 0.78rem;
      color: #888;
      margin-top: 5px;
      text-align: right;
    }

    @keyframes indeterminate {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(350%); }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Carica Documenti</h1>
    <p class="subtitle">Carica uno o più file PDF o DOCX per indicizzarli in Vectoria.</p>

    <div class="tenant-row">
      <label for="tenant-input">Tenant ID</label>
      <input id="tenant-input" type="text" value="default" placeholder="default" />
    </div>

    <div class="dropzone" id="dropzone">
      <div class="icon">📂</div>
      <p>Trascina i file qui oppure <span onclick="document.getElementById('file-input').click()">sfoglia</span></p>
      <p style="margin-top:6px;font-size:0.8rem;color:#555">PDF, DOCX, XLSX, PPTX, HTML, MD, TXT, CSV supportati</p>
    </div>
    <input type="file" id="file-input" multiple accept=".pdf,.docx,.xlsx,.pptx,.html,.htm,.md,.markdown,.txt,.csv" />

    <ul class="file-list" id="file-list"></ul>

    <button class="btn" id="upload-btn" disabled>Carica e Indicizza</button>
    <button class="btn-cancel" id="cancel-btn">✕ Annulla indicizzazione</button>

    <div class="progress" id="progress">
      <div class="progress-track">
        <div class="progress-bar" id="progress-bar"></div>
      </div>
      <div class="progress-label" id="progress-label"></div>
    </div>

    <div class="status" id="status"></div>
  </div>

  <script>
    const dropzone   = document.getElementById('dropzone');
    const fileInput  = document.getElementById('file-input');
    const fileList   = document.getElementById('file-list');
    const uploadBtn  = document.getElementById('upload-btn');
    const cancelBtn  = document.getElementById('cancel-btn');
    const statusBox  = document.getElementById('status');
    const progressEl = document.getElementById('progress');
    const progressBar = document.getElementById('progress-bar');
    const progressLabel = document.getElementById('progress-label');
    const tenantEl   = document.getElementById('tenant-input');

    let selectedFiles = [];
    let currentTenantId = null;
    let polling = false;

    function setProgress(pct, label) {
      progressEl.style.display = 'block';
      progressBar.classList.remove('indeterminate');
      progressBar.style.width = pct + '%';
      progressLabel.textContent = label || '';
    }

    function setProgressIndeterminate(label) {
      progressEl.style.display = 'block';
      progressBar.classList.add('indeterminate');
      progressBar.style.width = '40%';
      progressLabel.textContent = label || '';
    }

    function hideProgress() {
      progressEl.style.display = 'none';
      progressBar.style.width = '0%';
      progressLabel.textContent = '';
    }

    function showCancelBtn(show) {
      cancelBtn.style.display = show ? 'block' : 'none';
    }

    function renderList() {
      fileList.innerHTML = '';
      selectedFiles.forEach((f, i) => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="name">${f.name}</span><span class="remove" data-i="${i}">✕</span>`;
        fileList.appendChild(li);
      });
      uploadBtn.disabled = selectedFiles.length === 0;
      fileList.querySelectorAll('.remove').forEach(btn => {
        btn.addEventListener('click', () => {
          selectedFiles.splice(+btn.dataset.i, 1);
          renderList();
        });
      });
    }

    fileInput.addEventListener('change', () => {
      selectedFiles = [...selectedFiles, ...Array.from(fileInput.files)];
      fileInput.value = '';
      renderList();
    });

    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
    dropzone.addEventListener('drop', e => {
      e.preventDefault();
      dropzone.classList.remove('over');
      selectedFiles = [...selectedFiles, ...Array.from(e.dataTransfer.files)];
      renderList();
    });

    function setStatus(msg, type) {
      statusBox.textContent = msg;
      statusBox.className = 'status ' + type;
    }

    async function pollStatus(tenantId) {
      const url = `/documents/status`;
      polling = true;
      while (polling) {
        await new Promise(r => setTimeout(r, 1500));
        try {
          const res = await fetch(url, { headers: { 'X-Tenant-ID': tenantId } });
          const data = await res.json();
          if (data.progress) {
            const pct = data.progress.pct || 0;
            const step = data.progress.step || 'Indicizzazione...';
            setProgress(pct, `${step} (${pct}%)`);
          }
          if (data.status === 'ready')     { polling = false; return 'ready'; }
          if (data.status === 'error')     { polling = false; return 'error'; }
          if (data.status === 'cancelled') { polling = false; return 'cancelled'; }
        } catch (_) {}
      }
      return 'cancelled';
    }

    function uploadWithProgress(fd, tenantId) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/documents/ingest');
        xhr.setRequestHeader('X-Tenant-ID', tenantId);

        xhr.upload.addEventListener('progress', e => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            setProgress(pct, `Caricamento... ${pct}%`);
          }
        });

        xhr.upload.addEventListener('load', () => {
          setProgress(100, 'Caricamento completato. Indicizzazione in corso...');
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else if (xhr.status === 409) {
            let msg = 'Indicizzazione già in corso.';
            try { msg = JSON.parse(xhr.responseText).detail; } catch (_) {}
            reject({ detail: msg, retryable: true });
          } else {
            try {
              reject(JSON.parse(xhr.responseText));
            } catch (_) {
              reject({ detail: xhr.statusText });
            }
          }
        });

        xhr.addEventListener('error', () => reject({ detail: 'Errore di rete' }));
        xhr.send(fd);
      });
    }

    cancelBtn.addEventListener('click', async () => {
      if (!currentTenantId) return;
      polling = false;
      try {
        await fetch('/documents/cancel', {
          method: 'POST',
          headers: { 'X-Tenant-ID': currentTenantId },
        });
      } catch (_) {}
      showCancelBtn(false);
      hideProgress();
      setStatus('Indicizzazione annullata.', 'error');
      uploadBtn.disabled = selectedFiles.length === 0;
    });

    uploadBtn.addEventListener('click', async () => {
      if (selectedFiles.length === 0) return;

      currentTenantId = tenantEl.value.trim() || 'default';
      const fd = new FormData();
      selectedFiles.forEach(f => fd.append('files', f));

      uploadBtn.disabled = true;
      showCancelBtn(false);
      setStatus('Caricamento in corso...', 'info');

      try {
        await uploadWithProgress(fd, currentTenantId);
        setStatus('File caricati. Indicizzazione in corso...', 'info');
        setProgress(0, 'Avvio indicizzazione...');
        showCancelBtn(true);

        const result = await pollStatus(currentTenantId);
        showCancelBtn(false);
        hideProgress();

        if (result === 'ready') {
          const modelName = `vectoria-${currentTenantId}`;
          setStatus(`✓ Indicizzazione completata. Torna su OpenWebUI e seleziona il modello "${modelName}" per iniziare.`, 'success');
          selectedFiles = [];
          renderList();
        } else if (result === 'cancelled') {
          setStatus('Indicizzazione annullata.', 'error');
        } else {
          setStatus("Errore durante l'indicizzazione. Controlla i log del server.", 'error');
        }
      } catch (err) {
        showCancelBtn(false);
        hideProgress();
        setStatus('Errore: ' + (err.detail || JSON.stringify(err)), 'error');
      }

      uploadBtn.disabled = selectedFiles.length === 0;
    });
  </script>
</body>
</html>
"""


@router.get("/upload", response_class=HTMLResponse)
async def upload_ui():
    """Simple drag-and-drop upload page for users."""
    return HTMLResponse(content=_HTML)
