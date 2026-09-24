document.addEventListener("DOMContentLoaded", () => {
  loadRepoStats();
});

async function loadRepoStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const data = await res.json();

    const nameEl = document.getElementById("headerRepoName");
    const chunksEl = document.getElementById("headerTotalChunks");
    const codeEl = document.getElementById("statCodeChunks");
    const docEl = document.getElementById("statDocChunks");
    const ticketEl = document.getElementById("statTicketChunks");

    if (nameEl) nameEl.textContent = `${data.repo_name} (${data.repo_version})`;
    if (chunksEl) chunksEl.textContent = `${data.total_chunks.toLocaleString()} chunks`;
    if (codeEl) codeEl.textContent = data.code_chunks.toLocaleString();
    if (docEl) docEl.textContent = data.doc_chunks.toLocaleString();
    if (ticketEl) ticketEl.textContent = data.ticket_chunks.toLocaleString();
  } catch (err) {
    console.warn("Could not fetch repo stats:", err);
  }
}

function selectQuery(questionText) {
  const input = document.getElementById("queryInput");
  if (!input) return;
  input.value = questionText;
  input.focus();
  document.getElementById("queryForm").dispatchEvent(new Event("submit", { cancelable: true }));
}

async function submitQuery(e) {
  if (e) e.preventDefault();

  const input = document.getElementById("queryInput");
  const query = input.value.trim();
  if (!query) return;

  const chatMessages = document.getElementById("chatMessages");
  const welcomeHero = document.getElementById("welcomeHero");
  if (welcomeHero) welcomeHero.style.display = "none";

  appendUserMessage(query);
  input.value = "";

  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  if (statusIndicator) statusIndicator.style.display = "flex";
  if (statusText) statusText.textContent = "Searching ChromaDB vectors (1,562 chunks)...";

  const sendBtn = document.getElementById("sendBtn");
  if (sendBtn) sendBtn.disabled = true;
  input.disabled = true;

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 5 }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Query failed to execute.");
    }

    const data = await response.json();

    const retEl = document.getElementById("lastRetrievalMs");
    const genEl = document.getElementById("lastGenerationMs");
    const totEl = document.getElementById("lastTotalMs");
    if (retEl) retEl.textContent = `${data.retrieval_latency_ms.toFixed(1)} ms`;
    if (genEl) genEl.textContent = `${data.generation_latency_ms.toFixed(1)} ms`;
    if (totEl) totEl.textContent = `${data.total_latency_ms.toFixed(1)} ms`;

    appendAIMessage(data);

  } catch (err) {
    appendErrorMessage(err.message);
  } finally {
    if (statusIndicator) statusIndicator.style.display = "none";
    if (sendBtn) sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

function appendUserMessage(text) {
  const chatMessages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "user-msg-row";
  div.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAIMessage(data) {
  const chatMessages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "ai-msg-row";

  let renderedAnswer = escapeHtml(data.answer).replace(/\n/g, "<br/>");

  let citationsHtml = "";
  if (data.citations && data.citations.length > 0) {
    const cards = data.citations.map((c, idx) => {
      const typeClass =
        c.doc_type === "code" ? "code" : c.doc_type === "documentation" ? "doc" : "ticket";
      const snippetId = `snip-${Date.now()}-${idx}`;

      return `
        <div class="citation-card">
          <button type="button" class="citation-btn" onclick="toggleCitation('${snippetId}')">
            <div class="citation-info">
              <span class="citation-type ${typeClass}">${c.doc_type}</span>
              <span class="citation-path">${escapeHtml(c.file_path)}</span>
              <span class="citation-lines">(L${c.start_line}-L${c.end_line})</span>
            </div>
            <span class="citation-view-toggle">View ▾</span>
          </button>
          <div id="${snippetId}" class="citation-snippet">
            <pre>${escapeHtml(c.snippet)}</pre>
          </div>
        </div>
      `;
    }).join("");

    citationsHtml = `
      <div class="citations-wrapper">
        <div class="citations-title">
          <span>Retrieved Evidence & Ground Truth (${data.citations.length} sources)</span>
        </div>
        ${cards}
      </div>
    `;
  }

  div.innerHTML = `
    <div class="ai-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
    </div>

    <div class="ai-bubble">
      <div class="ai-meta">
        <div>
          <strong style="color: #ffffff;">RepoLens Engine</strong> • <span>${data.system_version}</span>
        </div>
        <div class="ai-timing">
          ⚡ ${(data.total_latency_ms / 1000).toFixed(2)}s (Ret: ${data.retrieval_latency_ms.toFixed(0)}ms, Gen: ${data.generation_latency_ms.toFixed(0)}ms)
        </div>
      </div>

      <div class="markdown-body">
        ${renderedAnswer}
      </div>

      ${citationsHtml}
    </div>
  `;

  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendErrorMessage(message) {
  const chatMessages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "ai-msg-row";
  div.innerHTML = `
    <div class="ai-bubble" style="border-color: #f43f5e; background-color: rgba(244, 63, 94, 0.1); color: #fecdd3;">
      <strong>Error:</strong> ${escapeHtml(message)}
    </div>
  `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function toggleCitation(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("open");
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
