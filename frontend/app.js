let currentSystem = "c";

document.addEventListener("DOMContentLoaded", () => {
  loadRepoStats();
});

function switchSystem(sys) {
  currentSystem = sys;
  const btnA = document.getElementById("btnSystemA");
  const btnB = document.getElementById("btnSystemB");
  const btnC = document.getElementById("btnSystemC");
  const badge = document.getElementById("headerSystemBadge");

  [btnA, btnB, btnC].forEach((b) => b && b.classList.remove("active"));

  if (sys === "c") {
    if (btnC) btnC.classList.add("active");
    if (badge) {
      badge.textContent = "System C (Cascading Intent Router)";
      badge.style.color = "#c084fc";
      badge.style.borderColor = "rgba(192, 132, 252, 0.4)";
      badge.style.backgroundColor = "rgba(192, 132, 252, 0.1)";
    }
  } else if (sys === "b") {
    if (btnB) btnB.classList.add("active");
    if (badge) {
      badge.textContent = "System B (Hybrid BM25+RRF)";
      badge.style.color = "#4ade80";
      badge.style.borderColor = "rgba(74, 222, 128, 0.4)";
      badge.style.backgroundColor = "rgba(74, 222, 128, 0.1)";
    }
  } else {
    if (btnA) btnA.classList.add("active");
    if (badge) {
      badge.textContent = "System A (Baseline Dense)";
      badge.style.color = "#60a5fa";
      badge.style.borderColor = "rgba(59, 130, 246, 0.4)";
      badge.style.backgroundColor = "rgba(59, 130, 246, 0.1)";
    }
  }
}

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
  if (statusText) {
    if (currentSystem === "c") {
      statusText.textContent = "Routing query via Cascading Router (Heuristics -> Micro-LLM)...";
    } else if (currentSystem === "b") {
      statusText.textContent = "Running Hybrid Search (Chroma Dense + BM25 Sparse with RRF k=60)...";
    } else {
      statusText.textContent = "Running ChromaDB Dense Vector Search (cosine similarity)...";
    }
  }

  const sendBtn = document.getElementById("sendBtn");
  if (sendBtn) sendBtn.disabled = true;
  input.disabled = true;

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 5, system: currentSystem }),
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

function renderMarkdown(rawText) {
  if (!rawText) return "";

  const codeBlocks = [];
  let text = rawText.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre class="code-block"><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`);
    return placeholder;
  });

  text = escapeHtml(text);
  text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\n\n/g, "<br/><br/>").replace(/\n/g, "<br/>");

  codeBlocks.forEach((block, idx) => {
    text = text.replace(`__CODE_BLOCK_${idx}__`, block);
  });

  return text;
}

function appendAIMessage(data) {
  const chatMessages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "ai-msg-row";

  let renderedAnswer = renderMarkdown(data.answer);

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

  const isSystemC = (data.system_version || "").includes("System C");
  const isSystemB = (data.system_version || "").includes("System B");
  const badgeColor = isSystemC ? "#c084fc" : isSystemB ? "#4ade80" : "#60a5fa";

  const routingTag = data.intent
    ? `<span style="font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); font-weight: 600; margin-left: 6px;">🎯 ${data.intent} (${data.route_source || 'routed'})</span>`
    : "";

  div.innerHTML = `
    <div class="ai-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${badgeColor}" stroke-width="2">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
    </div>

    <div class="ai-bubble">
      <div class="ai-meta">
        <div>
          <strong style="color: #ffffff;">RepoLens Engine</strong> • <span style="color: ${badgeColor}; font-weight: 500;">${data.system_version}</span>${routingTag}
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
