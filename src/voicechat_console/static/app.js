const runtimeGrid = document.getElementById("runtimeGrid");
const eventsTable = document.getElementById("eventsTable");
const configEditor = document.getElementById("configEditor");
const commandResult = document.getElementById("commandResult");
const configResult = document.getElementById("configResult");
const catalogResult = document.getElementById("catalogResult");
const eventType = document.getElementById("eventType");
const commandCatalog = document.getElementById("commandCatalog");
const heroRunning = document.getElementById("heroRunning");
const heroState = document.getElementById("heroState");
const heroMode = document.getElementById("heroMode");

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = data.reply || data.error || data.message || `request failed: ${response.status}`;
    const error = new Error(detail);
    error.payload = data;
    throw error;
  }
  return data;
}

function renderRuntime(state) {
  const runtime = state.state || {};
  heroRunning.textContent = state.running ? "稼働中" : "停止中";
  heroState.textContent = runtime.state || "-";
  heroMode.textContent = runtime.recognition_mode_label || runtime.recognition_mode || "-";
  const pairs = [
    ["稼働", state.running ? "yes" : "no"],
    ["状態", runtime.state || "-"],
    ["PID", runtime.pid || "-"],
    ["会話モード", runtime.active_mode || "-"],
    ["認識 backend", runtime.transcription_backend || "-"],
    ["認識モード", runtime.recognition_mode_label || runtime.recognition_mode || "-"],
    ["wake word", runtime.wake_word || "-"],
    ["LLM", [runtime.llm_provider, runtime.llm_model].filter(Boolean).join(" / ") || "-"],
    ["直近 fast", runtime.recognized_fast || "-"],
    ["直近 final", runtime.recognized_final || "-"],
    ["最後の reply", runtime.last_reply || "-"],
    ["更新時刻", runtime.updated_at || "-"],
  ];
  runtimeGrid.innerHTML = pairs
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value || ""))}</dd>`)
    .join("");
}

function renderEvents(items) {
  if (!items.length) {
    eventsTable.innerHTML = `<div class="row">no events</div>`;
    return;
  }
  eventsTable.innerHTML = items
    .map((item) => {
      const payload = item.payload || {};
      const title = payload.command_id || item.event_type;
      const body = payload.command_input_text || item.recognized || payload.recognized_final || "";
      const meta = [
        item.date,
        payload.source || "voice",
        payload.command_ok === true ? "success" : payload.command_ok === false ? "failed" : "",
      ]
        .filter(Boolean)
        .join(" / ");
      const statusClass = payload.command_ok === true ? "ok" : payload.command_ok === false ? "ng" : "";
      const statusLabel = payload.command_ok === true ? "OK" : payload.command_ok === false ? "NG" : "INFO";
      const metaLine = [
        `source=${payload.source || "voice"}`,
        `returncode=${payload.command_returncode ?? "-"}`,
      ]
        .filter(Boolean)
        .join(" / ");
      const detail = {
        reply: payload.command_reply || "",
        stdout: payload.command_stdout || "",
        stderr: payload.command_stderr || "",
        recognized_final: payload.recognized_final || "",
        action: payload.action_name || "",
      };
      return `
        <article class="row">
          <button class="row-toggle" type="button">
            <div class="row-head">
              <strong>${escapeHtml(title)}</strong>
              <span class="pill ${statusClass}">${escapeHtml(statusLabel)}</span>
              <span class="pill">${escapeHtml(item.event_type)}</span>
            </div>
            <span class="summary-line">${escapeHtml(body || "(empty)")}</span>
            <span class="meta">${escapeHtml(meta)}</span>
            <span class="meta">${escapeHtml(metaLine)}</span>
          </button>
          <pre class="detail" hidden>${escapeHtml(JSON.stringify(detail, null, 2))}</pre>
        </article>
      `;
    })
    .join("");

  eventsTable.querySelectorAll(".row-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const detail = button.parentElement.querySelector(".detail");
      detail.hidden = !detail.hidden;
    });
  });
}

function renderCatalog(items) {
  if (!items.length) {
    commandCatalog.innerHTML = `<div class="row">no commands</div>`;
    return;
  }
  commandCatalog.innerHTML = items
    .map((item) => `
      <article class="row">
        <div class="row-head">
          <strong>${escapeHtml(item.id || "")}</strong>
          <span class="pill">${escapeHtml(item.action_name || "")}</span>
        </div>
        <span class="summary-line">${escapeHtml(item.help_label || "")}</span>
        <span class="meta">action=${escapeHtml(item.action_name || "")} / type=${escapeHtml(item.action_type || "")}</span>
        <span class="meta">phrases: ${escapeHtml((item.phrases || []).join(", "))}</span>
        ${item.reply ? `<span class="meta">reply: ${escapeHtml(item.reply)}</span>` : ""}
      </article>
    `)
    .join("");
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function loadRuntime() {
  renderRuntime(await fetchJson("/api/runtime"));
}

async function loadEvents() {
  const type = eventType.value;
  const query = new URLSearchParams({ limit: "30" });
  if (type) {
    query.set("type", type);
  }
  const data = await fetchJson(`/api/events?${query.toString()}`);
  renderEvents(data.items || []);
}

async function loadConfig() {
  const data = await fetchJson("/api/config");
  configEditor.value = data.yaml_text || "";
}

async function loadCatalog() {
  const data = await fetchJson("/api/commands/catalog");
  renderCatalog(data.items || []);
}

async function refreshAll() {
  await Promise.all([loadRuntime(), loadEvents(), loadCatalog()]);
}

document.getElementById("refreshButton").addEventListener("click", async () => {
  await refreshAll();
});

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => {
    document.getElementById("commandInput").value = button.dataset.command || "";
    document.getElementById("commandInput").focus();
  });
});

document.getElementById("reloadCatalogButton").addEventListener("click", async () => {
  await loadCatalog();
});

eventType.addEventListener("change", async () => {
  await loadEvents();
});

document.getElementById("commandForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = document.getElementById("commandInput").value.trim();
  if (!text) {
    return;
  }
  try {
    const result = await fetchJson("/api/commands/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    commandResult.textContent = JSON.stringify(result, null, 2);
    await refreshAll();
  } catch (error) {
    commandResult.textContent = JSON.stringify(error.payload || { error: String(error) }, null, 2);
  }
});

document.getElementById("saveConfigButton").addEventListener("click", async () => {
  try {
    const result = await fetchJson("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_text: configEditor.value }),
    });
    configResult.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    configResult.textContent = JSON.stringify(error.payload || { error: String(error) }, null, 2);
  }
});

document.getElementById("catalogForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  let args = {};
  const rawArgs = document.getElementById("catalogArgs").value.trim();
  if (rawArgs) {
    try {
      args = JSON.parse(rawArgs);
    } catch (error) {
      catalogResult.textContent = JSON.stringify({ error: `args JSON parse failed: ${error}` }, null, 2);
      return;
    }
  }
  try {
    const result = await fetchJson("/api/commands/catalog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: document.getElementById("catalogId").value,
        help_label: document.getElementById("catalogLabel").value,
        phrases: document.getElementById("catalogPhrases").value,
        action_name: document.getElementById("catalogActionName").value,
        reply: document.getElementById("catalogReply").value,
        args,
      }),
    });
    catalogResult.textContent = JSON.stringify(result, null, 2);
    await Promise.all([loadCatalog(), loadConfig()]);
  } catch (error) {
    catalogResult.textContent = JSON.stringify(error.payload || { error: String(error) }, null, 2);
  }
});

loadConfig();
refreshAll();
setInterval(refreshAll, 3000);
