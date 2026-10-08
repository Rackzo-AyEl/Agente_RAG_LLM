const quickQuestions = [
  "¿Cómo creo mi negocio o tienda?",
  "¿Por qué no aparece mi negocio en el mapa?",
  "¿Cómo agrego menú, catálogo u ofertas?",
  "¿Cómo escaneo un menú desde una fotografía?",
  "¿Qué beneficios tiene el plan Plus?",
  "¿Cómo publico una aventura o experiencia?",
  "¿Cómo contacto a soporte?"
];

const $ = (id) => document.getElementById(id);
let lastResponse = null;
let sessionId = localStorage.getItem("helpRagSessionId") || newSessionId();
localStorage.setItem("helpRagSessionId", sessionId);

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function apiBase() {
  return $("apiBase").value.trim().replace(/\/$/, "");
}

function endpoint(path) {
  return `${apiBase()}${path}`;
}

function setStatus(text) {
  $("status").textContent = text;
}

function stringify(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function renderValue(container, value) {
  container.textContent = "";

  if (!value || (Array.isArray(value) && value.length === 0)) {
    container.textContent = "Sin datos";
    return;
  }

  const list = document.createElement(Array.isArray(value) ? "ol" : "pre");

  if (!Array.isArray(value)) {
    list.textContent = typeof value === "object" ? stringify(value) : String(value);
    container.append(list);
    return;
  }

  for (const item of value) {
    const li = document.createElement("li");
    li.textContent = typeof item === "object" ? stringify(item) : String(item);
    list.append(li);
  }

  container.append(list);
}

function mediaElement(item) {
  const src = endpoint(item.url);
  if (item.type === "video") {
    const video = document.createElement("video");
    video.controls = true;
    video.src = src;
    video.preload = "metadata";
    return video;
  }
  const image = document.createElement("img");
  image.src = src;
  image.alt = item.title || "Material de ayuda";
  return image;
}

function renderMedia(container, media) {
  container.textContent = "";
  if (!media?.length) {
    container.textContent = "Sin material sugerido";
    return;
  }

  for (const item of media) {
    const card = document.createElement("article");
    const title = document.createElement("strong");
    const prompt = document.createElement("p");
    const button = document.createElement("button");
    const holder = document.createElement("div");
    card.className = "media-card";
    holder.className = "media-holder";
    title.textContent = item.title || item.filename || "Material de ayuda";
    prompt.textContent = item.prompt || `¿Te gustaría ver este ${item.type === "video" ? "video" : "archivo"} de ayuda?`;
    button.type = "button";
    button.textContent = item.type === "video" ? "Sí, ver video" : "Sí, ver imagen";
    button.addEventListener("click", () => {
      holder.textContent = "";
      holder.append(mediaElement(item));
      button.hidden = true;
    });
    card.append(title, prompt, button, holder);
    container.append(card);
    if (item.auto_open) button.click();
  }
}

function appendInlineMarkdown(container, text) {
  const parts = String(text).split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  for (const part of parts) {
    if (part.startsWith("`") && part.endsWith("`")) {
      const code = document.createElement("code");
      code.textContent = part.slice(1, -1);
      container.append(code);
    } else if (part.startsWith("**") && part.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      container.append(strong);
    } else {
      container.append(document.createTextNode(part));
    }
  }
}

function renderMarkdown(container, text) {
  container.textContent = "";
  const lines = String(text || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
  let list = null;
  let listType = null;

  function closeList() {
    if (list) container.append(list);
    list = null;
    listType = null;
  }

  for (const line of lines) {
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const type = numbered ? "ol" : "ul";
      if (!list || listType !== type) {
        closeList();
        list = document.createElement(type);
        listType = type;
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, (bullet || numbered)[1]);
      list.append(item);
      continue;
    }
    closeList();
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, line.replace(/^#{1,6}\s+/, ""));
    container.append(paragraph);
  }
  closeList();
}

function renderBlocks(container, blocks) {
  container.textContent = "";
  if (!blocks?.length) {
    container.textContent = "Sin bloques";
    return;
  }
  for (const block of blocks) {
    const card = document.createElement("article");
    card.className = "block";
    const label = document.createElement("strong");
    label.textContent = block.type || "block";
    card.append(label);
    if (block.type === "steps" && Array.isArray(block.items)) {
      const list = document.createElement("ol");
      block.items.forEach((item) => {
        const li = document.createElement("li");
        appendInlineMarkdown(li, item);
        list.append(li);
      });
      card.append(list);
    } else {
      const content = document.createElement("div");
      renderMarkdown(content, block.text || stringify(block));
      card.append(content);
    }
    container.append(card);
  }
}

function metaValue(metadata, keys, fallback = "No reportado") {
  for (const key of keys) {
    if (metadata?.[key] !== undefined && metadata[key] !== null && metadata[key] !== "") {
      return typeof metadata[key] === "object" ? stringify(metadata[key]) : metadata[key];
    }
  }
  return fallback;
}

function renderMetadataCards(metadata) {
  const cards = [
    ["Cache", metadata?.cache_hit ? "Hit" : "Miss"],
    ["Memoria", metaValue(metadata, ["memory_mb", "rss_mb", "memory", "memory_usage_mb"])],
    ["Embedding", metaValue(metadata, ["embed_model"])],
    ["LLM/Ollama", metadata?.provider ? `${metadata.provider}: ${metadata.model}` : metaValue(metadata, ["llm_skipped", "llm_error"])],
    ["Dispositivo embeddings", metaValue(metadata, ["device"])],
    ["Hardware LLM", metaValue(metadata, ["llm_device"], metadata?.provider === "ollama" ? "Ollama auto (GPU si disponible)" : "No usado")],
    ["Sesión", metaValue(metadata, ["session_turns"])],
    ["Resumen", metaValue(metadata, ["session_summary_chars"])],
    ["Memoria usada", metadata?.memory_used ? "Sí" : "No"],
    ["Índice", metaValue(metadata, ["index_version"])],
    ["Top K", metaValue(metadata, ["top_k"])]
  ];

  $("metadataCards").textContent = "";
  for (const [label, value] of cards) {
    const card = document.createElement("div");
    const name = document.createElement("span");
    const detail = document.createElement("strong");
    card.className = "meta-card";
    name.textContent = label;
    detail.textContent = String(value);
    card.append(name, detail);
    $("metadataCards").append(card);
  }
}

function renderAnswer(data) {
  lastResponse = data;
  $("answer").hidden = false;
  $("feedbackStatus").textContent = "";
  $("title").textContent = data.title || "Respuesta";
  $("summary").textContent = "";
  appendInlineMarkdown($("summary"), data.summary || "");
  renderBlocks($("blocks"), data.blocks);
  renderValue($("sources"), data.sources);
  renderMedia($("media"), data.media);
  renderValue($("nextActions"), data.next_actions);
  renderMetadataCards(data.metadata || {});
  $("metadata").textContent = stringify(data.metadata);
}

async function postJson(path, body) {
  const response = await fetch(endpoint(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function getJson(path) {
  const response = await fetch(endpoint(path));
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function renderCorpus(data) {
  const container = $("corpus");
  container.hidden = false;
  container.textContent = "";

  const summary = document.createElement("p");
  summary.textContent = `${data.doc_count} documentos, ${data.chunk_count} chunks, ${data.media_count} archivos multimedia.`;
  container.append(summary);

  for (const doc of data.docs || []) {
    const details = document.createElement("details");
    const title = document.createElement("summary");
    const meta = document.createElement("p");
    const list = document.createElement("ol");
    title.textContent = `${doc.title} (${doc.id})`;
    meta.className = "hint";
    meta.textContent = [doc.product_area, doc.url].filter(Boolean).join(" · ");
    for (const chunk of doc.chunks || []) {
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      const text = document.createElement("p");
      heading.textContent = chunk.heading;
      text.textContent = chunk.text;
      item.append(heading, text);
      list.append(item);
    }
    details.append(title, meta, list);
    container.append(details);
  }

  if (data.media?.length) {
    const mediaTitle = document.createElement("h3");
    const mediaList = document.createElement("ul");
    mediaTitle.textContent = "Multimedia";
    for (const media of data.media) {
      const item = document.createElement("li");
      item.textContent = `${media.type}: ${media.title} (${media.filename})`;
      mediaList.append(item);
    }
    container.append(mediaTitle, mediaList);
  }
}

async function ask(question) {
  $("askButton").disabled = true;
  setStatus("Consultando...");

  try {
    renderAnswer(await postJson("/ask", { question, session_id: sessionId }));
    setStatus("Listo");
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    $("askButton").disabled = false;
  }
}

async function loadCorpus() {
  if (!$('corpus').hidden && $('corpus').children.length) {
    $('corpus').hidden = true;
    $('loadCorpus').textContent = 'Ver corpus cargado';
    return;
  }
  if ($('corpus').hidden && $('corpus').children.length) {
    $('corpus').hidden = false;
    $('loadCorpus').textContent = 'Minimizar corpus';
    return;
  }

  $("loadCorpus").disabled = true;
  setStatus("Cargando corpus...");
  try {
    renderCorpus(await getJson("/corpus"));
    $("loadCorpus").textContent = "Minimizar corpus";
    setStatus("Corpus cargado");
  } catch (error) {
    setStatus(`Error corpus: ${error.message}`);
  } finally {
    $("loadCorpus").disabled = false;
  }
}

async function sendFeedback(value) {
  if (!lastResponse) return;

  $("feedbackStatus").textContent = "Enviando feedback...";

  try {
    await postJson("/feedback", {
      answer_id: lastResponse.answer_id,
      feedback: value,
      request_id: lastResponse.metadata?.request_id || lastResponse.metadata?.requestId || null
    });
    $("feedbackStatus").textContent = "Feedback enviado";
  } catch (error) {
    $("feedbackStatus").textContent = `Error feedback: ${error.message}`;
  }
}

function init() {
  $("apiBase").value = localStorage.getItem("helpRagApiBase") || location.origin;
  $("question").addEventListener("input", () => {
    $("charCount").textContent = String($("question").value.length);
  });

  for (const question of quickQuestions) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = question;
    button.addEventListener("click", () => {
      $("question").value = question;
      $("charCount").textContent = String(question.length);
      ask(question);
    });
    $("quickQuestions").append(button);
  }

  $("apiBase").addEventListener("change", () => {
    localStorage.setItem("helpRagApiBase", apiBase());
  });

  $("askForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("question").value.trim();
    if (question) ask(question);
  });

  $("loadCorpus").addEventListener("click", loadCorpus);

  document.querySelectorAll("[data-feedback]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(button.dataset.feedback));
  });
}

init();
