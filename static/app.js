const $ = selector => document.querySelector(selector);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[char]));
const fileInput = $("#file"), drop = $("#drop"), status = $("#status"), toast = $("#toast");
let currentData = null;
let selectedFile = null;
let selectedMode = "image";
let toastTimer;
let activeRequestController = null;
let requestGeneration = 0;
let lastReportId = null;
let previewUrl = null;

if (window.lucide) lucide.createIcons();
const verdictColor = verdict => verdict === "Likely Deepfake" ? "#fb7185" : verdict === "Likely Real" ? "#34d399" : "#fbbf24";
const formatBytes = bytes => bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function notify(message, tone = "info") {
  toast.textContent = message;
  toast.className = `fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-xl border px-4 py-3 text-sm shadow-2xl ${tone === "error" ? "border-rose-700 bg-rose-950 text-rose-100" : "border-slate-700 bg-[#18253a] text-slate-100"}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 4500);
}

function setStatus(message, tone = "normal") {
  status.textContent = message;
  status.className = `mt-3 min-h-5 text-sm ${tone === "error" ? "text-rose-300" : "text-cyan-300"}`;
}

function analysisErrorMessage(error) {
  if (error?.name === "AbortError") return "";
  if (error instanceof TypeError && /fetch/i.test(error.message || "")) {
    return "Analysis service is unavailable. Please start DeepScan again and retry.";
  }
  return error?.message || "Analysis failed. Please try again.";
}

function setProgress(value) {
  const bar = $("#progress-bar");
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function closeLightbox() {
  $("#lightbox").classList.add("hidden");
  $("#lightbox").classList.remove("flex");
  $("#lightbox-image").removeAttribute("src");
  $("#lightbox-image").classList.remove("lightbox-zoomed");
}

function resetResults() {
  requestGeneration += 1;
  if (activeRequestController) {
    activeRequestController.abort();
    activeRequestController = null;
  }
  currentData = null;
  lastReportId = null;
  $("#result").classList.add("hidden");
  $("#verdict").textContent = "";
  $("#verdict").style.removeProperty("color");
  $("#score").textContent = "0.0/100";
  $("#gauge").style.setProperty("--gauge-color", "#fbbf24");
  $("#gauge").setAttribute("aria-valuenow", "0");
  $("#gauge-progress").style.strokeDashoffset = "251.327";
  $("#gauge-status").textContent = "Inconclusive";
  $("#gauge-status").style.color = "#fbbf24";
  $("#confidence-badge").textContent = "";
  $("#confidence-badge").removeAttribute("style");
  $("#confidence-meaning").textContent = "";
  $("#meta").textContent = "";
  $("#reason-chips").replaceChildren();
  $("#summary").replaceChildren();
  $("#summary-loading").classList.add("hidden");
  $("#fallback-notice").classList.add("hidden");
  $("#ai-label").classList.add("hidden");
  $("#ai-label").classList.remove("flex");
  $("#evidence").replaceChildren();
  $("#original").removeAttribute("src");
  $("#heatmap").removeAttribute("src");
  $("#gallery-image").removeAttribute("src");
  $("#gallery-image").classList.add("hidden");
  $("#gallery-caption").textContent = "";
  $("#face-warning").classList.add("hidden");
  $("#limitations").replaceChildren();
  $("#video-only").classList.add("hidden");
  $("#top-frames").replaceChildren();
  const canvas = $("#timeline");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  $("#technical-content").replaceChildren();
  const download = $("#download");
  download.removeAttribute("href");
  download.setAttribute("aria-disabled", "true");
  download.classList.add("pointer-events-none", "opacity-50");
  closeLightbox();
  clearTimeout(toastTimer);
  toast.textContent = "";
  toast.classList.add("hidden");
  setProgress(0);
  setStatus("");
}

function resetSelection() {
  resetResults();
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  fileInput.value = "";
  $("#selected-file").classList.add("hidden");
  $("#selected-file").classList.remove("flex");
  $("#preview-wrap").classList.add("hidden");
  $("#preview").removeAttribute("src");
  $("#file-name").textContent = "";
  $("#file-meta").textContent = "";
  $("#analyze-button").disabled = true;
}

function setMode(mode) {
  selectedMode = mode;
  fileInput.accept = mode === "image" ? "image/*" : "video/*";
  $("#drop-type").textContent = mode;
  document.querySelectorAll(".mode-button").forEach(button => {
    const active = button.dataset.mode === mode;
    button.className = `mode-button flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-bold ${active ? "bg-cyan-400 text-slate-950" : "text-slate-400"}`;
  });
  resetSelection();
}

function selectFile(file) {
  if (!file) return;
  resetResults();
  const isImage = file.type.startsWith("image/");
  const isVideo = file.type.startsWith("video/");
  if ((selectedMode === "image" && !isImage) || (selectedMode === "video" && !isVideo)) {
    fileInput.value = "";
    setStatus(`Please choose a ${selectedMode} file.`, "error");
    notify(`That file is not a ${selectedMode}.`, "error");
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    fileInput.value = "";
    setStatus("Maximum upload size is 100 MB.", "error");
    notify("File is larger than the 100 MB limit.", "error");
    return;
  }
  selectedFile = file;
  fileInput.value = "";
  $("#file-name").textContent = file.name;
  $("#file-meta").textContent = `${formatBytes(file.size)} · ${file.type || selectedMode}`;
  if (isImage) {
    previewUrl = URL.createObjectURL(file);
    $("#preview").src = previewUrl;
    $("#preview-wrap").classList.remove("hidden");
  }
  $("#selected-file").classList.remove("hidden");
  $("#selected-file").classList.add("flex");
  $("#analyze-button").disabled = false;
  setStatus("Ready to analyze.");
}

function renderEvidence(items = []) {
  const descriptions = { ensemble:"Pretrained detectors compare the media against learned real/fake visual patterns.", ela:"Checks whether parts of an image were compressed differently after editing or re-saving.", fft:"Examines repeating frequency patterns that can appear after generation or resizing.", noise:"Compares natural sensor noise in the face region with the surrounding image.", metadata:"Reads camera, editing, generator, and other file metadata when available.", boundary:"Checks for color and edge mismatch around the face region." };
  $("#evidence").innerHTML = items.length ? items.map(item => {
    const score = Number(item.score);
    const percent = Number.isFinite(score) ? Math.round(score * 100) : null;
    const kind = item.score_kind === "probability" ? "calibrated model probability" : "heuristic suspicion index";
    const color = percent === null ? "#64748b" : percent >= 65 ? "#fb7185" : percent <= 35 ? "#34d399" : "#fbbf24";
    return `<article class="rounded-2xl border border-slate-800 bg-[#0e1626] p-4"><div class="flex items-start justify-between gap-3"><div class="flex items-center gap-2"><h3 class="font-bold text-slate-100">${escapeHtml(item.signal)}</h3><span title="${escapeHtml(descriptions[String(item.signal).toLowerCase()] || "A forensic signal used in the combined result.")}" class="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-slate-600 text-xs text-slate-400">i</span></div><span class="text-right text-xs text-slate-400">${percent === null ? "Unavailable" : `${percent}% ${kind} · ${Math.round(Number(item.weight || 0) * 100)}% weight`}</span></div><div class="mt-4 h-2 overflow-hidden rounded-full bg-slate-800">${percent === null ? "" : `<div class="bar-fill h-full rounded-full" style="width:${Math.max(0, Math.min(100, percent))}%;background:${color}"></div>`}</div><p class="mt-3 text-sm leading-6 text-slate-400">${escapeHtml(item.finding || "No finding available.")}</p></article>`;
  }).join("") : `<p class="rounded-xl border border-slate-800 p-4 text-sm text-slate-500">No evidence items were returned.</p>`;
}

function renderSummary(data) {
  $("#summary-loading").classList.add("hidden");
  $("#fallback-notice").classList.toggle("hidden", !data.summary_is_fallback);
  $("#ai-label").classList.toggle("hidden", Boolean(data.summary_is_fallback));
  $("#ai-label").classList.toggle("flex", !data.summary_is_fallback);
  $("#summary").innerHTML = window.marked ? marked.parse(data.summary_markdown || "") : `<p>${escapeHtml(data.summary_markdown || "")}</p>`;
  if (window.lucide) lucide.createIcons();
}

function renderChips(chips = []) {
  $("#reason-chips").innerHTML = chips.map(chip => {
    const style = chip.direction === "fake" ? "border-rose-800/80 bg-rose-950/30 text-rose-200" : chip.direction === "real" ? "border-emerald-800/80 bg-emerald-950/30 text-emerald-200" : "border-slate-700 bg-slate-900 text-slate-300";
    const icon = chip.direction === "fake" ? "triangle-alert" : chip.direction === "real" ? "check-circle-2" : "circle-help";
    return `<span title="${escapeHtml(chip.detail)}" class="inline-flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-semibold ${style}"><i data-lucide="${icon}" class="h-3.5 w-3.5"></i>${escapeHtml(chip.label)}</span>`;
  }).join("");
  if (window.lucide) lucide.createIcons();
}

function showGallery(tab) {
  const image = currentData?.visualizations?.[tab] || "";
  $("#gallery-image").src = image;
  $("#gallery-image").classList.toggle("hidden", !image);
  $("#gallery-caption").textContent = currentData?.captions?.[tab] || "This visualization was not available for this analysis.";
  document.querySelectorAll(".gallery-tab").forEach(button => {
    const active = button.dataset.tab === tab;
    button.className = `gallery-tab rounded-lg px-3 py-2 text-sm font-semibold ${active ? "bg-cyan-400 text-slate-950" : "bg-slate-800 text-slate-300"}`;
  });
}

function renderVideo(data) {
  const section = $("#video-only");
  section.classList.toggle("hidden", data.type !== "video");
  if (data.type !== "video") return;
  const canvas = $("#timeline");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(600, rect.width * ratio);
  canvas.height = 220 * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = canvas.clientWidth || 600;
  const height = 220;
  const points = data.timeline || [];
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#22d3ee";
  context.lineWidth = 2;
  context.beginPath();
  points.forEach((point, index) => {
    const x = index * (width / Math.max(1, points.length - 1));
    const y = height - 25 - Number(point.score || 0) * (height - 55);
    index ? context.lineTo(x, y) : context.moveTo(x, y);
    context.fillStyle = Number(point.score) >= .65 ? "#fb7185" : "#22d3ee";
    context.fillRect(x - 3, y - 3, 6, 6);
  });
  context.stroke();
  $("#top-frames").innerHTML = (data.top_suspicious_frames || []).map(point => `<figure class="rounded-xl border border-slate-800 bg-[#0e1626] p-2"><img src="${point.image}" class="lightbox-image w-full cursor-zoom-in rounded-lg" alt="Suspicious frame at ${Number(point.timestamp).toFixed(2)} seconds"><figcaption class="px-1 pt-2 text-xs text-slate-400">${Number(point.timestamp).toFixed(2)}s · ${Math.round(Number(point.score) * 100)}% suspicious</figcaption></figure>`).join("");
  bindLightboxes();
}

function renderTechnical(data) {
  const models = (data.per_model || []).map(item => `<li>${escapeHtml(item.model)}: ${Math.round(Number(item.fake_probability) * 100)}% · ${escapeHtml(item.label || "label unavailable")}</li>`).join("");
  const timings = Object.entries(data.timings || {}).map(([key, value]) => `<li>${escapeHtml(key)}: ${escapeHtml(value)} ms</li>`).join("");
  $("#technical-content").innerHTML = `<div><h3 class="font-bold text-slate-200">Model outputs</h3><ul class="mt-2 list-disc space-y-1 pl-5 text-slate-400">${models || "<li>Unavailable</li>"}</ul></div><div><h3 class="font-bold text-slate-200">Processing timings</h3><ul class="mt-2 list-disc space-y-1 pl-5 text-slate-400">${timings || "<li>Unavailable</li>"}</ul></div>`;
}

function renderResult(data, generation) {
  if (generation !== requestGeneration) return;
  currentData = data;
  lastReportId = data.id || null;
  $("#result").classList.remove("hidden");
  const color = verdictColor(data.verdict);
  const score = Number(data.fake_score);
  const visualScore = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  $("#verdict").textContent = data.verdict;
  $("#verdict").style.color = color;
  $("#score").textContent = `${Number.isFinite(score) ? score.toFixed(1) : "0.0"}/100`;
  $("#gauge").style.setProperty("--gauge-color", color);
  $("#gauge").setAttribute("aria-valuenow", String(visualScore));
  $("#gauge-progress").style.strokeDashoffset = String(251.327 * (1 - visualScore / 100));
  $("#gauge-status").textContent = data.verdict;
  $("#gauge-status").style.color = color;
  $("#confidence-badge").textContent = `${data.confidence} confidence`;
  $("#confidence-badge").style.backgroundColor = `${color}22`;
  $("#confidence-badge").style.color = color;
  $("#confidence-meaning").textContent = data.confidence === "Low" ? "Low = results may be unreliable; review the limitations and evidence." : data.confidence === "Medium" ? "Medium = signals are mixed; treat this as a screening result." : "High = multiple signals agree, but this remains probabilistic.";
  $("#meta").textContent = `${Number(data.processing_time).toFixed(2)}s processing · ${data.type} analysis`;
  renderChips(data.reason_chips);
  renderSummary(data);
  renderEvidence(data.evidence);
  $("#original").src = data.original_image || "";
  $("#heatmap").src = data.visualizations?.heatmap || data.heatmaps?.[0] || "";
  $("#face-warning").classList.toggle("hidden", !data.no_face_detected);
  const limitations = [...new Set([...(data.input_warnings || []), ...(data.limitations || [])])];
  $("#limitations").innerHTML = limitations.length ? limitations.map(item => `<li>${escapeHtml(item)}</li>`).join("") : "<li>No additional limitations were reported.</li>";
  showGallery("face_crop");
  renderVideo(data);
  renderTechnical(data);
  if (lastReportId) {
    $("#download").href = `/api/report/${encodeURIComponent(lastReportId)}`;
    $("#download").removeAttribute("aria-disabled");
    $("#download").classList.remove("pointer-events-none", "opacity-50");
  }
  $("#result").scrollIntoView({ behavior:"smooth", block:"start" });
  if (window.lucide) lucide.createIcons();
}

async function analyze() {
  if (!selectedFile) return;
  resetResults();
  const generation = requestGeneration;
  activeRequestController = new AbortController();
  const button = $("#analyze-button");
  button.disabled = true;
  setStatus("Analyzing locally and collecting forensic signals…");
  setProgress(10);
  $("#summary-loading").classList.remove("hidden");
  const body = new FormData();
  body.append("file", selectedFile);
  try {
    const response = await fetch("/api/analyze", { method:"POST", body, signal:activeRequestController.signal });
    setProgress(75);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");
    if (generation !== requestGeneration) return;
    renderResult(data, generation);
    setProgress(100);
    setStatus("Analysis complete.");
    notify("Analysis complete.");
  } catch (error) {
    if (error.name === "AbortError" || generation !== requestGeneration) return;
    const message = analysisErrorMessage(error);
    setProgress(0);
    if (!message) return;
    setStatus(message, "error");
    notify(message, "error");
  } finally {
    if (generation === requestGeneration) {
      activeRequestController = null;
      button.disabled = false;
    }
  }
}

function bindLightboxes() {
  document.querySelectorAll(".lightbox-image").forEach(image => {
    image.onclick = () => {
      if (!image.src) return;
      $("#lightbox-image").src = image.src;
      $("#lightbox-image").classList.remove("lightbox-zoomed");
      $("#lightbox").classList.remove("hidden");
      $("#lightbox").classList.add("flex");
    };
  });
}

document.querySelectorAll(".mode-button").forEach(button => button.onclick = () => setMode(button.dataset.mode));
$("#pick").onclick = () => fileInput.click();
fileInput.onchange = () => selectFile(fileInput.files[0]);
$("#analyze-button").onclick = analyze;
$("#reset").onclick = resetSelection;
drop.ondragover = event => { event.preventDefault(); drop.classList.add("drop-active"); };
drop.ondragleave = () => drop.classList.remove("drop-active");
drop.ondrop = event => { event.preventDefault(); drop.classList.remove("drop-active"); selectFile(event.dataTransfer.files[0]); };
document.addEventListener("paste", event => {
  const file = [...(event.clipboardData?.files || [])].find(item => item.type.startsWith("image/") || item.type.startsWith("video/"));
  if (file) selectFile(file);
});
document.querySelectorAll(".gallery-tab").forEach(button => button.onclick = () => showGallery(button.dataset.tab));
$("#close-lightbox").onclick = closeLightbox;
$("#lightbox-image").onclick = () => $("#lightbox-image").classList.toggle("lightbox-zoomed");
document.addEventListener("keydown", event => { if (event.key === "Escape") closeLightbox(); });
$("#mobile-menu-button").onclick = () => {
  const menu = $("#mobile-menu");
  const open = menu.classList.toggle("hidden") === false;
  $("#mobile-menu-button").setAttribute("aria-expanded", String(open));
};
document.querySelectorAll("#mobile-menu a").forEach(link => link.onclick = () => $("#mobile-menu").classList.add("hidden"));

async function checkHealth() {
  const indicator = $("#api-status");
  const overview = $("#workspace-api-status");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    indicator.innerHTML = '<span class="h-2 w-2 rounded-full bg-emerald-400"></span> API operational';
    indicator.className = "hidden items-center gap-2 rounded-full border border-emerald-800/70 bg-emerald-950/30 px-3 py-1.5 text-xs text-emerald-300 sm:flex";
    if (overview) overview.innerHTML = '<span class="h-2 w-2 rounded-full bg-emerald-400"></span> API operational';
  } catch {
    indicator.innerHTML = '<span class="h-2 w-2 rounded-full bg-rose-400"></span> API unavailable';
    indicator.className = "hidden items-center gap-2 rounded-full border border-rose-800/70 bg-rose-950/30 px-3 py-1.5 text-xs text-rose-300 sm:flex";
    if (overview) overview.innerHTML = '<span class="h-2 w-2 rounded-full bg-rose-400"></span> API unavailable';
  }
}

resetResults();
checkHealth();
bindLightboxes();
