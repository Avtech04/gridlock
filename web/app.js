const API = "";
const violationTypes = [
  "Helmet Non-compliance",
  "Seatbelt Non-compliance",
  "Triple Riding",
  "Wrong-side Driving",
  "Stop-line Violation",
  "Red-light Violation",
  "Illegal Parking",
];

const state = {
  uploadMode: "image",
  file: null,
  batchFile: null,
  previewUrl: null,
  latest: null,
  latestBatch: null,
  singleRejectedIds: new Set(),
  batchRejectedIds: new Set(),
  videoFile: null,
  videoPreviewUrl: null,
  latestVideo: null,
  calibrationPreviewUrl: null,
  calibrationDrag: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function fmtPct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function fmtTime(ms) {
  if (!ms) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString();
}

function fmtBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactDetectorName(value) {
  const text = String(value || "").trim();
  if (!text) return "-";
  const models = text
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (models.length > 1) return `YOLO ensemble (${models.length} models)`;
  return text.split("/").pop() || text;
}

function isZipFile(file) {
  return Boolean(file?.name?.toLowerCase().endsWith(".zip"));
}

function violationId(violation) {
  return String(violation?.record_id || violation?.id || "");
}

function safeClass(value) {
  return String(value || "pending")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-");
}

function recordEvidenceUrl(record) {
  return record?.evidence_image_url || record?.annotated_image_url || record?.original_image_url || "";
}

function activeViolations(violations, rejectedIds) {
  return (violations || []).filter((violation) => {
    const id = violationId(violation);
    return !id || !rejectedIds.has(id);
  });
}

function resetEvidenceReport() {
  $("#violationCount").textContent = "0 found";
  $("#summaryGrid").innerHTML = `
    <div><span>Vehicles</span><strong>-</strong></div>
    <div><span>Pedestrians</span><strong>-</strong></div>
    <div><span>Traffic Light</span><strong>-</strong></div>
    <div><span>Processing</span><strong>-</strong></div>
  `;
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent =
    state.uploadMode === "batch" ? "Upload a ZIP archive and run batch analysis." : "Upload an image and run analysis.";
  $("#reviewPanel").classList.add("hidden");
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function setBusy(isBusy) {
  const btn = $("#analyzeBtn");
  const hasEvidence = state.uploadMode === "batch" ? Boolean(state.batchFile) : Boolean(state.file);
  btn.disabled = isBusy || !hasEvidence;
  if (state.uploadMode === "batch") {
    btn.innerHTML = isBusy
      ? `<span class="btn-icon analyze"></span>Processing ZIP...`
      : `<span class="btn-icon analyze"></span>Run Batch Analysis`;
    return;
  }
  btn.innerHTML = isBusy
    ? `<span class="btn-icon analyze"></span>Analyzing...`
    : `<span class="btn-icon analyze"></span>Run Analysis`;
}

function setVideoBusy(isBusy) {
  const btn = $("#analyzeVideoBtn");
  btn.disabled = isBusy || !state.videoFile;
  btn.innerHTML = isBusy
    ? `<span class="btn-icon video"></span>Tracking...`
    : `<span class="btn-icon video"></span>Run Video Tracking`;
}

function setSystem(status) {
  const dot = $("#statusDot");
  const rawModel = status.detector?.model || status.detector?.backend || "No detector loaded";
  dot.classList.toggle("ready", Boolean(status.engine_ready));
  dot.classList.toggle("error", !status.engine_ready);
  $("#systemState").textContent = status.engine_ready ? "Detector ready" : "Detector unavailable";
  $("#systemDetail").textContent = compactDetectorName(rawModel);
  $("#systemDetail").title = rawModel;
}

async function loadStatus() {
  try {
    const status = await requestJson("/api/status");
    setSystem(status);
  } catch (error) {
    $("#statusDot").classList.add("error");
    $("#systemState").textContent = "Backend offline";
    $("#systemDetail").textContent = "Start FastAPI on port 8000";
  }
}

function selectView(view) {
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
  if (view === "records") loadRecords();
  if (view === "analytics") loadAnalytics();
  if (view === "performance") loadPerformance();
  if (view === "config") loadCalibration();
}

function setUploadMode(mode) {
  state.uploadMode = mode === "batch" ? "batch" : "image";
  $$(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.uploadMode === state.uploadMode));
  $("#fileInput").accept = state.uploadMode === "batch" ? ".zip,application/zip" : "image/*";
  $("#dropTitle").textContent = state.uploadMode === "batch" ? "Drop a ZIP of traffic images" : "Drop traffic evidence here";
  $("#dropSubtitle").textContent =
    state.uploadMode === "batch"
      ? "ZIP archive with JPG, PNG, WebP, or BMP images"
      : "JPG, PNG, or WebP traffic camera frame";
  clearImage(false);
  resetEvidenceReport();
  setBusy(false);
}

function handleFile(file) {
  if (!file) return;
  if (state.uploadMode === "batch" || isZipFile(file)) {
    handleZipFile(file);
    return;
  }
  if (!file.type.startsWith("image/")) return;
  state.file = file;
  state.batchFile = null;
  state.singleRejectedIds.clear();
  state.batchRejectedIds.clear();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("#previewImage").src = state.previewUrl;
  $("#dropZone").classList.add("hidden");
  $("#batchReady").classList.add("hidden");
  $("#previewShell").classList.remove("hidden");
  $("#clearImage").classList.remove("hidden");
  $("#analyzeBtn").disabled = false;
  $("#boxLayer").innerHTML = "";
  $("#batchPanel").classList.add("hidden");
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent = "Ready to analyze.";
  setBusy(false);
}

function handleZipFile(file) {
  if (!isZipFile(file)) {
    $("#resultList").className = "result-list empty";
    $("#resultList").textContent = "Please choose a .zip archive for batch analysis.";
    return;
  }
  state.uploadMode = "batch";
  $$(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.uploadMode === "batch"));
  state.file = null;
  state.batchFile = file;
  state.latest = null;
  state.latestBatch = null;
  state.singleRejectedIds.clear();
  state.batchRejectedIds.clear();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("#fileInput").value = "";
  $("#previewShell").classList.add("hidden");
  $("#boxLayer").innerHTML = "";
  $("#dropZone").classList.add("hidden");
  $("#batchReady").classList.remove("hidden");
  $("#clearImage").classList.remove("hidden");
  $("#batchFileName").textContent = file.name;
  $("#batchFileDetail").textContent = `${fmtBytes(file.size)} ZIP package. Supported images will be processed one by one.`;
  $("#pipelinePanel").classList.add("hidden");
  $("#batchPanel").classList.add("hidden");
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent = "Ready to run batch analysis.";
  setBusy(false);
}

function clearImage(resetMode = true) {
  state.file = null;
  state.batchFile = null;
  state.latest = null;
  state.latestBatch = null;
  state.singleRejectedIds.clear();
  state.batchRejectedIds.clear();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("#fileInput").value = "";
  $("#dropZone").classList.remove("hidden");
  $("#previewShell").classList.add("hidden");
  $("#batchReady").classList.add("hidden");
  $("#clearImage").classList.add("hidden");
  $("#pipelinePanel").classList.add("hidden");
  $("#batchPanel").classList.add("hidden");
  $("#analyzeBtn").disabled = true;
  $("#boxLayer").innerHTML = "";
  if (resetMode) {
    state.uploadMode = "image";
    $$(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.uploadMode === "image"));
    $("#fileInput").accept = "image/*";
    $("#dropTitle").textContent = "Drop traffic evidence here";
    $("#dropSubtitle").textContent = "JPG, PNG, or WebP traffic camera frame";
  }
  resetEvidenceReport();
}

function handleVideoFile(file) {
  const extension = file?.name?.split(".").pop()?.toLowerCase();
  if (!file || (!file.type.startsWith("video/") && !["mp4", "mov", "m4v", "avi", "webm", "mkv"].includes(extension))) return;
  state.videoFile = file;
  state.latestVideo = null;
  if (state.videoPreviewUrl) URL.revokeObjectURL(state.videoPreviewUrl);
  state.videoPreviewUrl = URL.createObjectURL(file);
  $("#videoPreview").src = state.videoPreviewUrl;
  $("#videoDropZone").classList.add("hidden");
  $("#videoPreviewShell").classList.remove("hidden");
  $("#analyzeVideoBtn").disabled = false;
  $("#videoPipelinePanel").classList.add("hidden");
  $("#videoResultList").className = "result-list empty";
  $("#videoResultList").textContent = "Ready to track video.";
}

function clearVideo() {
  state.videoFile = null;
  state.latestVideo = null;
  if (state.videoPreviewUrl) URL.revokeObjectURL(state.videoPreviewUrl);
  state.videoPreviewUrl = null;
  $("#videoFileInput").value = "";
  $("#videoPreview").removeAttribute("src");
  $("#videoDropZone").classList.remove("hidden");
  $("#videoPreviewShell").classList.add("hidden");
  $("#videoPipelinePanel").classList.add("hidden");
  $("#analyzeVideoBtn").disabled = true;
  $("#videoResultList").className = "result-list empty";
  $("#videoResultList").textContent = "Upload a video and run tracking.";
}

function renderBoxes(violations) {
  $("#boxLayer").innerHTML = violations
    .map((item) => {
      const b = item.bbox_percent || {};
      return `<div class="box" style="left:${b.x || 0}%;top:${b.y || 0}%;width:${b.w || 0}%;height:${b.h || 0}%">
        <div class="box-label">${item.type} ${fmtPct(item.confidence)}</div>
      </div>`;
    })
    .join("");
}

function renderSummary(data) {
  const items = [
    ["Vehicles", data.detection?.total_vehicles ?? 0],
    ["Pedestrians", data.detection?.total_pedestrians ?? 0],
    ["Traffic Light", data.scene?.traffic_light || "not_visible"],
    ["Processing", fmtTime(data.processing_time_ms)],
  ];
  $("#summaryGrid").innerHTML = items.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function updateSingleReviewPanel(allViolations, visibleViolations) {
  if (!state.latest || !(allViolations || []).length) {
    $("#reviewPanel").classList.add("hidden");
    return;
  }
  const approved = state.latest.review?.review_status === "approved" || state.latest.review_status === "approved";
  const removedCount = Math.max(0, (allViolations || []).length - (visibleViolations || []).length);
  $("#reviewPanel").classList.remove("hidden", "approved");
  if (approved) {
    $("#reviewPanel").classList.add("approved");
    $("#reviewSummary").textContent = `Approved evidence: ${visibleViolations.length} kept, ${state.latest.review?.rejected_count || removedCount} rejected.`;
    $("#approveSessionBtn").disabled = true;
    $("#resetReviewBtn").disabled = true;
    return;
  }
  $("#reviewSummary").textContent = `${visibleViolations.length} pending violation(s), ${removedCount} marked wrong.`;
  $("#approveSessionBtn").disabled = false;
  $("#resetReviewBtn").disabled = removedCount === 0;
}

function renderViolations(violations) {
  const visibleViolations = activeViolations(violations, state.singleRejectedIds);
  $("#violationCount").textContent = `${visibleViolations.length} found`;
  if (!visibleViolations.length) {
    $("#resultList").className = "result-list empty";
    $("#resultList").textContent = violations.length
      ? "All detected violations are marked as wrong. Approve to reject them from final analysis."
      : "No violations detected by calibrated rules.";
    updateSingleReviewPanel(violations, visibleViolations);
    return;
  }
  $("#resultList").className = "result-list";
  $("#resultList").innerHTML = visibleViolations
    .map(
      (v) => `<article class="violation-card">
        <header>
          <strong>${escapeHtml(v.type)}</strong>
          <span>${fmtPct(v.confidence)}</span>
        </header>
        <div class="confidence"><div style="width:${Math.round((v.confidence || 0) * 100)}%"></div></div>
        <p>${escapeHtml(v.description || "Detector-grounded violation")}</p>
        <small>${escapeHtml(v.vehicle_type || "vehicle")}${v.license_plate ? ` · ${escapeHtml(v.license_plate)}` : ""} · ${escapeHtml(v.evidence || "evidence")}</small>
        ${
          violationId(v)
            ? `<button class="reject-chip" type="button" data-review-action="reject-single" data-record-id="${escapeHtml(violationId(v))}">Remove false ID</button>`
            : ""
        }
      </article>`
    )
    .join("");
  updateSingleReviewPanel(violations, visibleViolations);
}

function renderPipeline(data) {
  const detector = data.pipeline?.detector || {};
  const ocr = data.pipeline?.ocr || {};
  const review = data.pipeline?.vision_review || {};
  const rows = [
    ["Detector", detector.model || detector.backend || "unavailable"],
    ["OCR plates", data.detection?.recognized_plates?.length ?? 0],
    ["Vision review", review.configured ? review.provider : "off"],
    ["Preprocessing", data.preprocessing?.steps_applied?.join(", ") || "none"],
  ];
  $("#pipelineGrid").innerHTML = rows.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#evidenceLink").href = data.evidence?.annotated_image || "#";
  $("#pipelinePanel").classList.remove("hidden");
}

function renderBatchSummary(data) {
  const activeTotal = (data.results || []).reduce(
    (total, item) => total + activeViolations(item.violations || [], state.batchRejectedIds).length,
    0
  );
  const imagesWithActiveViolations = (data.results || []).filter(
    (item) => activeViolations(item.violations || [], state.batchRejectedIds).length > 0
  ).length;
  $("#violationCount").textContent = `${activeTotal} found`;
  const summary = data.summary || {};
  const items = [
    ["Images", data.processed ?? 0],
    ["Violations", activeTotal],
    ["Plates", summary.recognized_plates ?? 0],
    ["Processing", fmtTime(data.processing_time_ms)],
  ];
  $("#summaryGrid").innerHTML = items.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  const message = [
    `${data.processed || 0} image(s) processed`,
    `${imagesWithActiveViolations} image(s) with active violations`,
    `${data.failed || 0} skipped/failed`,
  ].join(" · ");
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent = message;
}

function renderBatchPanel(data) {
  const summary = data.summary || {};
  $("#batchCount").textContent = `${data.processed || 0} images`;
  const activeTotal = (data.results || []).reduce(
    (total, item) => total + activeViolations(item.violations || [], state.batchRejectedIds).length,
    0
  );
  const imagesWithActiveViolations = (data.results || []).filter(
    (item) => activeViolations(item.violations || [], state.batchRejectedIds).length > 0
  ).length;
  const removedCount = state.batchRejectedIds.size;
  const stats = [
    ["Images processed", data.processed || 0],
    ["Images with violations", imagesWithActiveViolations],
    ["Active violations", activeTotal],
    ["Removed false IDs", removedCount],
    ["Vehicles", summary.total_vehicles || 0],
    ["Recognized plates", summary.recognized_plates || 0],
    ["Failed/skipped", data.failed || 0],
  ];
  $("#batchStats").innerHTML = stats.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  const resultCards = (data.results || [])
    .map((item) => {
      const plates = item.detection?.recognized_plates || [];
      const plateText = plates.length ? plates.map((plate) => escapeHtml(plate.text || "plate")).join(", ") : "No plate read";
      const violations = activeViolations(item.violations || [], state.batchRejectedIds);
      const violationMarkup = violations.length
        ? violations
            .map(
              (v) => `<li>
                <b>${escapeHtml(v.type)}</b>
                <span>${fmtPct(v.confidence)}</span>
                ${
                  violationId(v)
                    ? `<button class="mini-reject" type="button" data-review-action="reject-batch" data-record-id="${escapeHtml(violationId(v))}">Remove</button>`
                    : ""
                }
              </li>`
            )
            .join("")
        : `<li><b>Clean frame</b><span>0%</span></li>`;
      const evidenceUrl = item.evidence?.annotated_image || "#";
      return `<article class="batch-card">
        <a class="batch-thumb" href="${escapeHtml(evidenceUrl)}" target="_blank" rel="noreferrer">
          ${evidenceUrl !== "#" ? `<img src="${escapeHtml(evidenceUrl)}" alt="${escapeHtml(item.source_filename)} annotated evidence" />` : `<span>No evidence</span>`}
        </a>
        <div class="batch-card-body">
          <header>
            <div>
              <strong>${escapeHtml(item.source_filename)}</strong>
              <small>${item.detection?.total_vehicles ?? 0} vehicle(s) · ${plateText}</small>
            </div>
            <span class="batch-score ${violations.length ? "alert" : "clean"}">${violations.length}</span>
          </header>
          <ul>${violationMarkup}</ul>
          <a class="link-btn compact-link" href="${escapeHtml(evidenceUrl)}" target="_blank" rel="noreferrer">Open evidence</a>
        </div>
      </article>`;
    })
    .join("");

  const failureCards = (data.failures || [])
    .map(
      (failure) => `<article class="batch-failure">
        <strong>${escapeHtml(failure.source_filename)}</strong>
        <span>${escapeHtml(failure.error)}</span>
      </article>`
    )
    .join("");

  $("#batchResults").innerHTML =
    resultCards || failureCards
      ? `${resultCards}${failureCards ? `<div class="batch-failures"><h4>Skipped or failed files</h4>${failureCards}</div>` : ""}`
      : `<div class="result-list empty">No batch results returned.</div>`;
  $("#batchPanel").classList.remove("hidden");
}

function renderVideoSummary(data) {
  const items = [
    ["Tracks", data.detection?.total_vehicles ?? 0],
    ["Frames", data.video?.sampled_frames ?? 0],
    ["Traffic Light", data.scene?.traffic_light || "not_visible"],
    ["Processing", fmtTime(data.processing_time_ms)],
  ];
  $("#videoSummaryGrid").innerHTML = items.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderVideoViolations(violations) {
  if (!violations.length) {
    $("#videoResultList").className = "result-list empty";
    $("#videoResultList").textContent = "No video violations detected by calibrated tracker rules.";
    return;
  }
  $("#videoResultList").className = "result-list";
  $("#videoResultList").innerHTML = violations
    .map(
      (v) => `<article class="violation-card">
        <header><strong>${v.type}</strong><span>${fmtPct(v.confidence)}</span></header>
        <div class="confidence"><div style="width:${Math.round((v.confidence || 0) * 100)}%"></div></div>
        <p>${v.description || "Tracker-grounded violation"}</p>
        <small>ID ${v.track_id || "-"} · ${v.vehicle_type || "vehicle"} · ${Number(v.time_s || 0).toFixed(1)}s · ${v.evidence || "tracker"}</small>
      </article>`
    )
    .join("");
}

function renderVideoPipeline(data) {
  const rows = [
    ["Duration", `${Number(data.video?.duration_s || 0).toFixed(1)}s`],
    ["Sample FPS", data.video?.sample_fps ?? "-"],
    ["Stable tracks", data.pipeline?.tracking?.stable_tracks ?? 0],
    ["Dwell rule", `${Number(data.pipeline?.tracking?.parking_dwell_seconds || 0).toFixed(0)}s`],
  ];
  $("#videoPipelineGrid").innerHTML = rows.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  const annotatedVideo = data.evidence?.annotated_video;
  const summaryImage = data.evidence?.summary_image;
  $("#videoEvidenceLink").href = annotatedVideo || summaryImage || "#";
  if (annotatedVideo) {
    $("#annotatedVideo").src = annotatedVideo;
    $("#annotatedVideo").classList.remove("hidden");
  } else {
    $("#annotatedVideo").classList.add("hidden");
  }
  if (summaryImage) {
    $("#videoSummaryImage").src = summaryImage;
    $("#videoSummaryImage").classList.remove("hidden");
  } else {
    $("#videoSummaryImage").classList.add("hidden");
  }
  $("#videoPipelinePanel").classList.remove("hidden");
}

async function analyzeVideo() {
  if (!state.videoFile) return;
  setVideoBusy(true);
  $("#videoResultList").className = "result-list empty";
  $("#videoResultList").textContent = "Running detector, tracker, and temporal rules...";
  try {
    const form = new FormData();
    form.append("file", state.videoFile);
    const data = await requestJson("/api/analyze-video", { method: "POST", body: form });
    state.latestVideo = data;
    renderVideoSummary(data);
    renderVideoViolations(data.violations || []);
    renderVideoPipeline(data);
    loadAnalytics();
  } catch (error) {
    $("#videoResultList").className = "result-list empty";
    $("#videoResultList").textContent = `Video analysis failed: ${error.message}`;
  } finally {
    setVideoBusy(false);
  }
}

async function analyze() {
  if (state.uploadMode === "batch") {
    await analyzeZip();
    return;
  }
  if (!state.file) return;
  setBusy(true);
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent = "Running detector, OCR, and rule engine...";
  try {
    const form = new FormData();
    form.append("file", state.file);
    const data = await requestJson("/api/analyze", { method: "POST", body: form });
    state.latest = data;
    state.singleRejectedIds.clear();
    renderSummary(data);
    renderViolations(data.violations || []);
    renderBoxes(activeViolations(data.violations || [], state.singleRejectedIds));
    renderPipeline(data);
    loadAnalytics();
  } catch (error) {
    $("#resultList").className = "result-list empty";
    $("#resultList").textContent = `Analysis failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

async function analyzeZip() {
  if (!state.batchFile) return;
  setBusy(true);
  $("#resultList").className = "result-list empty";
  $("#resultList").textContent = "Processing ZIP archive through detector, OCR, and rule engine...";
  $("#batchPanel").classList.add("hidden");
  $("#pipelinePanel").classList.add("hidden");
  try {
    const form = new FormData();
    form.append("file", state.batchFile);
    const data = await requestJson("/api/analyze-zip?max_images=60", { method: "POST", body: form });
    state.latestBatch = data;
    state.batchRejectedIds.clear();
    renderBatchSummary(data);
    renderBatchPanel(data);
    loadAnalytics();
  } catch (error) {
    $("#resultList").className = "result-list empty";
    $("#resultList").textContent = `Batch analysis failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function rejectSingleViolation(recordId) {
  if (!recordId || !state.latest) return;
  state.singleRejectedIds.add(recordId);
  const violations = state.latest.violations || [];
  renderViolations(violations);
  renderBoxes(activeViolations(violations, state.singleRejectedIds));
}

function rejectBatchViolation(recordId) {
  if (!recordId || !state.latestBatch) return;
  state.batchRejectedIds.add(recordId);
  renderBatchSummary(state.latestBatch);
  renderBatchPanel(state.latestBatch);
}

function resetSingleReview() {
  if (!state.latest) return;
  state.singleRejectedIds.clear();
  const violations = state.latest.violations || [];
  renderViolations(violations);
  renderBoxes(violations);
}

async function approveCurrentSession() {
  if (!state.latest?.session_id) return;
  $("#approveSessionBtn").disabled = true;
  $("#reviewSummary").textContent = "Approving reviewed evidence...";
  try {
    const result = await requestJson(`/api/sessions/${encodeURIComponent(state.latest.session_id)}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rejected_violation_ids: Array.from(state.singleRejectedIds),
        note: "Approved from Gridlock AI review console",
      }),
    });
    state.latest.violations = result.violations || [];
    state.latest.review = result;
    state.latest.review_status = "approved";
    if (result.evidence?.annotated_image) {
      state.latest.evidence = { ...(state.latest.evidence || {}), annotated_image: result.evidence.annotated_image };
      $("#evidenceLink").href = result.evidence.annotated_image;
    }
    state.singleRejectedIds.clear();
    renderViolations(state.latest.violations || []);
    renderBoxes(state.latest.violations || []);
    renderPipeline(state.latest);
    loadAnalytics();
  } catch (error) {
    $("#reviewSummary").textContent = `Approval failed: ${error.message}`;
    $("#approveSessionBtn").disabled = false;
  }
}

async function approveBatch() {
  if (!state.latestBatch?.results?.length) return;
  const button = $("#approveBatchBtn");
  button.disabled = true;
  button.textContent = "Approving...";
  try {
    const rejectedBySession = new Map();
    for (const item of state.latestBatch.results || []) {
      const ids = (item.violations || []).map(violationId).filter(Boolean);
      const rejected = ids.filter((id) => state.batchRejectedIds.has(id));
      rejectedBySession.set(item.session_id, rejected);
    }

    const approvals = [];
    for (const item of state.latestBatch.results || []) {
      if (!item.session_id) continue;
      approvals.push(
        requestJson(`/api/sessions/${encodeURIComponent(item.session_id)}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rejected_violation_ids: rejectedBySession.get(item.session_id) || [],
            note: "Approved from Gridlock AI batch review console",
          }),
        })
      );
    }

    const approvalResults = await Promise.all(approvals);
    const bySession = new Map(approvalResults.map((item) => [item.session_id, item]));
    state.latestBatch.results = (state.latestBatch.results || []).map((item) => {
      const approved = bySession.get(item.session_id);
      if (!approved) return item;
      return {
        ...item,
        violations: approved.violations || [],
        violation_count: approved.approved_count || 0,
        review: approved,
        evidence: {
          ...(item.evidence || {}),
          ...(approved.evidence || {}),
        },
      };
    });
    state.batchRejectedIds.clear();
    renderBatchSummary(state.latestBatch);
    renderBatchPanel(state.latestBatch);
    loadAnalytics();
  } catch (error) {
    $("#batchResults").insertAdjacentHTML(
      "afterbegin",
      `<div class="batch-failure"><strong>Approval failed</strong><span>${escapeHtml(error.message)}</span></div>`
    );
  } finally {
    button.disabled = false;
    button.textContent = "Approve Batch";
  }
}

async function loadRecords() {
  const type = $("#recordFilter").value;
  const search = $("#recordSearch").value.trim();
  const params = new URLSearchParams({ limit: "50" });
  if (type) params.set("violation_type", type);
  if (search) params.set("search", search);
  const data = await requestJson(`/api/violations?${params.toString()}`);
  const violations = data.violations || [];
  $("#recordTotal").textContent = `${violations.length} record${violations.length === 1 ? "" : "s"}`;
  renderRecordsStats(violations);
  renderRecordsGallery(violations);
  $("#recordsBody").innerHTML =
    violations
      .map(
        (v) => `<tr>
          <td>${escapeHtml(v.violation_type || "-")}</td>
          <td>${escapeHtml(v.vehicle_type || "-")}</td>
          <td>${escapeHtml(v.license_plate || "-")}</td>
          <td>${fmtPct(v.confidence)}</td>
          <td><span class="status-pill ${safeClass(v.review_status)}">${escapeHtml(v.review_status || "pending")}</span></td>
          <td>${escapeHtml(v.evidence || "-")}</td>
          <td>${fmtDate(v.timestamp)}</td>
        </tr>`
      )
      .join("") || "";
}

function renderRecordsStats(violations) {
  const approved = violations.filter((item) => item.review_status === "approved").length;
  const pending = violations.filter((item) => !item.review_status || item.review_status === "pending").length;
  const plates = violations.filter((item) => item.license_plate && item.license_plate !== "-").length;
  const typeCounts = violations.reduce((acc, item) => {
    const label = item.violation_type || "Unknown";
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});
  const topType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";
  const cards = [
    ["Total", violations.length],
    ["Approved", approved],
    ["Pending", pending],
    ["Plates", plates],
    ["Top type", topType],
  ];
  $("#recordsStats").innerHTML = cards
    .map(
      ([label, value]) => `<div>
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>`
    )
    .join("");
}

function renderRecordsGallery(violations) {
  if (!violations.length) {
    $("#recordsGallery").innerHTML = `<div class="result-list empty">No records match the current filter.</div>`;
    return;
  }
  $("#recordsGallery").innerHTML = violations
    .map((record) => {
      const imageUrl = recordEvidenceUrl(record);
      const status = record.review_status || "pending";
      const plate = record.license_plate || "Not captured";
      const vehicle = record.vehicle_type || "vehicle";
      const recordNo = record.id || record.record_id || "-";
      const imageMarkup = imageUrl
        ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(record.violation_type || "Violation")} evidence" loading="lazy" />`
        : `<div class="record-empty-thumb"><span>No evidence image</span></div>`;
      const linkMarkup = imageUrl
        ? `<a class="link-btn compact-link" href="${escapeHtml(imageUrl)}" target="_blank" rel="noreferrer">Open evidence</a>`
        : "";
      return `<article class="record-card">
        <div class="record-thumb">
          ${imageMarkup}
          <span class="record-confidence">${fmtPct(record.confidence)}</span>
        </div>
        <div class="record-card-body">
          <header>
            <div>
              <span class="record-kicker">Case #${escapeHtml(recordNo)}</span>
              <h4>${escapeHtml(record.violation_type || "Violation")}</h4>
            </div>
            <span class="status-pill ${safeClass(status)}">${escapeHtml(status)}</span>
          </header>
          <p>${escapeHtml(record.evidence || "Evidence captured from automated traffic image analysis.")}</p>
          <div class="record-meta">
            <div><span>Vehicle</span><strong>${escapeHtml(vehicle)}</strong></div>
            <div><span>Plate</span><strong>${escapeHtml(plate)}</strong></div>
            <div><span>Time</span><strong>${fmtDate(record.timestamp)}</strong></div>
          </div>
          <div class="record-actions">
            ${linkMarkup}
          </div>
        </div>
      </article>`;
    })
    .join("");
}

async function loadAnalytics() {
  const data = await requestJson("/api/analytics");
  const summary = data.summary || {};
  const perf = data.performance || {};
  const cards = [
    ["Images", summary.total_images_analyzed || 0],
    ["Violations", summary.total_violations || 0],
    ["Avg time", fmtTime(perf.avg_processing_ms)],
    ["Top type", data.violations_by_type?.[0]?.type || "-"],
  ];
  $("#analyticsStats").innerHTML = cards.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  const max = Math.max(...(data.violations_by_type || []).map((x) => x.count), 1);
  $("#typeBars").innerHTML = (data.violations_by_type || [])
    .map(
      (item) => `<div class="bar-row">
        <span>${item.type}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(item.count / max) * 100}%"></div></div>
        <strong>${item.count}</strong>
      </div>`
    )
    .join("") || `<p class="muted">No violation data yet.</p>`;
  loadAnalyticsBrief().catch((error) => {
    $("#briefSummary").textContent = `Could not generate brief: ${error.message}`;
  });
}

function renderBriefList(selector, items) {
  const list = Array.isArray(items) ? items : [];
  $(selector).innerHTML = list.length
    ? list.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : `<li>No insight yet. Analyze and approve more evidence.</li>`;
}

async function loadAnalyticsBrief() {
  $("#briefHeadline").textContent = "Generating enforcement brief";
  $("#briefSummary").textContent = "Reading current records, review status, throughput, and top violation trends...";
  $("#briefProvider").textContent = "working";
  const data = await requestJson("/api/analytics/brief");
  const brief = data.brief || {};
  $("#briefHeadline").textContent = brief.headline || "Gridlock AI enforcement brief";
  $("#briefSummary").textContent = brief.executive_summary || "No analytics summary available yet.";
  $("#briefProvider").textContent = data.provider || "local";
  $("#briefProvider").title = data.generated_at ? `Generated ${new Date(data.generated_at).toLocaleString()}` : "";
  renderBriefList("#briefActions", brief.priority_actions);
  renderBriefList("#briefRisks", brief.risk_signals);
  renderBriefList("#briefOps", brief.operations);
  renderBriefList("#briefNotes", brief.presentation_notes);
}

function renderOperationalBadges(operational = {}) {
  const items = [
    ["Detector", operational.detector_ready],
    ["OCR", operational.ocr_ready],
    ["Human review", operational.review_enabled],
    ["ZIP batch", operational.batch_enabled],
  ];
  return `<div class="readiness-strip">${items
    .map(
      ([label, ready]) =>
        `<span class="readiness-pill ${ready ? "ready" : "pending"}">${escapeHtml(label)} · ${
          ready ? "ready" : "pending"
        }</span>`
    )
    .join("")}</div>`;
}

function renderCoverageList(coverage = []) {
  if (!coverage.length) return "";
  return `<section class="performance-block">
    <h4>Requirement Coverage</h4>
    <div class="feature-coverage-list">
      ${coverage
        .map(
          (item) => `<article class="feature-coverage-item ${escapeHtml(item.status || "pending")}">
            <span>${escapeHtml(item.status || "pending")}</span>
            <strong>${escapeHtml(item.feature || "-")}</strong>
            <p>${escapeHtml(item.detail || "")}</p>
          </article>`
        )
        .join("")}
    </div>
  </section>`;
}

function renderEvaluationReadiness(readiness = {}) {
  const metrics = readiness.required_metrics || ["accuracy", "precision", "recall", "f1_score", "mAP50"];
  return `<section class="performance-block">
    <h4>Formal Evaluation Path</h4>
    <div class="evaluation-callout ${readiness.ground_truth_available ? "ready" : "pending"}">
      <strong>${readiness.ground_truth_available ? "Ground-truth run recorded" : "Ground-truth labels still required"}</strong>
      <p>Metrics covered: ${metrics.map(escapeHtml).join(", ")}.</p>
      <code>${escapeHtml(readiness.command || "python backend/scripts/evaluate_dataset.py --write-db")}</code>
    </div>
  </section>`;
}

async function loadPerformance() {
  const data = await requestJson("/api/performance");
  const throughput = data.throughput || {};
  const review = data.review || {};
  const detector = data.detector || {};
  const ocr = data.ocr || {};
  const detectorName = compactDetectorName(detector.model || detector.backend || "-");
  const ocrState = ocr.engine || ocr.status || (ocr.enabled ? "enabled" : "not enabled");
  const readiness = data.evaluation_readiness || {};
  const qualityCards = [
    ["Detector", detectorName],
    ["OCR", ocrState],
    ["Processed", throughput.total_images_processed || 0],
    ["Avg time", fmtTime(throughput.avg_processing_ms)],
    ["Throughput", `${throughput.images_per_minute || 0}/min`],
    ["Approved", review.approved || 0],
    ["Pending review", review.pending || 0],
    ["Rejected false IDs", review.rejected || 0],
  ];

  if (data.status === "needs_ground_truth") {
    $("#performancePanel").innerHTML = `<div class="panel-header performance-header">
        <div>
          <p class="panel-kicker">Operational readiness</p>
          <h3>Performance Console</h3>
        </div>
        <span class="status-pill pending">Formal metrics pending</span>
      </div>
      <p class="muted">${escapeHtml(data.message)}</p>
      ${renderOperationalBadges(data.operational)}
      <div class="pipeline-grid performance-grid">
        ${qualityCards.map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
      </div>
      ${renderCoverageList(data.coverage)}
      ${renderEvaluationReadiness(readiness)}`;
    return;
  }
  const overall = data.overall || {};
  const metricCards = [
    ["Accuracy", fmtPct(overall.accuracy)],
    ["Precision", fmtPct(overall.precision)],
    ["Recall", fmtPct(overall.recall)],
    ["F1", fmtPct(overall.f1_score)],
    ["mAP50", fmtPct(overall.mAP50)],
    ["Evaluated", data.created_at ? new Date(data.created_at).toLocaleString() : "-"],
    ["Dataset", data.dataset_name || "Evaluation"],
    ["Processed", throughput.total_images_processed || 0],
  ];
  $("#performancePanel").innerHTML = `<div class="panel-header performance-header">
      <div>
        <p class="panel-kicker">Ground-truth evaluation</p>
        <h3>${escapeHtml(data.dataset_name || "Evaluation")}</h3>
      </div>
      <span class="status-pill approved">Evaluated</span>
    </div>
    ${renderOperationalBadges(data.operational)}
    <div class="pipeline-grid performance-grid">
      ${metricCards.map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>
    <section class="performance-block">
      <h4>Operational Quality</h4>
      <div class="pipeline-grid performance-grid compact">
        ${qualityCards.map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
      </div>
    </section>
    ${renderCoverageList(data.coverage)}
    ${renderEvaluationReadiness(readiness)}`;
}

async function saveKey() {
  const key = $("#apiKeyInput").value.trim();
  if (!key) return;
  $("#configMessage").textContent = "Saving key...";
  try {
    const result = await requestJson(`/api/config?api_key=${encodeURIComponent(key)}`, { method: "POST" });
    $("#configMessage").textContent = result.message || "Configured.";
    loadStatus();
  } catch (error) {
    $("#configMessage").textContent = `Configuration failed: ${error.message}`;
  }
}

function numberValue(selector, fallback) {
  const value = Number($(selector).value);
  return Number.isFinite(value) ? value : fallback;
}

function calibrationPayloadFromInputs() {
  const roi = {
    x: numberValue("#roiX", 10),
    y: numberValue("#roiY", 60),
    w: numberValue("#roiW", 35),
    h: numberValue("#roiH", 25),
  };
  return {
    stop_line_rule_enabled: $("#stopLineEnabled").checked,
    red_light_rule_enabled: $("#redLightEnabled").checked,
    illegal_parking_rule_enabled: $("#parkingEnabled").checked,
    wrong_side_enabled: $("#wrongSideEnabled").checked,
    stop_line_y_percent: numberValue("#stopLineY", 60),
    red_light_vehicle_bottom_y_percent: numberValue("#redLightY", 52),
    expected_lane_direction: $("#laneDirection").value,
    traffic_light_state_override: $("#signalOverride").value,
    illegal_parking_rois: $("#parkingEnabled").checked ? [roi] : [],
    illegal_parking_dwell_seconds: numberValue("#parkingDwellSeconds", 8),
    tracking_sample_fps: numberValue("#trackingSampleFps", 1.5),
    tracking_max_frames: numberValue("#trackingMaxFrames", 90),
    wrong_side_min_travel_percent: numberValue("#wrongSideMinTravel", 8),
  };
}

function calibrationPercentFromEvent(event) {
  const rect = $("#calibrationPreviewShell").getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)),
    y: Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100)),
  };
}

function setNumber(selector, value) {
  $(selector).value = String(Math.round(value * 10) / 10);
}

function refreshCalibrationPreview(tempRoi = null) {
  const layer = $("#calibrationOverlay");
  if (!layer) return;
  const stopY = numberValue("#stopLineY", 60);
  const redY = numberValue("#redLightY", 52);
  const roi = tempRoi || {
    x: numberValue("#roiX", 10),
    y: numberValue("#roiY", 60),
    w: numberValue("#roiW", 35),
    h: numberValue("#roiH", 25),
  };
  const pieces = [];
  if ($("#stopLineEnabled").checked) {
    pieces.push(`<div class="calibration-line stop" style="top:${stopY}%"><span>Stop ${Math.round(stopY)}%</span></div>`);
  }
  if ($("#redLightEnabled").checked) {
    pieces.push(`<div class="calibration-line red" style="top:${redY}%"><span>Red ${Math.round(redY)}%</span></div>`);
  }
  if ($("#parkingEnabled").checked || tempRoi) {
    pieces.push(`<div class="calibration-roi" style="left:${roi.x}%;top:${roi.y}%;width:${roi.w}%;height:${roi.h}%"><span>Parking ROI</span></div>`);
  }
  layer.innerHTML = pieces.join("");
}

function handleCalibrationPreview(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (state.calibrationPreviewUrl) URL.revokeObjectURL(state.calibrationPreviewUrl);
  state.calibrationPreviewUrl = URL.createObjectURL(file);
  $("#calibrationPreviewImage").src = state.calibrationPreviewUrl;
  $("#calibrationPreviewShell").classList.remove("hidden");
  refreshCalibrationPreview();
}

function renderCalibration(calibration) {
  const roi = calibration.illegal_parking_rois?.[0] || { x: 10, y: 60, w: 35, h: 25 };
  $("#stopLineEnabled").checked = Boolean(calibration.stop_line_rule_enabled);
  $("#redLightEnabled").checked = Boolean(calibration.red_light_rule_enabled);
  $("#parkingEnabled").checked = Boolean(calibration.illegal_parking_rule_enabled);
  $("#wrongSideEnabled").checked = Boolean(calibration.wrong_side_enabled);
  $("#stopLineY").value = calibration.stop_line_y_percent ?? 60;
  $("#redLightY").value = calibration.red_light_vehicle_bottom_y_percent ?? 52;
  $("#laneDirection").value = calibration.expected_lane_direction || "left_to_right";
  $("#signalOverride").value = calibration.traffic_light_state_override || "auto";
  $("#roiX").value = roi.x ?? 10;
  $("#roiY").value = roi.y ?? 60;
  $("#roiW").value = roi.w ?? 35;
  $("#roiH").value = roi.h ?? 25;
  $("#parkingDwellSeconds").value = calibration.illegal_parking_dwell_seconds ?? 8;
  $("#trackingSampleFps").value = calibration.tracking_sample_fps ?? 1.5;
  $("#trackingMaxFrames").value = calibration.tracking_max_frames ?? 90;
  $("#wrongSideMinTravel").value = calibration.wrong_side_min_travel_percent ?? 8;
  refreshCalibrationPreview();
}

async function loadCalibration() {
  try {
    const data = await requestJson("/api/calibration");
    renderCalibration(data.calibration || {});
    $("#calibrationMessage").textContent = "Calibration loaded.";
  } catch (error) {
    $("#calibrationMessage").textContent = `Could not load calibration: ${error.message}`;
  }
}

async function saveCalibration() {
  const payload = calibrationPayloadFromInputs();
  $("#calibrationMessage").textContent = "Saving calibration...";
  try {
    const data = await requestJson("/api/calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderCalibration(data.calibration || {});
    $("#calibrationMessage").textContent = "Calibration saved. New analyses will use these rules.";
  } catch (error) {
    $("#calibrationMessage").textContent = `Save failed: ${error.message}`;
  }
}

function init() {
  violationTypes.forEach((type) => {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    $("#recordFilter").appendChild(option);
  });

  $$(".mode-tab").forEach((button) => button.addEventListener("click", () => setUploadMode(button.dataset.uploadMode)));
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
  $("#refreshStatus").addEventListener("click", loadStatus);
  $("#loadRecords").addEventListener("click", loadRecords);
  $("#loadAnalytics").addEventListener("click", loadAnalytics);
  $("#loadBrief").addEventListener("click", () => loadAnalyticsBrief().catch((error) => {
    $("#briefSummary").textContent = `Could not generate brief: ${error.message}`;
  }));
  $("#loadPerformance").addEventListener("click", loadPerformance);
  $("#approveSessionBtn").addEventListener("click", approveCurrentSession);
  $("#resetReviewBtn").addEventListener("click", resetSingleReview);
  $("#approveBatchBtn").addEventListener("click", approveBatch);
  $("#resultList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-review-action='reject-single']");
    if (button) rejectSingleViolation(button.dataset.recordId);
  });
  $("#batchResults").addEventListener("click", (event) => {
    const button = event.target.closest("[data-review-action='reject-batch']");
    if (button) rejectBatchViolation(button.dataset.recordId);
  });
  $("#saveKey").addEventListener("click", saveKey);
  $("#loadCalibration").addEventListener("click", loadCalibration);
  $("#saveCalibration").addEventListener("click", saveCalibration);
  $("#analyzeVideoBtn").addEventListener("click", analyzeVideo);
  $("#clearVideo").addEventListener("click", clearVideo);
  $("#videoDropZone").addEventListener("click", () => $("#videoFileInput").click());
  $("#videoFileInput").addEventListener("change", (event) => handleVideoFile(event.target.files?.[0]));
  $("#chooseCalibrationPreview").addEventListener("click", () => $("#calibrationPreviewInput").click());
  $("#calibrationPreviewInput").addEventListener("change", (event) => handleCalibrationPreview(event.target.files?.[0]));

  [
    "#stopLineEnabled",
    "#redLightEnabled",
    "#parkingEnabled",
    "#stopLineY",
    "#redLightY",
    "#roiX",
    "#roiY",
    "#roiW",
    "#roiH",
  ].forEach((selector) => $(selector).addEventListener("input", () => refreshCalibrationPreview()));

  $("#calibrationPreviewShell").addEventListener("pointerdown", (event) => {
    if ($("#calibrationPreviewShell").classList.contains("hidden")) return;
    const point = calibrationPercentFromEvent(event);
    const tool = $("#calibrationTool").value;
    if (tool === "stop") {
      setNumber("#stopLineY", point.y);
      $("#stopLineEnabled").checked = true;
      refreshCalibrationPreview();
      return;
    }
    if (tool === "red") {
      setNumber("#redLightY", point.y);
      $("#redLightEnabled").checked = true;
      refreshCalibrationPreview();
      return;
    }
    state.calibrationDrag = point;
    $("#calibrationPreviewShell").setPointerCapture(event.pointerId);
  });

  $("#calibrationPreviewShell").addEventListener("pointermove", (event) => {
    if (!state.calibrationDrag || $("#calibrationTool").value !== "parking") return;
    const point = calibrationPercentFromEvent(event);
    const x = Math.min(state.calibrationDrag.x, point.x);
    const y = Math.min(state.calibrationDrag.y, point.y);
    const roi = {
      x,
      y,
      w: Math.max(0.5, Math.abs(point.x - state.calibrationDrag.x)),
      h: Math.max(0.5, Math.abs(point.y - state.calibrationDrag.y)),
    };
    refreshCalibrationPreview(roi);
  });

  $("#calibrationPreviewShell").addEventListener("pointerup", (event) => {
    if (!state.calibrationDrag || $("#calibrationTool").value !== "parking") return;
    const point = calibrationPercentFromEvent(event);
    const x = Math.min(state.calibrationDrag.x, point.x);
    const y = Math.min(state.calibrationDrag.y, point.y);
    setNumber("#roiX", x);
    setNumber("#roiY", y);
    setNumber("#roiW", Math.max(0.5, Math.abs(point.x - state.calibrationDrag.x)));
    setNumber("#roiH", Math.max(0.5, Math.abs(point.y - state.calibrationDrag.y)));
    $("#parkingEnabled").checked = true;
    state.calibrationDrag = null;
    refreshCalibrationPreview();
  });

  $("#recordSearch").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadRecords();
  });
  $("#recordFilter").addEventListener("change", loadRecords);
  $("#analyzeBtn").addEventListener("click", analyze);
  $("#clearImage").addEventListener("click", clearImage);
  $("#dropZone").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", (event) => handleFile(event.target.files?.[0]));
  $("#dropZone").addEventListener("dragover", (event) => {
    event.preventDefault();
    $("#dropZone").classList.add("drag");
  });
  $("#dropZone").addEventListener("dragleave", () => $("#dropZone").classList.remove("drag"));
  $("#dropZone").addEventListener("drop", (event) => {
    event.preventDefault();
    $("#dropZone").classList.remove("drag");
    handleFile(event.dataTransfer.files?.[0]);
  });
  $("#videoDropZone").addEventListener("dragover", (event) => {
    event.preventDefault();
    $("#videoDropZone").classList.add("drag");
  });
  $("#videoDropZone").addEventListener("dragleave", () => $("#videoDropZone").classList.remove("drag"));
  $("#videoDropZone").addEventListener("drop", (event) => {
    event.preventDefault();
    $("#videoDropZone").classList.remove("drag");
    handleVideoFile(event.dataTransfer.files?.[0]);
  });

  loadStatus();
  loadAnalytics().catch(() => {});
  loadCalibration().catch(() => {});
}

init();
