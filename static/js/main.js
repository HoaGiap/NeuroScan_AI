// ─── NEUROSCAN AI — CLINICAL SUITE & MINIMALIST UI ENGINE ─────────────────────
const API_BASE = "http://127.0.0.1:5000";

const PASTEL_COLORS = {
  glioma: "#fca5a5",
  meningioma: "#fde047",
  pituitary: "#e9d5ff",
  notumor: "#6ee7b7",
};

const CLASS_VI = {
  glioma: "U thần kinh đệm",
  meningioma: "U màng não",
  pituitary: "U tuyến yên",
  notumor: "Không có u",
};

const CLASS_MEDICAL_TERMS = {
  glioma: "Glioma (U tế bào thần kinh đệm nội sọ)",
  meningioma: "Meningioma (U màng não ngoài trục)",
  pituitary: "Pituitary Adenoma (U tuyến yên vùng hố yên)",
  notumor: "Normal Brain Tissue (Không phát hiện tổn thương)",
};

let currentFile = null;
let currentFiles = [];
let currentSliceIndex = 0;
let batchAnalysisData = null;
let currentGradcamData = {};
let currentOriginalImgUrl = "";
let currentGradcamViewMode = "overlay";
let lastAnalysisData = null;

// Image Viewport Tool Settings
let imgZoom = 1.0;
let imgInvert = false;

// DOM Elements
const dropZone = document.getElementById("dropZone");
const imgWrapper = document.getElementById("imgWrapper");
const previewImg = document.getElementById("previewImg");
const btnRow = document.getElementById("btnRow");
const analyzeBtn = document.getElementById("analyzeBtn");
const scanOverlay = document.getElementById("scanOverlay");
const errorBox = document.getElementById("errorBox");
const imgBadge = document.getElementById("imgBadge");
const windowStatus = document.getElementById("windowStatus");
const medicalToolbar = document.getElementById("medicalToolbar");
const reportActionButtons = document.getElementById("reportActionButtons");
const sliceFilmstripContainer = document.getElementById("sliceFilmstripContainer");
const filmstripScroll = document.getElementById("filmstripScroll");
const sliceCountLabel = document.getElementById("sliceCountLabel");
const filmstripSummaryBadge = document.getElementById("filmstripSummaryBadge");

// Mouse wheel horizontal scroll on filmstrip
if (filmstripScroll) {
  filmstripScroll.addEventListener("wheel", (e) => {
    if (e.deltaY !== 0) {
      e.preventDefault();
      filmstripScroll.scrollLeft += e.deltaY;
    }
  }, { passive: false });
}

// Global Keyboard Shortcut (⌘ + U or Ctrl + U)
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "u") {
    e.preventDefault();
    document.getElementById("fileInput").click();
  }
});

// Drag and Drop
if (dropZone) {
  ["dragover", "dragenter"].forEach((e) =>
    dropZone.addEventListener(e, (ev) => {
      ev.preventDefault();
      dropZone.style.borderColor = "#94a3b8";
    })
  );
  ["dragleave", "drop"].forEach((e) =>
    dropZone.addEventListener(e, () => {
      dropZone.style.borderColor = "";
    })
  );
  dropZone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const files = ev.dataTransfer.files;
    if (files && files.length > 0) handleFiles(files);
  });
}

function handleFile(file) {
  if (file) handleFiles([file]);
}

async function handleFiles(filesList) {
  if (!filesList || filesList.length === 0) return;

  const validFiles = Array.from(filesList).filter((f) => {
    const isDicom = f.name.toLowerCase().endsWith(".dcm");
    const isImage = f.type.startsWith("image/") || /\.(png|jpe?g|bmp|tiff?)$/i.test(f.name);
    return isDicom || isImage;
  });

  if (validFiles.length === 0) {
    showError("Vui lòng chọn các tệp ảnh MRI hợp lệ (JPG, PNG, BMP, TIFF) hoặc file DICOM (.dcm)");
    return;
  }

  clearError();
  resetResults();
  resetImageFilters();

  currentFiles = validFiles;
  currentSliceIndex = 0;
  batchAnalysisData = null;
  currentFile = currentFiles[0];

  dropZone.style.display = "none";
  imgWrapper.classList.add("visible");
  imgBadge.style.display = "none";
  btnRow.style.display = "flex";
  if (medicalToolbar) medicalToolbar.style.display = "flex";

  if (currentFiles.length > 1) {
    // Multi-slice mode
    if (sliceFilmstripContainer) sliceFilmstripContainer.style.display = "block";
    if (sliceCountLabel) sliceCountLabel.textContent = currentFiles.length;
    if (filmstripSummaryBadge) filmstripSummaryBadge.textContent = "SẴN SÀNG QUÉT CHUỖI";
    if (windowStatus) windowStatus.textContent = `CHUỖI MRI (${currentFiles.length} LÁT CẮT)`;
    analyzeBtn.innerHTML = `<i class="fa-solid fa-layer-group"></i> Phân tích chuỗi (${currentFiles.length} lát)`;
    renderInitialFilmstrip();
    displaySlice(0);
  } else {
    // Single slice mode
    if (sliceFilmstripContainer) sliceFilmstripContainer.style.display = "none";
    if (windowStatus) windowStatus.textContent = currentFile.name.toUpperCase();
    analyzeBtn.innerHTML = `<i class="fa-solid fa-microscope"></i> Phân tích MRI`;
    displaySlice(0);
  }
}

function renderInitialFilmstrip() {
  if (!filmstripScroll) return;
  filmstripScroll.innerHTML = "";

  currentFiles.forEach((file, idx) => {
    const card = document.createElement("div");
    card.className = `slice-card ${idx === 0 ? "active" : ""}`;
    card.id = `slice-card-${idx}`;
    card.onclick = () => selectSlice(idx);

    const isDcm = file.name.toLowerCase().endsWith(".dcm");
    const thumbUrl = isDcm ? "/static/favicon.png" : URL.createObjectURL(file);

    card.innerHTML = `
      <div class="slice-thumb-wrap">
        <img src="${thumbUrl}" alt="Slice ${idx + 1}" />
      </div>
      <div class="slice-meta">
        <span class="slice-num">#${idx + 1}</span>
        <span class="slice-tag tag-pending" id="tag-${idx}">Chờ quét</span>
      </div>
    `;
    filmstripScroll.appendChild(card);
  });
}

function displaySlice(index) {
  if (index < 0 || index >= currentFiles.length) return;
  currentSliceIndex = index;
  currentFile = currentFiles[index];

  // Update active state in filmstrip
  document.querySelectorAll(".slice-card").forEach((c, i) => {
    c.classList.toggle("active", i === index);
  });

  const isDicom = currentFile.name.toLowerCase().endsWith(".dcm");

  if (batchAnalysisData && batchAnalysisData.slices && batchAnalysisData.slices[index]) {
    // If already analyzed, show analyzed preview & gradcam
    const sliceData = batchAnalysisData.slices[index];
    currentOriginalImgUrl = sliceData.preview_image;
    previewImg.src = currentOriginalImgUrl;
    currentGradcamData = sliceData.gradcam || {};
    renderGradcam();
    if (windowStatus) {
      windowStatus.textContent = `LÁT #${index + 1}: ${sliceData.filename.toUpperCase()} — ${sliceData.class_vi.toUpperCase()} (${sliceData.confidence}%)`;
    }
  } else {
    // Initial display
    if (isDicom) {
      previewDicomFile(currentFile);
    } else {
      currentOriginalImgUrl = URL.createObjectURL(currentFile);
      previewImg.src = currentOriginalImgUrl;
      if (windowStatus && currentFiles.length > 1) {
        windowStatus.textContent = `LÁT CẮT #${index + 1}: ${currentFile.name.toUpperCase()}`;
      }
    }
  }
}

function selectSlice(index) {
  displaySlice(index);
}

async function previewDicomFile(file) {
  try {
    const formData = new FormData();
    formData.append("image", file);
    if (windowStatus) windowStatus.textContent = "ĐANG GIẢI MÃ DICOM...";

    const res = await fetch(`${API_BASE}/preview_dicom`, {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const dcmData = await res.json();
      currentOriginalImgUrl = dcmData.preview_image;
      previewImg.src = currentOriginalImgUrl;
      if (windowStatus) windowStatus.textContent = file.name.toUpperCase() + " (DICOM)";
      if (dcmData.metadata) fillPatientMetadata(dcmData.metadata);
    } else {
      currentOriginalImgUrl = "";
      previewImg.src = "";
    }
  } catch (err) {
    console.warn("Lỗi preview DICOM:", err);
  }
}

function fillPatientMetadata(meta) {
  if (meta.patient_id && meta.patient_id.trim()) {
    document.getElementById("patientId").value = meta.patient_id.trim();
  }
  if (meta.patient_name && meta.patient_name.trim()) {
    document.getElementById("patientName").value = meta.patient_name.trim();
  }
  if (meta.patient_age && meta.patient_age.trim()) {
    document.getElementById("patientAge").value = meta.patient_age.trim().replace(/^0+/, "") + " tuổi";
  }
  if (meta.patient_sex && meta.patient_sex.trim()) {
    const s = meta.patient_sex.trim().toUpperCase();
    document.getElementById("patientGender").value = (s === "M" || s === "NAM") ? "Nam" : "Nữ";
  }
  if (meta.modality && meta.modality.trim()) {
    document.getElementById("patientModality").value = `${meta.modality.trim()} Sọ Não (Chuỗi MRI)`;
  }
}

// Preset Sample Loader
async function loadSample(className) {
  clearError();
  resetResults();
  resetImageFilters();

  const samplePath = `/static/samples/${className}.jpg`;
  try {
    const res = await fetch(samplePath);
    if (!res.ok) throw new Error("Không thể tải ảnh mẫu.");
    const blob = await res.blob();
    const file = new File([blob], `${className}_sample.jpg`, { type: "image/jpeg" });
    
    generateSamplePatient(className);
    handleFiles([file]);
    analyze();
  } catch (err) {
    showError(`Lỗi tải ảnh mẫu (${className}): ${err.message}`);
  }
}

function resetAll() {
  currentFile = null;
  currentFiles = [];
  currentSliceIndex = 0;
  batchAnalysisData = null;
  currentOriginalImgUrl = "";
  lastAnalysisData = null;
  previewImg.src = "";
  dropZone.style.display = "block";
  imgWrapper.classList.remove("visible");
  btnRow.style.display = "none";
  if (sliceFilmstripContainer) sliceFilmstripContainer.style.display = "none";
  if (medicalToolbar) medicalToolbar.style.display = "none";
  if (reportActionButtons) reportActionButtons.style.display = "none";
  document.getElementById("fileInput").value = "";
  if (windowStatus) windowStatus.textContent = "CHỜ TỆP ẢNH / DICOM";
  resetResults();
  resetImageFilters();
  clearError();
}

function resetResults() {
  document.getElementById("resultBody").innerHTML = `
  <div class="editorial-empty-state">
    <div class="empty-icon"><i class="fa-solid fa-file-waveform"></i></div>
    <h3 class="empty-title">Chưa có dữ liệu phân tích</h3>
    <p class="empty-desc">Tải ảnh MRI (hoặc DICOM .dcm) hoặc chọn mẫu thử nhanh để bắt đầu quy trình chẩn đoán AI.</p>
  </div>`;

  const gradcamCard = document.getElementById("gradcamCard");
  if (gradcamCard) gradcamCard.style.display = "none";
  if (reportActionButtons) reportActionButtons.style.display = "none";
  imgBadge.style.display = "none";
  scanOverlay.classList.remove("active");
}

// ─── ANALYSIS API CALL (SINGLE & BATCH) ────────────────────────────────────────
async function analyze() {
  if (!currentFiles || currentFiles.length === 0) return;
  setLoading(true);

  if (currentFiles.length > 1) {
    // ─── BATCH MULTI-SLICE PROCESSING ─────────────────────────────────────────
    const formData = new FormData();
    currentFiles.forEach((f) => formData.append("images", f));

    try {
      const resp = await fetch(`${API_BASE}/predict-batch?gradcam_model=both`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        let errMsg = `HTTP Error ${resp.status}`;
        try {
          const err = await resp.json();
          errMsg = err.error || errMsg;
        } catch (_) {
          if (resp.status === 404) {
            errMsg = "Chưa tìm thấy endpoint /predict-batch (Lỗi 404). Vui lòng nhấn Ctrl + C trong Terminal để tắt Flask và chạy lại: python app.py";
          } else if (resp.status === 413) {
            errMsg = "Dung lượng chuỗi ảnh quá lớn (Lỗi 413). Hãy giảm bớt số lượng ảnh tải lên.";
          } else {
            errMsg = `Lỗi từ server Flask (Mã lỗi ${resp.status}). Hãy kiểm tra log trong Terminal.`;
          }
        }
        throw new Error(errMsg);
      }

      const data = await resp.json();
      batchAnalysisData = data;
      renderBatchResults(data);
    } catch (e) {
      showError(`Lỗi khi quét chuỗi lát cắt: ${e.message}`);
      setLoading(false);
    }
  } else {
    // ─── SINGLE SLICE PROCESSING ──────────────────────────────────────────────
    const formData = new FormData();
    formData.append("image", currentFile);

    try {
      const resp = await fetch(`${API_BASE}/predict?gradcam_model=both`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        let errMsg = `HTTP Error ${resp.status}`;
        try {
          const err = await resp.json();
          errMsg = err.error || errMsg;
        } catch (_) {
          errMsg = `Lỗi từ server Flask (Mã lỗi ${resp.status}).`;
        }
        throw new Error(errMsg);
      }

      const data = await resp.json();
      if (data.preview_image && (!currentOriginalImgUrl || currentFile.name.toLowerCase().endsWith(".dcm"))) {
        currentOriginalImgUrl = data.preview_image;
        previewImg.src = currentOriginalImgUrl;
      }
      if (data.metadata) fillPatientMetadata(data.metadata);

      lastAnalysisData = data;
      renderResults(data);
    } catch (e) {
      showError(`Không thể kết nối đến server AI: ${e.message}`);
      setLoading(false);
    }
  }
}

function renderBatchResults(data) {
  setLoading(false);
  const summary = data.patient_summary;
  const cls = summary.class;
  const color = PASTEL_COLORS[cls] || "#f1f5f9";
  const sev = summary.severity;
  const total = data.total_slices;
  const tumorCount = summary.tumor_slice_count;
  const keyIdx = data.key_slice_index; // 1-based

  // Update badge
  imgBadge.style.display = "block";
  imgBadge.className = `status-overlay-tag ${summary.has_tumor ? "tag-positive" : "tag-negative"}`;
  imgBadge.textContent = summary.has_tumor ? `PHÁT HIỆN U (${tumorCount}/${total} LÁT)` : "CHUỖI BÌNH THƯỜNG";

  if (filmstripSummaryBadge) {
    filmstripSummaryBadge.textContent = summary.has_tumor ? `Phát hiện ${tumorCount}/${total} lát có u` : `Toàn bộ ${total} lát bình thường`;
  }

  if (reportActionButtons) reportActionButtons.style.display = "flex";

  // Update Filmstrip cards with analyzed results
  data.slices.forEach((s, i) => {
    const card = document.getElementById(`slice-card-${i}`);
    if (card) {
      card.classList.toggle("is-tumor", s.has_tumor);
      card.classList.toggle("is-key", s.index === keyIdx);
      const tag = document.getElementById(`tag-${i}`);
      if (tag) {
        tag.className = `slice-tag tag-${s.class}`;
        tag.textContent = `${s.class_vi} (${Math.round(s.confidence)}%)`;
      }
    }
  });

  // Automatically activate Key-Slice
  const key0Index = keyIdx - 1;
  displaySlice(key0Index);

  // 1. Render Summary Card in Right Column
  const keySlice = data.slices[key0Index];
  let html = `
  <div class="result-card-editorial">
    <div class="editorial-hero-banner bg-${cls}">
      <div class="diag-top-meta">
        <div>
          <div class="tag-label">KẾT LUẬN TOÀN BỘ CA BỆNH (${total} LÁT CẮT)</div>
          <h2 class="diag-serif-heading c-${cls}">${summary.class_vi}</h2>
          <div class="diag-sub-meta">
            ${summary.has_tumor ? `Tổn thương xuất hiện tại ${tumorCount}/${total} lát (${summary.tumor_slice_percentage}%)` : `Toàn bộ ${total} lát cắt không phát hiện khối u`}
          </div>
        </div>
        <div class="severity-pill sev-${sev.level}">
          ${sev.label.toUpperCase()}
        </div>
      </div>
    </div>

    <!-- Batch Series Metrics Box -->
    <div class="batch-summary-banner">
      <div class="batch-summary-row">
        <span class="batch-summary-label"><i class="fa-solid fa-layer-group"></i> Tổng số lát cắt:</span>
        <span class="batch-summary-val">${total} lát cắt</span>
      </div>
      <div class="batch-summary-row">
        <span class="batch-summary-label"><i class="fa-solid fa-circle-radiation"></i> Lát cắt tiêu biểu (Key-Slice):</span>
        <span class="batch-summary-val" style="color:${color}">Lát #${keyIdx} (${keySlice.filename}) &bull; ${keySlice.confidence}%</span>
      </div>
      <div class="batch-summary-row">
        <span class="batch-summary-label"><i class="fa-solid fa-chart-pie"></i> Phân bổ nhãn phát hiện:</span>
        <span class="batch-summary-val">${Object.entries(summary.class_counts).filter(([_, v]) => v > 0).map(([k, v]) => `${CLASS_VI[k]}: ${v}`).join(" | ")}</span>
      </div>
    </div>`;

  // 2. Key-Slice Confidence Spectrum Bars
  html += `
  <div class="conf-block">
    <div class="conf-block-title">XÁC SUẤT TẠI LÁT CẮT TIÊU BIỂU (#${keyIdx})</div>`;

  const sorted = Object.entries(keySlice.probabilities).sort((a, b) => b[1].score_pct - a[1].score_pct);
  for (const [cn, info] of sorted) {
    const c = PASTEL_COLORS[cn] || "#f1f5f9";
    html += `
    <div class="conf-item-row">
      <div class="conf-item-header">
        <span class="conf-item-name">${info.label_vi}</span>
        <span class="conf-item-pct" style="color:${c}">${info.score_pct}%</span>
      </div>
      <div class="conf-item-track">
        <div class="conf-item-fill" id="fill-${cn}" style="background:${c}"></div>
      </div>
    </div>`;
  }
  html += `</div>`;

  // 3. Clinical Recommendation Box
  html += `
  <div class="recom-box">
    <div class="recom-heading">
      <i class="fa-solid fa-compass"></i> KHUYẾN NGHỊ LÂM SÀNG
    </div>
    <p class="recom-body">${summary.recommendation}</p>
  </div>
  </div>`;

  document.getElementById("resultBody").innerHTML = html;

  requestAnimationFrame(() => {
    setTimeout(() => {
      for (const [cn, info] of sorted) {
        const el = document.getElementById(`fill-${cn}`);
        if (el) el.style.width = `${info.score_pct}%`;
      }
    }, 50);
  });

  // Prepare Last Analysis Data for PDF (format matching single predict)
  lastAnalysisData = {
    prediction: {
      class: summary.class,
      class_vi: summary.class_vi,
      confidence: summary.confidence,
      has_tumor: summary.has_tumor,
    },
    probabilities: keySlice.probabilities,
    recommendation: summary.recommendation,
    severity: summary.severity,
    gradcam: keySlice.gradcam,
    is_batch: true,
    total_slices: total,
    tumor_count: tumorCount,
    key_slice_idx: keyIdx,
    key_slice_name: keySlice.filename,
  };

  updateMedicalReportTemplate(lastAnalysisData);
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  if (on) {
    scanOverlay.classList.add("active");
    analyzeBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Đang quét ${currentFiles.length > 1 ? currentFiles.length + " lát..." : "MRI..."}`;
    document.getElementById("resultBody").innerHTML = `
    <div class="editorial-loading">
      <div class="loading-spinner-circle"></div>
      <div class="loading-title">Mạng nơ-ron đang trích xuất đặc trưng...</div>
      <div class="mono-label">${currentFiles.length > 1 ? "Đang xử lý chuỗi " + currentFiles.length + " lát cắt MRI" : "EfficientNet-V2-S &bull; Grad-CAM++"}</div>
    </div>`;
  } else {
    scanOverlay.classList.remove("active");
    if (currentFiles && currentFiles.length > 1) {
      analyzeBtn.innerHTML = `<i class="fa-solid fa-layer-group"></i> Phân tích chuỗi (${currentFiles.length} lát)`;
    } else {
      analyzeBtn.innerHTML = `<i class="fa-solid fa-microscope"></i> Phân tích MRI`;
    }
  }
}

// ─── SINGLE SLICE RESULT RENDERER ─────────────────────────────────────────────
function renderResults(data) {
  setLoading(false);
  const p = data.prediction;
  const cls = p.class;
  const color = PASTEL_COLORS[cls] || "#f1f5f9";
  const sev = data.severity;

  // Viewport status badge overlay
  imgBadge.style.display = "block";
  imgBadge.className = `status-overlay-tag ${p.has_tumor ? "tag-positive" : "tag-negative"}`;
  imgBadge.textContent = p.has_tumor ? "PHÁT HIỆN BẤT THƯỜNG" : "BÌNH THƯỜNG";

  // Reveal Report Action Buttons (PDF / Print)
  if (reportActionButtons) {
    reportActionButtons.style.display = "flex";
  }

  // 1. Editorial Hero Banner
  let html = `
  <div class="result-card-editorial">
    <div class="editorial-hero-banner bg-${cls}">
      <div class="diag-top-meta">
        <div>
          <div class="tag-label">KẾT QUẢ CHẨN ĐOÁN</div>
          <h2 class="diag-serif-heading c-${cls}">${p.class_vi}</h2>
          <div class="diag-sub-meta">${cls.toUpperCase()} &bull; <span style="color:${color}">${p.confidence}% tin cậy</span></div>
        </div>
        <div class="severity-pill sev-${sev.level}">
          ${sev.label.toUpperCase()}
        </div>
      </div>
    </div>`;

  // 2. Confidence Spectrum Bars
  html += `
  <div class="conf-block">
    <div class="conf-block-title">XÁC SUẤT BẤT THƯỜNG THEO LỚP</div>`;

  const sorted = Object.entries(data.probabilities).sort(
    (a, b) => b[1].score_pct - a[1].score_pct
  );

  for (const [cn, info] of sorted) {
    const c = PASTEL_COLORS[cn] || "#f1f5f9";
    html += `
    <div class="conf-item-row">
      <div class="conf-item-header">
        <span class="conf-item-name">${info.label_vi}</span>
        <span class="conf-item-pct" style="color:${c}">${info.score_pct}%</span>
      </div>
      <div class="conf-item-track">
        <div class="conf-item-fill" id="fill-${cn}" style="background:${c}"></div>
      </div>
    </div>`;
  }
  html += `</div>`;

  // 3. Clinical Recommendation Box
  html += `
  <div class="recom-box">
    <div class="recom-heading">
      <i class="fa-solid fa-compass"></i> KHUYẾN NGHỊ LÂM SÀNG
    </div>
    <p class="recom-body">${data.recommendation}</p>
  </div>
  </div>`;

  document.getElementById("resultBody").innerHTML = html;

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      for (const [cn, info] of sorted) {
        const el = document.getElementById(`fill-${cn}`);
        if (el) el.style.width = `${info.score_pct}%`;
      }
    }, 50);
  });

  // 4. Grad-CAM Visualization
  currentGradcamData = data.gradcam || {};
  renderGradcam();

  // 5. Update Hidden Report Data
  updateMedicalReportTemplate(data);
}

// ─── GRAD-CAM VISUALIZATION ───────────────────────────────────────────────────
function renderGradcam() {
  const keys = Object.keys(currentGradcamData).filter((k) => currentGradcamData[k]);
  if (!keys.length) return;

  const card = document.getElementById("gradcamCard");
  const tabs = document.getElementById("gradcamTabs");
  const img = document.getElementById("gradcamImg");
  const heatmapCompare = document.getElementById("gradcamHeatmapCompare");
  const originalCompare = document.getElementById("gradcamOriginalCompare");

  card.style.display = "block";
  tabs.innerHTML = "";

  const modelLabels = {
    resnet50: "ResNet50",
    efficientnet_v2_s: "EfficientNet-V2-S",
    efficientnet: "EfficientNet-B0",
    convnext_small: "ConvNeXt-Small",
  };

  keys.forEach((k, i) => {
    const btn = document.createElement("button");
    btn.className = `gtab ${i === 0 ? "active" : ""}`;
    btn.textContent = modelLabels[k] || k;
    btn.onclick = () => {
      document.querySelectorAll(".gtab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      img.src = currentGradcamData[k];
      if (heatmapCompare) heatmapCompare.src = currentGradcamData[k];
    };
    tabs.appendChild(btn);
  });

  const activeSrc = currentGradcamData[keys[0]];
  img.src = activeSrc;
  if (heatmapCompare) heatmapCompare.src = activeSrc;
  if (originalCompare) originalCompare.src = currentOriginalImgUrl || previewImg.src;

  setGradcamViewMode(currentGradcamViewMode);
}

function setGradcamViewMode(mode) {
  currentGradcamViewMode = mode;
  const singleDisp = document.getElementById("gradcamSingleDisplay");
  const splitDisp = document.getElementById("gradcamSplitDisplay");
  const overlayBtn = document.getElementById("viewOverlayBtn");
  const splitBtn = document.getElementById("viewSplitBtn");

  if (mode === "split") {
    singleDisp.style.display = "none";
    splitDisp.style.display = "grid";
    splitBtn.classList.add("active");
    overlayBtn.classList.remove("active");
  } else {
    singleDisp.style.display = "block";
    splitDisp.style.display = "none";
    overlayBtn.classList.add("active");
    splitBtn.classList.remove("active");
  }
}

// ─── MEDICAL VIEWPORT IMAGE MANIPULATION ───────────────────────────────────────
function applyImageFilters() {
  const b = document.getElementById("brightnessRange") ? document.getElementById("brightnessRange").value : 100;
  const c = document.getElementById("contrastRange") ? document.getElementById("contrastRange").value : 100;
  
  if (document.getElementById("brightnessVal")) {
    document.getElementById("brightnessVal").textContent = `${b}%`;
  }
  if (document.getElementById("contrastVal")) {
    document.getElementById("contrastVal").textContent = `${c}%`;
  }

  const invStr = imgInvert ? "invert(100%)" : "invert(0%)";
  previewImg.style.filter = `brightness(${b}%) contrast(${c}%) ${invStr}`;
  previewImg.style.transform = `scale(${imgZoom})`;
}

function toggleInvert() {
  imgInvert = !imgInvert;
  const btn = document.getElementById("invertBtn");
  if (btn) btn.classList.toggle("active", imgInvert);
  applyImageFilters();
}

function adjustZoom(delta) {
  imgZoom = Math.min(Math.max(imgZoom + delta, 0.6), 2.4);
  applyImageFilters();
}

function resetImageFilters() {
  imgZoom = 1.0;
  imgInvert = false;
  if (document.getElementById("brightnessRange")) document.getElementById("brightnessRange").value = 100;
  if (document.getElementById("contrastRange")) document.getElementById("contrastRange").value = 100;
  if (document.getElementById("brightnessVal")) document.getElementById("brightnessVal").textContent = "100%";
  if (document.getElementById("contrastVal")) document.getElementById("contrastVal").textContent = "100%";
  const btn = document.getElementById("invertBtn");
  if (btn) btn.classList.remove("active");
  applyImageFilters();
}

// ─── SAMPLE PATIENT GENERATOR ──────────────────────────────────────────────────
function generateSamplePatient(preferredClass) {
  const samplePatients = [
    {
      id: "BN-2026-0891",
      name: "Trần Thị Mai",
      age: "48 tuổi",
      gender: "Nữ",
      doctor: "BS. CKII Nguyễn Hoàng Nam",
      modality: "MRI Sọ Não Axial T1-CE",
      symptoms: "Đau đầu âm ỉ vùng trán đính 2 tháng, giảm thị lực mắt phải.",
    },
    {
      id: "BN-2026-0412",
      name: "Lê Văn Hùng",
      age: "56 tuổi",
      gender: "Nam",
      doctor: "TS. BS Phạm Hải Đăng",
      modality: "MRI 3.0T Brain T2-FLAIR",
      symptoms: "Co giật cục bộ nửa người trái, suy giảm trí nhớ gần, mệt mỏi.",
    },
    {
      id: "BN-2026-1077",
      name: "Hoàng Minh Tuấn",
      age: "39 tuổi",
      gender: "Nam",
      doctor: "BS. CKI Vũ Thu Hương",
      modality: "MRI Sọ Não T1 Coronal Dynamic",
      symptoms: "Rối loạn nội tiết, bán manh thái dương hai bên, đau đầu nhẹ.",
    },
    {
      id: "BN-2026-0205",
      name: "Nguyễn Thùy Dung",
      age: "32 tuổi",
      gender: "Nữ",
      doctor: "BS. CKII Nguyễn Hoàng Nam",
      modality: "MRI Não Khám Sức Khỏe Định Kỳ",
      symptoms: "Khám sàng lọc sức khỏe tổng quát định kỳ, không có triệu chứng thần kinh.",
    }
  ];

  let p;
  if (preferredClass === "meningioma") p = samplePatients[0];
  else if (preferredClass === "glioma") p = samplePatients[1];
  else if (preferredClass === "pituitary") p = samplePatients[2];
  else if (preferredClass === "notumor") p = samplePatients[3];
  else p = samplePatients[Math.floor(Math.random() * samplePatients.length)];

  document.getElementById("patientId").value = p.id;
  document.getElementById("patientName").value = p.name;
  document.getElementById("patientAge").value = p.age;
  document.getElementById("patientGender").value = p.gender;
  document.getElementById("patientDoctor").value = p.doctor;
  document.getElementById("patientModality").value = p.modality;
  document.getElementById("patientSymptoms").value = p.symptoms;
}

// ─── MEDICAL REPORT TEMPLATE SYNC ──────────────────────────────────────────────
function updateMedicalReportTemplate(data) {
  const pName = document.getElementById("patientName").value || "Bệnh nhân ẩn danh";
  const pId = document.getElementById("patientId").value || "BN-2026-0000";
  const pAge = document.getElementById("patientAge").value || "N/A";
  const pGender = document.getElementById("patientGender").value || "N/A";
  const pDoctor = document.getElementById("patientDoctor").value || "Bác sĩ phụ trách";
  const pModality = document.getElementById("patientModality").value || "MRI Sọ Não";
  const pSymptoms = document.getElementById("patientSymptoms").value || "Không ghi nhận";

  const now = new Date();
  const dateStr = `${String(now.getDate()).padStart(2, '0')}/${String(now.getMonth() + 1).padStart(2, '0')}/${now.getFullYear()} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
  const sigDateStr = `Hà Nội, ngày ${String(now.getDate()).padStart(2, '0')} tháng ${String(now.getMonth() + 1).padStart(2, '0')} năm ${now.getFullYear()}`;

  document.getElementById("repDocId").textContent = `SCAN-${now.getFullYear()}${String(now.getMonth()+1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${pId.replace(/[^0-9]/g, '').slice(-4) || '0001'}`;
  document.getElementById("repPatientName").textContent = pName;
  document.getElementById("repPatientId").textContent = pId;
  document.getElementById("repPatientAge").textContent = pAge;
  document.getElementById("repPatientGender").textContent = pGender;
  document.getElementById("repScanDate").textContent = dateStr;
  document.getElementById("repDoctor").textContent = pDoctor;
  document.getElementById("repModality").textContent = pModality;
  document.getElementById("repSymptoms").textContent = pSymptoms;

  // Images
  document.getElementById("repOriginalImg").src = currentOriginalImgUrl || previewImg.src;
  const firstGradcam = Object.values(currentGradcamData)[0] || currentOriginalImgUrl || previewImg.src;
  document.getElementById("repGradcamImg").src = firstGradcam;

  // Findings
  const p = data.prediction;
  document.getElementById("repClassVi").textContent = `${p.class_vi.toUpperCase()} (${p.class.toUpperCase()})`;
  document.getElementById("repConfBadge").textContent = `ĐỘ TIN CẬY: ${p.confidence}%`;
  document.getElementById("repRecommendation").textContent = data.recommendation;
  document.getElementById("repSigDoctor").textContent = pDoctor;
  document.getElementById("repSigDate").textContent = sigDateStr;

  // Probabilities Table
  let tbHtml = "";
  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1].score_pct - a[1].score_pct);
  for (const [cn, info] of sorted) {
    const isTop = cn === p.class;
    let riskTag = "Thấp";
    if (cn === "glioma") riskTag = isTop ? "<strong style='color:#dc2626'>Rất cao (Nguy hiểm)</strong>" : "Thấp";
    else if (cn === "meningioma" || cn === "pituitary") riskTag = isTop ? "<strong style='color:#d97706'>Trung bình</strong>" : "Thấp";
    else if (cn === "notumor") riskTag = isTop ? "<strong style='color:#16a34a'>Bình thường</strong>" : "Không áp dụng";

    tbHtml += `
    <tr ${isTop ? "style='background:#f0fdf4; font-weight:600;'" : ""}>
      <td>${info.label_vi} ${isTop ? " ★" : ""}</td>
      <td>${CLASS_MEDICAL_TERMS[cn] || cn}</td>
      <td style="font-family:'JetBrains Mono',monospace;">${info.score_pct}%</td>
      <td>${riskTag}</td>
    </tr>`;
  }
  document.getElementById("repProbTableBody").innerHTML = tbHtml;
}

// ─── EXPORT PDF & PRINT MEDICAL REPORT ─────────────────────────────────────────
function exportMedicalReportPDF() {
  if (!lastAnalysisData) {
    showError("Vui lòng phân tích ảnh MRI trước khi xuất báo cáo.");
    return;
  }

  updateMedicalReportTemplate(lastAnalysisData);

  const element = document.getElementById("medicalReportTemplate");
  const pId = document.getElementById("patientId").value || "BN";
  const safePId = pId.replace(/[^a-zA-Z0-9_-]/g, "_");

  const container = document.getElementById("medicalReportContainer");
  container.style.display = "block";

  const opt = {
    margin: [6, 8, 8, 8],
    filename: `BenhAn_${safePId}_${Date.now()}.pdf`,
    image: { type: "jpeg", quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true, logging: false },
    jsPDF: { unit: "mm", format: "a4", orientation: "portrait" }
  };

  html2pdf()
    .set(opt)
    .from(element)
    .save()
    .then(() => {
      container.style.display = "none";
    })
    .catch((err) => {
      container.style.display = "none";
      console.error("PDF Export Error:", err);
      showError("Lỗi khi xuất PDF. Hãy thử dùng chức năng 'In Phiếu' và chọn 'Save as PDF'.");
    });
}

function printMedicalReport() {
  if (!lastAnalysisData) {
    showError("Vui lòng phân tích ảnh MRI trước khi in báo cáo.");
    return;
  }
  updateMedicalReportTemplate(lastAnalysisData);
  window.print();
}

// Error Handling
function showError(msg) {
  errorBox.textContent = "⚠ " + msg;
  errorBox.classList.add("visible");
  setLoading(false);
}

function clearError() {
  errorBox.classList.remove("visible");
}


