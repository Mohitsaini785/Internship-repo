/**
 * VisionPulse AI - Interactive Frontend Controller
 * Handles drag-and-drop, webcam stream, sample gallery,
 * API requests, real-time predictions, and OpenCV diagnostic views.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements: Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // DOM Elements: Upload & Preview
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewArea = document.getElementById("previewArea");
    const imagePreview = document.getElementById("imagePreview");
    const clearPreviewBtn = document.getElementById("clearPreviewBtn");
    const previewFilename = document.getElementById("previewFilename");
    const quickClassifyBtn = document.getElementById("quickClassifyBtn");

    // DOM Elements: Camera
    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const startCamBtn = document.getElementById("startCamBtn");
    const captureCamBtn = document.getElementById("captureCamBtn");
    const stopCamBtn = document.getElementById("stopCamBtn");

    // DOM Elements: Sample Cards
    const sampleCards = document.querySelectorAll(".sample-card");

    // DOM Elements: Controls & Action
    const modelSelect = document.getElementById("modelSelect");
    const toggleClahe = document.getElementById("toggleClahe");
    const toggleSharpen = document.getElementById("toggleSharpen");
    const toggleDenoise = document.getElementById("toggleDenoise");
    const classifyBtn = document.getElementById("classifyBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnIcon = document.getElementById("btnIcon");
    const btnText = document.getElementById("btnText");

    // DOM Elements: Results
    const emptyState = document.getElementById("emptyState");
    const resultsContainer = document.getElementById("resultsContainer");
    const topLabel = document.getElementById("topLabel");
    const topClassId = document.getElementById("topClassId");
    const topConfidence = document.getElementById("topConfidence");
    const metaLatency = document.getElementById("metaLatency");
    const metaResolution = document.getElementById("metaResolution");
    const metaAspect = document.getElementById("metaAspect");
    const metaBrightness = document.getElementById("metaBrightness");
    const metaSize = document.getElementById("metaSize");
    const rankingsList = document.getElementById("rankingsList");
    const copyResultsBtn = document.getElementById("copyResultsBtn");

    // DOM Elements: Diagnostics
    const subTabs = document.querySelectorAll(".sub-tab");
    const diagImage = document.getElementById("diagImage");
    const diagLegendText = document.getElementById("diagLegendText");
    const roiSummary = document.getElementById("roiSummary");
    const roiTags = document.getElementById("roiTags");
    const toast = document.getElementById("toast");

    // State Variables
    let currentFile = null;
    let currentBase64 = null;
    let currentSampleId = null;
    let cameraStream = null;
    let currentDiagnostics = null;
    let lastClassificationReport = null;

    const DIAGNOSTIC_LEGENDS = {
        gradcam: "Grad-CAM: Warmer colors highlight regions that heavily influenced the neural network's classification decision.",
        edges: "OpenCV Canny: High-contrast wireframe identifying object boundaries, silhouettes, and sharp transitions.",
        contours: "OpenCV Contours: Salient Regions of Interest (ROI) segmented with HUD corner brackets.",
        histogram: "Color Distribution: Calculated intensity profiles across RGB channels.",
        processed: "Preprocessed Frame: Image after applying active enhancement filters."
    };

    // =========================================================================
    // Tab Switching
    // =========================================================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(`content${target.charAt(0).toUpperCase() + target.slice(1)}`).classList.add("active");

            // Stop webcam if leaving camera tab
            if (target !== "camera" && cameraStream) {
                stopWebcam();
            }
        });
    });

    // =========================================================================
    // Drag & Drop / File Upload
    // =========================================================================
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => {
        if (!currentFile && !currentBase64) fileInput.click();
    });

    ["dragenter", "dragover"].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            showToast("Please select a valid image file (PNG, JPG, WEBP, BMP)");
            return;
        }

        currentFile = file;
        currentBase64 = null;
        currentSampleId = null;
        sampleCards.forEach(c => c.classList.remove("selected"));

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.style.display = "none";
            previewArea.style.display = "block";
            previewFilename.textContent = `${file.name} (${formatBytes(file.size)})`;
            classifyBtn.disabled = false;
            // Automatically classify on upload
            runClassification();
        };
        reader.readAsDataURL(file);
    }

    clearPreviewBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetInput();
    });

    function resetInput() {
        currentFile = null;
        currentBase64 = null;
        currentSampleId = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropzonePrompt.style.display = "flex";
        previewArea.style.display = "none";
        classifyBtn.disabled = true;
        sampleCards.forEach(c => c.classList.remove("selected"));
    }

    // =========================================================================
    // Webcam Handler
    // =========================================================================
    startCamBtn.addEventListener("click", async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
                audio: false
            });
            webcamVideo.srcObject = cameraStream;
            startCamBtn.style.display = "none";
            captureCamBtn.style.display = "inline-flex";
            stopCamBtn.style.display = "inline-flex";
        } catch (err) {
            console.error("Camera access error:", err);
            showToast("Could not access camera: " + err.message);
        }
    });

    captureCamBtn.addEventListener("click", () => {
        if (!cameraStream) return;

        webcamCanvas.width = webcamVideo.videoWidth || 640;
        webcamCanvas.height = webcamVideo.videoHeight || 480;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        const dataUrl = webcamCanvas.toDataURL("image/jpeg", 0.92);
        currentBase64 = dataUrl;
        currentFile = null;
        currentSampleId = null;

        // Switch to upload preview tab to show captured snapshot
        imagePreview.src = dataUrl;
        dropzonePrompt.style.display = "none";
        previewArea.style.display = "block";
        previewFilename.textContent = `Webcam Snapshot (${webcamCanvas.width}x${webcamCanvas.height})`;
        classifyBtn.disabled = false;

        // Switch active tab back to upload
        tabBtns[0].click();
        stopWebcam();
        showToast("Snapshot captured! Analyzing image...");
        // Automatically trigger real-time classification!
        runClassification();
    });

    stopCamBtn.addEventListener("click", stopWebcam);

    function stopWebcam() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        webcamVideo.srcObject = null;
        startCamBtn.style.display = "inline-flex";
        captureCamBtn.style.display = "none";
        stopCamBtn.style.display = "none";
    }

    // =========================================================================
    // Sample Images Gallery
    // =========================================================================
    sampleCards.forEach(card => {
        card.addEventListener("click", () => {
            const sampleId = card.dataset.sample;
            sampleCards.forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");

            currentSampleId = sampleId;
            currentFile = null;
            currentBase64 = null;

            // Update preview image
            const sampleImg = card.querySelector("img");
            imagePreview.src = sampleImg.src;
            dropzonePrompt.style.display = "none";
            previewArea.style.display = "block";
            previewFilename.textContent = `${sampleId.replace('_', ' ').toUpperCase()} (Benchmark Sample)`;
            classifyBtn.disabled = false;

            // Automatically trigger classification on click for instant feedback!
            runClassification();
        });
    });

    // =========================================================================
    // Classification API Call
    // =========================================================================
    classifyBtn.addEventListener("click", runClassification);
    if (quickClassifyBtn) {
        quickClassifyBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            runClassification();
        });
    }

    async function runClassification() {
        if (!currentFile && !currentBase64 && !currentSampleId) {
            showToast("Please provide an image first.");
            return;
        }

        // UI Loading State
        setLoading(true);

        const selectedModel = modelSelect.value;
        const clahe = toggleClahe.checked;
        const sharpen = toggleSharpen.checked;
        const denoise = toggleDenoise.checked;

        try {
            let response;

            if (currentFile) {
                const formData = new FormData();
                formData.append("file", currentFile);
                formData.append("model", selectedModel);
                formData.append("clahe", clahe);
                formData.append("sharpen", sharpen);
                formData.append("denoise", denoise);

                response = await fetch("/api/classify", {
                    method: "POST",
                    body: formData
                });
            } else {
                const payload = {
                    model: selectedModel,
                    clahe: clahe,
                    sharpen: sharpen,
                    denoise: denoise
                };

                if (currentSampleId) {
                    payload.sample_id = currentSampleId;
                } else if (currentBase64) {
                    payload.image_base64 = currentBase64;
                }

                response = await fetch("/api/classify", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
            }

            const data = await response.json();

            if (!response.ok || data.status === "error") {
                throw new Error(data.error || "Inference failed");
            }

            renderResults(data);
            showToast(`Identified: ${data.prediction.top_prediction.label} (${data.prediction.top_prediction.probability}%)`);
        } catch (err) {
            console.error("Inference Error:", err);
            showToast("Classification error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        classifyBtn.disabled = isLoading;
        if (isLoading) {
            btnSpinner.style.display = "inline-block";
            btnIcon.style.display = "none";
            btnText.textContent = "Analyzing Neural Features...";
        } else {
            btnSpinner.style.display = "none";
            btnIcon.style.display = "inline-block";
            btnText.textContent = "Classify & Analyze Image";
        }
    }

    // =========================================================================
    // Render Results & Diagnostics
    // =========================================================================
    function renderResults(data) {
        emptyState.style.display = "none";
        resultsContainer.style.display = "flex";

        const top = data.prediction.top_prediction;
        const meta = data.metadata;

        // Spotlight Card
        topLabel.textContent = top.label;
        topClassId.textContent = `ID: ${top.class_id} · Model: ${data.prediction.model_name}`;
        topConfidence.textContent = `${top.probability}%`;

        // Metadata Strip
        metaLatency.textContent = `${data.prediction.inference_time_ms} ms`;
        metaResolution.textContent = `${meta.width} × ${meta.height}`;
        metaAspect.textContent = `${meta.aspect_ratio}:1`;
        metaBrightness.textContent = `${meta.brightness}`;
        metaSize.textContent = meta.file_size;

        // Top-5 Rankings
        rankingsList.innerHTML = "";
        data.prediction.predictions.forEach((pred) => {
            const item = document.createElement("div");
            item.className = "rank-item";
            item.innerHTML = `
                <div class="rank-row-top">
                    <div class="rank-title-group">
                        <span class="rank-badge">#${pred.rank}</span>
                        <span class="rank-name">${pred.label}</span>
                    </div>
                    <span class="rank-percent">${pred.probability}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: 0%;"></div>
                </div>
            `;
            rankingsList.appendChild(item);

            // Animate bar width smoothly
            setTimeout(() => {
                const bar = item.querySelector(".progress-bar");
                bar.style.width = `${Math.min(100, Math.max(2, pred.probability))}%`;
            }, 50 * pred.rank);
        });

        // Store diagnostic images
        currentDiagnostics = data.cv_diagnostics;
        lastClassificationReport = generateReportText(data);

        // Update active diagnostic view
        const activeSubTab = document.querySelector(".sub-tab.active") || subTabs[0];
        displayDiagnosticView(activeSubTab.dataset.diag);
    }

    // Diagnostic Sub-tabs
    subTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            subTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            displayDiagnosticView(tab.dataset.diag);
        });
    });

    function displayDiagnosticView(viewType) {
        if (!currentDiagnostics) return;

        let imgSrc = "";
        roiSummary.style.display = "none";

        switch (viewType) {
            case "gradcam":
                imgSrc = currentDiagnostics.gradcam_heatmap || currentDiagnostics.processed_preview;
                break;
            case "edges":
                imgSrc = currentDiagnostics.edges_canny;
                break;
            case "contours":
                imgSrc = currentDiagnostics.contours_hud;
                if (currentDiagnostics.detected_regions && currentDiagnostics.detected_regions.length > 0) {
                    roiSummary.style.display = "block";
                    roiTags.innerHTML = currentDiagnostics.detected_regions.map(r => 
                        `<span class="roi-tag">ROI #${r.id} (${r.width}×${r.height}px, ${r.area_percentage}%)</span>`
                    ).join("");
                }
                break;
            case "histogram":
                imgSrc = currentDiagnostics.color_histogram;
                break;
            case "processed":
                imgSrc = currentDiagnostics.processed_preview;
                break;
            default:
                imgSrc = currentDiagnostics.processed_preview;
        }

        diagImage.src = imgSrc;
        diagLegendText.textContent = DIAGNOSTIC_LEGENDS[viewType] || "";
    }

    // =========================================================================
    // Copy Report to Clipboard
    // =========================================================================
    copyResultsBtn.addEventListener("click", async () => {
        if (!lastClassificationReport) return;
        try {
            await navigator.clipboard.writeText(lastClassificationReport);
            showToast("Report copied to clipboard!");
        } catch (e) {
            showToast("Could not copy to clipboard");
        }
    });

    function generateReportText(data) {
        const top = data.prediction.top_prediction;
        const meta = data.metadata;
        const lines = [
            `# VisionPulse AI Recognition Summary`,
            `Model: ${data.prediction.model_name}`,
            `Inference Latency: ${data.prediction.inference_time_ms} ms`,
            `Resolution: ${meta.width}x${meta.height} (${meta.file_size})`,
            ``,
            `Top Prediction: ${top.label} (${top.probability}%)`,
            `Class ID: ${top.class_id}`,
            ``,
            `Top-5 Distribution:`,
            ...data.prediction.predictions.map(p => `  ${p.rank}. ${p.label}: ${p.probability}%`),
            ``,
            `Diagnostics: OpenCV Canny Edges, Contours HUD, Grad-CAM Attention Heatmap`
        ];
        return lines.join("\n");
    }

    // =========================================================================
    // Helpers
    // =========================================================================
    function formatBytes(bytes) {
        if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
        if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
        return bytes + " B";
    }

    // =========================================================================
    // 1,000 Classes Directory Explorer Modal
    // =========================================================================
    const exploreClassesBtn = document.getElementById("exploreClassesBtn");
    const classesModal = document.getElementById("classesModal");
    const closeClassesModalBtn = document.getElementById("closeClassesModalBtn");
    const classesSearchInput = document.getElementById("classesSearchInput");
    const searchMatchCount = document.getElementById("searchMatchCount");
    const classesPillsList = document.getElementById("classesPillsList");
    let allImageNetClasses = [];

    if (exploreClassesBtn && classesModal) {
        exploreClassesBtn.addEventListener("click", async () => {
            classesModal.style.display = "flex";
            if (allImageNetClasses.length === 0) {
                classesPillsList.innerHTML = `<span style="color: var(--text-dim); padding: 12px;">Loading 1,000 categories...</span>`;
                try {
                    const res = await fetch("/api/classes");
                    const data = await res.json();
                    if (data.status === "success") {
                        allImageNetClasses = data.classes;
                        renderClassesList(allImageNetClasses);
                    }
                } catch (err) {
                    classesPillsList.innerHTML = `<span style="color: var(--accent-pink);">Failed to load categories.</span>`;
                }
            } else {
                renderClassesList(allImageNetClasses);
            }
            classesSearchInput.focus();
        });

        closeClassesModalBtn.addEventListener("click", () => {
            classesModal.style.display = "none";
        });

        classesModal.addEventListener("click", (e) => {
            if (e.target === classesModal) {
                classesModal.style.display = "none";
            }
        });

        classesSearchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderClassesList(allImageNetClasses);
                return;
            }
            const filtered = allImageNetClasses.filter(c => c.label.toLowerCase().includes(query));
            renderClassesList(filtered);
        });
    }

    function renderClassesList(list) {
        searchMatchCount.textContent = `${list.length} of 1,000 matching`;
        if (list.length === 0) {
            classesPillsList.innerHTML = `<span style="color: var(--text-dim); padding: 16px;">No exact class found. Try related terms (e.g., 'phone' instead of 'iphone').</span>`;
            return;
        }
        classesPillsList.innerHTML = list.map(c => 
            `<span class="class-pill" title="ID: ${c.id}">${c.label}</span>`
        ).join("");
    }
});
