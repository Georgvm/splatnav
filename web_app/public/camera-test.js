const browserButton = document.querySelector("#browserButton");
const releaseButton = document.querySelector("#releaseButton");
const backendLiveButton = document.querySelector("#backendLiveButton");
const backendButton = document.querySelector("#backendButton");
const streamStatusButton = document.querySelector("#streamStatusButton");
const refreshButton = document.querySelector("#refreshButton");
const cameraSelect = document.querySelector("#cameraSelect");
const browserVideo = document.querySelector("#browserVideo");
const analysisCanvas = document.querySelector("#analysisCanvas");
const backendImage = document.querySelector("#backendImage");
const browserStatus = document.querySelector("#browserStatus");
const sdkStatus = document.querySelector("#sdkStatus");
const backendStatus = document.querySelector("#backendStatus");
const details = document.querySelector("#details");

let backendInFlight = false;
let browserStream = null;
let analysisTimer = null;
let backendLiveRunning = false;
let backendLiveTimer = null;

function shortJson(value) {
  return JSON.stringify(value, null, 2).replace(/\n/g, " ");
}

async function refreshStatus() {
  sdkStatus.textContent = "Checking RealSense SDK...";
  try {
    const response = await fetch("/api/realsense/status");
    const data = await response.json();
    details.textContent = JSON.stringify(data, null, 2);
    const usb = data.usb?.seen ? `USB sees ${data.usb.name} serial ${data.usb.serial || "unknown"}` : "USB does not see a D435i";
    const py = data.python_binding?.available ? `${data.python_binding.devices.length} SDK device(s)` : "Python SDK unavailable";
    const cliOk = data.librealsense_cli?.returncode === 0;
    const cli = cliOk ? "librealsense CLI can enumerate" : "librealsense CLI cannot open the device";
    sdkStatus.textContent = `${usb}. ${py}. ${cli}.`;
  } catch (error) {
    details.textContent = String(error);
    sdkStatus.textContent = `SDK status failed: ${error.message}`;
  }
}

async function refreshCameraDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    cameraSelect.innerHTML = '<option value="">Browser device list unavailable</option>';
    browserStatus.textContent = "Browser mediaDevices API unavailable.";
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  cameraSelect.innerHTML = "";
  if (cameras.length === 0) {
    cameraSelect.append(new Option("No browser cameras visible", ""));
    browserStatus.textContent = "No browser cameras visible.";
    return;
  }
  for (const [index, camera] of cameras.entries()) {
    const label = camera.label || `Camera ${index + 1}`;
    cameraSelect.append(new Option(label, camera.deviceId));
  }
  const realsense = [...cameraSelect.options].find((option) => /realsense|435i/i.test(option.textContent || ""));
  if (realsense) cameraSelect.value = realsense.value;
  const labels = cameras.map((camera, index) => camera.label || `Camera ${index + 1}`).join(", ");
  browserStatus.textContent = `${cameras.length} browser camera(s): ${labels}`;
}

function stopAnalysis() {
  if (analysisTimer !== null) {
    window.clearInterval(analysisTimer);
    analysisTimer = null;
  }
}

function stopBrowserStream(reason) {
  if (!browserStream) return;
  for (const track of browserStream.getTracks()) track.stop();
  browserStream = null;
  browserVideo.srcObject = null;
  stopAnalysis();
  browserButton.textContent = "Browser camera";
  if (reason) browserStatus.textContent = reason;
}

function stopBackendLive(reason = "Backend live stream stopped.") {
  if (!backendLiveRunning) return;
  backendLiveRunning = false;
  if (backendLiveTimer !== null) {
    window.clearTimeout(backendLiveTimer);
    backendLiveTimer = null;
  }
  const oldUrl = backendImage.src;
  backendImage.removeAttribute("src");
  if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
  backendLiveButton.textContent = "Backend live";
  backendButton.disabled = false;
  backendStatus.textContent = reason;
}

async function releaseBackendCameraOwners() {
  stopBrowserStream("Stopped browser camera before releasing RealSense ownership.");
  stopBackendLive("Stopped backend live stream before releasing RealSense ownership.");
  backendStatus.textContent = "Releasing macOS camera helpers and old RealSense helpers...";
  try {
    const response = await fetch("/api/realsense/release", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Release request failed");
    details.textContent = `${details.textContent}\n\nRelease camera:\n${JSON.stringify(data, null, 2)}`;
    backendStatus.textContent = "Released camera helpers. Try Backend live now.";
  } catch (error) {
    backendStatus.textContent = `Release failed: ${error.message}`;
  }
}

async function refreshStreamStatus(prefix = "Stream status") {
  try {
    const response = await fetch(`/api/realsense/stream-status?t=${Date.now()}`);
    const data = await response.json();
    details.textContent = `${details.textContent}\n\n${prefix}:\n${JSON.stringify(data, null, 2)}`;
    if (data.log) {
      backendStatus.textContent = `${prefix}: ${data.log.trim().split("\n").slice(-1)[0]}`;
    }
  } catch (error) {
    details.textContent = `${details.textContent}\n\n${prefix} failed: ${error.message}`;
  }
}

function startColorAnalysis(track) {
  stopAnalysis();
  const context = analysisCanvas.getContext("2d", { willReadFrequently: true });
  analysisTimer = window.setInterval(() => {
    if (!browserVideo.videoWidth || !browserVideo.videoHeight) return;
    const width = 96;
    const height = Math.max(1, Math.round(width * browserVideo.videoHeight / browserVideo.videoWidth));
    analysisCanvas.width = width;
    analysisCanvas.height = height;
    context.drawImage(browserVideo, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height).data;
    let colorDelta = 0;
    let lumaMean = 0;
    let count = 0;
    for (let index = 0; index < pixels.length; index += 16) {
      const r = pixels[index + 0];
      const g = pixels[index + 1];
      const b = pixels[index + 2];
      colorDelta += Math.abs(r - g) + Math.abs(g - b) + Math.abs(r - b);
      lumaMean += (r + g + b) / 3;
      count += 1;
    }
    const avgColorDelta = colorDelta / Math.max(count, 1);
    const avgLuma = lumaMean / Math.max(count, 1);
    const looksColor = avgColorDelta > 8;
    const settings = track.getSettings?.() ?? {};
    const label = settings.label || track.label || "camera";
    browserStatus.textContent =
      `Running: ${label} ${settings.width || "?"}x${settings.height || "?"}. ` +
      `Pixel test: ${looksColor ? "color" : "mostly grayscale"} ` +
      `(color delta ${avgColorDelta.toFixed(1)}, brightness ${avgLuma.toFixed(1)}).`;
  }, 500);
}

async function startBrowserCamera() {
  if (browserStream) {
    stopBrowserStream("Browser camera stopped.");
    return;
  }
  stopBackendLive("Stopped backend live stream so the browser camera can own the device.");
  browserStatus.textContent = "Requesting browser camera access...";
  try {
    stopAnalysis();
    const selectedDevice = cameraSelect.value;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
        width: { ideal: 848 },
        height: { ideal: 480 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    });
    browserStream = stream;
    browserVideo.srcObject = stream;
    browserButton.textContent = "Stop browser";
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    const capabilities = track?.getCapabilities?.() ?? {};
    await refreshCameraDevices();
    if (settings.deviceId) cameraSelect.value = settings.deviceId;
    startColorAnalysis(track);
    details.textContent =
      `${details.textContent}\n\nActive browser camera track:\n` +
      `label: ${track.label || settings.label || "unknown"}\n` +
      `settings: ${shortJson(settings)}\n` +
      `capabilities: ${shortJson(capabilities)}\n`;
  } catch (error) {
    browserStatus.textContent = `Failed: ${error.message}`;
    details.textContent = `${details.textContent}\n\nBrowser camera error:\n${error.stack || error.message}`;
  }
}

async function loadBackendFrame({ fast = false } = {}) {
  if (backendInFlight) return;
  backendInFlight = true;
  backendButton.disabled = true;
  backendButton.textContent = "Capturing...";
  backendStatus.textContent = backendLiveRunning
    ? "Capturing next RealSense RGB frame..."
    : "Requesting one RealSense RGB frame...";
  try {
    const frameStartedAt = performance.now();
    stopBrowserStream("Stopped browser camera so the RealSense SDK can own the device.");
    if (!backendLiveRunning) await fetch("/api/realsense/release", { method: "POST" }).catch(() => null);
    const params = new URLSearchParams({ t: String(Date.now()) });
    if (fast) {
      params.set("fast", "true");
      params.set("release", "false");
    }
    const response = await fetch(`/api/realsense/frame.jpg?${params.toString()}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }
    const blob = await response.blob();
    const oldUrl = backendImage.src;
    backendImage.src = URL.createObjectURL(blob);
    if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
    const elapsedSeconds = ((performance.now() - frameStartedAt) / 1000).toFixed(2);
    backendStatus.textContent = backendLiveRunning
      ? `Live polling: loaded latest RealSense RGB frame in ${elapsedSeconds}s.`
      : "Loaded backend RealSense frame.";
  } catch (error) {
    backendStatus.textContent = `Failed: ${error.message}`;
  } finally {
    backendInFlight = false;
    backendButton.disabled = false;
    backendButton.textContent = "Backend frame";
  }
}

async function pollBackendLiveFrame() {
  if (!backendLiveRunning || backendInFlight) return;
  const started = performance.now();
  await loadBackendFrame({ fast: true });
  if (!backendLiveRunning) return;
  const elapsed = performance.now() - started;
  const targetGapMs = backendStatus.textContent.startsWith("Failed") ? 1200 : 180;
  backendLiveTimer = window.setTimeout(
    pollBackendLiveFrame,
    Math.max(120, targetGapMs - elapsed)
  );
}

function toggleBackendLive() {
  if (backendLiveRunning) {
    stopBackendLive();
    return;
  }
  stopBrowserStream("Stopped browser camera so the RealSense SDK can own the device.");
  backendLiveRunning = true;
  backendLiveButton.textContent = "Stop live";
  backendButton.disabled = false;
  backendStatus.textContent = "Starting pyrealsense2 MJPEG stream from the working RealSense venv...";
  backendImage.onerror = () => {
    if (!backendLiveRunning) return;
    stopBackendLive("Backend pyrealsense2 stream failed. Try Release camera, then Backend live again.");
    refreshStreamStatus("Failed stream status");
  };
  backendImage.onload = () => {
    if (!backendLiveRunning) return;
    backendStatus.textContent = "Streaming backend RealSense RGB video through pyrealsense2.";
  };
  backendImage.src = `/api/realsense/stream?t=${Date.now()}`;
}

function startBackendFrame() {
  loadBackendFrame();
}

browserButton.addEventListener("click", startBrowserCamera);
releaseButton.addEventListener("click", releaseBackendCameraOwners);
backendLiveButton.addEventListener("click", toggleBackendLive);
backendButton.addEventListener("click", startBackendFrame);
streamStatusButton.addEventListener("click", () => refreshStreamStatus());
refreshButton.addEventListener("click", refreshStatus);
cameraSelect.addEventListener("change", () => {
  if (browserStream) startBrowserCamera();
});

refreshStatus();
refreshCameraDevices().catch((error) => {
  cameraSelect.innerHTML = "";
  cameraSelect.append(new Option(`Device list failed: ${error.message}`, ""));
  browserStatus.textContent = `Device list failed: ${error.message}`;
});
