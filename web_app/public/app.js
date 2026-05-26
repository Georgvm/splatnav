import * as THREE from "https://unpkg.com/three@0.165.0/build/three.module.js";

const fpCanvas = document.querySelector("#firstPersonScene");
const overviewCanvas = document.querySelector("#overviewScene");
const loadStatus = document.querySelector("#loadStatus");
const motionStatus = document.querySelector("#motionStatus");
const runStatus = document.querySelector("#runStatus");
const logToggle = document.querySelector("#logToggle");
const logPanel = document.querySelector("#logPanel");
const logOutput = document.querySelector("#logOutput");
const runButton = document.querySelector("#runButton");
const sourceToggle = document.querySelector("#sourceToggle");
const cameraButton = document.querySelector("#cameraButton");
const realFrameButton = document.querySelector("#realFrameButton");
const realsenseLiveButton = document.querySelector("#realsenseLiveButton");
const realsenseButton = document.querySelector("#realsenseButton");
const autoButton = document.querySelector("#autoButton");
const cameraPanel = document.querySelector("#cameraPanel");
const cameraVideo = document.querySelector("#cameraVideo");
const realsenseStreamImage = document.querySelector("#realsenseStreamImage");
const cameraCapture = document.querySelector("#cameraCapture");
const cameraStatus = document.querySelector("#cameraStatus");
const successMetric = document.querySelector("#successMetric");
const rotationMetric = document.querySelector("#rotationMetric");
const translationMetric = document.querySelector("#translationMetric");
const runtimeMetric = document.querySelector("#runtimeMetric");
const snapshotImage = document.querySelector("#snapshotImage");
const renderImage = document.querySelector("#renderImage");
const matchesImage = document.querySelector("#matchesImage");
const sceneName = new URLSearchParams(window.location.search).get("scene") || "old_union";

const fpRenderer = new THREE.WebGLRenderer({ canvas: fpCanvas, antialias: true });
const mapRenderer = new THREE.WebGLRenderer({ canvas: overviewCanvas, antialias: true });
for (const renderer of [fpRenderer, mapRenderer]) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0c0f10, 1);
}

const fpScene = new THREE.Scene();
const mapScene = new THREE.Scene();
fpScene.add(new THREE.AmbientLight(0xffffff, 0.9));
mapScene.add(new THREE.AmbientLight(0xffffff, 0.9));
mapScene.add(new THREE.AxesHelper(0.85));

const fpCamera = new THREE.PerspectiveCamera(64, 1, 0.01, 1000);
const mapCamera = new THREE.PerspectiveCamera(48, 1, 0.01, 1000);

const keys = new Set();
const euler = new THREE.Euler(0, 0, 0, "YXZ");
const velocity = new THREE.Vector3();
const moveVector = new THREE.Vector3();
const dragMoveVector = new THREE.Vector3();
const cameraRight = new THREE.Vector3();
const cameraUp = new THREE.Vector3();
const cameraForward = new THREE.Vector3();
const orbitPivot = new THREE.Vector3();
const orbitOffset = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
const clock = new THREE.Clock();
const SCENE_UP = new THREE.Vector3(0, 0, 1);

let bounds = null;
let radius = 1;
let cameraFrames = [];
let gtRig = null;
let predRig = null;
let initialRig = null;
let errorLine = null;
let fpPointCloud = null;
let isDraggingFirstPerson = false;
let isDraggingOverview = false;
let lastPoseSent = null;
let statusTimer = null;
let cameraStream = null;
let realsenseLiveRunning = false;
let realsenseLiveTimer = null;
let realsenseLiveFrameInFlight = false;
let autoTimer = null;
let realFrameInFlight = false;
let realsenseInFlight = false;
let localizationEnabled = sceneName === "old_union" || sceneName === "stanford";
let captureSource = "synthetic";
let sourceType = "unknown";

function resizeRenderer(renderer, camera, canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
}

function makePointCloud(geometry, pointSize) {
  return new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      sizeAttenuation: true,
    })
  );
}

function makePoseRig(color, scale = 1) {
  const group = new THREE.Group();
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.055 * scale, 18, 12),
    new THREE.MeshBasicMaterial({ color })
  );
  group.add(marker);

  const frustumPoints = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-0.12, -0.08, -0.24),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.12, -0.08, -0.24),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.12, 0.08, -0.24),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-0.12, 0.08, -0.24),
    new THREE.Vector3(-0.12, -0.08, -0.24),
    new THREE.Vector3(0.12, -0.08, -0.24),
    new THREE.Vector3(0.12, -0.08, -0.24),
    new THREE.Vector3(0.12, 0.08, -0.24),
    new THREE.Vector3(0.12, 0.08, -0.24),
    new THREE.Vector3(-0.12, 0.08, -0.24),
    new THREE.Vector3(-0.12, 0.08, -0.24),
    new THREE.Vector3(-0.12, -0.08, -0.24),
  ];
  const frustum = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(frustumPoints),
    new THREE.LineBasicMaterial({ color })
  );
  frustum.scale.setScalar(scale);
  group.add(frustum);

  const forward = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(0, 0, 0),
    0.34 * scale,
    color,
    0.08 * scale,
    0.045 * scale
  );
  group.add(forward);
  return group;
}

function setRigFromPose(rig, pose) {
  if (!rig || !pose) return;
  const matrix = new THREE.Matrix4().set(
    pose[0][0], pose[0][1], pose[0][2], pose[0][3],
    pose[1][0], pose[1][1], pose[1][2], pose[1][3],
    pose[2][0], pose[2][1], pose[2][2], pose[2][3],
    pose[3][0], pose[3][1], pose[3][2], pose[3][3]
  );
  rig.matrix.copy(matrix);
  rig.matrix.decompose(rig.position, rig.quaternion, rig.scale);
  rig.visible = true;
}

function matrixWorldToRows(camera) {
  camera.updateMatrixWorld(true);
  const e = camera.matrixWorld.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

function rowsToMatrix(rows) {
  return new THREE.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3]
  );
}

function aimFirstPersonAt(target) {
  fpCamera.up.copy(SCENE_UP);
  fpCamera.lookAt(target);
  updateGroundTruthRig();
}

function aimOverviewAt(target) {
  mapCamera.up.copy(SCENE_UP);
  mapCamera.lookAt(target);
}

function cameraBasis(camera) {
  camera.updateMatrixWorld(true);
  cameraRight.setFromMatrixColumn(camera.matrixWorld, 0).normalize();
  cameraUp.setFromMatrixColumn(camera.matrixWorld, 1).normalize();
  camera.getWorldDirection(cameraForward).normalize();
}

function rotateFirstPersonByDrag(deltaX, deltaY) {
  rotateCameraByDrag(fpCamera, deltaX, deltaY, true);
}

function rotateOverviewByDrag(deltaX, deltaY) {
  rotateCameraByDrag(mapCamera, deltaX, deltaY, false);
}

function orbitFirstPersonByDrag(deltaX, deltaY) {
  orbitCameraByDrag(fpCamera, deltaX, deltaY, true);
}

function orbitOverviewByDrag(deltaX, deltaY) {
  orbitCameraByDrag(mapCamera, deltaX, deltaY, false);
}

function rotateCameraByDrag(camera, deltaX, deltaY, updateGroundTruth) {
  const sensitivity = 0.002;
  camera.updateMatrixWorld(true);
  cameraUp.setFromMatrixColumn(camera.matrixWorld, 1).normalize();
  tempQuaternion.setFromAxisAngle(cameraUp, -deltaX * sensitivity);
  camera.quaternion.premultiply(tempQuaternion);

  camera.updateMatrixWorld(true);
  cameraRight.setFromMatrixColumn(camera.matrixWorld, 0).normalize();
  tempQuaternion.setFromAxisAngle(cameraRight, -deltaY * sensitivity);
  const proposed = camera.quaternion.clone().premultiply(tempQuaternion).normalize();
  camera.quaternion.copy(proposed);
  if (updateGroundTruth) {
    updateGroundTruthRig();
  }
}

function orbitCameraByDrag(camera, deltaX, deltaY, updateGroundTruth) {
  const sensitivity = 0.002;
  camera.getWorldDirection(cameraForward).normalize();
  const distance = Math.max(radius * 0.32, camera.near * 8);
  orbitPivot.copy(camera.position).addScaledVector(cameraForward, distance);
  orbitOffset.copy(camera.position).sub(orbitPivot);

  tempQuaternion.setFromAxisAngle(SCENE_UP, -deltaX * sensitivity);
  orbitOffset.applyQuaternion(tempQuaternion);

  cameraRight.copy(orbitOffset).cross(SCENE_UP).normalize();
  if (cameraRight.lengthSq() < 1e-8) {
    cameraRight.setFromMatrixColumn(camera.matrixWorld, 0).normalize();
  }
  tempQuaternion.setFromAxisAngle(cameraRight, deltaY * sensitivity);
  const proposedOffset = orbitOffset.clone().applyQuaternion(tempQuaternion);
  const proposedForward = proposedOffset.clone().negate().normalize();
  if (Math.abs(proposedForward.dot(SCENE_UP)) < 0.985) {
    orbitOffset.copy(proposedOffset);
  }

  camera.position.copy(orbitPivot).add(orbitOffset);
  camera.up.copy(SCENE_UP);
  camera.lookAt(orbitPivot);
  if (updateGroundTruth) {
    updateGroundTruthRig();
    motionStatus.textContent = `Orbiting ground truth around x ${orbitPivot.x.toFixed(2)}  y ${orbitPivot.y.toFixed(2)}  z ${orbitPivot.z.toFixed(2)}`;
  } else {
    motionStatus.textContent = `Orbiting overview around x ${orbitPivot.x.toFixed(2)}  y ${orbitPivot.y.toFixed(2)}  z ${orbitPivot.z.toFixed(2)}`;
  }
}

function translateFirstPersonByDrag(deltaX, deltaY, scale = 1, basisCamera = fpCamera) {
  cameraBasis(basisCamera);
  const amount = radius * 0.0016 * scale;
  dragMoveVector
    .copy(cameraRight)
    .multiplyScalar(-deltaX * amount)
    .addScaledVector(cameraUp, deltaY * amount);
  fpCamera.position.add(dragMoveVector);
  updateGroundTruthRig();
  motionStatus.textContent = `Moved ground truth: x ${fpCamera.position.x.toFixed(2)}  y ${fpCamera.position.y.toFixed(2)}  z ${fpCamera.position.z.toFixed(2)}`;
}

function translateCameraByDrag(camera, deltaX, deltaY, scale = 1) {
  cameraBasis(camera);
  const amount = radius * 0.0016 * scale;
  dragMoveVector
    .copy(cameraRight)
    .multiplyScalar(-deltaX * amount)
    .addScaledVector(cameraUp, deltaY * amount);
  camera.position.add(dragMoveVector);
}

function dollyCamera(camera, deltaY, scale = 1) {
  camera.getWorldDirection(cameraForward).normalize();
  camera.position.addScaledVector(cameraForward, deltaY * radius * -0.0015 * scale);
}

function posePosition(pose) {
  return new THREE.Vector3(pose[0][3], pose[1][3], pose[2][3]);
}

function updateErrorLine(gtPose, predictedPose) {
  if (!errorLine || !predictedPose) return;
  errorLine.geometry.setFromPoints([posePosition(gtPose), posePosition(predictedPose)]);
  errorLine.visible = true;
}

function setFirstPersonPoseFromRows(rows) {
  const matrix = rowsToMatrix(rows);
  const position = new THREE.Vector3();
  position.setFromMatrixPosition(matrix);
  fpCamera.position.copy(position);
  aimFirstPersonAt(bounds.center);
}

function initializeFirstPerson(data) {
  const firstPose = data.camera_positions.find((entry) => entry.pose)?.pose;
  if (firstPose) {
    setFirstPersonPoseFromRows(firstPose);
  } else {
    const fallback = bounds.center.clone().add(
      new THREE.Vector3(radius * 0.35, radius * -0.55, radius * 0.18)
    );
    const start = data.camera_positions[0]?.position ?? fallback.toArray();
    fpCamera.position.set(...start);
    aimFirstPersonAt(bounds.center);
  }
  fpCamera.up.copy(SCENE_UP);
  fpCamera.near = Math.max(radius / 3000, 0.001);
  fpCamera.far = radius * 20;
  fpCamera.updateProjectionMatrix();
  updateGroundTruthRig();
}

function updateGroundTruthRig() {
  const pose = matrixWorldToRows(fpCamera);
  setRigFromPose(gtRig, pose);
  if (lastPoseSent) updateErrorLine(pose, lastPoseSent.estimated_pose);
}

function buildScene(data) {
  localizationEnabled = sceneName === "stanford" || (data.localization_enabled ?? sceneName === "old_union");
  sourceType = data.source_type ?? "unknown";
  const positions = new Float32Array(data.points.length * 3);
  const colors = new Float32Array(data.colors.length * 3);
  for (let i = 0; i < data.points.length; i += 1) {
    const p = data.points[i];
    const c = data.colors[i];
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
    colors[i * 3 + 0] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  bounds = geometry.boundingSphere;
  radius = Math.max(bounds.radius, 1);

  fpPointCloud = makePointCloud(geometry, 0.034);
  fpScene.add(fpPointCloud);
  mapScene.add(makePointCloud(geometry, 0.024));

  cameraFrames = data.camera_positions;
  const pathPoints = cameraFrames.map((entry) => new THREE.Vector3(...entry.position));
  if (pathPoints.length > 1) {
    mapScene.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pathPoints),
        new THREE.LineBasicMaterial({ color: 0x5fa8ff, transparent: true, opacity: 0.55 })
      )
    );
  }

  gtRig = makePoseRig(0x8edbd1, 1.7);
  predRig = makePoseRig(0x62e878, 1.85);
  initialRig = makePoseRig(0xffa23a, 1.35);
  predRig.visible = false;
  initialRig.visible = false;
  mapScene.add(gtRig, predRig, initialRig);
  errorLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineBasicMaterial({ color: 0x62e878 })
  );
  errorLine.visible = false;
  mapScene.add(errorLine);

  initializeFirstPerson(data);
  updateGroundTruthRig();

  mapCamera.position.copy(bounds.center).add(new THREE.Vector3(radius * 0.72, radius * -0.92, radius * 0.34));
  mapCamera.up.copy(SCENE_UP);
  mapCamera.near = Math.max(radius / 3000, 0.001);
  mapCamera.far = radius * 18;
  mapCamera.updateProjectionMatrix();
  aimOverviewAt(bounds.center);

  loadStatus.textContent = `${data.points.length.toLocaleString()} splat points loaded`;
  motionStatus.textContent = "Drag to look. Hold Shift+drag to move. Hold E+drag to orbit around a point ahead.";
  runButton.disabled = !localizationEnabled;
  updateSourceControls();
  runStatus.textContent = localizationEnabled
    ? "Ready to localize current pose"
    : sourceType === "gaussian_splat_ply"
      ? "Gaussian-splat PLY preview loaded; localization needs matching camera calibration/poses wired into the Splat-Loc backend."
      : "Point-cloud-only scene loaded; localization needs a trained Gaussian splat model and camera poses.";
  addLog(`Scene loaded: ${data.points.length.toLocaleString()} points, ${cameraFrames.length} camera markers`);
}

function imageSrc(base64) {
  return base64 ? `data:image/png;base64,${base64}` : "";
}

function addLog(message) {
  const time = new Date().toLocaleTimeString();
  logOutput.textContent += `[${time}] ${message}\n`;
  logOutput.scrollTop = logOutput.scrollHeight;
}

function clearRunLogs() {
  logOutput.textContent = "";
}

function updateSourceControls() {
  sourceToggle.textContent = captureSource === "realsense" ? "Source: RealSense" : "Source: 3D scene";
  runButton.textContent = "Capture";
  runButton.disabled = !localizationEnabled;
}

function setRunning(running) {
  runButton.disabled = running;
  sourceToggle.disabled = running;
  runButton.textContent = running ? "Capturing..." : "Capture";
}

function setRunStatus(message, alsoLog = false) {
  runStatus.textContent = message;
  if (alsoLog) addLog(message);
}

function setCameraStatus(message, alsoLog = false) {
  cameraStatus.textContent = message;
  if (alsoLog) addLog(`Camera: ${message}`);
}

function setRealFrameRunning(running) {
  realFrameInFlight = running;
  if (realFrameButton) realFrameButton.disabled = running || !cameraStream || sceneName !== "stanford";
  if (autoButton) autoButton.disabled = running || !cameraStream || sceneName !== "stanford";
  if (realFrameButton) realFrameButton.textContent = running ? "Sending..." : "Real frame";
}

function setRealSenseRunning(running) {
  realsenseInFlight = running;
  runButton.disabled = running;
  sourceToggle.disabled = running;
  runButton.textContent = running ? "Capturing..." : "Capture";
  if (realsenseButton) realsenseButton.disabled = running || sceneName !== "stanford";
  if (realsenseButton) realsenseButton.textContent = running ? "Capturing..." : "D435i";
  if (realsenseLiveButton) realsenseLiveButton.disabled = running || sceneName !== "stanford";
}

function applyLocalizationResult(data, label) {
  lastPoseSent = data;
  setRigFromPose(gtRig, data.selected_pose);
  setRigFromPose(initialRig, data.initial_pose);
  initialRig.visible = true;
  if (data.estimated_pose) {
    setRigFromPose(predRig, data.estimated_pose);
    predRig.visible = true;
    updateErrorLine(data.selected_pose, data.estimated_pose);
  } else {
    predRig.visible = false;
    errorLine.visible = false;
  }

  snapshotImage.src = imageSrc(data.snapshot_png);
  renderImage.src = imageSrc(data.render_png);
  matchesImage.src = imageSrc(data.matches_png);
  successMetric.textContent = data.success ? "yes" : "no";
  rotationMetric.textContent = data.rotation_error_deg === null ? "-" : `${data.rotation_error_deg.toFixed(2)} deg`;
  translationMetric.textContent = data.translation_error_m === null ? "-" : `${data.translation_error_m.toFixed(3)} m`;
  runtimeMetric.textContent = `${data.seconds.toFixed(1)} s`;
  motionStatus.textContent = data.success ? "Prediction updated on overview" : `Failed: ${data.error}`;
  setRunStatus(
    data.success
      ? `${label}: matched features and solved PnP in ${data.seconds.toFixed(1)}s`
      : `${label}: localization failed: ${data.error}`,
    true
  );
  for (const line of data.logs ?? []) addLog(`Modal: ${line}`);
}

function startStatusTimer() {
  const stages = [
    [0, "Sending current 4x4 camera pose to local FastAPI server..."],
    [900, "Dispatching Modal H100 function..."],
    [3500, "Waiting for H100 container; cold starts may take a few minutes..."],
    [12000, "Loading GSplat checkpoint or warming cached container..."],
    [30000, "Compiling/loading CUDA extension if this is a new container..."],
    [60000, "Rendering query snapshot and running SIFT + PnP..."],
  ];
  const started = performance.now();
  let idx = -1;
  statusTimer = window.setInterval(() => {
    const elapsed = performance.now() - started;
    let next = 0;
    for (let i = 0; i < stages.length; i += 1) {
      if (elapsed >= stages[i][0]) next = i;
    }
    if (next !== idx) {
      idx = next;
      setRunStatus(stages[idx][1], true);
    }
  }, 250);
}

function stopStatusTimer() {
  if (statusTimer !== null) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
}

function stopBrowserCamera(reason = "") {
  if (!cameraStream) return;
  for (const track of cameraStream.getTracks()) track.stop();
  cameraStream = null;
  cameraVideo.srcObject = null;
  if (cameraButton) cameraButton.textContent = "Camera";
  if (realFrameButton) realFrameButton.disabled = true;
  if (autoButton) autoButton.disabled = true;
  if (reason) setCameraStatus(reason, true);
}

function stopRealSenseLive(reason = "D435i live stream stopped.") {
  if (!realsenseLiveRunning) return;
  realsenseLiveRunning = false;
  if (realsenseLiveTimer !== null) {
    window.clearTimeout(realsenseLiveTimer);
    realsenseLiveTimer = null;
  }
  const oldUrl = realsenseStreamImage.src;
  realsenseStreamImage.removeAttribute("src");
  if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
  realsenseStreamImage.hidden = true;
  cameraVideo.hidden = false;
  if (realsenseLiveButton) realsenseLiveButton.textContent = "D435i live";
  if (reason) setCameraStatus(reason, true);
}

async function startCamera() {
  if (cameraStream) return;
  stopRealSenseLive("Stopped D435i live stream so the browser camera can own the preview.");
  cameraPanel.hidden = false;
  cameraVideo.hidden = false;
  realsenseStreamImage.hidden = true;
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus("Browser camera capture is unavailable on this page.", true);
    return;
  }
  try {
    setCameraStatus("Requesting camera access...", true);
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    if (cameraButton) cameraButton.textContent = "Camera on";
    if (realFrameButton) realFrameButton.disabled = sceneName !== "stanford";
    if (autoButton) autoButton.disabled = sceneName !== "stanford";
    const track = cameraStream.getVideoTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    setCameraStatus(
      `${settings.label || "RGB stream"} ${settings.width || "?"}x${settings.height || "?"}`,
      true
    );
  } catch (error) {
    cameraStream = null;
    setCameraStatus(`Camera failed: ${error.message}`, true);
  }
}

async function pollRealSenseLiveFrame() {
  if (!realsenseLiveRunning || realsenseLiveFrameInFlight) return;
  realsenseLiveFrameInFlight = true;
  const started = performance.now();
  let failed = false;
  try {
    const params = new URLSearchParams({
      t: String(Date.now()),
      fast: "true",
      release: "false",
    });
    const response = await fetch(`/api/realsense/frame.jpg?${params.toString()}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }
    const blob = await response.blob();
    const oldUrl = realsenseStreamImage.src;
    realsenseStreamImage.src = URL.createObjectURL(blob);
    if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
    setCameraStatus(`D435i live polling: loaded latest RGB frame in ${((performance.now() - started) / 1000).toFixed(2)}s.`, false);
  } catch (error) {
    failed = true;
    setCameraStatus(`D435i live polling failed: ${error.message}`, true);
  } finally {
    realsenseLiveFrameInFlight = false;
  }
  if (!realsenseLiveRunning) return;
  const elapsed = performance.now() - started;
  const targetGapMs = failed ? 1200 : 180;
  realsenseLiveTimer = window.setTimeout(
    pollRealSenseLiveFrame,
    Math.max(120, targetGapMs - elapsed)
  );
}

function toggleRealSenseLive() {
  if (realsenseLiveRunning) {
    stopRealSenseLive();
    return;
  }
  if (sceneName !== "stanford") {
    setCameraStatus("D435i live preview is currently wired for the Stanford scene.", true);
    return;
  }
  stopBrowserCamera("Stopped browser camera so the native RealSense stream can own the device.");
  cameraPanel.hidden = false;
  cameraVideo.hidden = true;
  realsenseStreamImage.hidden = false;
  realsenseLiveRunning = true;
  if (realsenseLiveButton) realsenseLiveButton.textContent = "Stop D435i";
  setCameraStatus("Starting pyrealsense2 D435i MJPEG stream from the working RealSense venv...", true);
  realsenseStreamImage.onerror = () => {
    if (!realsenseLiveRunning) return;
    stopRealSenseLive("D435i pyrealsense2 stream failed. Try Release camera on /camera-test, then start it again.");
  };
  realsenseStreamImage.onload = () => {
    if (!realsenseLiveRunning) return;
    setCameraStatus("D435i RGB stream running through pyrealsense2. Localize uses the newest cached frame.", false);
  };
  realsenseStreamImage.src = `/api/realsense/stream?t=${Date.now()}`;
}

function showSyntheticSource() {
  stopBrowserCamera();
  stopRealSenseLive();
  cameraPanel.hidden = true;
  captureSource = "synthetic";
  updateSourceControls();
  setRunStatus("Source set to 3D scene snapshot.", true);
}

function showRealSenseSource() {
  if (sceneName !== "stanford") {
    setRunStatus("RealSense relocalization is currently wired for the Stanford splat.", true);
    return;
  }
  captureSource = "realsense";
  updateSourceControls();
  if (!realsenseLiveRunning) toggleRealSenseLive();
  setRunStatus("Source set to RealSense camera.", true);
}

function toggleCaptureSource() {
  if (captureSource === "realsense") {
    showSyntheticSource();
  } else {
    showRealSenseSource();
  }
}

function captureCameraFrame() {
  if (!cameraStream || cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error("Camera is not ready yet");
  }
  const srcWidth = cameraVideo.videoWidth;
  const srcHeight = cameraVideo.videoHeight;
  const maxSide = 960;
  const scale = Math.min(1, maxSide / Math.max(srcWidth, srcHeight));
  const width = Math.max(2, Math.round(srcWidth * scale));
  const height = Math.max(2, Math.round(srcHeight * scale));
  cameraCapture.width = width;
  cameraCapture.height = height;
  const context = cameraCapture.getContext("2d", { willReadFrequently: false });
  context.drawImage(cameraVideo, 0, 0, width, height);
  const imageBase64 = cameraCapture.toDataURL("image/jpeg", 0.88);
  const focal = Math.max(width, height) * 0.9;
  return {
    imageBase64,
    intrinsics: {
      width,
      height,
      fx: focal,
      fy: focal,
      cx: width / 2,
      cy: height / 2,
    },
  };
}

async function runRealFrameLocalization({ debugImages = true } = {}) {
  if (sceneName !== "stanford") {
    setRunStatus("Real-frame localization is currently wired for the Stanford splat.", true);
    return;
  }
  if (!cameraStream) {
    await startCamera();
    if (!cameraStream) return;
  }
  if (realFrameInFlight) return;
  setRealFrameRunning(true);
  clearRunLogs();
  startStatusTimer();
  successMetric.textContent = "-";
  rotationMetric.textContent = "-";
  translationMetric.textContent = "-";
  runtimeMetric.textContent = "-";
  try {
    const { imageBase64, intrinsics } = captureCameraFrame();
    const poseGuess = matrixWorldToRows(fpCamera);
    setRunStatus(`Sending ${intrinsics.width}x${intrinsics.height} real RGB frame to Modal...`, true);
    const response = await fetch("/api/localize-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: sceneName,
        image_base64: imageBase64,
        pose_matrix: poseGuess,
        intrinsics,
        debug_images: debugImages,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Real-frame localization request failed");
    stopStatusTimer();
    applyLocalizationResult(data, "Real frame");
  } catch (error) {
    stopStatusTimer();
    setRunStatus(`Real-frame request failed: ${error.message}`, true);
    successMetric.textContent = "error";
  } finally {
    stopStatusTimer();
    setRealFrameRunning(false);
  }
}

async function runRealSenseLocalization({ debugImages = true } = {}) {
  if (sceneName !== "stanford") {
    setRunStatus("D435i localization is currently wired for the Stanford splat.", true);
    return;
  }
  if (realsenseInFlight) return;
  setRealSenseRunning(true);
  clearRunLogs();
  startStatusTimer();
  successMetric.textContent = "-";
  rotationMetric.textContent = "-";
  translationMetric.textContent = "-";
  runtimeMetric.textContent = "-";
  try {
    const poseGuess = matrixWorldToRows(fpCamera);
    setRunStatus(
      realsenseLiveRunning
        ? "Sending newest cached D435i live RGB frame to Modal..."
        : "Capturing one D435i RGB frame locally, then sending it to Modal...",
      true
    );
    const response = await fetch("/api/localize-realsense-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: sceneName,
        pose_matrix: poseGuess,
        debug_images: debugImages,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "D435i localization request failed");
    stopStatusTimer();
    const intrinsics = data.realsense_intrinsics;
    if (intrinsics) {
      addLog(
        `RealSense intrinsics: ${intrinsics.width}x${intrinsics.height} fx=${intrinsics.fx.toFixed(2)} fy=${intrinsics.fy.toFixed(2)}`
      );
    }
    applyLocalizationResult(data, "D435i frame");
  } catch (error) {
    stopStatusTimer();
    setRunStatus(`D435i request failed: ${error.message}`, true);
    successMetric.textContent = "error";
  } finally {
    stopStatusTimer();
    setRealSenseRunning(false);
  }
}

function toggleAutoRealFrame() {
  if (autoTimer !== null) {
    window.clearInterval(autoTimer);
    autoTimer = null;
    if (autoButton) autoButton.textContent = "Auto 1 Hz";
    setCameraStatus("Auto relocalize stopped", true);
    return;
  }
  if (autoButton) autoButton.textContent = "Stop auto";
  setCameraStatus("Auto relocalize running at 1 Hz", true);
  runRealFrameLocalization({ debugImages: false });
  autoTimer = window.setInterval(() => {
    if (!realFrameInFlight) runRealFrameLocalization({ debugImages: false });
  }, 1000);
}

function updateMotion(deltaSeconds) {
  if (keys.size === 0) return;

  const speed = keys.has("AltLeft") || keys.has("AltRight") ? radius * 0.55 : radius * 0.22;
  let moved = false;
  moveVector.set(0, 0, 0);
  if (keys.has("KeyW")) moveVector.z -= 1;
  if (keys.has("KeyS")) moveVector.z += 1;
  if (keys.has("KeyA")) moveVector.x -= 1;
  if (keys.has("KeyD")) moveVector.x += 1;

  if (moveVector.lengthSq() > 0) {
    moveVector.normalize().multiplyScalar(speed * deltaSeconds);
    velocity.copy(moveVector).applyQuaternion(fpCamera.quaternion);
    fpCamera.position.add(velocity);
    moved = true;
  }
  if (keys.has("Space")) {
    fpCamera.position.addScaledVector(SCENE_UP, speed * deltaSeconds);
    moved = true;
  }
  if (keys.has("KeyC") || keys.has("ControlLeft") || keys.has("ControlRight")) {
    fpCamera.position.addScaledVector(SCENE_UP, -speed * deltaSeconds);
    moved = true;
  }
  if (moved) {
    updateGroundTruthRig();
    motionStatus.textContent = `x ${fpCamera.position.x.toFixed(2)}  y ${fpCamera.position.y.toFixed(2)}  z ${fpCamera.position.z.toFixed(2)}`;
  }
}

async function runLocalization() {
  if (!bounds) return;
  if (!localizationEnabled) {
    setRunStatus("Localization is disabled for this point-cloud-only scene.", true);
    return;
  }
  setRunning(true);
  clearRunLogs();
  startStatusTimer();
  setRunStatus("Preparing current first-person camera pose...", true);
  motionStatus.textContent = "Localizing current pose...";
  successMetric.textContent = "-";
  rotationMetric.textContent = "-";
  translationMetric.textContent = "-";
  runtimeMetric.textContent = "-";
  const gtPose = matrixWorldToRows(fpCamera);
  try {
    const response = await fetch("/api/localize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: sceneName,
        pose_matrix: gtPose,
        init_yaw_error_deg: 2,
        init_translation_error_m: 0.05,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Localization request failed");
    stopStatusTimer();
    applyLocalizationResult(data, "Synthetic test");
  } catch (error) {
    stopStatusTimer();
    motionStatus.textContent = error.message;
    setRunStatus(`Request failed: ${error.message}`, true);
    successMetric.textContent = "error";
  } finally {
    stopStatusTimer();
    setRunning(false);
  }
}

async function runCapture() {
  if (captureSource === "realsense") {
    await runRealSenseLocalization({ debugImages: true });
  } else {
    await runLocalization();
  }
}

function animate() {
  const delta = Math.min(clock.getDelta(), 0.05);
  updateMotion(delta);
  resizeRenderer(fpRenderer, fpCamera, fpCanvas);
  resizeRenderer(mapRenderer, mapCamera, overviewCanvas);
  fpRenderer.render(fpScene, fpCamera);
  mapRenderer.render(mapScene, mapCamera);
  requestAnimationFrame(animate);
}

runButton.addEventListener("click", runCapture);

fpCanvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  isDraggingFirstPerson = true;
  fpCanvas.classList.add("dragging");
  fpCanvas.setPointerCapture(event.pointerId);
});

fpCanvas.addEventListener("pointermove", (event) => {
  if (!isDraggingFirstPerson) return;
  if (event.shiftKey) {
    translateFirstPersonByDrag(event.movementX, event.movementY);
    return;
  }
  if (keys.has("KeyE")) {
    orbitFirstPersonByDrag(event.movementX, event.movementY);
    return;
  }
  rotateFirstPersonByDrag(event.movementX, event.movementY);
});

function stopFirstPersonDrag(event) {
  isDraggingFirstPerson = false;
  fpCanvas.classList.remove("dragging");
  if (event?.pointerId !== undefined && fpCanvas.hasPointerCapture(event.pointerId)) {
    fpCanvas.releasePointerCapture(event.pointerId);
  }
}

fpCanvas.addEventListener("pointerup", stopFirstPersonDrag);
fpCanvas.addEventListener("pointercancel", stopFirstPersonDrag);
fpCanvas.addEventListener("pointerleave", (event) => {
  if (event.buttons === 0) stopFirstPersonDrag(event);
});

overviewCanvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  isDraggingOverview = true;
  overviewCanvas.classList.add("dragging");
  overviewCanvas.setPointerCapture(event.pointerId);
});

overviewCanvas.addEventListener("pointermove", (event) => {
  if (!isDraggingOverview) return;
  if (event.shiftKey) {
    translateCameraByDrag(mapCamera, event.movementX, event.movementY, 1.4);
    motionStatus.textContent = `Moved overview camera: x ${mapCamera.position.x.toFixed(2)}  y ${mapCamera.position.y.toFixed(2)}  z ${mapCamera.position.z.toFixed(2)}`;
    return;
  }
  if (keys.has("KeyE")) {
    orbitOverviewByDrag(event.movementX, event.movementY);
    return;
  }
  rotateOverviewByDrag(event.movementX, event.movementY);
});

function stopOverviewDrag(event) {
  isDraggingOverview = false;
  overviewCanvas.classList.remove("dragging");
  if (event?.pointerId !== undefined && overviewCanvas.hasPointerCapture(event.pointerId)) {
    overviewCanvas.releasePointerCapture(event.pointerId);
  }
}

overviewCanvas.addEventListener("pointerup", stopOverviewDrag);
overviewCanvas.addEventListener("pointercancel", stopOverviewDrag);
overviewCanvas.addEventListener("pointerleave", (event) => {
  if (event.buttons === 0) stopOverviewDrag(event);
});

fpCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  dollyCamera(fpCamera, event.deltaY);
  updateGroundTruthRig();
  motionStatus.textContent = `Moved ground truth: x ${fpCamera.position.x.toFixed(2)}  y ${fpCamera.position.y.toFixed(2)}  z ${fpCamera.position.z.toFixed(2)}`;
}, { passive: false });

overviewCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  dollyCamera(mapCamera, event.deltaY, 2.0);
}, { passive: false });

document.addEventListener("keydown", (event) => {
  keys.add(event.code);
  if (event.code === "KeyL") runCapture();
  if (event.code === "Backquote") {
    logPanel.hidden = !logPanel.hidden;
  }
});

document.addEventListener("keyup", (event) => {
  keys.delete(event.code);
});

logToggle?.addEventListener("click", () => {
  logPanel.hidden = !logPanel.hidden;
});

sourceToggle.addEventListener("click", toggleCaptureSource);
cameraButton?.addEventListener("click", startCamera);
realFrameButton?.addEventListener("click", () => runRealFrameLocalization({ debugImages: true }));
realsenseLiveButton?.addEventListener("click", toggleRealSenseLive);
realsenseButton?.addEventListener("click", () => runRealSenseLocalization({ debugImages: true }));
autoButton?.addEventListener("click", toggleAutoRealFrame);

fetch(`/api/scene/${encodeURIComponent(sceneName)}`)
  .then((response) => {
    if (!response.ok) throw new Error("Could not load scene preview");
    return response.json();
  })
  .then(buildScene)
  .catch((error) => {
    loadStatus.textContent = error.message;
  });

animate();
