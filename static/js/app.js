const videoElement = document.getElementById('video');
const videoSelect = document.getElementById('videoSource');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const toggleDetectionButton = document.getElementById('toggleDetectionButton');
const qrDetectionButton = document.getElementById('qrDetectionButton');
const iconCountersElement = document.getElementById('iconCounters');
let currentStream;
let isDetectionEnabled = false;
let isQRDetectionEnabled = false;

// Function to handle devices
navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);

function gotDevices(deviceInfos) {
    const values = videoSelect.value;
    while (videoSelect.firstChild) {
        videoSelect.removeChild(videoSelect.firstChild);
    }
    for (let i = 0; i !== deviceInfos.length; ++i) {
        const deviceInfo = deviceInfos[i];
        const option = document.createElement('option');
        option.value = deviceInfo.deviceId;
        if (deviceInfo.kind === 'videoinput') {
            option.text = deviceInfo.label || `camera ${videoSelect.length + 1}`;
            videoSelect.appendChild(option);
        }
    }
    if (Array.prototype.slice.call(videoSelect.childNodes).some(n => n.value === values)) {
        videoSelect.value = values;
    }
    console.log("Devices populated:", deviceInfos);
}

function handleError(error) {
    console.error('Error: ', error);
}

async function startStream() {
    console.log("Starting stream...");
    if (typeof currentStream !== 'undefined') {
        console.log("Stopping existing stream before starting a new one...");
        stopStream();
    }
    const videoSource = videoSelect.value;
    const constraints = {
        video: {
            deviceId: videoSource ? { exact: videoSource } : undefined
        }
    };
    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = currentStream;
        videoSelect.disabled = true;  // Disable the dropdown
        const detectionParam = isDetectionEnabled ? 'true' : 'false';
        const qrDetectionParam = isQRDetectionEnabled ? 'true' : 'false';
        videoElement.src = `/video_feed?detection=${detectionParam}&qr_detection=${qrDetectionParam}`;
        console.log(`Stream started successfully with device: ${videoSource}, detection: ${detectionParam}, qr_detection: ${qrDetectionParam}`);
    } catch (error) {
        console.error('Error accessing the camera: ', error);
    }
}

function stopStream() {
    console.log("Stopping stream...");
    if (currentStream) {
        const tracks = currentStream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
        currentStream = null;
        videoSelect.disabled = false;  // Enable the dropdown
        videoElement.style.backgroundColor = 'black';  // Turn the video element black
        videoElement.src = "";  // Clear the src attribute
        console.log("Stream stopped and video element turned black.");
    }
}

function toggleDetection() {
    isDetectionEnabled = !isDetectionEnabled;
    toggleDetectionButton.textContent = isDetectionEnabled ? 'Disable Detection' : 'Enable Detection';
    toggleDetectionButton.className = isDetectionEnabled ? 'button-green' : 'button-red';
    console.log("Object detection enabled:", isDetectionEnabled);
}

function toggleQRDetection() {
    isQRDetectionEnabled = !isQRDetectionEnabled;
    qrDetectionButton.textContent = isQRDetectionEnabled ? 'Disable QR Detection' : 'Enable QR Detection';
    qrDetectionButton.className = isQRDetectionEnabled ? 'button-green' : 'button-yellow';
    console.log("QR detection enabled:", isQRDetectionEnabled);
}

// Event listeners for start, stop, toggle detection, and toggle QR detection buttons
startButton.addEventListener('click', startStream);
stopButton.addEventListener('click', stopStream);
toggleDetectionButton.addEventListener('click', toggleDetection);
qrDetectionButton.addEventListener('click', toggleQRDetection);

// Function to fetch and update icon counters
function updateIconCounters() {
    fetch('/icon_counters')
        .then(response => response.json())
        .then(data => {
            let counterText = "QR Codes Detected:<br>";
            for (const [icon, count] of Object.entries(data)) {
                counterText += `${icon}: ${count}<br>`;
            }
            iconCountersElement.innerHTML = counterText;
        })
        .catch(error => console.error('Error fetching icon counters:', error));
}

// Update icon counters every second
setInterval(updateIconCounters, 1000);