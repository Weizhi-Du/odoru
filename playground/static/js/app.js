const socket = io();
const video = document.getElementById('video');
const startButton = document.getElementById('start-btn');
const stopButton = document.getElementById('stop-btn');

let captureInterval;

// Start capturing video stream
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

// Function to capture and send video frames
function sendFrame() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  
  // Set canvas dimensions to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  // Draw the current video frame to the canvas
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert canvas to base64 string (compressed as JPEG)
  const frame = canvas.toDataURL('image/jpeg', 0.7); // Compress to 70% quality

  // Emit the frame via Socket.IO
  socket.emit('frame', { frame: frame.split(',')[1] });  // Send the base64 image without the data URL prefix
}

// Start capturing frames at 30 FPS when 'Start Recording' button is clicked
startButton.addEventListener('click', () => {
  captureInterval = setInterval(sendFrame, 1000 / 30);  // 30 fps
});

// Stop capturing frames when 'Stop Recording' is clicked
stopButton.addEventListener('click', () => {
  clearInterval(captureInterval);
});

// Handle the response from the server
socket.on('batch_score', (data) => {
  console.log('Batch score received:', data.score);
});
