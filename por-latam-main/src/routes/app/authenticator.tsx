import { Box, Typography } from "@mui/material";
import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

export const AuthenticatorPage = () => {
  return (
    <Box>
      <Typography variant="h3">Autenticación</Typography>
      <Webcam style={{ borderRadius: "10px", width: "70%" }} />
      <WebcamCapture />
    </Box>
  );
};

const WebcamCapture = () => {
  const [deviceId, setDeviceId] = useState<any>({});
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  const handleDevices = useCallback(
    (mediaDevices: MediaDeviceInfo[]) => setDevices(mediaDevices),
    // setDevices(mediaDevices.filter(({ kind }) => kind === "videoinput")),
    [setDevices]
  );

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(handleDevices);
  }, [handleDevices]);

  return (
    <>
      {devices.map((device, key) => (
        <div key={key}>
          {/* <Webcam
            audio={false}
            videoConstraints={{ deviceId: device.deviceId }}
          /> */}
          <p>{device.kind} a</p>
          <p>{device.label} b</p>
          <p>{device.deviceId} c</p>
          <p>{device.groupId} d</p>
        </div>
      ))}
    </>
  );
};

/* 
const CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: { min: 1024, max: 1920, ideal: 1200 },
    height: { min: 480, max: 1080, ideal: 720 },
  },
  audio: true,
};

export const AuthenticatorPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia();
      videoRef.current!.srcObject = stream;
    } catch (error) {
      console.log(error);
    }
  }

  useEffect(() => {
    if (canvasRef.current && videoRef.current && audioRef.current) {
      //   const webcam = new Webca();
      startWebcam();
    }
  }, []);
  return (
    <Box>
      <Typography variant="h3">Autenticación</Typography>
      <video
        ref={videoRef}
        id="webcam"
        autoPlay
        playsInline
        width="640"
        height="480"
      ></video>
      <canvas ref={canvasRef} id="canvas" className="d-none"></canvas>
      <audio
        ref={audioRef}
        id="snapSound"
        src="audio/snap.wav"
        preload="auto"
      ></audio>
    </Box>
  );
};
*/

/* 
const WebcamStreamCapture = () => {
  const webcamRef = React.useRef(null);
  const mediaRecorderRef = React.useRef(null);
  const [capturing, setCapturing] = React.useState(false);
  const [recordedChunks, setRecordedChunks] = React.useState([]);

  const handleStartCaptureClick = React.useCallback(() => {
    setCapturing(true);
    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: "video/webm"
    });
    mediaRecorderRef.current.addEventListener(
      "dataavailable",
      handleDataAvailable
    );
    mediaRecorderRef.current.start();
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = React.useCallback(
    ({ data }) => {
      if (data.size > 0) {
        setRecordedChunks((prev) => prev.concat(data));
      }
    },
    [setRecordedChunks]
  );

  const handleStopCaptureClick = React.useCallback(() => {
    mediaRecorderRef.current.stop();
    setCapturing(false);
  }, [mediaRecorderRef, webcamRef, setCapturing]);

  const handleDownload = React.useCallback(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: "video/webm"
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      document.body.appendChild(a);
      a.style = "display: none";
      a.href = url;
      a.download = "react-webcam-stream-capture.webm";
      a.click();
      window.URL.revokeObjectURL(url);
      setRecordedChunks([]);
    }
  }, [recordedChunks]);

  return (
    <>
      <Webcam audio={false} ref={webcamRef} />
      {capturing ? (
        <button onClick={handleStopCaptureClick}>Stop Capture</button>
      ) : (
        <button onClick={handleStartCaptureClick}>Start Capture</button>
      )}
      {recordedChunks.length > 0 && (
        <button onClick={handleDownload}>Download</button>
      )}
    </>
  );
};
*/
