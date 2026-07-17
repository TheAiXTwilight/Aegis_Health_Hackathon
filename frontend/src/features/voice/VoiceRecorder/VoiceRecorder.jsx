import { useState, useRef } from "react";
import "./VoiceRecorder.css";
import { Waveform } from "./Waveform";

export default function VoiceRecorder({ onComplete, onStatusChange, onAudioCapture }) {
  const [status, setStatus] = useState("idle"); // idle | recording | uploading
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const recognitionRef = useRef(null);
  const finalTranscriptRef = useRef(""); // tracks committed words only

  // ── Browser Speech Recognition (real-time transcription) ──────
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  const startSpeechRecognition = () => {
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let newFinalText = "";

      // event.results contains ALL results from the beginning.
      // event.results[i].isFinal tells us if that result has been committed.
      // We iterate from event.resultIndex (where new results start) to skip
      // results we've already processed.
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          // This is a committed phrase — add it to our permanent transcript
          if (!finalTranscriptRef.current.includes(transcript)) {
            finalTranscriptRef.current += (finalTranscriptRef.current ? " " : "") + transcript;
            newFinalText = finalTranscriptRef.current;
          }
        }
      }

      // Send the full committed transcript to parent (replaces old text, not appends)
      if (newFinalText) {
        onComplete?.(newFinalText, true); // true = replace existing text
      }
    };

    recognition.onerror = () => {
      // Fall back to recording-only mode silently
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        // Ignore
      }
      recognitionRef.current = null;
    }
  };

  // ── MediaRecorder (for backend audio upload) ──────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        streamRef.current?.getTracks().forEach((t) => t.stop());

        // Pass blob to parent for backend VoiceTranscriber submission.
        // Include the final status in the callback so the parent can safely
        // enable Submit only after the audio blob has been captured.
        if (onAudioCapture) {
          onAudioCapture(audioBlob);
        }

        updateStatus("ready");
      };

      mediaRecorder.start();
      updateStatus("recording");
    } catch (err) {
      console.error("Mic access denied:", err);
      alert("Please allow microphone access to record symptoms.");
      updateStatus("idle");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && status === "recording") {
      mediaRecorderRef.current.stop();
      stopSpeechRecognition();
      updateStatus("uploading");
    }
  };

  // ── Click handler ────────────────────────────────────────────
  const handleClick = () => {
    if (status === "idle" || status === "ready") {
      finalTranscriptRef.current = ""; // reset for new recording
      startRecording();
      startSpeechRecognition();
    } else if (status === "recording") {
      stopRecording();
    }
  };

  const updateStatus = (newStatus) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  };

  const isBusy = status === "uploading";
  const isRecording = status === "recording";

  return (
    <div className="voice-recorder-wrapper">
      <button
        type="button"
        onClick={handleClick}
        disabled={isBusy}
        className={`voice-rec-btn ${status}`}
        title={
          status === "idle" || status === "ready"
            ? "Start recording"
            : isRecording
            ? "Stop recording"
            : "Processing recording..."
        }
      >
        {(status === "idle" || status === "ready") && (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
          </svg>
        )}
        {isRecording && <span className="rec-stop-square"></span>}
        {isBusy && <span className="rec-spinner"></span>}
      </button>

      {isBusy && <span className="status-text">Processing...</span>}
      {status === "ready" && <span className="status-text">Voice ready</span>}
    </div>
  );
}

export { Waveform };
