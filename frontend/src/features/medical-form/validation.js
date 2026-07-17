/**
 * frontend/src/features/medical-form/validation.js
 * Client-side file validation and polling interval helpers.
 */

export function validateFile(file, kind, maxMb) {
  if (file.size > maxMb * 1024 * 1024) {
    return `${file.name} exceeds ${maxMb} MB size limit`;
  }
  if (kind === "pdf" && file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    return `${file.name} must be a valid PDF document`;
  }
  if (kind === "xray" && !file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".dcm") && !file.name.toLowerCase().endsWith(".dicom")) {
    return `${file.name} must be an image or DICOM (.dcm) file`;
  }
  return null;
}

export function nextPollInterval(status, submittedAtMs) {
  const ageMs = Date.now() - submittedAtMs;
  if (status === "completed" || status === "failed") return 0;
  if (ageMs < 2000) return 200;
  if (status === "queued") return 700;
  return 600;
}
