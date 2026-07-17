import "./ExportButtons.css";

export default function ExportButtons({ jobId, disabled }) {
  const handlePdfDownload = () => {
    window.open(`/export/pdf/${jobId}`, "_blank");
  };

  const handleFhirDownload = () => {
    // The FHIR endpoint needs a record_id, which comes from the result.
    // For now, open the export page — full wiring needs record_id from getJobResult.
    window.open(`/export/fhir/${jobId}`, "_blank");
  };

  return (
    <div className="export-buttons-row">
      <button
        className="export-btn"
        onClick={handlePdfDownload}
        disabled={disabled}
        title="Download Clinical Dossier PDF"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Download PDF
      </button>
      <button
        className="export-btn export-btn-outline"
        onClick={handleFhirDownload}
        disabled={disabled}
        title="Download FHIR Bundle"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
        </svg>
        FHIR Export
      </button>
    </div>
  );
}