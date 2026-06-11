from __future__ import annotations

from pydantic import BaseModel, Field


class XRayResult(BaseModel):
    """
    Output of XRayProcessor (Step 3).

    findings  — checklist items selected by the clinician
    free_text — unstructured clinician notes
    """

    findings: list[str] = Field(
        default_factory=list,
        description=(
            "Selected findings from the standard checklist. "
            "Valid items: Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax, "
            "Consolidation, Atelectasis, Infiltrates, Pulmonary Edema, "
            "Nodule / Mass, Fracture, Normal / No significant findings."
        ),
    )

    free_text: str | None = Field(
        default=None,
        description="Unstructured clinician findings not covered by the checklist.",
    )

    schema_version: str = "1.0"