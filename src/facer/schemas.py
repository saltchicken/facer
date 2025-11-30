from pydantic import BaseModel, Field
from typing import List, Optional


class FacePose(BaseModel):
    """Represents the orientation of the head."""

    yaw: float = Field(..., description="Left/Right rotation in degrees")
    pitch: float = Field(..., description="Up/Down rotation in degrees")
    roll: float = Field(..., description="Tilt rotation in degrees")
    direction_label: str = Field(
        ..., description="Text description like 'straight', 'left', etc."
    )


class FaceData(BaseModel):
    """
    All the data needed for a PostgreSQL record.
    """

    bbox: List[float] = Field(
        ..., description="[x1, y1, x2, y2] coordinates in the original image"
    )
    pose: FacePose
    is_valid_pose: bool = Field(
        ..., description="True if pose is within thresholds (looking at camera)"
    )
    embedding: Optional[List[float]] = Field(
        None, description="1x512 (or similar) vector for face recognition"
    )
    # ‼️ Added classification field to carry auto-detected names
    classification: Optional[str] = Field(
        None, description="Auto-detected classification from database match"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [100.0, 150.0, 300.0, 350.0],
                "pose": {
                    "yaw": 5.2,
                    "pitch": -2.1,
                    "roll": 0.5,
                    "direction_label": "straight",
                },
                "is_valid_pose": True,
                "embedding": [0.1238, -0.5432, "...", 0.9982],
                "classification": "John Doe",
            }
        }


class FaceUpdate(BaseModel):
    description: Optional[str] = None
    classification: Optional[str] = None
    keywords: Optional[str] = None


class AnalysisResponse(BaseModel):
    filename: str
    face_count: int
    results: List[FaceData]
    error: Optional[str] = None

