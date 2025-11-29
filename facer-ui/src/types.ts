export interface FacePose {
  yaw: number;
  pitch: number;
  roll: number;
  direction_label: string;
}

export interface FaceData {
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  pose: FacePose;
  is_valid_pose: boolean;
  embedding: number[] | null;
}

export interface AnalysisResponse {
  filename: string;
  face_count: number;
  results: FaceData[];
}

export interface FaceRecord {
  id: number;
  image_name: string;
  is_valid_pose: boolean;
  yaw: number;
  pitch: number;
  roll: number;
  created_at: string;
  description: string | null;
  direction: string;
  keywords: string | null;
  classification: string | null;
}
