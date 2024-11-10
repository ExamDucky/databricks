from pydantic import BaseModel
from typing import Optional, List, Dict

# Pydantic models
class PlagiarismResult(BaseModel):
    submission_id: str
    similarity_score: float
    matched_file: str
    similarity_type: str  # "Traditional" or "LLM-based_code" or "LLM-based_graph"
    timestamp: str

class SubmissionResponse(BaseModel):
    submission_id: str
    status: str
    file_path: str
    timestamp: str

class PlagiarismReport(BaseModel):
    submission_id: str
    traditional_matches: List[PlagiarismResult]
    llm_matches_code: List[PlagiarismResult]
    llm_matches_graph: List[PlagiarismResult]
    summary: dict
    charts_data: Optional[Dict] = None 
    
class HealthCheck(BaseModel):
    """
    Response model to validate and return when performing a health check.
    """
    status: str = "OK"