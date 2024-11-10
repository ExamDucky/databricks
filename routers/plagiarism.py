from fastapi import File, UploadFile, HTTPException, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
from models.models import SubmissionResponse
from fastapi.responses import HTMLResponse
from utils.plagiarism import download_file_from_dbfs
from utils.plagiarism import process_and_store_results, generate_html_report, parse_plagiarism_results, parse_similarity_data
from utils.shared import databricks_api, upload_to_databricks
from datetime import datetime
from typing import Optional
import tempfile
import aiofiles
import json
from config import settings
import os

router = APIRouter(
    prefix="/plagiarism",
    tags=['Plagiarism']
)

@router.get("/report/{submission_id}")
def get_detailed_report(submission_id: str, format: str = "json", inputs: str= "summary"):
    """
    Get a detailed plagiarism report for a submission.
    Parameters:
        submission_id: The ID of the submission
        format: Response format - 'json' or 'html' (default: 'json')
    """
    try:
        # Get results from DBFS
        results_path = f"{settings.DBFS_ROOT_PATH}results/{submission_id}"
        raw_results = download_file_from_dbfs(results_path)
        
        report_data = parse_plagiarism_results(raw_results)
        if format == "html":
            data = json.loads(raw_results)
            trad_report, llm_report_code, llm_report_graph = parse_similarity_data(data)
            return HTMLResponse(content=generate_html_report(submission_id, data, trad_report, llm_report_code, llm_report_graph), status_code=200)
        else:
            return JSONResponse(content=report_data)
            
    except Exception as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Report not found or still processing: {str(e)}"
        )
    
@router.get("/reports/summary")
def get_reports_summary(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get a summary of all plagiarism reports within a date range.
    Parameters:
        start_date: Optional start date (YYYY-MM-DD HH:MM)
        end_date: Optional end date (YYYY-MM-DD HH:MM)
    """
    try:
        # List all results in DBFS
        results_path = f"{settings.DBFS_ROOT_PATH}results/"
        response = databricks_api("/api/2.0/dbfs/list", "get", {"path": results_path})
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve reports")
            
        reports_summary = []
        for file in response.json().get("files", []):
            try:
                report_data = json.loads(download_file_from_dbfs(file["path"]))
                timestamp = report_data["summary"]["timestamp"]
                datetime_formated_timestamp = datetime.fromisoformat(timestamp)
                start_date_compare = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
                end_date_compare = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
                
                if start_date_compare and datetime_formated_timestamp < start_date_compare:
                    continue
                if end_date_compare and datetime_formated_timestamp > end_date_compare:
                    continue
                
                
                reports_summary.append({
                    "submission_id": report_data["submission_id"],
                    "timestamp": timestamp,
                    "total_matches": len(report_data["traditional_matches"]) + len(report_data["llm_matches_code"]) + len(report_data["llm_matches_graph"]),
                    "highest_similarity": max(
                        [m["similarity_score"] for m in report_data["traditional_matches"]] +
                        [m["similarity_score"] for m in report_data["llm_matches_code"]] +
                        [m["similarity_score"] for m in report_data["llm_matches_graph"]] +
                        [0]
                    )
                })
            except Exception as e:
                print(f"Error processing report {file['path']}: {str(e)}")
                continue
                
        return JSONResponse(content={
            "total_reports": len(reports_summary),
            "reports": sorted(reports_summary, key=lambda x: x["timestamp"], reverse=True)
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )

@router.get("/check-results/{submission_id}")
def get_plagiarism_results(submission_id: str):
    """Get the results of a plagiarism check"""
    try:
        results_path = f"{settings.DBFS_ROOT_PATH}results/{submission_id}"
        results = download_file_from_dbfs(results_path)
        return JSONResponse(content={"results": results})
    except Exception as _:
        return JSONResponse(content={"status": "processing"})
    
@router.post("/upload-and-check/")
async def upload_and_check_plagiarism(
    name: str,
    background_tasks: BackgroundTasks,
    test_file: UploadFile = File(...)
    
):
    try:
        # Save files temporarily
        temp_test_path = tempfile.NamedTemporaryFile(delete=False).name
        
        dbfs_path = f"dbfs:/baza/{name}/"
        
        # Create directory in DBFS
        databricks_api("/api/2.0/dbfs/mkdirs", "post", {"path": f"/baza/{name}"})
        
        # Save files locally first
        async with aiofiles.open(temp_test_path, 'wb') as out_file:
            content = await test_file.read()
            await out_file.write(content)
        
        
        # Upload to Databricks
        test_file_path = dbfs_path + test_file.filename
        
        upload_to_databricks(temp_test_path, test_file_path)
        
        # Queue plagiarism check
        submission_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(process_and_store_results, test_file_path, submission_id)
        
        return SubmissionResponse(
            submission_id=submission_id,
            status="processing",
            file_path=test_file_path,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_test_path):
            os.unlink(temp_test_path)