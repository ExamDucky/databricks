from fastapi import HTTPException
import tempfile
from typing import List, Tuple
from datetime import datetime

import requests
from utils.access import download_file_from_dbfs
from utils.shared import databricks_api, upload_to_databricks
from utils.shared import databricks_api, upload_to_databricks
from models.models import PlagiarismReport, PlagiarismResult
from transformers import AutoTokenizer, AutoModel
import torch 
import mlflow
import os
import json
from config import settings

os.environ["DATABRICKS_HOST"] = settings.DATABRICKS_URL
os.environ["DATABRICKS_TOKEN"] = settings.TOKEN
os.environ["OAUTH_TOKEN_CODE"] = settings.OAUTH_TOKEN_CODE
os.environ["OAUTH_TOKEN_GRAPH"] = settings.OAUTH_TOKEN_GRAPH
headers_code = {
    "Authorization": f"Bearer {settings.OAUTH_TOKEN_CODE}",
    "Content-Type": "application/json"
}

headers_graph = {
    "Authorization": f"Bearer {settings.OAUTH_TOKEN_GRAPH}",
    "Content-Type": "application/json"
}
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(settings.EXPERIMENT_URL)

# Load LLM model for embedding-based similarity detection
model_name = settings.model1_name
tokenizer_code = AutoTokenizer.from_pretrained(model_name)
model_code = AutoModel.from_pretrained(model_name)

model_name2 = settings.model2_name
tokenizer_graph = AutoTokenizer.from_pretrained(model_name2)
model_graph = AutoModel.from_pretrained(model_name2)

def get_ngrams(code: str, n: int = 5) -> List[str]:
    """Generate n-grams from code."""
    tokens = code.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]


def get_embedding(code_snippet: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate embedding for a code snippet using CodeBERT/GraphCodeBERT."""
    inputs_code = tokenizer_code(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    inputs_graph = tokenizer_graph(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs_code = model_code(**inputs_code)
        outputs_graph = model_graph(**inputs_graph)
    return outputs_code.last_hidden_state.mean(dim=1), outputs_graph.last_hidden_state.mean(dim=1)

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Calculate cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=1).item()


def get_timestamp(dbfs_path):
    try:
        response = databricks_api("/api/2.0/dbfs/get-status", "get", {"path": dbfs_path})
        modification_time_ms = response.json().get("modification_time", [])

        return {
            "path": dbfs_path,
            "modification_time": str(modification_time_ms)
        }

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"Failed to retrieve file info: {e.detail}")

async def process_submission(file_path: str, submission_id) -> PlagiarismReport:
    """Process a submission for plagiarism checking using both traditional and LLM methods."""
    code_content = download_file_from_dbfs(file_path)    
    code_embedding, graph_embedding = get_embedding(code_content)
    existing_submissions = get_all_submissions()
    traditional_matches = []
    llm_matches_code = []
    llm_matches_graph = []

    for submission in existing_submissions:
        if submission != file_path and submission.split("/")[2] not in file_path:
            sub = "dbfs:" + submission
            other_code = download_file_from_dbfs(sub)
            # empty submissions should not be analysed
            if len(other_code) < 5 or len(code_content) < 5:
                continue
            # Traditional n-gram similarity
            ngrams1 = set(get_ngrams(code_content))
            ngrams2 = set(get_ngrams(other_code))
            trad_similarity = len(ngrams1 & ngrams2) / (1 if min(len(ngrams1), len(ngrams2)) == 0 else min(len(ngrams1), len(ngrams2))) 
            time_created = get_timestamp(sub)["modification_time"]
            
            if trad_similarity > 0.6:
                traditional_matches.append(PlagiarismResult(
                    submission_id=submission.split('/')[-2],
                    similarity_score=trad_similarity,
                    matched_file=submission,
                    similarity_type="Traditional",
                    timestamp=time_created
                ))
            # LLM-based similarity
            other_code_embedding, other_graph_embedding = get_embedding(other_code)
            llm_similarity_code = cosine_similarity(code_embedding, other_code_embedding)          
            llm_similarity_graph = cosine_similarity(graph_embedding, other_graph_embedding)
            if llm_similarity_code > 0.7:
                llm_matches_code.append(PlagiarismResult(
                    submission_id=submission.split('/')[-2],
                    similarity_score=llm_similarity_code,
                    matched_file=submission,
                    similarity_type="LLM-based_code",
                    timestamp=time_created
                ))
            if llm_similarity_graph > 0.7:
                llm_matches_graph.append(PlagiarismResult(
                    submission_id=submission.split('/')[-2],
                    similarity_score=llm_similarity_graph,
                    matched_file=submission,
                    similarity_type="LLM-based_graph",
                    timestamp=time_created
                ))
    
    # Generate summary
    summary = {
        "total_submissions_checked": len(existing_submissions),
        "traditional_matches_found": len(traditional_matches),
        "llm_matches_found_code": len(llm_matches_code),
        "llm_matches_found_graph": len(llm_matches_graph),
        "highest_traditional_similarity": max([m.similarity_score for m in traditional_matches], default=0),
        "highest_llm_similarity_code": max([m.similarity_score for m in llm_matches_code], default=0),
        "highest_llm_similarity_graph": max([m.similarity_score for m in llm_matches_graph], default=0),
        "timestamp": datetime.now().isoformat()
    }
    
    return PlagiarismReport(
        submission_id=submission_id,
        traditional_matches=traditional_matches,
        llm_matches_code=llm_matches_code,
        llm_matches_graph=llm_matches_graph,
        summary=summary
    )

def parse_plagiarism_results(input_string: str) -> List[PlagiarismResult]:
    traditional_scores = []
    llm_scores = []
    data = json.loads(input_string)
    traditional_matches = data.get('traditional_matches', [])
    for match in traditional_matches:
        traditional_scores.append(match['similarity_score'])

    llm_matches = data.get('llm_matches', [])
    for match in llm_matches:
        llm_scores.append(match['similarity_score'])

    traditional_avg = sum(traditional_scores) / len(traditional_scores) if traditional_scores else 0
    llm_avg = sum(llm_scores) / len(llm_scores) if llm_scores else 0

    return {
        'similarity_distribution': {
            'traditional': {
                'scores': traditional_scores,
                'average': traditional_avg
            },
            'llm': {
                'scores': llm_scores,
                'average': llm_avg
            }
        },
        'matches_summary': {
            'total_matches': len(traditional_matches) + len(llm_matches),
            'traditional_matches': len(traditional_matches),
            'llm_matches': len(llm_matches)
        }
    }

def parse_similarity_data(data):
    trad_report = [x for x in data["traditional_matches"]]
    llm_report_code = [x for x in data["llm_matches_code"]]
    llm_report_graph = [x for x in data["llm_matches_graph"]]
    return trad_report, llm_report_code, llm_report_graph

def generate_visualization_data(report_data: dict) -> dict:
    """Generate data for visualizations in the report."""
    traditional_scores = [m.similarity_score for m in report_data if m.similarity_type == "Traditional"]
    llm_scores_code = [m.similarity_score for m in report_data if m.similarity_type == "LLM-based_code"]
    llm_scores_graph = [m.similarity_score for m in report_data if m.similarity_type == "LLM-based_graph"]
    return {
        "similarity_distribution": {
            "traditional": {
                "scores": traditional_scores,
                "average": sum(traditional_scores) / len(traditional_scores) if traditional_scores else 0
            },
            "llm_code": {
                "scores": llm_scores_code,
                "average": sum(llm_scores_code) / len(llm_scores_code) if llm_scores_code else 0
            },
            "llm_graph": {
                "scores": llm_scores_graph,
                "average": sum(llm_scores_graph) / len(llm_scores_graph) if llm_scores_graph else 0
            }
        },
        "matches_summary": {
            "total_matches": len(traditional_scores) + len(llm_scores_code) + len(llm_scores_graph),
            "traditional_matches": len(traditional_scores),
            "llm_matches_code": len(llm_scores_code),
            "llm_matches_graph": len(llm_scores_graph),
            "llm_matches": len(llm_scores_code) + len(llm_scores_graph)
        }
    }

def generate_html_report(submission_id, charts_data, traditional_matches, llm_matches_code, llm_matches_graph) -> str:
    """Generate an HTML report from the report data."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plagiarism Report - {submission_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background: #f5f5f5; padding: 1rem; border-radius: 4px; }}
            .matches {{ margin-top: 2rem; }}
            .match-item {{ border: 1px solid #ddd; padding: 1rem; margin: 1rem 0; }}
            .score {{ font-weight: bold; color: #d32f2f; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Plagiarism Detection Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Submission ID: {submission_id}</p>
                <p>Total Files Checked: {charts_data["summary"]["total_submissions_checked"]}</p>
                <p>Traditional Matches: {charts_data["summary"]["traditional_matches_found"]}</p>
                <p>LLM-based Matches: {charts_data["summary"]["llm_matches_found_code"]}</p>
                <p>LLM-based Matches: {charts_data["summary"]["llm_matches_found_graph"]}</p>
            </div>
            
            <div class="matches">
                <h2>Traditional Matches</h2>
                {generate_matches_html(traditional_matches)}
                
                <h2>LLM_code-based Matches</h2>
                {generate_matches_html(llm_matches_code)}

                <h2>LLM_graph-based Matches</h2>
                {generate_matches_html(llm_matches_graph)}
            </div>
        </div>
    </body>
    </html>
    """

def generate_matches_html(matches: List) -> str:
    """Generate HTML for matches section."""
    if not matches:
        return "<p>No matches found.</p>"
        
    return "\n".join([
        f"""
        <div class="match-item">
            <h3>Match with {match["matched_file"]}</h3>
            <p>Similarity Score: <span class="score">{match["similarity_score"]:.2%}</span></p>
            <p>Time submitted: <span class="score">{match["timestamp"]}</span></p>
            
        </div>
        """
        for match in matches
    ])

async def process_and_store_results(file_path: str, submission_id: str):
    """Process plagiarism check and store results with enhanced reporting."""
    try:
        with mlflow.start_run():
            mlflow.log_param("file_path", file_path)
            mlflow.log_param("submission_id", submission_id)
            results = await process_submission(file_path, submission_id)
            # Store results in Databricks
            results_path = f"{settings.DBFS_ROOT_PATH}results/{submission_id}"
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(results.json())
                temp_path = f.name
            mlflow.log_artifact(temp_path, "plagiarism_report.json")
            charts_data = generate_visualization_data(results.traditional_matches + results.llm_matches_code + results.llm_matches_graph)
            mlflow.log_metric("traditional_matches_found", charts_data["matches_summary"]["traditional_matches"])
            mlflow.log_metric("llm_matches_found_code", charts_data["matches_summary"]["llm_matches_code"])
            mlflow.log_metric("llm_matches_found_graph", charts_data["matches_summary"]["llm_matches_graph"])
            mlflow.log_metric("llm_matches_found_total", charts_data["matches_summary"]["llm_matches"])
            mlflow.log_metric("average_traditional_similarity", charts_data["similarity_distribution"]["traditional"]["average"])
            mlflow.log_metric("average_llm_code_similarity", charts_data["similarity_distribution"]["llm_code"]["average"])
            mlflow.log_metric("average_llm_graph_similarity", charts_data["similarity_distribution"]["llm_graph"]["average"])

            upload_to_databricks(temp_path, results_path)
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"Error processing submission {submission_id}: {str(e)}")

def get_all_submissions() -> List[str]:
    """Get all code submissions from DBFS"""
    response = databricks_api("/api/2.0/dbfs/list", "get", {"path": settings.DBFS_ROOT_PATH})
    if response.status_code != 200:
        return []
    
    submissions = []
    for folder in response.json().get("files", []):
        if folder["is_dir"]:
            folder_response = databricks_api("/api/2.0/dbfs/list", "get", {"path": folder["path"]})
            if folder_response.status_code == 200:
                for file in folder_response.json().get("files", []):
                    if file["path"].endswith('.c'):  # Only get C files
                        submissions.append(file["path"])
    
    return submissions