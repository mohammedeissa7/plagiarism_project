import os
import asyncio
from typing import List
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
import httpx
from run import PlagiarismChecker


app = FastAPI()
checker = PlagiarismChecker()


class Submission(BaseModel):
    studentId: str
    submissionId: str
    fileUrl: str


class CheckRequest(BaseModel):
    assignmentId: str
    submissions: List[Submission]


class Match(BaseModel):
    comparedWithStudent: str
    comparedWithSubmission: str
    similarityPercentage: float


class Result(BaseModel):
    studentId: str
    submissionId: str
    matches: List[Match]


class WebhookResponse(BaseModel):
    assignmentId: str
    hasError: bool
    errorMessage: str = ""
    results: List[Result] = []


async def download_file(url: str, dest: str):
    """Download file from URL to local path"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(response.content)


async def process_plagiarism(assignment_id: str, submissions: List[Submission], webhook_url: str):
    """Background task: download files, check plagiarism, send webhook"""
    temp_dir = f"/tmp/plag_{assignment_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download all files
        file_paths = []
        for sub in submissions:
            filename = f"{sub.studentId}_{sub.submissionId}.docx"
            filepath = os.path.join(temp_dir, filename)
            await download_file(sub.fileUrl, filepath)
            file_paths.append((sub, filepath))
        
        # Extract text from all files
        texts = []
        sub_map = []
        for sub, fpath in file_paths:
            text = checker.read_docx(fpath)
            texts.append(text)
            sub_map.append(sub)
        
        # Compute similarity matrix
        matrix = checker.compute_similarity_matrix(texts)
        
        # Build results
        results = []
        n = len(submissions)
        for i in range(n):
            matches = []
            for j in range(n):
                if i != j and matrix[i][j] > 0:  # Skip self-comparison and zeros
                    matches.append(Match(
                        comparedWithStudent=sub_map[j].studentId,
                        comparedWithSubmission=sub_map[j].submissionId,
                        similarityPercentage=round(matrix[i][j], 1)
                    ))
            
            results.append(Result(
                studentId=sub_map[i].studentId,
                submissionId=sub_map[i].submissionId,
                matches=matches
            ))
        
        # Send success webhook
        webhook_data = WebhookResponse(
            assignmentId=assignment_id,
            hasError=False,
            results=results
        )
        
    except Exception as e:
        # Send error webhook
        webhook_data = WebhookResponse(
            assignmentId=assignment_id,
            hasError=True,
            errorMessage=str(e)
        )
    
    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Send webhook
    async with httpx.AsyncClient() as client:
        webhook_secret = os.getenv("PLAGIARISM_WEBHOOK_SECRET")
        await client.post(
            webhook_url,
            json=webhook_data.dict(),
            headers={
            "x-plagiarism-webhook-secret": webhook_secret
        }
            )


@app.post("/check",status_code=status.HTTP_202_ACCEPTED)
async def check_plagiarism(req: CheckRequest, background_tasks: BackgroundTasks):
    """Accept plagiarism check request, return 202, process in background"""
    webhook_url = os.getenv("WEBHOOK_URL", "http://localhost:3000/api/v1/similarity/webhook")
    
    background_tasks.add_task(
        process_plagiarism,
        req.assignmentId,
        req.submissions,
        webhook_url
    )
    
    return {"status": "accepted"}


@app.get("/health")
def health():
    return {"status": "ok"}