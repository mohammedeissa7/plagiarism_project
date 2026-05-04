import os
import asyncio
from typing import List
from fastapi import FastAPI, BackgroundTasks, HTTPException
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
    
    webhook_data = None
    
    try:
        # Download all files
        file_paths = []
        for sub in submissions:
            filename = f"{sub.studentId}_{sub.submissionId}.docx"
            filepath = os.path.join(temp_dir, filename)
            
            try:
                await download_file(sub.fileUrl, filepath)
                file_paths.append((sub, filepath))
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP {e.response.status_code} downloading {sub.fileUrl}")
            except httpx.RequestError as e:
                raise Exception(f"Network error downloading {sub.fileUrl}: {str(e)}")
            except Exception as e:
                raise Exception(f"Failed to download {sub.fileUrl}: {str(e)}")
        
        # Extract text from all files
        texts = []
        sub_map = []
        for sub, fpath in file_paths:
            try:
                text = checker.read_docx(fpath)
                texts.append(text)
                sub_map.append(sub)
            except Exception as e:
                raise Exception(f"Failed to read DOCX {sub.submissionId}: {str(e)}")
        
        # Compute similarity matrix
        try:
            matrix = checker.compute_similarity_matrix(texts)
        except Exception as e:
            raise Exception(f"Failed to compute similarity: {str(e)}")
        
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
        
        # Build success webhook payload
        webhook_data = WebhookResponse(
            assignmentId=assignment_id,
            hasError=False,
            results=results
        )
        print(f"[✓] Plagiarism check completed for assignment {assignment_id}")
        
    except Exception as e:
        # Build error webhook payload
        error_msg = str(e)
        webhook_data = WebhookResponse(
            assignmentId=assignment_id,
            hasError=True,
            errorMessage=error_msg
        )
        print(f"[✗] Error processing assignment {assignment_id}: {error_msg}")
    
    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[✓] Cleaned up temp files for {assignment_id}")
            except Exception as e:
                print(f"[!] Failed to cleanup temp files: {e}")
    
    # Send webhook with retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                webhook_secret = os.getenv("PLAGIARISM_WEBHOOK_SECRET")
                response = await client.post(webhook_url,json=webhook_data.dict(),headers={"x-plagiarism-webhook-secret": webhook_secret})
                response.raise_for_status()
                print(f"[✓] Webhook sent successfully (attempt {attempt + 1})")
                break
        except httpx.HTTPStatusError as e:
            print(f"[!] Webhook HTTP error (attempt {attempt + 1}/{max_retries}): {e.response.status_code}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"[✗] Failed to send webhook after {max_retries} attempts: {e}")
        except httpx.RequestError as e:
            print(f"[!] Webhook request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"[✗] Failed to send webhook after {max_retries} attempts: {e}")
        except Exception as e:
            print(f"[!] Webhook unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"[✗] Failed to send webhook after {max_retries} attempts: {e}")


@app.post("/check", status_code=202)
async def check_plagiarism(req: CheckRequest, background_tasks: BackgroundTasks):
    """Accept plagiarism check request, return 202, process in background"""
    
    # Validate input
    if not req.submissions or len(req.submissions) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 submissions required for plagiarism check"
        )
    
    if len(req.submissions) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 submissions allowed per request"
        )
    
    # Get webhook URL from env
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        raise HTTPException(
            status_code=500,
            detail="WEBHOOK_URL not configured"
        )
    
    # Add background task
    background_tasks.add_task(
        process_plagiarism,
        req.assignmentId,
        req.submissions,
        webhook_url
    )
    
    print(f"[✓] Accepted plagiarism check for assignment {req.assignmentId} with {len(req.submissions)} submissions")
    
    return {"status": "accepted"}


@app.get("/health")
def health():
    return {"status": "ok"}