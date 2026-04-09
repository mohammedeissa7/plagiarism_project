import os
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from run import PlagiarismChecker

app = FastAPI(
    title="Plagiarism Checker API",
    description="Upload .docx files and receive a pairwise cosine-similarity matrix (%).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup (downloads on first run)
checker: PlagiarismChecker | None = None


@app.on_event("startup")
async def load_model():
    global checker
    print("[*] Loading sentence-transformer model …")
    checker = PlagiarismChecker(model_name=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    print("[+] Model ready.")


# ──────────────────────────────────────────────
# POST /check
# Body: multipart/form-data  →  files: List[UploadFile]
# ──────────────────────────────────────────────
@app.post(
    "/",
    summary="Run plagiarism check on uploaded .docx files",
    response_description="Pairwise similarity matrix in percent",
)
async def check_plagiarism(files: List[UploadFile] = File(...)):
    """
    Accepts **two or more** `.docx` files and returns:

    ```json
    {
      "files": ["a.docx", "b.docx"],
      "matrix": [[100.0, 72.4], [72.4, 100.0]]
    }
    ```

    - `files` — ordered list of uploaded file names
    - `matrix[i][j]` — similarity (%) between file *i* and file *j*
    """
    if len(files) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least 2 .docx files are required for comparison.",
        )

    non_docx = [f.filename for f in files if not f.filename.lower().endswith(".docx")]
    if non_docx:
        raise HTTPException(
            status_code=415,
            detail=f"Only .docx files are accepted. Rejected: {non_docx}",
        )

    # Save uploads to a temp directory so run.py helpers can read them
    tmp_dir = tempfile.mkdtemp(prefix="plagiarism_")
    try:
        file_names: List[str] = []
        texts: List[str] = []

        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            file_names.append(upload.filename)
            texts.append(checker.read_docx(dest))

        matrix_np = checker.compute_similarity_matrix(texts)
        matrix_list = matrix_np.tolist()

        return JSONResponse(
            content={
                "files": file_names,
                "matrix": matrix_list,

                "high_similarity_pairs": _high_pairs(file_names, matrix_list, threshold=80.0),
            }
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _high_pairs(names: List[str], matrix: List[List[float]], threshold: float):
    """Return pairs whose similarity exceeds `threshold` (excluding diagonal)."""
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            score = matrix[i][j]
            if score >= threshold:
                pairs.append(
                    {"file_a": names[i], "file_b": names[j], "similarity": score}
                )
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs


