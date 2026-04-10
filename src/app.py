"""
app.py
──────
FastAPI 網頁介面：上傳影片 → 設定標籤 → 執行 pipeline → 查看並人工校正結果

執行方式：
  uvicorn src.app:app --reload

然後開啟瀏覽器：http://localhost:8000
"""

import io
import shutil
import threading
import zipfile
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

app = FastAPI()

# ── Directories ────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRAMES_DIR = BASE_DIR / "data" / "extracted_frames"
OUTPUT_DIR = BASE_DIR / "data" / "labeled_output"
STATIC_DIR = Path(__file__).parent / "static"

for _d in [UPLOAD_DIR, FRAMES_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── In-memory job state ────────────────────────────────────
job: dict = {
    "status": "idle",   # idle | extracting | classifying | done | error
    "progress": 0,
    "total": 0,
    "message": "",
    "pos_folder": "positive",
    "neg_folder": "negative",
}

_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None
_conf: dict[str, float] = {}   # "folder/filename" -> positive probability


# ── Request schemas ────────────────────────────────────────
class RunConfig(BaseModel):
    video_path: str
    positive_label: str
    negative_labels: list[str]
    positive_folder: str = "positive"
    negative_folder: str = "negative"
    frame_interval: int = 10
    max_frames: int = 200
    confidence_threshold: float = 0.55


class MoveRequest(BaseModel):
    filenames: list[str]
    from_folder: str
    to_folder: str


class DeleteRequest(BaseModel):
    filenames: list[str]
    folder: str


# ── Pipeline ───────────────────────────────────────────────
def _run_pipeline(cfg: RunConfig) -> None:
    global job, _clip_model, _clip_processor, _conf

    try:
        # ── Step 1: Extract frames ──────────────────────────
        job.update(status="extracting", progress=0, total=0,
                   message="正在開啟影片...")

        if FRAMES_DIR.exists():
            shutil.rmtree(FRAMES_DIR)
        FRAMES_DIR.mkdir(parents=True)

        video_path = BASE_DIR / cfg.video_path
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片：{video_path}")

        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimated = total_raw // cfg.frame_interval
        if cfg.max_frames > 0:
            estimated = min(estimated, cfg.max_frames)
        job["total"] = estimated

        saved: list[Path] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % cfg.frame_interval == 0:
                dst = FRAMES_DIR / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(dst), frame)
                saved.append(dst)
                job["progress"] = len(saved)
                job["message"]  = f"抽取 frames... {len(saved)} / {estimated}"
                if cfg.max_frames > 0 and len(saved) >= cfg.max_frames:
                    break
            frame_idx += 1
        cap.release()

        # ── Step 2: Classify ────────────────────────────────
        job.update(status="classifying", progress=0, total=len(saved),
                   message="載入 CLIP 模型...",
                   pos_folder=cfg.positive_folder,
                   neg_folder=cfg.negative_folder)
        _conf.clear()

        for folder in [cfg.positive_folder, cfg.negative_folder]:
            p = OUTPUT_DIR / folder
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)

        if _clip_model is None:
            _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()

        labels = [cfg.positive_label] + cfg.negative_labels

        for i, img_path in enumerate(saved, 1):
            image  = Image.open(img_path).convert("RGB")
            inputs = _clip_processor(
                text=labels, images=image, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = _clip_model(**inputs)
            probs    = outputs.logits_per_image.softmax(dim=1)
            pos_prob = probs[0][0].item()

            target = cfg.positive_folder if pos_prob >= cfg.confidence_threshold \
                     else cfg.negative_folder
            shutil.copy2(img_path, OUTPUT_DIR / target / img_path.name)
            _conf[f"{target}/{img_path.name}"] = round(pos_prob, 4)

            job["progress"] = i
            job["message"]  = f"分類中... {i} / {len(saved)}"

        job.update(status="done",
                   message=f"完成！共處理 {len(saved)} 張 frame")

    except Exception as exc:
        job.update(status="error", message=str(exc))


# ── Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": f"data/uploads/{file.filename}"}


@app.post("/api/run")
async def run(cfg: RunConfig):
    if job["status"] in ("extracting", "classifying"):
        raise HTTPException(status_code=409, detail="Pipeline 正在執行中")
    threading.Thread(target=_run_pipeline, args=(cfg,), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    return job


@app.get("/api/results")
async def get_results():
    def list_images(folder_name: str) -> list[dict]:
        folder = OUTPUT_DIR / folder_name
        if not folder.exists():
            return []
        return [
            {"filename": f.name, "confidence": _conf.get(f"{folder_name}/{f.name}")}
            for f in sorted(folder.iterdir())
            if f.suffix.lower() in {".jpg", ".png"}
        ]

    pos = job.get("pos_folder", "positive")
    neg = job.get("neg_folder", "negative")
    return {
        "positive": {"folder": pos, "images": list_images(pos)},
        "negative": {"folder": neg, "images": list_images(neg)},
    }


@app.get("/api/image/{folder}/{filename}")
async def serve_image(folder: str, filename: str):
    path = OUTPUT_DIR / folder / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path)


@app.post("/api/move")
async def move_image(req: MoveRequest):
    (OUTPUT_DIR / req.to_folder).mkdir(parents=True, exist_ok=True)
    for filename in req.filenames:
        src = OUTPUT_DIR / req.from_folder / filename
        dst = OUTPUT_DIR / req.to_folder   / filename
        if src.exists():
            shutil.move(str(src), str(dst))
    return {"ok": True}


@app.post("/api/delete")
async def delete_image(req: DeleteRequest):
    for filename in req.filenames:
        path = OUTPUT_DIR / req.folder / filename
        if path.exists():
            path.unlink()
    return {"ok": True}


@app.get("/api/download")
async def download_results():
    pos = job.get("pos_folder", "positive")
    neg = job.get("neg_folder", "negative")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder_name in [pos, neg]:
            folder_path = OUTPUT_DIR / folder_name
            if not folder_path.exists():
                continue
            for img in sorted(folder_path.iterdir()):
                if img.suffix.lower() in {".jpg", ".png"}:
                    zf.write(img, f"{folder_name}/{img.name}")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="labeled_output.zip"'},
    )
