"""
WhisperX Batch Processing Service for NVIDIA DGX Spark (Blackwell GPU)
GPU DIARIZATION VERSION - Uses SM_90 spoof for full GPU acceleration

This service provides:
- Perfect transcription (Whisper large-v3)
- Word-level timestamps (Wav2Vec2 alignment)
- Speaker diarization (pyannote.audio) - GPU ACCELERATED!

Built from source to support ARM64 + Blackwell (SM_121) architecture.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import queue
import threading
import time
import uuid
import torch
import re

# Patch 1: PyTorch 2.6+ changed torch.load default to weights_only=True
# Pyannote models were saved with older PyTorch and need weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch 2: NVIDIA's torch version (2.6.0a0+ecf3bae40a) isn't valid semver
# Pyannote.audio tries to parse it and fails. Patch semver to handle this.
import semver.version
_original_semver_parse = semver.version.VersionInfo.parse
@classmethod
def _patched_semver_parse(cls, version):
    # Convert NVIDIA version format to valid semver
    # e.g., "2.6.0a0+ecf3bae40a" -> "2.6.0-alpha.0+ecf3bae40a"
    if isinstance(version, str) and 'a0+' in version:
        version = re.sub(r'(\d+\.\d+\.\d+)a0\+', r'\1-alpha.0+', version)
    elif isinstance(version, str) and 'a0' in version:
        version = re.sub(r'(\d+\.\d+\.\d+)a0', r'\1-alpha.0', version)
    return _original_semver_parse.__func__(cls, version)
semver.version.VersionInfo.parse = _patched_semver_parse

# Patch 3: torchaudio nightly removed APIs that pyannote.audio 3.3.2 still uses
# Create dummy implementations for compatibility
import torchaudio
from typing import NamedTuple

class AudioMetaData(NamedTuple):
    sample_rate: int
    num_frames: int
    num_channels: int

def list_audio_backends():
    return ["ffmpeg", "sox", "sox_io"]

torchaudio.AudioMetaData = AudioMetaData
torchaudio.list_audio_backends = list_audio_backends

import whisperx
import tempfile
import os
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models
whisperx_model = None
align_model = None
diarize_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# WhisperX 3.8.5 defaults to pyannote/speaker-diarization-community-1, which
# is gated. pyannote-community/speaker-diarization-community-1 is the same
# pipeline (config plus segmentation, embedding, and PLDA weights) and is public.
DIARIZE_MODEL = os.getenv(
    "WHISPERX_DIARIZE_MODEL",
    "pyannote-community/speaker-diarization-community-1",
).strip() or "pyannote-community/speaker-diarization-community-1"


def normalize_phrase(text: str) -> str:
    """Casefold and drop punctuation so list entries match Whisper text."""

    folded = text.casefold()
    return " ".join(re.sub(r"[^\w\s]+", " ", folded, flags=re.UNICODE).split())


def hallucination_keys(blob: str) -> tuple[set[str], set[str]]:
    """Return normalized phrases and exact casefolded lines from a newline list."""

    normalized: set[str] = set()
    exact: set[str] = set()
    for line in blob.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        exact.add(" ".join(raw.split()).casefold())
        key = normalize_phrase(raw)
        if key:
            normalized.add(key)
    return normalized, exact


def is_hallucination(text: str, normalized: set[str], exact: set[str]) -> bool:
    stripped = " ".join(str(text).split())
    if not stripped:
        return False
    if stripped.casefold() in exact:
        return True
    key = normalize_phrase(stripped)
    return bool(key) and key in normalized


def drop_hallucination_segments(segments: list, blob: str) -> tuple[list, list[str]]:
    """Remove segments whose whole text is a known hallucination phrase."""

    if not blob.strip():
        return segments, []
    normalized, exact = hallucination_keys(blob)
    if not normalized and not exact:
        return segments, []
    kept = []
    dropped: list[str] = []
    for segment in segments:
        text = segment.get("text", "") if isinstance(segment, dict) else ""
        if is_hallucination(text, normalized, exact):
            dropped.append(" ".join(str(text).split()))
            continue
        kept.append(segment)
    return kept, dropped


# One GPU job at a time. POST /transcribe returns immediately; the worker
# updates per-stage percents that GET /progress/{uid} reports.
STAGES = ("transcribe", "align", "diarize", "assign")
STAGE_LABELS = {
    "transcribe": "Transcribing",
    "align": "Aligning",
    "diarize": "Identifying speakers",
    "assign": "Assigning speakers",
}

_jobs_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_job_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False


class Job:
    def __init__(self, uid: str, tmp_path: str, filename: str, language: str,
                 num_speakers, min_speakers, max_speakers, hallucinations: str):
        self.uid = uid
        self.tmp_path = tmp_path
        self.filename = filename
        self.language = language
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hallucinations = hallucinations
        self.status = "queued"
        self.stage = "transcribe"
        self.percents = {name: 0.0 for name in STAGES}
        self.detail = None
        self.result = None
        self.updated = time.time()

    def mark(self, stage: str, percent: float) -> None:
        with _jobs_lock:
            if self.status == "error":
                return
            self.status = "running"
            self.stage = stage
            index = STAGES.index(stage)
            for name in STAGES[:index]:
                self.percents[name] = 100.0
            value = max(0.0, min(100.0, float(percent)))
            self.percents[stage] = max(self.percents[stage], value)
            self.updated = time.time()

    def finish(self, result: dict) -> None:
        with _jobs_lock:
            for name in STAGES:
                self.percents[name] = 100.0
            self.stage = STAGES[-1]
            self.result = result
            self.status = "done"
            self.updated = time.time()

    def fail(self, exc: BaseException) -> None:
        with _jobs_lock:
            self.status = "error"
            self.detail = str(exc)
            self.updated = time.time()

    def snapshot(self) -> dict:
        with _jobs_lock:
            return {
                "uid": self.uid,
                "status": self.status,
                "stage": self.stage,
                "filename": self.filename,
                "stages": [
                    {
                        "name": name,
                        "label": STAGE_LABELS[name],
                        "percent": round(self.percents[name], 1),
                    }
                    for name in STAGES
                ],
                "detail": self.detail,
            }


def _evict_finished() -> None:
    with _jobs_lock:
        finished = [job for job in _jobs.values() if job.status in {"done", "error"}]
        if len(finished) <= 10:
            return
        finished.sort(key=lambda job: job.updated)
        for job in finished[:-10]:
            _jobs.pop(job.uid, None)


def _run_job(job: Job) -> None:
    global align_model, diarize_pipeline

    job.mark("transcribe", 0)
    logger.info("🎤 %s transcribing %s", job.uid, job.filename)
    result = whisperx_model.transcribe(
        job.tmp_path,
        batch_size=16,
        language=None if job.language == "auto" else job.language,
        progress_callback=lambda percent: job.mark("transcribe", percent),
    )
    job.mark("transcribe", 100)

    language_detected = result["language"]
    logger.info("   Detected language: %s", language_detected)

    segments, dropped = drop_hallucination_segments(result.get("segments") or [], job.hallucinations)
    result["segments"] = segments
    if dropped:
        logger.info("   Dropped %s hallucination segment(s) before alignment", len(dropped))
        for phrase in dropped:
            logger.info("   hallucination: %s", phrase)

    job.mark("align", 0)
    logger.info("⏱️  %s aligning", job.uid)
    if align_model is None or align_model[1] != language_detected:
        align_model = whisperx.load_align_model(language_code=language_detected, device=device)
    if result["segments"]:
        result = whisperx.align(
            result["segments"],
            align_model[0],
            align_model[1],
            job.tmp_path,
            device,
            return_char_alignments=False,
            progress_callback=lambda percent: job.mark("align", percent),
        )
    else:
        logger.info("   No segments left to align")
        result["word_segments"] = []
    job.mark("align", 100)

    job.mark("diarize", 0)
    logger.info("👥 %s diarizing with %s", job.uid, DIARIZE_MODEL)
    if diarize_pipeline is None:
        diarize_pipeline = whisperx.diarize.DiarizationPipeline(
            model_name=DIARIZE_MODEL,
            token=os.getenv("HF_TOKEN") or None,
            device=device,
        )
    diarize_segments = diarize_pipeline(
        job.tmp_path,
        num_speakers=job.num_speakers,
        min_speakers=job.min_speakers,
        max_speakers=job.max_speakers,
        progress_callback=lambda percent: job.mark("diarize", percent),
    )
    job.mark("diarize", 100)

    job.mark("assign", 0)
    logger.info("🔗 %s assigning speakers", job.uid)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    job.mark("assign", 100)

    speakers = {}
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        bucket = speakers.setdefault(speaker, {"duration": 0, "segments": 0})
        bucket["duration"] += segment["end"] - segment["start"]
        bucket["segments"] += 1

    job.finish({
        "status": "success",
        "uid": job.uid,
        "language": language_detected,
        "segments": result["segments"],
        "word_segments": result.get("word_segments", []),
        "speakers": speakers,
        "num_speakers": len(speakers),
        "diarization_device": device,
        "dropped_hallucinations": dropped,
    })
    logger.info("✅ %s complete", job.uid)


def _job_worker() -> None:
    while True:
        uid = _job_queue.get()
        with _jobs_lock:
            job = _jobs.get(uid)
        if job is None:
            continue
        try:
            _run_job(job)
        except Exception as exc:
            logger.exception("job %s failed", uid)
            job.fail(exc)
        finally:
            if job.tmp_path and os.path.exists(job.tmp_path):
                os.remove(job.tmp_path)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            _evict_finished()


def _ensure_worker() -> None:
    global _worker_started
    with _jobs_lock:
        if _worker_started:
            return
        threading.Thread(target=_job_worker, name="whisperx-jobs", daemon=True).start()
        _worker_started = True


def _job_or_404(uid: str) -> Job:
    try:
        uuid.UUID(uid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown job") from exc
    with _jobs_lock:
        job = _jobs.get(uid)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global whisperx_model

    logger.info("🚀 Loading WhisperX models (GPU DIARIZATION ENABLED)...")
    logger.info(f"   Device: {device}")
    logger.info(f"   Compute type: {compute_type}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        # Log the spoofed capability
        cap = torch.cuda.get_device_capability(0)
        logger.info(f"   Compute Capability (reported): SM_{cap[0]}{cap[1]}")

    # Load Whisper model
    logger.info("📦 Loading Whisper large-v3...")
    whisperx_model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type
    )

    logger.info("✅ WhisperX ready for GPU-accelerated batch processing!")
    _ensure_worker()

    yield

    logger.info("👋 Shutting down WhisperX service...")


app = FastAPI(
    title="WhisperX Batch Processing Service (GPU)",
    description="Perfect transcription with GPU-accelerated speaker diarization. Built for NVIDIA DGX Spark (Blackwell).",
    version="1.1.0-gpu",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check"""
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    return {
        "status": "healthy" if whisperx_model else "loading",
        "service": "whisperx-batch-gpu",
        "device": device,
        "diarization_device": device,  # GPU diarization enabled!
        "model": "whisper-large-v3",
        "diarization_model": DIARIZE_MODEL,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": f"SM_{cap[0]}{cap[1]}"
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    num_speakers: int = Form(None),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None),
    hallucinations: str = Form(""),
    wait: bool = Form(False),
):
    """Queue one audio file and return its job id.

    The transcript is fetched later from GET /result/{uid}. Pass wait=true to
    block until that transcript is ready, which is the previous response shape.
    """
    logger.info(f"📥 Received file: {file.filename}")
    _ensure_worker()
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job = Job(
        uid=str(uuid.uuid4()),
        tmp_path=tmp_path,
        filename=file.filename or os.path.basename(tmp_path),
        language=language,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hallucinations=hallucinations,
    )
    with _jobs_lock:
        _jobs[job.uid] = job
    _job_queue.put(job.uid)

    if wait:
        while True:
            snap = job.snapshot()
            if snap["status"] == "done":
                return job.result
            if snap["status"] == "error":
                raise HTTPException(status_code=500, detail=snap["detail"])
            await asyncio.sleep(0.25)

    return JSONResponse(
        status_code=202,
        content={
            "uid": job.uid,
            "status": "queued",
            "progress": f"/progress/{job.uid}",
            "result": f"/result/{job.uid}",
        },
    )


@app.get("/progress/{uid}")
async def job_progress(uid: str):
    """Percent complete for each stage of one queued transcription."""
    return _job_or_404(uid).snapshot()


@app.get("/result/{uid}")
async def job_result(uid: str):
    """Transcript for a finished job. 202 while it is still running."""
    job = _job_or_404(uid)
    snap = job.snapshot()
    if snap["status"] in {"queued", "running"}:
        return JSONResponse(status_code=202, content=snap)
    if snap["status"] == "error":
        raise HTTPException(status_code=500, detail=snap["detail"])
    return job.result


@app.get("/")
async def root():
    """API info"""
    return {
        "service": "WhisperX Batch Processing (GPU)",
        "version": "1.1.0-gpu",
        "platform": "NVIDIA DGX Spark (Blackwell)",
        "endpoint": "POST /transcribe",
        "progress": "GET /progress/{uid}",
        "result": "GET /result/{uid}",
        "features": [
            "Perfect transcription (Whisper large-v3)",
            "Word-level timestamps (Wav2Vec2 alignment)",
            "Speaker diarization (pyannote.audio) - GPU ACCELERATED",
            "Per-stage progress by job id",
            "Full GPU acceleration (Blackwell SM_121 → SM_90 spoof)"
        ]
    }
