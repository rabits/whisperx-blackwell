# WhisperX on NVIDIA Blackwell (DGX Spark / GB10 / GB200)

Fork of original project: https://github.com/Mekopa/whisperx-blackwell with updates to build and
work properly. It integrates client with srt builder, switches to community diarization and
integrates faster-whisper hallucination suppressor.

## The Problem

Running legacy AI audio workloads (WhisperX, Pyannote) on next-generation NVIDIA Blackwell GPUs (SM_121) currently fails with:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Why?** The NVRTC compiler doesn't recognize `sm_121` (Blackwell) yet, even though:
- PyTorch can see the GPU
- CUDA toolkit supports it
- The hardware is ready

Standard Python monkeypatching fails because Jiterator queries hardware architecture directly from C++, bypassing Python-level patches.

## The Solution

This repository contains the **"Blackwell Bridge Patch"** - a surgical Dockerfile fix that:

1. **Architecture Spoofing:** Forces PyTorch's `get_device_capability()` to return `(9, 0)` (Hopper) instead of `(12, 1)` (Blackwell)
2. **JIT Bypass:** Patches `torchaudio` source code to avoid `.abs()` on complex tensors, which triggers the broken Jiterator path

**Result:** SM_90 (Hopper) code runs natively on SM_121 (Blackwell) due to binary compatibility.

## Performance

| Metric | CPU Fallback | GPU (Patched) | Speedup |
|--------|-------------|---------------|---------|
| **24 min audio** | ~2 hours | **62 seconds** | **~115x** |
| Transcription | GPU ✓ | GPU ✓ | - |
| Alignment | GPU ✓ | GPU ✓ | - |
| Diarization | **CPU only** | **GPU ✓** | 115x |

## Build & run

```bash
# Clone the repo
git clone https://github.com/mekopa/whisperx-blackwell.git
cd whisperx-blackwell

# Build the image
docker build -t whisperx-blackwell:latest .

# Run it
docker run -d \
  --name whisperx-gpu \
  --gpus all \
  --ipc=host \
  -p 8003:8003 \
  whisperx-blackwell:latest
```

Speaker diarization uses the public [pyannote-community/speaker-diarization-community-1](https://huggingface.co/pyannote-community/speaker-diarization-community-1)
pipeline. WhisperX 3.8.5 would otherwise download the gated `pyannote/speaker-diarization-community-1`
repo. A Hugging Face token is not required. Set `WHISPERX_DIARIZE_MODEL` to pick another pipeline,
and pass `HF_TOKEN` only when that pipeline is gated.

The image on Docker Hub is a previous build. The diarization mirror, the client, and hallucination
filtering below are in this tree and take effect after a local `docker build`.

## Usage

### Health Check

```bash
curl http://localhost:8003/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "whisperx-batch-gpu",
  "device": "cuda",
  "diarization_device": "cuda",
  "gpu": "NVIDIA GB10",
  "compute_capability": "SM_90",
  "diarization_model": "pyannote-community/speaker-diarization-community-1"
}
```

### Client

`client.py` uploads one audio file and writes subtitles next to it. For `interview.mp3` the
defaults are `interview.mp3.json` (the service JSON) and `interview.srt`. Each SRT cue keeps the
speaker id:

```
1
00:00:40,503 --> 00:00:43,044
[SPEAKER_01]: Hello folks.
```

```bash
./client.py --svc_url http://localhost:8003/ interview.mp3
./client.py --svc_url http://localhost:8003/ --language ru interview.mp3
```

`POST /transcribe` answers with a job id. The client then polls `GET /progress/{uid}`
and prints each stage as it moves, in 5% steps:

```
job 3f1c0a2e-1b4d-4e7a-9c20-6a8f0e5d2b11
queued
Transcribing              0.0%
Transcribing             40.0%
Transcribing            100.0%
Aligning                 15.0%
Aligning                100.0%
Identifying speakers     50.0%
Identifying speakers    100.0%
Assigning speakers      100.0%
```

`--json` and `--srt` replace those paths. An empty path skips that file (`--json ''` keeps only the
SRT). Other flags match the form fields: `--language`, `--num-speakers`, `--min-speakers`,
`--max-speakers`, and `--timeout` (seconds to wait for the job after the upload; omitted means
wait until the service finishes).

By default the client also sends [`whisper-hallucinations-ru.lst`](whisper-hallucinations-ru.lst).
The service drops a segment before alignment when the whole cue matches a line in that list.
Matching ignores case and punctuation, so phrase is removed and never reaches the aligner. A longer
sentence that merely contains one of those words is kept. `--hallucinations other.lst` uses another
file. `--hallucinations ''` turns the filter off.

The JSON response includes `dropped_hallucinations` with the removed cue texts. If the running
image is older and ignores the form field, the client still strips the same phrases from the JSON
and SRT it writes.

### Transcribe with curl

```bash
curl -s -X POST "http://localhost:8003/transcribe" \
  -F "file=@your_audio.mp3" \
  -F "language=ru" \
  -F "hallucinations=$(cat whisper-hallucinations-ru.lst)"
# {"uid":"...","status":"queued","progress":"/progress/...","result":"/result/..."}

curl -s "http://localhost:8003/progress/UID"
curl -s "http://localhost:8003/result/UID" -o transcription.json
```

`GET /progress/{uid}` reports `queued`, `running`, `done`, or `error`, plus a percent for
`transcribe`, `align`, `diarize`, and `assign`. `GET /result/{uid}` is the transcript once
`status` is `done`, and `202` with the same progress object while the job is still running.

`wait=true` blocks the POST until the transcript is ready and returns that JSON directly:

```bash
curl -X POST "http://localhost:8003/transcribe" \
  -F "file=@your_audio.mp3" \
  -F "language=ru" \
  -F "wait=true" \
  -o transcription.json
```

The transcript includes:
- Word-level timestamps
- Speaker labels (`SPEAKER_00`, `SPEAKER_01`, ...)
- Language detection
- `dropped_hallucinations` when a phrase list was sent
- `uid` of the job

## Technical Details

### The Patches

#### 1. PyTorch Capability Spoof (`Dockerfile` lines 88-99)

```python
# Forces get_device_capability() to return (9, 0) for SM_121
def get_device_capability(device=None):
    major, minor = _original_get_device_capability(device)
    if major == 12 and minor == 1:
        return (9, 0)  # Pretend to be Hopper H100
    return (major, minor)
```

#### 2. Torchaudio Jiterator Bypass (`Dockerfile` lines 113-118)

```python
# OLD (crashes on SM_121):
spectrum = torch.fft.rfft(strided_input).abs()

# NEW (works):
fft_result = torch.fft.rfft(strided_input)
spectrum = torch.sqrt(fft_result.real**2 + fft_result.imag**2)
```

### Why This Works

1. **Binary Compatibility:** NVIDIA designed Blackwell to execute Hopper (SM_90) code natively
2. **JIT Avoidance:** Computing `.abs()` manually uses standard CUDA kernels instead of runtime-compiled jiterator kernels
3. **No Performance Loss:** The manual computation is mathematically identical and equally fast

### Tested Hardware

- ✅ NVIDIA DGX Spark (ARM64, Blackwell GB10)
- ✅ Should work on GB200, GB202, GB203 (untested)
- ✅ Should work on any SM_121 Blackwell GPU

### Tested Software

- PyTorch 2.6.0 (NVIDIA container 25.01)
- WhisperX 3.8.5
- Pyannote.audio 4.0.4
- transformers 4.48.x (kept below 4.50; 5.x needs `huggingface-hub` 1.x, which WhisperX 3.8.5 cannot use)
- scipy 1.15.1 and scikit-learn 1.6.1
- CUDA 13.0
- Python 3.12

`torchcodec` is not installed. After pip, the image puts NVIDIA's CUDA torch and NumPy 1.x back,
then selects the aarch64 OpenBLAS/LAPACK builds so diarization can link on DGX Spark.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  WhisperX Pipeline (GPU-Accelerated)                      │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Step 1: Whisper large-v3       → GPU (Blackwell/Hopper)  │
│  Step 2: Drop hallucination cues, whole segment           │
│  Step 3: Wav2Vec2 alignment     → GPU (Blackwell/Hopper)  │
│  Step 4: Pyannote community-1   → GPU (public mirror)     │
│                                                           │
│  Patches Applied:                                         │
│  - SM_121 → SM_90 capability spoof                        │
│  - Torchaudio jiterator bypass                            │
│  - OpenBLAS / LAPACK on aarch64                           │
└───────────────────────────────────────────────────────────┘
```

## Known Limitations

1. **Temporary Fix:** This will become obsolete when NVIDIA updates NVRTC to recognize SM_121
2. **Binary Compatibility:** Relies on Blackwell executing Hopper code (safe, but not optimized)
3. **Torchaudio Version:** The line numbers in the patch are for `torchaudio==2.6.0` from the NVIDIA container

## When to Use This

✅ **Use this if:**
- You have Blackwell hardware (DGX Spark, GB10, GB200)
- You're getting `nvrtc: error: invalid value for --gpu-architecture`
- You want GPU-accelerated speaker diarization

❌ **Don't use this if:**
- You have Hopper (H100) or older GPUs - use standard WhisperX
- You're on x86_64 architecture - rebuild for your arch
- NVIDIA has officially released SM_121 support (check PyTorch release notes)

## Future Work

This patch will become obsolete when:
- PyTorch updates to recognize SM_121 natively
- Torchaudio stops using jiterator for complex number operations
- NVIDIA releases updated NVRTC compiler

Until then, this is the **only known way** to run GPU speaker diarization on Blackwell.

## Contributing

Found this useful? Here's how to help:

1. ⭐ **Star the repo** if this saved you time
2. 🐛 **Report issues** if you find edge cases
3. 📝 **Share results** from other Blackwell GPUs (GB200, GB202, etc.)
4. 🔧 **Submit PRs** for improvements

## Credits

- **WhisperX:** https://github.com/m-bain/whisperX
- **Pyannote.audio:** https://github.com/pyannote/pyannote-audio
- **Patch Discovery:** Community effort to unlock Blackwell for legacy workloads

## License

MIT License - Free to use, modify, and distribute.

**Disclaimer:** This is a community patch for early-adopter hardware. Use at your own risk. Not
affiliated with NVIDIA or WhisperX maintainers.

---

**Need help?** Open an issue or check the [Discussions](https://github.com/mekopa/whisperx-blackwell/discussions) tab.
