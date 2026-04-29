# 🎬 Agent Opus

> **The free, local, open-source alternative to Opus Clip.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.1-76b900)](https://developer.nvidia.com/cuda-toolkit)

| Feature | **Agent Opus** | Opus Clip |
|---|---|---|
| **Price** | **FREE forever** | $15–$49/month |
| **Privacy** | **100% Local — your video never leaves your PC** | Cloud upload |
| **Watermark** | **None** | Yes (free plan) |
| **Animated captions** | ✅ Yellow word-highlight, burned in | ✅ |
| **Face / person tracking** | ✅ YOLOv8 + smooth scipy interpolation | ✅ |
| **Virality scoring** | ✅ Local heuristic + optional Llama 3 | ✅ AI |
| **Custom LLM** | ✅ Any Ollama model | ❌ |
| **GPU acceleration** | ✅ CUDA (RTX 3090 tested) | ❌ cloud |
| **Aspect ratios** | ✅ 9:16 · 1:1 · 16:9 | ✅ |
| **Open source** | ✅ MIT License | ❌ |
| **Offline** | ✅ 100% | ❌ |
| **Max video length** | Unlimited | Limited on free tier |

---

## ✨ Features

- **Auto-identify viral moments** — Llama 3 (local) + heuristic fallback
- **Smooth face tracking** — YOLOv8 person detection + scipy interpolation keeps the speaker centred
- **Animated word-level captions** — yellow highlight follows each word as it's spoken
- **GPU-accelerated transcription** — Faster-Whisper on CUDA
- **Download any platform** — YouTube, TikTok, Twitter via yt-dlp
- **Multiple aspect ratios** — 9:16 (Shorts/TikTok), 1:1 (Instagram), 16:9 (YouTube)
- **Batch output** — up to 10 clips per run, ranked by virality score
- **No watermarks. No subscription. No cloud.**

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/agentopus/agent-opus.git
cd agent-opus
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** FFmpeg is pre-configured. Edit the `FFMPEG_DIR` constant at the top of `clipping_tool/clipper.py` if you move the folder.

### 3. (Optional but recommended) Install Ollama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3
```

Without Ollama, Agent Opus automatically falls back to a local heuristic scorer — no API key, no internet needed.

### 4. Launch the web UI

```bash
python clipping_tool/app.py
```

Open **http://localhost:7860** — paste a YouTube URL or upload a video, choose options, click **Generate Viral Clips**.

---

## 🖥️ CLI Usage

```bash
# YouTube URL → 6 clips in 9:16
python clipping_tool/clipper.py --url "https://youtu.be/XXXXXXXXXXX"

# Local file → 8 clips in 1:1 (Instagram)
python clipping_tool/clipper.py --file podcast.mp4 --clips 8 --aspect 1:1

# Use a lighter Whisper model (faster, less accurate)
python clipping_tool/clipper.py --file interview.mp4 --model medium
```

---

## 🧠 How It Works

```
Input video / URL
      │
      ▼
yt-dlp download (any platform)
      │
      ▼
Faster-Whisper GPU transcription (word-level timestamps)
      │
      ▼
Llama 3 (local) → find best 25–60s segments
      │  (Heuristic fallback if Ollama not running)
      ▼
YOLOv8 person detection → scipy smooth interpolation → face-tracking crop
      │
      ▼
Per-word PIL caption frames pre-rendered as ImageClips
      │
      ▼
MoviePy composite + FFmpeg encode at CRF 18 (broadcast quality)
      │
      ▼
output/[score]_title.mp4  (+ _thumb.jpg)
```

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Linux / macOS | Windows 11 |
| Python | 3.10 | 3.11 |
| GPU | NVIDIA 6 GB VRAM (CPU works, ~10× slower) | RTX 3090 24 GB |
| CUDA | 11.8 | 12.1 |
| RAM | 8 GB | 32 GB |
| Disk | 5 GB (models) | 20 GB |

---

## 📁 Project Structure

```
agent-opus/
├── clipping_tool/
│   ├── clipper.py          # Core engine (download, transcribe, track, render)
│   └── app.py              # Gradio web UI
├── generative_studio/
│   └── studio.py           # CrewAI + Ollama autonomous video generation
├── requirements.txt
├── setup.bat               # Windows one-click setup
└── README.md
```

---

## 🔧 Configuration

At the top of `clipping_tool/clipper.py`:

```python
FFMPEG_DIR = r"path\to\ffmpeg\bin"   # Windows path to FFmpeg
OUTPUT_DIR = r"path\to\output"        # Where clips are saved
```

---

## 🤝 Contributing

PRs welcome! Ideas for next features:

- [ ] Speaker diarisation (multi-person tracking)
- [ ] ComfyUI integration for AI B-roll generation
- [ ] Auto-upload to TikTok / YouTube Shorts
- [ ] Caption styles (neon, subtitle bar, emoji captions)
- [ ] Batch playlist processing

---

## 📜 License

MIT — free forever. Do whatever you want with it.

---

## 🙏 Credits

Built on the shoulders of giants:

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — GPU transcription
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) — person detection
- [MoviePy](https://github.com/Zulko/moviepy) — video editing
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — video download
- [Gradio](https://gradio.app) — web UI
- [Ollama](https://ollama.com) — local LLM
- [CrewAI](https://crewai.com) — multi-agent orchestration
