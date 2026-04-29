"""
Agent Opus — Gradio Web UI
Free, local, open-source alternative to Opus Clip.
"""

import gradio as gr
import os
import zipfile
import traceback
from clipper import AgentOpusClipper, OUTPUT_DIR

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1280px !important; }
footer { display: none !important; }
.score-badge { font-size: 1.4em; font-weight: 700; color: #ff4757; }
#hero h1 { font-size: 2.4em; font-weight: 800; margin-bottom: 0; }
#hero p  { font-size: 1.1em; opacity: 0.8; }
"""

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(url, uploaded_file, max_clips, aspect_ratio,
                 whisper_size, ollama_model, progress=gr.Progress()):
    logs = []

    def update(msg, pct):
        progress(pct / 100, desc=msg)
        logs.append(f"[{pct:3d}%] {msg}")
        print(f"[{pct:3d}%] {msg}")

    try:
        clipper = AgentOpusClipper(
            whisper_model_size=whisper_size,
            progress_callback=update,
        )

        # Resolve source
        if url and url.strip():
            source = url.strip(); is_url = True
        elif uploaded_file:
            source = uploaded_file; is_url = False
        else:
            return "❌ Please provide a YouTube URL or upload a video.", [], None, ""

        results = clipper.run(
            source, is_url=is_url,
            max_clips=int(max_clips),
            aspect=aspect_ratio,
            ollama_model=ollama_model,
        )

        # ZIP all clips
        zip_path = os.path.join(OUTPUT_DIR, "agent_opus_clips.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                zf.write(r["path"], os.path.basename(r["path"]))

        # Summary report
        lines = [f"✅ Generated {len(results)} viral clips!\n",
                 "=" * 50]
        for r in results:
            lines.append(
                f"\n🎬  [{r['virality_score']}/100]  {r['title']}\n"
                f"    ⏱  {r['start']:.1f}s – {r['end']:.1f}s\n"
                f"    💡 {r['reason']}\n"
                + (f"    🎣 Hook: {r['hook']}\n" if r.get("hook") else "")
            )

        summary  = "\n".join(lines)
        thumbs   = [r["thumb"] for r in results if os.path.exists(r.get("thumb",""))]
        log_text = "\n".join(logs)
        return summary, thumbs, zip_path, log_text

    except Exception:
        err = traceback.format_exc()
        return f"❌ Error:\n\n{err}", [], None, "\n".join(logs)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Agent Opus — Free Opus Clip Alternative") as demo:

    # ── Hero ──
    with gr.Column(elem_id="hero"):
        gr.Markdown("""
# 🎬 Agent Opus
### The Free, Local, Open-Source Alternative to Opus Clip
**100% private · No subscription · GPU-accelerated · Animated captions · No watermarks · Runs on YOUR machine**
        """)

    gr.Markdown("---")

    # ── Main tabs ──
    with gr.Tabs():

        # ── Tab 1: Clip Generator ──────────────────────────────────────────
        with gr.Tab("✂️ Clip Generator"):
            with gr.Row(equal_height=False):

                # Left column — inputs
                with gr.Column(scale=1, min_width=340):
                    gr.Markdown("### 📥 Source")
                    video_url = gr.Textbox(
                        label="YouTube / TikTok / Twitter URL",
                        placeholder="https://www.youtube.com/watch?v=…",
                    )
                    gr.Markdown("**— or —**")
                    upload_file = gr.File(
                        label="Upload Local Video",
                        file_types=["video"],
                    )

                    gr.Markdown("### ⚙️ Options")
                    with gr.Row():
                        max_clips_slider = gr.Slider(
                            minimum=1, maximum=10, value=6, step=1,
                            label="Max Clips",
                        )
                        aspect_ratio = gr.Radio(
                            choices=["9:16", "1:1", "16:9"],
                            value="9:16",
                            label="Aspect Ratio",
                        )

                    with gr.Accordion("🔬 Advanced", open=False):
                        whisper_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium",
                                     "large-v2", "large-v3"],
                            value="large-v3",
                            label="Whisper Model (larger = more accurate, slower)",
                        )
                        ollama_model = gr.Dropdown(
                            choices=["llama3", "llama3.1", "mistral",
                                     "gemma2", "phi3", "none"],
                            value="llama3",
                            label="Local LLM (Ollama) — 'none' uses heuristic scorer",
                        )

                    run_btn = gr.Button(
                        "🚀 Generate Viral Clips", variant="primary", size="lg"
                    )

                # Right column — outputs
                with gr.Column(scale=2):
                    gr.Markdown("### 📤 Results")
                    status_box = gr.Textbox(
                        label="Virality Report",
                        lines=14,
                        interactive=False,
                    )
                    thumb_gallery = gr.Gallery(
                        label="Clip Thumbnails",
                        columns=3, height=360,
                        object_fit="cover",
                    )
                    download_zip = gr.File(
                        label="⬇️ Download All Clips (ZIP)",
                    )
                    log_box = gr.Textbox(
                        label="Processing Log",
                        lines=6,
                        interactive=False,
                        visible=False,
                    )
                    show_log_btn = gr.Button("Show/Hide Log", size="sm")
                    show_log_btn.click(
                        fn=lambda v: gr.update(visible=not v),
                        inputs=[log_box],
                        outputs=[log_box],
                    )

            run_btn.click(
                fn=run_pipeline,
                inputs=[video_url, upload_file, max_clips_slider,
                        aspect_ratio, whisper_size, ollama_model],
                outputs=[status_box, thumb_gallery, download_zip, log_box],
            )

        # ── Tab 2: Compare vs Opus Clip ────────────────────────────────────
        with gr.Tab("📊 vs Opus Clip"):
            gr.Markdown("""
## Agent Opus vs. Opus Clip — Feature Comparison

| Feature | **Agent Opus** ✅ | Opus Clip |
|---|---|---|
| **Price** | **FREE forever** | $15–$49/month |
| **Privacy** | **100% Local — data never leaves your PC** | Cloud — videos uploaded |
| **Watermark** | **None** | Yes (on free plan) |
| **Animated captions** | **✅ Yellow word-highlight** | ✅ |
| **Face / person tracking** | **✅ YOLOv8 + smooth interpolation** | ✅ |
| **Virality scoring** | **✅ Local heuristic + LLM** | ✅ AI |
| **Custom LLM** | **✅ Any Ollama model** | ❌ locked |
| **GPU acceleration** | **✅ CUDA (RTX 3090 tested)** | ❌ cloud |
| **Aspect ratios** | **✅ 9:16, 1:1, 16:9** | ✅ |
| **Open source** | **✅ MIT License** | ❌ |
| **Batch / CLI** | **✅ python clipper.py --url …** | Limited |
| **Offline** | **✅ 100%** | ❌ |
| **Max video length** | **Unlimited** | Limited on free tier |

## How It Works

1. **Download** — yt-dlp fetches any public video at up to 1080p  
2. **Transcribe** — Faster-Whisper runs entirely on your GPU (word-level timestamps)  
3. **Analyse** — Llama 3 (local Ollama) scores viral potential + fallback heuristic  
4. **Track** — YOLOv8 detects the speaker; scipy smooths the pan  
5. **Captions** — Per-word animated yellow highlights, burned into the video  
6. **Render** — MoviePy + FFmpeg encodes 9:16 clips at CRF 18 (broadcast quality)

## System Requirements

- Windows 10/11, Linux, or macOS  
- Python 3.10+  
- NVIDIA GPU with 8 GB+ VRAM (CPU fallback available, slower)  
- [Ollama](https://ollama.com) + `ollama pull llama3` *(optional but recommended)*  
- FFmpeg (bundled on Windows)
            """)

        # ── Tab 3: Quick Start ─────────────────────────────────────────────
        with gr.Tab("🚀 Quick Start"):
            gr.Markdown("""
## Install & Run

```bash
# 1. Clone
git clone https://github.com/agentopus/agent-opus.git
cd agent-opus

# 2. Install
pip install -r requirements.txt

# 3. (Optional) Install Ollama for AI analysis
# https://ollama.com  →  ollama pull llama3

# 4. Launch
python clipping_tool/app.py
```

## CLI Usage

```bash
# From a YouTube URL
python clipping_tool/clipper.py --url "https://youtu.be/XXXXXXXXXXX" --clips 6

# From a local file
python clipping_tool/clipper.py --file my_podcast.mp4 --aspect 9:16

# Square format
python clipping_tool/clipper.py --file interview.mp4 --aspect 1:1 --clips 8
```

## GitHub

⭐ Star us: **https://github.com/agentopus/agent-opus**

MIT License — free forever.
            """)

if __name__ == "__main__":
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print("\n" + "="*60)
    print("  🎬  Agent Opus is starting…")
    print("="*60)
    print(f"  Local:   http://localhost:7860")
    print(f"  Network: http://{local_ip}:7860")
    print("  Public:  generating share link… (check below)")
    print("="*60 + "\n")
    demo.launch(
        server_port=7860,
        share=True,           # ← public *.gradio.live URL for anyone on the internet
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
        css=CSS,
    )
