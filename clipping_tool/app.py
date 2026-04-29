"""
Agent Opus — Gradio Web UI  v2.0
Free, local, open-source alternative to Opus Clip.
"""

import gradio as gr
import os
import zipfile
import traceback
from clipper import AgentOpusClipper, OUTPUT_DIR, PLATFORM_PRESETS

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1300px !important; }
footer { display: none !important; }
#hero h1 { font-size: 2.5em; font-weight: 900; margin-bottom: 0; }
#hero p  { font-size: 1.1em; opacity: 0.75; }
.score-badge { font-size: 1.5em; font-weight: 800; color: #ff4757; }
.keyword-chip {
    display: inline-block; background: #ff4757; color: white;
    border-radius: 12px; padding: 2px 10px; margin: 2px;
    font-size: 0.85em; font-weight: 700;
}
"""

# ── Clipper cache — one instance per Whisper model size ──────────────────────
_clipper_cache: dict = {}

def _get_clipper(whisper_size: str, callback) -> AgentOpusClipper:
    if whisper_size not in _clipper_cache:
        _clipper_cache[whisper_size] = AgentOpusClipper(
            whisper_model_size=whisper_size,
            progress_callback=callback,
        )
    inst = _clipper_cache[whisper_size]
    inst._cb = callback
    return inst


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(url, uploaded_file, max_clips, aspect_ratio,
                 whisper_size, ollama_model, caption_style,
                 enable_zoom, min_dur, max_dur,
                 progress=gr.Progress()):
    logs = []

    def update(msg, pct):
        progress(pct / 100, desc=msg)
        logs.append(f"[{pct:3d}%] {msg}")
        print(f"[{pct:3d}%] {msg}")

    try:
        clipper = _get_clipper(whisper_size, update)

        if url and url.strip():
            # Handle multiple URLs (batch mode — newline separated)
            urls = [u.strip() for u in url.strip().splitlines() if u.strip()]
            if len(urls) > 1:
                all_results = []
                for i, u in enumerate(urls[:5]):  # cap at 5 URLs
                    update(f"Processing URL {i+1}/{len(urls)}: {u[:60]}…", 10)
                    r = clipper.run(
                        u, is_url=True,
                        max_clips=max(1, int(max_clips) // len(urls)),
                        aspect=aspect_ratio,
                        ollama_model=ollama_model if ollama_model != "none" else "none",
                        min_dur=int(min_dur), max_dur=int(max_dur),
                        caption_style=caption_style,
                        enable_zoom=enable_zoom,
                    )
                    all_results.extend(r)
                results = all_results
            else:
                source = urls[0]; is_url = True
                results = clipper.run(
                    source, is_url=is_url,
                    max_clips=int(max_clips),
                    aspect=aspect_ratio,
                    ollama_model=ollama_model if ollama_model != "none" else "none",
                    min_dur=int(min_dur), max_dur=int(max_dur),
                    caption_style=caption_style,
                    enable_zoom=enable_zoom,
                )
        elif uploaded_file:
            results = clipper.run(
                uploaded_file, is_url=False,
                max_clips=int(max_clips),
                aspect=aspect_ratio,
                ollama_model=ollama_model if ollama_model != "none" else "none",
                min_dur=int(min_dur), max_dur=int(max_dur),
                caption_style=caption_style,
                enable_zoom=enable_zoom,
            )
        else:
            return ("❌ Please provide a YouTube URL or upload a video.",
                    [], None, "", None, "")

        # ZIP all clips
        zip_path = os.path.join(OUTPUT_DIR, "agent_opus_clips.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                if os.path.exists(r["path"]):
                    zf.write(r["path"], os.path.basename(r["path"]))

        # Build rich report
        lines = [f"✅ Generated {len(results)} viral clips!\n", "=" * 52]
        for r in results:
            kw_str = "  •  ".join(r.get("keywords_hit", [])[:8]) or "—"
            hook_t = r.get("hook_moment", 0.0)
            lines.append(
                f"\n🎬  [{r['virality_score']}/100]  {r['title']}\n"
                f"    ⏱  {r['start']:.1f}s – {r['end']:.1f}s  "
                f"|  🪝 Hook at {hook_t:.1f}s into clip\n"
                f"    💡 {r['reason']}\n"
                f"    🔑 Keywords: {kw_str}\n"
                + (f"    🎣 {r['hook']}\n" if r.get("hook") else "")
            )

        summary      = "\n".join(lines)
        thumbs       = [r["thumb"] for r in results if os.path.exists(r.get("thumb", ""))]
        log_text     = "\n".join(logs)

        # Top clip for preview
        top_clip = results[0]["path"] if results else None

        # Read transcript if available
        transcript_path = os.path.join(OUTPUT_DIR, "transcript.json")
        transcript_text = ""
        if os.path.exists(transcript_path):
            try:
                import json
                with open(transcript_path, encoding="utf-8") as f:
                    tj = json.load(f)
                segs = tj.get("segments", [])
                transcript_text = "\n".join(
                    f"[{s['start']:.1f}s]  {s['text']}" for s in segs
                )
            except Exception:
                transcript_text = "(error reading transcript)"

        return summary, thumbs, zip_path, log_text, top_clip, transcript_text

    except Exception:
        err = traceback.format_exc()
        return (f"❌ Error:\n\n{err}", [], None, "\n".join(logs), None, "")


# ── Platform preset handler ───────────────────────────────────────────────────
def apply_platform_preset(preset):
    presets = {
        "TikTok":           ("9:16", 6,  15, 60),
        "YouTube Shorts":   ("9:16", 6,  15, 60),
        "Instagram Reels":  ("9:16", 8,  15, 90),
        "LinkedIn":         ("16:9", 8,  30, 600),
        "Custom":           ("9:16", 6,  25, 60),
    }
    aspect, clips, mn, mx = presets.get(preset, ("9:16", 6, 25, 60))
    return (
        gr.update(value=aspect),
        gr.update(value=clips),
        gr.update(value=mn),
        gr.update(value=mx),
    )


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Agent Opus — Free Opus Clip Alternative") as demo:

    with gr.Column(elem_id="hero"):
        gr.Markdown("""
# 🎬 Agent Opus  v2.0
### The Free, Local, Open-Source Opus Clip Killer
**Dynamic zoom · 3 caption styles · Hook detection · Platform presets · Batch URLs · 100% private · No subscription · GPU-accelerated**
        """)

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1: Clip Generator ──────────────────────────────────────────
        with gr.Tab("✂️ Clip Generator"):
            with gr.Row(equal_height=False):

                # ── Left: inputs ──────────────────────────────────────────
                with gr.Column(scale=1, min_width=360):
                    gr.Markdown("### 📥 Source")
                    video_url = gr.Textbox(
                        label="YouTube / TikTok / Twitter URL  (one per line for batch)",
                        placeholder="https://youtu.be/…\nhttps://youtu.be/… (optional 2nd URL)",
                        lines=2,
                    )
                    gr.Markdown("**— or —**")
                    upload_file = gr.File(
                        label="Upload Local Video",
                        file_types=[".mp4", ".mov", ".avi", ".mkv",
                                    ".webm", ".m4v", ".ts"],
                    )

                    gr.Markdown("### 🎯 Platform Preset")
                    platform_preset = gr.Dropdown(
                        choices=["Custom", "TikTok", "YouTube Shorts",
                                 "Instagram Reels", "LinkedIn"],
                        value="Custom",
                        label="Auto-configure for platform",
                    )

                    gr.Markdown("### ⚙️ Options")
                    with gr.Row():
                        max_clips_slider = gr.Slider(
                            minimum=1, maximum=12, value=6, step=1,
                            label="Max Clips",
                        )
                        aspect_ratio = gr.Radio(
                            choices=["9:16", "1:1", "16:9"],
                            value="9:16",
                            label="Aspect Ratio",
                        )

                    with gr.Row():
                        min_dur_slider = gr.Slider(
                            minimum=5, maximum=60, value=25, step=5,
                            label="Min clip length (s)",
                        )
                        max_dur_slider = gr.Slider(
                            minimum=15, maximum=600, value=60, step=5,
                            label="Max clip length (s)",
                        )

                    # Wire platform preset → auto-set options
                    platform_preset.change(
                        fn=apply_platform_preset,
                        inputs=[platform_preset],
                        outputs=[aspect_ratio, max_clips_slider,
                                 min_dur_slider, max_dur_slider],
                    )

                    gr.Markdown("### 🎨 Caption Style")
                    caption_style = gr.Radio(
                        choices=["karaoke", "bold_box", "minimal"],
                        value="karaoke",
                        label="Caption Style",
                        info="karaoke=yellow highlight  |  bold_box=TikTok pills  |  minimal=clean outline",
                    )

                    enable_zoom = gr.Checkbox(
                        value=True,
                        label="🔭 Dynamic Zoom on power moments",
                        info="Opus Clip's signature effect — free here",
                    )

                    with gr.Accordion("🔬 Advanced", open=False):
                        whisper_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium",
                                     "large-v2", "large-v3"],
                            value="large-v3",
                            label="Whisper Model",
                        )
                        ollama_model = gr.Dropdown(
                            choices=["llama3", "llama3.1", "mistral",
                                     "gemma2", "phi3", "none"],
                            value="llama3",
                            label="Local LLM (Ollama) — 'none' = heuristic scorer",
                        )

                    run_btn = gr.Button(
                        "🚀 Generate Viral Clips", variant="primary", size="lg"
                    )

                # ── Right: outputs ────────────────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("### 📤 Results")
                    status_box = gr.Textbox(
                        label="Virality Report (score / keywords / hook moment)",
                        lines=16,
                        interactive=False,
                    )
                    thumb_gallery = gr.Gallery(
                        label="Clip Thumbnails",
                        columns=3, height=320,
                        object_fit="cover",
                    )

                    gr.Markdown("### 🎥 Top Clip Preview")
                    top_clip_preview = gr.Video(
                        label="Preview — top-scored clip",
                        height=360,
                    )

                    download_zip = gr.File(
                        label="⬇️ Download All Clips (ZIP)",
                    )

                    log_visible = gr.State(False)
                    log_box = gr.Textbox(
                        label="Processing Log",
                        lines=6,
                        interactive=False,
                        visible=False,
                    )
                    show_log_btn = gr.Button("Show/Hide Log", size="sm")
                    show_log_btn.click(
                        fn=lambda v: (gr.update(visible=not v), not v),
                        inputs=[log_visible],
                        outputs=[log_box, log_visible],
                    )

            run_btn.click(
                fn=run_pipeline,
                inputs=[video_url, upload_file, max_clips_slider,
                        aspect_ratio, whisper_size, ollama_model,
                        caption_style, enable_zoom,
                        min_dur_slider, max_dur_slider],
                outputs=[status_box, thumb_gallery, download_zip,
                         log_box, top_clip_preview, gr.Textbox(visible=False)],
            )

        # ── Tab 2: Transcript Viewer ───────────────────────────────────────
        with gr.Tab("📝 Transcript"):
            gr.Markdown("### Full Transcript\nRun the Clip Generator first — the transcript appears here automatically.")
            transcript_box = gr.Textbox(
                label="Transcript (timestamps + text)",
                lines=30,
                interactive=False,
                placeholder="Transcript will appear here after processing…",
            )
            # Wire transcript output from run_pipeline
            run_btn.click(
                fn=lambda *args: run_pipeline(*args)[5],
                inputs=[video_url, upload_file, max_clips_slider,
                        aspect_ratio, whisper_size, ollama_model,
                        caption_style, enable_zoom,
                        min_dur_slider, max_dur_slider],
                outputs=[transcript_box],
            )

        # ── Tab 3: Compare vs Opus Clip ───────────────────────────────────
        with gr.Tab("📊 vs Opus Clip"):
            gr.Markdown("""
## Agent Opus v2 vs. Opus Clip — Feature Comparison

| Feature | **Agent Opus v2** ✅ | Opus Clip |
|---|---|---|
| **Price** | **FREE forever** | $15–$49/month |
| **Privacy** | **100% Local — data never leaves your PC** | Cloud uploads |
| **Watermark** | **None** | Yes (free plan) |
| **Dynamic zoom on power moments** | **✅ Built-in FREE** | ✅ Paid feature |
| **Caption styles** | **✅ Karaoke · Bold Box · Minimal** | ✅ Paid styles |
| **Hook moment detection** | **✅ Per-clip, with timestamp** | ✅ |
| **Keyword virality breakdown** | **✅ Shows exactly why a clip scores high** | ❌ black box |
| **Animated captions** | **✅ GPU-rendered** | ✅ |
| **Face / person tracking** | **✅ YOLOv8 + smooth interpolation** | ✅ |
| **Platform presets** | **✅ TikTok / Shorts / Reels / LinkedIn** | ✅ |
| **Batch URL processing** | **✅ Up to 5 URLs at once** | Limited (paid) |
| **Transcript viewer** | **✅ Timestamped, searchable** | Limited |
| **Custom clip duration** | **✅ 5s–600s configurable** | Limited |
| **Custom LLM** | **✅ Any Ollama model** | ❌ locked |
| **GPU acceleration** | **✅ CUDA (RTX 3090 tested)** | ❌ cloud |
| **Open source** | **✅ MIT License** | ❌ |
| **Offline** | **✅ 100%** | ❌ |
| **Any video genre** | **✅ Podcasts, vlogs, gaming, sports** | Mostly podcasts |

---

## Features Opus Clip Has (Paid) That We're Building Next

- 🔜 **AI B-Roll insertion** — auto-search and splice relevant stock footage
- 🔜 **AI Voice-over** — add narration to silent segments  
- 🔜 **Brand templates** — intro/outro frames with logo
- 🔜 **Direct social publish** — TikTok / YouTube API post
- 🔜 **Audio enhancement** — noise removal, level normalization
            """)

        # ── Tab 4: Quick Start ─────────────────────────────────────────────
        with gr.Tab("🚀 Quick Start"):
            gr.Markdown("""
## Install & Run

```bash
# 1. Clone
git clone https://github.com/atherion005-byte/agent-opus.git
cd agent-opus

# 2. Install
pip install -r requirements.txt

# 3. (Optional) Ollama for AI analysis
# https://ollama.com → ollama pull llama3

# 4. Launch
python clipping_tool/app.py
```

## CLI v2 — New Options

```bash
# TikTok preset (9:16, 15–60s) with bold-box captions + dynamic zoom
python clipping_tool/clipper.py --url "https://youtu.be/XXX" \\
    --platform tiktok --caption-style bold_box

# Instagram Reels, heuristic only (no Ollama needed)
python clipping_tool/clipper.py --file podcast.mp4 \\
    --platform instagram_reels --llm none --clips 8

# LinkedIn (16:9, up to 10min clips) minimal captions, no zoom
python clipping_tool/clipper.py --file webinar.mp4 \\
    --platform linkedin --caption-style minimal --no-zoom

# Square format, karaoke captions (default)
python clipping_tool/clipper.py --file interview.mp4 --aspect 1:1 --clips 6
```

## Caption Styles

| Style | Description | Best For |
|---|---|---|
| `karaoke` | Yellow word highlight on dark bg | Podcasts, interviews |
| `bold_box` | Each word in its own TikTok pill | High-energy content, Gen Z audience |
| `minimal` | Clean white text with outline only | Cinematic, B-roll heavy |

## GitHub

⭐ **https://github.com/atherion005-byte/agent-opus**  
MIT License — free forever.
            """)


if __name__ == "__main__":
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "localhost"
    print("\n" + "=" * 62)
    print("  🎬  Agent Opus v2.0 — Free Opus Clip Killer")
    print("=" * 62)
    print(f"  Local:   http://localhost:7860")
    print(f"  Network: http://{local_ip}:7860")
    print("  Public:  generating share link…")
    print("=" * 62 + "\n")
    demo.launch(
        server_port=7860,
        share=True,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
        css=CSS,
    )
