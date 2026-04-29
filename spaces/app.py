"""
Agent Opus — Hugging Face Spaces version
Cross-platform, cloud-compatible. Runs on CPU (use 'small' Whisper for speed)
or GPU (use 'large-v3' for accuracy).
Full source: https://github.com/atherion005-byte/agent-opus
"""

import os
import re
import json
import zipfile
import tempfile
import traceback
import torch
import numpy as np
import gradio as gr
from faster_whisper import WhisperModel
import yt_dlp
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# ── Paths (cloud-safe) ────────────────────────────────────────────────────────
OUTPUT_DIR = tempfile.mkdtemp(prefix="agent_opus_")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Virality word lists ───────────────────────────────────────────────────────
_HOOK_WORDS    = {"secret","never","always","stop","wait","truth","lie","fake","real",
                  "hidden","warning","mistake","shocking","unbelievable","incredible",
                  "exposed","leaked","banned","controversial","dangerous","illegal"}
_EMOTION_WORDS = {"amazing","love","hate","fear","angry","sad","happy","excited",
                  "terrified","surprised","devastated","overwhelmed","frustrated",
                  "passionate","inspired","destroyed","heartbroken"}
_VALUE_WORDS   = {"free","save","earn","win","lose","rich","poor","easy","hard",
                  "fast","slow","better","best","worst","cheap","expensive","profit"}
_POWER_WORDS   = _HOOK_WORDS | _EMOTION_WORDS | _VALUE_WORDS

# ── Load models once at startup ───────────────────────────────────────────────
WHISPER_SIZE = os.environ.get("WHISPER_MODEL", "small")   # override via HF Space secret
print(f"[Agent Opus] Loading Whisper '{WHISPER_SIZE}' on {DEVICE}…")
_whisper = WhisperModel(
    WHISPER_SIZE, device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8",
)

print("[Agent Opus] Loading YOLOv8…")
_yolo = YOLO("yolov8n.pt")   # auto-downloads on first run

def _find_font():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "C:\\Windows\\Fonts\\arialbd.ttf",
              "C:\\Windows\\Fonts\\arial.ttf"]:
        if os.path.exists(p):
            return p
    return None

_FONT_PATH = _find_font()

def _make_font(size=52):
    try:
        if _FONT_PATH:
            return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        pass
    return ImageFont.load_default()

# ── Virality scorer ───────────────────────────────────────────────────────────
def _virality_score(segments, t_start, t_end) -> int:
    score    = 50
    words_in = [w for s in segments for w in s.get("words", [])
                if t_start <= w["start"] <= t_end]
    if not words_in:
        return score
    dur  = max(t_end - t_start, 1)
    wps  = len(words_in) / dur
    score += min(int((wps - 1.5) * 6), 12)
    text  = " ".join(w["word"].lower() for w in words_in)
    score += min(sum(1 for pw in _POWER_WORDS if pw in text) * 4, 20)
    if "?" in text:                score += 8
    if "!" in text:                score += 5
    if re.search(r"\b\d+\b", text): score += 4
    conf  = np.mean([w.get("probability", 0.85) for w in words_in])
    if conf > 0.90:                score += 5
    return min(int(score), 99)

# ── Core pipeline ─────────────────────────────────────────────────────────────
def _transcribe(video_path, progress):
    progress(0.25, desc="Transcribing audio with Whisper…")
    seg_iter, _ = _whisper.transcribe(
        video_path, beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
    )
    words, segments = [], []
    for seg in seg_iter:
        sw = []
        if seg.words:
            for w in seg.words:
                e = {"word": w.word.strip(), "start": round(w.start, 3),
                     "end": round(w.end, 3), "probability": round(w.probability, 3)}
                words.append(e); sw.append(e)
        segments.append({"text": seg.text.strip(), "start": seg.start,
                         "end": seg.end, "words": sw})
    return words, segments

def _find_highlights(words, segments, max_clips, progress):
    progress(0.40, desc="Scoring viral moments…")
    if not segments:
        return []
    total_dur = segments[-1]["end"]
    window    = min(55, max(20, total_dur * 0.4))
    step      = max(5.0, window * 0.2)
    scored    = []
    t = 0.0
    while t + window <= total_dur:
        scored.append((t, t + window, _virality_score(segments, t, t + window)))
        t += step
    if not scored:
        scored = [(0, min(total_dur, 60), 50)]
    scored.sort(key=lambda x: x[2], reverse=True)
    selected, out = [], []
    for s, e, sc in scored:
        if not any(not (e <= sl[0] or s >= sl[1]) for sl in selected):
            selected.append((s, e))
            out.append({"start": s, "end": e, "title": f"Viral Moment {len(out)+1}",
                        "reason": "High engagement density", "virality_score": sc})
            if len(out) >= max_clips:
                break
    return out

def _compute_face_track(video):
    dur, t_list, c_list = video.duration, [], []
    for t in np.arange(0, dur, 0.2):
        frame   = video.get_frame(min(float(t), dur - 0.03))
        results = _yolo(frame, classes=[0], verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
            best  = boxes[np.argmax(areas)]
            t_list.append(float(t))
            c_list.append(float((best[0]+best[2])/2))
    if len(c_list) < 2:
        return None
    smoothed = uniform_filter1d(c_list, size=max(1, 10))
    return interpolate.interp1d(t_list, smoothed, kind="linear",
                                bounds_error=False,
                                fill_value=(smoothed[0], smoothed[-1]))

def _render_caption(phrase_words, cur_idx, fw, cap_h=150, fsize=50):
    img  = Image.new("RGBA", (fw, cap_h), (0, 0, 0, 0))
    bg   = Image.new("RGBA", (fw, cap_h), (10, 10, 10, 200))
    img.paste(bg)
    draw = ImageDraw.Draw(img)
    fnt  = _make_font(fsize)
    full = " ".join(w["word"] for w in phrase_words)
    bb   = draw.textbbox((0, 0), full, font=fnt)
    x    = max(20, (fw - (bb[2]-bb[0])) // 2)
    y    = (cap_h - (bb[3]-bb[1])) // 2
    for i, w in enumerate(phrase_words):
        ws    = w["word"] + (" " if i < len(phrase_words)-1 else "")
        color = (255, 220, 0, 255) if i == cur_idx else (255, 255, 255, 230)
        if i == cur_idx:
            draw.text((x+2, y+2), ws, font=fnt, fill=(0, 0, 0, 160))
        draw.text((x, y), ws, font=fnt, fill=color)
        wb = draw.textbbox((0, 0), ws, font=fnt)
        x += wb[2]-wb[0]
    out = Image.new("RGB", (fw, cap_h), (0, 0, 0))
    out.paste(img, mask=img.split()[3])
    return np.array(out)

def _render_clip(video_path, h, words, aspect, out_dir, progress, pct):
    t0, t1 = h["start"], h["end"]
    sc     = h.get("virality_score", 50)
    title  = re.sub(r"[^\w\s-]", "", h.get("title","clip")).strip().replace(" ","_")[:40]
    out    = os.path.join(out_dir, f"[{sc}]_{title}.mp4")

    progress(pct, desc=f'Rendering "{h["title"]}"…')

    video        = VideoFileClip(video_path).subclipped(t0, t1)
    src_w, src_h = video.size

    if aspect == "9:16":
        tgt_w, tgt_h = int(src_h*9/16), src_h
    elif aspect == "1:1":
        side = min(src_w, src_h); tgt_w = tgt_h = side
    else:
        tgt_w, tgt_h = src_w, src_h

    face_fn = _compute_face_track(video) if aspect != "16:9" else None

    def crop(get_frame, t):
        frame = get_frame(t)
        if aspect == "16:9":
            return frame
        cx = float(face_fn(t)) if face_fn is not None else src_w/2
        cx = max(tgt_w/2, min(src_w-tgt_w/2, cx))
        x0 = int(max(0, min(src_w-tgt_w, cx-tgt_w/2)))
        if aspect == "9:16":
            return frame[:, x0:x0+tgt_w]
        cy = src_h//2
        y0 = int(max(0, min(src_h-tgt_h, cy-tgt_h//2)))
        return frame[y0:y0+tgt_h, x0:x0+tgt_w]

    cropped = video.transform(crop)
    fw, fh  = {"9:16":(1080,1920),"1:1":(1080,1080),"16:9":(1920,1080)}.get(aspect,(1080,1920))
    cropped = cropped.resized((fw, fh))

    words_in = [{**w, "start": w["start"]-t0, "end": w["end"]-t0}
                for w in words if t0 <= w["start"] <= t1]
    cap_clips = []
    cap_y     = fh - 150 - 50
    for i, w in enumerate(words_in):
        dur   = max(w["end"]-w["start"], 0.04)
        ph    = words_in[max(0,i-3):min(len(words_in),i+5)]
        frame = _render_caption(ph, i-max(0,i-3), fw)
        cap_clips.append(
            ImageClip(frame, duration=dur)
            .with_start(w["start"])
            .with_position((0, cap_y))
        )

    final = CompositeVideoClip([cropped]+cap_clips).with_audio(video.audio)
    final.write_videofile(out, codec="libx264", audio_codec="aac", fps=30,
                          threads=4, ffmpeg_params=["-crf","20","-preset","fast"],
                          logger=None)
    video.close()
    return out

# ── Gradio pipeline ───────────────────────────────────────────────────────────
def run_pipeline(url, uploaded_file, max_clips, aspect, progress=gr.Progress()):
    try:
        progress(0.05, desc="Starting…")
        out_dir = tempfile.mkdtemp(prefix="agent_opus_run_")

        # Resolve source
        if url and url.strip():
            progress(0.15, desc="Downloading video…")
            video_path = os.path.join(out_dir, "source.mp4")
            ydl_opts = {
                "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "outtmpl": video_path,
                "merge_output_format": "mp4",
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url.strip()])
        elif uploaded_file:
            video_path = uploaded_file
        else:
            return "❌ Please provide a YouTube URL or upload a video.", [], None

        words, segments = _transcribe(video_path, progress)
        highlights      = _find_highlights(words, segments, int(max_clips), progress)

        results = []
        for i, h in enumerate(highlights):
            pct  = 0.55 + (i / max(len(highlights),1)) * 0.38
            path = _render_clip(video_path, h, words, aspect, out_dir, progress, pct)
            results.append({"path": path, "title": h["title"],
                            "score": h["virality_score"], "reason": h["reason"]})

        # ZIP
        zip_path = os.path.join(out_dir, "agent_opus_clips.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                zf.write(r["path"], os.path.basename(r["path"]))

        report = (f"✅ Generated {len(results)} viral clips!\n" + "="*48
                  + "".join(f"\n\n🎬 [{r['score']}/100]  {r['title']}\n   {r['reason']}"
                            for r in results))

        progress(1.0, desc="Done!")
        return report, [r["path"] for r in results], zip_path

    except Exception:
        return f"❌ Error:\n\n{traceback.format_exc()}", [], None


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
footer{display:none!important}
#hero h1{font-size:2.2em;font-weight:800}
"""

with gr.Blocks(title="Agent Opus — Free Viral Clip Generator", css=CSS) as demo:

    gr.Markdown("""
# 🎬 Agent Opus
### Free · Local · Open-Source alternative to Opus Clip
**No subscription · No watermark · Animated captions · Face tracking · 100% private**
> ⚡ Running on **CPU** (free tier). For GPU speed, clone and run locally: [github.com/atherion005-byte/agent-opus](https://github.com/atherion005-byte/agent-opus)
    """)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Source")
            url_in    = gr.Textbox(label="YouTube / TikTok URL",
                                   placeholder="https://www.youtube.com/watch?v=…")
            file_in   = gr.File(label="Or upload a video", file_types=["video"])
            max_clips = gr.Slider(1, 8, value=4, step=1, label="Max clips")
            aspect    = gr.Radio(["9:16","1:1","16:9"], value="9:16", label="Aspect ratio")
            run_btn   = gr.Button("🚀 Generate Viral Clips", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 📤 Results")
            report_out = gr.Textbox(label="Virality Report", lines=12, interactive=False)
            video_out  = gr.Gallery(label="Generated Clips", columns=3,
                                    height=400, object_fit="cover")
            zip_out    = gr.File(label="⬇️ Download All (ZIP)")

    run_btn.click(
        fn=run_pipeline,
        inputs=[url_in, file_in, max_clips, aspect],
        outputs=[report_out, video_out, zip_out],
    )

    gr.Markdown("""
---
### 📊 vs Opus Clip
| | **Agent Opus** | Opus Clip |
|---|---|---|
| Price | **FREE** | $15–$49/mo |
| Privacy | **100% local** | Cloud upload |
| Watermark | **None** | Yes |
| Open source | **MIT** | ❌ |
| GPU | **Your machine** | Their cloud |

[⭐ Star on GitHub](https://github.com/atherion005-byte/agent-opus) · MIT License
    """)

if __name__ == "__main__":
    demo.launch()
