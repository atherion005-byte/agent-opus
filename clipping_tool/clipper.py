"""
Agent Opus — Core Clipping Engine
Beats Opus Clip: 100% local, no subscription, GPU-accelerated,
animated captions, smooth AI face tracking, virality scoring, no watermarks.
"""

import os
import re
import json
import torch
import numpy as np
from faster_whisper import WhisperModel
import yt_dlp
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from scipy.ndimage import uniform_filter1d
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ── Optional Ollama (graceful if not installed) ───────────────────────────────
try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

# ─── FFmpeg ───────────────────────────────────────────────────────────────────
FFMPEG_DIR  = r"D:\AI\AgentOpus\ffmpeg\ffmpeg-8.1-essentials_build\bin"
FFMPEG_BIN  = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE_BIN = os.path.join(FFMPEG_DIR, "ffprobe.exe")

if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("FFMPEG_BINARY",  FFMPEG_BIN)
os.environ.setdefault("FFPROBE_BINARY", FFPROBE_BIN)

# ─── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = r"D:\AI\AgentOpus\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Virality word lists ──────────────────────────────────────────────────────
_HOOK_WORDS    = {"secret","never","always","stop","wait","truth","lie","fake","real",
                  "hidden","warning","mistake","shocking","unbelievable","incredible",
                  "exposed","leaked","banned","controversial","dangerous","illegal"}
_EMOTION_WORDS = {"amazing","love","hate","fear","angry","sad","happy","excited",
                  "terrified","surprised","devastated","overwhelmed","frustrated",
                  "passionate","inspired","destroyed","heartbroken"}
_VALUE_WORDS   = {"free","save","earn","win","lose","rich","poor","easy","hard",
                  "fast","slow","better","best","worst","cheap","expensive","profit"}
_POWER_WORDS   = _HOOK_WORDS | _EMOTION_WORDS | _VALUE_WORDS


# ─── Main engine ─────────────────────────────────────────────────────────────
class AgentOpusClipper:
    """
    Full pipeline:
      download → transcribe → AI highlight detection →
      smooth face tracking → animated captions → render.
    """

    def __init__(self, whisper_model_size: str = "large-v3",
                 progress_callback=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cb    = progress_callback or (lambda msg, pct: None)
        self._font  = self._find_font()

        self._cb(f"Initialising on {self.device.upper()}…", 3)

        self._cb("Loading Whisper model…", 5)
        self.whisper = WhisperModel(
            whisper_model_size, device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
        )

        self._cb("Loading YOLOv8 person detector…", 10)
        yolo_pt = r"D:\yolov8n.pt"
        if not os.path.exists(yolo_pt):
            yolo_pt = "yolov8n.pt"
        self.yolo = YOLO(yolo_pt)

    # ── Font helper ───────────────────────────────────────────────────────────
    def _find_font(self):
        for p in [r"C:\Windows\Fonts\arialbd.ttf",
                  r"C:\Windows\Fonts\calibrib.ttf",
                  r"C:\Windows\Fonts\verdanab.ttf",
                  r"C:\Windows\Fonts\arial.ttf"]:
            if os.path.exists(p):
                return p
        return None

    def _make_font(self, size=52):
        try:
            if self._font:
                return ImageFont.truetype(self._font, size)
        except Exception:
            pass
        return ImageFont.load_default()

    # ── Download ──────────────────────────────────────────────────────────────
    def download_video(self, url: str, out_path: str = None) -> str:
        if out_path is None:
            out_path = os.path.join(OUTPUT_DIR, "source_video.mp4")
        self._cb("Downloading video…", 15)
        ydl_opts = {
            "format": ("bestvideo[height<=1080][ext=mp4]"
                       "+bestaudio[ext=m4a]"
                       "/best[height<=1080][ext=mp4]/best"),
            "outtmpl":             out_path,
            "ffmpeg_location":     FFMPEG_DIR,
            "merge_output_format": "mp4",
            "noplaylist":          True,   # never download a whole playlist
            "quiet":               True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return out_path

    # ── Transcribe ────────────────────────────────────────────────────────────
    def transcribe(self, video_path: str):
        self._cb("Transcribing with Whisper (GPU)…", 25)
        seg_iter, _ = self.whisper.transcribe(
            video_path, beam_size=5, word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )

        words, segments = [], []
        for seg in seg_iter:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    entry = {
                        "word":        w.word.strip(),
                        "start":       round(w.start, 3),
                        "end":         round(w.end,   3),
                        "probability": round(w.probability, 3),
                    }
                    words.append(entry)
                    seg_words.append(entry)
            segments.append({
                "text":  seg.text.strip(),
                "start": seg.start,
                "end":   seg.end,
                "words": seg_words,
            })

        with open(os.path.join(OUTPUT_DIR, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump({"words": words, "segments": segments}, f, indent=2)
        return words, segments

    # ── Virality scoring ──────────────────────────────────────────────────────
    def _virality_score(self, segments, t_start, t_end) -> int:
        score = 50
        words_in = [
            w for s in segments for w in s.get("words", [])
            if t_start <= w["start"] <= t_end
        ]
        if not words_in:
            return score

        dur = max(t_end - t_start, 1)
        wps = len(words_in) / dur
        score += min(int((wps - 1.5) * 6), 12)

        text    = " ".join(w["word"].lower() for w in words_in)
        pw_hits = sum(1 for pw in _POWER_WORDS if pw in text)
        score  += min(pw_hits * 4, 20)

        if "?" in text:                  score += 8
        if "!" in text:                  score += 5
        if re.search(r"\b\d+\b", text):  score += 4

        conf = np.mean([w.get("probability", 0.85) for w in words_in])
        if conf > 0.90:  score += 5

        return min(int(score), 99)

    # ── AI highlight analysis ─────────────────────────────────────────────────
    def analyze_highlights(self, words, segments,
                           max_clips=8, min_dur=25, max_dur=60,
                           ollama_model="llama3"):
        self._cb("AI analysing viral potential…", 40)

        # ─ Try Ollama (skipped if not installed or model == "none") ─
        use_ollama = _OLLAMA_AVAILABLE and ollama_model.lower() != "none"
        if use_ollama:
            try:
                ts_text = " ".join(
                    f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}"
                    for s in segments[:80]
                )
                prompt = (
                    f"You are a viral content expert. Find the {max_clips} best segments "
                    f"for YouTube Shorts/TikTok.\n"
                    f"Rules: {min_dur}–{max_dur} seconds each, strong hook, complete thought.\n"
                    f"Return ONLY valid JSON array. Each object: "
                    f'"start"(float),"end"(float),"title"(≤6 words),'
                    f'"reason"(why viral),"hook"(opening line).\n\n'
                    f"Transcript:\n{ts_text[:6000]}"
                )
                resp = _ollama.chat(
                    model=ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3},
                )
                # ollama 0.2+ returns an object; 0.1.x returned a dict — handle both
                try:
                    raw = resp.message.content
                except AttributeError:
                    raw = resp["message"]["content"]

                s_idx = raw.find("["); e_idx = raw.rfind("]") + 1
                if s_idx >= 0 and e_idx > s_idx:
                    candidates = json.loads(raw[s_idx:e_idx])
                    valid = []
                    for h in candidates:
                        s = float(h.get("start", 0))
                        e = float(h.get("end",   0))
                        if min_dur <= (e - s) <= max_dur:
                            h["virality_score"] = self._virality_score(segments, s, e)
                            h.setdefault("title", f"Clip {len(valid)+1}")
                            valid.append(h)
                    if valid:
                        return sorted(valid, key=lambda x: x["virality_score"], reverse=True)
            except Exception as ex:
                print(f"[Ollama] {ex} — heuristic scorer activated.")

        # ─ Heuristic fallback (no LLM needed) ─
        if not segments:
            return []

        total_dur = segments[-1]["end"]
        window = min(max_dur, max(min_dur, total_dur * 0.4))
        step   = max(5.0, window * 0.2)
        scored = []
        t = 0.0
        while t + window <= total_dur:
            scored.append((t, t + window, self._virality_score(segments, t, t + window)))
            t += step

        scored.sort(key=lambda x: x[2], reverse=True)
        selected, highlights = [], []
        for s, e, score in scored:
            # No overlap with already-selected segments
            if all(e <= sel[0] or s >= sel[1] for sel in selected):
                selected.append((s, e))
                idx = len(selected)
                highlights.append({
                    "start": s, "end": e,
                    "title": f"Viral Moment {idx}",
                    "reason": "Heuristic: high engagement density",
                    "hook": "",
                    "virality_score": score,
                })
                if len(selected) >= max_clips:
                    break

        return sorted(highlights, key=lambda x: x["virality_score"], reverse=True)

    # ── Face tracking ─────────────────────────────────────────────────────────
    def _compute_face_track(self, video, sample_fps=5):
        """Returns a closure t→center_x (float), or None if no faces found."""
        dur = video.duration
        times_raw, centers_raw = [], []

        for t in np.arange(0, dur, 1.0 / sample_fps):
            frame   = video.get_frame(min(float(t), dur - 0.03))
            results = self.yolo(frame, classes=[0], verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best  = boxes[np.argmax(areas)]
                times_raw.append(float(t))
                centers_raw.append(float((best[0] + best[2]) / 2))

        if len(centers_raw) < 2:
            return None

        t_arr    = np.array(times_raw,   dtype=float)
        c_arr    = np.array(centers_raw, dtype=float)
        smoothed = uniform_filter1d(c_arr, size=max(1, sample_fps * 2))

        # np.interp replaces the deprecated scipy.interpolate.interp1d
        def _interp(t: float) -> float:
            return float(np.interp(t, t_arr, smoothed))

        return _interp

    # ── Caption rendering ─────────────────────────────────────────────────────
    def _render_caption_image(self, phrase_words, current_idx,
                              frame_w, cap_h=160, font_size=54):
        """Return (H, W, 3) uint8 array for one caption frame."""
        img  = Image.new("RGBA", (frame_w, cap_h), (0, 0, 0, 0))
        bg   = Image.new("RGBA", (frame_w, cap_h), (10, 10, 10, 200))
        img.paste(bg)
        draw = ImageDraw.Draw(img)
        fnt  = self._make_font(font_size)

        # Build token list — trailing space on every word except the last
        tokens = [
            w["word"] + (" " if i < len(phrase_words) - 1 else "")
            for i, w in enumerate(phrase_words)
        ]

        # Centre-align based on the joined full string
        full_text = "".join(tokens)
        bbox      = draw.textbbox((0, 0), full_text, font=fnt)
        total_w   = bbox[2] - bbox[0]
        x = max(24, (frame_w - total_w) // 2)
        y = (cap_h - (bbox[3] - bbox[1])) // 2

        for i, tok in enumerate(tokens):
            is_cur = (i == current_idx)
            color  = (255, 220, 0, 255) if is_cur else (255, 255, 255, 230)
            if is_cur:
                draw.text((x + 2, y + 2), tok, font=fnt, fill=(0, 0, 0, 160))
            draw.text((x, y), tok, font=fnt, fill=color)
            wb = draw.textbbox((0, 0), tok, font=fnt)
            x += wb[2] - wb[0]

        out = Image.new("RGB", (frame_w, cap_h), (0, 0, 0))
        out.paste(img, mask=img.split()[3])
        return np.array(out)

    def _build_caption_clips(self, words_in_clip, final_w, final_h,
                             cap_h=160, clip_dur=None):
        """Pre-render one ImageClip per word (no per-frame PIL overhead)."""
        cap_y = final_h - cap_h - 50
        clips = []

        for i, word in enumerate(words_in_clip):
            w_start = word["start"]
            w_end   = word["end"]
            # Clamp to clip duration so captions never exceed the video length
            if clip_dur is not None:
                w_end = min(w_end, clip_dur)
            dur = max(w_end - w_start, 0.04)

            ph_s    = max(0, i - 3)
            ph_e    = min(len(words_in_clip), i + 5)
            phrase  = words_in_clip[ph_s:ph_e]
            cur_idx = i - ph_s

            frame = self._render_caption_image(phrase, cur_idx, final_w, cap_h)
            clips.append(
                ImageClip(frame, duration=dur)
                .with_start(w_start)
                .with_position((0, cap_y))
            )

        return clips

    # ── Render one clip ───────────────────────────────────────────────────────
    def create_clip(self, video_path, highlight, words,
                    aspect="9:16", output_dir=None, progress_pct=70):
        if output_dir is None:
            output_dir = OUTPUT_DIR

        t_start  = highlight["start"]
        t_end    = highlight["end"]
        score    = highlight.get("virality_score", 0)
        title    = re.sub(r"[^\w\s-]", "", highlight.get("title", "clip")).strip()
        title    = title.replace(" ", "_")[:40]
        out_path = os.path.join(output_dir, f"[{score}]_{title}.mp4")

        self._cb(f'Rendering "{title}" ({t_start:.0f}s–{t_end:.0f}s)…', progress_pct)

        video        = VideoFileClip(video_path).subclipped(t_start, t_end)
        src_w, src_h = video.size
        clip_dur     = video.duration

        # Output resolution lookup
        final_w, final_h = {"9:16": (1080, 1920),
                            "1:1":  (1080, 1080),
                            "16:9": (1920, 1080)}.get(aspect, (1080, 1920))

        # ── Build cropped/resized video ────────────────────────────────────
        if aspect == "16:9":
            # No crop needed — just resize to target resolution
            cropped = video.resized((final_w, final_h))
        else:
            tgt_w = int(src_h * 9 / 16) if aspect == "9:16" else min(src_w, src_h)
            tgt_h = src_h               if aspect == "9:16" else tgt_w

            face_fn = self._compute_face_track(video)

            def crop_frame(get_frame, t):
                frame = get_frame(t)
                cx    = float(face_fn(t)) if face_fn is not None else src_w / 2
                cx    = max(tgt_w / 2, min(src_w - tgt_w / 2, cx))
                x0    = int(max(0, min(src_w - tgt_w, cx - tgt_w / 2)))

                if aspect == "9:16":
                    return frame[:, x0:x0 + tgt_w]
                else:  # 1:1
                    cy = src_h // 2
                    y0 = int(max(0, min(src_h - tgt_h, cy - tgt_h // 2)))
                    return frame[y0:y0 + tgt_h, x0:x0 + tgt_w]

            cropped = video.transform(crop_frame).resized((final_w, final_h))

        # ── Animated captions ─────────────────────────────────────────────
        words_in = [
            {**w, "start": w["start"] - t_start, "end": w["end"] - t_start}
            for w in words if t_start <= w["start"] < t_end
        ]
        cap_clips = self._build_caption_clips(words_in, final_w, final_h,
                                              clip_dur=clip_dur)

        final     = CompositeVideoClip([cropped] + cap_clips)
        has_audio = video.audio is not None
        if has_audio:
            final = final.with_audio(video.audio)

        try:
            final.write_videofile(
                out_path,
                codec="libx264",
                audio_codec="aac" if has_audio else None,
                fps=30,
                threads=8,
                ffmpeg_params=["-crf", "18", "-preset", "fast"],
                logger=None,
            )
        finally:
            # Always release file handles — critical on Windows
            for clip in (final, cropped, video):
                try: clip.close()
                except Exception: pass

        return out_path

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    def generate_thumbnail(self, clip_path: str) -> str:
        thumb = clip_path.replace(".mp4", "_thumb.jpg")
        try:
            clip = VideoFileClip(clip_path)
            t    = clip.duration * 0.08
            Image.fromarray(clip.get_frame(t)).save(thumb, quality=85)
            clip.close()
        except Exception:
            pass
        return thumb

    # ── Full pipeline (public API) ─────────────────────────────────────────────
    def run(self, source: str, is_url=False, max_clips=6,
            aspect="9:16", ollama_model="llama3"):
        """
        Returns list of dicts with keys:
          path, thumb, title, virality_score, reason, hook, start, end
        """
        if is_url:
            video_path = self.download_video(source)
        else:
            video_path = source

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        words, segments = self.transcribe(video_path)
        highlights      = self.analyze_highlights(
            words, segments,
            max_clips=max_clips,
            ollama_model=ollama_model,
        )
        if not highlights:
            if segments:
                total = segments[-1]["end"]
                highlights = [{
                    "start": 0, "end": min(total, 60),
                    "title": "Full Video", "reason": "Short video — full clip",
                    "hook": "", "virality_score": 50,
                }]
            else:
                raise ValueError("No highlights found — is the video too short or silent?")

        results = []
        for i, h in enumerate(highlights):
            pct       = 65 + int((i / max(len(highlights), 1)) * 30)
            clip_path = self.create_clip(video_path, h, words,
                                         aspect=aspect, progress_pct=pct)
            results.append({
                "path":           clip_path,
                "thumb":          self.generate_thumbnail(clip_path),
                "title":          h.get("title", f"Clip {i+1}"),
                "virality_score": h.get("virality_score", 0),
                "reason":         h.get("reason", ""),
                "hook":           h.get("hook", ""),
                "start":          h["start"],
                "end":            h["end"],
            })

        self._cb("✅ All clips ready!", 100)
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Agent Opus CLI")
    p.add_argument("--url",    help="YouTube / TikTok URL")
    p.add_argument("--file",   help="Local video path")
    p.add_argument("--clips",  type=int, default=6)
    p.add_argument("--aspect", default="9:16", choices=["9:16", "1:1", "16:9"])
    p.add_argument("--model",  default="large-v3")
    p.add_argument("--llm",    default="llama3",
                   help="Ollama model name, or 'none' to use heuristic scorer only")
    args = p.parse_args()

    def log(msg, pct): print(f"[{pct:3d}%] {msg}")
    clipper = AgentOpusClipper(whisper_model_size=args.model, progress_callback=log)

    if args.url:
        res = clipper.run(args.url,  is_url=True,  max_clips=args.clips,
                          aspect=args.aspect, ollama_model=args.llm)
    elif args.file:
        res = clipper.run(args.file, is_url=False, max_clips=args.clips,
                          aspect=args.aspect, ollama_model=args.llm)
    else:
        p.print_help(); raise SystemExit(1)

    print("\n=== DONE ===")
    for r in res:
        print(f"  [{r['virality_score']}/100] {r['title']}  →  {r['path']}")
