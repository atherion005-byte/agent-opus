"""
Agent Opus — Core Clipping Engine  v2.0
Free, local, GPU-accelerated alternative to Opus Clip.

v2 additions vs Opus Clip:
  ✅ Dynamic zoom on power moments  (Opus Clip signature, they charge $49/mo)
  ✅ 3 caption styles: karaoke, bold_box, minimal
  ✅ Hook moment detection per clip
  ✅ Keyword breakdown per clip (why it scored high)
  ✅ Platform presets: TikTok, YouTube Shorts, Instagram Reels, LinkedIn
  ✅ Any-genre support (not just talking-head podcasts)
  ✅ 100% local, 100% private, 100% free
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

# ── Optional Ollama ───────────────────────────────────────────────────────────
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

OUTPUT_DIR = r"D:\AI\AgentOpus\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Platform presets ────────────────────────────────────────────────────────
PLATFORM_PRESETS = {
    "tiktok":          {"aspect": "9:16",  "min_dur": 15, "max_dur": 60},
    "youtube_shorts":  {"aspect": "9:16",  "min_dur": 15, "max_dur": 60},
    "instagram_reels": {"aspect": "9:16",  "min_dur": 15, "max_dur": 90},
    "linkedin":        {"aspect": "16:9",  "min_dur": 30, "max_dur": 600},
    "custom":          {"aspect": "9:16",  "min_dur": 25, "max_dur": 90},
}

# ─── Virality word lists ──────────────────────────────────────────────────────
_HOOK_WORDS    = {"secret","never","always","stop","wait","truth","lie","fake","real",
                  "hidden","warning","mistake","shocking","unbelievable","incredible",
                  "exposed","leaked","banned","controversial","dangerous","illegal",
                  "insane","wild","crazy","mindblowing","finally","actually","literally"}
_EMOTION_WORDS = {"amazing","love","hate","fear","angry","sad","happy","excited",
                  "terrified","surprised","devastated","overwhelmed","frustrated",
                  "passionate","inspired","destroyed","heartbroken","obsessed","proud"}
_VALUE_WORDS   = {"free","save","earn","win","lose","rich","poor","easy","hard",
                  "fast","slow","better","best","worst","cheap","expensive","profit",
                  "million","billion","percent","guarantee","proven","results","strategy"}
_QUESTION_STARTERS = {"why","how","what","when","who","where","can","could","should","would"}
_POWER_WORDS   = _HOOK_WORDS | _EMOTION_WORDS | _VALUE_WORDS


# ─── Main engine ─────────────────────────────────────────────────────────────
class AgentOpusClipper:
    """
    Full pipeline:
      download → transcribe → AI highlight detection →
      dynamic zoom + smooth face tracking → animated captions → render.
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

    # ── Font ─────────────────────────────────────────────────────────────────
    def _find_font(self):
        for p in [r"C:\Windows\Fonts\arialbd.ttf",
                  r"C:\Windows\Fonts\calibrib.ttf",
                  r"C:\Windows\Fonts\verdanab.ttf",
                  r"C:\Windows\Fonts\arial.ttf"]:
            if os.path.exists(p):
                return p
        return None

    def _make_font(self, size: int = 52):
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
                       "+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"),
            "outtmpl":             out_path,
            "ffmpeg_location":     FFMPEG_DIR,
            "merge_output_format": "mp4",
            "noplaylist":          True,
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

    # ── Virality scoring (returns score + keywords hit) ───────────────────────
    def _virality_details(self, segments, t_start, t_end):
        """Return (score: int, keywords_hit: list[str])."""
        score = 50
        words_in = [
            w for s in segments for w in s.get("words", [])
            if t_start <= w["start"] <= t_end
        ]
        if not words_in:
            return score, []

        dur  = max(t_end - t_start, 1)
        text = " ".join(w["word"].lower() for w in words_in)

        # Speech pace
        wps = len(words_in) / dur
        score += min(int((wps - 1.5) * 6), 12)

        # Power word hits
        hit_words = sorted({pw for pw in _POWER_WORDS if pw in text})
        score    += min(len(hit_words) * 4, 20)

        # Question starters in first 10 words
        first_ten = " ".join(w["word"].lower() for w in words_in[:10])
        if any(qs in first_ten for qs in _QUESTION_STARTERS):
            score += 7
            hit_words = ["[question hook]"] + hit_words

        # Punctuation energy
        if "?" in text:              score += 6
        if "!" in text:              score += 4
        if re.search(r"\b\d+\b", text): score += 4

        # Speaker confidence
        conf = float(np.mean([w.get("probability", 0.85) for w in words_in]))
        if conf > 0.92: score += 6
        elif conf > 0.85: score += 3

        # Personal pronouns (makes it feel direct / intimate)
        pronouns = {"i","you","we","my","your","our","me","us"}
        if any(p in text.split() for p in pronouns):
            score += 4

        return min(int(score), 99), hit_words

    def _virality_score(self, segments, t_start, t_end) -> int:
        return self._virality_details(segments, t_start, t_end)[0]

    # ── Hook moment detection ─────────────────────────────────────────────────
    def _find_hook_moment(self, words_in_clip) -> float:
        """Return clip-local seconds of the best opening hook (0 = no hook found)."""
        for w in words_in_clip[:25]:
            clean = w["word"].lower().strip(".,!?'\"")
            if clean in _HOOK_WORDS or clean in _QUESTION_STARTERS:
                return max(0.0, w["start"] - 0.4)
            if w.get("probability", 0) > 0.96:
                return max(0.0, w["start"] - 0.2)
        return 0.0

    # ── AI highlight analysis ─────────────────────────────────────────────────
    def analyze_highlights(self, words, segments,
                           max_clips=8, min_dur=25, max_dur=60,
                           ollama_model="llama3"):
        self._cb("AI analysing viral potential…", 40)

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
                            sc, kw = self._virality_details(segments, s, e)
                            h["virality_score"] = sc
                            h["keywords_hit"]   = kw
                            h.setdefault("title", f"Clip {len(valid)+1}")
                            valid.append(h)
                    if valid:
                        return sorted(valid, key=lambda x: x["virality_score"], reverse=True)
            except Exception as ex:
                print(f"[Ollama] {ex} — heuristic scorer activated.")

        # ── Heuristic fallback ────────────────────────────────────────────
        if not segments:
            return []

        total_dur = segments[-1]["end"]
        window = min(max_dur, max(min_dur, total_dur * 0.4))
        step   = max(5.0, window * 0.2)

        scored = []
        t = 0.0
        while t + window <= total_dur:
            sc, kw = self._virality_details(segments, t, t + window)
            scored.append((t, t + window, sc, kw))
            t += step

        scored.sort(key=lambda x: x[2], reverse=True)
        selected, highlights = [], []
        for s, e, score, kw in scored:
            if all(e <= sel[0] or s >= sel[1] for sel in selected):
                selected.append((s, e))
                idx = len(selected)
                highlights.append({
                    "start": s, "end": e,
                    "title": f"Viral Moment {idx}",
                    "reason": "Heuristic: high engagement density",
                    "hook": "",
                    "virality_score": score,
                    "keywords_hit": kw,
                })
                if len(selected) >= max_clips:
                    break

        return sorted(highlights, key=lambda x: x["virality_score"], reverse=True)

    # ── Face tracking ─────────────────────────────────────────────────────────
    def _compute_face_track(self, video, sample_fps=5):
        """Returns closure t→center_x, or None if no faces found."""
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

        def _interp(t: float) -> float:
            return float(np.interp(t, t_arr, smoothed))
        return _interp

    # ── Dynamic zoom track (Opus Clip signature — implemented FREE here) ──────
    def _compute_zoom_track(self, words_in_clip, clip_dur,
                            max_zoom: float = 1.15, sample_rate: float = 0.1):
        """
        Returns closure t→zoom_level in [1.0, max_zoom].
        Zooms smoothly into power words and high-confidence speech bursts.
        """
        times = np.arange(0, clip_dur + sample_rate, sample_rate)
        energy = np.ones(len(times))

        for w in words_in_clip:
            conf     = w.get("probability", 0.85)
            is_power = w["word"].lower().strip(".,!?'\"") in _POWER_WORDS
            boost    = (0.35 if is_power else 0.0) + max(0.0, (conf - 0.85) * 1.5)
            if boost < 0.01:
                continue
            t_mid = (w["start"] + w["end"]) / 2.0
            for i, t in enumerate(times):
                dist = abs(t - t_mid)
                if dist < 2.5:
                    energy[i] += boost * max(0.0, 1.0 - dist / 2.5)

        e_min, e_max = energy.min(), energy.max()
        if e_max > e_min:
            norm = (energy - e_min) / (e_max - e_min)
        else:
            norm = np.zeros_like(energy)

        smoothed = uniform_filter1d(norm, size=max(1, int(0.8 / sample_rate)))
        zoom_arr = 1.0 + smoothed * (max_zoom - 1.0)

        def _zoom(t: float) -> float:
            return float(np.interp(t, times, zoom_arr))
        return _zoom

    # ── Caption styles ────────────────────────────────────────────────────────

    def _render_karaoke_caption(self, phrase_words, current_idx,
                                frame_w, cap_h=160, font_size=54):
        """Classic karaoke: current word yellow, others white, dark semi-transparent bg."""
        img  = Image.new("RGBA", (frame_w, cap_h), (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (frame_w, cap_h), (10, 10, 10, 200)))
        draw = ImageDraw.Draw(img)
        fnt  = self._make_font(font_size)

        tokens    = [w["word"] + (" " if i < len(phrase_words)-1 else "")
                     for i, w in enumerate(phrase_words)]
        bbox      = draw.textbbox((0, 0), "".join(tokens), font=fnt)
        x = max(24, (frame_w - (bbox[2] - bbox[0])) // 2)
        y = (cap_h - (bbox[3] - bbox[1])) // 2

        for i, tok in enumerate(tokens):
            is_cur = (i == current_idx)
            color  = (255, 220, 0, 255) if is_cur else (255, 255, 255, 230)
            if is_cur:
                draw.text((x + 2, y + 2), tok, font=fnt, fill=(0, 0, 0, 160))
            draw.text((x, y), tok, font=fnt, fill=color)
            x += draw.textbbox((0, 0), tok, font=fnt)[2]

        out = Image.new("RGB", (frame_w, cap_h), (0, 0, 0))
        out.paste(img, mask=img.split()[3])
        return np.array(out)

    def _render_bold_box_caption(self, phrase_words, current_idx,
                                 frame_w, cap_h=180, font_size=56):
        """
        TikTok-style bold box: each word in its own pill.
        Active word = bright yellow pill with black text.
        Others = dark pill with white text.
        """
        fnt     = self._make_font(font_size)
        padding = 18
        gap     = 10

        # Pre-measure pill widths
        dummy = Image.new("RGBA", (1, 1))
        dd    = ImageDraw.Draw(dummy)
        pill_data = []
        for w in phrase_words:
            bb = dd.textbbox((0, 0), w["word"], font=fnt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            pill_data.append((w["word"], tw + padding * 2, th + padding, tw, th))

        total_w = sum(p[1] for p in pill_data) + gap * (len(pill_data) - 1)
        x0      = max(10, (frame_w - total_w) // 2)
        y_pill  = (cap_h - (pill_data[0][4] + padding)) // 2

        img  = Image.new("RGBA", (frame_w, cap_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        x = x0
        for i, (word, pw, ph, tw, th) in enumerate(pill_data):
            is_cur   = (i == current_idx)
            bg_color = (255, 220, 0, 255) if is_cur else (30, 30, 30, 220)
            fg_color = (0, 0, 0, 255)     if is_cur else (255, 255, 255, 240)

            # Rounded rectangle pill
            r  = ph // 2
            rx = [x, y_pill, x + pw, y_pill + ph]
            draw.rounded_rectangle(rx, radius=r, fill=bg_color)

            # Text centred in pill
            tx = x + (pw - tw) // 2
            ty = y_pill + (ph - th) // 2
            draw.text((tx, ty), word, font=fnt, fill=fg_color)
            x += pw + gap

        out = Image.new("RGB", (frame_w, cap_h), (0, 0, 0))
        out.paste(img, mask=img.split()[3])
        return np.array(out)

    def _render_minimal_caption(self, phrase_words, current_idx,
                                frame_w, cap_h=120, font_size=52):
        """
        Minimal: white text with thick black outline — no background box.
        Active word slightly larger + bright white.
        """
        img  = Image.new("RGBA", (frame_w, cap_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fnt  = self._make_font(font_size)
        fnt_big = self._make_font(font_size + 6)

        tokens = [w["word"] + (" " if i < len(phrase_words)-1 else "")
                  for i, w in enumerate(phrase_words)]
        total_text = "".join(tokens)
        bb   = draw.textbbox((0, 0), total_text, font=fnt)
        x    = max(20, (frame_w - (bb[2] - bb[0])) // 2)
        y    = (cap_h - (bb[3] - bb[1])) // 2

        for i, tok in enumerate(tokens):
            is_cur = (i == current_idx)
            f      = fnt_big if is_cur else fnt
            color  = (255, 255, 255, 255) if is_cur else (220, 220, 220, 230)
            # Thick black outline (8-way shadow)
            for dx, dy in [(-3,-3),(3,-3),(-3,3),(3,3),(0,-3),(0,3),(-3,0),(3,0)]:
                draw.text((x+dx, y+dy), tok, font=f, fill=(0, 0, 0, 230))
            draw.text((x, y), tok, font=f, fill=color)
            x += draw.textbbox((0, 0), tok, font=f)[2]

        out = Image.new("RGB", (frame_w, cap_h), (0, 0, 0))
        out.paste(img, mask=img.split()[3])
        return np.array(out)

    def _render_caption_image(self, phrase_words, current_idx,
                              frame_w, style="karaoke", cap_h=None):
        """Dispatch to selected caption style renderer."""
        if style == "bold_box":
            h = cap_h or 180
            return self._render_bold_box_caption(phrase_words, current_idx, frame_w, h)
        elif style == "minimal":
            h = cap_h or 120
            return self._render_minimal_caption(phrase_words, current_idx, frame_w, h)
        else:  # karaoke (default)
            h = cap_h or 160
            return self._render_karaoke_caption(phrase_words, current_idx, frame_w, h)

    def _cap_h_for_style(self, style: str) -> int:
        return {"bold_box": 180, "minimal": 120}.get(style, 160)

    def _build_caption_clips(self, words_in_clip, final_w, final_h,
                             style="karaoke", clip_dur=None):
        cap_h  = self._cap_h_for_style(style)
        cap_y  = final_h - cap_h - 50
        clips  = []

        for i, word in enumerate(words_in_clip):
            w_start = word["start"]
            w_end   = word["end"]
            if clip_dur is not None:
                w_end = min(w_end, clip_dur)
            dur = max(w_end - w_start, 0.04)

            ph_s    = max(0, i - 3)
            ph_e    = min(len(words_in_clip), i + 5)
            phrase  = words_in_clip[ph_s:ph_e]
            cur_idx = i - ph_s

            frame = self._render_caption_image(phrase, cur_idx, final_w, style, cap_h)
            clips.append(
                ImageClip(frame, duration=dur)
                .with_start(w_start)
                .with_position((0, cap_y))
            )
        return clips

    # ── Render one clip ───────────────────────────────────────────────────────
    def create_clip(self, video_path, highlight, words,
                    aspect="9:16", output_dir=None, progress_pct=70,
                    caption_style="karaoke", enable_zoom=True):
        if output_dir is None:
            output_dir = OUTPUT_DIR

        t_start  = highlight["start"]
        t_end    = highlight["end"]
        score    = highlight.get("virality_score", 0)
        title    = re.sub(r"[^\w\s-]", "", highlight.get("title", "clip")).strip()
        title    = title.replace(" ", "_")[:40]
        out_path = os.path.join(output_dir, f"[{score}]_{title}.mp4")

        self._cb(f'Rendering "{title}" ({t_start:.0f}s–{t_end:.0f}s) [{caption_style}]…',
                 progress_pct)

        video        = VideoFileClip(video_path).subclipped(t_start, t_end)
        src_w, src_h = video.size
        clip_dur     = video.duration

        final_w, final_h = {"9:16": (1080, 1920),
                            "1:1":  (1080, 1080),
                            "16:9": (1920, 1080)}.get(aspect, (1080, 1920))

        # ── Words for this clip (clip-local timestamps) ────────────────────
        words_in = [
            {**w, "start": w["start"] - t_start, "end": w["end"] - t_start}
            for w in words if t_start <= w["start"] < t_end
        ]

        # ── Build zoom track ───────────────────────────────────────────────
        zoom_fn = None
        if enable_zoom and words_in:
            zoom_fn = self._compute_zoom_track(words_in, clip_dur)

        # ── Build video clip with crop + zoom ──────────────────────────────
        if aspect == "16:9":
            if zoom_fn is not None:
                def zoom_frame_16(get_frame, t):
                    frame = get_frame(t)
                    z = zoom_fn(t)
                    if z > 1.001:
                        fh, fw = frame.shape[:2]
                        nw, nh = int(fw / z), int(fh / z)
                        x0 = (fw - nw) // 2; y0 = (fh - nh) // 2
                        frame = np.array(
                            Image.fromarray(frame[y0:y0+nh, x0:x0+nw])
                            .resize((fw, fh), Image.LANCZOS)
                        )
                    return frame
                cropped = video.transform(zoom_frame_16).resized((final_w, final_h))
            else:
                cropped = video.resized((final_w, final_h))
        else:
            tgt_w = int(src_h * 9 / 16) if aspect == "9:16" else min(src_w, src_h)
            tgt_h = src_h               if aspect == "9:16" else tgt_w
            face_fn = self._compute_face_track(video)

            def crop_frame(get_frame, t):
                frame = get_frame(t)

                # 1. Dynamic zoom (zoom in on centre before face-crop)
                if zoom_fn is not None:
                    z = zoom_fn(t)
                    if z > 1.001:
                        fh, fw = frame.shape[:2]
                        nw, nh = int(fw / z), int(fh / z)
                        x0z = (fw - nw) // 2; y0z = (fh - nh) // 2
                        frame = np.array(
                            Image.fromarray(frame[y0z:y0z+nh, x0z:x0z+nw])
                            .resize((fw, fh), Image.LANCZOS)
                        )

                # 2. Face-tracking crop
                cx = float(face_fn(t)) if face_fn is not None else src_w / 2
                cx = max(tgt_w / 2, min(src_w - tgt_w / 2, cx))
                x0 = int(max(0, min(src_w - tgt_w, cx - tgt_w / 2)))

                if aspect == "9:16":
                    return frame[:, x0:x0 + tgt_w]
                else:  # 1:1
                    cy = src_h // 2
                    y0 = int(max(0, min(src_h - tgt_h, cy - tgt_h // 2)))
                    return frame[y0:y0 + tgt_h, x0:x0 + tgt_w]

            cropped = video.transform(crop_frame).resized((final_w, final_h))

        # ── Animated captions ──────────────────────────────────────────────
        cap_clips = self._build_caption_clips(words_in, final_w, final_h,
                                              style=caption_style,
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
            for clip in (final, cropped, video):
                try: clip.close()
                except Exception: pass

        return out_path

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    def generate_thumbnail(self, clip_path: str) -> str:
        thumb = clip_path.replace(".mp4", "_thumb.jpg")
        try:
            clip = VideoFileClip(clip_path)
            t    = min(clip.duration * 0.12, clip.duration - 0.1)
            Image.fromarray(clip.get_frame(t)).save(thumb, quality=88)
            clip.close()
        except Exception:
            pass
        return thumb

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, source: str, is_url=False, max_clips=6,
            aspect="9:16", ollama_model="llama3",
            min_dur=25, max_dur=60,
            caption_style="karaoke", enable_zoom=True):
        """
        Returns list of dicts:
          path, thumb, title, virality_score, reason, hook,
          keywords_hit, hook_moment, start, end
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
            min_dur=min_dur, max_dur=max_dur,
            ollama_model=ollama_model,
        )
        if not highlights:
            if segments:
                total = segments[-1]["end"]
                _, kw = self._virality_details(segments, 0, min(total, 60))
                highlights = [{
                    "start": 0, "end": min(total, 60),
                    "title": "Full Video", "reason": "Short video — full clip",
                    "hook": "", "virality_score": 50, "keywords_hit": kw,
                }]
            else:
                raise ValueError("No highlights found — is the video too short or silent?")

        results = []
        for i, h in enumerate(highlights):
            pct = 65 + int((i / max(len(highlights), 1)) * 30)

            # Find hook moment
            t_start = h["start"]
            t_end   = h["end"]
            words_in = [
                {**w, "start": w["start"] - t_start, "end": w["end"] - t_start}
                for w in words if t_start <= w["start"] < t_end
            ]
            hook_moment = self._find_hook_moment(words_in)

            clip_path = self.create_clip(
                video_path, h, words,
                aspect=aspect, progress_pct=pct,
                caption_style=caption_style,
                enable_zoom=enable_zoom,
            )
            results.append({
                "path":           clip_path,
                "thumb":          self.generate_thumbnail(clip_path),
                "title":          h.get("title", f"Clip {i+1}"),
                "virality_score": h.get("virality_score", 0),
                "reason":         h.get("reason", ""),
                "hook":           h.get("hook", ""),
                "keywords_hit":   h.get("keywords_hit", []),
                "hook_moment":    hook_moment,
                "start":          h["start"],
                "end":            h["end"],
            })

        self._cb("✅ All clips ready!", 100)
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Agent Opus CLI v2")
    p.add_argument("--url",     help="YouTube / TikTok URL")
    p.add_argument("--file",    help="Local video path")
    p.add_argument("--clips",   type=int, default=6)
    p.add_argument("--aspect",  default="9:16", choices=["9:16", "1:1", "16:9"])
    p.add_argument("--model",   default="large-v3")
    p.add_argument("--llm",     default="llama3",
                   help="Ollama model, or 'none' for heuristic only")
    p.add_argument("--platform", default="custom",
                   choices=list(PLATFORM_PRESETS.keys()),
                   help="Platform preset (overrides --aspect, --min-dur, --max-dur)")
    p.add_argument("--caption-style", default="karaoke",
                   choices=["karaoke", "bold_box", "minimal"])
    p.add_argument("--no-zoom", action="store_true",
                   help="Disable dynamic zoom effect")
    p.add_argument("--min-dur", type=int, default=25)
    p.add_argument("--max-dur", type=int, default=60)
    args = p.parse_args()

    # Platform preset overrides
    preset = PLATFORM_PRESETS[args.platform]
    aspect   = preset["aspect"]
    min_dur  = preset["min_dur"]
    max_dur  = preset["max_dur"]
    # Manual overrides still win
    if args.aspect != "9:16":  aspect  = args.aspect
    if args.min_dur != 25:     min_dur = args.min_dur
    if args.max_dur != 60:     max_dur = args.max_dur

    def log(msg, pct): print(f"[{pct:3d}%] {msg}")
    clipper = AgentOpusClipper(whisper_model_size=args.model, progress_callback=log)

    src    = args.url or args.file
    is_url = bool(args.url)
    if not src:
        p.print_help(); raise SystemExit(1)

    res = clipper.run(
        src, is_url=is_url,
        max_clips=args.clips, aspect=aspect,
        ollama_model=args.llm,
        min_dur=min_dur, max_dur=max_dur,
        caption_style=args.caption_style,
        enable_zoom=not args.no_zoom,
    )

    print("\n=== DONE ===")
    for r in res:
        kw = ", ".join(r["keywords_hit"][:6]) or "none"
        print(f"  [{r['virality_score']}/100] {r['title']}")
        print(f"    keywords: {kw}  |  hook@{r['hook_moment']:.1f}s  →  {r['path']}")
