"""
Agent Opus — Autonomous Generative Studio
Multi-agent CrewAI pipeline: research → script → storyboard → render.
"""

import os
import json
import time
import sys
import traceback

# ── FFmpeg ────────────────────────────────────────────────────────────────────
FFMPEG_DIR = r"D:\AI\AgentOpus\ffmpeg\ffmpeg-8.1-essentials_build\bin"
if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("FFMPEG_BINARY",  os.path.join(FFMPEG_DIR, "ffmpeg.exe"))
os.environ.setdefault("FFPROBE_BINARY", os.path.join(FFMPEG_DIR, "ffprobe.exe"))

from crewai import Agent, Task, Crew, Process

# ── LLM — lazy import, graceful if Ollama is not running ─────────────────────
def _load_ollama_llm(model: str = "llama3"):
    """Load the Ollama LLM. Returns None if Ollama is not available."""
    try:
        from langchain_ollama import OllamaLLM as Ollama
    except ImportError:
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            return None
    try:
        return Ollama(model=model)
    except Exception as ex:
        print(f"[Studio] Ollama load failed: {ex}")
        return None

import gradio as gr
from moviepy import ColorClip, CompositeVideoClip, concatenate_videoclips
import torch

# ── Absolute output directory (never depends on cwd) ─────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GenerativeStudio:
    def __init__(self):
        self._llm = None   # lazy-loaded on first generate call

    def _get_llm(self, model: str = "llama3"):
        if self._llm is None:
            self._llm = _load_ollama_llm(model)
        return self._llm

    def create_creative_squad(self, prompt: str, llm):
        # 1. Researcher Agent
        researcher = Agent(
            role="Researcher",
            goal=f"Gather deep insights and facts about: {prompt}",
            backstory=(
                "Expert at finding niche details and compelling statistics "
                "to make content authoritative."
            ),
            llm=llm,
            verbose=True,
        )

        # 2. Scriptwriter Agent
        writer = Agent(
            role="Scriptwriter",
            goal="Write a punchy, engaging 60-second video script based on research.",
            backstory=(
                "Master of storytelling and emotional hooks. "
                "Expert at writing for visual engagement."
            ),
            llm=llm,
            verbose=True,
        )

        # 3. Motion Designer / Storyboarder
        storyboarder = Agent(
            role="Motion Designer",
            goal="Break the script into 5 visual scenes with detailed image prompts.",
            backstory=(
                "Expert in visual aesthetics. Knows exactly what prompts work best "
                "for Stable Diffusion and AI image generation."
            ),
            llm=llm,
            verbose=True,
        )

        # Tasks
        research_task = Task(
            description=f"Research compelling facts for: {prompt}",
            agent=researcher,
            expected_output="A list of 5 key facts with sources.",
        )
        write_task = Task(
            description="Write a 60-second script based on the research.",
            agent=writer,
            expected_output="A script with dialogue and scene descriptions.",
        )
        storyboard_task = Task(
            description="Create detailed AI image prompts for 5 scenes from the script.",
            agent=storyboarder,
            expected_output="A JSON-like list of 5 image prompts.",
        )

        crew = Crew(
            agents=[researcher, writer, storyboarder],
            tasks=[research_task, write_task, storyboard_task],
            process=Process.sequential,
        )
        return crew.kickoff()

    def generate_video(self, prompt: str, style: str, llm_model: str = "llama3"):
        print(f"[Studio] Starting production: {prompt!r}  (Style: {style})")

        # ── Brains phase ──────────────────────────────────────────────────
        llm = self._get_llm(llm_model)
        production_plan_str = "(LLM unavailable — heuristic mode)"

        if llm is not None:
            try:
                result = self.create_creative_squad(prompt, llm)
                production_plan_str = str(result)
                print("[Studio] Production plan ready.")
            except Exception as ex:
                production_plan_str = f"(CrewAI error: {ex})\n\nProceeding with placeholder render."
                print(f"[Studio] CrewAI error: {ex}")
        else:
            print("[Studio] Ollama not available — generating placeholder video only.")

        # ── Rendering phase — placeholder scenes ──────────────────────────
        # In production: send storyboard prompts to ComfyUI API here.
        palette = [
            (20,  20,  80),   # scene 1 — deep blue
            (80,  20,  20),   # scene 2 — deep red
            (20,  80,  20),   # scene 3 — deep green
            (80,  60,   0),   # scene 4 — amber
            (40,  20,  80),   # scene 5 — violet
        ]
        clips = [
            ColorClip(size=(1080, 1920), color=color).with_duration(5)
            for color in palette
        ]

        final_video = concatenate_videoclips(clips)
        output_path = os.path.join(OUTPUT_DIR, f"generated_{int(time.time())}.mp4")

        try:
            final_video.write_videofile(
                output_path,
                fps=24,
                codec="libx264",
                audio=False,
                logger=None,
            )
        finally:
            try: final_video.close()
            except Exception: pass
            for c in clips:
                try: c.close()
                except Exception: pass

        return output_path, production_plan_str


# ── Singleton studio instance ─────────────────────────────────────────────────
studio = GenerativeStudio()


def studio_ui(prompt: str, style: str, llm_model: str):
    if not prompt.strip():
        return None, "❌ Please enter a topic."
    try:
        video_path, logs = studio.generate_video(prompt.strip(), style, llm_model)
        return video_path, logs
    except Exception:
        return None, f"❌ Error:\n\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Agent Opus — Generative Studio") as demo:
    gr.Markdown("# 🎨 Agent Opus: Autonomous Generative Studio")
    gr.Markdown(
        "Describe a topic, and the Creative Squad will research, write, "
        "and render your video automatically."
    )

    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(
                label="What is the video about?",
                placeholder="The history of SpaceX, Why coffee is good for you…",
            )
            style = gr.Dropdown(
                label="Art Style",
                choices=["Space Cinematic", "Plastic Blocks", "Claymation",
                         "Collage", "3D Animation"],
                value="Space Cinematic",
            )
            llm_model = gr.Dropdown(
                label="Ollama LLM",
                choices=["llama3", "llama3.1", "mistral", "gemma2", "phi3"],
                value="llama3",
            )
            generate_btn = gr.Button("🎬 Start Autonomous Production",
                                     variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Final Production")
            log_output   = gr.Textbox(label="Creative Squad Logs", lines=12,
                                      interactive=False)

    generate_btn.click(
        fn=studio_ui,
        inputs=[topic, style, llm_model],
        outputs=[video_output, log_output],
    )


if __name__ == "__main__":
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "localhost"
    print(f"\n[Studio] Running on http://localhost:7861  |  network: http://{local_ip}:7861\n")
    demo.launch(server_port=7861, share=False, inbrowser=True, show_error=True)
