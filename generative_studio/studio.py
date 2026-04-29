import os
import json
import time
import sys

# Inject ffmpeg into PATH
FFMPEG_DIR = r"D:\AI\AgentOpus\ffmpeg\ffmpeg-8.1-essentials_build\bin"
if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("FFMPEG_BINARY",  os.path.join(FFMPEG_DIR, "ffmpeg.exe"))
os.environ.setdefault("FFPROBE_BINARY", os.path.join(FFMPEG_DIR, "ffprobe.exe"))

from crewai import Agent, Task, Crew, Process
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama   # fallback
import gradio as gr
from moviepy import ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips
import torch

# Initialize Local LLM for Agents
llama3 = Ollama(model="llama3")

class GenerativeStudio:
    def __init__(self):
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_creative_squad(self, prompt):
        # 1. Researcher Agent
        researcher = Agent(
            role='Researcher',
            goal=f'Gather deep insights and facts about: {prompt}',
            backstory='Expert at finding niche details and compelling statistics to make content authoritative.',
            llm=llama3,
            verbose=True
        )
        
        # 2. Scriptwriter Agent
        writer = Agent(
            role='Scriptwriter',
            goal='Write a punchy, engaging 60-second video script based on research.',
            backstory='Master of storytelling and emotional hooks. Expert at writing for visual engagement.',
            llm=llama3,
            verbose=True
        )
        
        # 3. Motion Designer (Storyboarder)
        storyboarder = Agent(
            role='Motion Designer',
            goal='Break the script into 5 visual scenes with detailed image prompts for AI generation.',
            backstory='Expert in visual aesthetics. Knows exactly what prompts work best for Stable Diffusion.',
            llm=llama3,
            verbose=True
        )
        
        # Tasks
        research_task = Task(description=f'Research facts for: {prompt}', agent=researcher, expected_output="A list of 5 key facts.")
        write_task = Task(description='Write a 60s script.', agent=writer, expected_output="A script with dialogue and scene descriptions.")
        storyboard_task = Task(description='Create image prompts for 5 scenes.', agent=storyboarder, expected_output="A JSON-like list of 5 image prompts.")
        
        crew = Crew(
            agents=[researcher, writer, storyboarder],
            tasks=[research_task, write_task, storyboard_task],
            process=Process.sequential
        )
        
        return crew.kickoff()

    def generate_video(self, prompt, style):
        print(f"Starting production for: {prompt} (Style: {style})")
        
        # 1. Brains Phase
        production_plan = self.create_creative_squad(prompt)
        print("Production Plan Ready!")
        
        # 2. Rendering Phase (Mocking ComfyUI call for now, as it's a separate background process)
        # In a real scenario, we send 'storyboard_task' output to ComfyUI API
        # Here we'll generate a placeholder video using MoviePy and the script text
        
        clips = []
        for i in range(3):   # Create 3 placeholder scenes
            bg_clip  = ColorClip(size=(1080, 1920), color=(20, 20, min(i * 50, 255))).with_duration(5)
            clips.append(bg_clip)

        final_video = concatenate_videoclips(clips)
        output_path = os.path.join(self.output_dir, f"generated_{int(time.time())}.mp4")

        final_video.write_videofile(
            output_path, fps=24, codec="libx264",
            audio=False, logger=None,
        )
        
        return output_path, str(production_plan)

studio = GenerativeStudio()

def studio_ui(prompt, style):
    video_path, logs = studio.generate_video(prompt, style)
    return video_path, logs

with gr.Blocks(title="Agent Opus - Generative Studio") as demo:
    gr.Markdown("# 🎨 Agent Opus: Autonomous Generative Studio")
    gr.Markdown("Describe a topic, and the Creative Squad will research, write, and render your video.")
    
    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(label="What is the video about?", placeholder="The history of SpaceX, Why coffee is good for you...")
            style = gr.Dropdown(label="Art Style", choices=["Space Cinematic", "Plastic Blocks", "Claymation", "Collage", "3D Animation"], value="Space Cinematic")
            generate_btn = gr.Button("🎬 Start Autonomous Production", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Final Production")
            log_output = gr.Textbox(label="Creative Squad Logs", lines=10)
            
    generate_btn.click(
        fn=studio_ui,
        inputs=[topic, style],
        outputs=[video_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
