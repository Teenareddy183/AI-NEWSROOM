import asyncio
import os
import uuid
import edge_tts
from pathlib import Path

def generate_audio_summary(topic: str, overview: str, findings: list, audio_type: str = 'brief', angle: str = '') -> str:
    """
    Generates an MP3 audio summary of the report using edge-tts.
    Returns the relative path to the generated audio file (e.g., 'audio/summary.mp3').
    """
    text_to_speak = f"Here is your AI Newsroom audio brief on {topic}. "
    text_to_speak += f"Overview: {overview} "
    
    if audio_type == 'detailed':
        if angle:
            text_to_speak += f"Recommended Angle: {angle} "
        if findings:
            text_to_speak += "Here are all the key findings: "
            for i, finding in enumerate(findings):
                headline = finding.get('headline', '')
                if headline:
                    text_to_speak += f"Finding {i+1}: {headline}. "
    elif audio_type == 'brief':
        if findings:
            text_to_speak += "Here are the key findings: "
            for i, finding in enumerate(findings[:3]):
                headline = finding.get('headline', '')
                if headline:
                    text_to_speak += f"Finding {i+1}: {headline}. "
    # If audio_type == 'short', we just stick to topic and overview
    
    text_to_speak += "End of brief."

    # Define paths
    project_root = Path(__file__).resolve().parent.parent
    audio_dir = project_root / "static" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = audio_dir / filename

    # Run the async edge-tts generation synchronously
    # "en-US-AriaNeural" is a good professional voice
    voice = "en-US-AriaNeural"
    
    async def _amain():
        communicate = edge_tts.Communicate(text_to_speak, voice)
        await communicate.save(str(filepath))

    asyncio.run(_amain())

    return f"audio/{filename}"
