import os
import asyncio
import tempfile
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
import edge_tts

# Load environment variables from .env file (for local testing)
load_dotenv()

# Initialize Groq client
# The client automatically picks up GROQ_API_KEY from the environment
try:
    client = Groq()
except Exception as e:
    print(f"Failed to initialize Groq client: {e}. Please make sure GROQ_API_KEY is set.")
    client = None

async def generate_tts(text: str, voice: str) -> str:
    """Generate TTS audio and save it to a temporary file."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        # Create a temporary file with a reliable extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            output_path = tmp_file.name
        
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def process_audio(audio_filepath, direction, high_accuracy):
    if client is None:
        return "Error: Groq client not initialized.", "Please check your API key.", None
    
    if not audio_filepath:
        return "No audio provided.", "Please record or upload an audio file.", None
        
    try:
        # Step 1: Transcription using Whisper on Groq
        with open(audio_filepath, "rb") as file:
            transcription_response = client.audio.transcriptions.create(
                file=(audio_filepath, file.read()),
                model="whisper-large-v3-turbo",
                response_format="json"
            )
        
        original_text = transcription_response.text
        
        if not original_text or original_text.strip() == "":
            return "No speech detected.", "Could not transcribe audio.", None

        # Step 2: Translation using Llama 3.3 or DeepSeek-R1 on Groq
        system_prompt = ""
        tts_voice = ""
        
        # Determine the model to use
        # Llama 4 Scout is the latest reasoning-optimized model on Groq as of 2026
        selected_model = "meta-llama/llama-4-scout-17b-16e-instruct" if high_accuracy else "llama-3.3-70b-versatile"
        
        if direction == "English to Hebrew":
            system_prompt = (
                "You are an expert translator specializing in translating English to Hebrew. "
                "Translate the following English text into grammatically correct, natural-sounding Hebrew. "
                "Ensure the tone matches the context (everyday conversation vs professional recruiter). "
                "Provide ONLY the Hebrew translation. Do not include any introductory text, explanations, or quotes."
            )
            tts_voice = "he-IL-AvriNeural" # "he-IL-HilaNeural" is also an option
        else:
            system_prompt = (
                "You are an expert translator specializing in translating Hebrew to English. "
                "You are highly sensitive to Hebrew nuances and common phonetic/orthographic mistakes. "
                "For example, differentiate correctly between 'בוחן' (testing/examining) and 'בוכה' (crying) based on the surrounding sentence context. "
                "Translate the following Hebrew text into grammatically correct, natural-sounding English. "
                "Provide ONLY the English translation. Do not include any introductory text, explanations, or quotes."
            )
            tts_voice = "en-US-AriaNeural"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_text}
            ],
            model=selected_model,
            temperature=0.1, # Low temperature for more accurate/deterministic translation
            max_tokens=1024,
        )
        
        translated_text = chat_completion.choices[0].message.content.strip()

        # If using DeepSeek-R1, we might need to strip the <think> tags if they appear in the response
        if "<think>" in translated_text and "</think>" in translated_text:
            translated_text = translated_text.split("</think>")[-1].strip()

        # Step 3: Text-to-Speech using edge-tts
        # Gradio functions run synchronously by default unless defined as async, 
        # but edge-tts requires asyncio. We use asyncio.run to bridge it.
        tts_audio_path = asyncio.run(generate_tts(translated_text, tts_voice))

        return original_text, translated_text, tts_audio_path

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        return error_msg, "Translation failed.", None

# --- Gradio UI Setup ---
with gr.Blocks(title="English-Hebrew Speech Translator") as demo:
    gr.Markdown("# 🎤 English <-> Hebrew Speech-to-Speech Translator")
    gr.Markdown("Powered by **Groq** (Whisper-large-v3-turbo + Llama-3.3-70B) and **Edge-TTS**.")
    gr.Markdown("Record your voice or upload an audio file to get instant, accurate text translation and spoken output.")
    
    with gr.Row():
        with gr.Column():
            direction_radio = gr.Radio(
                choices=["English to Hebrew", "Hebrew to English"],
                value="English to Hebrew",
                label="Translation Direction"
            )
            high_accuracy_checkbox = gr.Checkbox(
                label="High Accuracy Mode (Llama 4 Reasoning)",
                value=False
            )
            audio_input = gr.Audio(
                sources=["microphone", "upload"], # Modern Gradio syntax
                type="filepath",
                label="Input Audio"
            )
            translate_btn = gr.Button("Translate", variant="primary")
            
        with gr.Column():
            original_text_output = gr.Textbox(label="Original Transcription", lines=3)
            translated_text_output = gr.Textbox(label="Translated Text", lines=3)
            audio_output = gr.Audio(label="Spoken Translation", type="filepath")

    with gr.Accordion("🛠️ Technical Specifications", open=False):
        gr.Markdown(
            "### System Architecture\n"
            "- **ASR:** [OpenAI Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)\n"
            "- **LLM:** [Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) / [Llama 4 Scout](https://groq.com/)\n"
            "- **Inference:** [Groq LPU™ (Language Processing Unit)](https://groq.com/)\n\n"
            "--- \n"
            "**Developer:** Josh Chamo - joshchamo@gmail.com and Google Gemini"
        )

    # Connect the UI elements to the function
    translate_btn.click(
        fn=process_audio,
        inputs=[audio_input, direction_radio, high_accuracy_checkbox],
        outputs=[original_text_output, translated_text_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
