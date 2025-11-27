import gradio as gr
import os
import shutil
from TTS.api import TTS
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Configure Tesseract OCR for multiple languages
tesseract_path = shutil.which("tesseract")  # Auto-detect Tesseract path
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise RuntimeError("Tesseract OCR not found. Please install it.")

ocr_languages = "eng+hin+mar"  # English, Hindi, Marathi

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file.name)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else "Error: No text extracted. Try an image-based PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to extract text from an image file
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image, lang=ocr_languages)
        return text.strip() if text.strip() else "Error: No text found in image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

# Function to perform voice cloning and generate speech
def generate_cloned_speech(speaker_audio, input_file):
    if not speaker_audio:
        return "Error: Please upload a valid audio file.", None
    if not input_file:
        return "Error: Please upload a valid PDF or image file.", None

    file_extension = os.path.splitext(input_file.name)[1].lower()
    
    # Determine input file type and extract text
    if file_extension == ".pdf":
        input_text = extract_text_from_pdf(input_file)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        input_text = extract_text_from_image(input_file)
    else:
        return "Error: Unsupported file type. Upload a PDF, JPG, JPEG, or PNG file.", None

    if not input_text or "Error" in input_text:
        return f"Error: {input_text}", None

    # Validate audio format (must be .wav)
    audio_extension = os.path.splitext(speaker_audio.name)[1].lower()
    if audio_extension != ".wav":
        return "Error: Please upload a valid .wav audio file.", None

    # Get the path of the uploaded audio file
    speaker_wav_path = speaker_audio.name  
    output_file = "output.wav"

    try:
        tts.tts_to_file(
            text=input_text,
            file_path=output_file,
            speaker_wav=speaker_wav_path,
            language="en",
            split_sentences=True
        )
        return "Speech generation completed successfully!", output_file
    except Exception as e:
        return f"Error during speech generation: {str(e)}", None

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("## Voice Cloning with Text Extraction from PDF or Images")
    gr.Markdown("Upload a speaker's audio file and a PDF or image file to extract text. The text will be converted to speech in the provided speaker's voice.")

    with gr.Row():
        speaker_audio = gr.File(label="Upload Speaker Audio (.wav)", file_types=[".wav"])
        input_file = gr.File(label="Upload PDF or Image File", file_types=[".pdf", ".jpg", ".jpeg", ".png"])

    generate_button = gr.Button("Generate Speech")
    output_message = gr.Textbox(label="Status")
    output_audio = gr.Audio(label="Generated Audio", type="filepath", interactive=False)

    generate_button.click(
        fn=generate_cloned_speech,
        inputs=[speaker_audio, input_file],
        outputs=[output_message, output_audio]
    )

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(debug=True, share=True)
