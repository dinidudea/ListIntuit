from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import texttospeech
from google.oauth2 import service_account
import os
import json
import io
import nltk
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Configure credentials for both services
credentials_path = "credentials/google_credentials.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Set environment variable for Text-to-Speech
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Configure Gemini AI with the same credentials
genai.configure(credentials=credentials)
model = genai.GenerativeModel('gemini-pro')

# Settings file path
SETTINGS_FILE = 'settings.json'

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {
        "voice": "en-US-Studio-O",
        "language": "en-US",
        "speed": 1.0,
        "pitch": 0
    }

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def generate_ssml(text):
    prompt = f"""
    Convert the following text into SSML markup with appropriate prosody, emphasis, and breaks based on the text's sentiment and meaning:
    {text}
    Return only the SSML markup without any explanation.
    """
    response = model.generate_content(prompt)
    return response.text

def split_text(text, max_bytes=4800):
    try:
        if not text:
            return [""]

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence
            if len(test_chunk.encode('utf-8')) < max_bytes:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return [text[:max_bytes]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    try:
        data = request.json
        text = data.get('text', '')
        settings = data.get('settings', {})

        logger.debug(f"Received text length: {len(text)}")
        logger.debug(f"Received settings: {settings}")

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        save_settings(settings)

        client = texttospeech.TextToSpeechClient()
        chunks = split_text(text)
        logger.debug(f"Split into {len(chunks)} chunks")

        # Process all chunks and combine their audio content
        combined_audio = io.BytesIO()

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i} of {len(chunks)}")

            ssml = generate_ssml(chunk)
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
            voice = texttospeech.VoiceSelectionParams(
                language_code=settings['language'],
                name=settings['voice']
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=float(settings['speed']),
                pitch=float(settings['pitch'])
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Write this chunk's audio content to our combined buffer
            combined_audio.write(response.audio_content)

        # Prepare the final audio for sending
        combined_audio.seek(0)

        return send_file(
            combined_audio,
            mimetype='audio/mp3',
            as_attachment=True,
            download_name='speech.mp3'
        )

    except Exception as e:
        logger.error(f"Error in synthesize_speech: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_settings')
def get_settings():
    return jsonify(load_settings())

if __name__ == '__main__':
    app.run(debug=True)