import os
import base64
import json
import re
import time
from flask import Flask, request, render_template, jsonify
import boto3
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pdfplumber
from PIL import Image

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AWS credentials from .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("MODEL_ID")

# AWS Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Extract text from PDFs
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Resize and compress images before encoding
def optimize_image(input_path, output_path, max_size=(1024, 1024), quality=75):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)
        img.save(output_path, format='JPEG', quality=quality)

# Function to clean markdown special characters from model output
import re

def clean_markdown(text):
    # Remove markdown headers (#)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove bold ** or __
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)

    # Remove italic * or _
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Replace unordered list markers (-, *, +) with bullet points
    text = re.sub(r'^[\s]*[-*+]\s+', '• ', text, flags=re.MULTILINE)

    # Replace ordered lists: 1. or 2. etc with bullet points or keep numbers
    text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)

    # Replace multiple newlines with max two newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove backticks `code`
    text = re.sub(r'`(.+?)`', r'\1', text)

    # Strip leading/trailing whitespace
    return text.strip()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('message', '').strip()
    uploaded_file = request.files.get('file')
    file_text = ""
    ext = ""
    image_base64 = None

    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf':
                file_text = extract_text_from_pdf(filepath)
                if not file_text.strip():
                    raise ValueError("PDF content empty, fallback to OCR")
            elif ext in ['.png', '.jpg', '.jpeg']:
                optimized_path = os.path.join(app.config['UPLOAD_FOLDER'], "optimized_" + filename)
                optimize_image(filepath, optimized_path)
                with open(optimized_path, "rb") as image_file:
                    image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                file_text = "[Image uploaded]"
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            return jsonify({'response': f"⚠️ Failed to process file: {str(e)}"})

        # Truncate large text files
        max_chars = 4000
        if len(file_text) > max_chars:
            file_text = file_text[:max_chars] + "\n\n[...Truncated due to length limit]"

    # Prepare Bedrock payload
    if uploaded_file and ext in ['.png', '.jpg', '.jpeg']:
        prompt = (
            "You are a helpful and accurate assistant. "
            "Describe the content of the provided image in detail. "
            "Do not extract text—just describe what you see."
        )
        media_type = "image/jpeg"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {
                            "toolUse": {
                                "name": "image",
                                "toolUseId": "image1",
                                "input": {
                                    "image_base64": image_base64,
                                    "media_type": media_type
                                }
                            }
                        }
                    ]
                }
            ]
        }
    else:
        prompt = f"""
You are a helpful and accurate assistant. Answer the user's question based only on the content below.

--- Begin Content ---
{file_text}
--- End Content ---

User Question: {user_input}
Answer:"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        }

    max_retries = 5
    answer = ""
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                body=json.dumps(payload),
                modelId=MODEL_ID,
                accept='application/json',
                contentType='application/json'
            )
            result = json.loads(response['body'].read())
            text_chunks = result.get('output', {}).get('message', {}).get('content', [])
            answer = ' '.join(chunk.get('text', '') for chunk in text_chunks)
            if not answer:
                answer = str(result)
            break
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                answer = f"Error invoking model: {str(e)}"
                break

    # Clean answer before sending to frontend
    answer = clean_markdown(answer)

    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(debug=True)
