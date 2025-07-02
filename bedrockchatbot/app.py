from flask import Flask, request, render_template, jsonify
import boto3
import json
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = Flask(__name__)

# Access credentials from .env
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)


MODEL_ID = os.getenv('MODEL_ID')

# Your remaining code...


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return "Server is running!"

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message', '')

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": user_input}
                ]
            }
        ]
    }

    try:
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        result = json.loads(response['body'].read())

        # Extract the assistant's reply text cleanly
        answer = ""
        if 'output' in result:
            message = result['output'].get('message', {})
            contents = message.get('content', [])
            texts = [item.get('text', '') for item in contents if 'text' in item]
            answer = "\n".join(texts).strip()

        if not answer:
            answer = result.get('response') or result.get('completion') or str(result)

    except Exception as e:
        answer = f"Error invoking model: {str(e)}"

    return jsonify({'response': answer})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
