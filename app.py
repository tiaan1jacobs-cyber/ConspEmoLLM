from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Load model (using the exact method from README)
MODEL_PATH = "lzw1008/ConspEmoLLM-v2"
print("Loading ConspEmoLLM...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map='auto',
        torch_dtype=torch.float16,  # Use half precision to save memory
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

@app.route('/analyze', methods=['POST'])
def analyze_text():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(
                inputs["input_ids"], 
                max_new_tokens=100,  # Changed from max_length to max_new_tokens
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        return jsonify({
            'result': response,
            'original_text': text,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'ConspEmoLLM API is running',
        'endpoints': {
            'analyze': 'POST /analyze',
            'health': 'GET /health'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
