import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load the ConspEmoLLM model
MODEL_PATH = "lzw1008/ConspEmoLLM-v2"
print("Loading ConspEmoLLM...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    print("Model loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"Error: {e}")
    tokenizer = None
    model = None
    model_loaded = False

def analyze_text(text):
    """Analyze text with ConspEmoLLM"""
    if not model_loaded:
        return {"error": "Model not loaded", "status": "error"}
    
    if not text.strip():
        return {"error": "No text provided", "status": "error"}
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], 
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return {
            "result": result,
            "original_text": text,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

def api_interface(text):
    """Main interface for API calls"""
    if text.strip().lower() == "/health":
        return json.dumps({"status": "healthy", "model_loaded": model_loaded}, indent=2)
    
    result = analyze_text(text)
    return json.dumps(result, indent=2)

# Create simple Gradio interface
demo = gr.Interface(
    fn=api_interface,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Enter text to analyze...\nOr type '/health' to check status",
        label="Input Text"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Analysis Result (JSON)"
    ),
    title="ConspEmoLLM API",
    description="Analyze text for conspiracy theories and emotions",
    examples=[
        ["/health"],
        ["I think the government is hiding vaccine information and this makes me worried."],
        ["The media manipulates climate data to control us and I feel angry about it."]
    ]
)

if __name__ == "__main__":
    demo.launch()
