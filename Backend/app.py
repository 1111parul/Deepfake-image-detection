from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

#Inirialise flask app
import os
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

#Define the model and processor directory
model_dir ="./deepfake_vs_real_image_detection"

# Load the model using safetensors
model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

# Load the processor
processor = ViTImageProcessor.from_pretrained(model_dir)
print("Model and processor loaded successfully!")
@app.route('/')
def home():
    return render_template('index.html')

#Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request"}), 400

        # Read the image file from the request
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        # Print analyzing message in backend terminal
        print("Analyzing image...")
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()

        # Map the predicted class ID to the corresponding label
        predicted_label = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

        # Print result in backend terminal
        print(f"Result: {predicted_label} | Confidence: {confidence*100:.2f}%")

        # Return the prediction as JSON
        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
