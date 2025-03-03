import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import time
import os

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pretrained EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

MODEL_PATH = "best_cnn.pt"
NUM_CLASSES = 10

CLASS_NAMES = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def initialize_model():
    """Initialize the model once at startup."""
    global model
    try:
        model = CNNModel(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess the input image for the EfficientNet model."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    input_tensor = preprocess(image)
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict(image, progress=gr.Progress()):
    """Make predictions with visual feedback for the prediction process."""
    if image is None:
        return None, "No image provided. Please upload an image."
    
    try:
        # Step 1: Update progress (20%)
        progress(0.2, "Preprocessing image...")
        
        # Preprocess the image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        input_batch = preprocess_image(image)
        input_batch = input_batch.to(device)
        
        # Step 2: Update progress (50%)
        progress(0.5, "Running prediction model...")
        
        # Add a small delay to make the prediction step visible
        time.sleep(0.5)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
        
        # Step 3: Update progress (80%)
        progress(0.8, "Processing results...")
        
        # Get predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_indices = torch.topk(probabilities, min(1, NUM_CLASSES))
        
        # Format the results with class names
        results = {}
        for prob, idx in zip(top_prob, top_indices):
            class_name = CLASS_NAMES[idx.item()]
            confidence = prob.item() * 100
            results[class_name] = f"{confidence:.2f}%"
            
        # Format results as HTML for better display
        html_result = "<div style='text-align:left; padding:10px; background:#f9f9f9; border-radius:5px;'>"
        html_result += "<h3>Prediction Results:</h3>"
        html_result += "<ul style='list-style-type:none; padding-left:10px;'>"
        
        for class_name, confidence in results.items():
            # Extract the numerical value for the width of the progress bar
            confidence_value = float(confidence.strip('%'))
            bar_width = confidence_value  # Width as percentage
            
            html_result += f"""
            <li style='margin-bottom:10px;'>
                <div style='display:flex; align-items:center;'>
                    <div style='width:150px; font-weight:bold;'>{class_name}:</div>
                    <div style='flex-grow:1;'>
                        <div style='background:#e0e0e0; border-radius:3px; height:20px; width:100%;'>
                            <div style='background:linear-gradient(90deg, #4CAF50, #8BC34A); 
                                        height:20px; width:{bar_width}%; 
                                        border-radius:3px; text-align:right;'>
                                <span style='padding-right:5px; color:white; font-weight:bold;'>{confidence}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </li>
            """
        
        html_result += "</ul></div>"
        
        # Step 4: Update progress (100%)
        progress(1.0, "Done!")
        
        return image, html_result
    
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        return None, f"<div style='color:red; padding:10px;'>{error_message}</div>"

# Create Gradio interface
def create_app():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("""# Paddy Disease Classification
        Upload an image to find out which disease your crop has.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(type="pil", label="Upload Image")
                predict_btn = gr.Button("Predict", variant="primary")
                
            with gr.Column(scale=1):
                # Output components
                # output_image = gr.Image(type="pil", label="Processed Image")
                output_html = gr.HTML(label="Prediction Results")
                
        # Set up the prediction flow
        predict_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_html,]
        )
        
        gr.Examples(
            examples=[],
            inputs=input_image
        )
        
        # Clear button
        gr.ClearButton(components=[input_image, output_html])
    
    return demo

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please ensure the model file is in the correct location.")
        exit(1)
    
    # Initialize model
    if initialize_model():
        # Create and launch the app
        app = create_app()
        app.launch(share=True)  # Set share=False if you don't want to create a public link
    else:
        print("Application startup failed due to model initialization errors.")