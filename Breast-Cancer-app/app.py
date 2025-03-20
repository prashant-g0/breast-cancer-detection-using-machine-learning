from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import fitz  # PyMuPDF for PDF processing
import re  # For extracting numbers
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = joblib.load("breast_cancer_model.pkl")

# List of all 29 features
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst"
]

@app.route("/")
def home():
    return render_template("index.html", feature_names=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        app.logger.debug("Received a prediction request.")

        if "pdf_file" in request.files and request.files["pdf_file"].filename != "":
            app.logger.debug("PDF file uploaded.")
            pdf_file = request.files["pdf_file"]
            app.logger.debug(f"File received: {pdf_file.filename}")

            # Extract features from the uploaded PDF
            input_features = extract_features_from_pdf(pdf_file)
            app.logger.debug(f"Extracted features: {input_features}")
        else:
            app.logger.debug("Manual entry form submitted.")
            # Get input data from form fields (manual entry)
            input_features = [float(request.form[f]) for f in features]
            app.logger.debug(f"Manual input features: {input_features}")

        # Convert to numpy array for prediction
        input_array = np.array(input_features).reshape(1, -1)
        app.logger.debug(f"Input array for prediction: {input_array}")

        # Make prediction
        prediction = model.predict(input_array)[0]
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
        app.logger.debug(f"Prediction result: {result}")

        # Return JSON response
        return jsonify({"prediction_text": f"Prediction: {result}"})
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"prediction_text": f"Error: {str(e)}"})

def extract_features_from_pdf(pdf_file):
    """Extracts only required feature values from a PDF file by matching feature names."""
    app.logger.debug("Extracting features from PDF.")
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    
    # Extract text from all pages
    for page in doc:
        text += page.get_text("text") + "\n"

    app.logger.debug(f"Extracted text from PDF: {text}")

    # Dictionary to store extracted feature values
    extracted_data = {}

    for feature in features:
        # Create regex pattern to match feature names and capture their numerical values
        pattern = rf"({feature})\s*[:\-]?\s*([\d\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            extracted_data[feature] = float(match.group(2))
            app.logger.debug(f"Extracted {feature}: {extracted_data[feature]}")
        else:
            app.logger.warning(f"Could not extract feature: {feature}")

    # Ensure we have all required features
    if len(extracted_data) < len(features):
        raise ValueError("Could not extract all required features from the PDF.")

    # Return values in the correct order
    return [extracted_data[feature] for feature in features]

if __name__ == "__main__":
    app.run(debug=True)