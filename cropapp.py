import streamlit as st
import numpy as np
import joblib
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Custom CSS for better styling
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 2rem;
    }
    h3 {
        color: #1B5E20;
    }
    .stNumberInput>div>div>input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Agentic AI Components
class CropRecommendationAgent:
    def __init__(self):
        self.feedback_history = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.load_feedback_history()
        
    def load_feedback_history(self):
        try:
            if os.path.exists('feedback_history.json'):
                with open('feedback_history.json', 'r') as f:
                    self.feedback_history = json.load(f)
        except Exception as e:
            st.warning(f"Could not load feedback history: {e}")
            self.feedback_history = []

    def save_feedback_history(self):
        try:
            with open('feedback_history.json', 'w') as f:
                json.dump(self.feedback_history, f)
        except Exception as e:
            st.warning(f"Could not save feedback history: {e}")

    def get_similar_cases(self, current_features):
        if not self.feedback_history:
            return []
        
        # Convert feedback history to feature vectors
        historical_features = np.array([case['features'] for case in self.feedback_history])
        current_features = np.array(current_features).reshape(1, -1)
        
        # Calculate similarity scores
        similarities = cosine_similarity(current_features, historical_features)[0]
        
        # Get top 3 similar cases
        top_indices = np.argsort(similarities)[-3:][::-1]
        return [(self.feedback_history[i], similarities[i]) for i in top_indices]

    def adjust_recommendation(self, prediction, confidence, features):
        similar_cases = self.get_similar_cases(features)
        
        if similar_cases and confidence < self.confidence_threshold:
            # Consider historical feedback for similar cases
            successful_crops = [case['crop'] for case, sim in similar_cases 
                              if case['feedback'] == 'success' and sim > 0.8]
            
            if successful_crops:
                # If we have successful cases with high similarity, adjust recommendation
                return successful_crops[0], confidence + 0.1
        
        return prediction, confidence

    def record_feedback(self, features, crop, feedback, confidence):
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'crop': crop,
            'feedback': feedback,
            'confidence': confidence
        }
        self.feedback_history.append(feedback_entry)
        self.save_feedback_history()

# Initialize the agent
@st.cache_resource
def initialize_agent():
    return CropRecommendationAgent()

agent = initialize_agent()

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load("/Users/shreeya/Desktop/crop_project/best_model.pkl")

model = load_model()

# Crop explanation dictionary
CROP_REASONS = {
    "rice": "Rice grows well in areas with high humidity, heavy rainfall (more than 150 mm), and slightly acidic to neutral soil (pH 5.5 to 7). It requires warm temperatures (around 20-30°C) and nitrogen-rich soil.",
    "maize": "Maize prefers warm temperatures (21-27°C), moderate rainfall, and slightly acidic to neutral soil (pH 5.5 to 7.5). It grows well in nitrogen-rich, well-drained soils with good sunlight exposure.",
    "chickpea": "Chickpeas require cooler, dry climates and grow best in loamy, well-drained soils. Ideal temperature is 18-30°C, and they tolerate moderate rainfall. Optimal pH is around 6 to 7.5.",
    "kidneybeans": "Kidney beans grow well in warm climates with moderate rainfall. They require well-drained, fertile soil with a pH between 6 and 7.5, and prefer moderate nitrogen content.",
    "pigeonpeas": "Pigeon peas require a warm climate (25-35°C), moderate rainfall, and well-drained soil. Ideal soil pH is 5.5 to 7. They are drought-resistant and suited for semi-arid conditions.",
    "mothbeans": "Moth beans are drought-resistant and grow in dry, arid regions with high temperatures and low rainfall. They prefer sandy or loamy soil with a pH range of 6.2 to 7.5.",
    "mungbean": "Mung beans prefer warm temperatures (25-35°C), moderate humidity, and loamy soil with good drainage. They grow best in soil with pH 6.2 to 7.2 and require moderate rainfall.",
    "blackgram": "Black gram grows in tropical climates with moderate rainfall and warm temperatures (25-30°C). It requires fertile, loamy soil with a pH of 6.0 to 7.5.",
    "lentil": "Lentils require cool weather during early stages and warm weather during maturation. Ideal pH is 6.0 to 8.0. They prefer loamy soils with moderate nitrogen content and low humidity.",
    "pomegranate": "Pomegranates thrive in hot, dry climates with less humidity. They need well-drained sandy loam soil with a pH of 5.5 to 7.2 and moderate water supply.",
    "banana": "Bananas need a humid, tropical climate with high temperatures (26-30°C) and high rainfall. They prefer deep, well-drained loamy soil with a pH of 6.0 to 7.5.",
    "mango": "Mangoes prefer tropical to subtropical climates, moderate rainfall, and well-drained soil. Ideal temperature is 24-30°C with pH ranging from 5.5 to 7.5.",
    "grapes": "Grapes require a warm, dry climate and loamy soil with good drainage. Optimal temperature is 20-30°C, and the ideal soil pH is 5.5 to 6.5.",
    "watermelon": "Watermelons thrive in hot climates with temperatures between 25-35°C and require sandy loam soil. They need low to moderate rainfall and soil pH of 6.0 to 7.5.",
    "muskmelon": "Muskmelons require warm temperatures (25-30°C), well-drained sandy loam soil, and low to moderate rainfall. Ideal pH is 6.0 to 7.0.",
    "apple": "Apples grow best in cool climates with cold winters and mild summers. Ideal pH is 6.0 to 7.0, and they need well-drained loamy soils and moderate humidity.",
    "orange": "Oranges grow well in subtropical climates with moderate humidity. Ideal temperature is 20-30°C. They require sandy loam soil and a pH range of 5.5 to 7.0.",
    "papaya": "Papayas prefer warm temperatures (25-35°C), high humidity, and well-drained soil. The ideal pH is 6.0 to 6.5 and they need moderate to high rainfall.",
    "coconut": "Coconuts grow well in coastal tropical climates with high humidity and rainfall. They need sandy, well-drained soil and pH between 5.5 and 7.0.",
    "cotton": "Cotton thrives in warm climates with low to moderate rainfall and plenty of sunshine. It requires well-drained, loamy soil with pH between 5.8 to 8.0.",
    "jute": "Jute grows in hot, humid climates with high rainfall and requires loamy alluvial soil. Ideal temperature is 24-37°C, and pH should be 6.0 to 7.5.",
    "coffee": "Coffee requires a tropical climate with moderate rainfall and temperature between 15-28°C. It grows in fertile, well-drained soil with pH 6.0 to 6.5."
}

# Load text2text model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def explain_prediction(crop, features):
    context = CROP_REASONS.get(crop.lower(), "No specific information available.")
    prompt = (
        f"The recommended crop is {crop}. "
        f"The field has Nitrogen={features[0]}, Phosphorus={features[1]}, Potassium={features[2]}, "
        f"Temperature={features[3]}°C, Humidity={features[4]}%, pH={features[5]}, Rainfall={features[6]} mm. "
        f"Given this data, and knowing that {context} Explain in simple terms why this crop fits these conditions."
    )
    output = generator(prompt, max_length=120)
    return output[0]["generated_text"]

# Streamlit UI
st.title("🌾 Smart Crop Recommendation System")

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Field Parameters")
    st.markdown("Enter your field's characteristics below:")
    
    # Create a container for input fields with a nice background
    with st.container():
        st.markdown("""
            <style>
            div[data-testid="stVerticalBlock"] > div:nth-child(1) {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=65.0)
        ph = st.number_input("Soil pH", value=6.5)
        rainfall = st.number_input("Rainfall (mm)", value=100.0)

with col2:
    st.markdown("### 📈 Field Overview")
    st.markdown("Visual representation of your field parameters:")
    
    # Create a container for the radar chart
    with st.container():
        st.markdown("""
            <style>
            div[data-testid="stVerticalBlock"] > div:nth-child(2) {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Radar chart visualization
        labels = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
        values = [N, P, K, temperature, humidity, ph, rainfall]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color='#4CAF50')
        ax.fill(angles, values, alpha=0.25, color='#4CAF50')
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("🌿 Field Condition Overview", pad=20, color='#2E7D32')
        st.pyplot(fig)

# Center the button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🌿 Get Crop Recommendation", use_container_width=True):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = [N, P, K, temperature, humidity, ph, rainfall]
    
    # Get initial prediction
    prediction = model.predict(input_data)[0]
    confidence = 0.0
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        confidence = round(np.max(probs) * 100, 2) / 100
    
    # Let the agent adjust the recommendation
    prediction, confidence = agent.adjust_recommendation(prediction, confidence, features)
    
    # Create a success message with custom styling
    st.markdown(f"""
        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h2 style='color: #2E7D32; text-align: center;'>✅ Recommended Crop: {prediction.upper()}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Confidence score with custom styling
    confidence_percentage = round(confidence * 100, 2)
    st.markdown(f"""
        <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1565C0; text-align: center;'>🌡️ Suitability Confidence: {confidence_percentage}%</h3>
        </div>
    """, unsafe_allow_html=True)

    # Show similar cases if available
    similar_cases = agent.get_similar_cases(features)
    if similar_cases:
        st.markdown("### 📊 Similar Historical Cases")
        for case, similarity in similar_cases:
            feedback_color = "#4CAF50" if case['feedback'] == 'success' else "#F44336"
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <p><strong>Crop:</strong> {case['crop'].upper()}</p>
                    <p><strong>Similarity:</strong> {round(similarity * 100, 2)}%</p>
                    <p><strong>Feedback:</strong> <span style='color: {feedback_color};'>{case['feedback'].upper()}</span></p>
                    <p><strong>Date:</strong> {datetime.fromisoformat(case['timestamp']).strftime('%Y-%m-%d')}</p>
                </div>
            """, unsafe_allow_html=True)

    # LLM Explanation with custom styling
    with st.spinner("🔍 Generating explanation..."):
        explanation = explain_prediction(prediction, features)
        st.markdown("""
            <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: #E65100; text-align: center;'>🧠 Why this crop?</h3>
                <p style='color: #333;'>{}</p>
            </div>
        """.format(explanation), unsafe_allow_html=True)

    # Feedback collection
    st.markdown("### 💭 Was this recommendation helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, it was successful", use_container_width=True):
            agent.record_feedback(features, prediction, "success", confidence)
            st.success("Thank you for your feedback! This helps improve our recommendations.")
    with col2:
        if st.button("❌ No, it wasn't successful", use_container_width=True):
            agent.record_feedback(features, prediction, "failure", confidence)
            st.error("Thank you for your feedback. We'll use this to improve our recommendations.")

st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>🌾 Smart Crop Recommendation System | Powered by Machine Learning & Agentic AI</p>
    </div>
""", unsafe_allow_html=True)
