import os
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask import Flask, render_template, request, jsonify 
import io
import pickle
import cv2  
app = Flask(__name__)
model = load_model('model.h5')
class_labels = ['nostress', 'stress']
# Load the trained model
try:
    with open('catboost_stress_model.pkl', 'rb') as f:
        model2 = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")
    model2 = None # Handle case where model loading fails

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/face', methods=['GET', 'POST'])
def face_detection():
    print("into facedetection")
    prediction = None
    if request.method == 'POST':
        image_data = request.form.get('image_data')
        if image_data:
            # Decode base64 image
            image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img_np = np.array(img)

            # Face detection using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img_np, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = img_np[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                print(f"Face region: x={x}, y={y}, w={w}, h={h}")
                print("Face image shape:", face_img.shape)
                img_array = img_to_array(face_img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                print("Prediction probabilities:", pred)
                pred_class = class_labels[np.argmax(pred)]
                print("Predicted class:", pred_class)
                prediction = f"Prediction: {pred_class} (confidence: {np.max(pred):.2f})"
            else:
                prediction = "No face detected. Please try again."
    return render_template('face.html', prediction=prediction)


def get_ai_insight(stress_score):
    """Provides a basic AI-driven insight based on the stress score."""
    if stress_score <= 5:
        return "Excellent! Your stress levels are very low. Keep up your positive habits."
    elif stress_score <= 12:
        return "You're experiencing moderate stress. It's a good time to consider relaxation techniques."
    elif stress_score <= 20:
        return "Your stress level is high. This calls for active self-care and potentially seeking support."
    else:
        return "Very high stress detected. Please prioritize your well-being and consider professional help."

def get_chatbot_response(user_message):
    user_message_lower = user_message.lower()
    if "hello" in user_message_lower or "hi" in user_message_lower:
        return "Hi there! How can I help you today regarding stress?"
    elif any(word in user_message_lower for word in ["how are you", "how r u"]):
        return "I'm a chatbot, so I don't have feelings, but I'm ready to assist you! How are you feeling?"
    elif "thank you" in user_message_lower or "thanks" in user_message_lower:
        return "You're welcome! Is there anything else about stress or well-being I can help with?"
    elif "stress" in user_message_lower and ("relief" in user_message_lower or "cope" in user_message_lower):
        return "Stress relief can involve deep breathing, mindfulness, exercise, or talking to someone. What kind of strategies are you interested in?"
    elif "sad" in user_message_lower or "depressed" in user_message_lower or "down" in user_message_lower:
        return "I'm sorry to hear you're feeling that way. It's important to reach out for support if you're feeling persistently sad or depressed. Consider talking to a mental health professional or a trusted friend."
    elif "anxious" in user_message_lower or "worried" in user_message_lower:
        return "Anxiety can be tough. Try practicing grounding techniques, focusing on your breath, or identifying the root cause of your worries. If it's severe, professional help is recommended."
    elif "tired" in user_message_lower or "exhausted" in user_message_lower:
        return "Feeling tired can be a sign of stress. Ensure you're getting enough sleep and taking breaks throughout your day."
    elif "help" in user_message_lower:
        return "I can help you assess your stress levels or provide general information and tips related to stress management. What do you need help with?"
    elif "bye" in user_message_lower or "goodbye" in user_message_lower:
        return "Goodbye! Take care of yourself. Feel free to chat again anytime."
    elif "tell me more" in user_message_lower or "what else" in user_message_lower:
        return "Is there a specific aspect of stress or well-being you'd like to explore further? I can discuss symptoms, causes, or coping mechanisms."
    elif "quiz" in user_message_lower or "assessment" in user_message_lower or "start" in user_message_lower:
        return "If you'd like to start the stress assessment, please go to the main page or type 'yes' when prompted to restart."
    return "I'm still learning! For now, I can help you with stress assessment via specific options. Or you can ask general questions about stress, feelings, or well-being. Try asking 'What helps with stress?' or 'I feel anxious'."

@app.route('/handle-chat-input', methods=['POST'])
def handle_chat_input():
    user_message = request.json.get('message', '')
    print(f"Received general user message: '{user_message}'")
    bot_response = get_chatbot_response(user_message)
    return jsonify({'status': 'success', 'bot_message': bot_response})

# --- Save Chat Route ---
@app.route('/save-chat', methods=['POST'])
def save_chat():
    data = request.json
    chat_history = data.get('chat', [])
    received_stress_score = data.get('stressScore')
    print("Received chat data from frontend:")
    if received_stress_score is not None:
        print(f"Final Stress Score received: {received_stress_score}")
        ai_recommendation = get_ai_insight(received_stress_score)
        print(f"AI Recommendation (from assessment): {ai_recommendation}")
        # Here you would typically save chat_history, received_stress_score, ai_recommendation to a database.
        return jsonify({'status': 'success', 'message': 'Chat data processed', 'ai_insight': ai_recommendation})
    else:
        return jsonify({'status': 'error', 'message': 'Stress score not provided.'}), 400


@app.route('/boot.html')
def boot_page():
    return render_template('boot.html')
@app.route('/questionnaire', methods=['GET'])  
def questionnaire_page(): 
    
    return render_template('questionnaire.html', message=None)
@app.route('/questionnaire-detect', methods=['POST'])
def questionnaire_detect():
    """
    Handles the submission of the questionnaire, processes the input,
    and makes a prediction using the loaded CatBoost model.
    """
    if model2 is None:
        return jsonify({'success': False, 'message': "Error: Stress detection model not loaded. Please check server logs."}), 500

    try:
        feature_names = [
            'snoring_range',
            'respiration_rate',
            'body_temperature',
            'limb_movement',
            'blood_oxygen',
            'eye_movement',
            'hours_of_sleep',
            'heart_rate'
        ]

        features = []
        for feature_name in feature_names:
            if feature_name not in request.form:
                raise KeyError(f"Missing form data for '{feature_name}'. Please ensure all fields are filled.")
            features.append(float(request.form[feature_name]))
        print(f"Extracted features: {features}")
        
        pred = model2.predict([features])[0] 
        stress_level = int(pred)
        message = f"Predicted Stress Level: {stress_level}"
        print(f"Prediction successful: {message}")
        
        # Return a JSON response for success
        return jsonify({'success': True, 'message': message})

    except KeyError as e:
        message = f"Error: Missing form data. {e}. Please ensure all fields are filled."
        print(f"KeyError: {message}")
        # Return a JSON response for an error
        return jsonify({'success': False, 'message': message}), 400
    except ValueError as e:
        message = f"Error: Invalid input data. Please enter numerical values. Details: {e}"
        print(f"ValueError: {message}")
        # Return a JSON response for an error
        return jsonify({'success': False, 'message': message}), 400
    except Exception as e:
        message = f"An unexpected error occurred during prediction: {e}"
        print(f"General Error: {message}")
        # Return a JSON response for a general error
        return jsonify({'success': False, 'message': message}), 500

if __name__ == '__main__':
    app.run(debug=True)