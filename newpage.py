import os
import numpy as np
import base64
import json
import random

from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle
import cv2

app = Flask(__name__)

try:
    model = load_model('model/stress_detection_model.h5')
    class_labels = ['nostress', 'stress']
except Exception as e:
    print(f"Error loading Keras model: {e}")
    model = None

try:
    with open('model/questionar_stress_model.pkl', 'rb') as f:
        model2 = pickle.load(f)
        
except Exception as e:
    print(f"Error loading the CatBoost model: {e}")
    model2 = None

user_data = {}


def get_ai_insight(stress_score):
    if stress_score <= 5:
        return "Excellent! Your stress levels are very low. Keep up your positive habits."
    elif stress_score <= 12:
        return "You're experiencing moderate stress. Consider relaxation techniques."
    elif stress_score <= 20:
        return "Your stress level is high. Prioritize self-care."
    else:
        return "Very high stress detected. Consider professional help."

def get_chatbot_response(user_message):
    user_message_lower = user_message.lower()
    if "hello" in user_message_lower or "hi" in user_message_lower:
        return "Hi there! How can I help you today regarding stress?"
    elif any(word in user_message_lower for word in ["how are you", "how r u"]):
        return "I'm a chatbot, so I don't have feelings, but I'm ready to assist you!"
    elif "thank you" in user_message_lower or "thanks" in user_message_lower:
        return "You're welcome! Is there anything else I can help with?"
    elif "stress" in user_message_lower and ("relief" in user_message_lower or "cope" in user_message_lower):
        return "Stress relief can involve breathing, mindfulness, or talking to someone."
    elif "sad" in user_message_lower or "depressed" in user_message_lower:
        return "I'm sorry you're feeling this way. Consider reaching out for help."
    elif "anxious" in user_message_lower or "worried" in user_message_lower:
        return "Anxiety can be tough. Try grounding techniques and breathing."
    elif "tired" in user_message_lower:
        return "Feeling tired can be a sign of stress. Ensure rest and self-care."
    elif "help" in user_message_lower:
        return "I can help assess your stress levels or provide coping tips."
    elif "bye" in user_message_lower:
        return "Goodbye! Take care of yourself."
    elif "tell me more" in user_message_lower:
        return "Is there a specific aspect of stress you'd like to explore?"
    elif "quiz" in user_message_lower or "assessment" in user_message_lower or "start" in user_message_lower:
        return "Please go to the main page or type 'yes' to restart assessment."
    return "I'm still learning! Try asking about stress, feelings, or coping."

def calculate_holistic_score_and_message(chatbot_score, questionnaire_score, face_score):
    print("chatbot_score :  ",chatbot_score, "questionnaire_score :  ",questionnaire_score,"face_score :  ",face_score  )
    weights = {'chatbot': 0.3, 'questionnaire': 0.4, 'face': 0.3}
    holistic_score = (
        chatbot_score * weights['chatbot']
        + questionnaire_score * weights['questionnaire']
        + face_score * weights['face']
    )

    if holistic_score < 0.3:
        final_message = "Low stress. Great job!"
    elif holistic_score < 0.6:
        final_message = "Moderate stress. Consider relaxing."
    else:
        final_message = "High stress. Professional advice may help."

    return final_message, holistic_score

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/boot.html')
def boot_page():
    return render_template('boot.html')

@app.route('/handle-chat-input', methods=['POST'])
def handle_chat_input():
    user_message = request.json.get('message', '')
    bot_response = get_chatbot_response(user_message)
    return jsonify({'status': 'success', 'bot_message': bot_response})

@app.route('/save-chat', methods=['POST'])
def save_chat():
    data = request.json
    received_stress_score = data.get('stressScore')
    if received_stress_score is not None:
        # Normalize 0–1
        user_data['chatbot_score'] = received_stress_score / 64.0
        ai_recommendation = get_ai_insight(received_stress_score)
        return jsonify({'status': 'success', 'message': 'Chat data processed', 'ai_insight': ai_recommendation})
    return jsonify({'status': 'error', 'message': 'Stress score not provided.'}), 400

@app.route('/questionnaire')
def questionnaire_page():
    return render_template('questionnaire.html', message=None)
@app.route('/save-questionnaire', methods=['POST'])
def save_questionnaire():
    try:
        data = request.json
        print("Received questionnaire data:", data)
        return jsonify({'status': 'success', 'message': 'Questionnaire data saved successfully.'})
    except Exception as e:
        print(f"Error saving questionnaire data: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to save questionnaire data.'}), 500

@app.route('/questionnaire-detect', methods=['POST'])
def questionnaire_detect():
    if model2 is None:
        return jsonify({'success': False, 'message': "Error: Model not loaded."}), 500
    try:
        data = request.json
        feature_names = [
            'snoring_range', 'respiration_rate', 'body_temperature',
            'limb_movement', 'blood_oxygen', 'eye_movement',
            'hours_of_sleep', 'heart_rate'
        ]
        features = [float(data[name]) for name in feature_names]

        print("Feayure requested :",features)
        predicted_stress_level = int(model2.predict([features])) 
        print("predicted_stress_level",predicted_stress_level)
        user_data['questionnaire_score'] = predicted_stress_level/3
        return jsonify({'success': True, 'message': f"Physiological Questionnaire Stress Level: {predicted_stress_level}"})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/face', methods=['GET', 'POST'])
def face_detection():
    prediction = None
    if request.method == 'POST':
        image_data = request.form.get('image_data')
        if image_data:
            image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img_np = np.array(img)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img_np, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = img_np[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                img_array = img_to_array(face_img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                pred_class_index = np.argmax(pred)
                pred_class = class_labels[pred_class_index]
                confidence = float(np.max(pred))
                if pred_class == 'stress':
                    face_score = confidence
                else:
                    face_score = 1 - confidence
                user_data['face_score'] = face_score
                prediction = f"Facial Prediction: {pred_class} (confidence: {confidence:.2f})."
            else:
                prediction = "No face detected."
    return render_template('face.html', prediction=prediction)

@app.route('/holistic_report', methods=['GET', 'POST'])
def holistic_report():
    print("User data:", user_data)
    
    # Initialize variables to None to prevent UndefinedError
    final_message = "Not enough data to complete assessment."
    holistic_score = None
    chatbot_score = None
    questionnaire_score = None
    face_score = None

    if all(k in user_data for k in ('chatbot_score', 'questionnaire_score', 'face_score')):
        final_message, holistic_score = calculate_holistic_score_and_message(
            user_data['chatbot_score'],
            user_data['questionnaire_score'],
            user_data['face_score']
        )
        chatbot_score = user_data['chatbot_score']
        questionnaire_score = user_data['questionnaire_score']
        face_score = user_data['face_score']
        print(f"Final message: {final_message}, Holistic score: {holistic_score}")
    # Optional: Clear session after viewing
    user_data.clear()
    
    return render_template('holistic_report.html',
                           final_message=final_message,
                           holistic_score=holistic_score,
                           chatbot_score=chatbot_score,
                           questionnaire_score=questionnaire_score,
                           face_score=face_score)

if __name__ == '__main__':
    app.run(debug=True)
