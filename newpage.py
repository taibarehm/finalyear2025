# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import re
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime # For timestamps in chat history

# Ensure NLTK 'punkt' is downloaded (run `python -c "import nltk; nltk.download('punkt')"` once)
# If you didn't download it globally, you might need to handle it here or ensure it's available.

app = Flask(__name__)

# --- Simple AI Logic (Rule-Based for Stress Assessment) ---
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

# --- New: Basic NLU for general chat ---
def get_chatbot_response(user_message):
    user_message_lower = user_message.lower()

    # Keyword-based responses
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

    # Fallback response
    return "I'm still learning! For now, I can help you with stress assessment via specific options. Or you can ask general questions about stress, feelings, or well-being. Try asking 'What helps with stress?' or 'I feel anxious'."


# --- Routes ---

@app.route('/')
def index():
    """Serves your main chatbot HTML page."""
    return render_template('index.html')

@app.route('/handle-chat-input', methods=['POST'])
def handle_chat_input():
    """
    Handles general chat input from the frontend when the assessment is NOT active.
    This is where the new NLU logic will be applied.
    """
    user_message = request.json.get('message', '')
    
    print(f"Received general user message: '{user_message}'")
    
    # Process the message using our basic NLU
    bot_response = get_chatbot_response(user_message)
    
    return jsonify({'status': 'success', 'bot_message': bot_response})


@app.route('/save-chat', methods=['POST'])
def save_chat():
    """
    Handles the POST request from your frontend to save chat data after the assessment.
    This is where your 'AI' processing for the collected assessment data occurs.
    """
    data = request.json
    chat_history = data.get('chat', [])
    received_stress_score = data.get('stressScore')
    
    print("Received chat data from frontend:")
    # print(chat_history) # Uncomment to see full chat history in console
    
    if received_stress_score is not None:
        print(f"Final Stress Score received: {received_stress_score}")
        ai_recommendation = get_ai_insight(received_stress_score)
        print(f"AI Recommendation (from assessment): {ai_recommendation}")
        
        # Here you would typically save `chat_history`, `received_stress_score`,
        # and `ai_recommendation` to a database.
        
        return jsonify({'status': 'success', 'message': 'Chat data processed', 'ai_insight': ai_recommendation})
    else:
        return jsonify({'status': 'error', 'message': 'Stress score not provided.'}), 400


@app.route('/boot.html')
def boot_page():
    """Serves the boot.html page after the chat is saved."""
    return render_template('boot.html')

if __name__ == '__main__':
    app.run(debug=True)