import pytest
from newpage import app as flask_app
import os
from newone import app as flask_app
import numpy as np
import base64
import json
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from newpage import app as flask_app
from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle
import cv2
# --- Load Models ---
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

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_index_page(client):
    """Test if the index page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
   
    assert b"<title>" in response.data

def test_chatbot_response(client):
    """Test the chatbot's response to a simple greeting."""
    response = client.post('/handle-chat-input', json={'message': 'hello'})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'success'
    assert "Hi there!" in json_data['bot_message']

def test_questionnaire_page_loads(client):
    """Test if the questionnaire page loads correctly."""
    response = client.get('/questionnaire')
    assert response.status_code == 200
    
    assert b"<body>" in response.data
    
def test_physiological_submission(client):
    """Test submitting physiological data and receiving a prediction."""
    sample_data = {
        'snoring_range': 20,
        'respiration_rate': 18,
        'body_temperature': 98,
        'limb_movement': 10,
        'blood_oxygen': 98,
        'eye_movement': 13,
        'hours_of_sleep': 7,
        'heart_rate': 72
    }
    response = client.post('/submit-questionnaire', json=sample_data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert json_data['status'] == 'success'
def test_image_upload(client):
    """Test uploading an image for stress detection."""
    # Create a dummy image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = base64.b64encode(img_encoded).decode('utf-8')

    response = client.post('/predict-image', json={'image': img_bytes})
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert json_data['status'] == 'success'
def test_invalid_image_upload(client):
    """Test uploading invalid image data."""
    response = client.post('/predict-image', json={'image': 'not_base64_data'})
    assert response.status_code == 400 or response.get_json()['status'] == 'error'
def test_model_availability():
    """Test if models are loaded properly."""
    assert model is not None, "Keras model failed to load"
    assert model2 is not None, "CatBoost model failed to load"
    
def test_chatbot_unknown_input(client):
    """Test chatbot with an unknown message."""
    response = client.post('/handle-chat-input', json={'message': 'What is the meaning of life?'})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'success'
    assert isinstance(json_data['bot_message'], str)