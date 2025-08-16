import pytest
from newpage import app as flask_app

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
