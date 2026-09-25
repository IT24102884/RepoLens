from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy', 'service': 'RepoLens Engine'}

def test_stats_endpoint():
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.json()
    assert data['repo_name'] == 'encode/httpx'
    assert data['total_chunks'] > 0

def test_empty_query_validation():
    response = client.post('/api/query', json={'query': '   '})
    assert response.status_code == 400

def test_frontend_index_serving():
    response = client.get('/')
    assert response.status_code == 200
    assert 'RepoLens' in response.text
    assert len(response.text) > 1000

def test_query_system_c_out_of_scope():
    response = client.post('/api/query', json={'query': 'What is the weather in Tokyo?', 'system': 'c'})
    assert response.status_code == 200
    data = response.json()
    assert 'System C' in data['system_version']
    assert data['intent'] == 'OUT_OF_SCOPE'
    assert 'outside the scope' in data['answer']
    assert len(data['citations']) == 0
