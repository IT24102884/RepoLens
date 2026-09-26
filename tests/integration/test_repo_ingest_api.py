from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_list_repos_endpoint():
    response = client.get("/api/repos")
    assert response.status_code == 200
    data = response.json()
    assert "repos" in data
    assert any(r["name"] == "encode/httpx" for r in data["repos"])


def test_switch_repo_default():
    response = client.post("/api/repo/switch", json={"repo_slug": "default"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["active_repo"] == "encode/httpx"


def test_switch_repo_not_found():
    response = client.post("/api/repo/switch", json={"repo_slug": "non_existent_repo_999"})
    assert response.status_code == 404


def test_ingest_repo_invalid_url():
    response = client.post("/api/repo/ingest", json={"repo_url": "invalid_url"})
    assert response.status_code == 400


@patch("backend.api.main.RepoCloner.clone")
@patch("backend.api.main.RepoCloner.collect_source_files")
@patch("backend.api.main.ChromaRetriever.index_chunks")
def test_ingest_repo_mocked_success(mock_index, mock_collect, mock_clone, tmp_path):
    # Setup mock repo directory
    mock_clone.return_value = (tmp_path, "testorg/testlib")
    dummy_file = tmp_path / "main.py"
    dummy_file.write_text("def hello(): return 'world'\n", encoding="utf-8")
    mock_collect.return_value = [dummy_file]

    response = client.post("/api/repo/ingest", json={"repo_url": "https://github.com/testorg/testlib"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["repo_name"] == "testorg/testlib"
    assert data["total_chunks"] > 0
    assert "Successfully ingested" in data["message"]

    # Verify active repo in /api/stats
    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["repo_name"] == "testorg/testlib"

    # Reset back to default
    switch_resp = client.post("/api/repo/switch", json={"repo_slug": "default"})
    assert switch_resp.status_code == 200
    assert switch_resp.json()["active_repo"] == "encode/httpx"
