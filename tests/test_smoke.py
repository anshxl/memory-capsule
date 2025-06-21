import httpx  # type: ignore
import pytest # type: ignore 

BASE = "http://127.0.0.1:8000"
USER = "test_user"

@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE)

def test_stats(client):
    r = client.get(f"/stats/{USER}")
    assert r.status_code == 200
    data = r.json()
    assert "total_entries" in data and "streak" in data and "badges" in data

@pytest.mark.parametrize("mode,payload_key", [
    ("ai", "answers"),
    ("manual", "content"),
])
def test_entry_modes(client, mode, payload_key):
    if mode == "manual":
        body = {"mode": "manual", "user_id":USER,
                "content":"Today was a good day!"}
    else:
        answers = [f"A{i}" for i in range(10)]
        body = {"mode": "ai", "user_id": USER, "answers": answers}
    
    r = client.post("/entry", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "entry_id" in data and "text" in data

def test_flashback(client):
    # Assuming at least one entry exists
    r = client.get(f"/flashback/{USER}", params={"q":"good", "k":2})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)

    # Each items should have 'entry_id' and 'content
    for item in data:
        assert "entry_id" in item and "content" in item
    
def test_finetune(client):
    samples = [f"Sample line {i}" for i in range(10)]
    r = client.post(f"/tune/{USER}", json={"samples": samples})
    assert r.status_code == 200
    data = r.json()
    assert "adapter_repo" in data and data["status"] == "completed"