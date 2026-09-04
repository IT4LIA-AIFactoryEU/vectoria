#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vectoria_lib.api.routers import agent as agent_router

app = FastAPI()
app.include_router(agent_router.router)
client = TestClient(app)


def test_health_endpoint_still_works():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK"}


def test_chat_endpoint_is_removed():
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}], "thread_id": "t1"})
    assert resp.status_code == 404
