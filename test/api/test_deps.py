#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from vectoria_lib.api.deps import _check_token, _get_tenant_id

app = FastAPI()


@app.get("/protected", dependencies=[Depends(_check_token)])
def protected():
    return {"ok": True}


@app.get("/tenant")
def tenant(tenant_id: str = Depends(_get_tenant_id)):
    return {"tenant_id": tenant_id}


client = TestClient(app)


# --- _check_token ---

def test_auth_disabled_by_default_when_permitted_keys_unset(monkeypatch):
    monkeypatch.delenv("PERMITTED_KEYS", raising=False)
    resp = client.get("/protected")
    assert resp.status_code == 200


def test_auth_rejects_missing_token_when_permitted_keys_set(monkeypatch):
    monkeypatch.setenv("PERMITTED_KEYS", "secret1,secret2")
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("PERMITTED_KEYS", "secret1,secret2")
    resp = client.get("/protected", headers={"x-token": "wrong"})
    assert resp.status_code == 401


def test_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("PERMITTED_KEYS", "secret1,secret2")
    resp = client.get("/protected", headers={"x-token": "secret2"})
    assert resp.status_code == 200


# --- _get_tenant_id ---

def test_tenant_id_defaults_to_default():
    resp = client.get("/tenant")
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "default"


def test_tenant_id_accepts_valid_header():
    resp = client.get("/tenant", headers={"x-tenant-id": "company-a_1"})
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "company-a_1"


def test_tenant_id_is_lowercased():
    resp = client.get("/tenant", headers={"x-tenant-id": "Company-A"})
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "company-a"


def test_tenant_id_rejects_path_traversal():
    resp = client.get("/tenant", headers={"x-tenant-id": "../../etc/passwd"})
    assert resp.status_code == 400


def test_tenant_id_rejects_too_long():
    resp = client.get("/tenant", headers={"x-tenant-id": "a" * 65})
    assert resp.status_code == 400
