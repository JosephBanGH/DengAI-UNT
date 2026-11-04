"""Pruebas de integridad de la bitácora de auditoría (M03)."""
from __future__ import annotations

from app.db.session import SessionLocal
from app.models.security import AuditEvent
from app.services import audit as audit_service


def test_audit_chain_is_valid_after_operations(client, admin_headers):
    client.get("/api/v1/dashboard/overview", headers=admin_headers)
    response = client.get("/api/v1/audit/verify", headers=admin_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["checked"] > 0


def test_audit_detects_tampering(client, admin_headers):
    with SessionLocal() as session:
        event = session.query(AuditEvent).order_by(AuditEvent.id.asc()).first()
        assert event is not None
        original = event.description
        event.description = "descripción alterada por la prueba"
        session.commit()

    response = client.get("/api/v1/audit/verify", headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["valid"] is False

    # Se restaura el valor para no afectar las demás pruebas de la sesión.
    with SessionLocal() as session:
        event = session.query(AuditEvent).order_by(AuditEvent.id.asc()).first()
        event.description = original
        session.commit()


def test_audit_requires_permission(client, consult_headers):
    response = client.get("/api/v1/audit", headers=consult_headers)
    assert response.status_code == 403


def test_audit_event_includes_hash_and_previous_hash(client, admin_headers):
    response = client.get("/api/v1/audit", headers=admin_headers, params={"size": 5})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert all(len(item["hash"]) == 64 for item in items)


def test_module_event_records_duration(client, admin_headers):
    enter = client.post(
        "/api/v1/audit/module-event",
        headers=admin_headers,
        json={"event_type": "MODULE_ENTER", "module": "/laboratorio", "action": "test"},
    )
    assert enter.status_code == 200
    exit_response = client.post(
        "/api/v1/audit/module-event",
        headers=admin_headers,
        json={"event_type": "MODULE_EXIT", "module": "/laboratorio", "action": "test"},
    )
    assert exit_response.status_code == 200

    listing = client.get(
        "/api/v1/audit", headers=admin_headers, params={"event_type": "MODULE_EXIT", "size": 5}
    ).json()
    assert listing["items"]
