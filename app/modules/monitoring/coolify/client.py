"""Coolify API client — credentials stay server-side only."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CoolifyClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("COOLIFY_URL") or "").rstrip("/")
        self.token = token or os.environ.get("COOLIFY_TOKEN") or ""
        self.timeout = timeout
        self.app_uuids = {
            "api": os.environ.get("COOLIFY_APP_UUID") or "",
            "worker": os.environ.get("COOLIFY_WORKER_UUID") or "",
            "voice": os.environ.get("COOLIFY_VOICE_UUID") or os.environ.get("COOLIFY_APP_UUID_VOICE") or "",
            "frontend": os.environ.get("COOLIFY_FRONTEND_UUID") or "",
        }

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def _get(self, path: str) -> dict[str, Any] | list[Any] | None:
        if not self.configured:
            return None
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, headers=self._headers())
                if resp.status_code >= 400:
                    logger.warning("Coolify GET %s -> %s", path, resp.status_code)
                    return None
                return resp.json()
        except Exception as exc:
            logger.warning("Coolify request failed (non-fatal): %s", exc)
            return None

    def list_applications(self) -> list[dict[str, Any]]:
        data = self._get("/api/v1/applications")
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def get_application(self, uuid: str) -> dict[str, Any] | None:
        if not uuid:
            return None
        data = self._get(f"/api/v1/applications/{uuid}")
        return data if isinstance(data, dict) else None

    def list_deployments(self, app_uuid: str | None = None) -> list[dict[str, Any]]:
        path = "/api/v1/deployments"
        if app_uuid:
            path = f"/api/v1/applications/{app_uuid}/deployments"
        data = self._get(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        data = self._get(f"/api/v1/deployments/{deployment_id}")
        return data if isinstance(data, dict) else None

    def list_servers(self) -> list[dict[str, Any]]:
        data = self._get("/api/v1/servers")
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def server_resources(self, server_uuid: str | None = None) -> dict[str, Any] | None:
        uuid = server_uuid or os.environ.get("COOLIFY_SERVER_UUID") or ""
        if not uuid:
            servers = self.list_servers()
            if not servers:
                return None
            uuid = str(servers[0].get("uuid") or servers[0].get("id") or "")
        if not uuid:
            return None
        data = self._get(f"/api/v1/servers/{uuid}/resources")
        return data if isinstance(data, dict) else None

    def infrastructure_snapshot(self) -> dict[str, Any]:
        if not self.configured:
            return {
                "unavailable": ["coolify"],
                "message": "COOLIFY_URL / COOLIFY_TOKEN not configured",
                "containers": [],
                "cpu_percent": None,
                "ram_percent": None,
                "disk_percent": None,
                "deployments": [],
            }
        containers: list[dict[str, Any]] = []
        for key, uuid in self.app_uuids.items():
            if not uuid:
                continue
            app = self.get_application(uuid)
            if not app:
                containers.append({"service_key": key, "uuid": uuid, "status": "unknown"})
                continue
            containers.append(
                {
                    "service_key": key,
                    "uuid": uuid,
                    "name": app.get("name") or key,
                    "status": app.get("status") or app.get("fqdn") or "unknown",
                    "fqdn": app.get("fqdn"),
                    "restarts": app.get("restart_count") or app.get("restarts"),
                }
            )
        resources = self.server_resources() or {}
        # Coolify resource payload shapes vary; map common keys safely.
        cpu = resources.get("cpu") or resources.get("cpu_percent") or resources.get("cpuUsage")
        ram = resources.get("memory") or resources.get("ram_percent") or resources.get("memoryUsage")
        disk = resources.get("disk") or resources.get("disk_percent") or resources.get("diskUsage")
        deployments: list[dict[str, Any]] = []
        for key, uuid in self.app_uuids.items():
            if not uuid:
                continue
            for dep in self.list_deployments(uuid)[:5]:
                deployments.append(
                    {
                        "service_key": key,
                        "id": dep.get("uuid") or dep.get("id") or dep.get("deployment_uuid"),
                        "status": dep.get("status"),
                        "created_at": dep.get("created_at") or dep.get("started_at"),
                        "commit": dep.get("commit") or dep.get("git_commit_sha"),
                    }
                )
        return {
            "unavailable": [],
            "message": None,
            "containers": containers,
            "cpu_percent": _as_float(cpu),
            "ram_percent": _as_float(ram),
            "disk_percent": _as_float(disk),
            "network": resources.get("network"),
            "deployments": deployments,
            "raw_resources_keys": list(resources.keys()) if resources else [],
        }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for k in ("percent", "usage", "value", "usedPercent"):
            if k in value:
                return _as_float(value[k])
    try:
        return float(str(value).replace("%", "").strip())
    except Exception:
        return None


def get_coolify_client() -> CoolifyClient:
    return CoolifyClient()
