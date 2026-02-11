from typing import Any
from urllib.parse import urlparse

import consul

from app.core.config import settings


class ConsulService:
    def __init__(self) -> None:
        parsed = urlparse(settings.consul_http_addr)
        host = parsed.hostname or "localhost"
        port = parsed.port or 47528
        self.client = consul.Consul(host=host, port=port)

    def list_nodes(self) -> list[dict[str, Any]]:
        _, nodes = self.client.catalog.nodes()
        return nodes or []

    def list_service(self, service_name: str) -> list[dict[str, Any]]:
        _, services = self.client.catalog.service(service_name)
        return services or []

    def service_health(self, service_name: str) -> dict[str, list[str]]:
        _, services = self.client.health.service(service_name, passing=False)
        results: dict[str, list[str]] = {}
        for service in services or []:
            service_id = service.get("Service", {}).get("ID")
            checks = [
                check.get("Status")
                for check in service.get("Checks", [])
                if check.get("Status")
            ]
            if service_id:
                results[service_id] = checks
        return results

    def list_services(self) -> dict[str, list[str]]:
        _, services = self.client.catalog.services()
        return services or {}


consul_service = ConsulService()
