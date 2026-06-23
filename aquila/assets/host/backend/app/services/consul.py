import os
from typing import Any
from urllib.parse import urlparse

import consul

from app.core.config import settings


class ConsulService:
    def __init__(self) -> None:
        parsed = urlparse(settings.consul_http_addr)
        host = parsed.hostname or "localhost"
        port = parsed.port or 47528
        # python-consul2 also reads CONSUL_HTTP_ADDR internally and chokes on
        # the http:// scheme prefix.  Hide it so the library uses our explicit
        # host/port instead.
        saved = os.environ.pop("CONSUL_HTTP_ADDR", None)
        try:
            self.client = consul.Consul(host=host, port=port)
        finally:
            if saved is not None:
                os.environ["CONSUL_HTTP_ADDR"] = saved

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

    def deregister_service(self, service_id: str) -> None:
        """Drop a service registration (used when removing a stale node)."""
        self.client.agent.service.deregister(service_id)


consul_service = ConsulService()
