from urllib.parse import urlparse
import asyncio
import logging
import socket

import consul

from app.config import settings

logger = logging.getLogger("vllm-cluster-client")


def _resolve_advertise_ip() -> str:
    if settings.advertise_host:
        return settings.advertise_host

    parsed = urlparse(settings.consul_http_addr)
    consul_host = parsed.hostname
    if consul_host:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((consul_host, parsed.port or 47528))
                return sock.getsockname()[0]
        except OSError:
            pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return "127.0.0.1"


def register_node() -> None:
    parsed = urlparse(settings.consul_http_addr)
    host = parsed.hostname or "localhost"
    port = parsed.port or 47528
    client = consul.Consul(host=host, port=port)

    advertise = _resolve_advertise_ip()
    service_id = settings.node_name or socket.gethostname()

    client.agent.service.register(
        name="vllm-satellite",
        service_id=service_id,
        address=advertise,
        port=settings.port,
        check=consul.Check.http(
            f"http://{advertise}:{settings.port}/health",
            "15s",
            timeout="30s",
            deregister="5m",
        ),
    )


async def register_loop(interval_seconds: int = 10) -> None:
    consecutive_failures = 0
    while True:
        try:
            register_node()
            if consecutive_failures:
                logger.info("Re-registered with Consul after %d failure(s)", consecutive_failures)
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            # Log the first failure loudly, then back off the log volume so a
            # persistently unreachable Consul does not flood the journal.
            if consecutive_failures == 1 or consecutive_failures % 12 == 0:
                logger.warning(
                    "Consul registration failed (attempt %d): %s",
                    consecutive_failures,
                    exc,
                )
        await asyncio.sleep(interval_seconds)
