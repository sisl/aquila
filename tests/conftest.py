"""Shared pytest configuration."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
