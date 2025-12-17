"""REST API for plugin management."""

from vllm_dynamic_loader.api.routes import create_app, create_router

__all__ = [
    "create_router",
    "create_app",
]
