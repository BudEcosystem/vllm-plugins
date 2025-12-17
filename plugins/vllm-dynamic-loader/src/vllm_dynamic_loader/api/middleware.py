"""ASGI Middleware for injecting plugin management routes into vLLM."""

import json
import logging
import os
import sys

# Configure logging for this module to match vLLM's log level
_log_level_str = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(_log_level)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_log_level)
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s [%(name)s] %(message)s", datefmt="%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class PluginAPIMiddleware:
    """ASGI Middleware that adds plugin management routes to vLLM's server.

    This middleware intercepts requests to /plugins/* and handles them
    before they reach vLLM's main application.

    Usage:
        The middleware is automatically applied when the dynamic loader
        registers with vLLM. It patches vLLM's ASGI application.
    """

    def __init__(self, app):
        """Initialize the middleware.

        Args:
            app: The wrapped ASGI application (vLLM's FastAPI app).
        """
        self.app = app
        self._fastapi_app = None
        self._router = None

    async def __call__(self, scope, receive, send):
        """Handle ASGI requests.

        Intercepts /plugins/* routes and handles them with our router.
        All other requests pass through to the wrapped application.
        """
        if scope["type"] == "http":
            path = scope.get("path", "")

            # Handle plugin management routes
            if path.startswith("/plugins"):
                await self._handle_plugin_request(scope, receive, send)
                return

        # Pass through to wrapped application
        await self.app(scope, receive, send)

    async def _handle_plugin_request(self, scope, receive, send):
        """Handle plugin management requests."""
        # Lazy-load FastAPI app to avoid import issues at startup
        if self._fastapi_app is None:
            self._fastapi_app = self._create_fastapi_app()

        if self._fastapi_app:
            await self._fastapi_app(scope, receive, send)
        else:
            # FastAPI not available, return error
            await self._send_error(
                send,
                503,
                "Plugin API not available. Install with: pip install vllm-dynamic-loader[api]",
            )

    def _create_fastapi_app(self):
        """Create a minimal FastAPI app for plugin routes."""
        try:
            from fastapi import FastAPI

            from vllm_dynamic_loader.api.routes import create_router

            app = FastAPI(
                title="vLLM Plugin Management",
                docs_url="/plugins/docs",
                openapi_url="/plugins/openapi.json",
            )
            router = create_router()
            app.include_router(router)

            logger.info("Plugin management API initialized")
            return app

        except ImportError as e:
            logger.warning(f"FastAPI not available for plugin API: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create plugin API: {e}")
            return None

    async def _send_error(self, send, status_code: int, message: str):
        """Send an error response."""
        body = json.dumps({"error": message}).encode()

        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def patch_vllm_app() -> bool:
    """Patch vLLM's ASGI application to include plugin management routes.

    This function attempts to wrap vLLM's FastAPI application with our
    middleware. It should be called during plugin registration.

    Returns:
        True if patching succeeded, False otherwise.
    """
    try:
        # Try to find and patch vLLM's app
        # vLLM typically creates app in vllm.entrypoints.openai.api_server

        logger.debug("Attempting to patch vLLM app with plugin middleware...")

        # Method 1: Try to patch the app factory
        try:
            logger.debug("Method 1: Trying to patch api_server.app...")
            from vllm.entrypoints.openai import api_server

            if hasattr(api_server, "app"):
                original_app = api_server.app
                api_server.app = PluginAPIMiddleware(original_app)
                logger.info("Patched vLLM app with plugin middleware (Method 1)")
                return True
            else:
                logger.debug("api_server.app not found")
        except ImportError as e:
            logger.debug(f"Method 1 failed - ImportError: {e}")
        except AttributeError as e:
            logger.debug(f"Method 1 failed - AttributeError: {e}")

        # Method 2: Try to find app in run_server
        try:
            logger.debug("Method 2: Trying to patch build_app function...")
            from vllm.entrypoints.openai.api_server import build_app

            original_build_app = build_app

            def patched_build_app(*args, **kwargs):
                app = original_build_app(*args, **kwargs)
                logger.info("Plugin middleware applied to vLLM app")
                return PluginAPIMiddleware(app)

            import vllm.entrypoints.openai.api_server as server_module

            server_module.build_app = patched_build_app
            logger.info("Patched vLLM build_app with plugin middleware (Method 2)")
            return True
        except ImportError as e:
            logger.debug(f"Method 2 failed - ImportError: {e}")
        except AttributeError as e:
            logger.debug(f"Method 2 failed - AttributeError: {e}")

        logger.warning(
            "Could not patch vLLM app. Plugin API will be available via standalone server."
        )
        return False

    except Exception as e:
        logger.error(f"Error patching vLLM app: {e}", exc_info=True)
        return False


def create_standalone_server(host: str = "0.0.0.0", port: int = 8001):
    """Create a standalone server for plugin management.

    Use this if patching vLLM's app fails or if you want a separate
    management interface.

    Args:
        host: Host to bind to.
        port: Port to bind to.

    Returns:
        Uvicorn server instance (not started).
    """
    try:
        import uvicorn

        from vllm_dynamic_loader.api.routes import create_app

        app = create_app()

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        logger.info(f"Standalone plugin API server configured on {host}:{port}")
        return server

    except ImportError as e:
        logger.error(f"Cannot create standalone server: {e}")
        raise


async def run_standalone_server_async(host: str = "0.0.0.0", port: int = 8001):
    """Run standalone server asynchronously.

    Args:
        host: Host to bind to.
        port: Port to bind to.
    """
    server = create_standalone_server(host, port)
    await server.serve()


def run_standalone_server(host: str = "0.0.0.0", port: int = 8001):
    """Run standalone server (blocking).

    Args:
        host: Host to bind to.
        port: Port to bind to.
    """
    try:
        import uvicorn

        from vllm_dynamic_loader.api.routes import create_app

        app = create_app()
        uvicorn.run(app, host=host, port=port)

    except ImportError as e:
        logger.error(f"Cannot run standalone server: {e}")
        raise
