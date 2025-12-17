"""REST API routes for plugin management."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_router():
    """Create FastAPI router for plugin management.

    Returns:
        FastAPI APIRouter with all plugin management endpoints.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    try:
        from fastapi import APIRouter, BackgroundTasks, HTTPException
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not available. Install with: pip install vllm-dynamic-loader[api]")
        raise

    router = APIRouter(prefix="/plugins", tags=["plugins"])

    # ==================== Request/Response Models ====================

    class PluginResponse(BaseModel):
        id: str
        name: str
        source: str
        state: str
        version: Optional[str] = None
        entry_points: dict = {}
        error: Optional[str] = None

    class InstallFromPyPIRequest(BaseModel):
        package: str
        version: Optional[str] = None
        upgrade: bool = False
        auto_load: bool = True

    class InstallFromURLRequest(BaseModel):
        url: str
        upgrade: bool = False
        auto_load: bool = True

    class InstallFromGitRequest(BaseModel):
        repo_url: str
        branch: Optional[str] = None
        tag: Optional[str] = None
        commit: Optional[str] = None
        subdirectory: Optional[str] = None
        auto_load: bool = True

    class HotSwapRequest(BaseModel):
        processor: Optional[str] = None  # None for passthrough
        graceful: bool = True
        timeout_ms: int = 5000

    class StatusResponse(BaseModel):
        total_plugins: int
        loaded: int
        failed: int
        volume_watcher_active: bool
        watch_directory: str
        registry_path: str

    class ProcessorInfoResponse(BaseModel):
        delegate_type: Optional[str]
        batch_size: int
        requests_in_flight: int
        swap_in_progress: bool
        device: str

    # ==================== Helper Functions ====================

    def get_loader():
        """Get the dynamic plugin loader instance."""
        from vllm_dynamic_loader.register import get_loader

        return get_loader()

    # ==================== Endpoints ====================

    @router.get("/", response_model=List[PluginResponse])
    async def list_plugins():
        """List all registered plugins."""
        loader = get_loader()
        plugins = loader.list_plugins()
        return [
            PluginResponse(
                id=p["id"],
                name=p["name"],
                source=p["source"],
                state=p["state"],
                version=p.get("version"),
                entry_points=p.get("entry_points", {}),
                error=p.get("error"),
            )
            for p in plugins
        ]

    @router.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get overall loader status."""
        loader = get_loader()
        status = loader.get_status()
        return StatusResponse(
            total_plugins=status["total_plugins"],
            loaded=status["loaded"],
            failed=status["failed"],
            volume_watcher_active=status["volume_watcher_active"],
            watch_directory=status["watch_directory"],
            registry_path=status["registry_path"],
        )

    @router.get("/{plugin_id}", response_model=PluginResponse)
    async def get_plugin(plugin_id: str):
        """Get details of a specific plugin."""
        loader = get_loader()
        plugin = loader.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return PluginResponse(
            id=plugin["id"],
            name=plugin["name"],
            source=plugin["source"],
            state=plugin["state"],
            version=plugin.get("version"),
            entry_points=plugin.get("entry_points", {}),
            error=plugin.get("error"),
        )

    @router.post("/{plugin_id}/load")
    async def load_plugin(plugin_id: str, background_tasks: BackgroundTasks):
        """Load a discovered plugin."""
        loader = get_loader()
        plugin = loader.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        if plugin["state"] == "loaded":
            return {"status": "already_active", "message": f"Plugin {plugin_id} already loaded"}

        # Load in background to avoid blocking
        background_tasks.add_task(loader.load_plugin, plugin_id)
        return {"status": "loading", "message": f"Loading {plugin_id}"}

    @router.post("/{plugin_id}/unload")
    async def unload_plugin(plugin_id: str):
        """Unload an active plugin."""
        loader = get_loader()
        if loader.unload_plugin(plugin_id):
            return {"status": "unloaded", "message": f"Plugin {plugin_id} unloaded"}
        raise HTTPException(status_code=400, detail="Failed to unload plugin")

    @router.delete("/{plugin_id}")
    async def uninstall_plugin(plugin_id: str):
        """Uninstall a plugin."""
        loader = get_loader()
        if loader.uninstall(plugin_id):
            return {"status": "uninstalled", "message": f"Plugin {plugin_id} uninstalled"}
        raise HTTPException(status_code=400, detail="Failed to uninstall plugin")

    @router.post("/{plugin_id}/update")
    async def update_plugin(plugin_id: str, background_tasks: BackgroundTasks):
        """Update a plugin to the latest version."""
        loader = get_loader()
        plugin = loader.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Update in background
        background_tasks.add_task(loader.update, plugin_id)
        return {"status": "updating", "message": f"Updating {plugin_id}"}

    @router.post("/install/pypi", response_model=PluginResponse)
    async def install_from_pypi(request: InstallFromPyPIRequest):
        """Install a plugin from PyPI."""
        loader = get_loader()

        package_spec = request.package
        if request.version:
            package_spec = f"{request.package}=={request.version}"

        result = loader.install_from_pypi(package_spec, upgrade=request.upgrade)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Installation failed")

        if result.plugin_info:
            if request.auto_load:
                loader.load_plugin(result.plugin_info.id)
            return PluginResponse(
                id=result.plugin_info.id,
                name=result.plugin_info.name,
                source=result.plugin_info.source.value,
                state=result.plugin_info.state.value,
                version=result.plugin_info.version,
            )

        raise HTTPException(status_code=400, detail="No plugin found in package")

    @router.post("/install/url", response_model=PluginResponse)
    async def install_from_url(request: InstallFromURLRequest):
        """Install a plugin from a wheel URL."""
        loader = get_loader()

        result = loader.install_from_url(request.url, upgrade=request.upgrade)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Installation failed")

        if result.plugin_info:
            if request.auto_load:
                loader.load_plugin(result.plugin_info.id)
            return PluginResponse(
                id=result.plugin_info.id,
                name=result.plugin_info.name,
                source=result.plugin_info.source.value,
                state=result.plugin_info.state.value,
                version=result.plugin_info.version,
            )

        raise HTTPException(status_code=400, detail="No plugin found in package")

    @router.post("/install/git", response_model=PluginResponse)
    async def install_from_git(request: InstallFromGitRequest):
        """Install a plugin from a Git repository."""
        loader = get_loader()

        kwargs = {}
        if request.branch:
            kwargs["branch"] = request.branch
        if request.tag:
            kwargs["tag"] = request.tag
        if request.commit:
            kwargs["commit"] = request.commit

        # Build URL with subdirectory if specified
        url = request.repo_url
        if request.subdirectory:
            url = f"{url}#subdirectory={request.subdirectory}"

        result = loader.install_from_git(url, **kwargs)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Installation failed")

        if result.plugin_info:
            if request.auto_load:
                loader.load_plugin(result.plugin_info.id)
            return PluginResponse(
                id=result.plugin_info.id,
                name=result.plugin_info.name,
                source=result.plugin_info.source.value,
                state=result.plugin_info.state.value,
                version=result.plugin_info.version,
            )

        raise HTTPException(status_code=400, detail="No plugin found in repository")

    # ==================== Hot-Swap Endpoints ====================

    @router.post("/hot-swap")
    async def hot_swap_processor(request: HotSwapRequest):
        """Hot-swap the active logits processor.

        Send processor=null or omit to switch to passthrough mode.
        """
        from vllm_dynamic_loader.hotswap.coordinator import SwapCoordinator

        coordinator = SwapCoordinator()
        result = coordinator.request_swap(
            request.processor,
            timeout_seconds=request.timeout_ms / 1000.0,
        )

        return {
            "status": "swapped",
            "swap_id": result["swap_id"],
            "processor": result["processor"],
            "results": result["results"],
        }

    @router.get("/processors/available")
    async def list_available_processors():
        """List all available processors for hot-swapping."""
        from vllm_dynamic_loader.hotswap.processor_registry import ProcessorRegistry

        return {"processors": ProcessorRegistry.list_available()}

    @router.get("/processors/active", response_model=ProcessorInfoResponse)
    async def get_active_processor():
        """Get information about the currently active logits processor."""
        from vllm_dynamic_loader.hotswap.proxy import HotSwapProxyProcessor

        instances = HotSwapProxyProcessor.get_all_instances()
        if not instances:
            return ProcessorInfoResponse(
                delegate_type=None,
                batch_size=0,
                requests_in_flight=0,
                swap_in_progress=False,
                device="unknown",
            )

        # Return info from first instance (they should all be the same)
        info = instances[0].get_delegate_info()
        return ProcessorInfoResponse(**info)

    return router


def create_app():
    """Create a standalone FastAPI app for plugin management.

    This can be used for testing or as a standalone management API.

    Returns:
        FastAPI application.
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError("FastAPI required. Install with: pip install vllm-dynamic-loader[api]")

    app = FastAPI(
        title="vLLM Dynamic Plugin Loader",
        description="REST API for managing vLLM plugins at runtime",
        version="0.1.0",
    )

    router = create_router()
    app.include_router(router)

    return app
