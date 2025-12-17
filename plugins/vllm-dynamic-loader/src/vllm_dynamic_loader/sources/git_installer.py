"""Git repository installer for plugins."""

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

from vllm_dynamic_loader.core.registry import PluginInfo, PluginSource, PluginState
from vllm_dynamic_loader.sources.base import InstallResult, SourceHandler
from vllm_dynamic_loader.utils.pip_wrapper import PipWrapper

logger = logging.getLogger(__name__)


@dataclass
class GitRef:
    """Parsed git reference."""

    url: str
    branch: Optional[str] = None
    tag: Optional[str] = None
    commit: Optional[str] = None
    subdirectory: Optional[str] = None


def parse_git_url(url: str) -> GitRef:
    """Parse a git URL with optional branch/tag/commit specification.

    Formats supported:
        - https://github.com/user/repo.git
        - https://github.com/user/repo.git@branch
        - https://github.com/user/repo.git@tag
        - https://github.com/user/repo.git@commit
        - git+https://github.com/user/repo.git@branch#subdirectory=plugins/foo
    """
    # Handle git+https prefix
    if url.startswith("git+"):
        url = url[4:]

    subdirectory = None
    if "#subdirectory=" in url:
        url, subdirectory = url.split("#subdirectory=", 1)

    # Parse ref (branch/tag/commit)
    ref = None
    if "@" in url:
        url, ref = url.rsplit("@", 1)

    git_ref = GitRef(url=url, subdirectory=subdirectory)

    if ref:
        # Heuristics to determine ref type
        if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
            git_ref.commit = ref
        elif ref.startswith("v") and any(c.isdigit() for c in ref):
            git_ref.tag = ref
        else:
            git_ref.branch = ref

    return git_ref


class GitInstaller(SourceHandler):
    """Installs plugins from git repositories."""

    def __init__(
        self,
        on_install: Optional[Callable[[PluginInfo], None]] = None,
        on_uninstall: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[PluginInfo], None]] = None,
        clone_dir: Optional[str] = None,
        editable: bool = True,
    ):
        """Initialize the git installer.

        Args:
            on_install: Callback when a plugin is installed.
            on_uninstall: Callback when a plugin is uninstalled.
            on_update: Callback when a plugin is updated.
            clone_dir: Directory for cloning repos (temp if not specified).
            editable: Whether to install in editable mode (allows updates).
        """
        super().__init__(on_install, on_uninstall, on_update)

        self._pip = PipWrapper()
        self._clone_dir = Path(clone_dir) if clone_dir else None
        self._editable = editable
        self._installed: Dict[str, PluginInfo] = {}
        self._clone_paths: Dict[str, Path] = {}  # plugin_id -> clone path

    @property
    def source_type(self) -> PluginSource:
        return PluginSource.GIT

    def install(self, source: str, **kwargs) -> InstallResult:
        """Install a plugin from a git repository.

        Args:
            source: Git URL with optional ref specification.
            **kwargs:
                branch: Override branch
                tag: Override tag
                commit: Override commit
                editable: Override editable mode
                depth: Shallow clone depth (default: 1)

        Returns:
            InstallResult with installation outcome.
        """
        branch = kwargs.get("branch")
        tag = kwargs.get("tag")
        commit = kwargs.get("commit")
        editable = kwargs.get("editable", self._editable)
        depth = kwargs.get("depth", 1)

        try:
            git_ref = parse_git_url(source)

            # Override ref from kwargs if provided
            if branch:
                git_ref.branch = branch
                git_ref.tag = None
                git_ref.commit = None
            elif tag:
                git_ref.tag = tag
                git_ref.branch = None
                git_ref.commit = None
            elif commit:
                git_ref.commit = commit
                git_ref.branch = None
                git_ref.tag = None

            # Clone the repository
            clone_path = self._clone_repository(git_ref, depth)

            # Determine install path (may be subdirectory)
            install_path = clone_path
            if git_ref.subdirectory:
                install_path = clone_path / git_ref.subdirectory
                if not install_path.exists():
                    return InstallResult(
                        success=False, error=f"Subdirectory not found: {git_ref.subdirectory}"
                    )

            # Install the package
            result = self._pip.install_local(str(install_path), editable=editable)

            if result.success:
                plugin_id = f"git_{hashlib.md5(source.encode()).hexdigest()[:8]}"

                plugin_info = PluginInfo(
                    id=plugin_id,
                    name=result.package_name or clone_path.name,
                    source=PluginSource.GIT,
                    source_path=source,
                    install_path=str(install_path),
                    version=result.version,
                    state=PluginState.INSTALLED,
                )

                self._installed[plugin_id] = plugin_info
                self._clone_paths[plugin_id] = clone_path
                self._notify_install(plugin_info)

                return InstallResult(
                    success=True,
                    plugin_info=plugin_info,
                    package_name=result.package_name,
                    install_path=str(install_path),
                    version=result.version,
                )
            else:
                # Cleanup on failure
                shutil.rmtree(clone_path, ignore_errors=True)
                return InstallResult(success=False, error=result.error)

        except Exception as e:
            error_msg = f"Failed to install from git: {e}"
            logger.error(error_msg)
            return InstallResult(success=False, error=error_msg)

    def _clone_repository(self, git_ref: GitRef, depth: int = 1) -> Path:
        """Clone a git repository."""
        import tempfile

        # Determine clone directory
        if self._clone_dir:
            base_dir = self._clone_dir
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path(tempfile.mkdtemp(prefix="vllm_plugin_git_"))

        # Generate unique clone path
        url_hash = hashlib.md5(git_ref.url.encode()).hexdigest()[:8]
        repo_name = Path(urlparse(git_ref.url).path).stem
        clone_path = base_dir / f"{repo_name}_{url_hash}"

        # Remove existing clone if present
        if clone_path.exists():
            shutil.rmtree(clone_path)

        # Build clone command
        cmd = ["git", "clone"]

        if depth > 0:
            cmd.extend(["--depth", str(depth)])

        if git_ref.branch:
            cmd.extend(["--branch", git_ref.branch])
        elif git_ref.tag:
            cmd.extend(["--branch", git_ref.tag])

        cmd.extend([git_ref.url, str(clone_path)])

        logger.info(f"Cloning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

        # Checkout specific commit if specified
        if git_ref.commit:
            # Need full history for specific commit
            if depth > 0:
                subprocess.run(
                    ["git", "fetch", "--unshallow"],
                    cwd=clone_path,
                    capture_output=True,
                    timeout=300,
                )

            checkout_result = subprocess.run(
                ["git", "checkout", git_ref.commit],
                cwd=clone_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if checkout_result.returncode != 0:
                raise RuntimeError(f"Git checkout failed: {checkout_result.stderr}")

        logger.info(f"Cloned to {clone_path}")
        return clone_path

    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin and clean up the clone."""
        if plugin_id not in self._installed:
            logger.warning(f"Plugin {plugin_id} not found")
            return False

        plugin_info = self._installed[plugin_id]

        # Uninstall the package
        result = self._pip.uninstall(plugin_info.name)

        # Clean up clone directory
        if plugin_id in self._clone_paths:
            clone_path = self._clone_paths[plugin_id]
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            del self._clone_paths[plugin_id]

        if result.success:
            del self._installed[plugin_id]
            self._notify_uninstall(plugin_id)
            return True
        else:
            logger.error(f"Failed to uninstall {plugin_info.name}: {result.error}")
            return False

    def update(self, plugin_id: str) -> InstallResult:
        """Update a plugin by pulling latest changes."""
        if plugin_id not in self._installed:
            return InstallResult(success=False, error=f"Plugin {plugin_id} not found")

        if plugin_id not in self._clone_paths:
            return InstallResult(success=False, error="Clone path not found, reinstall required")

        plugin_info = self._installed[plugin_id]
        clone_path = self._clone_paths[plugin_id]

        try:
            # Pull latest changes
            logger.info(f"Pulling updates for {plugin_info.name}")
            result = subprocess.run(
                ["git", "pull"],
                cwd=clone_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return InstallResult(success=False, error=f"Git pull failed: {result.stderr}")

            # Reinstall to pick up changes
            install_path = (
                Path(plugin_info.install_path) if plugin_info.install_path else clone_path
            )
            pip_result = self._pip.install_local(str(install_path), editable=self._editable)

            if pip_result.success:
                plugin_info.version = pip_result.version
                self._notify_update(plugin_info)
                return InstallResult(
                    success=True,
                    plugin_info=plugin_info,
                    package_name=plugin_info.name,
                    version=pip_result.version,
                )
            else:
                return InstallResult(success=False, error=pip_result.error)

        except Exception as e:
            error_msg = f"Failed to update: {e}"
            logger.error(error_msg)
            return InstallResult(success=False, error=error_msg)

    def get_installed(self) -> Dict[str, PluginInfo]:
        """Get all installed plugins."""
        return self._installed.copy()

    def get_clone_path(self, plugin_id: str) -> Optional[Path]:
        """Get the clone path for a plugin."""
        return self._clone_paths.get(plugin_id)
