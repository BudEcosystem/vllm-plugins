"""Safe pip operations wrapper."""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


@dataclass
class PipResult:
    """Result of a pip operation."""

    success: bool
    package_name: Optional[str] = None
    version: Optional[str] = None
    install_path: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None


class PipWrapper:
    """Wrapper for safe pip operations."""

    def __init__(self, python_executable: Optional[str] = None):
        """Initialize the pip wrapper.

        Args:
            python_executable: Python executable to use (default: sys.executable).
        """
        self.python = python_executable or sys.executable

    def _run_pip(self, args: List[str], timeout: int = 600) -> PipResult:
        """Run a pip command.

        Args:
            args: Arguments to pass to pip.
            timeout: Command timeout in seconds.

        Returns:
            PipResult with operation outcome.
        """
        cmd = [self.python, "-m", "pip"] + args
        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                return PipResult(success=True, output=result.stdout)
            else:
                return PipResult(
                    success=False,
                    error=result.stderr or result.stdout,
                    output=result.stdout,
                )

        except subprocess.TimeoutExpired:
            return PipResult(success=False, error="Command timed out")
        except Exception as e:
            return PipResult(success=False, error=str(e))

    def install_package(
        self,
        package_spec: str,
        upgrade: bool = False,
        force_reinstall: bool = False,
        extra_index_url: Optional[str] = None,
    ) -> PipResult:
        """Install a package from PyPI.

        Args:
            package_spec: Package specification (e.g., "package>=1.0").
            upgrade: Upgrade if already installed.
            force_reinstall: Force reinstall.
            extra_index_url: Additional PyPI index URL.

        Returns:
            PipResult with installation outcome.
        """
        args = ["install"]

        if upgrade:
            args.append("--upgrade")
        if force_reinstall:
            args.append("--force-reinstall")
        if extra_index_url:
            args.extend(["--extra-index-url", extra_index_url])

        args.append(package_spec)

        result = self._run_pip(args)

        if result.success:
            # Extract package name and version
            package_name = (
                package_spec.split("[")[0].split("=")[0].split("<")[0].split(">")[0].split("!")[0]
            )
            result.package_name = package_name
            result.version = self._get_installed_version(package_name)

        return result

    def install_wheel(
        self,
        wheel_path: str,
        upgrade: bool = False,
        force_reinstall: bool = False,
    ) -> PipResult:
        """Install a wheel file.

        Args:
            wheel_path: Path to the wheel file.
            upgrade: Upgrade if already installed.
            force_reinstall: Force reinstall.

        Returns:
            PipResult with installation outcome.
        """
        args = ["install"]

        if upgrade:
            args.append("--upgrade")
        if force_reinstall:
            args.append("--force-reinstall")

        args.append(wheel_path)

        result = self._run_pip(args)

        if result.success:
            # Extract package name from wheel filename
            wheel_name = Path(wheel_path).stem
            # Wheel format: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}
            parts = wheel_name.split("-")
            if len(parts) >= 2:
                result.package_name = parts[0].replace("_", "-")
                result.version = parts[1]

        return result

    def install_local(self, path: str, editable: bool = False) -> PipResult:
        """Install a local package.

        Args:
            path: Path to the package directory.
            editable: Install in editable mode.

        Returns:
            PipResult with installation outcome.
        """
        import importlib

        args = ["install"]

        if editable:
            args.append("-e")

        args.append(path)

        result = self._run_pip(args)

        if result.success:
            result.install_path = path

            # CRITICAL: For editable installs at runtime, add to sys.path
            # so the module can be imported immediately without restart
            if editable and path not in sys.path:
                sys.path.insert(0, path)
                logger.info(f"Added {path} to sys.path for runtime import")

            # Invalidate import caches so new modules can be found
            importlib.invalidate_caches()

            # Try to get package name from pyproject.toml or setup.py
            result.package_name = self._extract_package_name(path)
            if result.package_name:
                result.version = self._get_installed_version(result.package_name)

        return result

    def uninstall(self, package_name: str) -> PipResult:
        """Uninstall a package.

        Args:
            package_name: Name of the package to uninstall.

        Returns:
            PipResult with uninstallation outcome.
        """
        args = ["uninstall", "-y", package_name]
        return self._run_pip(args)

    def _get_installed_version(self, package_name: str) -> Optional[str]:
        """Get the installed version of a package.

        Args:
            package_name: Package name to check.

        Returns:
            Version string if installed, None otherwise.
        """
        try:
            import importlib.metadata

            return importlib.metadata.version(package_name)
        except Exception:
            return None

    def _extract_package_name(self, path: str) -> Optional[str]:
        """Extract package name from a local package.

        Args:
            path: Path to the package directory.

        Returns:
            Package name if found, None otherwise.
        """
        path_obj = Path(path)

        # Try pyproject.toml
        pyproject = path_obj / "pyproject.toml"
        if pyproject.exists():
            try:
                # Use tomllib in Python 3.11+ or fallback
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib

                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                name: Optional[str] = data.get("project", {}).get("name")
                return name
            except Exception:
                pass

        # Try setup.py (parse for name=)
        setup_py = path_obj / "setup.py"
        if setup_py.exists():
            try:
                content = setup_py.read_text()
                for line in content.split("\n"):
                    if "name=" in line or "name =" in line:
                        # Extract quoted value
                        for quote in ['"', "'"]:
                            if quote in line:
                                start = line.index(quote) + 1
                                end = line.index(quote, start)
                                return line[start:end]
            except Exception:
                pass

        # Fallback to directory name
        return path_obj.name

    def is_installed(self, package_name: str) -> bool:
        """Check if a package is installed.

        Args:
            package_name: Package name to check.

        Returns:
            True if installed, False otherwise.
        """
        return self._get_installed_version(package_name) is not None
