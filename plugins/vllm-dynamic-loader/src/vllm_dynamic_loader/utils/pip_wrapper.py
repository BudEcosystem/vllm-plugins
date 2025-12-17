"""Safe pip operations wrapper."""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from packaging.version import Version, InvalidVersion

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


# Packages that should never be downgraded by plugin installations
PROTECTED_PACKAGES = [
    "vllm",
    "torch",
    "transformers",
    "numpy",
    "triton",
]


@dataclass
class PipResult:
    """Result of a pip operation."""

    success: bool
    package_name: Optional[str] = None
    version: Optional[str] = None
    install_path: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None
    downgrades_prevented: Dict[str, tuple] = field(default_factory=dict)  # pkg -> (before, after)


class PipWrapper:
    """Wrapper for safe pip operations."""

    def __init__(
        self,
        python_executable: Optional[str] = None,
        protected_packages: Optional[List[str]] = None,
    ):
        """Initialize the pip wrapper.

        Args:
            python_executable: Python executable to use (default: sys.executable).
            protected_packages: List of packages to protect from downgrades.
                               Defaults to PROTECTED_PACKAGES.
        """
        self.python = python_executable or sys.executable
        self.protected_packages = protected_packages or PROTECTED_PACKAGES

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

    def _snapshot_protected_versions(self) -> Dict[str, Optional[str]]:
        """Capture current versions of protected packages."""
        versions = {}
        for pkg in self.protected_packages:
            versions[pkg] = self._get_installed_version(pkg)
        return versions

    def _check_for_downgrades(
        self, before: Dict[str, Optional[str]]
    ) -> Dict[str, tuple]:
        """Check if any protected packages were downgraded.

        Returns:
            Dict mapping package name to (before_version, after_version) for downgrades.
        """
        downgrades = {}
        for pkg in self.protected_packages:
            before_ver = before.get(pkg)
            after_ver = self._get_installed_version(pkg)

            if before_ver and after_ver and before_ver != after_ver:
                try:
                    if Version(after_ver) < Version(before_ver):
                        downgrades[pkg] = (before_ver, after_ver)
                except InvalidVersion:
                    # Can't compare versions, skip
                    pass

        return downgrades

    def _restore_versions(self, versions: Dict[str, Optional[str]]) -> List[str]:
        """Restore packages to their original versions.

        Returns:
            List of packages that failed to restore.
        """
        failed = []
        for pkg, version in versions.items():
            if version:
                result = self._run_pip(["install", f"{pkg}=={version}"])
                if not result.success:
                    logger.error(f"Failed to restore {pkg} to {version}")
                    failed.append(pkg)
                else:
                    logger.info(f"Restored {pkg} to {version}")
        return failed

    def _parse_dry_run_downgrades(
        self, output: str, before_versions: Dict[str, Optional[str]]
    ) -> Dict[str, tuple]:
        """Parse pip dry-run output to detect downgrades.

        Args:
            output: Output from pip install --dry-run.
            before_versions: Current versions of protected packages.

        Returns:
            Dict mapping package name to (before_version, after_version) for downgrades.
        """
        import re

        downgrades = {}

        # Match lines like "Would install package-1.2.3" or "Collecting package==1.2.3"
        # Also match "package-1.2.3" in the summary line
        for pkg in self.protected_packages:
            before_ver = before_versions.get(pkg)
            if not before_ver:
                continue

            # Normalize package name for regex (handle - vs _)
            pkg_pattern = pkg.replace("-", "[-_]")

            # Look for version being installed
            patterns = [
                rf"Would install[^\n]*{pkg_pattern}-(\d+\.\d+[^\s,]*)",
                rf"Collecting {pkg_pattern}==([^\s]+)",
                rf"Downloading {pkg_pattern}-([^\s-]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    after_ver = match.group(1)
                    try:
                        if Version(after_ver) < Version(before_ver):
                            downgrades[pkg] = (before_ver, after_ver)
                    except InvalidVersion:
                        pass
                    break

        return downgrades

    def _safe_install(self, args: List[str]) -> PipResult:
        """Run pip install with protection against downgrades.

        Uses --dry-run first to detect downgrades BEFORE making any changes.

        Args:
            args: Arguments for pip install (without 'install' prefix).

        Returns:
            PipResult with operation outcome.
        """
        # Snapshot protected package versions
        before_versions = self._snapshot_protected_versions()

        # First, do a dry-run to check what would be installed
        dry_run_result = self._run_pip(["install", "--dry-run"] + args)

        if dry_run_result.success and dry_run_result.output:
            # Check for downgrades in dry-run output
            downgrades = self._parse_dry_run_downgrades(
                dry_run_result.output, before_versions
            )

            if downgrades:
                downgrade_msgs = [
                    f"{pkg}: {old} -> {new}" for pkg, (old, new) in downgrades.items()
                ]
                logger.error(
                    f"Plugin would downgrade protected packages: {', '.join(downgrade_msgs)}. "
                    "Installation blocked."
                )
                return PipResult(
                    success=False,
                    error=(
                        f"Installation blocked: would downgrade protected packages: "
                        f"{', '.join(downgrade_msgs)}"
                    ),
                    downgrades_prevented=downgrades,
                )

        # Safe to proceed with actual install
        result = self._run_pip(["install"] + args)

        # Double-check after install (in case dry-run missed something)
        if result.success:
            actual_downgrades = self._check_for_downgrades(before_versions)
            if actual_downgrades:
                # This shouldn't happen if dry-run worked, but handle it
                downgrade_msgs = [
                    f"{pkg}: {old} -> {new}"
                    for pkg, (old, new) in actual_downgrades.items()
                ]
                logger.error(
                    f"Unexpected downgrades after install: {', '.join(downgrade_msgs)}. "
                    "Attempting to restore..."
                )
                self._restore_versions(
                    {pkg: before_versions[pkg] for pkg in actual_downgrades}
                )
                result.success = False
                result.error = f"Unexpected downgrades: {', '.join(downgrade_msgs)}"
                result.downgrades_prevented = actual_downgrades

        return result

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
        args = []

        if upgrade:
            args.append("--upgrade")
        if force_reinstall:
            args.append("--force-reinstall")
        if extra_index_url:
            args.extend(["--extra-index-url", extra_index_url])

        args.append(package_spec)

        result = self._safe_install(args)

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
        args = []

        if upgrade:
            args.append("--upgrade")
        if force_reinstall:
            args.append("--force-reinstall")

        args.append(wheel_path)

        result = self._safe_install(args)

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

        args = []

        if editable:
            args.append("-e")

        args.append(path)

        result = self._safe_install(args)

        if result.success:
            result.install_path = path

            # CRITICAL: For editable installs at runtime, add to sys.path
            # so the module can be imported immediately without restart
            if editable:
                # Determine the correct path to add:
                # - For src/ layout: add the src/ directory
                # - For flat layout: add the root directory
                path_obj = Path(path)
                src_dir = path_obj / "src"
                if src_dir.is_dir():
                    import_path = str(src_dir)
                else:
                    import_path = path

                if import_path not in sys.path:
                    sys.path.insert(0, import_path)
                    logger.info(f"Added {import_path} to sys.path for runtime import")

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
