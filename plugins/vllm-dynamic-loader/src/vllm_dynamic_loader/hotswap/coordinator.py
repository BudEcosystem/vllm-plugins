"""Coordinates hot-swap across multiple vLLM processes."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SwapCoordinator:
    """Coordinates hot-swap across multiple vLLM processes.

    Multi-Process Communication Strategy:
    Since vLLM worker processes are separate OS processes, we use
    file-based signaling for coordination:

    1. Control plane writes swap instruction to shared file
    2. Each process watches the file (via polling)
    3. Processes acknowledge completion via per-process status files
    4. Control plane waits for all acknowledgments
    """

    def __init__(
        self,
        swap_dir: Optional[str] = None,
        poll_interval_ms: int = 100,
    ):
        """Initialize the swap coordinator.

        Args:
            swap_dir: Directory for swap coordination files.
            poll_interval_ms: Polling interval for file changes.
        """
        from vllm_dynamic_loader.config import get_config

        config = get_config()

        if swap_dir:
            self.swap_dir = Path(swap_dir)
        else:
            # Use directory next to registry
            registry_path = Path(config.registry_path)
            self.swap_dir = registry_path.parent / "swap"

        self.swap_dir.mkdir(parents=True, exist_ok=True)

        self.poll_interval = poll_interval_ms / 1000.0
        self.process_id = os.getpid()

        # Coordination files
        self.swap_request_file = self.swap_dir / "swap_request.json"
        self.status_file = self.swap_dir / f"status_{self.process_id}.json"

        # Internal state
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        self._last_swap_id: Optional[str] = None
        self._swap_callback: Optional[Callable[[Optional[str]], None]] = None

        logger.info(f"SwapCoordinator initialized: dir={self.swap_dir}, pid={self.process_id}")

    def start_watching(self, callback: Callable[[Optional[str]], None]) -> None:
        """Start watching for swap requests.

        Args:
            callback: Function to call with processor name when swap requested.
                     None means switch to passthrough mode.
        """
        self._swap_callback = callback
        self._stop_watching.clear()

        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name=f"hotswap-watcher-{self.process_id}",
        )
        self._watch_thread.start()

        logger.info("Started watching for swap requests")

    def stop_watching(self) -> None:
        """Stop watching for swap requests."""
        self._stop_watching.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)
        logger.info("Stopped watching for swap requests")

    def _watch_loop(self) -> None:
        """Main watching loop - polls for swap requests."""
        while not self._stop_watching.is_set():
            try:
                self._check_for_swap_request()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")

            self._stop_watching.wait(self.poll_interval)

    def _check_for_swap_request(self) -> None:
        """Check if there's a pending swap request."""
        if not self.swap_request_file.exists():
            return

        try:
            with open(self.swap_request_file, "r") as f:
                request = json.load(f)

            swap_id = request.get("swap_id")

            # Skip if we've already processed this swap
            if swap_id == self._last_swap_id:
                return

            processor_name = request.get("processor")  # None for passthrough
            timestamp = request.get("timestamp", 0)

            # Check if request is too old (> 60 seconds)
            if time.time() - timestamp > 60:
                logger.warning(f"Stale swap request ignored: {swap_id}")
                self._last_swap_id = swap_id
                return

            logger.info(f"Swap request detected: id={swap_id}, processor={processor_name}")

            # Execute the swap
            if self._swap_callback:
                try:
                    self._swap_callback(processor_name)
                    self._report_status(swap_id, "success")
                except Exception as e:
                    logger.error(f"Swap failed: {e}")
                    self._report_status(swap_id, "error", str(e))

            self._last_swap_id = swap_id

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid swap request file: {e}")
        except Exception as e:
            logger.error(f"Error processing swap request: {e}")

    def _report_status(self, swap_id: str, status: str, error: Optional[str] = None) -> None:
        """Report swap status for this process."""
        status_data = {
            "swap_id": swap_id,
            "process_id": self.process_id,
            "status": status,
            "timestamp": time.time(),
        }
        if error:
            status_data["error"] = error

        with open(self.status_file, "w") as f:
            json.dump(status_data, f)

        logger.info(f"Reported status: {status}")

    # ==================== Control Plane Methods ====================

    def request_swap(
        self,
        processor_name: Optional[str],
        timeout_seconds: float = 30.0,
    ) -> Dict:
        """Request a swap across all processes.

        This is called by the control plane (e.g., REST API handler).

        Args:
            processor_name: Name of processor to swap to (None for passthrough).
            timeout_seconds: Maximum time to wait for all processes.

        Returns:
            Dict with swap results per process.
        """
        swap_id = str(uuid.uuid4())[:8]

        # Clear old status files
        for status_file in self.swap_dir.glob("status_*.json"):
            try:
                status_file.unlink()
            except Exception:
                pass

        # Write swap request
        request = {
            "swap_id": swap_id,
            "processor": processor_name,
            "timestamp": time.time(),
        }

        with open(self.swap_request_file, "w") as f:
            json.dump(request, f)

        logger.info(f"Initiated swap request: id={swap_id}, processor={processor_name}")

        # Wait for all processes to acknowledge
        results = self._wait_for_acknowledgments(swap_id, timeout_seconds)

        return {
            "swap_id": swap_id,
            "processor": processor_name,
            "results": results,
        }

    def _wait_for_acknowledgments(self, swap_id: str, timeout: float) -> Dict:
        """Wait for all processes to acknowledge the swap."""
        start = time.time()
        results = {}

        while time.time() - start < timeout:
            for status_file in self.swap_dir.glob("status_*.json"):
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)

                    if status.get("swap_id") == swap_id:
                        pid = status.get("process_id")
                        if pid not in results:
                            results[pid] = status
                            logger.info(f"Process {pid} acknowledged: {status.get('status')}")
                except Exception:
                    continue

            # Small delay before next check
            time.sleep(0.1)

        return results

    def get_current_swap_id(self) -> Optional[str]:
        """Get the current swap ID."""
        return self._last_swap_id

    def cleanup_old_files(self, max_age_seconds: float = 3600) -> int:
        """Clean up old coordination files.

        Args:
            max_age_seconds: Maximum age of files to keep.

        Returns:
            Number of files removed.
        """
        removed = 0
        cutoff = time.time() - max_age_seconds

        for status_file in self.swap_dir.glob("status_*.json"):
            try:
                if status_file.stat().st_mtime < cutoff:
                    status_file.unlink()
                    removed += 1
            except Exception:
                pass

        return removed
