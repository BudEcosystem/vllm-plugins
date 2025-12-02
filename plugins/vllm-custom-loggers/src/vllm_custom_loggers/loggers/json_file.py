"""JSON file logger - writes vLLM stats to JSON files.

This logger writes metrics to JSON files for later analysis or ingestion
by external monitoring systems.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JsonFileLogger:
    """Stat logger that writes metrics to JSON files.

    This is a custom stat logger implementation that can be registered
    with vLLM via the stat_logger_plugins entry point.

    Configuration via environment variables:
        VLLM_JSON_LOG_DIR: Directory for log files (default: ./vllm_logs)
        VLLM_JSON_LOG_INTERVAL: Minimum seconds between writes (default: 10)

    Output format:
        Each line is a JSON object with timestamp and metrics.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: float = 10.0,
    ):
        """Initialize the JSON file logger.

        Args:
            log_dir: Directory to write log files. Defaults to VLLM_JSON_LOG_DIR
                     env var or ./vllm_logs.
            log_interval: Minimum seconds between log writes.
        """
        self.log_dir = Path(
            log_dir if log_dir is not None else os.environ.get("VLLM_JSON_LOG_DIR", "./vllm_logs")
        )
        self.log_interval = float(os.environ.get("VLLM_JSON_LOG_INTERVAL", log_interval))

        self._last_log_time: float = 0
        self._log_file: Optional[Path] = None
        self._buffer: list[dict[str, Any]] = []

        self._ensure_log_dir()
        self._init_log_file()

        logger.info(f"JsonFileLogger initialized: {self._log_file}")

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _init_log_file(self) -> None:
        """Initialize a new log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self.log_dir / f"vllm_stats_{timestamp}.jsonl"

    def log(self, stats: dict[str, Any]) -> None:
        """Log stats to the JSON file.

        This method is called by vLLM's stat logging system with
        current metrics.

        Args:
            stats: Dictionary of metric names to values.
        """
        import time

        current_time = time.time()

        # Rate limit logging
        if current_time - self._last_log_time < self.log_interval:
            self._buffer.append(stats)
            return

        self._last_log_time = current_time

        # Prepare log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }

        # Include buffered entries if any
        if self._buffer:
            entry["buffered_count"] = len(self._buffer)  # type: ignore[assignment]
            self._buffer.clear()

        # Write to file
        try:
            assert self._log_file is not None
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write stats to {self._log_file}: {e}")

    def info(self, key: str, value: Any) -> None:
        """Log an informational metric.

        Args:
            key: Metric name.
            value: Metric value.
        """
        self.log({key: value})

    def close(self) -> None:
        """Clean up resources and flush any remaining buffered data."""
        if self._buffer:
            # Flush remaining buffer
            entry = {
                "timestamp": datetime.now().isoformat(),
                "stats": {"final_flush": True},
                "buffered_entries": self._buffer,
            }
            try:
                assert self._log_file is not None
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to flush buffer: {e}")
            finally:
                self._buffer.clear()

        logger.info(f"JsonFileLogger closed: {self._log_file}")
