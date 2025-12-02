"""Tests for JSON file logger."""

import json
import tempfile
from pathlib import Path

from vllm_custom_loggers.loggers.json_file import JsonFileLogger


class TestJsonFileLogger:
    def test_creates_log_directory(self):
        """Test that logger creates log directory if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            JsonFileLogger(log_dir=str(log_dir))

            assert log_dir.exists()

    def test_creates_log_file(self):
        """Test that logger creates a log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonFileLogger(log_dir=tmpdir)

            assert logger._log_file is not None
            assert logger._log_file.parent == Path(tmpdir)
            assert logger._log_file.suffix == ".jsonl"

    def test_logs_stats(self):
        """Test that stats are written to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonFileLogger(log_dir=tmpdir, log_interval=0)

            logger.log({"requests": 100, "latency_ms": 50.5})
            logger.close()

            # Read the log file
            with open(logger._log_file) as f:
                lines = f.readlines()

            assert len(lines) >= 1
            entry = json.loads(lines[0])
            assert "timestamp" in entry
            assert "stats" in entry

    def test_info_method(self):
        """Test the info convenience method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonFileLogger(log_dir=tmpdir, log_interval=0)

            logger.info("test_metric", 42)
            logger.close()

            with open(logger._log_file) as f:
                lines = f.readlines()

            assert len(lines) >= 1

    def test_close_flushes_buffer(self):
        """Test that close flushes buffered entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # High interval to ensure buffering
            logger = JsonFileLogger(log_dir=tmpdir, log_interval=3600)

            # These should be buffered
            logger.log({"metric1": 1})
            logger.log({"metric2": 2})

            # This should flush everything
            logger.close()

            with open(logger._log_file) as f:
                content = f.read()

            assert "final_flush" in content or "buffered_entries" in content
