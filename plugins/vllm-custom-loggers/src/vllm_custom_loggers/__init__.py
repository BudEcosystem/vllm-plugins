"""vLLM Custom Loggers - Custom stat loggers for metrics and monitoring.

This plugin provides custom implementations of vLLM's StatLoggerBase for
exporting metrics to various backends (files, external services, etc.).

Stat logger plugins use the entry point group: vllm.stat_logger_plugins
"""

__version__ = "0.1.0"

from vllm_custom_loggers.loggers.json_file import JsonFileLogger

__all__ = [
    "JsonFileLogger",
]
