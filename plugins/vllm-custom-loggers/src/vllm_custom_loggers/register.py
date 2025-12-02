"""Plugin registration for custom loggers.

Note: Stat logger plugins work differently from general plugins.
The entry point should directly reference the logger class, not a
registration function. vLLM will instantiate the class directly.

Entry point format in pyproject.toml:
    [project.entry-points."vllm.stat_logger_plugins"]
    my_logger = "my_package.module:MyLoggerClass"

The class should be compatible with vLLM's StatLoggerBase interface.
"""

# Re-export logger classes for entry point access
from vllm_custom_loggers.loggers.json_file import JsonFileLogger

__all__ = ["JsonFileLogger"]
