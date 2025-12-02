# vLLM Custom Loggers

Custom stat loggers for vLLM metrics and monitoring. This plugin provides implementations for exporting vLLM statistics to various backends.

## Installation

```bash
# Basic installation
pip install -e .

# With Prometheus support
pip install -e ".[prometheus]"

# With Datadog support
pip install -e ".[datadog]"
```

## Included Loggers

### JsonFileLogger

Writes vLLM statistics to JSON Lines files for later analysis.

**Configuration:**

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VLLM_JSON_LOG_DIR` | `./vllm_logs` | Directory for log files |
| `VLLM_JSON_LOG_INTERVAL` | `10` | Minimum seconds between writes |

**Output format:**
```json
{"timestamp": "2024-01-15T10:30:00", "stats": {"requests": 100, "latency_ms": 45.2}}
```

## Adding a Custom Logger

1. Create your logger class in `src/vllm_custom_loggers/loggers/`:

```python
# src/vllm_custom_loggers/loggers/my_logger.py
from typing import Dict, Any

class MyCustomLogger:
    """Custom logger implementation."""

    def __init__(self):
        # Initialize your logger
        pass

    def log(self, stats: Dict[str, Any]) -> None:
        """Called by vLLM with current statistics."""
        # Process stats here
        pass

    def info(self, key: str, value: Any) -> None:
        """Log a single metric."""
        self.log({key: value})

    def close(self) -> None:
        """Clean up resources."""
        pass
```

2. Register the logger in `pyproject.toml`:

```toml
[project.entry-points."vllm.stat_logger_plugins"]
my_logger = "vllm_custom_loggers.loggers.my_logger:MyCustomLogger"
```

## Usage

Once installed, custom loggers are automatically discovered by vLLM. Configure which logger to use via vLLM's configuration options.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
