# vLLM Plugin Development Guide

This guide covers everything you need to know about developing plugins for vLLM, including best practices learned from the official documentation and community blog posts.

## Table of Contents

1. [Plugin System Overview](#plugin-system-overview)
2. [Types of Plugins](#types-of-plugins)
3. [How vLLM Discovers Plugins](#how-vllm-discovers-plugins)
4. [Creating Your First Plugin](#creating-your-first-plugin)
5. [Best Practices](#best-practices)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Plugin System Overview

vLLM's plugin system uses Python's standard `entry_points` mechanism to discover and load extensions. This enables:

- **Clean modifications**: Extend vLLM without forking the codebase
- **Runtime activation**: Plugins load automatically when vLLM starts
- **Distributed compatibility**: Plugins load in all processes (main, workers, etc.)
- **Selective loading**: Use environment variables to control which plugins activate

### Plugin Lifecycle

1. **Discovery**: vLLM reads entry points from installed packages
2. **Loading**: `load_general_plugins()` is called before initialization
3. **Registration**: Plugin functions register models, patches, or other extensions
4. **Execution**: Registered functionality is available throughout vLLM's runtime

**Important**: Plugin registration happens in **every vLLM process** including:
- Main process
- Worker processes
- GPU/CPU workers
- Auxiliary processes

This ensures consistent behavior across distributed deployments.

---

## Types of Plugins

vLLM supports several plugin entry point groups:

| Entry Point Group | Purpose | Registration Target |
|-------------------|---------|---------------------|
| `vllm.general_plugins` | General extensions, custom models, patches | Function that performs registration |
| `vllm.platform_plugins` | Hardware backend integrations | Function returning platform class if supported |
| `vllm.stat_logger_plugins` | Custom metrics/logging | Logger class (StatLoggerBase subclass) |
| `vllm.logits_processors` | Custom decoding strategies | LogitsProcessor subclass |
| `vllm.io_processor_plugins` | Input/output processing | IO processor implementation |

### General Plugins (`vllm.general_plugins`)

The most common plugin type. Use for:
- Registering custom model architectures
- Applying patches to vLLM classes
- Adding custom samplers or processors

```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_package.register:register"
```

### Platform Plugins (`vllm.platform_plugins`)

For hardware backend integrations (NPU, custom accelerators):

```toml
[project.entry-points."vllm.platform_plugins"]
my_platform = "my_package.platform:register"
```

Requires implementing:
- `Platform` class
- `WorkerBase`
- `ModelRunnerBase`
- `AttentionBackend`
- `CommunicatorBase`

### Stat Logger Plugins (`vllm.stat_logger_plugins`)

For custom metrics collection and export:

```toml
[project.entry-points."vllm.stat_logger_plugins"]
my_logger = "my_package.loggers:MyLoggerClass"
```

Note: Entry point should reference the class directly, not a registration function.

### Logits Processor Plugins (`vllm.logits_processors`)

**vLLM v1 Only** - For custom decoding strategies that modify logits before sampling.

```toml
[project.entry-points."vllm.logits_processors"]
my_decoder = "my_package.processor:MyLogitsProcessor"
```

**Important Characteristics:**
- **Global Application**: Plugins apply to ALL requests when installed
- **No Per-Request Selection**: vLLM v1 does NOT support per-request logits processor selection via the OpenAI API
- **One Plugin Per Deployment**: Install only ONE decoding strategy plugin per vLLM deployment
- **Must Inherit Base Class**: Your processor MUST inherit from `LogitsProcessor`

See [Logits Processor Plugins (vLLM v1)](#logits-processor-plugins-vllm-v1) for detailed implementation guide.

---

## How vLLM Discovers Plugins

vLLM uses Python's `importlib.metadata.entry_points()` to discover plugins:

```python
# Simplified discovery logic
from importlib.metadata import entry_points

eps = entry_points(group='vllm.general_plugins')
for ep in eps:
    register_func = ep.load()
    register_func()
```

### Environment Variable Control

- `VLLM_PLUGINS`: Comma-separated list of plugin names to load
  - If not set, all discovered plugins are loaded
  - Use to selectively enable plugins: `VLLM_PLUGINS=my_plugin,other_plugin`

---

## Creating Your First Plugin

### Step 1: Project Structure

```
my-vllm-plugin/
├── pyproject.toml
├── src/
│   └── my_vllm_plugin/
│       ├── __init__.py
│       └── register.py
└── tests/
    └── test_plugin.py
```

### Step 2: Define Entry Point

```toml
# pyproject.toml
[project]
name = "my-vllm-plugin"
version = "0.1.0"
dependencies = ["vllm>=0.8.0"]

[project.entry-points."vllm.general_plugins"]
my_plugin = "my_vllm_plugin.register:register"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

### Step 3: Implement Registration

```python
# src/my_vllm_plugin/register.py
import logging

logger = logging.getLogger(__name__)
_registered = False

def register() -> None:
    """Register plugin with vLLM."""
    global _registered

    # Ensure re-entrancy
    if _registered:
        return

    logger.info("Registering my vLLM plugin")

    # Your registration logic here
    # Example: Register a custom model
    from vllm import ModelRegistry
    if "MyModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyModel",
            "my_vllm_plugin.models:MyModelForCausalLM"
        )

    _registered = True
```

### Step 4: Install and Test

```bash
pip install -e .
python -c "import vllm; print('Plugin loaded!')"
```

---

## Best Practices

### 1. Re-Entrant Registration Functions

Your registration function **must** be safe to call multiple times:

```python
_registered = False

def register():
    global _registered
    if _registered:
        return  # Already registered

    # ... registration logic ...

    _registered = True
```

**Why?** vLLM may call your function in multiple processes.

### 2. Version Compatibility

Always specify and check vLLM version requirements:

```python
from packaging.version import Version
import vllm

def register():
    current = Version(vllm.__version__)
    required = Version("0.9.0")

    if current < required:
        logger.warning(f"Plugin requires vLLM >= 0.9.0, got {current}")
        return

    # ... registration logic ...
```

Or use the decorator pattern:

```python
@min_vllm_version("0.9.0")
class MyPatch(VLLMPatch[Scheduler]):
    pass
```

### 3. Minimal Patches

When patching vLLM classes:

- **Do**: Add single methods, override specific behavior
- **Don't**: Duplicate entire classes, make sweeping changes

```python
# Good: Minimal patch
class PriorityPatch(VLLMPatch[Scheduler]):
    def get_priority(self, request):
        return request.metadata.get("priority", 0)

# Bad: Reimplementing entire class
class MyScheduler(Scheduler):
    # ... hundreds of lines ...
```

### 4. Configuration via Environment Variables

Use environment variables for runtime configuration:

```python
import os

def register():
    enabled = os.environ.get("MY_PLUGIN_ENABLED", "true").lower() == "true"
    if not enabled:
        return

    # ... registration logic ...
```

### 5. Graceful Degradation

Handle missing dependencies gracefully:

```python
def register():
    try:
        from vllm import ModelRegistry
    except ImportError:
        logger.warning("vLLM not available, skipping registration")
        return

    # ... registration logic ...
```

### 6. Logging

Use Python's logging module for visibility:

```python
import logging

logger = logging.getLogger(__name__)

def register():
    logger.info("Starting plugin registration")
    # ...
    logger.info("Plugin registered successfully")
```

### 7. Testing

Always test:
- Re-entrancy (multiple calls)
- Without vLLM installed
- With different vLLM versions

```python
def test_register_is_reentrant():
    from my_plugin.register import register
    register()
    register()  # Should not raise

def test_handles_missing_vllm(monkeypatch):
    monkeypatch.setattr('builtins.__import__', mock_import_error)
    # Should not raise, just log warning
```

---

## Advanced Topics

### Surgical Patching with VLLMPatch

For modifying existing vLLM classes without forking:

```python
from vllm.core.scheduler import Scheduler

class PrioritySchedulerPatch(VLLMPatch[Scheduler]):
    """Add priority-based scheduling."""

    def get_priority(self, request) -> int:
        """New method added to Scheduler."""
        return request.metadata.get("priority", 0)

    def schedule(self, waiting_queue):
        """Override existing method."""
        sorted_queue = sorted(
            waiting_queue,
            key=lambda r: self.get_priority(r),
            reverse=True
        )
        # Call original via _original_ prefix
        return self._original_schedule(sorted_queue)

# Apply at registration time
PrioritySchedulerPatch.apply()
```

### Runtime Patch Control

Control patches via environment variables:

```bash
# Enable specific patches
VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch,CustomSamplerPatch python app.py

# Enable all patches
VLLM_CUSTOM_PATCHES=* python app.py
```

### Multi-Model Support

Different models can enable different patches:

```python
def register():
    model_type = os.environ.get("MODEL_TYPE", "")

    if model_type == "priority":
        PrioritySchedulerPatch.apply()
    elif model_type == "batch":
        BatchOptimizationPatch.apply()
```

### Docker Configuration

```dockerfile
FROM vllm/vllm-openai:latest

# Install plugins
COPY plugins/ /plugins/
RUN pip install /plugins/vllm-custom-models /plugins/vllm-patches

# Configure patches
ENV VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch
```

### Logits Processor Plugins (vLLM v1)

Logits processors modify the model's output logits before sampling. vLLM v1 has a specific interface that must be followed.

#### Required Interface

Your processor **MUST**:
1. **Inherit from `LogitsProcessor`** - Not just implement the methods
2. **Use the exact constructor signature**
3. **Implement all required methods**

```python
from typing import Optional
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

# TYPE_CHECKING imports to avoid circular dependencies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate


class MyLogitsProcessor(LogitsProcessor):
    """Custom logits processor for vLLM v1."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool
    ):
        """Initialize the processor.

        Args:
            vllm_config: vLLM configuration object
            device: Target device for tensors (cuda:0, cpu, etc.)
            is_pin_memory: Whether to use pinned memory for CPU tensors
        """
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.batch_size = 0

        # Load configuration from environment variables
        import os
        self.my_param = float(os.environ.get("MY_PROCESSOR_PARAM", "1.0"))

    def is_argmax_invariant(self) -> bool:
        """Return whether this processor preserves the argmax.

        Returns:
            True: Processor never changes which token has highest logit
                  (can be skipped during greedy/beam search)
            False: Processor may change the argmax
                   (must always be applied)
        """
        return False  # Most custom processors should return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        """Update internal state when batch composition changes.

        Called at the start of each engine step BEFORE apply().

        Args:
            batch_update: Contains info about added, removed, moved requests.
                         None if no changes to the batch.

        The BatchUpdate contains:
            - batch_size: Current number of requests
            - added: List of (index, SamplingParams, output_tok_ids, req_id)
            - removed: List of removed request indices
            - moved: List of (from_idx, to_idx, direction) for reordered requests
        """
        if batch_update:
            self.batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the logits processing.

        Args:
            logits: Tensor of shape (batch_size, vocab_size)

        Returns:
            Modified logits tensor with same shape
        """
        if logits.size(0) == 0:
            return logits

        # Your processing logic here
        modified_logits = logits / self.my_param
        return modified_logits

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Validate sampling parameters at request creation time.

        Args:
            sampling_params: The sampling parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate any custom parameters in sampling_params.extra_args
        pass
```

#### Common Mistakes

| Mistake | Error Message | Fix |
|---------|---------------|-----|
| Not inheriting from base class | `must be a subclass of LogitsProcessor` | Add `(LogitsProcessor)` to class definition |
| Missing `is_argmax_invariant()` | `has no attribute 'is_argmax_invariant'` | Add the method, return `False` |
| Missing `update_state()` | `has no attribute 'update_state'` | Add the method, track batch_size |
| Wrong constructor signature | Various init errors | Use `(vllm_config, device, is_pin_memory)` |
| Using `__call__` instead of `apply` | Processor not called | Rename to `apply()` |

#### Configuration via Environment Variables

Since processors are instantiated by vLLM (not by your code), you cannot pass custom constructor parameters. Use environment variables instead:

```python
import os

class MyProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        # Configuration from environment
        self.temperature = float(os.environ.get("MY_PROCESSOR_TEMP", "0.8"))
        self.threshold = float(os.environ.get("MY_PROCESSOR_THRESHOLD", "0.5"))
```

```bash
# Configure at runtime
MY_PROCESSOR_TEMP=0.7 MY_PROCESSOR_THRESHOLD=0.3 python -m vllm.entrypoints.openai.api_server ...
```

#### Per-Request State (Advanced)

If you need per-request configuration, use the `BatchUpdate` in `update_state()`:

```python
def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
    if not batch_update:
        return

    # Track per-request state
    for index, params, output_tokens, req_id in batch_update.added:
        # params.extra_args contains custom per-request parameters
        threshold = params.extra_args.get("my_threshold", 0.5) if params.extra_args else 0.5
        self.req_state[index] = {"threshold": threshold}

    for index in batch_update.removed:
        self.req_state.pop(index, None)
```

#### Entry Point Registration

```toml
# pyproject.toml
[project.entry-points."vllm.logits_processors"]
my_decoder = "my_package.processor:MyLogitsProcessor"
```

The entry point name (`my_decoder`) is used for identification but cannot be selected per-request in vLLM v1.

---

## Troubleshooting

### Plugin Not Loading

1. **Check installation**: `pip list | grep your-plugin`
2. **Check entry points**: `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.general_plugins')))"`
3. **Check VLLM_PLUGINS env var**: May be filtering your plugin

### Registration Errors

1. **Check logs**: Look for registration messages
2. **Test import**: `python -c "from your_plugin.register import register; register()"`
3. **Check vLLM version**: Ensure compatibility

### Patch Not Applied

1. **Check VLLM_CUSTOM_PATCHES**: Must include your patch name
2. **Check version decorator**: May be blocking due to version mismatch
3. **Check PatchManager**: `PatchManager.is_applied("YourPatch")`

### Distributed Issues

If plugin works locally but not in distributed mode:
1. Ensure re-entrancy
2. Check all workers have plugin installed
3. Verify environment variables propagate to workers

### Logits Processor Issues (vLLM v1)

#### "must be a subclass of LogitsProcessor"

Your class must inherit from the base class:

```python
# Wrong
class MyProcessor:
    ...

# Correct
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

class MyProcessor(LogitsProcessor):
    ...
```

#### "has no attribute 'is_argmax_invariant'"

Add the required method:

```python
def is_argmax_invariant(self) -> bool:
    return False
```

#### "has no attribute 'update_state'"

Add the required method:

```python
def update_state(self, batch_update):
    if batch_update:
        self.batch_size = batch_update.batch_size
```

#### Processor not being called

Ensure you're using `apply()` not `__call__`:

```python
# Wrong
def __call__(self, logits):
    ...

# Correct
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    ...
```

#### Per-request selection not working

vLLM v1 does NOT support per-request logits processor selection via the OpenAI API. Processors apply globally to all requests. To use different strategies:
- Deploy separate vLLM instances with different plugins
- Use different Docker images per strategy

---

## References

- [vLLM Plugin System Documentation](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [Building Clean vLLM Modifications (Blog)](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html)
- [vLLM Hardware Plugin Guide](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)
