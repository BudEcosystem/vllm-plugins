# vLLM Patches

Surgical patches for vLLM classes without maintaining a fork. This plugin provides a framework for applying targeted modifications to vLLM at runtime.

## Features

- **Type-safe patching**: Use `VLLMPatch[TargetClass]` for clear, typed patches
- **Version compatibility**: `@min_vllm_version` decorator ensures patches only apply to compatible versions
- **Runtime activation**: Control patches via `VLLM_CUSTOM_PATCHES` environment variable
- **Patch registry**: Track which patches are applied for debugging
- **Original preservation**: Access original methods via `_original_<method>` attributes

## Installation

```bash
pip install -e .
```

## Creating a Patch

1. Create a patch class in `src/vllm_patches/patches/`:

```python
# src/vllm_patches/patches/priority_scheduler.py
from vllm.core.scheduler import Scheduler
from vllm_patches import VLLMPatch, min_vllm_version

@min_vllm_version("0.9.0")
class PrioritySchedulerPatch(VLLMPatch[Scheduler]):
    """Add priority-based request scheduling."""

    def get_priority(self, request) -> int:
        """New method: extract priority from request metadata."""
        return request.metadata.get("priority", 0)

    def schedule(self, waiting_queue):
        """Override: sort by priority before scheduling."""
        sorted_queue = sorted(
            waiting_queue,
            key=lambda r: self.get_priority(r),
            reverse=True
        )
        # Call original implementation
        return self._original_schedule(sorted_queue)
```

2. Register the patch in `src/vllm_patches/register.py`:

```python
from vllm_patches.patches.priority_scheduler import PrioritySchedulerPatch

AVAILABLE_PATCHES = {
    "PrioritySchedulerPatch": PrioritySchedulerPatch,
}
```

## Usage

### Enable specific patches

```bash
VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch python app.py
```

### Enable multiple patches

```bash
VLLM_CUSTOM_PATCHES=PrioritySchedulerPatch,CustomSamplerPatch python app.py
```

### Enable all patches

```bash
VLLM_CUSTOM_PATCHES=* python app.py
```

### Programmatic application

```python
from vllm_patches.patches.priority_scheduler import PrioritySchedulerPatch

# Apply manually
PrioritySchedulerPatch.apply()
```

## Best Practices

1. **Keep patches minimal**: Only modify what's necessary
2. **Declare version requirements**: Always use `@min_vllm_version`
3. **Test across versions**: Verify patches work with different vLLM releases
4. **Document changes**: Clearly explain what each patch modifies
5. **Preserve originals**: Use `_original_*` methods when you need the original behavior

## Debugging

Check applied patches:

```python
from vllm_patches import PatchManager

# List all applied patches
print(PatchManager.get_applied_patches())

# Check if specific patch is applied
print(PatchManager.is_applied("PrioritySchedulerPatch"))
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
