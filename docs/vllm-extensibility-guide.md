# vLLM Extensibility Guide: Plugins vs Pull Requests

This document provides a comprehensive reference for what can be extended in vLLM via the plugin system versus what requires a pull request to the core repository.

## Table of Contents

1. [Overview](#overview)
2. [Plugin-Extensible Modules](#plugin-extensible-modules)
3. [Modules Requiring Pull Requests](#modules-requiring-pull-requests)
4. [Plugin System Architecture](#plugin-system-architecture)
5. [Decision Matrix](#decision-matrix)
6. [Implementation Examples](#implementation-examples)

---

## Overview

vLLM provides a robust plugin system using Python's `entry_points` mechanism. This allows developers to extend vLLM's functionality without forking the codebase. However, not all components are designed for plugin-based extension—some require direct contributions via pull requests.

### Key Principle

> **Plugin System**: Extend vLLM with custom implementations that follow documented interfaces.
>
> **Pull Requests**: Modify core algorithms, add built-in features, or change fundamental architecture.

---

## Plugin-Extensible Modules

### 1. General Plugins (`vllm.general_plugins`)

**Purpose**: Register custom model architectures and general extensions.

**What You Can Do**:
- Register custom model architectures via `ModelRegistry.register_model()`
- Register custom operations using the `CustomOp` class
- Apply patches to existing vLLM classes

**Entry Point**:
```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_package.register:register"
```

**Example**:
```python
def register():
    from vllm import ModelRegistry

    if "MyCustomModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyCustomModel",
            "my_package.models:MyCustomModelForCausalLM"
        )
```

**Loading Behavior**: Loaded in ALL processes (main, engine core, workers).

---

### 2. Platform Plugins (`vllm.platform_plugins`)

**Purpose**: Add support for custom hardware platforms and accelerators.

This is the most comprehensive plugin type, enabling full hardware stack customization.

**What You Can Do**:

| Component | Description | Base Class/Interface |
|-----------|-------------|---------------------|
| **Hardware Platform** | New accelerators (beyond NVIDIA/AMD/Intel/TPU/CPU) | `vllm.platforms.interface.Platform` |
| **Worker Implementation** | Custom model execution workers | `vllm.v1.worker.worker_base.WorkerBase` |
| **Attention Backend** | Custom attention mechanisms | `vllm.attention.backends.abstract.AttentionBackend` |
| **Device Communicator** | Custom collective communication ops | `DeviceCommunicatorBase` |
| **Custom Operations** | PyTorch ops, C++/CUDA kernels | `CustomOp`, `vllm._custom_ops` |

**Entry Point**:
```toml
[project.entry-points."vllm.platform_plugins"]
my_platform = "my_package.platform:register"
```

**Example**:
```python
def register():
    # Return None if platform not supported, otherwise return class path
    return "my_package.platform:MyCustomPlatform"
```

**Platform Class Requirements**:
```python
class MyCustomPlatform(Platform):
    _enum = PlatformEnum.OOT  # Out-of-tree platform

    @property
    def device_type(self) -> str:
        return "my_device"

    @property
    def device_name(self) -> str:
        return "my_device"

    def check_and_update_config(self, vllm_config) -> None:
        # Set worker_cls and other configurations
        vllm_config.parallel_config.worker_cls = (
            "my_package.worker:MyWorker"
        )

    def get_attn_backend_cls(self) -> str:
        return "my_package.attention:MyAttentionBackend"

    def get_device_communicator_cls(self) -> str:
        return "my_package.communicator:MyDeviceCommunicator"
```

**Worker Class Required Methods**:
- `init_device()` - Set up the device
- `initialize_cache()` - Configure cache
- `load_model()` - Load model weights
- `get_kv_cache_spaces()` - Generate KV cache spaces
- `determine_available_memory()` - Profile memory usage
- `initialize_from_config()` - Allocate KV cache
- `execute_model()` - Run inference

**Optional Worker Methods**:
- `sleep()` / `wakeup()` - Power management
- `compile_or_warm_up_model()` - Graph mode support
- `take_draft_token_ids()` - Speculative decoding
- `add_lora()` / `remove_lora()` / `list_loras()` / `pin_lora()` - LoRA support
- `execute_dummy_batch()` - Data parallelism

**Loading Behavior**: Loaded when `current_platform` is called.

---

### 3. IO Processor Plugins (`vllm.io_processor_plugins`)

**Purpose**: Custom pre-processing and post-processing for pooling models.

**What You Can Do**:
- Convert custom inputs to model prompts
- Transform model outputs to custom formats
- Handle multimodal data (images, audio, video)
- Implement custom request validation

**Entry Point**:
```toml
[project.entry-points."vllm.io_processor_plugins"]
my_processor = "my_package.io:MyIOProcessor"
```

**Required Interface**:
```python
from vllm.plugins.io_processors.interface import IOProcessor

class MyIOProcessor(IOProcessor[MyInput, MyOutput]):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def pre_process(self, prompt: MyInput, request_id: str = None, **kwargs):
        # Convert custom input to model prompts
        return converted_prompts

    def post_process(self, model_output, request_id: str = None, **kwargs):
        # Transform model output
        return custom_output

    def parse_request(self, request: Any) -> MyInput:
        # Parse incoming request
        return parsed_input

    def output_to_response(self, plugin_output: MyOutput):
        # Convert to API response
        return response
```

**Loading Behavior**: Loaded in process0 only.

---

### 4. Stat Logger Plugins (`vllm.stat_logger_plugins`)

**Purpose**: Custom metrics collection and logging backends.

**What You Can Do**:
- Implement custom logging backends
- Connect to monitoring systems (Prometheus, Ray, custom systems)
- Track custom metrics

**Entry Point**:
```toml
[project.entry-points."vllm.stat_logger_plugins"]
my_logger = "my_package.loggers:MyStatLogger"
```

**Required Interface**:
```python
from vllm.v1.metrics.loggers import StatLoggerBase

class MyStatLogger(StatLoggerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log(self, stats: Stats) -> None:
        # Log metrics to your backend
        pass
```

**Loading Behavior**: Loaded in process0 only for async serving.

---

### 5. Logits Processor Plugins (`vllm.logits_processors`)

**Purpose**: Custom decoding strategies that modify logits before sampling.

**What You Can Do**:
- Implement custom temperature scaling
- Add entropy-based adaptive decoding
- Create confidence-weighted sampling
- Apply custom token filtering/biasing

**Entry Point**:
```toml
[project.entry-points."vllm.logits_processors"]
my_decoder = "my_package.processor:MyLogitsProcessor"
```

**Required Interface** (vLLM v1):
```python
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.batch_size = 0

    def is_argmax_invariant(self) -> bool:
        return False  # Most processors should return False

    def update_state(self, batch_update) -> None:
        if batch_update:
            self.batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Modify and return logits
        return modified_logits

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        pass
```

**Important Limitations**:
- **Global Application**: Plugins apply to ALL requests when installed
- **No Per-Request Selection**: vLLM v1 does NOT support per-request processor selection
- **One Plugin Per Deployment**: Install only ONE decoding strategy per server
- **Configuration via Environment Variables**: Since vLLM instantiates processors

**Loading Behavior**: Loaded globally when plugin is installed.

---

### 6. LoRA Resolvers (Built-in Plugin System)

**Purpose**: Dynamic loading of LoRA adapters from various sources.

**What You Can Do**:
- Load LoRA adapters from filesystem
- Load from cloud storage (S3, GCS, etc.)
- Implement custom adapter resolution logic

**Entry Point**:
```toml
[project.entry-points."vllm.general_plugins"]
my_lora_resolver = "my_package.lora:register_resolver"
```

**Environment Variables**:
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` - Enable runtime LoRA updates
- `VLLM_PLUGINS=lora_filesystem_resolver` - Enable built-in resolver
- `VLLM_LORA_RESOLVER_CACHE_DIR` - Set cache directory

---

### 7. Tool Parsers (Plugin Support)

**Purpose**: Custom tool calling parsers for function calling.

**What You Can Do**:
- Implement custom tool call parsing logic
- Support model-specific tool formats

**Command Line**:
```bash
--tool-parser-plugin my_package.parsers
--tool-call-parser my_parser_name
```

---

## Modules Requiring Pull Requests

The following components require direct contributions to the vLLM repository:

### 1. Core Engine Components

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| `LLMEngine` | Request orchestration | Core inference loop |
| `AsyncLLMEngine` | Async request handling | Fundamental architecture |
| `EngineCore` | Core engine logic | Central coordination |
| `Scheduler` | Request scheduling | Core algorithm |
| `ModelRunner` | Model execution | Core execution path |

### 2. Built-in Model Support

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| `_VLLM_MODELS` registry | Official model list | Documentation & testing guarantees |
| Model implementations | Built-in architectures | Official support & maintenance |
| Multimodal processors | Vision/audio handling | Core multimodal support |

**Note**: You CAN register custom models via plugins. Adding to `_VLLM_MODELS` requires a PR for official support.

### 3. Attention Backend Registry

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| Built-in attention backends | FlashAttention, PagedAttention, etc. | Core performance critical |
| `AttentionBackendEnum` | Backend enumeration | Registration mechanism |
| Backend selection logic | Auto-detection logic | Core functionality |

**Note**: You CAN add custom attention via platform plugins. Adding to built-in backends requires a PR.

### 4. Configuration System

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| `VllmConfig` | Central configuration | All components depend on it |
| `SamplingParams` | Sampling configuration | API contract |
| Config validation | Input validation | Safety guarantees |

### 5. Core Sampling Logic

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| `Sampler` class | Token sampling | Core generation logic |
| Built-in sampling methods | Top-k, top-p, beam search | Algorithmic correctness |
| Speculative decoding core | Draft/verify mechanism | Performance critical |

### 6. API Endpoints

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| OpenAI-compatible API | `/v1/completions`, etc. | API compatibility |
| Request/Response schemas | API contracts | Stability guarantees |
| Streaming implementation | SSE handling | Core functionality |

### 7. Distributed Execution

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| Tensor parallelism | Model sharding | Core distributed logic |
| Pipeline parallelism | Stage execution | Core distributed logic |
| Ray integration | Cluster management | Infrastructure |

### 8. KV Cache Management

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| Block manager | Memory allocation | Core memory management |
| Cache eviction policies | Memory pressure handling | Performance critical |
| Prefix caching core | Shared prefix logic | Core optimization |

### 9. Reasoning Parsers

| Component | Description | Why PR Required |
|-----------|-------------|-----------------|
| Built-in parsers | DeepSeek, Qwen3, etc. | Model-specific logic |
| `ReasoningParserManager` | Parser registry | Registration mechanism |

---

## Plugin System Architecture

### Plugin Loading Flow

```
vLLM Startup
    │
    ├─► load_plugins_by_group("vllm.general_plugins")
    │       └─► Loaded in ALL processes
    │
    ├─► load_plugins_by_group("vllm.platform_plugins")
    │       └─► Loaded when current_platform accessed
    │
    ├─► load_plugins_by_group("vllm.io_processor_plugins")
    │       └─► Loaded in process0 only
    │
    ├─► load_plugins_by_group("vllm.stat_logger_plugins")
    │       └─► Loaded in process0 for async serving
    │
    └─► load_plugins_by_group("vllm.logits_processors")
            └─► Loaded globally when installed
```

### Environment Variable Control

| Variable | Purpose |
|----------|---------|
| `VLLM_PLUGINS` | Comma-separated list of plugins to load (empty = all) |
| `VLLM_CUSTOM_PATCHES` | Control which patches apply (`*` for all) |
| `VLLM_ATTENTION_BACKEND` | Select attention backend |

---

## Decision Matrix

Use this matrix to determine whether to use a plugin or submit a PR:

| Scenario | Plugin | PR | Notes |
|----------|--------|-----|-------|
| Custom model architecture | ✅ | | Use `ModelRegistry.register_model()` |
| New hardware platform | ✅ | | Platform plugin with full stack |
| Custom attention for new hardware | ✅ | | Via platform plugin |
| Custom attention algorithm for CUDA | | ✅ | Core performance path |
| Custom logging backend | ✅ | | Stat logger plugin |
| Custom decoding strategy | ✅ | | Logits processor plugin |
| Modify scheduler algorithm | | ✅ | Core scheduling logic |
| Add new sampling method | | ✅ | Core sampling logic |
| Custom I/O processing | ✅ | | IO processor plugin |
| LoRA resolver | ✅ | | General plugin |
| Tool parser | ✅ | | Via `--tool-parser-plugin` |
| New API endpoint | | ✅ | API surface |
| Config field addition | | ✅ | Central configuration |
| Built-in model support | | ✅ | Official model list |
| KV cache algorithm | | ✅ | Core memory management |
| Distributed execution | | ✅ | Core infrastructure |

---

## Implementation Examples

### Example 1: Custom Model via Plugin

```python
# my_plugin/register.py
_registered = False

def register():
    global _registered
    if _registered:
        return

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MyCustomLlama",
        "my_plugin.models:MyCustomLlamaForCausalLM"
    )

    _registered = True
```

### Example 2: Custom Attention via Platform Plugin

```python
# my_platform/register.py
def register():
    import torch
    if not torch.cuda.is_available():
        return None
    return "my_platform.platform:MyCustomPlatform"

# my_platform/platform.py
class MyCustomPlatform(Platform):
    _enum = PlatformEnum.OOT

    def get_attn_backend_cls(self) -> str:
        return "my_platform.attention:MyCustomAttention"
```

### Example 3: Custom Logits Processor

```python
# my_decoder/processor.py
import os
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

class EntropyAdaptiveProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.threshold = float(os.environ.get("ENTROPY_THRESHOLD", "2.0"))
        self.batch_size = 0

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        if batch_update:
            self.batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Adaptive temperature based on entropy
        scale = torch.where(
            entropy > self.threshold,
            torch.ones_like(entropy) * 0.8,
            torch.ones_like(entropy) * 1.2
        )

        return logits * scale.unsqueeze(-1)
```

---

## Summary

### Use Plugins When:
- Adding custom model architectures
- Supporting new hardware platforms
- Implementing custom logging/metrics
- Creating custom decoding strategies
- Adding I/O processing logic
- Implementing LoRA resolvers

### Submit PRs When:
- Modifying core engine components
- Adding built-in model support
- Changing sampling algorithms
- Adding API endpoints
- Modifying distributed execution
- Changing KV cache management

---

## References

- [vLLM Plugin System Documentation](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [vLLM Hardware Plugin Guide](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)
- [Building Clean vLLM Modifications](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html)
- [vLLM Contributing Guide](https://docs.vllm.ai/en/latest/contributing/README.html)
