# vLLM Entropy Decoder

Adaptive entropy-based decoding strategy for vLLM. This plugin dynamically adjusts sampling behavior based on model uncertainty metrics.

## Features

- **Entropy-aware sampling**: Adapts temperature and filtering based on model confidence
- **Varentropy tracking**: Monitors consistency of uncertainty across tokens
- **Automatic adaptation**: No manual tuning required during inference
- **vLLM native**: Integrates via the `vllm.logits_processors` plugin system

## Important: Global Behavior

This plugin applies to **ALL requests** when installed. vLLM v1 does not support per-request logits processor selection via the OpenAI API.

**Deployment pattern:**
- Install only ONE decoding strategy plugin per vLLM deployment
- To switch strategies, use different Docker images or server configurations
- The plugin is automatically loaded and applied at engine startup

## Installation

```bash
pip install -e .
```

## How It Works

The processor analyzes two key metrics at each decoding step:

| Metric | Description |
|--------|-------------|
| **Entropy** | Measures overall uncertainty in token distribution |
| **Varentropy** | Variance of entropy, indicating consistency of uncertainty |

Based on these metrics, the processor adapts:

| Entropy | Varentropy | Behavior |
|---------|------------|----------|
| Low | Low | **Sharpen** - Model is confident, reinforce top choices |
| High | Low | **Explore with temperature** - Consistently uncertain, allow diversity |
| Low | High | **Moderate adjustment** - Mixed signals, slight exploration |
| High | High | **Flatten** - Very uncertain, maximum exploration |

## Usage

Once installed, the processor is automatically applied to all requests:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model")

# The entropy decoder is automatically applied - no need to specify it
sampling_params = SamplingParams(
    temperature=0.8,
)

output = llm.generate("Your prompt here", sampling_params)
```

With Docker:
```bash
# Build image with entropy decoder
docker build -t vllm-entropy .

# Run - plugin applies to all requests automatically
docker run --gpus all -p 8000:8000 vllm-entropy
```

## Configuration

The processor accepts these parameters:

```python
EntropyLogitsProcessor(
    temperature=0.666,           # Base temperature
    top_p=0.90,                  # Reference nucleus sampling (not directly used)
    top_k=27,                    # Reference top-k (not directly used)
    min_p=0.03,                  # Minimum probability threshold
    entropy_threshold_low=0.1,   # Below this = high confidence
    entropy_threshold_high=3.0,  # Above this = high uncertainty
    varentropy_threshold=5.0,    # Above this = inconsistent uncertainty
)
```

## Limitations

- Not supported with speculative decoding enabled
- Not supported with pooling models
- Adds slight computational overhead for entropy calculations

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## References

Based on entropy-based adaptive sampling research. The core insight is that model uncertainty (measured via entropy) provides a signal for when to explore vs exploit during text generation.
