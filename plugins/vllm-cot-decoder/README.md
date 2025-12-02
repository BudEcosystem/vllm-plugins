# vLLM CoT Decoder

Confidence-weighted Chain-of-Thought decoding strategy for vLLM. This plugin adapts sampling behavior based on the model's confidence margin between top predictions.

## Features

- **Confidence-aware sampling**: Adjusts distribution based on prediction certainty
- **Adaptive temperature**: Sharpens when confident, flattens when uncertain
- **CoT-inspired**: Captures the scoring essence of Chain-of-Thought decoding
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

The processor analyzes the **confidence margin** at each decoding step:

```
confidence_margin = P(top_token) - P(second_token)
```

| Confidence | Behavior |
|------------|----------|
| High (> threshold) | **Sharpen** - Boost top token, trust the model |
| Low (< threshold) | **Flatten** - Encourage exploration of alternatives |
| Very Low | **Maximum exploration** - Model is guessing |

This approach captures the core insight from CoT-decoding research: confident
predictions are more likely correct, so we should amplify them; uncertain
predictions benefit from exploring alternatives.

## Usage

Once installed, the processor is automatically applied to all requests:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model")

# The CoT decoder is automatically applied - no need to specify it
sampling_params = SamplingParams(
    temperature=0.8,
)

output = llm.generate("Your prompt here", sampling_params)
```

With Docker:
```bash
# Build image with CoT decoder
docker build -t vllm-cot .

# Run - plugin applies to all requests automatically
docker run --gpus all -p 8000:8000 vllm-cot
```

## Configuration

The processor accepts these parameters:

```python
CoTLogitsProcessor(
    confidence_threshold=0.3,    # Margin above this = high confidence
    sharpening_factor=1.5,       # How much to boost confident predictions
    exploration_factor=0.8,      # How much to flatten uncertain predictions
    min_confidence=0.05,         # Below this = maximum exploration
)
```

### Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `confidence_threshold` | 0.0-1.0 | Higher = more strict definition of "confident" |
| `sharpening_factor` | > 1.0 | Higher = more aggressive sharpening when confident |
| `exploration_factor` | 0.0-1.0 | Lower = more flattening when uncertain |

## Comparison with Full CoT Decoding

Traditional CoT decoding explores k parallel reasoning paths and scores them.
This plugin captures the **confidence scoring** aspect without the computational
cost of multiple generations:

| Aspect | Full CoT | This Plugin |
|--------|----------|-------------|
| Path exploration | k parallel paths | Single path |
| Scoring | Full sequence confidence | Per-token confidence |
| Compute cost | k × generation cost | ~1.1× generation cost |
| Best for | Complex reasoning | General generation |

## Limitations

- Not supported with speculative decoding enabled
- Not supported with pooling models
- Single-path (doesn't explore multiple reasoning branches)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## References

Inspired by Chain-of-Thought decoding research, which demonstrates that
confidence margins between top predictions correlate with reasoning quality.
