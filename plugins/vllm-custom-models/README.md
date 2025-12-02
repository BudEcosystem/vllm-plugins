# vLLM Custom Models

Custom model implementations for vLLM, extending support beyond built-in architectures.

## Installation

```bash
pip install -e .
```

## Adding a Custom Model

1. Create your model class in `src/vllm_custom_models/models/`:

```python
# src/vllm_custom_models/models/my_model.py
from vllm.model_executor.models import LlamaForCausalLM

class MyCustomLlamaForCausalLM(LlamaForCausalLM):
    """Custom Llama variant."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Custom initialization
```

2. Register the model in `src/vllm_custom_models/register.py`:

```python
MODEL_REGISTRY = {
    "MyCustomLlama": "vllm_custom_models.models.my_model:MyCustomLlamaForCausalLM",
}
```

3. Ensure your model's `config.json` has the matching architecture:

```json
{
    "architectures": ["MyCustomLlama"],
    ...
}
```

## Usage

After installation, your custom models are automatically available in vLLM:

```python
from vllm import LLM

# Load your custom model
llm = LLM(model="path/to/your/model")
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
