"""Custom patches for vLLM.

Add your patch implementations here. Each patch should:
1. Inherit from VLLMPatch[TargetClass]
2. Use @min_vllm_version decorator for version safety
3. Be registered in ../register.py AVAILABLE_PATCHES

Example patch:

```python
from vllm.core.scheduler import Scheduler
from vllm_patches import VLLMPatch, min_vllm_version

@min_vllm_version("0.9.0")
class MySchedulerPatch(VLLMPatch[Scheduler]):
    '''Custom scheduler modifications.'''

    def my_custom_method(self):
        return "custom behavior"
```
"""
