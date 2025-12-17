"""Registry for available logits processors that can be hot-swapped."""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Registry for available logits processors that can be hot-swapped.

    Manages discovery, validation, and instantiation of processors.
    Supports:
    - Built-in processors from this repo
    - Dynamically discovered processors via entry points
    - Direct class registration
    """

    _processors: Dict[str, Type] = {}
    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry with available processors."""
        if cls._initialized:
            return

        cls._initialized = True

        # Register built-in processors
        cls._register_builtin_processors()

        # Discover additional processors from entry points
        cls._discover_entry_point_processors()

        logger.info(f"ProcessorRegistry initialized with {len(cls._processors)} processors")

    @classmethod
    def _register_builtin_processors(cls) -> None:
        """Register processors from this monorepo."""
        builtin = {
            "entropy": ("vllm_entropy_decoder.processor", "EntropyLogitsProcessor"),
            "cot": ("vllm_cot_decoder.processor", "CoTLogitsProcessor"),
        }

        for name, (module_path, class_name) in builtin.items():
            try:
                module = importlib.import_module(module_path)
                processor_class = getattr(module, class_name)
                cls.register(name, processor_class)
            except ImportError as e:
                logger.debug(f"Built-in processor '{name}' not available: {e}")
            except AttributeError as e:
                logger.warning(f"Processor class '{class_name}' not found in {module_path}: {e}")

    @classmethod
    def _discover_entry_point_processors(cls) -> None:
        """Discover processors via Python entry points."""
        try:
            from importlib.metadata import entry_points

            # Look for processors in a dedicated group for hot-swap
            all_eps = entry_points()
            # Python 3.10+ returns SelectableGroups with select() method
            # Python 3.9 returns a dict-like object
            if hasattr(all_eps, "select"):
                eps = list(all_eps.select(group="vllm_hotswap.processors"))
            else:
                # Python 3.9 compatibility
                eps = list(all_eps.get("vllm_hotswap.processors", []))  # type: ignore[union-attr]

            for ep in eps:
                try:
                    processor_class = ep.load()
                    cls.register(ep.name, processor_class)
                except Exception as e:
                    logger.warning(f"Failed to load processor '{ep.name}': {e}")
        except Exception as e:
            logger.debug(f"Entry point discovery failed: {e}")

    @classmethod
    def register(cls, name: str, processor_class: Type) -> None:
        """Register a processor class.

        Args:
            name: Unique name for the processor.
            processor_class: LogitsProcessor subclass.

        Raises:
            TypeError: If processor_class doesn't have required methods.
        """
        # Validate required methods
        required_methods = ["apply", "update_state", "is_argmax_invariant"]
        for method in required_methods:
            if not hasattr(processor_class, method):
                logger.warning(
                    f"Processor '{name}' missing method '{method}', registration may fail"
                )

        cls._processors[name] = processor_class
        logger.info(f"Registered processor: {name} -> {processor_class.__name__}")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a processor.

        Args:
            name: Processor name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in cls._processors:
            del cls._processors[name]
            logger.info(f"Unregistered processor: {name}")
            return True
        return False

    @classmethod
    def load(
        cls,
        name: str,
        vllm_config: "VllmConfig",
        device: "torch.device",
        is_pin_memory: bool,
    ) -> Any:
        """Load and instantiate a processor by name.

        Args:
            name: Registered processor name.
            vllm_config: vLLM configuration.
            device: Target device.
            is_pin_memory: Pinned memory flag.

        Returns:
            Instantiated processor.

        Raises:
            KeyError: If processor not found.
            Exception: If instantiation fails.
        """
        cls.initialize()

        if name not in cls._processors:
            raise KeyError(
                f"Processor '{name}' not found. Available: {list(cls._processors.keys())}"
            )

        processor_class = cls._processors[name]

        logger.info(f"Loading processor: {name} ({processor_class.__name__})")

        return processor_class(vllm_config, device, is_pin_memory)

    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """List all available processors.

        Returns:
            Dict mapping processor names to class names.
        """
        cls.initialize()
        return {name: pc.__name__ for name, pc in cls._processors.items()}

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a processor is available.

        Args:
            name: Processor name to check.

        Returns:
            True if available.
        """
        cls.initialize()
        return name in cls._processors

    @classmethod
    def get_class(cls, name: str) -> Optional[Type]:
        """Get the processor class without instantiating.

        Args:
            name: Processor name.

        Returns:
            Processor class or None.
        """
        cls.initialize()
        return cls._processors.get(name)

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        cls._processors.clear()
        cls._initialized = False
