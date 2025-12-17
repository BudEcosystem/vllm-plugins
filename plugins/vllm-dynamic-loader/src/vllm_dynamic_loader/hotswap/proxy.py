"""Hot-swappable proxy processor for vLLM logits processors."""

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    from vllm_dynamic_loader.hotswap.coordinator import SwapCoordinator

logger = logging.getLogger(__name__)


class HotSwapProxyProcessor:
    """Proxy processor that enables hot-swapping of delegate processors.

    This processor is registered as the vLLM entry point and delegates
    all operations to a dynamically-loaded processor that can be swapped
    at runtime without restarting the service.

    Thread Safety:
        - Uses RLock for all delegate access
        - Atomic swap via swap_delegate()
        - Read operations use lock for consistency

    Multi-Process Coordination:
        - Each process has its own instance
        - Coordination via SwapCoordinator (file-based signaling)
    """

    # Class-level tracking for instances
    _instances: list = []
    _instance_lock = threading.Lock()

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        """Initialize the proxy processor.

        Args:
            vllm_config: vLLM configuration (passed to delegates).
            device: Target device for tensors.
            is_pin_memory: Whether to use pinned memory.
        """
        self._vllm_config = vllm_config
        self._device = device
        self._is_pin_memory = is_pin_memory

        # Delegate management
        self._delegate = None
        self._delegate_lock = threading.RLock()

        # State tracking
        self._batch_size = 0
        self._last_batch_update = None

        # Swap state tracking
        self._swap_in_progress = threading.Event()
        self._requests_in_flight = 0
        self._requests_lock = threading.Lock()

        # Coordinator for multi-process sync
        self._coordinator: Optional["SwapCoordinator"] = None

        # Register this instance
        with HotSwapProxyProcessor._instance_lock:
            HotSwapProxyProcessor._instances.append(self)

        # Initialize with default processor from environment
        self._initialize_default_delegate()

        # Start swap coordinator
        self._start_coordinator()

        logger.info(
            f"HotSwapProxyProcessor initialized on {device}, "
            f"delegate={type(self._delegate).__name__ if self._delegate else 'passthrough'}"
        )

    def _initialize_default_delegate(self) -> None:
        """Load the default processor based on environment configuration."""
        from vllm_dynamic_loader.config import get_config

        config = get_config()
        default_processor = config.hotswap_default_processor

        if default_processor and default_processor != "passthrough":
            try:
                from vllm_dynamic_loader.hotswap.processor_registry import ProcessorRegistry

                self._delegate = ProcessorRegistry.load(
                    default_processor,
                    self._vllm_config,
                    self._device,
                    self._is_pin_memory,
                )
                logger.info(f"Loaded default processor: {default_processor}")
            except Exception as e:
                logger.warning(f"Failed to load default processor '{default_processor}': {e}")
                self._delegate = None

    def _start_coordinator(self) -> None:
        """Start the swap coordinator for multi-process sync."""
        try:
            from vllm_dynamic_loader.hotswap.coordinator import SwapCoordinator

            self._coordinator = SwapCoordinator()
            self._coordinator.start_watching(self._handle_swap_request)
        except Exception as e:
            logger.error(f"Failed to start swap coordinator: {e}")

    def _handle_swap_request(self, processor_name: Optional[str]) -> None:
        """Handle a swap request from the coordinator.

        Args:
            processor_name: Name of processor to load, or None for passthrough.
        """
        if processor_name is None:
            # Switch to passthrough mode
            self.swap_delegate(None)
        else:
            try:
                from vllm_dynamic_loader.hotswap.processor_registry import ProcessorRegistry

                # Refresh registry to discover newly installed processors
                ProcessorRegistry.refresh()

                new_delegate = ProcessorRegistry.load(
                    processor_name,
                    self._vllm_config,
                    self._device,
                    self._is_pin_memory,
                )
                self.swap_delegate(new_delegate)
            except Exception as e:
                logger.error(f"Failed to load processor '{processor_name}': {e}")
                raise

    def is_argmax_invariant(self) -> bool:
        """Delegate to current processor, or True if pass-through."""
        with self._delegate_lock:
            if self._delegate is None:
                return True  # Pass-through preserves argmax
            return self._delegate.is_argmax_invariant()

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        """Update state and propagate to delegate.

        Args:
            batch_update: Information about batch changes.
        """
        with self._delegate_lock:
            if batch_update:
                self._batch_size = batch_update.batch_size
                self._last_batch_update = batch_update

            if self._delegate is not None:
                self._delegate.update_state(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply processing via current delegate.

        Thread-safe application that handles:
        - Pass-through when no delegate
        - Delegate swap during processing
        - Request tracking for graceful swap

        Args:
            logits: Raw logits tensor (batch_size, vocab_size).

        Returns:
            Processed logits tensor.
        """
        # Track in-flight requests for graceful swap
        with self._requests_lock:
            self._requests_in_flight += 1

        try:
            with self._delegate_lock:
                if self._delegate is None:
                    # Pass-through mode: return logits unchanged
                    return logits

                # Delegate processing
                return self._delegate.apply(logits)
        finally:
            with self._requests_lock:
                self._requests_in_flight -= 1

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Validate sampling parameters.

        Args:
            sampling_params: The sampling parameters to validate.
        """
        # Delegate validation is done by the concrete processor
        # during swap, not at request time
        pass

    # ==================== HOT-SWAP API ====================

    def swap_delegate(
        self,
        new_delegate,
        graceful: bool = True,
        timeout_ms: int = 5000,
    ) -> bool:
        """Atomically swap the delegate processor.

        Args:
            new_delegate: New processor to use (None for pass-through).
            graceful: Wait for in-flight requests to complete.
            timeout_ms: Maximum wait time for graceful swap.

        Returns:
            True if swap successful, False on timeout.
        """
        old_name = type(self._delegate).__name__ if self._delegate else "passthrough"
        new_name = type(new_delegate).__name__ if new_delegate else "passthrough"

        logger.info(f"Swapping delegate: {old_name} -> {new_name}")

        self._swap_in_progress.set()

        try:
            if graceful:
                # Wait for in-flight requests to complete
                import time

                start = time.time()
                while self._requests_in_flight > 0:
                    if (time.time() - start) * 1000 > timeout_ms:
                        logger.warning(
                            f"Swap timeout: {self._requests_in_flight} requests still in flight"
                        )
                        return False
                    time.sleep(0.001)  # 1ms polling

            with self._delegate_lock:
                _old_delegate = self._delegate  # noqa: F841 - kept for potential future use

                # Migrate state to new delegate if applicable
                if new_delegate is not None and self._last_batch_update is not None:
                    try:
                        new_delegate.update_state(self._last_batch_update)
                    except Exception as e:
                        logger.warning(f"State migration failed: {e}")

                # Atomic swap
                self._delegate = new_delegate

                logger.info(f"Swap complete: {old_name} -> {new_name}")

                return True
        finally:
            self._swap_in_progress.clear()

    def get_delegate_info(self) -> dict:
        """Get information about the current delegate."""
        with self._delegate_lock:
            return {
                "delegate_type": type(self._delegate).__name__ if self._delegate else None,
                "batch_size": self._batch_size,
                "requests_in_flight": self._requests_in_flight,
                "swap_in_progress": self._swap_in_progress.is_set(),
                "device": str(self._device),
            }

    def get_delegate(self):
        """Get the current delegate processor."""
        with self._delegate_lock:
            return self._delegate

    @classmethod
    def get_all_instances(cls) -> list:
        """Get all proxy instances (for testing/debugging)."""
        with cls._instance_lock:
            return cls._instances.copy()

    def shutdown(self) -> None:
        """Shutdown the proxy and cleanup resources."""
        if self._coordinator:
            self._coordinator.stop_watching()

        with HotSwapProxyProcessor._instance_lock:
            if self in HotSwapProxyProcessor._instances:
                HotSwapProxyProcessor._instances.remove(self)

        logger.info("HotSwapProxyProcessor shutdown complete")
