# Dockerfile for testing vLLM plugins
# Build: docker build -t vllm-plugins-test .
# Run:   docker run --gpus all -it vllm-plugins-test

FROM budstudio/vllm-cpu:0.2.0

WORKDIR /app

# Copy all plugins
COPY plugins/ /app/plugins/
COPY shared/ /app/shared/

# Install shared utilities
RUN pip install -e /app/shared/

# Install decoding strategy plugins
RUN pip install -e /app/plugins/vllm-entropy-decoder/
# RUN pip install -e /app/plugins/vllm-cot-decoder/
RUN pip install -e /app/plugins/vllm-dynamic-loader/.[api]

# Verify plugins are installed
RUN python -c "from importlib.metadata import entry_points; \
    eps = entry_points(group='vllm.logits_processors'); \
    print('Installed logits processors:', [ep.name for ep in eps])"

# Default command: start vLLM server
# Override model with: docker run --gpus all -e MODEL=your-model vllm-plugins-test
ENV MODEL=Qwen/Qwen2.5-0.5B-Instruct

# CMD ["sh", "-c", "python -m vllm.entrypoints.openai.api_server --model $MODEL --host 0.0.0.0 --port 8000"]
