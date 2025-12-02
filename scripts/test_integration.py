#!/usr/bin/env python3
"""Integration test script for vLLM plugins.

This script tests the entropy and CoT decoder plugins against a running vLLM server.

IMPORTANT: vLLM v1 does not support per-request logits processor selection.
Plugins apply GLOBALLY to all requests when installed. This test verifies that
completions work correctly when decoder plugins are installed at the server level.

Usage:
    # Start the server first:
    docker-compose up -d

    # Then run this script:
    python scripts/test_integration.py

    # Or specify a different endpoint:
    python scripts/test_integration.py --base-url http://localhost:8000
"""

import argparse
import sys

import requests


def test_basic_completion(base_url: str) -> bool:
    """Test basic completion without custom logits processor."""
    print("\n=== Test: Basic Completion ===")

    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.7,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        print(f"Response: {result['choices'][0]['text']}")
        print("✓ Basic completion works")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_entropy_decoder(base_url: str) -> bool:
    """Test completion with entropy decoder plugin.

    Note: The entropy decoder is applied GLOBALLY when installed.
    vLLM v1 does not support per-request logits processor selection.
    """
    print("\n=== Test: Entropy Decoder (Global) ===")

    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt": "Explain quantum computing in simple terms:",
                "max_tokens": 50,
                "temperature": 0.8,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        print(f"Response: {result['choices'][0]['text']}")
        print("✓ Completion works (entropy decoder applied globally if installed)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_cot_decoder(base_url: str) -> bool:
    """Test completion with CoT decoder plugin.

    Note: The CoT decoder is applied GLOBALLY when installed.
    vLLM v1 does not support per-request logits processor selection.
    """
    print("\n=== Test: CoT Decoder (Global) ===")

    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt": "What is 15 + 27? Let me think step by step:",
                "max_tokens": 50,
                "temperature": 0.8,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        print(f"Response: {result['choices'][0]['text']}")
        print("✓ Completion works (CoT decoder applied globally if installed)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_health(base_url: str) -> bool:
    """Test server health endpoint."""
    print("\n=== Test: Server Health ===")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        print("✓ Server is healthy")
        return True
    except Exception as e:
        print(f"✗ Server not reachable: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM plugins integration")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM server",
    )
    args = parser.parse_args()

    print(f"Testing vLLM plugins at {args.base_url}")
    print("=" * 50)

    results = []

    # Test health first
    if not test_health(args.base_url):
        print("\n❌ Server is not running. Start it with: docker-compose up -d")
        sys.exit(1)

    results.append(("Basic Completion", test_basic_completion(args.base_url)))
    results.append(("Entropy Decoder", test_entropy_decoder(args.base_url)))
    results.append(("CoT Decoder", test_cot_decoder(args.base_url)))

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    print(f"\nPassed: {passed}/{total}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
