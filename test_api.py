#!/usr/bin/env python3
"""
test_api.py - Quick test to verify your TritonAI API key works.
Run this FIRST before anything else.

Usage:
    export TRITONAI_API_KEY="your-key-here"
    python test_api.py
"""

import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from agent.brain import Brain


def main():
    api_key = os.environ.get("TRITONAI_API_KEY")
    if not api_key:
        print("❌ Set TRITONAI_API_KEY first:")
        print("   export TRITONAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("Testing TritonAI API connection...")
    print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")

    brain = Brain(
        api_key=api_key,
        api_url="https://tritonai-api.ucsd.edu/v1/chat/completions",
        model="api-llama-4-scout",
        max_tokens=100,
        temperature=0.3,
    )

    # Test 1: Simple query
    print("\n─── Test 1: Simple query ───")
    result = brain.query(
        system_prompt="You are a helpful Minecraft assistant.",
        user_prompt="What tool do I need to mine stone in Minecraft? Reply in one sentence."
    )
    if result["error"]:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    print(f"✓ Response: {result['content']}")
    print(f"  Tokens: {result['tokens_used']}")
    print(f"  Latency: {result['latency']:.2f}s")

    # Test 2: JSON action output
    print("\n─── Test 2: JSON action parsing ───")
    result = brain.query(
        system_prompt=(
            "You are a Minecraft bot. Respond with ONLY a JSON action.\n"
            "Example: {\"name\": \"find_and_mine_block\", \"params\": {\"block_name\": \"dirt\", \"count\": 1}}"
        ),
        user_prompt=(
            "GOAL: Mine 3 dirt blocks\n"
            "CURRENT STATE: Position: (0, 64, 0) | Inventory: [empty]\n"
            "Choose your next action (JSON only):"
        )
    )
    if result["error"]:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)

    print(f"  Raw: {result['content']}")
    parsed = brain.parse_json_response(result["content"])
    if parsed:
        print(f"✓ Parsed action: {parsed}")
    else:
        print(f"⚠ Could not parse JSON (may need prompt tuning)")

    print(f"\n─── Stats ───")
    stats = brain.get_stats()
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg latency: {stats['avg_latency']:.2f}s")
    print("\n✅ API connection working! Ready to run the agent.")


if __name__ == "__main__":
    main()
