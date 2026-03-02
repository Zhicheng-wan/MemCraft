#!/usr/bin/env python3
"""
diagnose.py - Thorough diagnostic for MemCraft.
Tests every layer before you spend any API credits.

Usage:
    python diagnose.py --host localhost --port 25565
    python diagnose.py --host localhost --port 25565 --skip-api
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import requests
from pathlib import Path


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓ {msg}{RESET}")
def fail(msg): print(f"  {RED}✗ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠ {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ {msg}{RESET}")


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_prerequisites():
    """Test 1: Check Node.js, npm, Python deps."""
    header("TEST 1: Prerequisites")
    all_good = True

    # Node.js
    node = shutil.which("node")
    if node:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        ver = result.stdout.strip()
        ok(f"Node.js found: {ver} ({node})")
        # Check version >= 18
        major = int(ver.lstrip("v").split(".")[0])
        if major < 18:
            warn(f"Node.js {ver} may be too old. Recommend v18+.")
    else:
        fail("Node.js not found! Install: https://nodejs.org/")
        all_good = False

    # npm
    npm = shutil.which("npm")
    if npm:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        ok(f"npm found: v{result.stdout.strip()}")
    else:
        fail("npm not found!")
        all_good = False

    # Python deps
    for pkg in ["requests", "rank_bm25", "numpy"]:
        try:
            __import__(pkg)
            ok(f"Python package '{pkg}' installed")
        except ImportError:
            fail(f"Python package '{pkg}' missing. Run: pip install -r requirements.txt")
            all_good = False

    return all_good


def test_npm_install():
    """Test 2: Check/install node_modules."""
    header("TEST 2: Mineflayer npm Dependencies")
    bridge_dir = Path(__file__).parent / "mineflayer_bridge"

    if not bridge_dir.exists():
        fail(f"Bridge directory not found: {bridge_dir}")
        return False

    package_json = bridge_dir / "package.json"
    if not package_json.exists():
        fail("package.json missing in mineflayer_bridge/")
        return False
    ok("package.json found")

    node_modules = bridge_dir / "node_modules"
    if not node_modules.exists():
        info("node_modules not found, running npm install...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=bridge_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            fail(f"npm install failed:\n{result.stderr}")
            return False
        ok("npm install succeeded")
    else:
        ok("node_modules exists")

    # Check key packages
    for pkg in ["mineflayer", "mineflayer-pathfinder", "express"]:
        pkg_dir = node_modules / pkg
        if pkg_dir.exists():
            ok(f"  {pkg} installed")
        else:
            fail(f"  {pkg} missing! Run: cd mineflayer_bridge && npm install")
            return False

    return True


def test_bridge_starts(http_port):
    """Test 3: Check if the bridge Node.js process starts without errors."""
    header("TEST 3: Bridge Process Starts")
    bridge_dir = Path(__file__).parent / "mineflayer_bridge"

    # Check bot.js syntax by running node --check
    result = subprocess.run(
        ["node", "--check", "bot.js"],
        cwd=bridge_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"bot.js has syntax errors:\n{result.stderr}")
        return False
    ok("bot.js syntax OK")

    result = subprocess.run(
        ["node", "--check", "actions.js"],
        cwd=bridge_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"actions.js has syntax errors:\n{result.stderr}")
        return False
    ok("actions.js syntax OK")

    # Check if http_port is already in use
    try:
        resp = requests.get(f"http://localhost:{http_port}/health", timeout=2)
        warn(f"Port {http_port} already in use (bridge already running?)")
        info(f"Response: {resp.json()}")
        return True
    except:
        ok(f"Port {http_port} is free")

    return True


def test_minecraft_connection(host, port, username, version, http_port):
    """Test 4: Actually start the bridge and connect to Minecraft."""
    header("TEST 4: Minecraft Server Connection")

    # First test: can we reach the server at all?
    import socket
    info(f"Testing TCP connection to {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((host, port))
        sock.close()
        ok(f"TCP connection to {host}:{port} succeeded")
    except socket.timeout:
        fail(f"Connection to {host}:{port} timed out!")
        info("Is your Minecraft server running?")
        info("If using singleplayer, did you Open to LAN?")
        return False
    except ConnectionRefusedError:
        fail(f"Connection to {host}:{port} refused!")
        info("Minecraft server is not listening on this port.")
        info("Check: 1) Server is running  2) Port is correct  3) online-mode=false in server.properties")
        return False
    except Exception as e:
        fail(f"Connection error: {e}")
        return False

    # Now start the actual bridge
    bridge_dir = Path(__file__).parent / "mineflayer_bridge"
    info("Starting Mineflayer bridge...")

    proc = subprocess.Popen(
        [
            "node", "bot.js",
            f"--host={host}",
            f"--port={port}",
            f"--username={username}",
            f"--version={version}",
            f"--http_port={http_port}",
        ],
        cwd=bridge_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait and collect output
    bot_ready = False
    output_lines = []
    start = time.time()
    timeout = 20  # seconds

    while time.time() - start < timeout:
        # Check if process died
        if proc.poll() is not None:
            remaining = proc.stdout.read()
            output_lines.append(remaining)
            fail(f"Bridge process exited with code {proc.returncode}")
            print(f"\n  {YELLOW}─── Bridge output ───{RESET}")
            for line in "".join(output_lines).strip().split("\n"):
                print(f"  {line}")
            print(f"  {YELLOW}─────────────────────{RESET}")
            return False

        # Non-blocking read
        import select
        if select.select([proc.stdout], [], [], 0.5)[0]:
            line = proc.stdout.readline()
            if line:
                output_lines.append(line)
                stripped = line.strip()
                if stripped:
                    info(f"Bridge: {stripped}")

        # Try the health endpoint
        try:
            resp = requests.get(f"http://localhost:{http_port}/health", timeout=2)
            data = resp.json()
            if data.get("ready"):
                bot_ready = True
                break
            elif data.get("error"):
                warn(f"Bot error: {data['error']}")
        except requests.exceptions.ConnectionError:
            pass  # HTTP server not up yet
        except Exception:
            pass

    if bot_ready:
        ok("Bot connected and ready!")

        # Test observation
        info("Testing observation endpoint...")
        try:
            resp = requests.get(f"http://localhost:{http_port}/observe", timeout=5)
            obs = resp.json()
            if "error" in obs:
                fail(f"Observation error: {obs['error']}")
            else:
                ok("Observation working!")
                pos = obs.get("position", {})
                info(f"Bot position: ({pos.get('x')}, {pos.get('y')}, {pos.get('z')})")
                info(f"Health: {obs.get('stats', {}).get('health')}/20")
                inv = obs.get("inventory", {})
                info(f"Inventory: {inv if inv else 'empty'}")
                nearby = obs.get("nearby_blocks", {})
                top_blocks = sorted(nearby.items(), key=lambda x: -x[1])[:5]
                info(f"Nearby blocks: {', '.join(f'{k}:{v}' for k,v in top_blocks)}")
        except Exception as e:
            fail(f"Observation failed: {e}")

        # Test a safe action (scan surroundings)
        info("Testing action endpoint (scan_surroundings)...")
        try:
            resp = requests.post(
                f"http://localhost:{http_port}/action",
                json={"name": "scan_surroundings", "params": {"radius": 4}},
                timeout=10,
            )
            result = resp.json()
            if result.get("success"):
                ok(f"Action works! Result: {result.get('message', '')[:80]}")
            else:
                warn(f"Action returned failure: {result.get('message')}")
        except Exception as e:
            fail(f"Action failed: {e}")

        # Disconnect
        info("Disconnecting bot...")
        try:
            requests.post(f"http://localhost:{http_port}/disconnect", timeout=3)
        except:
            pass

    else:
        fail(f"Bot did not become ready within {timeout}s")
        print(f"\n  {YELLOW}─── Bridge output ───{RESET}")
        for line in "".join(output_lines).strip().split("\n"):
            print(f"  {line}")
        print(f"  {YELLOW}─────────────────────{RESET}")
        info("Common causes:")
        info("  - Wrong Minecraft version (try --version 1.20.4 or match your server)")
        info("  - Server has online-mode=true (set to false in server.properties)")
        info("  - Server hasn't finished loading yet")

    # Cleanup
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except:
        proc.kill()

    return bot_ready


def test_api(skip=False):
    """Test 5: Check TritonAI API connection."""
    header("TEST 5: TritonAI API Connection")

    if skip:
        info("Skipped (--skip-api flag)")
        return True

    api_key = os.environ.get("TRITONAI_API_KEY")
    if not api_key:
        warn("TRITONAI_API_KEY not set. Skipping API test.")
        info("Set it with: export TRITONAI_API_KEY='your-key-here'")
        return True  # Not fatal for diagnostics

    info(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    # Test the endpoint
    url = "https://tritonai-api.ucsd.edu/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "api-llama-4-scout",
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    info(f"Testing {url}...")
    try:
        start = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        latency = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            ok(f"API working! Response: '{content.strip()}'")
            info(f"Latency: {latency:.2f}s")
            info(f"Tokens used: prompt={usage.get('prompt_tokens', '?')}, "
                 f"completion={usage.get('completion_tokens', '?')}")
            return True
        elif resp.status_code == 401:
            fail("Authentication failed (401). Check your API key.")
            return False
        elif resp.status_code == 429:
            warn("Rate limited (429). Wait and try again.")
            return True
        else:
            fail(f"API returned {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        fail("API request timed out (30s)")
        info("The TritonAI server might be down or slow.")
        return False
    except requests.exceptions.ConnectionError as e:
        fail(f"Cannot reach API: {e}")
        info("Check your internet connection / VPN.")
        return False


def test_file_structure():
    """Test 6: Verify all project files exist."""
    header("TEST 6: Project File Structure")
    root = Path(__file__).parent
    all_good = True

    required_files = [
        "run_agent.py",
        "evaluate.py",
        "requirements.txt",
        "configs/default.json",
        "agent/__init__.py",
        "agent/brain.py",
        "agent/memory.py",
        "agent/observer.py",
        "agent/retrieval.py",
        "agent/consolidation.py",
        "agent/agent.py",
        "mineflayer_bridge/package.json",
        "mineflayer_bridge/bot.js",
        "mineflayer_bridge/actions.js",
    ]

    for f in required_files:
        path = root / f
        if path.exists():
            ok(f"{f}")
        else:
            fail(f"{f} MISSING")
            all_good = False

    # Check directories
    for d in ["memories", "logs"]:
        path = root / d
        if path.exists():
            ok(f"{d}/ directory exists")
        else:
            info(f"Creating {d}/ directory...")
            path.mkdir(exist_ok=True)
            ok(f"{d}/ created")

    return all_good


def main():
    parser = argparse.ArgumentParser(description="MemCraft Diagnostics")
    parser.add_argument("--host", default="localhost", help="MC server host")
    parser.add_argument("--port", type=int, default=25565, help="MC server port")
    parser.add_argument("--username", default="MemAgent", help="Bot username")
    parser.add_argument("--version", default="1.20.4", help="MC version")
    parser.add_argument("--http-port", type=int, default=3001, help="Bridge HTTP port")
    parser.add_argument("--skip-api", action="store_true", help="Skip API test")
    parser.add_argument("--skip-mc", action="store_true", help="Skip Minecraft connection test")
    args = parser.parse_args()

    print(f"\n{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║      MemCraft Diagnostic Tool                ║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════╝{RESET}")

    results = {}

    # Test 1: Prerequisites
    results["prerequisites"] = test_prerequisites()

    # Test 2: npm deps
    results["npm"] = test_npm_install()

    # Test 3: Bridge syntax
    results["bridge_syntax"] = test_bridge_starts(args.http_port)

    # Test 4: Minecraft connection
    if args.skip_mc:
        header("TEST 4: Minecraft Connection")
        info("Skipped (--skip-mc)")
        results["minecraft"] = None
    elif not results["prerequisites"] or not results["npm"]:
        header("TEST 4: Minecraft Connection")
        warn("Skipped due to earlier failures")
        results["minecraft"] = False
    else:
        results["minecraft"] = test_minecraft_connection(
            args.host, args.port, args.username, args.version, args.http_port
        )

    # Test 5: API
    results["api"] = test_api(skip=args.skip_api)

    # Test 6: File structure
    results["files"] = test_file_structure()

    # ─── Summary ───
    header("SUMMARY")
    all_pass = True
    checks = [
        ("Prerequisites (Node, npm, Python pkgs)", results["prerequisites"]),
        ("npm dependencies", results["npm"]),
        ("Bridge syntax check", results["bridge_syntax"]),
        ("Minecraft server connection", results["minecraft"]),
        ("TritonAI API", results["api"]),
        ("Project files", results["files"]),
    ]

    for name, passed in checks:
        if passed is None:
            info(f"{name}: SKIPPED")
        elif passed:
            ok(name)
        else:
            fail(name)
            all_pass = False

    if all_pass:
        print(f"\n  {GREEN}All checks passed! Ready to run:{RESET}")
        print(f"  {CYAN}python run_agent.py --task 'mine 5 dirt blocks' "
              f"--agent memagent --port {args.port}{RESET}\n")
    else:
        print(f"\n  {RED}Some checks failed. Fix the issues above before running.{RESET}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
