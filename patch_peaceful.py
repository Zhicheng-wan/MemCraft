#!/usr/bin/env python3
"""
Run this on your VM: python3 patch_peaceful.py
Adds peaceful difficulty to reset_minecraft_world() in run_agent.py
"""
import re

filepath = "run_agent.py"

with open(filepath, "r") as f:
    content = f.read()

# Check if already patched
if "difficulty=peaceful" in content:
    print("✓ Already patched - peaceful mode is present in run_agent.py")
    exit(0)

# Find the seed setting block and add peaceful after it
old = '''            check=True, capture_output=True, timeout=10
        )
        # Delete world'''

new = '''            check=True, capture_output=True, timeout=10
        )
        # Set peaceful difficulty (no hostile mobs)
        subprocess.run(
            ["sudo", "docker", "exec", container, "sh", "-c",
             'sed -i "s/difficulty=.*/difficulty=peaceful/" /data/server.properties'],
            check=True, capture_output=True, timeout=10
        )
        # Delete world'''

if old in content:
    content = content.replace(old, new, 1)
    with open(filepath, "w") as f:
        f.write(content)
    print("✓ Patched! Peaceful difficulty added to reset_minecraft_world()")
else:
    print("✗ Could not find insertion point.")
    print("  Add this manually after the 'Set seed' block in reset_minecraft_world():\n")
    print('        # Set peaceful difficulty (no hostile mobs)')
    print('        subprocess.run(')
    print('            ["sudo", "docker", "exec", container, "sh", "-c",')
    print("             'sed -i \"s/difficulty=.*/difficulty=peaceful/\" /data/server.properties'],")
    print('            check=True, capture_output=True, timeout=10')
    print('        )')