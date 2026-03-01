"""
MineDojo environment wrapper.
Converts raw observations into structured text representation.
Maps LLM string actions to MineDojo action arrays.
"""
import numpy as np
from typing import Tuple, Optional

try:
    import minedojo
except ImportError:
    minedojo = None
    print("[WARN] minedojo not installed. Using MockEnv for development.")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import IMAGE_SIZE, MAX_STEPS_PER_EPISODE, NEARBY_ENTITIES_TOP_N


# ─── Biome ID → Name mapping (common ones) ───────────────────────────
BIOME_NAMES = {
    0: "ocean", 1: "plains", 2: "desert", 3: "extreme_hills", 4: "forest",
    5: "taiga", 6: "swampland", 7: "river", 10: "frozen_ocean",
    12: "ice_plains", 14: "mushroom_island", 16: "beach",
    21: "jungle", 29: "roofed_forest", 35: "savanna", 36: "savanna_plateau",
    127: "the_void", 129: "sunflower_plains",
}


class TextObservation:
    """Structured text observation matching the slide's 'Data' schema."""
    
    def __init__(self):
        self.inventory: dict = {}       # item → count
        self.position: dict = {}        # x, y, z, pitch, yaw
        self.stats: dict = {}           # health, food, oxygen
        self.environment: dict = {}     # time, biome, raining
        self.equipment: dict = {}       # mainhand, offhand, armor
        self.nearby_entities: list = [] # [{type, distance}]
        self.raw_obs: dict = {}
    
    def to_text(self) -> str:
        """Convert to compact text representation for LLM prompt."""
        lines = []
        
        # Inventory (skip air/empty)
        inv_items = {k: v for k, v in self.inventory.items()
                     if k != "air" and v > 0}
        if inv_items:
            inv_str = ", ".join(f"{k}: {v}" for k, v in inv_items.items())
            lines.append(f"inventory: {{{inv_str}}}")
        else:
            lines.append("inventory: {empty}")
        
        # Position
        pos = self.position
        lines.append(
            f"position: {{x={pos.get('x', 0):.1f}, "
            f"y={pos.get('y', 0):.1f}, "
            f"z={pos.get('z', 0):.1f}, "
            f"pitch={pos.get('pitch', 0):.0f}, "
            f"yaw={pos.get('yaw', 0):.0f}}}"
        )
        
        # Stats
        s = self.stats
        lines.append(
            f"stats: {{health={s.get('health', 20):.0f}/20, "
            f"food={s.get('food', 20):.0f}/20, "
            f"oxygen={s.get('oxygen', 300):.0f}/300}}"
        )
        
        # Environment
        e = self.environment
        time_str = "day" if 0 <= e.get("time", 0) < 12000 else "night"
        biome = BIOME_NAMES.get(e.get("biome_id", 1), "unknown")
        raining = "yes" if e.get("raining", False) else "no"
        lines.append(
            f"environment: {{time={time_str}, biome={biome}, raining={raining}}}"
        )
        
        # Equipment
        eq = self.equipment
        lines.append(
            f"equipment: {{mainhand={eq.get('mainhand', 'air')}, "
            f"offhand={eq.get('offhand', 'air')}, "
            f"armor=[{', '.join(eq.get('armor', ['air']*4))}]}}"
        )
        
        # Nearby entities
        if self.nearby_entities:
            ents = [f"{e['type']}({e['distance']:.1f}m)"
                    for e in self.nearby_entities[:NEARBY_ENTITIES_TOP_N]]
            lines.append(f"nearby_entities: [{', '.join(ents)}]")
        else:
            lines.append("nearby_entities: [none]")
        
        return "\n".join(lines)
    
    def get_inventory_terms(self) -> list:
        """Extract inventory item names for BM25 query."""
        return [k for k, v in self.inventory.items()
                if k != "air" and v > 0]
    
    def get_entity_terms(self) -> list:
        """Extract nearby entity types for BM25 query."""
        return [e["type"] for e in self.nearby_entities]


def parse_observation(raw_obs: dict) -> TextObservation:
    """Convert MineDojo raw observation dict → TextObservation."""
    tobs = TextObservation()
    tobs.raw_obs = raw_obs
    
    # ── Inventory ──
    if "inventory" in raw_obs:
        inv = raw_obs["inventory"]
        names = inv.get("name", [])
        quantities = inv.get("quantity", [])
        for name, qty in zip(names, quantities):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name and name != "air" and qty > 0:
                tobs.inventory[name] = tobs.inventory.get(name, 0) + int(qty)
    
    # ── Position ──
    if "location_stats" in raw_obs:
        loc = raw_obs["location_stats"]
        pos_arr = loc.get("pos", [0, 0, 0])
        tobs.position = {
            "x": float(pos_arr[0]) if len(pos_arr) > 0 else 0,
            "y": float(pos_arr[1]) if len(pos_arr) > 1 else 0,
            "z": float(pos_arr[2]) if len(pos_arr) > 2 else 0,
            "pitch": float(loc.get("pitch", [0])[0]) if "pitch" in loc else 0,
            "yaw": float(loc.get("yaw", [0])[0]) if "yaw" in loc else 0,
        }
    
    # ── Life Stats ──
    if "life_stats" in raw_obs:
        ls = raw_obs["life_stats"]
        tobs.stats = {
            "health": float(ls.get("life", [20])[0]) if "life" in ls else 20,
            "food": float(ls.get("food", [20])[0]) if "food" in ls else 20,
            "oxygen": float(ls.get("oxygen", [300])[0]) if "oxygen" in ls else 300,
        }
    
    # ── Environment ──
    if "location_stats" in raw_obs:
        loc = raw_obs["location_stats"]
        tobs.environment = {
            "time": int(loc.get("world_time", [6000])[0]) if "world_time" in loc else 6000,
            "biome_id": int(loc.get("biome_id", 1)),
            "raining": bool(loc.get("is_raining", [False])[0]) if "is_raining" in loc else False,
        }
    
    # ── Equipment ──
    if "equipment" in raw_obs:
        eq = raw_obs["equipment"]
        names = eq.get("name", ["air"] * 6)
        decoded = []
        for n in names:
            if isinstance(n, bytes):
                n = n.decode("utf-8")
            decoded.append(n if n else "air")
        tobs.equipment = {
            "mainhand": decoded[0] if len(decoded) > 0 else "air",
            "offhand": decoded[5] if len(decoded) > 5 else "air",
            "armor": decoded[1:5] if len(decoded) >= 5 else ["air"] * 4,
        }
    
    # ── Nearby Entities (from entity_info if available) ──
    if "entity_info" in raw_obs:
        ei = raw_obs["entity_info"]
        entity_names = ei.get("entity_name", [])
        entity_dists = ei.get("entity_distance", [])
        for ename, edist in zip(entity_names, entity_dists):
            if isinstance(ename, bytes):
                ename = ename.decode("utf-8")
            if ename:
                tobs.nearby_entities.append({
                    "type": ename,
                    "distance": float(edist)
                })
        # Sort by distance
        tobs.nearby_entities.sort(key=lambda e: e["distance"])
    
    return tobs


def compute_delta(prev: TextObservation, curr: TextObservation) -> dict:
    """
    Compute what changed between two observations.
    Returns a dict of changed fields for delta-based step memory filtering.
    """
    delta = {}
    
    # Inventory changes
    all_items = set(list(prev.inventory.keys()) + list(curr.inventory.keys()))
    inv_changes = {}
    for item in all_items:
        old = prev.inventory.get(item, 0)
        new = curr.inventory.get(item, 0)
        if old != new:
            inv_changes[item] = {"old": old, "new": new, "diff": new - old}
    if inv_changes:
        delta["inventory_changes"] = inv_changes
    
    # Position change (significant movement)
    if prev.position and curr.position:
        dx = abs(curr.position.get("x", 0) - prev.position.get("x", 0))
        dy = abs(curr.position.get("y", 0) - prev.position.get("y", 0))
        dz = abs(curr.position.get("z", 0) - prev.position.get("z", 0))
        dist = (dx**2 + dy**2 + dz**2) ** 0.5
        if dist > 1.0:
            delta["moved"] = f"{dist:.1f} blocks"
    
    # Health change
    old_hp = prev.stats.get("health", 20)
    new_hp = curr.stats.get("health", 20)
    if abs(old_hp - new_hp) > 0.5:
        delta["health_change"] = new_hp - old_hp
    
    # Entity changes
    old_ents = set(e["type"] for e in prev.nearby_entities)
    new_ents = set(e["type"] for e in curr.nearby_entities)
    if old_ents != new_ents:
        appeared = new_ents - old_ents
        disappeared = old_ents - new_ents
        if appeared:
            delta["entities_appeared"] = list(appeared)
        if disappeared:
            delta["entities_disappeared"] = list(disappeared)
    
    # Equipment change
    if prev.equipment.get("mainhand") != curr.equipment.get("mainhand"):
        delta["mainhand_changed"] = {
            "from": prev.equipment.get("mainhand", "air"),
            "to": curr.equipment.get("mainhand", "air"),
        }
    
    return delta


# ─── Action Mapping ──────────────────────────────────────────────────

def map_action_to_minedojo(action_str: str, env) -> np.ndarray:
    """
    Map a string action from LLM to MineDojo action array.
    
    MineDojo action space (compound):
    [0] forward/back: 0=noop, 1=forward, 2=back
    [1] left/right:   0=noop, 1=left, 2=right
    [2] jump/sneak/sprint: 0=noop, 1=jump, 2=sneak, 3=sprint
    [3] camera pitch delta (continuous, degrees)
    [4] camera yaw delta (continuous, degrees)
    [5] functional: 0=noop, 1=use, 2=drop, 3=attack
    [6] craft arg (int index)
    [7] equip arg (int index)
    """
    act = env.action_space.no_op()
    action_str = action_str.strip().lower()
    
    if action_str == "forward":
        act[0] = 1
    elif action_str == "back":
        act[0] = 2
    elif action_str == "left":
        act[1] = 1
    elif action_str == "right":
        act[1] = 2
    elif action_str == "jump":
        act[2] = 1
    elif action_str == "sneak":
        act[2] = 2
    elif action_str == "sprint":
        act[0] = 1  # forward
        act[2] = 3  # sprint
    elif action_str == "attack":
        act[5] = 3
    elif action_str == "use":
        act[5] = 1
    elif action_str == "drop":
        act[5] = 2
    elif action_str.startswith("camera_up"):
        act[3] = -15  # pitch up
    elif action_str.startswith("camera_down"):
        act[3] = 15   # pitch down
    elif action_str.startswith("camera_left"):
        act[4] = -15  # yaw left
    elif action_str.startswith("camera_right"):
        act[4] = 15   # yaw right
    elif action_str.startswith("craft "):
        # For simplicity, craft actions need the item index
        # This requires looking up the recipe - simplified here
        act[5] = 1  # use (open crafting table, then craft)
    elif action_str.startswith("equip "):
        # Try to equip from hotbar
        item_name = action_str.replace("equip ", "").strip()
        # Would need inventory lookup - simplified
        pass
    # else: noop (default)
    
    return act


class MockEnv:
    """
    Mock environment for development/testing without MineDojo installed.
    Simulates basic Minecraft observations.
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task_prompt = f"Complete task: {task_id}"
        self.task_guidance = "Explore and accomplish the goal."
        self.step_count = 0
        self._done = False
        self._inventory = {"shears": 1} if "shears" in task_id else {}
        self._pos = [100.0, 65.0, 100.0]
        self._health = 20.0
        
    class _ActionSpace:
        def no_op(self):
            return np.zeros(8, dtype=np.float32)
    
    action_space = _ActionSpace()
    
    def reset(self):
        self.step_count = 0
        self._done = False
        self._health = 20.0
        self._pos = [100.0, 65.0, 100.0]
        return self._make_obs()
    
    def step(self, action):
        self.step_count += 1
        # Simulate movement
        if action[0] == 1:  # forward
            self._pos[0] += 1.0
        # Simulate finding things
        reward = 0.0
        if self.step_count > 20 and np.random.random() < 0.05:
            reward = 1.0
            self._done = True
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            self._done = True
        
        obs = self._make_obs()
        return obs, reward, self._done, {"step": self.step_count}
    
    def _make_obs(self):
        entities = []
        if np.random.random() < 0.3:
            etype = np.random.choice(["sheep", "cow", "pig", "spider", "zombie"])
            entities.append(etype)
        
        return {
            "inventory": {
                "name": list(self._inventory.keys()) + ["air"] * (36 - len(self._inventory)),
                "quantity": list(self._inventory.values()) + [0] * (36 - len(self._inventory)),
            },
            "location_stats": {
                "pos": self._pos,
                "pitch": [0.0],
                "yaw": [np.random.uniform(-180, 180)],
                "biome_id": 1,
                "world_time": [6000 + self.step_count * 20],
                "is_raining": [False],
            },
            "life_stats": {
                "life": [self._health],
                "food": [18.0],
                "oxygen": [300.0],
            },
            "equipment": {
                "name": ["air"] * 6,
            },
            "entity_info": {
                "entity_name": entities,
                "entity_distance": [np.random.uniform(3, 15)] * len(entities),
            },
        }
    
    def close(self):
        pass


def make_env(task_id: str):
    """Create environment - uses MineDojo if available, else MockEnv."""
    if minedojo is not None:
        try:
            env = minedojo.make(task_id=task_id, image_size=IMAGE_SIZE)
            return env
        except Exception as e:
            print(f"[ENV] Failed to create MineDojo env: {e}")
            print("[ENV] Falling back to MockEnv")
    
    return MockEnv(task_id)
