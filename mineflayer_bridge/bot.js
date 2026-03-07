/**
 * bot.js - Mineflayer bot with HTTP bridge for Python agent
 * 
 * The Python agent communicates with this via HTTP:
 *   GET  /observe       -> get current structured observation
 *   POST /action        -> execute an action { name, params }
 *   GET  /health        -> check if bot is alive
 *   POST /disconnect    -> disconnect bot
 */

const mineflayer = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');
const express = require('express');
const { setupActions } = require('./actions');

// ─── Parse CLI args ───
const args = {};
process.argv.slice(2).forEach(arg => {
    const [key, val] = arg.split('=');
    args[key.replace('--', '')] = val;
});

const MC_HOST = args.host || 'localhost';
const MC_PORT = parseInt(args.port || '25565');
const MC_USERNAME = args.username || 'MemAgent';
const MC_VERSION = args.version || '1.20.4';
const HTTP_PORT = parseInt(args.http_port || '3001');

console.log(`[Bridge] Connecting to ${MC_HOST}:${MC_PORT} as ${MC_USERNAME} (MC ${MC_VERSION})`);
console.log(`[Bridge] HTTP API on port ${HTTP_PORT}`);

// ─── Create Bot ───
const bot = mineflayer.createBot({
    host: MC_HOST,
    port: MC_PORT,
    username: MC_USERNAME,
    version: MC_VERSION,
});

bot.loadPlugin(pathfinder);

let mcData = null;
let actions = null;
let botReady = false;
let lastError = null;

bot.once('spawn', () => {
    mcData = require('minecraft-data')(bot.version);
    actions = setupActions(bot, mcData);
    botReady = true;
    console.log('[Bridge] Bot spawned and ready!');
});

bot.on('error', (err) => {
    console.error('[Bridge] Bot error:', err.message);
    lastError = err.message;
});

bot.on('kicked', (reason) => {
    console.error('[Bridge] Bot kicked:', reason);
    lastError = `Kicked: ${reason}`;
});

let deathCount = 0;
let lastDeathTime = 0;

bot.on('death', () => {
    deathCount++;
    lastDeathTime = Date.now();
    lastError = `Bot died (death #${deathCount}). All items lost. Will respawn at spawn point.`;
    console.log(`[Bridge] ${lastError}`);
});

// After respawn, log it
bot.on('respawn', () => {
    console.log(`[Bridge] Respawned after death #${deathCount}. Position: ${bot.entity.position}`);
});

// ─── Observation Builder ───

function getObservation() {
    if (!botReady) return { error: 'Bot not ready' };

    const pos = bot.entity.position;
    const inv = {};
    bot.inventory.items().forEach(item => {
        inv[item.name] = (inv[item.name] || 0) + item.count;
    });

    // Nearby entities (top 10 by distance)
    const nearbyEntities = Object.values(bot.entities)
        .filter(e => e !== bot.entity && e.position.distanceTo(pos) < 32)
        .sort((a, b) => a.position.distanceTo(pos) - b.position.distanceTo(pos))
        .slice(0, 10)
        .map(e => ({
            type: e.name || e.displayName || 'unknown',
            distance: Math.round(e.position.distanceTo(pos) * 10) / 10,
            position: {
                x: Math.round(e.position.x),
                y: Math.round(e.position.y),
                z: Math.round(e.position.z)
            }
        }));

    // Nearby blocks scan (compact - just counts in a 6-block radius)
    const blockPos = pos.floored();
    const nearbyBlocks = {};
    for (let dx = -6; dx <= 6; dx++) {
        for (let dy = -2; dy <= 2; dy++) {
            for (let dz = -6; dz <= 6; dz++) {
                const block = bot.blockAt(blockPos.offset(dx, dy, dz));
                if (block && block.name !== 'air') {
                    nearbyBlocks[block.name] = (nearbyBlocks[block.name] || 0) + 1;
                }
            }
        }
    }

    // Equipment
    const equipment = {
        mainhand: bot.heldItem ? bot.heldItem.name : 'empty',
        offhand: bot.inventory.slots[45] ? bot.inventory.slots[45].name : 'empty',
        helmet: bot.inventory.slots[5] ? bot.inventory.slots[5].name : 'empty',
        chestplate: bot.inventory.slots[6] ? bot.inventory.slots[6].name : 'empty',
        leggings: bot.inventory.slots[7] ? bot.inventory.slots[7].name : 'empty',
        boots: bot.inventory.slots[8] ? bot.inventory.slots[8].name : 'empty',
    };

    return {
        position: {
            x: Math.round(pos.x * 10) / 10,
            y: Math.round(pos.y * 10) / 10,
            z: Math.round(pos.z * 10) / 10,
            pitch: Math.round(bot.entity.pitch * 100) / 100,
            yaw: Math.round(bot.entity.yaw * 100) / 100
        },
        inventory: inv,
        stats: {
            health: Math.round(bot.health * 10) / 10,
            food: bot.food,
            oxygen: bot.oxygenLevel || 20
        },
        equipment: equipment,
        environment: {
            time: bot.time.timeOfDay,
            is_day: bot.time.timeOfDay < 13000,
            biome: bot.blockAt(blockPos) ? bot.blockAt(blockPos).biome?.name || 'unknown' : 'unknown',
            is_raining: bot.isRaining
        },
        nearby_entities: nearbyEntities,
        nearby_blocks: nearbyBlocks,
        last_error: lastError,
        deaths: deathCount,
        recently_died: (Date.now() - lastDeathTime) < 10000  // true if died in last 10s
    };
}

// ─── HTTP Server ───

const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ ready: botReady, error: lastError });
});

app.get('/observe', (req, res) => {
    try {
        const obs = getObservation();
        res.json(obs);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/action', async (req, res) => {
    const { name, params } = req.body;
    if (!botReady) {
        return res.status(503).json({ success: false, message: 'Bot not ready' });
    }
    if (!actions[name]) {
        return res.status(400).json({
            success: false,
            message: `Unknown action: ${name}. Available: ${Object.keys(actions).join(', ')}`
        });
    }
    try {
        lastError = null;
        const result = await Promise.race([
            actions[name](params || {}),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Action timeout (90s)')), 90000))
        ]);
        res.json(result);
    } catch (e) {
        lastError = e.message;
        res.json({ success: false, message: e.message });
    }
});

app.post('/disconnect', (req, res) => {
    bot.quit();
    res.json({ success: true, message: 'Disconnected' });
    setTimeout(() => process.exit(0), 1000);
});

app.post('/reset', async (req, res) => {
    // Reset: clear inventory, remove drops, teleport to surface
    try {
        // Set time to day and kill hostile mobs to prevent deaths during episode
        bot.chat('/time set day');
        await new Promise(r => setTimeout(r, 300));
        bot.chat('/kill @e[type=!player]');
        await new Promise(r => setTimeout(r, 500));

        // Reset death counter for new episode
        deathCount = 0;
        lastDeathTime = 0;

        // Clear inventory
        bot.chat('/clear');
        await new Promise(r => setTimeout(r, 500));

        // Teleport to a random surface location nearby (spread out for variety)
        // Pick a random offset so each episode starts at a slightly different spot
        const offsetX = Math.floor(Math.random() * 200) - 100;
        const offsetZ = Math.floor(Math.random() * 200) - 100;
        bot.chat(`/tp ${bot.username} ${offsetX} 200 ${offsetZ}`);
        await new Promise(r => setTimeout(r, 1000));

        // Now tp down to the actual surface (200 is above ground, let gravity work
        // or use ~ ~ ~ after a spreadplayers)
        // Better: use spreadplayers to find a safe surface spot
        bot.chat(`/spreadplayers ${offsetX} ${offsetZ} 0 50 false ${bot.username}`);
        await new Promise(r => setTimeout(r, 2000));

        // Clear again after teleport (in case of death from fall)
        bot.chat('/clear');
        await new Promise(r => setTimeout(r, 500));

        const pos = bot.entity.position;
        const items = bot.inventory.items();
        lastError = null;
        res.json({ 
            success: true, 
            message: `Reset complete. Position: (${Math.round(pos.x)}, ${Math.round(pos.y)}, ${Math.round(pos.z)}). Inventory: ${items.length} items.`,
            inventory_empty: items.length === 0
        });
    } catch (e) {
        res.json({ success: false, message: `Reset failed: ${e.message}` });
    }
});

app.listen(HTTP_PORT, () => {
    console.log(`[Bridge] HTTP server listening on port ${HTTP_PORT}`);
});