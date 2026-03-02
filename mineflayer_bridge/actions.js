/**
 * actions.js - Available bot actions for MemCraft agent
 * Each action returns a result object: { success: bool, message: string, data?: any }
 */

const { goals: { GoalNear, GoalBlock, GoalXZ } } = require('mineflayer-pathfinder');
const { Movements } = require('mineflayer-pathfinder');
const vec3 = require('vec3');

function setupActions(bot, mcData) {
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);

    const actions = {};

    // ─── MOVEMENT ───

    actions.move_to = async ({ x, y, z }) => {
        try {
            await bot.pathfinder.goto(new GoalNear(x, y, z, 1));
            return { success: true, message: `Moved to (${x}, ${y}, ${z})` };
        } catch (e) {
            return { success: false, message: `Failed to move: ${e.message}` };
        }
    };

    actions.move_forward = async ({ steps = 1 }) => {
        try {
            bot.setControlState('forward', true);
            await new Promise(r => setTimeout(r, steps * 400));
            bot.setControlState('forward', false);
            return { success: true, message: `Moved forward ${steps} steps` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.jump = async () => {
        try {
            bot.setControlState('jump', true);
            await new Promise(r => setTimeout(r, 500));
            bot.setControlState('jump', false);
            return { success: true, message: 'Jumped' };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.look_at = async ({ x, y, z }) => {
        try {
            await bot.lookAt(vec3(x, y, z));
            return { success: true, message: `Looking at (${x}, ${y}, ${z})` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── MINING / DIGGING ───

    actions.dig_block = async ({ x, y, z }) => {
        try {
            const block = bot.blockAt(vec3(x, y, z));
            if (!block || block.name === 'air') {
                return { success: false, message: `No block at (${x}, ${y}, ${z})` };
            }
            if (!bot.canDigBlock(block)) {
                return { success: false, message: `Cannot dig ${block.name} (need better tool?)` };
            }
            await bot.dig(block);
            return { success: true, message: `Dug ${block.name} at (${x}, ${y}, ${z})` };
        } catch (e) {
            return { success: false, message: `Failed to dig: ${e.message}` };
        }
    };

    actions.dig_block_below = async () => {
        try {
            const pos = bot.entity.position.floored();
            const block = bot.blockAt(pos.offset(0, -1, 0));
            if (!block || block.name === 'air') {
                return { success: false, message: 'No block below' };
            }
            await bot.dig(block);
            return { success: true, message: `Dug ${block.name} below` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.find_and_mine_block = async ({ block_name, count = 1 }) => {
        try {
            // Block name aliases: some items drop differently than the block name
            const blockAliases = {
                'cobblestone': 'stone',      // mining stone drops cobblestone
                'raw_iron': 'iron_ore',      // mining iron_ore drops raw_iron
                'raw_copper': 'copper_ore',
                'raw_gold': 'gold_ore',
                'diamond': 'diamond_ore',
                'coal': 'coal_ore',
                'redstone': 'redstone_ore',
                'lapis_lazuli': 'lapis_ore',
            };
            const actualBlock = blockAliases[block_name] || block_name;
            
            const blockType = mcData.blocksByName[actualBlock];
            if (!blockType) {
                return { success: false, message: `Unknown block: ${block_name} (tried ${actualBlock})` };
            }

            // Auto-equip the best tool for this block type
            const toolPriority = ['diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe',
                                  'diamond_axe', 'iron_axe', 'stone_axe', 'wooden_axe',
                                  'diamond_shovel', 'iron_shovel', 'stone_shovel', 'wooden_shovel'];
            const needsPickaxe = ['stone', 'cobblestone', 'iron_ore', 'coal_ore', 'copper_ore', 
                                  'gold_ore', 'diamond_ore', 'lapis_ore', 'redstone_ore', 'furnace',
                                  'deepslate', 'andesite', 'granite', 'diorite'];
            const needsAxe = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 
                              'dark_oak_log', 'mangrove_log', 'oak_planks', 'birch_planks', 'crafting_table'];
            const needsShovel = ['dirt', 'grass_block', 'sand', 'gravel', 'clay', 'red_sand'];

            let bestTool = null;
            if (needsPickaxe.includes(block_name)) {
                bestTool = bot.inventory.items().find(i => i.name.includes('pickaxe'));
            } else if (needsAxe.includes(block_name)) {
                bestTool = bot.inventory.items().find(i => i.name.includes('axe') && !i.name.includes('pickaxe'));
            } else if (needsShovel.includes(block_name)) {
                bestTool = bot.inventory.items().find(i => i.name.includes('shovel'));
            }
            if (bestTool) {
                try { await bot.equip(bestTool, 'hand'); } catch(_) {}
            }

            let mined = 0;
            for (let i = 0; i < count; i++) {
                const block = bot.findBlock({
                    matching: blockType.id,
                    maxDistance: 32
                });
                if (!block) {
                    return {
                        success: mined > 0,
                        message: mined > 0 
                            ? `Mined ${mined}/${count} ${block_name}. No more nearby.`
                            : `No ${block_name} found nearby.`
                    };
                }
                const targetPos = block.position.clone();
                // Move close enough
                await bot.pathfinder.goto(new GoalNear(targetPos.x, targetPos.y, targetPos.z, 4));
                // Re-fetch the block at the target position (reference may be stale after moving)
                const freshBlock = bot.blockAt(targetPos);
                if (!freshBlock || freshBlock.name === 'air') {
                    // Block was already mined (maybe by falling or another cause)
                    continue;
                }
                try {
                    await bot.dig(freshBlock);
                    mined++;
                    // Wait for item pickup
                    await new Promise(r => setTimeout(r, 500));
                    // Try to collect nearby dropped items
                    const nearbyItems = Object.values(bot.entities).filter(
                        e => e.name === 'item' && e.position.distanceTo(bot.entity.position) < 5
                    );
                    for (const item of nearbyItems.slice(0, 3)) {
                        try {
                            await bot.pathfinder.goto(
                                new GoalNear(item.position.x, item.position.y, item.position.z, 0)
                            );
                        } catch (_) {}
                    }
                    await new Promise(r => setTimeout(r, 300));
                } catch (digErr) {
                    return {
                        success: mined > 0,
                        message: `Mined ${mined}/${count}. Dig failed: ${digErr.message}`
                    };
                }
            }
            return { success: true, message: `Mined ${mined} ${block_name}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── INVENTORY & CRAFTING ───

    actions.equip_item = async ({ item_name, destination = 'hand' }) => {
        try {
            const item = bot.inventory.items().find(i => i.name.includes(item_name));
            if (!item) {
                return { success: false, message: `No ${item_name} in inventory` };
            }
            await bot.equip(item, destination);
            return { success: true, message: `Equipped ${item.name}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.craft_item = async ({ item_name, count = 1 }) => {
        try {
            const item = mcData.itemsByName[item_name];
            if (!item) {
                return { success: false, message: `Unknown item: ${item_name}. Use underscores (e.g. oak_planks)` };
            }

            const getInvStr = () => bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');
            
            // First try without crafting table (2x2 recipes: planks, sticks, crafting_table)
            let recipes = bot.recipesFor(item.id, null, 1, null);
            if (recipes.length > 0) {
                for (let i = 0; i < count; i++) {
                    await bot.craft(recipes[0], 1, null);
                }
                return { success: true, message: `Crafted ${count} ${item_name}` };
            }

            // Need a crafting table (3x3 recipe) - find placed one, or place from inventory, or craft one
            let craftingTable = bot.findBlock({
                matching: mcData.blocksByName.crafting_table?.id,
                maxDistance: 32
            });

            if (!craftingTable) {
                // Check if we already have a crafting table in inventory
                let tableInv = bot.inventory.items().find(i => i.name === 'crafting_table');
                
                if (!tableInv) {
                    // Need to craft one - requires 4 planks
                    const hasPlanks = bot.inventory.items().find(i => i.name.includes('planks'));
                    if (!hasPlanks || hasPlanks.count < 4) {
                        return { success: false, message: `Need crafting table. Have no table and < 4 planks. Have: [${getInvStr()}]. Craft more planks from logs first!` };
                    }
                    const tableItem = mcData.itemsByName.crafting_table;
                    const tableRecipes = bot.recipesFor(tableItem.id, null, 1, null);
                    if (tableRecipes.length === 0) {
                        return { success: false, message: `Cannot craft crafting_table. Have: [${getInvStr()}]` };
                    }
                    await bot.craft(tableRecipes[0], 1, null);
                    tableInv = bot.inventory.items().find(i => i.name === 'crafting_table');
                }

                // Place the crafting table from inventory
                if (tableInv) {
                    await bot.equip(tableInv, 'hand');
                    const pos = bot.entity.position.floored();
                    let placed = false;
                    
                    // Try multiple placement spots
                    const offsets = [
                        { block: pos.offset(1, -1, 0), face: vec3(0, 1, 0) },
                        { block: pos.offset(0, -1, 1), face: vec3(0, 1, 0) },
                        { block: pos.offset(-1, -1, 0), face: vec3(0, 1, 0) },
                        { block: pos.offset(0, -1, -1), face: vec3(0, 1, 0) },
                        { block: pos.offset(0, -1, 0), face: vec3(1, 0, 0) },
                    ];
                    for (const { block: bpos, face } of offsets) {
                        const b = bot.blockAt(bpos);
                        if (b && b.name !== 'air') {
                            try {
                                await bot.placeBlock(b, face);
                                await new Promise(r => setTimeout(r, 500));
                                placed = true;
                                break;
                            } catch (_) {}
                        }
                    }
                    if (!placed) {
                        return { success: false, message: `Have crafting_table but failed to place it` };
                    }
                }

                // Find the placed table
                craftingTable = bot.findBlock({
                    matching: mcData.blocksByName.crafting_table?.id,
                    maxDistance: 10
                });
                if (!craftingTable) {
                    return { success: false, message: `Crafting table not found after placing` };
                }
            }

            // Move to crafting table and craft
            await bot.pathfinder.goto(
                new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 3)
            );

            recipes = bot.recipesFor(item.id, null, 1, craftingTable);
            if (recipes.length === 0) {
                const invStr = getInvStr();
                return { success: false, message: `No recipe for ${item_name} at crafting table. Have: [${invStr}]. Missing materials?` };
            }

            for (let i = 0; i < count; i++) {
                await bot.craft(recipes[0], 1, craftingTable);
            }
            return { success: true, message: `Crafted ${count} ${item_name} (at crafting table)` };
        } catch (e) {
            const invStr = bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');
            return { success: false, message: `Failed: ${e.message}. Inventory: [${invStr}]` };
        }
    };

    // ─── PLACEMENT ───

    actions.place_block = async ({ x, y, z, block_name }) => {
        try {
            const item = bot.inventory.items().find(i => i.name.includes(block_name));
            if (!item) {
                return { success: false, message: `No ${block_name} in inventory` };
            }
            await bot.equip(item, 'hand');
            const referenceBlock = bot.blockAt(vec3(x, y, z));
            if (!referenceBlock || referenceBlock.name === 'air') {
                return { success: false, message: `No reference block at (${x}, ${y}, ${z})` };
            }
            await bot.placeBlock(referenceBlock, vec3(0, 1, 0));
            return { success: true, message: `Placed ${block_name}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── COMBAT ───

    actions.attack_nearest = async ({ entity_type }) => {
        try {
            const entity = bot.nearestEntity(e =>
                e.name === entity_type || e.displayName === entity_type
            );
            if (!entity) {
                return { success: false, message: `No ${entity_type} nearby` };
            }
            await bot.pathfinder.goto(new GoalNear(entity.position.x, entity.position.y, entity.position.z, 2));
            bot.attack(entity);
            return { success: true, message: `Attacked ${entity_type}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── SMELTING ───

    actions.smelt_item = async ({ item_name, fuel = 'auto', count = 1 }) => {
        try {
            const getInvStr = () => bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');

            // Find the input item in inventory
            const inputItem = bot.inventory.items().find(i => i.name === item_name);
            if (!inputItem || inputItem.count < count) {
                return { success: false, message: `Need ${count} ${item_name} but have ${inputItem ? inputItem.count : 0}. Inventory: [${getInvStr()}]` };
            }

            // Find or craft+place a furnace
            let furnaceBlock = bot.findBlock({
                matching: mcData.blocksByName.furnace?.id,
                maxDistance: 32
            });

            if (!furnaceBlock) {
                // Try to craft a furnace (needs 8 cobblestone)
                const cobble = bot.inventory.items().find(i => i.name === 'cobblestone');
                if (!cobble || cobble.count < 8) {
                    return { success: false, message: `Need furnace but don't have 8 cobblestone to craft one. Have: [${getInvStr()}]` };
                }

                // Need crafting table for furnace recipe
                let craftingTable = bot.findBlock({
                    matching: mcData.blocksByName.crafting_table?.id,
                    maxDistance: 32
                });
                if (!craftingTable) {
                    // Try to place one
                    const tableInv = bot.inventory.items().find(i => i.name === 'crafting_table');
                    if (!tableInv) {
                        const planks = bot.inventory.items().find(i => i.name.includes('planks'));
                        if (!planks || planks.count < 4) {
                            return { success: false, message: `Need crafting table to craft furnace but no table or planks. Have: [${getInvStr()}]` };
                        }
                        const tableItem = mcData.itemsByName.crafting_table;
                        const tableRecipes = bot.recipesFor(tableItem.id, null, 1, null);
                        if (tableRecipes.length > 0) await bot.craft(tableRecipes[0], 1, null);
                    }
                    // Place crafting table
                    const tInv = bot.inventory.items().find(i => i.name === 'crafting_table');
                    if (tInv) {
                        await bot.equip(tInv, 'hand');
                        const pos = bot.entity.position.floored();
                        const below = bot.blockAt(pos.offset(1, -1, 0));
                        if (below && below.name !== 'air') {
                            try { await bot.placeBlock(below, vec3(0, 1, 0)); } catch(_) {}
                            await new Promise(r => setTimeout(r, 500));
                        }
                    }
                    craftingTable = bot.findBlock({ matching: mcData.blocksByName.crafting_table?.id, maxDistance: 10 });
                }

                if (craftingTable) {
                    await bot.pathfinder.goto(new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 3));
                    const furnaceItem = mcData.itemsByName.furnace;
                    const furnaceRecipes = bot.recipesFor(furnaceItem.id, null, 1, craftingTable);
                    if (furnaceRecipes.length > 0) {
                        await bot.craft(furnaceRecipes[0], 1, craftingTable);
                    } else {
                        return { success: false, message: `No furnace recipe available. Have: [${getInvStr()}]` };
                    }
                }

                // Place the furnace
                const furnaceInv = bot.inventory.items().find(i => i.name === 'furnace');
                if (furnaceInv) {
                    await bot.equip(furnaceInv, 'hand');
                    const pos = bot.entity.position.floored();
                    const offsets = [
                        { block: pos.offset(2, -1, 0), face: vec3(0, 1, 0) },
                        { block: pos.offset(0, -1, 2), face: vec3(0, 1, 0) },
                        { block: pos.offset(-2, -1, 0), face: vec3(0, 1, 0) },
                        { block: pos.offset(0, -1, 0), face: vec3(1, 0, 0) },
                    ];
                    for (const { block: bpos, face } of offsets) {
                        const b = bot.blockAt(bpos);
                        if (b && b.name !== 'air') {
                            try {
                                await bot.placeBlock(b, face);
                                await new Promise(r => setTimeout(r, 500));
                                break;
                            } catch (_) {}
                        }
                    }
                }

                furnaceBlock = bot.findBlock({ matching: mcData.blocksByName.furnace?.id, maxDistance: 10 });
                if (!furnaceBlock) {
                    return { success: false, message: `Failed to place furnace` };
                }
            }

            // Move to furnace
            await bot.pathfinder.goto(new GoalNear(furnaceBlock.position.x, furnaceBlock.position.y, furnaceBlock.position.z, 3));

            // Open furnace
            const furnace = await bot.openFurnace(furnaceBlock);

            // Find fuel
            let fuelItem;
            if (fuel === 'auto') {
                // Priority: coal > charcoal > oak_planks > any planks > any log
                const fuelPriority = ['coal', 'charcoal', 'oak_planks', 'birch_planks', 'acacia_planks',
                                      'spruce_planks', 'jungle_planks', 'dark_oak_planks',
                                      'oak_log', 'birch_log', 'acacia_log', 'spruce_log'];
                for (const fname of fuelPriority) {
                    fuelItem = bot.inventory.items().find(i => i.name === fname);
                    if (fuelItem) break;
                }
                if (!fuelItem) {
                    // Any wood-like item
                    fuelItem = bot.inventory.items().find(i => 
                        i.name.includes('planks') || i.name.includes('log') || i.name === 'coal' || i.name === 'stick'
                    );
                }
            } else {
                fuelItem = bot.inventory.items().find(i => i.name === fuel);
            }

            if (!fuelItem) {
                furnace.close();
                return { success: false, message: `No fuel available. Have: [${getInvStr()}]. Need coal, planks, or logs as fuel.` };
            }

            // Put fuel and input
            await furnace.putFuel(fuelItem.type, null, Math.min(fuelItem.count, count));
            await new Promise(r => setTimeout(r, 200));
            await furnace.putInput(inputItem.type, null, count);

            // Wait for smelting (each item takes ~10s)
            const waitTime = count * 11000;
            await new Promise(r => setTimeout(r, waitTime));

            // Take output
            const output = furnace.outputItem();
            if (output) {
                await furnace.takeOutput();
            }

            furnace.close();
            await new Promise(r => setTimeout(r, 500));

            const newInv = getInvStr();
            return { success: true, message: `Smelted ${count} ${item_name}. Inventory: [${newInv}]` };
        } catch (e) {
            const invStr = bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');
            return { success: false, message: `Smelt failed: ${e.message}. Inventory: [${invStr}]` };
        }
    };

    // ─── UTILITY ───

    actions.wait = async ({ seconds = 1 }) => {
        await new Promise(r => setTimeout(r, seconds * 1000));
        return { success: true, message: `Waited ${seconds}s` };
    };

    actions.collect_nearby_items = async () => {
        try {
            // Walk toward nearby items
            const items = Object.values(bot.entities).filter(e => e.name === 'item');
            if (items.length === 0) {
                return { success: true, message: 'No items nearby' };
            }
            for (const item of items.slice(0, 5)) {
                try {
                    await bot.pathfinder.goto(
                        new GoalNear(item.position.x, item.position.y, item.position.z, 0)
                    );
                } catch (_) { /* might be picked up already */ }
            }
            await new Promise(r => setTimeout(r, 500));
            return { success: true, message: `Collected nearby items` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.scan_surroundings = async ({ radius = 8 }) => {
        try {
            const pos = bot.entity.position.floored();
            const blocks = {};
            for (let dx = -radius; dx <= radius; dx++) {
                for (let dy = -3; dy <= 3; dy++) {
                    for (let dz = -radius; dz <= radius; dz++) {
                        const block = bot.blockAt(pos.offset(dx, dy, dz));
                        if (block && block.name !== 'air') {
                            blocks[block.name] = (blocks[block.name] || 0) + 1;
                        }
                    }
                }
            }
            // Sort by count, top 15
            const sorted = Object.entries(blocks)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15)
                .map(([name, count]) => `${name}: ${count}`);
            return {
                success: true,
                message: `Nearby blocks (r=${radius}): ${sorted.join(', ')}`,
                data: blocks
            };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    actions.drop_item = async ({ item_name, count = 1 }) => {
        try {
            const item = bot.inventory.items().find(i => i.name.includes(item_name));
            if (!item) return { success: false, message: `No ${item_name} in inventory` };
            await bot.tossStack(item);
            return { success: true, message: `Dropped ${item_name}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    return actions;
}

module.exports = { setupActions };