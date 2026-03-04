/**
 * actions.js - Available bot actions for MemCraft agent
 * v2: Fixes item pickup, block placement, smelt recipe resolution
 * Each action returns a result object: { success: bool, message: string, data?: any }
 */

const { goals: { GoalNear, GoalBlock, GoalXZ } } = require('mineflayer-pathfinder');
const { Movements } = require('mineflayer-pathfinder');
const vec3 = require('vec3');

function setupActions(bot, mcData) {
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);

    const actions = {};

    // ─── HELPERS ───

    /**
     * Try to place a block item from inventory onto a nearby solid surface.
     * Tries many offsets, and if all fail, digs a spot to create a placement surface.
     * Returns the placed block position or null.
     */
    async function placeBlockFromInventory(itemName) {
        const item = bot.inventory.items().find(i => i.name === itemName);
        if (!item) return null;

        await bot.equip(item, 'hand');
        const pos = bot.entity.position.floored();

        // Expanded offset list: try adjacent, then further out
        const offsets = [
            // Standard: place on top of block below adjacent
            { block: pos.offset(1, -1, 0), face: vec3(0, 1, 0) },
            { block: pos.offset(-1, -1, 0), face: vec3(0, 1, 0) },
            { block: pos.offset(0, -1, 1), face: vec3(0, 1, 0) },
            { block: pos.offset(0, -1, -1), face: vec3(0, 1, 0) },
            // Place on side of block next to us
            { block: pos.offset(1, 0, 0), face: vec3(-1, 0, 0) },
            { block: pos.offset(-1, 0, 0), face: vec3(1, 0, 0) },
            { block: pos.offset(0, 0, 1), face: vec3(0, 0, -1) },
            { block: pos.offset(0, 0, -1), face: vec3(0, 0, 1) },
            // Place on block directly below us
            { block: pos.offset(0, -1, 0), face: vec3(0, 1, 0) },
            // Further out
            { block: pos.offset(2, -1, 0), face: vec3(0, 1, 0) },
            { block: pos.offset(-2, -1, 0), face: vec3(0, 1, 0) },
            { block: pos.offset(0, -1, 2), face: vec3(0, 1, 0) },
            { block: pos.offset(0, -1, -2), face: vec3(0, 1, 0) },
        ];

        for (const { block: bpos, face } of offsets) {
            const b = bot.blockAt(bpos);
            if (b && b.name !== 'air' && b.name !== 'water' && b.name !== 'lava') {
                try {
                    await bot.placeBlock(b, face);
                    await new Promise(r => setTimeout(r, 500));
                    return bpos.plus(face);
                } catch (_) {}
            }
        }

        // Last resort: dig a block in front to create a wall surface, then place against it
        try {
            const frontPos = pos.offset(1, 0, 0);
            const frontBlock = bot.blockAt(frontPos);
            if (frontBlock && frontBlock.name !== 'air') {
                await bot.dig(frontBlock);
                await new Promise(r => setTimeout(r, 300));
                const wallBlock = bot.blockAt(pos.offset(2, 0, 0));
                if (wallBlock && wallBlock.name !== 'air') {
                    const reItem = bot.inventory.items().find(i => i.name === itemName);
                    if (reItem) {
                        await bot.equip(reItem, 'hand');
                        await bot.placeBlock(wallBlock, vec3(-1, 0, 0));
                        await new Promise(r => setTimeout(r, 500));
                        return frontPos;
                    }
                }
            }
        } catch (_) {}

        return null;
    }

    /**
     * Count total of an item across all inventory stacks.
     */
    function countInventoryItem(itemName) {
        return bot.inventory.items()
            .filter(i => i.name === itemName)
            .reduce((sum, i) => sum + i.count, 0);
    }

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
            // Block name aliases: map what agent asks for -> what block exists in world
            const blockAliases = {
                'cobblestone': 'stone',
                'raw_iron': 'iron_ore',
                'raw_copper': 'copper_ore',
                'raw_gold': 'gold_ore',
                'diamond': 'diamond_ore',
                'coal': 'coal_ore',
                'redstone': 'redstone_ore',
                'lapis_lazuli': 'lapis_ore',
            };

            // What item the agent actually receives from mining
            const dropMap = {
                'stone': 'cobblestone',
                'iron_ore': 'raw_iron',
                'copper_ore': 'raw_copper',
                'gold_ore': 'raw_gold',
                'diamond_ore': 'diamond',
                'coal_ore': 'coal',
                'redstone_ore': 'redstone',
                'lapis_ore': 'lapis_lazuli',
            };

            const actualBlock = blockAliases[block_name] || block_name;
            const expectedDrop = dropMap[actualBlock] || actualBlock;

            const blockType = mcData.blocksByName[actualBlock];
            if (!blockType) {
                return { success: false, message: `Unknown block: ${block_name} (tried ${actualBlock}). Use stone instead of cobblestone.` };
            }

            // Auto-equip the best tool
            const needsPickaxe = ['stone', 'cobblestone', 'iron_ore', 'coal_ore', 'copper_ore',
                                  'gold_ore', 'diamond_ore', 'lapis_ore', 'redstone_ore', 'furnace',
                                  'deepslate', 'andesite', 'granite', 'diorite'];
            const needsAxe = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log',
                              'dark_oak_log', 'mangrove_log', 'oak_planks', 'birch_planks',
                              'spruce_planks', 'crafting_table'];

            if (needsPickaxe.includes(actualBlock)) {
                // Pick the best pickaxe available
                const pickPriority = ['diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe'];
                for (const pickName of pickPriority) {
                    const pick = bot.inventory.items().find(i => i.name === pickName);
                    if (pick) { try { await bot.equip(pick, 'hand'); } catch(_) {} break; }
                }
            } else if (needsAxe.includes(actualBlock)) {
                const axe = bot.inventory.items().find(i => i.name.includes('axe') && !i.name.includes('pickaxe'));
                if (axe) try { await bot.equip(axe, 'hand'); } catch(_) {}
            }

            const beforeCount = countInventoryItem(expectedDrop);
            let mined = 0;

            for (let i = 0; i < count; i++) {
                const block = bot.findBlock({
                    matching: blockType.id,
                    maxDistance: 32
                });
                if (!block) {
                    const gained = countInventoryItem(expectedDrop) - beforeCount;
                    return {
                        success: gained > 0,
                        message: gained > 0
                            ? `Found and broke ${mined} ${actualBlock}, collected ${gained} ${expectedDrop}. No more nearby.`
                            : `No ${actualBlock} found within 32 blocks. Try moving to a new area or mining 'stone' for cobblestone.`
                    };
                }

                const targetPos = block.position.clone();

                try {
                    await bot.pathfinder.goto(new GoalNear(targetPos.x, targetPos.y, targetPos.z, 3));
                } catch (moveErr) {
                    continue;
                }

                const freshBlock = bot.blockAt(targetPos);
                if (!freshBlock || freshBlock.name === 'air') {
                    continue;
                }

                try {
                    await bot.dig(freshBlock);
                    mined++;

                    // Wait for drop + actively collect
                    await new Promise(r => setTimeout(r, 600));
                    for (let pickup = 0; pickup < 3; pickup++) {
                        const nearbyItems = Object.values(bot.entities).filter(
                            e => e.name === 'item' && e.position.distanceTo(bot.entity.position) < 6
                        );
                        if (nearbyItems.length === 0) break;
                        for (const item of nearbyItems.slice(0, 3)) {
                            try {
                                await bot.pathfinder.goto(
                                    new GoalNear(item.position.x, item.position.y, item.position.z, 0)
                                );
                            } catch (_) {}
                        }
                        await new Promise(r => setTimeout(r, 400));
                    }
                    await new Promise(r => setTimeout(r, 300));
                } catch (digErr) {
                    const gained = countInventoryItem(expectedDrop) - beforeCount;
                    return {
                        success: gained > 0,
                        message: `Broke ${mined}/${count}. Dig failed: ${digErr.message}. Collected ${gained} ${expectedDrop}.`
                    };
                }
            }

            const gained = countInventoryItem(expectedDrop) - beforeCount;
            if (gained < mined && gained < count) {
                return {
                    success: gained > 0,
                    message: `Broke ${mined} ${actualBlock} but only picked up ${gained} ${expectedDrop}. Some items may have fallen into holes. Try collect_nearby_items.`
                };
            }
            return { success: true, message: `Mined ${mined} ${block_name} (got ${gained} ${expectedDrop})` };
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
                return { success: false, message: `Unknown item: ${item_name}. Use exact names with underscores (e.g. oak_planks not planks, stick not sticks)` };
            }

            const getInvStr = () => bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');

            // First try without crafting table (2x2 recipes)
            let recipes = bot.recipesFor(item.id, null, 1, null);
            if (recipes.length > 0) {
                let crafted = 0;
                for (let i = 0; i < count; i++) {
                    recipes = bot.recipesFor(item.id, null, 1, null);
                    if (recipes.length === 0) break;
                    await bot.craft(recipes[0], 1, null);
                    crafted++;
                }
                if (crafted < count) {
                    return { success: crafted > 0, message: `Crafted ${crafted}/${count} ${item_name} (ran out of materials). Have: [${getInvStr()}]` };
                }
                return { success: true, message: `Crafted ${count} ${item_name}` };
            }

            // Need a crafting table (3x3 recipe)
            let craftingTable = bot.findBlock({
                matching: mcData.blocksByName.crafting_table?.id,
                maxDistance: 32
            });
            let wePlacedTable = false;

            if (!craftingTable) {
                let tableInv = bot.inventory.items().find(i => i.name === 'crafting_table');

                if (!tableInv) {
                    const totalPlanks = bot.inventory.items()
                        .filter(i => i.name.includes('planks'))
                        .reduce((sum, i) => sum + i.count, 0);
                    if (totalPlanks < 4) {
                        return { success: false, message: `Need crafting table. No table and only ${totalPlanks}/4 planks. Have: [${getInvStr()}]. Mine logs and craft planks first.` };
                    }
                    const tableItem = mcData.itemsByName.crafting_table;
                    const tableRecipes = bot.recipesFor(tableItem.id, null, 1, null);
                    if (tableRecipes.length === 0) {
                        return { success: false, message: `Cannot craft crafting_table. Have: [${getInvStr()}]` };
                    }
                    await bot.craft(tableRecipes[0], 1, null);
                }

                // Place using robust helper
                const placed = await placeBlockFromInventory('crafting_table');
                if (!placed) {
                    return { success: false, message: `Have crafting_table but failed to place it. Move to flat open ground and try again.` };
                }
                wePlacedTable = true;

                craftingTable = bot.findBlock({
                    matching: mcData.blocksByName.crafting_table?.id,
                    maxDistance: 10
                });
                if (!craftingTable) {
                    return { success: false, message: `Crafting table not found after placing` };
                }
            }

            await bot.pathfinder.goto(
                new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 3)
            );

            recipes = bot.recipesFor(item.id, null, 1, craftingTable);
            if (recipes.length === 0) {
                return { success: false, message: `No recipe for ${item_name} at crafting table. Have: [${getInvStr()}]. Missing materials?` };
            }

            let crafted = 0;
            for (let i = 0; i < count; i++) {
                recipes = bot.recipesFor(item.id, null, 1, craftingTable);
                if (recipes.length === 0) break;
                await bot.craft(recipes[0], 1, craftingTable);
                crafted++;
            }

            // Pick up crafting table after use so we carry it with us
            if (wePlacedTable && craftingTable) {
                try {
                    const tableBlock = bot.blockAt(craftingTable.position);
                    if (tableBlock && tableBlock.name === 'crafting_table') {
                        await bot.dig(tableBlock);
                        await new Promise(r => setTimeout(r, 500));
                        // Walk to pick up the drop
                        const drops = Object.values(bot.entities).filter(
                            e => e.name === 'item' && e.position.distanceTo(bot.entity.position) < 5
                        );
                        for (const drop of drops.slice(0, 2)) {
                            try { await bot.pathfinder.goto(new GoalNear(drop.position.x, drop.position.y, drop.position.z, 0)); } catch(_) {}
                        }
                        await new Promise(r => setTimeout(r, 300));
                    }
                } catch (_) { /* non-critical if pickup fails */ }
            }

            if (crafted < count) {
                return { success: crafted > 0, message: `Crafted ${crafted}/${count} ${item_name} (ran out of materials). Have: [${getInvStr()}]` };
            }
            return { success: true, message: `Crafted ${count} ${item_name} (at crafting table)` };
        } catch (e) {
            const invStr = bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');
            return { success: false, message: `Failed: ${e.message}. Inventory: [${invStr}]` };
        }
    };

    // ─── SMELTING ───

    const SMELT_RECIPES = {
        'iron_ingot': 'raw_iron',
        'gold_ingot': 'raw_gold',
        'copper_ingot': 'raw_copper',
        'glass': 'sand',
        'stone': 'cobblestone',
        'smooth_stone': 'stone',
        'cooked_beef': 'beef',
        'cooked_porkchop': 'porkchop',
        'cooked_chicken': 'chicken',
        'cooked_mutton': 'mutton',
        'cooked_cod': 'cod',
        'cooked_salmon': 'salmon',
        'brick': 'clay_ball',
        'charcoal': 'oak_log',
        'dried_kelp': 'kelp',
    };

    const SMELT_INPUTS = {};
    for (const [output, input] of Object.entries(SMELT_RECIPES)) {
        SMELT_INPUTS[input] = output;
    }

    // Common wrong names agents use
    const SMELT_ALIASES = {
        'iron_ore': 'raw_iron',
        'gold_ore': 'raw_gold',
        'copper_ore': 'raw_copper',
    };

    actions.smelt_item = async ({ item_name, count = 1 }) => {
        try {
            const getInvStr = () => bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');

            // Resolve input/output names
            let inputItem, outputName;
            if (SMELT_ALIASES[item_name]) {
                inputItem = SMELT_ALIASES[item_name];
                outputName = SMELT_INPUTS[inputItem] || inputItem;
            } else if (SMELT_RECIPES[item_name]) {
                inputItem = SMELT_RECIPES[item_name];
                outputName = item_name;
            } else if (SMELT_INPUTS[item_name]) {
                inputItem = item_name;
                outputName = SMELT_INPUTS[item_name];
            } else {
                inputItem = item_name;
                outputName = item_name;
            }

            // Check input material (sum across stacks)
            const inputCount = countInventoryItem(inputItem);
            if (inputCount < count) {
                return {
                    success: false,
                    message: `Need ${count} ${inputItem} (to smelt into ${outputName}) but have ${inputCount}. Inventory: [${getInvStr()}]`
                };
            }

            // Find fuel
            const fuelPriority = ['coal', 'charcoal', 'oak_planks', 'birch_planks', 'spruce_planks',
                                  'jungle_planks', 'acacia_planks', 'dark_oak_planks',
                                  'oak_log', 'birch_log', 'spruce_log', 'stick'];
            let fuelItem = null;
            for (const fuelName of fuelPriority) {
                fuelItem = bot.inventory.items().find(i => i.name === fuelName);
                if (fuelItem) break;
            }
            if (!fuelItem) {
                fuelItem = bot.inventory.items().find(i =>
                    i.name.includes('planks') || i.name.includes('_log') || i.name === 'coal'
                );
            }
            if (!fuelItem) {
                return { success: false, message: `No fuel for smelting. Need coal, planks, or logs. Inventory: [${getInvStr()}]` };
            }

            // Find or place furnace
            let furnaceBlock = bot.findBlock({
                matching: mcData.blocksByName.furnace?.id,
                maxDistance: 32
            });

            if (!furnaceBlock) {
                let furnaceInv = bot.inventory.items().find(i => i.name === 'furnace');

                if (!furnaceInv) {
                    const cobbleTotal = countInventoryItem('cobblestone');
                    if (cobbleTotal < 8) {
                        return { success: false, message: `Need furnace. Have ${cobbleTotal}/8 cobblestone. Mine more stone first. Inventory: [${getInvStr()}]` };
                    }

                    let craftingTable = bot.findBlock({
                        matching: mcData.blocksByName.crafting_table?.id,
                        maxDistance: 32
                    });
                    if (!craftingTable) {
                        let tableInv = bot.inventory.items().find(i => i.name === 'crafting_table');
                        if (!tableInv) {
                            const totalPlanks = bot.inventory.items()
                                .filter(i => i.name.includes('planks'))
                                .reduce((sum, i) => sum + i.count, 0);
                            if (totalPlanks < 4) {
                                return { success: false, message: `Need crafting table for furnace but no planks. Inventory: [${getInvStr()}]` };
                            }
                            const tableItemData = mcData.itemsByName.crafting_table;
                            const tableRecipes = bot.recipesFor(tableItemData.id, null, 1, null);
                            if (tableRecipes.length > 0) await bot.craft(tableRecipes[0], 1, null);
                        }
                        const placed = await placeBlockFromInventory('crafting_table');
                        if (!placed) return { success: false, message: `Cannot place crafting_table for furnace. Move to flat ground.` };
                        craftingTable = bot.findBlock({ matching: mcData.blocksByName.crafting_table?.id, maxDistance: 10 });
                    }

                    if (craftingTable) {
                        await bot.pathfinder.goto(new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 3));
                        const furnaceItemData = mcData.itemsByName.furnace;
                        const furnaceRecipes = bot.recipesFor(furnaceItemData.id, null, 1, craftingTable);
                        if (furnaceRecipes.length > 0) {
                            await bot.craft(furnaceRecipes[0], 1, craftingTable);
                            furnaceInv = bot.inventory.items().find(i => i.name === 'furnace');
                        }
                    }
                    if (!furnaceInv) return { success: false, message: `Failed to craft furnace. Inventory: [${getInvStr()}]` };
                }

                const placed = await placeBlockFromInventory('furnace');
                if (!placed) return { success: false, message: `Have furnace but failed to place it. Move to flat open ground and try again.` };

                furnaceBlock = bot.findBlock({ matching: mcData.blocksByName.furnace?.id, maxDistance: 10 });
                if (!furnaceBlock) return { success: false, message: `Furnace not found after placing` };
            }

            await bot.pathfinder.goto(new GoalNear(furnaceBlock.position.x, furnaceBlock.position.y, furnaceBlock.position.z, 3));

            const furnaceEntity = await bot.openFurnace(furnaceBlock);

            const inputSlotItem = bot.inventory.items().find(i => i.name === inputItem);
            if (inputSlotItem) await furnaceEntity.putInput(inputSlotItem.type, null, count);

            await new Promise(r => setTimeout(r, 200));
            const currentFuel = bot.inventory.items().find(i => i.name === fuelItem.name);
            if (currentFuel) {
                const fuelNeeded = Math.max(Math.ceil(count / 8), 1);
                await furnaceEntity.putFuel(currentFuel.type, null, Math.min(currentFuel.count, fuelNeeded));
            }

            await new Promise(r => setTimeout(r, count * 11 * 1000));

            const output = furnaceEntity.outputItem();
            let smelted = 0;
            if (output) {
                smelted = output.count;
                await furnaceEntity.takeOutput();
            }

            try {
                if (furnaceEntity.inputItem()) await furnaceEntity.takeInput();
                if (furnaceEntity.fuelItem()) await furnaceEntity.takeFuel();
            } catch (_) {}

            furnaceEntity.close();

            if (smelted >= count) return { success: true, message: `Smelted ${smelted} ${outputName} (from ${inputItem})` };
            if (smelted > 0) return { success: true, message: `Partially smelted ${smelted}/${count} ${outputName}. May need more fuel.` };
            return { success: false, message: `Smelting produced 0 ${outputName}. Inventory: [${getInvStr()}]` };
        } catch (e) {
            const invStr = bot.inventory.items().map(i => `${i.name}:${i.count}`).join(', ');
            return { success: false, message: `Smelt failed: ${e.message}. Inventory: [${invStr}]` };
        }
    };

    // ─── PLACEMENT ───

    actions.place_block = async ({ x, y, z, block_name }) => {
        try {
            const item = bot.inventory.items().find(i => i.name.includes(block_name));
            if (!item) return { success: false, message: `No ${block_name} in inventory` };
            await bot.equip(item, 'hand');
            const referenceBlock = bot.blockAt(vec3(x, y, z));
            if (!referenceBlock || referenceBlock.name === 'air') return { success: false, message: `No reference block at (${x}, ${y}, ${z})` };
            await bot.placeBlock(referenceBlock, vec3(0, 1, 0));
            return { success: true, message: `Placed ${block_name}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── COMBAT ───

    actions.attack_nearest = async ({ entity_type }) => {
        try {
            const entity = bot.nearestEntity(e => e.name === entity_type || e.displayName === entity_type);
            if (!entity) return { success: false, message: `No ${entity_type} nearby` };
            await bot.pathfinder.goto(new GoalNear(entity.position.x, entity.position.y, entity.position.z, 2));
            bot.attack(entity);
            return { success: true, message: `Attacked ${entity_type}` };
        } catch (e) {
            return { success: false, message: `Failed: ${e.message}` };
        }
    };

    // ─── UTILITY ───

    actions.wait = async ({ seconds = 1 }) => {
        await new Promise(r => setTimeout(r, seconds * 1000));
        return { success: true, message: `Waited ${seconds}s` };
    };

    actions.collect_nearby_items = async () => {
        try {
            const items = Object.values(bot.entities).filter(e => e.name === 'item');
            if (items.length === 0) return { success: true, message: 'No items nearby' };
            let collected = 0;
            for (const item of items.slice(0, 8)) {
                try {
                    await bot.pathfinder.goto(new GoalNear(item.position.x, item.position.y, item.position.z, 0));
                    collected++;
                } catch (_) {}
            }
            await new Promise(r => setTimeout(r, 500));
            return { success: true, message: `Walked to ${collected} nearby items` };
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
            const sorted = Object.entries(blocks)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15)
                .map(([name, count]) => `${name}: ${count}`);
            return { success: true, message: `Nearby blocks (r=${radius}): ${sorted.join(', ')}`, data: blocks };
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