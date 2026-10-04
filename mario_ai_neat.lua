-- Super Mario AI using NEAT plays Super Mario Bros. autonomously in FCEUX.
-- It learns controller decisions from RAM observations, fitness, and evolution.
-- Load this Lua 5.1 file in FCEUX with a compatible NES SMB1 ROM already open.

local AI = {}


-- Load modular components
local smb1_memory = require("scripts.smb1.memory")
local neat_genome = require("scripts.neat.genome")
local neat_mutation = require("scripts.neat.mutation")
local neat_species = require("scripts.neat.species")
local neat_observation = require("scripts.neat.observation")
local neat_population = require("scripts.neat.population")
local q_memory = require("scripts.qmemory")
local hud_graphics = require("scripts.hud")
local persistence = require("scripts.storage")
-- Keep game rules intact. Training resets with a savestate after an attempt.
local SET_TIMER_TO_999_PER_EPISODE = false
local TESTING_INFINITE_LIVES = false
-- FCEUX slot 9 is reserved for the AI's fixed training start. Named slots
-- reload safely without the persist() call that crashes some FCEUX builds.
local USE_FIXED_TRAINING_STATE = true
local TRAINING_SAVESTATE_SLOT = 9
-- Save a training start only near the beginning of a level. If the script is
-- loaded mid-level, wait for a manual reset or Mario's normal respawn.
local LEVEL_START_MAX_X = 128
-- Set true after training to replay the strongest saved genome only.
local PLAY_CHAMPION_ONLY = false
-- Click the small HUD tab in the upper-right corner to show/hide the live
-- network inspector. Drawing can be disabled here to maximize training speed.
local SHOW_NEURAL_INSPECTOR = os.getenv("MARIO_AI_HIDE_HUD")~="1"
local HUD_CLICK_COOLDOWN = 0

local RAM = {
  game_engine_subroutine=0x000E, enemy_present=0x000F, enemy_id=0x0016,
  enemy_state=0x001E, player_page=0x006D, enemy_page=0x006E,
  player_x=0x0086, enemy_x=0x0087, player_vx=0x0057,
  player_vy=0x009F, enemy_y=0x00CF, player_y=0x03B8,
  player_screen_x=0x03AD, death_music=0x0712,
  tiles=0x0500,
  player_size=0x0754, power=0x0756, operation_mode=0x0770,
  world_number=0x075F, level_number=0x075C,
  timer_hundreds=0x07F8, timer_tens=0x07F9, timer_ones=0x07FA,
  lives=0x075A,
}

local ENEMY_NAME = {
  [0x00]="green koopa", [0x02]="buzzy beetle", [0x03]="red koopa",
  [0x05]="hammer bro", [0x06]="goomba", [0x07]="bloober",
  [0x08]="bullet bill", [0x09]="paratroopa", [0x0A]="cheep-cheep",
  [0x0B]="cheep-cheep", [0x0C]="podoboo", [0x0D]="piranha plant",
  [0x0E]="jumping paratroopa", [0x0F]="red paratroopa",
  [0x10]="flying paratroopa", [0x11]="lakitu", [0x12]="spiny",
  [0x14]="flying cheep-cheep", [0x15]="bowser flame", [0x2D]="bowser",
  [0x33]="bullet bill",
}

local NON_SOLID_TILES = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}
local INACTIVE_ENEMY_STATES = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}

local function toSignedByte(value) return value >= 128 and value - 256 or value end
local function clamp(value, low, high) return math.max(low, math.min(high, value)) end

local function readByte(address) return memory.readbyte(address) end

-- Convert the SMB1 RAM layout into one readable snapshot for the learner.
function AI.observe(frameNumber)
  local marioWorldX = readByte(RAM.player_page) * 256 + readByte(RAM.player_x)
  local operationMode = readByte(RAM.operation_mode)
  local playerState = readByte(RAM.game_engine_subroutine)
  local state = {frame=frameNumber,worldX=marioWorldX,worldY=readByte(RAM.player_y)+16,
    screenX=readByte(RAM.player_screen_x),
    horizontalVelocity=toSignedByte(readByte(RAM.player_vx))/16,
    verticalVelocity=toSignedByte(readByte(RAM.player_vy)),playerState=playerState,
    worldNumber=readByte(RAM.world_number),levelNumber=readByte(RAM.level_number),
    size=readByte(RAM.player_size),power=readByte(RAM.power),operationMode=operationMode,
    enemies={},items={},tiles={},phase="playing"}

  if operationMode == 0 then
    state.phase = "title"
  elseif readByte(RAM.death_music) == 1 or playerState == 0x0B then
    state.phase = "death"
  elseif operationMode ~= 1 then
    state.phase = "transition"
  elseif playerState ~= 0x08 then
    state.phase = "locked"
  end
  -- SMB1's game engine uses 4 for flagpole slide and 5 for level end.
  -- $010e/$070f are flagpole animation/collision bytes, not a victory flag.
  -- https://gist.github.com/1wErt3r/4048722
  if operationMode == 1 and (playerState == 0x04 or playerState == 0x05) then
    state.phase = "victory"
  end

  -- Read both metatile pages in one FCEUX call. Keep zero-based tile indices
  -- because the observation and collision code use the SMB1 RAM layout.
  local tileBytes=memory.readbyterange and memory.readbyterange(RAM.tiles,416)
  if type(tileBytes)=="string" and #tileBytes==416 then
    local tileValues={string.byte(tileBytes,1,416)}
    for tileIndex=0,415 do state.tiles[tileIndex]=tileValues[tileIndex+1] end
  else
    for tileIndex=0,415 do state.tiles[tileIndex]=readByte(RAM.tiles+tileIndex) end
  end
  state.grounded = AI.isGrounded(state)
  for enemySlot = 0, 4 do
    if readByte(RAM.enemy_present + enemySlot) ~= 0 then
      local enemyId = readByte(RAM.enemy_id + enemySlot)
      state.enemies[#state.enemies+1] = {
        slot=enemySlot,
        id=enemyId,
        name=ENEMY_NAME[enemyId] or "unknown object",
        status=readByte(RAM.enemy_state + enemySlot),
        worldX=readByte(RAM.enemy_page + enemySlot)*256 + readByte(RAM.enemy_x + enemySlot),
        worldY=readByte(RAM.enemy_y + enemySlot)+24,
        horizontalVelocity=toSignedByte(readByte(0x0058 + enemySlot))/16,
      }
    end
  end
  if readByte(0x0014) == 1 and readByte(0x001B) == 0x2E then
    local itemWorldX = math.floor(marioWorldX/256)*256 + readByte(0x008C)
    if itemWorldX < marioWorldX-128 then
      itemWorldX = itemWorldX+256
    elseif itemWorldX > marioWorldX+128 then
      itemWorldX = itemWorldX-256
    end
    state.items[1] = {kind="powerup",type=readByte(0x0039),worldX=itemWorldX,
      worldY=readByte(0x03BE)+16,heading=readByte(0x004B)}
  end
  return state
end


-- The policy learns with NEAT-style neuroevolution. Each genome controls a
-- real SMB1 play segment; episode fitness selects parents for the next
-- generation. RAM writes are limited to the optional timer and lives test aids.
local SENSOR_RADIUS_TILES = 6
local GRID_WIDTH = SENSOR_RADIUS_TILES * 2 + 1
local GRID_INPUT_COUNT = GRID_WIDTH * GRID_WIDTH
local GLOBAL_INPUT_COUNT = 15
local OBSERVATION_INPUT_COUNT = GRID_INPUT_COUNT + GLOBAL_INPUT_COUNT
local NEURAL_INPUT_COUNT = OBSERVATION_INPUT_COUNT + 1 -- final input is the bias node
local ACTION_COUNT = 6
local DURATION_OUTPUT_COUNT = 4
local NETWORK_OUTPUT_COUNT = ACTION_COUNT + DURATION_OUTPUT_COUNT
local OUTPUT_NODE_OFFSET = 1000000
-- Reserve a high, non-overlapping range for transient temporal inputs. They
-- are synthesized from recent SMB1 observations and are never stored as genes.
local MEMORY_INPUT_OFFSET = 900000
local MEMORY_FEATURE_COUNT = GLOBAL_INPUT_COUNT
local MEMORY_LAGS = {1, 4}
local MEMORY_INPUT_COUNT = MEMORY_FEATURE_COUNT * #MEMORY_LAGS
local ACTION_HOLD_FRAMES = {1, 2, 4, 6}
local NOVELTY_ARCHIVE_LIMIT = 48
local EPISODE_HISTORY_LIMIT = 1000
local TOP_PERFORMER_LIMIT = 5
local EXPERIENCE_CONTEXT_LIMIT = 512
local EXPERIENCE_TRACE_LIMIT = 64
local DEFAULT_POPULATION_SIZE = 300
local SPECIES_DISTANCE_THRESHOLD = 1.0
local MAX_STALE_GENERATIONS = 15
local SAVE_INTERVAL_FRAMES = 600
local INITIAL_PROGRESS_DEADLINE_FRAMES = 180
local MIN_INITIAL_PROGRESS_PIXELS = 16

local ACTION_OPTIONS = {
  {name="run", right=true, B=true},
  {name="jump_run", right=true, B=true, A=true},
  {name="retreat", left=true, B=true},
  {name="brake"},
  {name="jump_place", A=true},
  {name="walk", right=true},
}
local NON_STOMPABLE_ENEMIES = {[0x07]=true,[0x0C]=true,[0x0D]=true,[0x11]=true,[0x12]=true}

local function safeWriteLine(file, value) file:write(value, "\n") end

local function currentTimestamp()
  return os and os.date and os.date("%Y-%m-%d %H:%M:%S") or "unknown"
end

local function writeFile(path,contents)
  local file=io and io.open and io.open(path,"w")
  if not file then return false end
  local wrote=file:write(contents)
  file:close()
  return wrote~=nil
end

local function backupFile(path)
  local source=io and io.open and io.open(path,"r")
  if not source then return true end
  local contents=source:read("*a")
  source:close()
  local backupPath=path..".bak"
  local temporaryBackupPath=backupPath..".tmp"
  if not writeFile(temporaryBackupPath,contents) then return false end
  local renamed=os and os.rename and os.rename(temporaryBackupPath,backupPath)
  if renamed then return true end
  local copied=writeFile(backupPath,contents)
  if os and os.remove then os.remove(temporaryBackupPath) end
  return copied
end

local function memoryNode(index) return MEMORY_INPUT_OFFSET + index end
local function isMemoryInput(nodeId)
  return nodeId >= MEMORY_INPUT_OFFSET and nodeId < MEMORY_INPUT_OFFSET + MEMORY_INPUT_COUNT
end
local function isHiddenNode(nodeId)
  return nodeId > NEURAL_INPUT_COUNT and nodeId < MEMORY_INPUT_OFFSET
end

function AI.inputCount() return NEURAL_INPUT_COUNT end
function AI.outputNode(index) return OUTPUT_NODE_OFFSET + index end
function AI.memoryInputNode(index) return memoryNode(index) end
function AI.sensorIndex(horizontalOffset, verticalOffset)
  local columnIndex = math.floor((horizontalOffset + SENSOR_RADIUS_TILES * 16) / 16)
  local rowIndex = math.floor((verticalOffset + SENSOR_RADIUS_TILES * 16) / 16)
  return rowIndex * GRID_WIDTH + columnIndex + 1
end

local function isSolidTileAtOffset(state, horizontalOffset, verticalOffset)
  local sampledWorldX = state.worldX + horizontalOffset
  local sampledWorldY = state.worldY + verticalOffset - 16
  local columnIndex = math.floor((sampledWorldX+8)/16)
  local rowIndex = math.floor((sampledWorldY-32)/16)
  if rowIndex < 0 or rowIndex >= 13 then return false end
  local tileIndex = (math.floor(columnIndex/16)%2)*208 + rowIndex*16 + columnIndex%16
  local tileId = state.tiles[tileIndex]
  return tileId ~= nil and tileId ~= 0 and not NON_SOLID_TILES[tileId]
end

-- Ground support is inferred from the feet and vertical speed. This gives the
-- network a real landing signal without writing to SMB1 RAM.
function AI.isGrounded(state)
  return state.verticalVelocity==0 and
    (isSolidTileAtOffset(state,-6,16) or isSolidTileAtOffset(state,6,16))
end

local function hasGapAhead(state)
  for horizontalOffset=16,64,16 do
    if not isSolidTileAtOffset(state,horizontalOffset,16) then return 1 end
  end
  return 0
end

local function encodeGridCell(state, horizontalOffset, verticalOffset)
  local sampledWorldX = state.worldX + horizontalOffset
  local sampledWorldY = state.worldY + verticalOffset - 16
  local columnIndex = math.floor((sampledWorldX + 8) / 16)
  local rowIndex = math.floor((sampledWorldY - 32) / 16)
  if rowIndex < 0 or rowIndex >= 13 then return 0 end
  local pageIndex = math.floor(columnIndex / 16) % 2
  local columnWithinPage = columnIndex % 16
  local tileId = state.tiles[pageIndex * 208 + rowIndex * 16 + columnWithinPage]
  local isOccupied = tileId ~= nil and tileId ~= 0 and not NON_SOLID_TILES[tileId]
  local encodedValue = isOccupied and 1 or 0
  for _, enemy in ipairs(state.enemies) do
    if not INACTIVE_ENEMY_STATES[enemy.status]
      and math.abs(enemy.worldX - (state.worldX + horizontalOffset)) <= 8
      and math.abs(enemy.worldY - (state.worldY + verticalOffset)) <= 8 then
      encodedValue = -1
      break
    end
  end
  return encodedValue
end

local function findNearestEnemy(state)
  local nearestEnemy, nearestDistance
  for _, enemy in ipairs(state.enemies) do
    if not INACTIVE_ENEMY_STATES[enemy.status] then
      local horizontalOffset, verticalOffset = enemy.worldX - state.worldX, enemy.worldY - state.worldY
      local weightedDistance = math.abs(horizontalOffset) + math.abs(verticalOffset) * 1.5
      if weightedDistance < 240 and (not nearestDistance or weightedDistance < nearestDistance) then
        nearestEnemy, nearestDistance = enemy, weightedDistance
      end
    end
  end
  return nearestEnemy, nearestDistance
end

-- Build the 184 observed values. AI.evaluateGenome adds the constant bias node.
function AI.buildObservationInputs(state)
  local inputValues = {}
  for verticalOffset=-SENSOR_RADIUS_TILES*16,SENSOR_RADIUS_TILES*16,16 do
    for horizontalOffset=-SENSOR_RADIUS_TILES*16,SENSOR_RADIUS_TILES*16,16 do
      inputValues[#inputValues+1] = encodeGridCell(state,horizontalOffset,verticalOffset)
    end
  end
  local nearestEnemy = findNearestEnemy(state)
  local enemyHorizontalOffset, enemyVerticalOffset = 0, 0
  local enemyHorizontalVelocity, enemyTypeValue = 0, 0
  if nearestEnemy then
    enemyHorizontalOffset = clamp((nearestEnemy.worldX-state.worldX)/128,-1,1)
    enemyVerticalOffset = clamp((nearestEnemy.worldY-state.worldY)/96,-1,1)
    enemyHorizontalVelocity = clamp(nearestEnemy.horizontalVelocity/4,-1,1)
    enemyTypeValue = (nearestEnemy.id or 0)/51
  end
  inputValues[#inputValues+1] = clamp(state.horizontalVelocity/4,-1,1)
  inputValues[#inputValues+1] = clamp(state.verticalVelocity/8,-1,1)
  inputValues[#inputValues+1] = state.grounded and 1 or -1
  inputValues[#inputValues+1] = state.size == 1 and -1 or 1
  inputValues[#inputValues+1] = state.power == 2 and 1 or (state.power == 1 and 0 or -1)
  inputValues[#inputValues+1] = enemyHorizontalOffset
  inputValues[#inputValues+1] = enemyVerticalOffset
  inputValues[#inputValues+1] = enemyHorizontalVelocity
  inputValues[#inputValues+1] = enemyTypeValue
  local visibleItem = state.items and state.items[1]
  if visibleItem then
    inputValues[#inputValues+1] = 1
    inputValues[#inputValues+1] = clamp((visibleItem.worldX-state.worldX)/160,-1,1)
    inputValues[#inputValues+1] = clamp((visibleItem.worldY-state.worldY)/96,-1,1)
    inputValues[#inputValues+1] = clamp((visibleItem.type or 0)/3,-1,1)
  else
    inputValues[#inputValues+1] = -1
    inputValues[#inputValues+1] = 0
    inputValues[#inputValues+1] = 0
    inputValues[#inputValues+1] = 0
  end
  inputValues[#inputValues+1] = hasGapAhead(state)
  inputValues[#inputValues+1] = nearestEnemy~=nil and nearestEnemy.worldX>state.worldX
    and nearestEnemy.worldX-state.worldX<32 and 1 or 0
  return inputValues
end

local function sigmoid(value)
  value = clamp(value,-60,60)
  return 2/(1+math.exp(-4.9*value))-1
end

-- Follow enabled connections when evaluating each output. A split can add a
-- newer hidden node before an older hidden target, so numeric order is unsafe.
function AI.evaluateGenome(genome, inputValues, memoryValues)
  local nodeValues, incomingConnections = {}, {}
  for inputIndex = 1, OBSERVATION_INPUT_COUNT do
    nodeValues[inputIndex] = inputValues[inputIndex] or 0
  end
  nodeValues[NEURAL_INPUT_COUNT] = 1
  for memoryIndex = 1, MEMORY_INPUT_COUNT do
    nodeValues[memoryNode(memoryIndex)] = memoryValues and memoryValues[memoryIndex] or 0
  end
  for _, gene in ipairs(genome.genes) do
    if gene.enabled then
      incomingConnections[gene.targetNode] = incomingConnections[gene.targetNode] or {}
      incomingConnections[gene.targetNode][#incomingConnections[gene.targetNode]+1] = gene
    end
  end
  local evaluating={}
  local function evaluateNode(nodeId)
    if nodeValues[nodeId]~=nil then return nodeValues[nodeId] end
    if nodeId<=NEURAL_INPUT_COUNT or evaluating[nodeId] then return 0 end
    evaluating[nodeId]=true
    local nodeConnections = incomingConnections[nodeId]
    if nodeConnections then
      local weightedSum = 0
      for _, gene in ipairs(nodeConnections) do
        weightedSum = weightedSum + evaluateNode(gene.sourceNode)*gene.weight
      end
      nodeValues[nodeId] = sigmoid(weightedSum)
    else
      nodeValues[nodeId]=0
    end
    evaluating[nodeId]=nil
    return nodeValues[nodeId]
  end
  local actionScores, durationScores = {}, {}
  for actionIndex = 1, ACTION_COUNT do
    actionScores[actionIndex] = evaluateNode(OUTPUT_NODE_OFFSET+actionIndex)
  end
  for durationIndex = 1, DURATION_OUTPUT_COUNT do
    durationScores[durationIndex] = evaluateNode(OUTPUT_NODE_OFFSET+ACTION_COUNT+durationIndex)
  end
  return actionScores,nodeValues,durationScores
end

local function createEmptyGenome()
  return {genes={},fitness=0,adjustedFitness=0,highestHiddenNode=NEURAL_INPUT_COUNT,
    mutationRates={connections=0.8,link=1.0,bias=0.4,node=0.12,enable=0.2,disable=0.2,step=0.1}}
end

local function connectionKey(sourceNode, targetNode)
  return tostring(sourceNode)..":"..tostring(targetNode)
end

-- Reuse one historical number for the same structural connection. This is the
-- NEAT mechanism that aligns homologous genes during crossover.
local function getInnovationNumber(populationState, sourceNode, targetNode)
  local connectionIdentifier = connectionKey(sourceNode,targetNode)
  if not populationState.innovations[connectionIdentifier] then
    populationState.nextInnovation = populationState.nextInnovation + 1
    populationState.innovations[connectionIdentifier] = populationState.nextInnovation
  end
  return populationState.innovations[connectionIdentifier]
end

local function createGene(sourceNode, targetNode, weight, innovationNumber)
  return {sourceNode=sourceNode,targetNode=targetNode,weight=weight,
    enabled=true,innovation=innovationNumber}
end

local function hasConnection(genome, sourceNode, targetNode)
  for _, gene in ipairs(genome.genes) do
    if gene.sourceNode==sourceNode and gene.targetNode==targetNode then return true end
  end
  return false
end

local function chooseRandomNode(genome, populationState, inputOnly)
  -- Most grid cells are empty in a given frame. Give compact movement, enemy,
  -- item, and gap features a better chance of receiving a new connection.
  if not inputOnly and math.random()<0.4 then
    return math.random(GRID_INPUT_COUNT+1,NEURAL_INPUT_COUNT)
  end
  local candidates = {}
  for inputIndex = 1, NEURAL_INPUT_COUNT do
    candidates[#candidates+1] = inputIndex
  end
  for memoryIndex = 1, MEMORY_INPUT_COUNT do
    candidates[#candidates+1] = memoryNode(memoryIndex)
  end
  if not inputOnly then
    for _, gene in ipairs(genome.genes) do
      if isHiddenNode(gene.sourceNode) then candidates[#candidates+1]=gene.sourceNode end
      if isHiddenNode(gene.targetNode) then candidates[#candidates+1]=gene.targetNode end
    end
  end
  return candidates[math.random(#candidates)]
end

local function mutateAddConnection(genome, populationState, biasOnly)
  local sourceNode = biasOnly and NEURAL_INPUT_COUNT or chooseRandomNode(genome,populationState,false)
  local targetNode = OUTPUT_NODE_OFFSET + math.random(NETWORK_OUTPUT_COUNT)
  if not biasOnly then
    local hiddenNodes = {}
    for _, gene in ipairs(genome.genes) do
      if isHiddenNode(gene.sourceNode) then
        hiddenNodes[#hiddenNodes+1] = gene.sourceNode
      end
      if isHiddenNode(gene.targetNode) then
        hiddenNodes[#hiddenNodes+1] = gene.targetNode
      end
    end
    if #hiddenNodes > 0 and math.random(2) == 1 then
      targetNode = hiddenNodes[math.random(#hiddenNodes)]
    end
  end
  if (sourceNode >= targetNode and not isMemoryInput(sourceNode))
    or hasConnection(genome,sourceNode,targetNode) then return end
  table.insert(genome.genes,createGene(sourceNode,targetNode,math.random()*4-2,
    getInnovationNumber(populationState,sourceNode,targetNode)))
end

local function mutateAddNode(genome, populationState)
  local enabledGenes = {}
  for _, gene in ipairs(genome.genes) do
    local existingNode=populationState.splitHistory[gene.innovation]
    if gene.enabled and (not existingNode
      or not hasConnection(genome,gene.sourceNode,existingNode)) then
      enabledGenes[#enabledGenes+1] = gene
    end
  end
  if #enabledGenes == 0 then return end
  local splitGene = enabledGenes[math.random(#enabledGenes)]
  local newHiddenNode=populationState.splitHistory[splitGene.innovation]
  if not newHiddenNode then
    newHiddenNode=(populationState.nextHiddenNode or NEURAL_INPUT_COUNT)+1
    if newHiddenNode>=OUTPUT_NODE_OFFSET then return end
    populationState.nextHiddenNode=newHiddenNode
    populationState.splitHistory[splitGene.innovation]=newHiddenNode
  end
  splitGene.enabled=false
  genome.highestHiddenNode=math.max(genome.highestHiddenNode,newHiddenNode)
  table.insert(genome.genes,createGene(splitGene.sourceNode,newHiddenNode,1,
    getInnovationNumber(populationState,splitGene.sourceNode,newHiddenNode)))
  table.insert(genome.genes,createGene(newHiddenNode,splitGene.targetNode,splitGene.weight,
    getInnovationNumber(populationState,newHiddenNode,splitGene.targetNode)))
end

local function repeatMutationByRate(rate, callback)
  while rate > 0 do
    if math.random() < math.min(1,rate) then callback() end
    rate = rate - 1
  end
end

function AI.mutate(genome, populationState)
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    if mutationName ~= "step" then
      genome.mutationRates[mutationName] = mutationRate
        * (math.random(2) == 1 and 0.95 or 1.05263)
    end
  end
  if math.random() < genome.mutationRates.connections then
    for _, gene in ipairs(genome.genes) do
      if math.random() < 0.9 then
        gene.weight = gene.weight + (math.random()*2-1)*genome.mutationRates.step
      else
        gene.weight = math.random()*4-2
      end
    end
  end
  repeatMutationByRate(genome.mutationRates.link,function() mutateAddConnection(genome,populationState,false) end)
  repeatMutationByRate(genome.mutationRates.bias,function() mutateAddConnection(genome,populationState,true) end)
  repeatMutationByRate(genome.mutationRates.node,function() mutateAddNode(genome,populationState) end)
  repeatMutationByRate(genome.mutationRates.enable,function()
    local disabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if not gene.enabled then disabledGenes[#disabledGenes+1] = gene end
    end
    if #disabledGenes > 0 then disabledGenes[math.random(#disabledGenes)].enabled=true end
  end)
  repeatMutationByRate(genome.mutationRates.disable,function()
    local enabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if gene.enabled then enabledGenes[#enabledGenes+1] = gene end
    end
    if #enabledGenes > 0 then enabledGenes[math.random(#enabledGenes)].enabled=false end
  end)
end

local function cloneGenome(genome)
  local genomeCopy={genes={},fitness=genome.fitness or 0,adjustedFitness=0,
    highestHiddenNode=genome.highestHiddenNode,mutationRates={}}
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    genomeCopy.mutationRates[mutationName] = mutationRate
  end
  for _, gene in ipairs(genome.genes) do
    genomeCopy.genes[#genomeCopy.genes+1]={sourceNode=gene.sourceNode,targetNode=gene.targetNode,weight=gene.weight,
      enabled=gene.enabled,innovation=gene.innovation}
  end
  return genomeCopy
end

-- The signature describes the active policy. Disabled genes do not change
-- current behavior, and rounding avoids treating serialization noise as novelty.
local function genomeSignature(genome)
  local activeGenes={}
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      activeGenes[#activeGenes+1]=table.concat({gene.sourceNode,gene.targetNode,
        string.format("%.4f",gene.weight)},":")
    end
  end
  table.sort(activeGenes)
  return table.concat(activeGenes,"|")
end

function AI.policySignature(genome)
  return genomeSignature(genome)
end

function AI.newGenome(populationState)
  local genome=createEmptyGenome()
  -- Sparse initial networks keep the first generation diverse and inexpensive.
  for actionIndex = 1, ACTION_COUNT do
    local sourceNode = math.random(NEURAL_INPUT_COUNT)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,
      (math.random()*2-1)*0.5,getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  return genome
end

local function createSeededGenome(populationState)
  local genome=createEmptyGenome()
  local enemyHorizontalOffsetInput = GRID_INPUT_COUNT+6
  local itemHorizontalOffsetInput = GRID_INPUT_COUNT+11
  local gapAheadInput = GRID_INPUT_COUNT+14
  local enemyContactInput = GRID_INPUT_COUNT+15
  local function connect(sourceNode, actionIndex, weight)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,weight,
      getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  -- Start from sensible SMB1 play: run on clear ground, jump for an enemy or
  -- pit, and turn back toward a visible reward. Evolution can change all links.
  connect(NEURAL_INPUT_COUNT,1,0.55)
  connect(NEURAL_INPUT_COUNT,2,-0.28)
  connect(enemyHorizontalOffsetInput,2,3.4)
  connect(gapAheadInput,2,3.0)
  connect(itemHorizontalOffsetInput,3,-1.2)
  connect(enemyHorizontalOffsetInput,3,-2.0)
  connect(NEURAL_INPUT_COUNT,5,-0.2)
  connect(enemyContactInput,5,1.5)
  return genome
end

local function calculateGenomeDistance(firstGenome, secondGenome)
  local secondGenesByInnovation = {}
  for _, gene in ipairs(secondGenome.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  local matchingGeneCount, totalWeightDifference, unmatchedGeneCount = 0, 0, 0
  for _, gene in ipairs(firstGenome.genes) do
    local matchingGene = secondGenesByInnovation[gene.innovation]
    if matchingGene then
      matchingGeneCount = matchingGeneCount+1
      totalWeightDifference = totalWeightDifference+math.abs(gene.weight-matchingGene.weight)
    else
      unmatchedGeneCount = unmatchedGeneCount+1
    end
  end
  unmatchedGeneCount = unmatchedGeneCount
    + math.max(0,#secondGenome.genes-matchingGeneCount)
  local genomeSize = math.max(1,#firstGenome.genes,#secondGenome.genes)
  local averageWeightDifference = matchingGeneCount > 0
    and totalWeightDifference/matchingGeneCount or 0
  return 2*unmatchedGeneCount/genomeSize + 0.4*averageWeightDifference
end

local function assignSpecies(populationState)
  local previousSpecies = populationState.species or {}
  local currentSpecies = {}
  for _, genome in ipairs(populationState.genomes) do
    local matchingSpecies
    for _, speciesGroup in ipairs(currentSpecies) do
      if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
        matchingSpecies=speciesGroup
        break
      end
    end
    if not matchingSpecies then
      local previousMatch
      for _, speciesGroup in ipairs(previousSpecies) do
        if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
          previousMatch=speciesGroup
          break
        end
      end
      matchingSpecies={id=previousMatch and previousMatch.id or (#currentSpecies+1),genomes={},
        topFitness=previousMatch and previousMatch.topFitness or 0,
        staleness=previousMatch and previousMatch.staleness or 0,
        representative=cloneGenome(genome)}
      currentSpecies[#currentSpecies+1]=matchingSpecies
    end
    matchingSpecies.genomes[#matchingSpecies.genomes+1]=genome
    genome.species=matchingSpecies.id
  end
  populationState.species=currentSpecies
end

-- Create one seeded SMB1 controller, then mutate clones until the requested
-- population size is reached. The first genome provides a useful prior; NEAT
-- is free to replace that prior during later generations.
function AI.newPopulation(populationSize)
  local populationState = {generation=1,nextInnovation=ACTION_COUNT,innovations={},genomes={},species={},
    bestFitness=0,population=populationSize or DEFAULT_POPULATION_SIZE,nextGenomeIndex=1,
    nextHiddenNode=NEURAL_INPUT_COUNT,splitHistory={},behaviorArchive={},
    episodeHistory={},topPerformers={},experienceMemory={},legacyLogImported=false}
  for genomeIndex = 1, populationState.population do
    local genome
    if genomeIndex == 1 then
      genome = createSeededGenome(populationState)
    else
      genome = cloneGenome(populationState.genomes[1])
      AI.mutate(genome,populationState)
    end
    populationState.genomes[#populationState.genomes+1]=genome
  end
  assignSpecies(populationState)
  return populationState
end

local function crossover(first,second)
  if second.fitness > first.fitness then first, second = second, first end
  local child = createEmptyGenome()
  local secondGenesByInnovation = {}
  for _, gene in ipairs(second.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  for _, firstGene in ipairs(first.genes) do
    local matchingGene = secondGenesByInnovation[firstGene.innovation]
    local selectedGene = matchingGene and math.random(2) == 1 and matchingGene or firstGene
    local childGene = {sourceNode=selectedGene.sourceNode,targetNode=selectedGene.targetNode,
      weight=selectedGene.weight,enabled=selectedGene.enabled,innovation=selectedGene.innovation}
    if matchingGene and ((not firstGene.enabled) or (not matchingGene.enabled)) and math.random() < 0.75 then
      childGene.enabled = false
    end
    child.genes[#child.genes+1] = childGene
  end
  child.highestHiddenNode = math.max(first.highestHiddenNode,second.highestHiddenNode)
  for mutationName, mutationRate in pairs(first.mutationRates) do
    child.mutationRates[mutationName] = mutationRate
  end
  return child
end

local function rankSpecies(populationState)
  table.sort(populationState.genomes,function(firstGenome,secondGenome)
    return firstGenome.fitness > secondGenome.fitness
  end)
  for rank, genome in ipairs(populationState.genomes) do
    genome.globalRank = #populationState.genomes-rank+1
  end
  local champion = populationState.genomes[1]
  populationState.bestFitness = math.max(populationState.bestFitness or 0,
    champion and champion.fitness or 0)
  for _, speciesGroup in ipairs(populationState.species) do
    table.sort(speciesGroup.genomes,function(firstGenome,secondGenome)
      return firstGenome.fitness > secondGenome.fitness
    end)
    local topFitness = speciesGroup.genomes[1] and speciesGroup.genomes[1].fitness or 0
    if topFitness > speciesGroup.topFitness then
      speciesGroup.topFitness = topFitness
      speciesGroup.staleness = 0
    else
      speciesGroup.staleness = (speciesGroup.staleness or 0)+1
    end
    local adjustedFitnessTotal = 0
    for _, genome in ipairs(speciesGroup.genomes) do
      genome.adjustedFitness = genome.globalRank/math.max(1,#speciesGroup.genomes)
      adjustedFitnessTotal = adjustedFitnessTotal+genome.adjustedFitness
    end
    -- Adjusted fitness already divides each rank by species size. Summing it
    -- gives the species one fair breeding weight; averaging would divide by
    -- species size twice and overproduce one-member species.
    speciesGroup.averageFitness = adjustedFitnessTotal
  end
end

local function chooseSpeciesForBreeding(species)
  local totalFitness = 0
  for _, speciesGroup in ipairs(species) do
    totalFitness = totalFitness + math.max(0,speciesGroup.averageFitness or 0)
  end
  if totalFitness <= 0 then return species[math.random(#species)] end
  local selectionPoint = math.random()*totalFitness
  for _, speciesGroup in ipairs(species) do
    selectionPoint = selectionPoint-math.max(0,speciesGroup.averageFitness or 0)
    if selectionPoint <= 0 then return speciesGroup end
  end
  return species[#species]
end

local function breedChild(group,populationState)
  local speciesMembers = group.genomes
  if #speciesMembers == 1 then
    local child = cloneGenome(speciesMembers[1])
    child.fitness, child.adjustedFitness = 0, 0
    -- A newly formed species must keep exploring rather than make exact copies.
    AI.mutate(child,populationState)
    return child
  end
  local function selectParent()
    local first = speciesMembers[math.random(#speciesMembers)]
    local second = speciesMembers[math.random(#speciesMembers)]
    return first.fitness >= second.fitness and first or second
  end
  local firstParent = selectParent()
  local secondParent = selectParent()
  local child = math.random() < 0.75
    and crossover(firstParent,secondParent) or cloneGenome(firstParent)
  child.fitness, child.adjustedFitness = 0, 0
  AI.mutate(child,populationState)
  return child
end

function AI.nextGeneration(populationState)
  rankSpecies(populationState)
  local champion = populationState.genomes[1]
  local historicChampion=populationState.topPerformers and populationState.topPerformers[1]
  if historicChampion and historicChampion.genome
    and (historicChampion.fitness or 0)>(champion.fitness or 0) then
    champion=cloneGenome(historicChampion.genome)
    champion.fitness=historicChampion.fitness
  end
  local survivingSpecies = {}
  for _, speciesGroup in ipairs(populationState.species) do
    if speciesGroup.staleness < MAX_STALE_GENERATIONS or speciesGroup.genomes[1] == champion then
      local survivorCount = math.max(1,math.ceil(#speciesGroup.genomes/2))
      for memberIndex = #speciesGroup.genomes, survivorCount+1, -1 do
        speciesGroup.genomes[memberIndex] = nil
      end
      survivingSpecies[#survivingSpecies+1] = speciesGroup
    end
  end
  if #survivingSpecies == 0 then
    survivingSpecies = {{id=1,genomes={champion},topFitness=champion.fitness,staleness=0,
      representative=cloneGenome(champion),averageFitness=1}}
  end
  local nextPopulation = {generation=populationState.generation+1,
    nextInnovation=populationState.nextInnovation,innovations=populationState.innovations,
    nextHiddenNode=populationState.nextHiddenNode,splitHistory=populationState.splitHistory,
    species=survivingSpecies,genomes={cloneGenome(champion)},
    bestFitness=populationState.bestFitness,population=populationState.population,
    nextGenomeIndex=1,behaviorArchive=populationState.behaviorArchive or {},
    episodeHistory=populationState.episodeHistory or {},topPerformers=populationState.topPerformers or {},
    experienceMemory=populationState.experienceMemory or {},
    legacyLogImported=populationState.legacyLogImported or false}
  local targetPopulationSize = populationState.population or DEFAULT_POPULATION_SIZE
  local activePolicies={[genomeSignature(champion)]=true}
  while #nextPopulation.genomes < targetPopulationSize do
    local speciesGroup = chooseSpeciesForBreeding(survivingSpecies)
    local child=breedChild(speciesGroup,nextPopulation)
    local signature=genomeSignature(child)
    local retryCount=0
    while activePolicies[signature] and retryCount<32 do
      child=breedChild(speciesGroup,nextPopulation)
      signature=genomeSignature(child)
      retryCount=retryCount+1
    end
    -- A fresh sparse genome is a final escape from a clone-heavy species.
    if activePolicies[signature] then
      retryCount=0
      repeat
        child=AI.newGenome(nextPopulation)
        AI.mutate(child,nextPopulation)
        signature=genomeSignature(child)
        retryCount=retryCount+1
      until not activePolicies[signature] or retryCount>=128
    end
    assert(not activePolicies[signature],"unable to generate a distinct NEAT policy")
    activePolicies[signature]=true
    nextPopulation.genomes[#nextPopulation.genomes+1]=child
  end
  assignSpecies(nextPopulation)
  return nextPopulation
end

local function getDatabasePath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.db"
  end
  return "mario_ai_neat.db"
end

local function getLogPath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.log"
  end
  return "mario_ai_neat.log"
end

function AI.appendLog(message,path)
  path=path or getLogPath()
  if not io or not io.open then return false end
  local file=io.open(path,"a")
  if not file then return false end
  local stamp=os and os.date and os.date("%Y-%m-%d %H:%M:%S") or "time-unknown"
  file:write("[",stamp,"] ",tostring(message),"\n")
  file:flush()
  file:close()
  return true
end

-- object(slot) is the current FCEUX API. create(slot) is retained by FCEUX
-- for older builds and uses an offset slot number. Never call persist().
function AI.createStateAdapter(api,slot)
  if type(api)~="table" or type(api.save)~="function" or type(api.load)~="function" then
    return nil,"savestate API unavailable"
  end
  local stateConstructor, constructorArgument, adapterKind
  if type(api.object)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.object, slot, "object"
  elseif type(api.create)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.create, slot+1, "create"
  else return nil,"savestate constructor unavailable" end
  local succeeded, stateHandle = pcall(stateConstructor,constructorArgument)
  if not succeeded or stateHandle == nil then return nil,"could not create slot handle" end
  local adapter={handle=stateHandle,slot=slot,kind=adapterKind}
  function adapter:save()
    local succeeded, result = pcall(api.save,self.handle)
    return succeeded and result ~= false
  end
  function adapter:load()
    local succeeded, result = pcall(api.load,self.handle)
    return succeeded and result ~= false
  end
  return adapter
end

function AI.setTimerTo999()
  if not SET_TIMER_TO_999_PER_EPISODE or not memory or not memory.writebyte then return false end
  memory.writebyte(RAM.timer_hundreds,0x09)
  memory.writebyte(RAM.timer_tens,0x09)
  memory.writebyte(RAM.timer_ones,0x09)
  return true
end

function AI.keepLivesForTesting()
  if not TESTING_INFINITE_LIVES or not memory or not memory.writebyte then return false end
  -- The inherited LuaRio test section identifies 0x075A as the lives byte.
  -- Refreshing 9 gives the practical effect of infinite lives during tests.
  memory.writebyte(RAM.lives,0x09)
  return true
end

function AI.save(populationState,path)
  -- Write a complete temporary checkpoint first. The rename keeps the last
  -- valid population available if FCEUX stops while the file is being written.
  path = path or getDatabasePath()
  local temporaryPath = path..".tmp"
  local databaseFile = io and io.open and io.open(temporaryPath,"w")
  if not databaseFile then return false end
  safeWriteLine(databaseFile,table.concat({"MARIO_AI_NEAT_V1",populationState.generation,populationState.nextInnovation,
    populationState.bestFitness or 0,populationState.population or #populationState.genomes,#populationState.genomes},","))
  -- Older V1 readers ignore this optional line; new readers resume the next
  -- unevaluated genome after a stopped or crashed FCEUX session.
  safeWriteLine(databaseFile,"P,"..tostring(populationState.nextGenomeIndex or 1))
  safeWriteLine(databaseFile,"H,"..tostring(populationState.nextHiddenNode or NEURAL_INPUT_COUNT))
  safeWriteLine(databaseFile,"L,"..(populationState.legacyLogImported and "1" or "0"))
  if populationState.scoreOnlyHistoricalBest then
    safeWriteLine(databaseFile,"K,"..tostring(populationState.scoreOnlyHistoricalBest))
  end
  for splitInnovation,hiddenNode in pairs(populationState.splitHistory or {}) do
    safeWriteLine(databaseFile,table.concat({"S",splitInnovation,hiddenNode},","))
  end
  -- Novelty-search archive concept: https://arxiv.org/abs/1504.04909
  -- Store compact behavior summaries, not replay states or external source code.
  for _,descriptor in ipairs(populationState.behaviorArchive or {}) do
    safeWriteLine(databaseFile,table.concat({"B",unpack(descriptor)},","))
  end
  -- Keep enough compact episode evidence to audit progress after FCEUX closes.
  for _,episode in ipairs(populationState.episodeHistory or {}) do
    safeWriteLine(databaseFile,table.concat({"E",episode.generation or 0,episode.genomeIndex or 0,
      episode.fitness or 0,episode.startWorldX or 0,episode.maxWorldX or 0,episode.frames or 0,
      episode.reason or "unknown",episode.power or 0,episode.jumps or 0,episode.retreats or 0,
      episode.passedEnemies or 0,episode.landings or 0,episode.powerUps or 0,
      episode.episodeReward or 0,episode.novelty or 0,episode.timestamp or "unknown"},","))
  end
  -- Persist full connections for the distinct highest-scoring policies, not
  -- just a historical score that can no longer be replayed.
  for rank,performer in ipairs(populationState.topPerformers or {}) do
    local genome=performer.genome
    safeWriteLine(databaseFile,table.concat({"T",rank,performer.generation or 0,
      performer.genomeIndex or 0,performer.fitness or 0,performer.maxWorldX or 0,
      performer.progress or 0,performer.reason or "unknown",performer.frames or 0,
      performer.power or 0,performer.jumps or 0,performer.retreats or 0,
      performer.passedEnemies or 0,performer.landings or 0,performer.powerUps or 0,
      performer.episodeReward or 0,performer.novelty or 0,
      genome.highestHiddenNode or NEURAL_INPUT_COUNT,genome.species or 0,#genome.genes,
      performer.timestamp or "unknown"},","))
    for mutationName,mutationRate in pairs(genome.mutationRates or {}) do
      safeWriteLine(databaseFile,table.concat({"TR",rank,mutationName,mutationRate},","))
    end
    for _,gene in ipairs(genome.genes or {}) do
      safeWriteLine(databaseFile,table.concat({"TN",rank,gene.sourceNode,gene.targetNode,
        string.format("%.17g",gene.weight),gene.enabled and 1 or 0,gene.innovation},","))
    end
  end
  local experienceKeys={}
  for contextKey in pairs(populationState.experienceMemory or {}) do
    experienceKeys[#experienceKeys+1]=contextKey
  end
  table.sort(experienceKeys)
  for _,contextKey in ipairs(experienceKeys) do
    local choiceIndices={}
    for choice in pairs(populationState.experienceMemory[contextKey]) do
      choiceIndices[#choiceIndices+1]=choice
    end
    table.sort(choiceIndices)
    for _,choice in ipairs(choiceIndices) do
      local evidence=populationState.experienceMemory[contextKey][choice]
      safeWriteLine(databaseFile,table.concat({"X",contextKey,choice,evidence.attempts,
        evidence.successes,evidence.failures,string.format("%.8f",evidence.rewardMean or 0),
        string.format("%.8f",evidence.qValue or 0),evidence.qVisits or 0},","))
    end
  end
  for genomeIndex, genome in ipairs(populationState.genomes) do
    safeWriteLine(databaseFile,table.concat({"G",genomeIndex,genome.fitness or 0,
      genome.highestHiddenNode or NEURAL_INPUT_COUNT,genome.species or 0},","))
    for mutationName, mutationRate in pairs(genome.mutationRates) do
      safeWriteLine(databaseFile,table.concat({"R",genomeIndex,mutationName,mutationRate},","))
    end
    for _, gene in ipairs(genome.genes) do
      safeWriteLine(databaseFile,table.concat({"N",genomeIndex,gene.sourceNode,gene.targetNode,string.format("%.17g",gene.weight),
        gene.enabled and 1 or 0,gene.innovation},","))
    end
  end
  databaseFile:close()
  if not backupFile(path) then
    if os and os.remove then os.remove(temporaryPath) end
    return false
  end
  local renamed = os and os.rename and os.rename(temporaryPath,path)
  if not renamed then
    local temporaryFile = io.open(temporaryPath,"r")
    local destinationFile = io.open(path,"w")
    if not temporaryFile or not destinationFile then
      if temporaryFile then temporaryFile:close() end
      if destinationFile then destinationFile:close() end
      return false
    end
    destinationFile:write(temporaryFile:read("*a"))
    temporaryFile:close()
    destinationFile:close()
    os.remove(temporaryPath)
  end
  return true
end

function AI.load(path)
  -- Loading is deliberately tolerant of missing or malformed checkpoints: a
  -- failed load returns nil and AI.run creates a fresh population instead.
  path = path or getDatabasePath()
  if not io or not io.open then return nil end
  local databaseFile = io.open(path,"r")
  if not databaseFile then return nil end
  local header = databaseFile:read("*l") or ""
  local generation, innovationId, bestFitness, populationSize, genomeCount = header:match(
    "^MARIO_AI_NEAT_V1,(%d+),([%d%.]+),([%d%.%-]+),(%d+),(%d+)$")
  if not generation then databaseFile:close();return nil end
  local populationState={generation=tonumber(generation),nextInnovation=tonumber(innovationId),
    bestFitness=tonumber(bestFitness),population=tonumber(populationSize),genomes={},species={},innovations={},
    nextHiddenNode=NEURAL_INPUT_COUNT,splitHistory={},behaviorArchive={},episodeHistory={},
    topPerformers={},experienceMemory={},legacyLogImported=false}
  for genomeIndex = 1, tonumber(genomeCount) do
    populationState.genomes[genomeIndex] = createEmptyGenome()
    populationState.genomes[genomeIndex].fitness = 0
  end
  for line in databaseFile:lines() do
    local fields={}
    for field in (line..","):gmatch("(.-),") do fields[#fields+1]=field end
    if fields[1]=="P" then
      local savedIndex=tonumber(fields[2])
      if savedIndex and savedIndex>=1 and savedIndex<=#populationState.genomes then
        populationState.nextGenomeIndex=math.floor(savedIndex)
      end
    elseif fields[1]=="H" then
      local savedNode=tonumber(fields[2])
      if savedNode and isHiddenNode(savedNode) then
        populationState.nextHiddenNode=math.floor(savedNode)
      end
    elseif fields[1]=="L" then
      populationState.legacyLogImported=tonumber(fields[2])==1
    elseif fields[1]=="K" then
      populationState.scoreOnlyHistoricalBest=tonumber(fields[2])
    elseif fields[1]=="B" then
      local descriptor={}
      for fieldIndex=2,#fields do
        local value=tonumber(fields[fieldIndex])
        if value then descriptor[#descriptor+1]=value end
      end
      if #descriptor==6 and #populationState.behaviorArchive<NOVELTY_ARCHIVE_LIMIT then
        populationState.behaviorArchive[#populationState.behaviorArchive+1]=descriptor
      end
    elseif fields[1]=="E" then
      if #populationState.episodeHistory<EPISODE_HISTORY_LIMIT then
        local episode={generation=tonumber(fields[2]),genomeIndex=tonumber(fields[3]),
          fitness=tonumber(fields[4]),startWorldX=tonumber(fields[5]),maxWorldX=tonumber(fields[6]),
          frames=tonumber(fields[7]),reason=fields[8],power=tonumber(fields[9]),
          jumps=tonumber(fields[10]),retreats=tonumber(fields[11]),passedEnemies=tonumber(fields[12]),
          landings=tonumber(fields[13]),powerUps=tonumber(fields[14]),
          episodeReward=tonumber(fields[15]),novelty=tonumber(fields[16]),timestamp=fields[17]}
        if episode.generation and episode.fitness and episode.maxWorldX then
          populationState.episodeHistory[#populationState.episodeHistory+1]=episode
        end
      end
    elseif fields[1]=="T" then
      local rank=tonumber(fields[2])
      if rank and rank>=1 and rank<=TOP_PERFORMER_LIMIT then
        local genome=createEmptyGenome()
        genome.fitness=tonumber(fields[5]) or 0
        genome.highestHiddenNode=tonumber(fields[18]) or NEURAL_INPUT_COUNT
        genome.species=tonumber(fields[19]) or 0
        populationState.topPerformers[rank]={generation=tonumber(fields[3]),
          genomeIndex=tonumber(fields[4]),fitness=tonumber(fields[5]),maxWorldX=tonumber(fields[6]),
          progress=tonumber(fields[7]),reason=fields[8],frames=tonumber(fields[9]),
          power=tonumber(fields[10]),jumps=tonumber(fields[11]),retreats=tonumber(fields[12]),
          passedEnemies=tonumber(fields[13]),landings=tonumber(fields[14]),
          powerUps=tonumber(fields[15]),episodeReward=tonumber(fields[16]),
          novelty=tonumber(fields[17]),timestamp=fields[21],genome=genome}
      end
    elseif fields[1]=="TR" then
      local rank=tonumber(fields[2])
      local performer=rank and populationState.topPerformers[rank]
      if performer and fields[3] then performer.genome.mutationRates[fields[3]]=tonumber(fields[4]) or 0 end
    elseif fields[1]=="TN" then
      local rank=tonumber(fields[2])
      local performer=rank and populationState.topPerformers[rank]
      if performer then
        performer.genome.genes[#performer.genome.genes+1]={sourceNode=tonumber(fields[3]),
          targetNode=tonumber(fields[4]),weight=tonumber(fields[5]),enabled=tonumber(fields[6])==1,
          innovation=tonumber(fields[7])}
      end
    elseif fields[1]=="X" then
      local contextKey=fields[2]
      local choice,attempts=tonumber(fields[3]),tonumber(fields[4])
      local successes,failures=tonumber(fields[5]),tonumber(fields[6])
      local rewardMean=tonumber(fields[7])
      local qValue,qVisits=tonumber(fields[8]) or 0,tonumber(fields[9]) or 0
      if type(contextKey)=="string" and #contextKey>0 and #contextKey<=96
        and choice and choice>=1 and choice<=ACTION_COUNT*DURATION_OUTPUT_COUNT
        and attempts and attempts>=0 and attempts<=100000000
        and successes and failures and successes>=0 and failures>=0
        and successes+failures<=attempts and rewardMean and (attempts>0 or qVisits>0) then
        local choices=populationState.experienceMemory[contextKey]
        if not choices and AI.experienceMemorySize(populationState)<EXPERIENCE_CONTEXT_LIMIT then
          choices={};populationState.experienceMemory[contextKey]=choices
        end
        if choices then
          choices[math.floor(choice)]={attempts=math.floor(attempts),
            successes=math.floor(successes),failures=math.floor(failures),rewardMean=rewardMean,
            qValue=clamp(qValue,-1,1),qVisits=math.max(0,math.floor(qVisits))}
        end
      end
    elseif fields[1]=="S" then
      local splitInnovation,hiddenNode=tonumber(fields[2]),tonumber(fields[3])
      if splitInnovation and hiddenNode and hiddenNode>NEURAL_INPUT_COUNT
        and hiddenNode<OUTPUT_NODE_OFFSET then
        populationState.splitHistory[splitInnovation]=math.floor(hiddenNode)
      end
    elseif fields[1]=="G" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        populationState.genomes[genomeIndex].fitness=tonumber(fields[3]) or 0
        populationState.genomes[genomeIndex].highestHiddenNode=tonumber(fields[4]) or NEURAL_INPUT_COUNT
        populationState.genomes[genomeIndex].species=tonumber(fields[5]) or 0
      end
    elseif fields[1]=="R" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        populationState.genomes[genomeIndex].mutationRates[fields[3]]=tonumber(fields[4]) or 0
      end
    elseif fields[1]=="N" then
      local genomeIndex = tonumber(fields[2])
      if populationState.genomes[genomeIndex] then
        local gene={sourceNode=tonumber(fields[3]),targetNode=tonumber(fields[4]),weight=tonumber(fields[5]),
          enabled=tonumber(fields[6])==1,innovation=tonumber(fields[7])}
        populationState.genomes[genomeIndex].genes[#populationState.genomes[genomeIndex].genes+1]=gene
        populationState.innovations[connectionKey(gene.sourceNode,gene.targetNode)]=gene.innovation
        if isHiddenNode(gene.sourceNode) then
          populationState.nextHiddenNode=math.max(populationState.nextHiddenNode,gene.sourceNode)
        end
        if isHiddenNode(gene.targetNode) then
          populationState.nextHiddenNode=math.max(populationState.nextHiddenNode,gene.targetNode)
        end
      end
    end
  end
  databaseFile:close()
  assignSpecies(populationState)
  return populationState
end

local function findClosestThreat(state)
  local nearestEnemy, nearestDistance
  for _, enemyCandidate in ipairs(state.enemies) do
    local horizontalOffset = enemyCandidate.worldX-state.worldX
    local verticalOffset = math.abs(enemyCandidate.worldY-state.worldY)
    if not INACTIVE_ENEMY_STATES[enemyCandidate.status] and verticalOffset<72
      and horizontalOffset>-32 and horizontalOffset<144 then
      local weightedDistance = math.abs(horizontalOffset)+verticalOffset
      if not nearestDistance or weightedDistance<nearestDistance then
        nearestEnemy, nearestDistance = enemyCandidate, weightedDistance
      end
    end
  end
  return nearestEnemy
end

local function distanceBand(distance)
  if distance <= 24 then return 0 end
  if distance <= 40 then return 1 end
  if distance <= 64 then return 2 end
  if distance <= 96 then return 3 end
  return 4
end

-- Context keys omit absolute world position so similar situations can share
-- experience across different level locations and training sessions.
local function describeExperienceContext(state)
  local grounded=state.grounded and "g" or "a"
  local speedBand=math.floor((clamp(state.horizontalVelocity or 0,-4,4)+4)/2)
  local powerBand=state.power==2 and 2 or (state.power==1 and 1 or 0)
  local enemy=findClosestThreat(state)
  if enemy and enemy.worldX>state.worldX then
    local enemyDistance=enemy.worldX-state.worldX
    local enemyClass=NON_STOMPABLE_ENEMIES[enemy.id] and "air" or "ground"
    local verticalBand=math.abs(enemy.worldY-state.worldY)>24 and "high" or "level"
    local key=string.format("enemy:%s:%s:%d:%s:v%d:p%d",enemyClass,verticalBand,
      distanceBand(enemyDistance),grounded,speedBand,powerBand)
    return key,"enemy",enemy.worldX
  end

  local firstGapDistance
  local gapWidth=0
  for horizontalOffset=16,96,16 do
    if not isSolidTileAtOffset(state,horizontalOffset,16) then
      if not firstGapDistance then firstGapDistance=horizontalOffset end
      gapWidth=gapWidth+1
    elseif firstGapDistance then
      break
    end
  end
  if firstGapDistance then
    return string.format("gap:w%d:d%d:%s:v%d:p%d",math.min(gapWidth,5),
      distanceBand(firstGapDistance),grounded,speedBand,powerBand),"gap",
      state.worldX+firstGapDistance+gapWidth*16
  end

  -- Sample above the floor to separate a pipe/block face from ordinary ground.
  for horizontalOffset=16,96,16 do
    local obstacleHeight=0
    for _,verticalOffset in ipairs({-48,-32,-16}) do
      if isSolidTileAtOffset(state,horizontalOffset,verticalOffset) then
        obstacleHeight=obstacleHeight+1
      end
    end
    if obstacleHeight>0 then
      local obstacleWidth=1
      for nextOffset=horizontalOffset+16,96,16 do
        local nextColumnHasObstacle=false
        for _,verticalOffset in ipairs({-48,-32,-16}) do
          if isSolidTileAtOffset(state,nextOffset,verticalOffset) then
            nextColumnHasObstacle=true
            break
          end
        end
        if not nextColumnHasObstacle then break end
        obstacleWidth=obstacleWidth+1
      end
      return string.format("obstacle:h%d:d%d:%s:v%d:p%d",obstacleHeight,
        distanceBand(horizontalOffset),grounded,speedBand,powerBand),"obstacle",
        state.worldX+horizontalOffset+obstacleWidth*16
    end
  end

  local key=string.format("clear:%s:v%d:p%d",grounded,speedBand,powerBand)
  return key,nil,nil
end

function AI.experienceContextKey(state)
  return describeExperienceContext(state)
end

function AI.experienceMemorySize(populationState)
  local count=0
  for _ in pairs(populationState.experienceMemory or {}) do count=count+1 end
  return count
end

local function experienceChoice(actionIndex,durationIndex)
  return (actionIndex-1)*DURATION_OUTPUT_COUNT+durationIndex
end

-- Parse the compact sensor context into comparable features. Exact string
-- matching wasted useful experience whenever a Goomba was one distance band
-- closer on a later attempt, so memory now uses a small, class-aware kernel.
local experienceFeatureCache={}
local function experienceContextFeatures(contextKey)
  local cached=experienceFeatureCache[contextKey]
  if cached~=nil then return cached or nil end
  local features
  local enemyClass,vertical,distance,grounded,speed,power=contextKey:match(
    "^enemy:([^:]+):([^:]+):(%d+):([^:]+):v(%d+):p(%d+)$")
  if enemyClass then
    features={kind="enemy",class=enemyClass,vertical=vertical,distance=tonumber(distance),
      grounded=grounded,speed=tonumber(speed),power=tonumber(power)}
  else
  local width,distance,grounded,speed,power=contextKey:match(
    "^gap:w(%d+):d(%d+):([^:]+):v(%d+):p(%d+)$")
  if width then
    features={kind="gap",width=tonumber(width),distance=tonumber(distance),grounded=grounded,
      speed=tonumber(speed),power=tonumber(power)}
  else
  local height,distance,grounded,speed,power=contextKey:match(
    "^obstacle:h(%d+):d(%d+):([^:]+):v(%d+):p(%d+)$")
  if height then
    features={kind="obstacle",height=tonumber(height),distance=tonumber(distance),grounded=grounded,
      speed=tonumber(speed),power=tonumber(power)}
  else
  grounded,speed,power=contextKey:match("^clear:([^:]+):v(%d+):p(%d+)$")
  if grounded then
    features={kind="clear",grounded=grounded,speed=tonumber(speed),power=tonumber(power)}
  end
  end
  end
  end
  experienceFeatureCache[contextKey]=features or false
  return features
end

local function experienceContextSimilarity(firstKey,secondKey)
  if firstKey==secondKey then return 1 end
  local first=experienceContextFeatures(firstKey)
  local second=experienceContextFeatures(secondKey)
  if not first or not second or first.kind~=second.kind
    or first.grounded~=second.grounded or first.power~=second.power then return 0 end
  local speedDifference=math.abs(first.speed-second.speed)
  if speedDifference>2 then return 0 end
  local similarity=0.9^speedDifference
  if first.kind=="enemy" then
    if first.class~=second.class or first.vertical~=second.vertical then return 0 end
    local distanceDifference=math.abs(first.distance-second.distance)
    if distanceDifference>2 then return 0 end
    return similarity*0.82^distanceDifference
  elseif first.kind=="gap" then
    local widthDifference=math.abs(first.width-second.width)
    local distanceDifference=math.abs(first.distance-second.distance)
    if widthDifference>1 or distanceDifference>2 then return 0 end
    return similarity*0.78^widthDifference*0.84^distanceDifference
  elseif first.kind=="obstacle" then
    local heightDifference=math.abs(first.height-second.height)
    local distanceDifference=math.abs(first.distance-second.distance)
    if heightDifference>1 or distanceDifference>2 then return 0 end
    return similarity*0.78^heightDifference*0.84^distanceDifference
  end
  return similarity
end

local function trimExperienceMemory(memory)
  local contextCount=0
  for _ in pairs(memory) do contextCount=contextCount+1 end
  while contextCount>EXPERIENCE_CONTEXT_LIMIT do
    local weakestKey,weakestEvidence
    for contextKey,choices in pairs(memory) do
      local evidence=0
      for _,record in pairs(choices) do
        evidence=evidence+(record.attempts or 0)+(record.qVisits or 0)
      end
      if weakestEvidence==nil or evidence<weakestEvidence then
        weakestKey,weakestEvidence=contextKey,evidence
      end
    end
    if not weakestKey then break end
    memory[weakestKey]=nil
    contextCount=contextCount-1
  end
end

function AI.updateExperienceMemory(populationState,contextKey,choice,succeeded)
  if type(contextKey)~="string" or #contextKey==0 or #contextKey>96
    or not choice or choice<1 or choice>ACTION_COUNT*DURATION_OUTPUT_COUNT then return false end
  local memory=populationState.experienceMemory or {}
  populationState.experienceMemory=memory
  local choices=memory[contextKey]
  if not choices then choices={};memory[contextKey]=choices end
  local record=choices[choice] or {attempts=0,successes=0,failures=0,rewardMean=0}
  record.attempts=record.attempts+1
  if succeeded then
    record.successes=record.successes+1
  else
    record.failures=record.failures+1
  end
  local reward=succeeded and 1 or -1
  record.rewardMean=record.rewardMean+(reward-record.rewardMean)/record.attempts
  choices[choice]=record
  trimExperienceMemory(memory)
  return true
end

-- A decision evaluates 24 action and duration pairs against the same context.
-- Calculate the matching contexts once, then reuse their weights for every pair.
local function similarExperienceContexts(memory,contextKey)
  local similar={}
  for rememberedContext,choices in pairs(memory or {}) do
    local similarity=experienceContextSimilarity(contextKey,rememberedContext)
    if similarity>0 then
      similar[#similar+1]={choices=choices,weight=similarity*similarity}
    end
  end
  return similar
end

local function experienceBiasFromMemory(memory,contextKey,actionIndex,durationIndex,similarContexts)
  if not memory then return 0 end
  local choice=experienceChoice(actionIndex,durationIndex)
  local weightedSuccesses,weightedFailures,totalWeight=0,0,0
  local weightedQ,totalQWeight=0,0
  for _,similarContext in ipairs(similarContexts or similarExperienceContexts(memory,contextKey)) do
    local record=similarContext.choices[choice]
    if record then
      local evidenceWeight=similarContext.weight
      if (record.attempts or 0)>0 then
        weightedSuccesses=weightedSuccesses+(record.successes or 0)*evidenceWeight
        weightedFailures=weightedFailures+(record.failures or 0)*evidenceWeight
        totalWeight=totalWeight+(record.attempts or 0)*evidenceWeight
      end
      if (record.qVisits or 0)>0 then
        local qWeight=evidenceWeight*math.min(record.qVisits,12)
        weightedQ=weightedQ+(record.qValue or 0)*qWeight
        totalQWeight=totalQWeight+qWeight
      end
    end
  end
  local outcomeBias=0
  if totalWeight>0 then
    -- A uniform Beta(1,1) prior keeps one lucky outcome from dominating.
    local successRate=(weightedSuccesses+1)/(weightedSuccesses+weightedFailures+2)
    local confidence=math.min(1,totalWeight/6)
    outcomeBias=(successRate-0.5)*2*confidence*0.8
  end
  local qBias=0
  if totalQWeight>0 then
    local confidence=totalQWeight/(totalQWeight+4)
    qBias=(weightedQ/totalQWeight)*confidence*0.8
  end
  return clamp(outcomeBias*0.5+qBias*0.5,-0.8,0.8)
end

local function bestExperienceQValue(memory,contextKey)
  local weightedValues,totalWeights={},{}
  for _,similarContext in ipairs(similarExperienceContexts(memory,contextKey)) do
    for choice,record in pairs(similarContext.choices) do
      if (record.qVisits or 0)>0 then
        local weight=similarContext.weight*math.min(record.qVisits,12)
        weightedValues[choice]=(weightedValues[choice] or 0)+(record.qValue or 0)*weight
        totalWeights[choice]=(totalWeights[choice] or 0)+weight
      end
    end
  end
  local best=0
  for choice=1,ACTION_COUNT*DURATION_OUTPUT_COUNT do
    if (totalWeights[choice] or 0)>0 then
      best=math.max(best,weightedValues[choice]/totalWeights[choice])
    end
  end
  return best
end

-- Online contextual Q-learning lets an attempt improve the shared memory
-- immediately; NEAT still evolves the broader neural policy across episodes.
function AI.updateExperienceQ(populationState,contextKey,choice,reward,nextContextKey,terminal,discount)
  if type(contextKey)~="string" or #contextKey==0 or #contextKey>96
    or not choice or choice<1 or choice>ACTION_COUNT*DURATION_OUTPUT_COUNT then return false end
  local memory=populationState.experienceMemory or {}
  populationState.experienceMemory=memory
  local choices=memory[contextKey]
  if not choices then
    if AI.experienceMemorySize(populationState)>=EXPERIENCE_CONTEXT_LIMIT then trimExperienceMemory(memory) end
    choices={};memory[contextKey]=choices
  end
  local record=choices[choice] or {attempts=0,successes=0,failures=0,rewardMean=0,qValue=0,qVisits=0}
  local bootstrap=not terminal and nextContextKey
    and bestExperienceQValue(memory,nextContextKey) or 0
  local target=clamp((reward or 0)+(discount or 0.9)*bootstrap,-1,1)
  local learningRate=0.25
  record.qValue=clamp((record.qValue or 0)+learningRate*(target-(record.qValue or 0)),-1,1)
  record.qVisits=(record.qVisits or 0)+1
  choices[choice]=record
  trimExperienceMemory(memory)
  return true,record.qValue
end

function AI.experienceBias(populationState,contextKey,actionIndex,durationIndex)
  return experienceBiasFromMemory(populationState.experienceMemory,contextKey,actionIndex,durationIndex)
end

local function labelChallenge(aiState,succeeded)
  local challenge=aiState.activeChallenge
  if not challenge then return end
  for _,choiceRecord in pairs(challenge.choices) do
    AI.updateExperienceMemory(aiState.populationState,choiceRecord.context,
      choiceRecord.choice,succeeded)
  end
  aiState.activeChallenge=nil
end

local function trackChallenge(aiState,state,currentKind,targetWorldX,passedEnemy)
  local challenge=aiState.activeChallenge
  if challenge then
    local passedTarget=state.worldX>=challenge.targetWorldX+8
    local successfulEnemy=challenge.kind=="enemy" and passedEnemy
    local clearLanding=state.grounded and currentKind~=challenge.kind
      and state.worldX>=challenge.targetWorldX+8
    if successfulEnemy or (challenge.kind=="enemy" and passedTarget) or clearLanding then
      labelChallenge(aiState,true)
      challenge=nil
    end
  end
  if not aiState.activeChallenge and currentKind and state.grounded then
    aiState.activeChallenge={kind=currentKind,targetWorldX=targetWorldX or state.worldX,
      choices={},choiceOrder={},startedFrame=aiState.episodeFrames}
  end
end

local function rememberChallengeChoice(aiState,contextKey,actionIndex,durationIndex)
  local challenge=aiState.activeChallenge
  if not challenge then return end
  local choice=experienceChoice(actionIndex,durationIndex)
  local pairKey=contextKey..":"..choice
  if not challenge.choices[pairKey] and #challenge.choiceOrder<EXPERIENCE_TRACE_LIMIT then
    challenge.choices[pairKey]={context=contextKey,choice=choice}
    challenge.choiceOrder[#challenge.choiceOrder+1]=pairKey
  end
end

local function finishPendingExperience(aiState,state,terminalReason)
  local pending=aiState.pendingExperience
  if not pending then return end
  aiState.pendingExperience=nil
  if pending.kind then return end -- Hazard choices get outcome-level credit below.
  local moved=(state.worldX or pending.worldX)-pending.worldX
  local eventReward=(aiState.episodeReward or 0)-pending.episodeReward
  if terminalReason then
    if terminalReason=="death" or terminalReason=="stuck" or terminalReason=="timeout" then
      AI.updateExperienceMemory(aiState.populationState,pending.context,pending.choice,false)
    elseif terminalReason=="victory" then
      AI.updateExperienceMemory(aiState.populationState,pending.context,pending.choice,true)
    end
  elseif moved>0 or eventReward>0 then
    AI.updateExperienceMemory(aiState.populationState,pending.context,pending.choice,true)
  end
end

local function finishPendingQTransition(aiState,state,nextContextKey,terminalReason)
  local pending=aiState.pendingTransition
  if not pending then return end
  aiState.pendingTransition=nil
  local terminal=terminalReason~=nil
  local progressReward=clamp(((state.worldX or pending.worldX)-pending.worldX)/24,-0.5,0.5)
  local eventReward=clamp(((aiState.episodeReward or 0)-pending.episodeReward)/24,-0.25,0.5)
  local reward=progressReward+eventReward
  if terminalReason=="death" then reward=-0.9
  elseif terminalReason=="stuck" or terminalReason=="timeout" then reward=-0.55
  elseif terminalReason=="victory" then reward=1 end
  local elapsedFrames=math.max(1,(aiState.episodeFrames or pending.frame)-pending.frame)
  local discount=0.9^(clamp(elapsedFrames/4,1,16))
  AI.updateExperienceQ(aiState.populationState,pending.context,pending.choice,reward,
    nextContextKey,terminal,discount)
end

local function calculateAllowedActions(state,enemy)
  -- This is a narrow safety shield, not a second controller. It removes only
  -- actions that would make an observed close threat immediately unavoidable.
  local allowed={}
  if enemy then
    local horizontalOffset=enemy.worldX-state.worldX
    if horizontalOffset>0 and horizontalOffset<112 then
      -- Preserve the original AI's stomp response for ground enemies. Close
      -- contact and non-stompable enemies remove forward motion from the policy.
      allowed[3],allowed[4]=true,true
    if not NON_STOMPABLE_ENEMIES[enemy.id] and horizontalOffset>=28 then
        allowed[2]=true
        if state.grounded then allowed[5]=true end
      elseif state.grounded then
        allowed[5]=true
      end
      if state.power==2 and horizontalOffset>36 and horizontalOffset>=28 then allowed[1]=true end
      return allowed
    elseif horizontalOffset<=0 and horizontalOffset>-32 then
      allowed[1],allowed[2],allowed[3],allowed[4],allowed[5]=true,true,true,true,true
      return allowed
    end
  end
  for actionIndex = 1, ACTION_COUNT do allowed[actionIndex] = true end
  return allowed
end

local function chooseAction(genome,state,observationInputs,memoryInputs,experienceMemory)
  local enemy=findClosestThreat(state)
  local allowed=calculateAllowedActions(state,enemy)
  local actionScores,nodeValues,durationScores=AI.evaluateGenome(genome,observationInputs,memoryInputs)
  if state.power==2 and enemy and enemy.worldX-state.worldX>36 then actionScores[1]=actionScores[1]+0.3 end
  local contextKey=describeExperienceContext(state)
  local similarContexts=experienceMemory and similarExperienceContexts(experienceMemory,contextKey)
  local selectedActionIndex,selectedDurationIndex,highestPairScore
  for actionIndex=1,ACTION_COUNT do
    if allowed[actionIndex] then
      for durationIndex=1,DURATION_OUTPUT_COUNT do
        local memoryBias=experienceBiasFromMemory(experienceMemory,contextKey,actionIndex,durationIndex,similarContexts)
        local durationPrior=durationScores[durationIndex]
        if durationScores[1]==0 and durationScores[2]==0
          and durationScores[3]==0 and durationScores[4]==0 and durationIndex==2 then
          durationPrior=0.001
        end
        local pairScore=actionScores[actionIndex]+durationPrior+memoryBias
        if not highestPairScore or pairScore>highestPairScore then
          selectedActionIndex,selectedDurationIndex,highestPairScore=actionIndex,durationIndex,pairScore
        end
      end
    end
  end
  if not selectedActionIndex then selectedActionIndex,selectedDurationIndex=4,2 end
  return ACTION_OPTIONS[selectedActionIndex],selectedActionIndex,enemy,
    actionScores,nodeValues,observationInputs,ACTION_HOLD_FRAMES[selectedDurationIndex],selectedDurationIndex,contextKey
end

local ACTION_LABEL = {
  run="run with speed", jump_run="running jump", retreat="retreat and reassess",
  brake="brake for control", jump_place="vertical jump", walk="controlled walk",
}

function AI.learningStatus(aiState,state,action)
  local enemy=findClosestThreat(state)
  local gapAhead=hasGapAhead(state)>0
  local visibleItem=state.items and state.items[1]
  local lesson="forward movement and momentum control"
  local sensing="clear ground ahead"
  if enemy then
    sensing=string.format("%s %d px ahead",enemy.name,math.max(0,enemy.worldX-state.worldX))
    if action.name=="jump_run" or action.name=="jump_place" then
      lesson="jump timing to clear an enemy"
    elseif action.name=="retreat" or action.name=="brake" then
      lesson="safe spacing and enemy avoidance"
    else
      lesson="choosing a safe response to an enemy"
    end
  elseif gapAhead then
    sensing="gap or missing floor ahead"
    lesson=action.name=="jump_run" and "jump timing and landing distance" or "safe gap approach"
  elseif visibleItem then
    sensing="power-up visible"
    lesson=action.name=="retreat" and "positioning for a power-up" or "power-up value versus safe progress"
  elseif not state.grounded then
    sensing="Mario is airborne"
    lesson="air control and landing alignment"
  end
  return {lesson=lesson,sensing=sensing,action=ACTION_LABEL[action.name] or action.name,
    progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))}
end

function AI.new(populationState)
  populationState=populationState or AI.newPopulation()
  populationState.experienceMemory=populationState.experienceMemory or {}
  return {populationState=populationState,
    genomeIndex=populationState.nextGenomeIndex or 1,episodeFrames=0,episodeDecisions=0,episodeStartTime=0,
    startWorldX=nil,furthestWorldX=nil,lastProgressFrame=0,episodeReward=0,totalEpisodes=0,
    lastAction=nil,frames=0,finished=false,episodeActive=false,championMode=false,
    recentGlobalInputs={},previousEpisodeState=nil,activeChallenge=nil,pendingExperience=nil,
    pendingTransition=nil,
    behavior={jumps=0,retreats=0,
      passedEnemies=0,landings=0,powerUps=0},behaviorArchive=populationState.behaviorArchive or {}}
end

function AI.bestGenomeIndex(populationState)
  local championIndex,bestFitness=1,nil
  for genomeIndex,genome in ipairs(populationState.genomes) do
    if bestFitness==nil or (genome.fitness or 0)>bestFitness then
      championIndex,bestFitness=genomeIndex,genome.fitness or 0
    end
  end
  return championIndex,bestFitness or 0
end

function AI.bestPerformer(populationState)
  local genomeIndex,currentFitness=AI.bestGenomeIndex(populationState)
  local best={genome=populationState.genomes[genomeIndex],fitness=currentFitness,
    genomeIndex=genomeIndex,source="population"}
  for _,performer in ipairs(populationState.topPerformers or {}) do
    if performer.genome and (performer.fitness or 0)>best.fitness then
      best={genome=performer.genome,fitness=performer.fitness,genomeIndex=performer.genomeIndex,
        generation=performer.generation,source="archive"}
    end
  end
  return best
end

local function updateEpisodeHistory(populationState,episode)
  populationState.episodeHistory=populationState.episodeHistory or {}
  populationState.episodeHistory[#populationState.episodeHistory+1]=episode
  while #populationState.episodeHistory>EPISODE_HISTORY_LIMIT do
    table.remove(populationState.episodeHistory,1)
  end
end

local function updateTopPerformers(populationState,episode,genome)
  populationState.topPerformers=populationState.topPerformers or {}
  populationState.bestFitness=math.max(populationState.bestFitness or 0,episode.fitness or 0)
  local signature=genomeSignature(genome)
  for index,performer in ipairs(populationState.topPerformers) do
    if genomeSignature(performer.genome)==signature then
      if episode.fitness>(performer.fitness or 0) then
        episode.genome=cloneGenome(genome)
        populationState.topPerformers[index]=episode
      end
      table.sort(populationState.topPerformers,function(first,second)
        return first.fitness>second.fitness
      end)
      return
    end
  end
  episode.genome=cloneGenome(genome)
  populationState.topPerformers[#populationState.topPerformers+1]=episode
  table.sort(populationState.topPerformers,function(first,second)
    return first.fitness>second.fitness
  end)
  while #populationState.topPerformers>TOP_PERFORMER_LIMIT do
    table.remove(populationState.topPerformers)
  end
end

local function activeGenome(aiState)
  if aiState.championMode and aiState.championGenome then return aiState.championGenome end
  return aiState.populationState.genomes[aiState.genomeIndex]
end

function AI.importLegacyLogHistory(populationState,path)
  if populationState.legacyLogImported then return 0,0 end
  local logFile=io and io.open and io.open(path,"r")
  local importedEpisodes=0
  local importedCurrentGenomes=0
  local bestForCurrentGenome={}
  populationState.episodeHistory=populationState.episodeHistory or {}
  if logFile then
    local activeStart
    for line in logFile:lines() do
      local timestamp,generation,genomeIndex,_,startWorldX,power=line:match(
        "^%[([^%]]+)%] episode start | generation=(%d+) | genome=(%d+)/(%d+) | x=(%d+) | power=(%d+)")
      if timestamp then
        activeStart={timestamp=timestamp,generation=tonumber(generation),
          genomeIndex=tonumber(genomeIndex),startWorldX=tonumber(startWorldX),power=tonumber(power)}
      else
        local endTimestamp,reason,fitness,maxWorldX,frames=line:match(
          "^%[([^%]]+)%] episode end | reason=([^|]+) | fitness=([%-%d%.]+) | max_x=(%d+) | frames=(%d+)")
        if endTimestamp and activeStart then
          reason=reason:match("^%s*(.-)%s*$")
          fitness,maxWorldX,frames=tonumber(fitness),tonumber(maxWorldX),tonumber(frames)
          local episode={generation=activeStart.generation,genomeIndex=activeStart.genomeIndex,
            fitness=fitness,startWorldX=activeStart.startWorldX,maxWorldX=maxWorldX,
            progress=math.max(0,maxWorldX-activeStart.startWorldX),frames=frames,reason=reason,
            power=activeStart.power,jumps=-1,retreats=-1,passedEnemies=-1,landings=-1,
            powerUps=-1,episodeReward=-1,novelty=-1,timestamp=endTimestamp}
          updateEpisodeHistory(populationState,episode)
          importedEpisodes=importedEpisodes+1
          if episode.generation==populationState.generation
            and episode.genomeIndex<(populationState.nextGenomeIndex or 1) then
            local old=bestForCurrentGenome[episode.genomeIndex]
            if not old or episode.fitness>old.fitness then
              bestForCurrentGenome[episode.genomeIndex]=episode
            end
          end
          activeStart=nil
        end
      end
    end
    logFile:close()
  end
  for genomeIndex,episode in pairs(bestForCurrentGenome) do
    local genome=populationState.genomes[genomeIndex]
    if genome then
      genome.fitness=episode.fitness
      updateTopPerformers(populationState,episode,genome)
      importedCurrentGenomes=importedCurrentGenomes+1
    end
  end
  local recoverableBest=0
  for _,genome in ipairs(populationState.genomes) do
    recoverableBest=math.max(recoverableBest,genome.fitness or 0)
  end
  for _,performer in ipairs(populationState.topPerformers) do
    recoverableBest=math.max(recoverableBest,performer.fitness or 0)
  end
  if (populationState.bestFitness or 0)>recoverableBest then
    -- Older files kept the score but not necessarily the corresponding genome.
    populationState.scoreOnlyHistoricalBest=math.max(
      populationState.scoreOnlyHistoricalBest or 0,populationState.bestFitness or 0)
    populationState.bestFitness=recoverableBest
  end
  populationState.legacyLogImported=true
  return importedEpisodes,importedCurrentGenomes
end

function AI.beginEpisode(aiState,state)
  aiState.episodeFrames=0;aiState.episodeReward=0;aiState.startWorldX=state.worldX;aiState.furthestWorldX=state.worldX
  aiState.episodeDecisions=0
  aiState.episodeStartTime=os and os.time and os.time() or 0
  aiState.lastProgressFrame=0;aiState.finished=false
  aiState.episodeActive=true
  aiState.bestForm=state.power==2 and 2 or (state.size==1 and 0 or 1)
  aiState.previousEpisodeState=nil
  aiState.recentGlobalInputs={}
  aiState.cachedAction=nil
  aiState.actionFramesRemaining=0
  aiState.behavior={jumps=0,retreats=0,passedEnemies=0,landings=0,powerUps=0}
  aiState.activeChallenge=nil
  aiState.pendingExperience=nil
  aiState.pendingTransition=nil
  aiState.nextLandmark=math.floor(state.worldX/128)+1
end

function AI.isValidTrainingStart(state)
  return state~=nil and state.phase=="playing" and state.worldX<=LEVEL_START_MAX_X
end

local function buildTemporalInputs(observationInputs,recentGlobalInputs)
  local temporalInputs={}
  local currentGlobals={}
  for featureIndex=1,GLOBAL_INPUT_COUNT do
    currentGlobals[featureIndex]=observationInputs[GRID_INPUT_COUNT+featureIndex] or 0
  end
  local temporalIndex=0
  -- Two short history taps give the feed-forward NEAT network motion context
  -- without breaking legacy 185-input genomes. This is an SMB1-specific
  -- alternative to changing the project's topology into a recurrent network.
  for _,lag in ipairs(MEMORY_LAGS) do
    local pastGlobals=recentGlobalInputs[#recentGlobalInputs-lag+1]
    for featureIndex=1,GLOBAL_INPUT_COUNT do
      temporalIndex=temporalIndex+1
      local pastValue=pastGlobals and pastGlobals[featureIndex]
      temporalInputs[temporalIndex]=pastValue
        and clamp(currentGlobals[featureIndex]-pastValue,-1,1) or 0
    end
  end
  recentGlobalInputs[#recentGlobalInputs+1]=currentGlobals
  while #recentGlobalInputs>math.max(unpack(MEMORY_LAGS)) do table.remove(recentGlobalInputs,1) end
  return temporalInputs
end

local function recordBehaviorEvents(aiState,state)
  local previousState=aiState.previousEpisodeState
  if previousState then
    if previousState.grounded and not state.grounded then aiState.behavior.jumps=aiState.behavior.jumps+1 end
    if not previousState.grounded and state.grounded then
      aiState.behavior.landings=aiState.behavior.landings+1
      if state.worldX>=previousState.worldX then aiState.episodeReward=aiState.episodeReward+1 end
    end
    if state.power>(previousState.power or 0) then
      aiState.behavior.powerUps=aiState.behavior.powerUps+1
      aiState.episodeReward=aiState.episodeReward+8
    end
    local enemiesBySlot={}
    for _,enemy in ipairs(state.enemies or {}) do enemiesBySlot[enemy.slot]=enemy end
    for _,oldEnemy in ipairs(previousState.enemies or {}) do
      local priorDx=oldEnemy.worldX-previousState.worldX
      local currentEnemy=enemiesBySlot[oldEnemy.slot]
      local currentDx=currentEnemy and currentEnemy.worldX-state.worldX or nil
      if currentEnemy and oldEnemy.id==currentEnemy.id and state.worldX>previousState.worldX
        and priorDx>0 and currentDx and currentDx<=0 and priorDx-currentDx<128 then
        aiState.behavior.passedEnemies=aiState.behavior.passedEnemies+1
        aiState.episodeReward=aiState.episodeReward+12
      end
    end
  end
  while state.worldX>=aiState.nextLandmark*128 do
    aiState.episodeReward=aiState.episodeReward+3
    aiState.nextLandmark=aiState.nextLandmark+1
  end
  aiState.previousEpisodeState=state
end

local function behaviorDescriptor(aiState,progress)
  local behavior=aiState.behavior or {}
  return {math.floor(progress/128),math.min(10,behavior.jumps or 0),
    math.min(10,behavior.retreats or 0),math.min(10,behavior.passedEnemies or 0),
    math.min(5,behavior.powerUps or 0),math.min(10,(aiState.episodeFrames or 0)/600)}
end

local function addNoveltyAndArchive(populationState,descriptor)
  local archive=populationState.behaviorArchive or {}
  local distances={}
  for _,archived in ipairs(archive) do
    local distance=0
    for index,value in ipairs(descriptor) do
      distance=distance+math.abs(value-(archived[index] or 0))/(index==1 and 8 or 10)
    end
    distances[#distances+1]=distance/#descriptor
  end
  table.sort(distances)
  local neighborCount=math.min(5,#distances)
  local novelty=1
  if neighborCount>0 then
    novelty=0
    for index=1,neighborCount do novelty=novelty+distances[index] end
    novelty=novelty/neighborCount
  end
  local nearestDistance=distances[1] or 1
  if nearestDistance>=0.12 then
    archive[#archive+1]=descriptor
    while #archive>NOVELTY_ARCHIVE_LIMIT do table.remove(archive,1) end
  end
  populationState.behaviorArchive=archive
  return novelty
end

function AI.decide(aiState,state)
  aiState.frames=aiState.frames+1
  if state.phase~="playing" then
    return {reason=state.phase}
  end
  if aiState.startWorldX==nil then AI.beginEpisode(aiState,state) end
  aiState.episodeFrames=aiState.episodeFrames+1
  if state.worldX>aiState.furthestWorldX then aiState.furthestWorldX=state.worldX;aiState.lastProgressFrame=aiState.episodeFrames end
  aiState.bestForm=math.max(aiState.bestForm or 0,state.power==2 and 2 or (state.size==1 and 0 or 1))
  local passedEnemiesBefore=aiState.behavior.passedEnemies
  recordBehaviorEvents(aiState,state)
  local contextKey,contextKind,targetWorldX=describeExperienceContext(state)
  if not aiState.championMode then
    trackChallenge(aiState,state,contextKind,targetWorldX,
      aiState.behavior.passedEnemies>passedEnemiesBefore)
    finishPendingQTransition(aiState,state,contextKey,nil)
  end
  local observationInputs=AI.buildObservationInputs(state)
  local temporalInputs=buildTemporalInputs(observationInputs,aiState.recentGlobalInputs)
  local memoryInputs={}
  for memoryIndex=1,MEMORY_INPUT_COUNT do memoryInputs[memoryIndex]=temporalInputs[memoryIndex] end
  aiState.lastObservationInputs=observationInputs
  aiState.lastTemporalInputs=memoryInputs
  local genome=activeGenome(aiState)
  local urgentEnemy=findClosestThreat(state)
  local immediateHazard=(urgentEnemy and urgentEnemy.worldX-state.worldX<56)
    or hasGapAhead(state)>0
  local cachedAction=aiState.cachedAction
  if cachedAction and aiState.actionFramesRemaining>0
    and (not immediateHazard or aiState.championMode) then
    aiState.actionFramesRemaining=aiState.actionFramesRemaining-1
    aiState.lastState=state
    aiState.lastObservationInputs=observationInputs
    local heldAction={}
    for key,value in pairs(cachedAction) do heldAction[key]=value end
    heldAction.reason="held learned "..heldAction.name
    return heldAction
  end
  -- This extends MarI/O's FCEUX observe/evaluate/joypad cycle with an evolved
  -- action horizon. The horizon is an SMB1-specific extension, not copied code:
  -- https://github.com/juvester/mari-o-fceux/blob/master/neatevolve.lua
  if not aiState.championMode then finishPendingExperience(aiState,state,nil) end
  local action,actionIndex,enemy,actionScores,nodeValues,_,holdFrames,durationIndex,selectedContext=
    chooseAction(genome,state,observationInputs,memoryInputs,
      not aiState.championMode and aiState.populationState.experienceMemory or nil)
  action.name=ACTION_OPTIONS[actionIndex].name
  aiState.episodeDecisions=(aiState.episodeDecisions or 0)+1
  action.reason=enemy and ("learned "..action.name.." | threat "..enemy.name)
    or ("learned "..action.name)
  aiState.lastAction=actionIndex
  aiState.lastActionScores=actionScores
  aiState.lastNodeValues=nodeValues
  aiState.lastObservationInputs=observationInputs
  aiState.lastTemporalInputs=memoryInputs
  -- Champion evaluation uses the Python bridge's fixed 12-frame action step;
  -- training continues to evolve its own action horizon.
  aiState.actionFramesRemaining=aiState.championMode and 11 or holdFrames-1
  aiState.cachedAction=action
  aiState.lastHoldFrames=holdFrames
  if not aiState.championMode then
    rememberChallengeChoice(aiState,selectedContext,actionIndex,durationIndex)
    local activeKind=contextKind or (aiState.activeChallenge and aiState.activeChallenge.kind)
    if not activeKind then
      aiState.pendingExperience={context=selectedContext,
        choice=experienceChoice(actionIndex,durationIndex),worldX=state.worldX,
        episodeReward=aiState.episodeReward or 0}
    end
    aiState.pendingTransition={context=selectedContext,
      choice=experienceChoice(actionIndex,durationIndex),worldX=state.worldX,
      episodeReward=aiState.episodeReward or 0,frame=aiState.episodeFrames}
  end
  if action.name=="retreat" then
    aiState.behavior.retreats=aiState.behavior.retreats+1
  end
  aiState.lastState=state
  return action
end

-- FCEUX uses ARGB colors for gui.drawbox/drawline. MarI/O's port converts to
-- the byte order expected by FCEUX; keep that conversion local to the HUD.
local HUD_COLOR_CACHE={}
local function hudColor(argb)
  local cached=HUD_COLOR_CACHE[argb]
  if cached then return cached end
  local alpha=math.floor(argb/0x1000000)%256
  local red=math.floor(argb/0x10000)%256
  local green=math.floor(argb/0x100)%256
  local blue=argb%256
  local converted=alpha+blue*0x100+green*0x10000+red*0x1000000
  HUD_COLOR_CACHE[argb]=converted
  return converted
end

local function hudText(guiApi,x,y,value,foreground,background)
  local text=tostring(value)
  local palette={white=0xFFFFFFFF,cyan=0xFF00FFFF,yellow=0xFFFFFF00,green=0xFF42FF70,
    red=0xFFFF5050,gray=0xFF9AA7B2,black=0xFF101010}
  if guiApi.drawtext then
    guiApi.drawtext(x,y,text,hudColor(palette[foreground] or 0xFFFFFFFF),0)
  elseif guiApi.text then
    guiApi.text(x,y,text,foreground or "white",background or "black")
  end
end

local function hudBox(guiApi,x1,y1,x2,y2,fill,outline)
  local draw=guiApi.drawbox or guiApi.box
  if draw then draw(x1,y1,x2,y2,hudColor(fill),hudColor(outline or fill)) end
end

local function hudLine(guiApi,x1,y1,x2,y2,color)
  local draw=guiApi.drawline or guiApi.line
  if draw then draw(x1,y1,x2,y2,hudColor(color)) end
end

local HUD_NODE_POSITION_CACHE=setmetatable({}, {__mode="k"})
local function makeHudNodePositions(genome)
  local cached=HUD_NODE_POSITION_CACHE[genome]
  if cached and cached.geneCount==#(genome.genes or {}) then
    return cached.positions,cached.hiddenCount
  end
  local positions={}
  local gridStartX,gridStartY,gridStep=4,50,2
  for inputIndex=1,GRID_INPUT_COUNT do
    local cellIndex=inputIndex-1
    local row=math.floor(cellIndex/GRID_WIDTH)
    local column=cellIndex%GRID_WIDTH
    positions[inputIndex]={x=gridStartX+column*gridStep,y=gridStartY+row*gridStep}
  end
  for featureIndex=1,GLOBAL_INPUT_COUNT do
    local inputIndex=GRID_INPUT_COUNT+featureIndex
    local column=math.floor((featureIndex-1)/8)
    local row=(featureIndex-1)%8
    positions[inputIndex]={x=35+column*41,y=49+row*7}
  end
  positions[NEURAL_INPUT_COUNT]={x=76,y=98}
  -- Two compact rows expose the delayed global-feature inputs in the same graph.
  for memoryIndex=1,MEMORY_INPUT_COUNT do
    local featureIndex=(memoryIndex-1)%MEMORY_FEATURE_COUNT
    local lagColumn=math.floor((memoryIndex-1)/MEMORY_FEATURE_COUNT)
    positions[memoryNode(memoryIndex)]={x=103+lagColumn*10,y=48+featureIndex*3}
  end

  local hiddenNodes={}
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      for _,nodeId in ipairs({gene.sourceNode,gene.targetNode}) do
        if isHiddenNode(nodeId) and not positions[nodeId] then
          positions[nodeId]={pending=true}
          hiddenNodes[#hiddenNodes+1]=nodeId
        end
      end
    end
  end
  table.sort(hiddenNodes)
  for hiddenIndex,nodeId in ipairs(hiddenNodes) do
    local column=math.floor((hiddenIndex-1)/12)
    local row=(hiddenIndex-1)%12
    if hiddenIndex<=48 then positions[nodeId]={x=120+column*6,y=50+row*4} end
  end
  for actionIndex=1,ACTION_COUNT do
    positions[OUTPUT_NODE_OFFSET+actionIndex]={x=145,y=50+(actionIndex-1)*8}
  end
  HUD_NODE_POSITION_CACHE[genome]={positions=positions,hiddenCount=#hiddenNodes,
    geneCount=#(genome.genes or {})}
  return positions,#hiddenNodes
end

-- A tiny NES-style controller mirrors the exact buttons sent to joypad.set.
local function drawMiniController(guiApi,buttons)
  buttons=buttons or {}
  -- NTSC's default visible area ends at x=255, y=231 in FCEUX. Anchor the
  -- controller there so no edge is clipped or left floating from the corner.
  local controllerX,controllerY=255-66,231-20
  hudBox(guiApi,controllerX,controllerY,controllerX+66,controllerY+20,
    0xFFD1CEC5,0xFF20262D)

  -- The four ends of the cross light up independently.
  local idleDirection,activeDirection=0xFF252A30,0xFF3ED484
  hudBox(guiApi,controllerX+9,controllerY+4,controllerX+14,controllerY+17,
    idleDirection,idleDirection)
  hudBox(guiApi,controllerX+4,controllerY+8,controllerX+19,controllerY+14,
    idleDirection,idleDirection)
  local directions={
    {name="up",left=9,top=4,right=14,bottom=7},
    {name="down",left=9,top=15,right=14,bottom=17},
    {name="left",left=4,top=8,right=8,bottom=14},
    {name="right",left=15,top=8,right=19,bottom=14},
  }
  for _,direction in ipairs(directions) do
    if buttons[direction.name] then
      hudBox(guiApi,controllerX+direction.left,controllerY+direction.top,
        controllerX+direction.right,controllerY+direction.bottom,
        activeDirection,activeDirection)
    end
  end

  -- Select and Start are shown for the controller shape; training never
  -- presses Start automatically after Mario dies.
  hudBox(guiApi,controllerX+25,controllerY+10,controllerX+30,controllerY+13,
    buttons.select and activeDirection or idleDirection,idleDirection)
  hudBox(guiApi,controllerX+33,controllerY+10,controllerX+38,controllerY+13,
    buttons.start and activeDirection or idleDirection,idleDirection)
  for _,button in ipairs({{name="B",left=43},{name="A",left=56}}) do
    local pressed=buttons[button.name]
    hudBox(guiApi,controllerX+button.left,controllerY+6,
      controllerX+button.left+8,controllerY+16,
      pressed and 0xFFFF5A55 or 0xFF9B4445,pressed and 0xFFFFDF70 or 0xFF5A292E)
    hudText(guiApi,controllerX+button.left+1,controllerY+7,
      button.name,pressed and "black" or "white","black")
  end
end

-- FCEUX's gui.drawtext has no font-size argument. A tiny 3x5 pixel font keeps
-- sensor names inside the existing network panel instead of covering the game.
local SENSOR_LABELS={
  "SPEED X","SPEED Y","GROUND","SIZE","POWER","ENEMY DX","ENEMY DY","ENEMY VX",
  "ENEMY ID","ITEM?","ITEM DX","ITEM DY","ITEM ID","GAP","NEAR","BIAS",
}

local SMALL_GLYPHS={
  A={"010","101","111","101","101"}, B={"110","101","110","101","110"},
  D={"110","101","101","101","110"}, E={"111","100","110","100","111"},
  G={"011","100","101","101","011"}, I={"111","010","010","010","111"},
  M={"101","111","111","101","101"}, N={"101","111","111","111","101"},
  O={"010","101","101","101","010"}, P={"110","101","110","100","100"},
  R={"110","101","110","101","101"}, S={"011","100","010","001","110"},
  T={"111","010","010","010","010"}, U={"101","101","101","101","111"},
  V={"101","101","101","101","010"}, W={"101","101","111","111","101"},
  X={"101","101","010","101","101"}, Y={"101","101","010","010","010"},
  Z={"111","001","010","100","111"}, ["?"]={"110","001","010","000","010"},
}

local SMALL_GLYPH_RUNS={}
for character,glyph in pairs(SMALL_GLYPHS) do
  local runs={}
  for row=1,5 do
    local column=1
    while column<=3 do
      if glyph[row]:sub(column,column)=="1" then
        local runStart=column
        repeat column=column+1 until column>3 or glyph[row]:sub(column,column)~="1"
        runs[#runs+1]={runStart-1,row-1,column-2}
      else
        column=column+1
      end
    end
  end
  SMALL_GLYPH_RUNS[character]=runs
end

local function drawSmallText(guiApi,x,y,value,color)
  for characterIndex=1,#value do
    local runs=SMALL_GLYPH_RUNS[value:sub(characterIndex,characterIndex)]
    if runs then
      local characterX=x+(characterIndex-1)*4
      for _,run in ipairs(runs) do
        hudBox(guiApi,characterX+run[1],y+run[2],
          characterX+run[3],y+run[2],color,color)
      end
    end
  end
end

-- Rasterize the fixed sensor names once. FCEUX can draw a truecolor GD image
-- in one call, avoiding hundreds of tiny drawbox calls on every game frame.
local function buildSensorLabelImage()
  local imageX,imageY,imageWidth,imageHeight=39,46,77,57
  local transparent=string.char(127,0,0,0)
  local backing=string.char(24,0x10,0x2D,0x4A)
  local white=string.char(0,255,255,255)
  local yellow=string.char(0,255,255,0)
  local pixels={}
  for pixelIndex=1,imageWidth*imageHeight do pixels[pixelIndex]=transparent end
  local function setPixel(screenX,screenY,color)
    local column,row=screenX-imageX,screenY-imageY
    if column>=0 and column<imageWidth and row>=0 and row<imageHeight then
      pixels[row*imageWidth+column+1]=color
    end
  end
  for sensorIndex,sensorLabel in ipairs(SENSOR_LABELS) do
    local column=math.floor((sensorIndex-1)/8)
    local row=(sensorIndex-1)%8
    local labelX,labelY=40+column*41,47+row*7
    local labelWidth=#sensorLabel*4-1
    for backgroundY=labelY-1,labelY+5 do
      for backgroundX=labelX-1,labelX+labelWidth do
        setPixel(backgroundX,backgroundY,backing)
      end
    end
    local textColor=sensorIndex==16 and yellow or white
    for characterIndex=1,#sensorLabel do
      local runs=SMALL_GLYPH_RUNS[sensorLabel:sub(characterIndex,characterIndex)]
      if runs then
        local characterX=labelX+(characterIndex-1)*4
        for _,run in ipairs(runs) do
          for pixelX=characterX+run[1],characterX+run[3] do
            setPixel(pixelX,labelY+run[2],textColor)
          end
        end
      end
    end
  end
  -- GD 2.x truecolor header: signature, width, height, truecolor, transparent.
  local header=string.char(255,254,0,imageWidth,0,imageHeight,1,255,255,255,255)
  return imageX,imageY,header..table.concat(pixels)
end

local SENSOR_LABEL_IMAGE_X,SENSOR_LABEL_IMAGE_Y,SENSOR_LABEL_IMAGE=buildSensorLabelImage()
local SENSOR_LABEL_IMAGE_SUPPORTED=nil
local function drawSensorLabels(guiApi)
  if guiApi.drawimage and SENSOR_LABEL_IMAGE_SUPPORTED~=false then
    if SENSOR_LABEL_IMAGE_SUPPORTED then
      guiApi.drawimage(SENSOR_LABEL_IMAGE_X,SENSOR_LABEL_IMAGE_Y,SENSOR_LABEL_IMAGE)
      return
    end
    local success=pcall(guiApi.drawimage,
      SENSOR_LABEL_IMAGE_X,SENSOR_LABEL_IMAGE_Y,SENSOR_LABEL_IMAGE)
    SENSOR_LABEL_IMAGE_SUPPORTED=success
    if success then return end
  end
  for sensorIndex,sensorLabel in ipairs(SENSOR_LABELS) do
    local column=math.floor((sensorIndex-1)/8)
    local row=(sensorIndex-1)%8
    local labelX=40+column*41
    local labelY=47+row*7
    local labelWidth=#sensorLabel*4-1
    -- A narrow backing hides crossing links behind each label; this is part
    -- of the existing graph, not another panel over the game picture.
    hudBox(guiApi,labelX-1,labelY-1,labelX+labelWidth,labelY+5,
      0xD0102D4A,0xD0102D4A)
    drawSmallText(guiApi,labelX,labelY,sensorLabel,
      sensorIndex==16 and 0xFFFFFF00 or 0xFFFFFFFF)
  end
end

-- Draw a compact live inspector in the upper-left, leaving the game view clear.
function AI.drawNeuralInspector(guiApi,aiState,state,action,buttons)
  if not guiApi or not (guiApi.text or guiApi.drawtext) then return false end
  if not SHOW_NEURAL_INSPECTOR then
    hudText(guiApi,224,12,"[AI]","cyan","black")
    return false
  end
  local genome=activeGenome(aiState)
  if not genome then return false end
  local nodePositions=makeHudNodePositions(genome)
  -- The graph and labeled sensors stay above Mario's ground-level play area.
  hudBox(guiApi,0,10,135,38,0xB0000000,0xB0000000)
  hudBox(guiApi,0,40,180,106,0x90000000,0x90000000)
  hudText(guiApi,2,12,"MARIO AI  NEAT","cyan","black")
  hudText(guiApi,2,20,string.format("G%d #%d/%d S%d",
    aiState.populationState.generation,aiState.genomeIndex,#aiState.populationState.genomes,
    genome.species or 0),"white","black")
  local progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))
  local actionName=action and action.name or "idle"
  local shortAction=({run="RUN",jump_run="JUMP",retreat="BACK",brake="STOP",
    jump_place="HOP",walk="WALK"})[actionName] or "IDLE"
  hudText(guiApi,2,28,string.format("X%d +%d %s M%d",state.worldX,progress,
    shortAction,AI.experienceMemorySize(aiState.populationState)),"white","black")

  -- Keep the network rendering lightweight so the game remains responsive.
  local drawnConnections=0
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      local source,target=nodePositions[gene.sourceNode],nodePositions[gene.targetNode]
      if source and target and not source.pending and not target.pending and drawnConnections<50 then
        local color=gene.weight>=0 and 0x6000DD55 or 0x60FF453A
        hudLine(guiApi,source.x,source.y,target.x,target.y,color)
        drawnConnections=drawnConnections+1
      end
    end
  end

  -- Grid is the local tile view: dark empty, green solid, red enemy.
  hudText(guiApi,2,41,"ACTIONS","white","black")
  for inputIndex=1,GRID_INPUT_COUNT do
    local point=nodePositions[inputIndex]
    local value=(aiState.lastObservationInputs or {})[inputIndex] or 0
    local color=value<0 and 0xFFFF5C64 or (value>0 and 0xFF56D69A or 0xFF344454)
    hudBox(guiApi,point.x,point.y,point.x+1,point.y+1,color,color)
  end

  -- Global features are a small activation column beside the 13x13 grid.
  for featureIndex=1,GLOBAL_INPUT_COUNT do
    local inputIndex=GRID_INPUT_COUNT+featureIndex
    local point=nodePositions[inputIndex]
    local value=(aiState.lastObservationInputs or {})[inputIndex] or 0
    local color=value<0 and 0xFFFF6873 or (value>0 and 0xFF62D6A5 or 0xFF617180)
    hudBox(guiApi,point.x,point.y,point.x+1,point.y+1,color,color)
  end
  local biasPoint=nodePositions[NEURAL_INPUT_COUNT]
  hudBox(guiApi,biasPoint.x,biasPoint.y,biasPoint.x+2,biasPoint.y+2,0xFFFFFF00,0xFFFFFF00)

  -- Hidden activations and output nodes are colored by their current value.
  for nodeId,point in pairs(nodePositions) do
    if (isHiddenNode(nodeId) or isMemoryInput(nodeId)) and not point.pending then
      local activation=(aiState.lastNodeValues or {})[nodeId] or 0
      local color=activation>=0 and 0xFF62D6A5 or 0xFFFF6873
      hudBox(guiApi,point.x-1,point.y-1,point.x+1,point.y+1,color,color)
    end
  end
  local outputLabels={"run","jrun","back","stop","hop","walk"}
  for actionIndex,option in ipairs(ACTION_OPTIONS) do
    local nodeId=OUTPUT_NODE_OFFSET+actionIndex
    local point=nodePositions[nodeId]
    local selected=action and action.name==option.name
    hudText(guiApi,point.x+4,point.y-3,outputLabels[actionIndex],
      selected and "yellow" or "white","black")
  end

  drawSensorLabels(guiApi)
  drawMiniController(guiApi,buttons)
  return true
end

local function processHudClick()
  if HUD_CLICK_COOLDOWN>0 then HUD_CLICK_COOLDOWN=HUD_CLICK_COOLDOWN-1;return end
  if not input or not input.get then return end
  local mouse=input.get()
  if mouse and mouse.click==1 and mouse.xmouse>=190 and mouse.xmouse<=255
    and mouse.ymouse>=8 and mouse.ymouse<=26 then
    SHOW_NEURAL_INSPECTOR=not SHOW_NEURAL_INSPECTOR
    HUD_CLICK_COOLDOWN=10
  end
end

function AI.finishEpisode(aiState,state,forced_reason)
  if not aiState.episodeActive then return nil end
  local genome=activeGenome(aiState)
  local progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))
  local outcomeReason=forced_reason or state.phase
  if not aiState.championMode then
    finishPendingQTransition(aiState,state,nil,outcomeReason)
    finishPendingExperience(aiState,state,outcomeReason)
    if aiState.activeChallenge then
      local challengeCleared=outcomeReason=="victory"
        or (state.worldX>=aiState.activeChallenge.targetWorldX+8 and state.grounded)
      labelChallenge(aiState,challengeCleared)
    end
  end
  local survival=math.min(aiState.episodeFrames,12000)*0.02
  local power=(aiState.bestForm or 0)*150
  local descriptor=behaviorDescriptor(aiState,progress)
  local novelty=addNoveltyAndArchive(aiState.populationState,descriptor)
  local fitness=progress*10+survival+power+aiState.episodeReward+novelty*6
  if state and state.phase=="death" then fitness=fitness-120
  elseif state and state.phase=="victory" then fitness=fitness+10000 end
  if forced_reason=="stuck" then fitness=fitness-20 end
  if forced_reason=="timeout" then fitness=fitness-80 end
  aiState.totalEpisodes=aiState.totalEpisodes+1
  local episode={generation=aiState.populationState.generation,genomeIndex=aiState.genomeIndex,
    fitness=fitness,startWorldX=aiState.startWorldX or state.worldX,
    maxWorldX=aiState.furthestWorldX or state.worldX,progress=progress,
    frames=aiState.episodeFrames,reason=forced_reason or state.phase,
    power=aiState.bestForm or 0,jumps=aiState.behavior.jumps or 0,
    retreats=aiState.behavior.retreats or 0,passedEnemies=aiState.behavior.passedEnemies or 0,
    landings=aiState.behavior.landings or 0,powerUps=aiState.behavior.powerUps or 0,
    episodeReward=aiState.episodeReward or 0,novelty=novelty,timestamp=currentTimestamp()}
  updateEpisodeHistory(aiState.populationState,episode)
  aiState.startWorldX=nil
  aiState.episodeActive=false
  aiState.lastFitness=fitness
  aiState.lastNovelty=novelty
  if not aiState.championMode then
    genome.fitness=fitness
    updateTopPerformers(aiState.populationState,episode,genome)
    aiState.genomeIndex=aiState.genomeIndex+1
    if aiState.genomeIndex>#aiState.populationState.genomes then
      aiState.populationState=AI.nextGeneration(aiState.populationState)
      aiState.genomeIndex=1
    end
    aiState.populationState.nextGenomeIndex=aiState.genomeIndex
  end
  aiState.finished=true
  return fitness
end

function AI.episodeStopReason(aiState,state)
  if aiState.episodeFrames>=12000 then return "timeout" end
  local progress=(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX)
  if aiState.episodeFrames>=INITIAL_PROGRESS_DEADLINE_FRAMES
    and progress<MIN_INITIAL_PROGRESS_PIXELS then return "stuck" end
  if aiState.episodeFrames-aiState.lastProgressFrame>600 then return "stuck" end
  return nil
end

-- A checkpoint taken during the death animation can still look playable in
-- RAM. Discard it when Mario dies almost immediately without making progress.
function AI.isUnsafeTrainingStart(aiState,state)
  local progress=math.max(0,(aiState.furthestWorldX or state.worldX)-(aiState.startWorldX or state.worldX))
  return aiState.episodeActive and aiState.episodeFrames<=12 and progress<=16
end

function AI.abandonEpisode(aiState)
  aiState.startWorldX=nil
  aiState.furthestWorldX=nil
  aiState.episodeFrames=0
  aiState.episodeReward=0
  aiState.episodeActive=false
  aiState.finished=false
  aiState.cachedAction=nil
  aiState.actionFramesRemaining=0
  aiState.recentGlobalInputs={}
  aiState.previousEpisodeState=nil
end

function AI.run()
  assert(memory and memory.readbyte and joypad and joypad.set and emu and emu.frameadvance
    and emu.registerexit,
    "Load Mario AI NEAT in FCEUX with an NES SMB1 ROM open")
  math.randomseed(os.time())
  local databasePath=getDatabasePath()
  local loaded=AI.load(databasePath)
  local aiState=AI.new(loaded or AI.newPopulation())
  local logPath=getLogPath()
  local needsHistoryMigration=loaded and not aiState.populationState.legacyLogImported
  local importedEpisodes,archivedGenomes=0,0
  if needsHistoryMigration then
    importedEpisodes,archivedGenomes=AI.importLegacyLogHistory(aiState.populationState,logPath)
  end
  local stateAdapter,stateProblem
  -- Champion play must be able to finish the flagpole sequence and enter the
  -- next level; only training restores a fixed start after each attempt.
  if USE_FIXED_TRAINING_STATE then
    stateAdapter,stateProblem=AI.createStateAdapter(savestate,TRAINING_SAVESTATE_SLOT)
  end
  local fixedTraining=stateAdapter~=nil
  local stateSaved=false
  local waitingForRespawn=false
  local awaitingNextLevel=false
  local flagpoleWorldX=nil
  if PLAY_CHAMPION_ONLY and loaded then
    aiState.championMode=true
    local champion=AI.bestPerformer(aiState.populationState)
    aiState.genomeIndex=champion.genomeIndex or 1
    aiState.championGenome=cloneGenome(champion.genome)
    AI.appendLog(string.format("champion selected | source=%s | generation=%s | fitness=%.2f",
      champion.source,tostring(champion.generation or aiState.populationState.generation),champion.fitness),logPath)
  end
  local function savePopulation(context)
    aiState.databaseOK=AI.save(aiState.populationState,databasePath)
    if not aiState.databaseOK then AI.appendLog("database save failed: "..context,logPath) end
    return aiState.databaseOK
  end
  if needsHistoryMigration then
    AI.appendLog(string.format("checkpoint history migrated | episodes=%d | current-genome snapshots=%d",
      importedEpisodes,archivedGenomes),logPath)
    savePopulation("legacy checkpoint history migration")
  end
  AI.appendLog(string.format("started | database=%s | generation=%d | population=%d | mode=%s | state=%s | experience_contexts=%d | timer_aid=%s | test_lives=%s",
    loaded and "loaded" or "new",aiState.populationState.generation,#aiState.populationState.genomes,
    aiState.championMode and "champion" or "training",
    fixedTraining and ("slot "..TRAINING_SAVESTATE_SLOT.." via "..stateAdapter.kind) or (stateProblem or "continuous"),
    AI.experienceMemorySize(aiState.populationState),
    SET_TIMER_TO_999_PER_EPISODE and "999 per episode" or "off",
    TESTING_INFINITE_LIVES and "refreshed" or "off"),logPath)
  if not loaded then savePopulation("initial population") end
  emu.registerexit(function()
    savePopulation("FCEUX exit")
    AI.appendLog("stopped | episodes="..aiState.totalEpisodes.." | generation="..aiState.populationState.generation,logPath)
  end)
  local function restoreTrainingState(reason)
    if not fixedTraining then return end
    joypad.set(1,{})
    if stateAdapter:load() then
      AI.keepLivesForTesting()
      AI.setTimerTo999()
      AI.appendLog("restored training slot "..TRAINING_SAVESTATE_SLOT.." after "..reason,logPath)
    else
      fixedTraining=false
      AI.appendLog("training slot restore failed; switched to continuous training",logPath)
    end
  end
  local lastSaveFrame=0
  local waitingForLevelStartLogged=false
  while true do
    local state=AI.observe(aiState.frames+1)
    AI.keepLivesForTesting()
    local nextLevelReady=awaitingNextLevel and state.phase=="playing"
      and state.worldX<=LEVEL_START_MAX_X
      and state.worldX<(flagpoleWorldX or state.worldX)-128
    local restoredAfterVictory=false
    if nextLevelReady then
      awaitingNextLevel=false
      flagpoleWorldX=nil
      if not aiState.championMode then
        restoreTrainingState("victory")
        restoredAfterVictory=true
      end
    end
    if waitingForRespawn and state.phase=="playing" then
      waitingForRespawn=false
      AI.appendLog("Mario respawned; finding a new training start",logPath)
    end
    local waitingForLevelStart=false
    if state.phase=="playing" and not aiState.episodeActive
      and not awaitingNextLevel and not restoredAfterVictory then
      if state.worldX>LEVEL_START_MAX_X or (not aiState.championMode and not AI.isValidTrainingStart(state)) then
        waitingForLevelStart=true
        if not waitingForLevelStartLogged then
          waitingForLevelStartLogged=true
          AI.appendLog(string.format("waiting for level start | current_x=%d | reset SMB1 to the selected world's start and restart the Lua script",
            state.worldX),logPath)
        end
      else
        waitingForLevelStartLogged=false
        if fixedTraining and not stateSaved then
          if stateAdapter:save() then
            stateSaved=true
            AI.appendLog("saved fixed training start in slot "..TRAINING_SAVESTATE_SLOT,logPath)
          else
            fixedTraining=false
            AI.appendLog("training slot save failed; switched to continuous training",logPath)
          end
        end
        AI.beginEpisode(aiState,state)
        AI.setTimerTo999()
        AI.appendLog(string.format("episode start | generation=%d | genome=%d/%d | world=%d | level=%d | x=%d | power=%d | action_repeat=12",
          aiState.populationState.generation,aiState.genomeIndex,#aiState.populationState.genomes,
          (state.worldNumber or 0)+1,(state.levelNumber or 0)+1,state.worldX,state.power),logPath)
      end
    end
    if waitingForLevelStart then
      joypad.set(1,{})
    elseif awaitingNextLevel or restoredAfterVictory then
      joypad.set(1,{})
    elseif state.phase=="playing" then
      local action=AI.decide(aiState,state)
      local buttons={}
      for _,name in ipairs({"left","right","up","down","A","B","select"}) do
        if action[name] then buttons[name]=true end
      end
      joypad.set(1,buttons)
      processHudClick()
      if gui then AI.drawNeuralInspector(gui,aiState,state,action,buttons) end
      local reason=AI.episodeStopReason(aiState,state)
      if reason then
        local fitness=AI.finishEpisode(aiState,state,reason)
        AI.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d | decisions=%d | elapsed_seconds=%d",
          reason,fitness or 0,aiState.furthestWorldX or state.worldX,aiState.episodeFrames,
          aiState.episodeDecisions or 0,os and os.difftime and os.difftime(os.time(),aiState.episodeStartTime or os.time()) or 0),logPath)
        if not aiState.championMode then savePopulation("episode "..reason) end
        if fixedTraining then restoreTrainingState(reason) end
      end
    elseif state.phase=="death" and fixedTraining and stateSaved
      and AI.isUnsafeTrainingStart(aiState,state) then
      AI.appendLog(string.format("discarded unsafe training start | x=%d | frames=%d | waiting for respawn",
        aiState.startWorldX or state.worldX,aiState.episodeFrames),logPath)
      AI.abandonEpisode(aiState)
      stateSaved=false
      waitingForRespawn=true
      joypad.set(1,{})
    elseif (state.phase=="death" or state.phase=="victory") and aiState.episodeActive then
      joypad.set(1,{})
      local fitness=AI.finishEpisode(aiState,state)
      AI.appendLog(string.format("episode end | reason=%s | fitness=%.2f | max_x=%d | frames=%d | decisions=%d | elapsed_seconds=%d",
        state.phase,fitness or 0,aiState.furthestWorldX or state.worldX,aiState.episodeFrames,
        aiState.episodeDecisions or 0,os and os.difftime and os.difftime(os.time(),aiState.episodeStartTime or os.time()) or 0),logPath)
      if not aiState.championMode then savePopulation("episode "..state.phase) end
      if state.phase=="victory" then
        if aiState.championMode and fixedTraining then
          restoreTrainingState("victory")
          AI.appendLog("champion evaluation restored clean level start after victory",logPath)
        else
          awaitingNextLevel=true
          flagpoleWorldX=state.worldX
          AI.appendLog("flagpole touched; waiting for SMB1 level transition",logPath)
        end
      else
        if fixedTraining then restoreTrainingState(state.phase) end
      end
    else
      local action=AI.decide(aiState,state)
      joypad.set(1,{})
    end
    if aiState.frames-lastSaveFrame>=SAVE_INTERVAL_FRAMES then
      savePopulation("periodic checkpoint");lastSaveFrame=aiState.frames
    end
    emu.frameadvance()
  end
end

if rawget(_G,"MARIO_AI_TEST") then return AI end
AI.run()
