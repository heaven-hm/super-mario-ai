local function install(AI,C)
local SENSOR_RADIUS_TILES=C.SENSOR_RADIUS_TILES
local GRID_WIDTH=C.GRID_WIDTH
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local GLOBAL_INPUT_COUNT=C.GLOBAL_INPUT_COUNT
local OBSERVATION_INPUT_COUNT=C.OBSERVATION_INPUT_COUNT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local clamp=C.clamp
local RAM=AI.RAM
local NON_SOLID_TILES=AI.NON_SOLID_TILES
local INACTIVE_ENEMY_STATES=AI.INACTIVE_ENEMY_STATES
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
AI.isSolidTileAtOffset=isSolidTileAtOffset

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
AI.hasGapAhead=hasGapAhead

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
AI.findNearestEnemy=findNearestEnemy

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
end

return install
