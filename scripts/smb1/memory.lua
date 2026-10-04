-- SMB1 RAM addresses, tile observation encoding, and state reading

local memory = {}

-- SMB1 RAM addresses
memory.RAM = {
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

-- SMB1 enemy names by ID
memory.ENEMY_NAME = {
  [0x00]="green koopa", [0x02]="buzzy beetle", [0x03]="red koopa",
  [0x05]="hammer bro", [0x06]="goomba", [0x07]="bloober",
  [0x08]="bullet bill", [0x09]="paratroopa", [0x0A]="cheep-cheep",
  [0x0B]="cheep-cheep", [0x0C]="podoboo", [0x0D]="piranha plant",
  [0x0E]="jumping paratroopa", [0x0F]="red paratroopa",
  [0x10]="flying paratroopa", [0x11]="lakitu", [0x12]="spiny",
  [0x14]="flying cheep-cheep", [0x15]="bowser flame", [0x2D]="bowser",
  [0x33]="bullet bill",
}

-- Tile encoding: which tiles are not solid
memory.NON_SOLID_TILES = {[0x00]=true,[0x08]=true,[0x24]=true,[0x25]=true,[0x26]=true,
  [0x88]=true,[0xC2]=true,[0xC3]=true,[0xC5]=true}

-- Enemy states that are inactive (dead, in shell, etc)
memory.INACTIVE_ENEMY_STATES = {[0x02]=true,[0x03]=true,[0x04]=true,[0x20]=true,[0x22]=true,
  [0x23]=true,[0x83]=true,[0x84]=true,[0xC4]=true}

-- Enemies that cannot be defeated by jumping on them
memory.NON_STOMPABLE_ENEMIES = {[0x07]=true,[0x0C]=true,[0x0D]=true,[0x11]=true,[0x12]=true}

local function toSignedByte(value)
  return value >= 128 and value - 256 or value
end

local function readByte(address)
  return memory.readbyte and memory.readbyte(address) or 0
end

-- Convert SMB1 RAM layout into one readable snapshot
function memory.observe(frameNumber)
  local RAM = memory.RAM
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
  if operationMode == 1 and (playerState == 0x04 or playerState == 0x05) then
    state.phase = "victory"
  end

  -- Read both metatile pages in one FCEUX call
  local tileBytes=memory.readbyterange and memory.readbyterange(RAM.tiles,416)
  if type(tileBytes)=="string" and #tileBytes==416 then
    local tileValues={string.byte(tileBytes,1,416)}
    for tileIndex=0,415 do state.tiles[tileIndex]=tileValues[tileIndex+1] end
  else
    for tileIndex=0,415 do state.tiles[tileIndex]=readByte(RAM.tiles+tileIndex) end
  end

  for enemySlot = 0, 4 do
    if readByte(RAM.enemy_present + enemySlot) ~= 0 then
      local enemyId = readByte(RAM.enemy_id + enemySlot)
      state.enemies[#state.enemies+1] = {
        slot=enemySlot,
        id=enemyId,
        name=memory.ENEMY_NAME[enemyId] or "unknown object",
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

return memory
