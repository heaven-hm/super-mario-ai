local function install(AI,C)
local ACTION_COUNT=C.ACTION_COUNT
local ACTION_OPTIONS=C.ACTION_OPTIONS
local OUTPUT_NODE_OFFSET=C.OUTPUT_NODE_OFFSET
local GLOBAL_INPUT_COUNT=C.GLOBAL_INPUT_COUNT
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local GRID_WIDTH=C.GRID_WIDTH
local MEMORY_FEATURE_COUNT=C.MEMORY_FEATURE_COUNT
local MEMORY_INPUT_COUNT=C.MEMORY_INPUT_COUNT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local activeGenome=AI.activeGenome
local memoryNode=AI.memoryNode
local isHiddenNode=AI.isHiddenNode
local isMemoryInput=AI.isMemoryInput
local clamp=C.clamp
local SHOW_NEURAL_INSPECTOR=os.getenv("MARIO_AI_HIDE_HUD")~="1"
local HUD_CLICK_COOLDOWN=0
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
AI.processHudClick=processHudClick
end

return install
