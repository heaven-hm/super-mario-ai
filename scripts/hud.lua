-- FCEUX overlay HUD: neural network inspector and controller visualization

local hud = {}

local HUD_WIDTH = 320
local HUD_PANEL_HEIGHT = 240
local CELL_SIZE = 8
local NEURON_RADIUS = 3

local function clamp(value, low, high)
  return math.max(low, math.min(high, value))
end

-- Draw neural network visualization on HUD
function hud.drawNeuralNetwork(gui, aiState, state, action, buttons)
  if not gui then return end
  if not aiState or not aiState.runner or not aiState.runner.genome then return end
  
  gui.defaultBackground(0x88000000)
  gui.defaultTextBackground(0xFF000000)
  gui.drawRectangle(0,0,HUD_WIDTH,HUD_PANEL_HEIGHT,0xFF222222,0xFF000000)
  
  local y = 4
  gui.text(4,y,"GEN:"..tostring(aiState.populationState.generation or 0),0xFFFFFFFF)
  y = y + 12
  gui.text(4,y,"ID:"..tostring(aiState.genomeIndex or 0),0xFFFFFFFF)
  y = y + 12
  gui.text(4,y,"X:"..tostring(math.floor(state.worldX or 0)),0xFFFFFFFF)
  y = y + 12
  gui.text(4,y,"FIT:"..string.format("%.0f",aiState.runner.genome.fitness or 0),0xFFFFFFFF)
  y = y + 12
  
  if action then
    gui.text(4,y,"ACT:"..tostring(action.name or ""),0xFFFFFFFF)
  end
  
  -- Draw simple network topology: inputs on left, outputs on right
  local inputY = 60
  local outputY = 60
  local neuronWidth = 30
  
  -- Draw output nodes
  for i = 1, 6 do
    local color = action and action.buttonIndex == i and 0xFF00FF00 or 0xFFFFFFFF
    gui.drawRectangle(290-neuronWidth, outputY + (i-1)*16, 20, 12, color, color)
  end
  
  gui.text(4, HUD_PANEL_HEIGHT-12, "CLICK TO TOGGLE", 0xFFFFFFFF)
end

-- Check if user clicked the HUD toggle
function hud.getClickInput()
  if not input or not input.get then return nil end
  local clicked = input.get()
  if clicked and clicked.xmouse and clicked.ymouse then
    return clicked.xmouse, clicked.ymouse
  end
  return nil
end

-- Draw controller button states on screen
function hud.drawControllerState(gui, buttons)
  if not gui or not buttons then return end
  local buttonNames = {"A", "B", "Select", "Start", "Up", "Down", "Left", "Right"}
  local y = 200
  for i, name in ipairs(buttonNames) do
    local color = buttons[name] and 0xFF00FF00 or 0xFF888888
    gui.text(10 + (i-1)*30, y, name, color)
  end
end

-- Get HUD state and visibility from environment
function hud.isVisible()
  return (os.getenv("MARIO_AI_HIDE_HUD") or "") ~= "1"
end

return hud
