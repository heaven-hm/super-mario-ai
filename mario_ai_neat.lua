-- FCEUX-compatible entry point. Public AI methods are installed by focused modules.
local AI={}
local C=require("scripts.neat.config")
local PLAY_CHAMPION_ONLY = false
C.PLAY_CHAMPION_ONLY=PLAY_CHAMPION_ONLY
local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
local script=source:sub(1,1)=="@" and source:sub(2) or ""
C.PROJECT_DIR=script:match("^(.*[/\\])") or ""

require("scripts.smb1.memory")(AI,C)
require("scripts.neat.observation")(AI,C)
require("scripts.neat.genome")(AI,C)
require("scripts.neat.mutation")(AI,C)
require("scripts.neat.species")(AI,C)
require("scripts.neat.population")(AI,C)
require("scripts.storage")(AI,C)
require("scripts.qmemory")(AI,C)
require("scripts.neat.controller")(AI,C)
require("scripts.hud")(AI,C)
require("scripts.neat.runtime")(AI,C)

if rawget(_G,"MARIO_AI_TEST") then return AI end
AI.run()
