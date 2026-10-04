local function install(AI,C)
local USE_FIXED_TRAINING_STATE=C.USE_FIXED_TRAINING_STATE
local TRAINING_SAVESTATE_SLOT=C.TRAINING_SAVESTATE_SLOT
local LEVEL_START_MAX_X=C.LEVEL_START_MAX_X
local PLAY_CHAMPION_ONLY=C.PLAY_CHAMPION_ONLY
local SAVE_INTERVAL_FRAMES=C.SAVE_INTERVAL_FRAMES
local SET_TIMER_TO_999_PER_EPISODE=C.SET_TIMER_TO_999_PER_EPISODE
local TESTING_INFINITE_LIVES=C.TESTING_INFINITE_LIVES
local getDatabasePath=AI.getDatabasePath
local getLogPath=AI.getLogPath
local cloneGenome=AI.cloneGenome
local processHudClick=AI.processHudClick
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
end

return install
