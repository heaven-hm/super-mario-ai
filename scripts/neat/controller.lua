local function install(AI,C)
local ACTION_COUNT=C.ACTION_COUNT
local DURATION_OUTPUT_COUNT=C.DURATION_OUTPUT_COUNT
local ACTION_OPTIONS=C.ACTION_OPTIONS
local ACTION_HOLD_FRAMES=C.ACTION_HOLD_FRAMES
local EPISODE_HISTORY_LIMIT=C.EPISODE_HISTORY_LIMIT
local EXPERIENCE_TRACE_LIMIT=C.EXPERIENCE_TRACE_LIMIT
local GLOBAL_INPUT_COUNT=C.GLOBAL_INPUT_COUNT
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local LEVEL_START_MAX_X=C.LEVEL_START_MAX_X
local NON_STOMPABLE_ENEMIES=AI.NON_STOMPABLE_ENEMIES
local MEMORY_INPUT_COUNT=C.MEMORY_INPUT_COUNT
local MEMORY_LAGS=C.MEMORY_LAGS
local NOVELTY_ARCHIVE_LIMIT=C.NOVELTY_ARCHIVE_LIMIT
local TOP_PERFORMER_LIMIT=C.TOP_PERFORMER_LIMIT
local INITIAL_PROGRESS_DEADLINE_FRAMES=C.INITIAL_PROGRESS_DEADLINE_FRAMES
local MIN_INITIAL_PROGRESS_PIXELS=C.MIN_INITIAL_PROGRESS_PIXELS
local clamp=C.clamp
local findClosestThreat=AI.findClosestThreat
local describeExperienceContext=AI.describeExperienceContext
local experienceChoice=AI.experienceChoice
local similarExperienceContexts=AI.similarExperienceContexts
local experienceBiasFromMemory=AI.experienceBiasFromMemory
local isSolidTileAtOffset=AI.isSolidTileAtOffset
local hasGapAhead=AI.hasGapAhead
local cloneGenome=AI.cloneGenome
local genomeSignature=AI.genomeSignature
local currentTimestamp=AI.currentTimestamp
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
AI.activeGenome=activeGenome
end

return install
