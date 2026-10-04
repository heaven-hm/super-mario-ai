local function install(AI,C)
local ACTION_COUNT=C.ACTION_COUNT
local DURATION_OUTPUT_COUNT=C.DURATION_OUTPUT_COUNT
local EXPERIENCE_CONTEXT_LIMIT=C.EXPERIENCE_CONTEXT_LIMIT
local EXPERIENCE_TRACE_LIMIT=C.EXPERIENCE_TRACE_LIMIT
local NON_STOMPABLE_ENEMIES=AI.NON_STOMPABLE_ENEMIES
local INACTIVE_ENEMY_STATES=AI.INACTIVE_ENEMY_STATES
local clamp=C.clamp
local isSolidTileAtOffset=AI.isSolidTileAtOffset
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
AI.findClosestThreat=findClosestThreat
AI.describeExperienceContext=describeExperienceContext
AI.experienceChoice=experienceChoice
AI.similarExperienceContexts=similarExperienceContexts
AI.experienceBiasFromMemory=experienceBiasFromMemory
end

return install
