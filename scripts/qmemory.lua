-- Contextual Q-value lookup table: experience memory indexed by game state context

local qmemory = {}

local DURATION_OUTPUT_COUNT = 4
local EXPERIENCE_CONTEXT_LIMIT = 512
local EXPERIENCE_TRACE_LIMIT = 64
local NOVELTY_ARCHIVE_LIMIT = 48

-- Experience choice encoding
local function experienceChoice(actionIndex,durationIndex)
  return (actionIndex-1)*DURATION_OUTPUT_COUNT+durationIndex
end

-- Parse context features from compact string key
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
      features={kind="gap",width=tonumber(width),distance=tonumber(distance),
        grounded=grounded,speed=tonumber(speed),power=tonumber(power)}
    else
      local height,distance,grounded,speed,power=contextKey:match(
        "^obstacle:h(%d+):d(%d+):([^:]+):v(%d+):p(%d+)$")
      if height then
        features={kind="obstacle",height=tonumber(height),distance=tonumber(distance),
          grounded=grounded,speed=tonumber(speed),power=tonumber(power)}
      else
        local grounded,speed,power=contextKey:match(
          "^clear:([^:]+):v(%d+):p(%d+)$")
        if grounded then
          features={kind="clear",grounded=grounded,speed=tonumber(speed),power=tonumber(power)}
        end
      end
    end
  end
  experienceFeatureCache[contextKey]=features or false
  return features
end

-- Check if two contexts are similar (neighbor in experience space)
local function isContextNeighbor(firstContext, secondContext)
  local firstFeatures=experienceContextFeatures(firstContext)
  local secondFeatures=experienceContextFeatures(secondContext)
  if not firstFeatures or not secondFeatures or firstFeatures.kind~=secondFeatures.kind then
    return false
  end
  if firstFeatures.kind=="enemy" then
    return firstFeatures.class==secondFeatures.class
      and firstFeatures.vertical==secondFeatures.vertical
      and math.abs(firstFeatures.distance-secondFeatures.distance)<=1
      and firstFeatures.grounded==secondFeatures.grounded
      and math.abs(firstFeatures.speed-secondFeatures.speed)<=1
      and firstFeatures.power==secondFeatures.power
  elseif firstFeatures.kind=="gap" then
    return math.abs(firstFeatures.width-secondFeatures.width)<=1
      and math.abs(firstFeatures.distance-secondFeatures.distance)<=1
      and firstFeatures.grounded==secondFeatures.grounded
      and math.abs(firstFeatures.speed-secondFeatures.speed)<=1
      and firstFeatures.power==secondFeatures.power
  elseif firstFeatures.kind=="obstacle" then
    return math.abs(firstFeatures.height-secondFeatures.height)<=1
      and math.abs(firstFeatures.distance-secondFeatures.distance)<=1
      and firstFeatures.grounded==secondFeatures.grounded
      and math.abs(firstFeatures.speed-secondFeatures.speed)<=1
      and firstFeatures.power==secondFeatures.power
  end
  return firstFeatures.grounded==secondFeatures.grounded
    and math.abs(firstFeatures.speed-secondFeatures.speed)<=1
    and firstFeatures.power==secondFeatures.power
end

-- Get Q-values for a context, including neighbors
function qmemory.lookup(contextKey, experienceMemory)
  local memory=experienceMemory or {}
  local contextMemory=memory[contextKey] or {}
  local qvalues={}
  for actionChoice=1,24 do
    qvalues[actionChoice]={reward=0,count=0}
  end
  for choice=1,24 do
    if contextMemory[choice] then
      qvalues[choice].reward=contextMemory[choice].reward or 0
      qvalues[choice].count=contextMemory[choice].count or 0
    end
  end
  local neighborCount=0
  for otherContext in pairs(memory) do
    if otherContext~=contextKey and isContextNeighbor(contextKey,otherContext) then
      neighborCount=neighborCount+1
      if neighborCount<=EXPERIENCE_TRACE_LIMIT then
        local neighborMemory=memory[otherContext]
        for choice=1,24 do
          if neighborMemory[choice] then
            qvalues[choice].count=qvalues[choice].count+(neighborMemory[choice].count or 0)*0.5
            qvalues[choice].reward=qvalues[choice].reward+(neighborMemory[choice].reward or 0)*0.5
          end
        end
      end
    end
  end
  return qvalues
end

-- Update experience memory with episode result
function qmemory.updateEpisode(contextKey, choice, reward, experienceMemory)
  if not contextKey or not choice or not reward then return end
  local memory=experienceMemory or {}
  memory[contextKey]=memory[contextKey] or {}
  if not memory[contextKey][choice] then
    memory[contextKey][choice]={reward=reward,count=1}
  else
    local entry=memory[contextKey][choice]
    entry.reward=(entry.reward*(entry.count or 1)+reward)/(entry.count+1)
    entry.count=(entry.count or 1)+1
  end
  local memorySize=0
  for _ in pairs(memory) do memorySize=memorySize+1 end
  if memorySize>EXPERIENCE_CONTEXT_LIMIT then
    local worstContext,worstScore
    for ctx in pairs(memory) do
      local score=0
      for _ in pairs(memory[ctx]) do score=score+1 end
      if not worstScore or score<worstScore then
        worstContext,worstScore=ctx,score
      end
    end
    if worstContext then memory[worstContext]=nil end
  end
  return memory
end

-- Track behavioral novelty
function qmemory.recordBehavior(genome, behavior, behaviorArchive)
  behaviorArchive=behaviorArchive or {}
  local key=tostring(genome)..":"..table.concat(behavior or {},":")
  if not behaviorArchive[key] then
    behaviorArchive[key]=true
    local archiveSize=0
    for _ in pairs(behaviorArchive) do archiveSize=archiveSize+1 end
    if archiveSize>NOVELTY_ARCHIVE_LIMIT then
      local toRemove
      for k in pairs(behaviorArchive) do
        toRemove=k
        break
      end
      if toRemove then behaviorArchive[toRemove]=nil end
    end
    return true
  end
  return false
end

return qmemory
