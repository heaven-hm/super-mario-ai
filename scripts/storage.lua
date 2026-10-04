local function install(AI,C)
local ACTION_COUNT=C.ACTION_COUNT
local DURATION_OUTPUT_COUNT=C.DURATION_OUTPUT_COUNT
local EPISODE_HISTORY_LIMIT=C.EPISODE_HISTORY_LIMIT
local EXPERIENCE_CONTEXT_LIMIT=C.EXPERIENCE_CONTEXT_LIMIT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local NOVELTY_ARCHIVE_LIMIT=C.NOVELTY_ARCHIVE_LIMIT
local OUTPUT_NODE_OFFSET=C.OUTPUT_NODE_OFFSET
local TOP_PERFORMER_LIMIT=C.TOP_PERFORMER_LIMIT
local SET_TIMER_TO_999_PER_EPISODE=C.SET_TIMER_TO_999_PER_EPISODE
local TESTING_INFINITE_LIVES=C.TESTING_INFINITE_LIVES
local clamp=C.clamp
local createEmptyGenome=AI.createEmptyGenome
local isHiddenNode=AI.isHiddenNode
local connectionKey=AI.connectionKey
local assignSpecies=AI.assignSpecies
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

local function getDatabasePath()
  return C.PROJECT_DIR.."mario_ai_neat.db"
end

local function getLogPath()
  return C.PROJECT_DIR.."mario_ai_neat.log"
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
AI.getDatabasePath=getDatabasePath
AI.getLogPath=getLogPath
AI.currentTimestamp=currentTimestamp
end

return install
