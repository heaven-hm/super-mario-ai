local function install(AI,C)
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local OUTPUT_NODE_OFFSET=C.OUTPUT_NODE_OFFSET
local NETWORK_OUTPUT_COUNT=C.NETWORK_OUTPUT_COUNT
local MEMORY_INPUT_COUNT=C.MEMORY_INPUT_COUNT
local clamp=C.clamp
local memoryNode=AI.memoryNode
local isMemoryInput=AI.isMemoryInput
local isHiddenNode=AI.isHiddenNode
local function connectionKey(sourceNode, targetNode)
  return tostring(sourceNode)..":"..tostring(targetNode)
end

-- Reuse one historical number for the same structural connection. This is the
-- NEAT mechanism that aligns homologous genes during crossover.
local function getInnovationNumber(populationState, sourceNode, targetNode)
  local connectionIdentifier = connectionKey(sourceNode,targetNode)
  if not populationState.innovations[connectionIdentifier] then
    populationState.nextInnovation = populationState.nextInnovation + 1
    populationState.innovations[connectionIdentifier] = populationState.nextInnovation
  end
  return populationState.innovations[connectionIdentifier]
end

local function createGene(sourceNode, targetNode, weight, innovationNumber)
  return {sourceNode=sourceNode,targetNode=targetNode,weight=weight,
    enabled=true,innovation=innovationNumber}
end

local function hasConnection(genome, sourceNode, targetNode)
  for _, gene in ipairs(genome.genes) do
    if gene.sourceNode==sourceNode and gene.targetNode==targetNode then return true end
  end
  return false
end

local function chooseRandomNode(genome, populationState, inputOnly)
  -- Most grid cells are empty in a given frame. Give compact movement, enemy,
  -- item, and gap features a better chance of receiving a new connection.
  if not inputOnly and math.random()<0.4 then
    return math.random(GRID_INPUT_COUNT+1,NEURAL_INPUT_COUNT)
  end
  local candidates = {}
  for inputIndex = 1, NEURAL_INPUT_COUNT do
    candidates[#candidates+1] = inputIndex
  end
  for memoryIndex = 1, MEMORY_INPUT_COUNT do
    candidates[#candidates+1] = memoryNode(memoryIndex)
  end
  if not inputOnly then
    for _, gene in ipairs(genome.genes) do
      if isHiddenNode(gene.sourceNode) then candidates[#candidates+1]=gene.sourceNode end
      if isHiddenNode(gene.targetNode) then candidates[#candidates+1]=gene.targetNode end
    end
  end
  return candidates[math.random(#candidates)]
end

local function mutateAddConnection(genome, populationState, biasOnly)
  local sourceNode = biasOnly and NEURAL_INPUT_COUNT or chooseRandomNode(genome,populationState,false)
  local targetNode = OUTPUT_NODE_OFFSET + math.random(NETWORK_OUTPUT_COUNT)
  if not biasOnly then
    local hiddenNodes = {}
    for _, gene in ipairs(genome.genes) do
      if isHiddenNode(gene.sourceNode) then
        hiddenNodes[#hiddenNodes+1] = gene.sourceNode
      end
      if isHiddenNode(gene.targetNode) then
        hiddenNodes[#hiddenNodes+1] = gene.targetNode
      end
    end
    if #hiddenNodes > 0 and math.random(2) == 1 then
      targetNode = hiddenNodes[math.random(#hiddenNodes)]
    end
  end
  if (sourceNode >= targetNode and not isMemoryInput(sourceNode))
    or hasConnection(genome,sourceNode,targetNode) then return end
  table.insert(genome.genes,createGene(sourceNode,targetNode,math.random()*4-2,
    getInnovationNumber(populationState,sourceNode,targetNode)))
end

local function mutateAddNode(genome, populationState)
  local enabledGenes = {}
  for _, gene in ipairs(genome.genes) do
    local existingNode=populationState.splitHistory[gene.innovation]
    if gene.enabled and (not existingNode
      or not hasConnection(genome,gene.sourceNode,existingNode)) then
      enabledGenes[#enabledGenes+1] = gene
    end
  end
  if #enabledGenes == 0 then return end
  local splitGene = enabledGenes[math.random(#enabledGenes)]
  local newHiddenNode=populationState.splitHistory[splitGene.innovation]
  if not newHiddenNode then
    newHiddenNode=(populationState.nextHiddenNode or NEURAL_INPUT_COUNT)+1
    if newHiddenNode>=OUTPUT_NODE_OFFSET then return end
    populationState.nextHiddenNode=newHiddenNode
    populationState.splitHistory[splitGene.innovation]=newHiddenNode
  end
  splitGene.enabled=false
  genome.highestHiddenNode=math.max(genome.highestHiddenNode,newHiddenNode)
  table.insert(genome.genes,createGene(splitGene.sourceNode,newHiddenNode,1,
    getInnovationNumber(populationState,splitGene.sourceNode,newHiddenNode)))
  table.insert(genome.genes,createGene(newHiddenNode,splitGene.targetNode,splitGene.weight,
    getInnovationNumber(populationState,newHiddenNode,splitGene.targetNode)))
end

local function repeatMutationByRate(rate, callback)
  while rate > 0 do
    if math.random() < math.min(1,rate) then callback() end
    rate = rate - 1
  end
end

function AI.mutate(genome, populationState)
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    if mutationName ~= "step" then
      genome.mutationRates[mutationName] = mutationRate
        * (math.random(2) == 1 and 0.95 or 1.05263)
    end
  end
  if math.random() < genome.mutationRates.connections then
    for _, gene in ipairs(genome.genes) do
      if math.random() < 0.9 then
        gene.weight = gene.weight + (math.random()*2-1)*genome.mutationRates.step
      else
        gene.weight = math.random()*4-2
      end
    end
  end
  repeatMutationByRate(genome.mutationRates.link,function() mutateAddConnection(genome,populationState,false) end)
  repeatMutationByRate(genome.mutationRates.bias,function() mutateAddConnection(genome,populationState,true) end)
  repeatMutationByRate(genome.mutationRates.node,function() mutateAddNode(genome,populationState) end)
  repeatMutationByRate(genome.mutationRates.enable,function()
    local disabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if not gene.enabled then disabledGenes[#disabledGenes+1] = gene end
    end
    if #disabledGenes > 0 then disabledGenes[math.random(#disabledGenes)].enabled=true end
  end)
  repeatMutationByRate(genome.mutationRates.disable,function()
    local enabledGenes = {}
    for _, gene in ipairs(genome.genes) do
      if gene.enabled then enabledGenes[#enabledGenes+1] = gene end
    end
    if #enabledGenes > 0 then enabledGenes[math.random(#enabledGenes)].enabled=false end
  end)
end
AI.connectionKey=connectionKey
AI.getInnovationNumber=getInnovationNumber
AI.createGene=createGene
end

return install
