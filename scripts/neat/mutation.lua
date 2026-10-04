-- NEAT mutation operators: add connection, add node, weight mutation

local mutation = {}

local genome_module = require("scripts.neat.genome")

local function isMemoryInput(nodeId)
  return nodeId >= genome_module.MEMORY_INPUT_OFFSET and nodeId < genome_module.MEMORY_INPUT_OFFSET + genome_module.MEMORY_INPUT_COUNT
end

local function isHiddenNode(nodeId)
  return nodeId > genome_module.NEURAL_INPUT_COUNT and nodeId < genome_module.MEMORY_INPUT_OFFSET
end

-- Helper to get unique key for a connection
local function connectionKey(sourceNode, targetNode)
  return tostring(sourceNode)..":"..tostring(targetNode)
end

-- Get or assign innovation number for a connection
local function getInnovationNumber(populationState, sourceNode, targetNode)
  local connectionIdentifier = connectionKey(sourceNode,targetNode)
  if not populationState.innovations[connectionIdentifier] then
    populationState.nextInnovation = populationState.nextInnovation + 1
    populationState.innovations[connectionIdentifier] = populationState.nextInnovation
  end
  return populationState.innovations[connectionIdentifier]
end

-- Create a gene (connection)
local function createGene(sourceNode, targetNode, weight, innovationNumber)
  return {sourceNode=sourceNode,targetNode=targetNode,weight=weight,
    enabled=true,innovation=innovationNumber}
end

-- Check if genome already has a connection
local function hasConnection(gen, sourceNode, targetNode)
  for _, gene in ipairs(gen.genes) do
    if gene.sourceNode==sourceNode and gene.targetNode==targetNode then return true end
  end
  return false
end

-- Choose a random node from genome to connect to
local function chooseRandomNode(gen, populationState, inputOnly)
  -- Give compact features a better chance of receiving connections
  if not inputOnly and math.random()<0.4 then
    return math.random(genome_module.GRID_INPUT_COUNT+1,genome_module.NEURAL_INPUT_COUNT)
  end
  local candidates = {}
  for inputIndex = 1, genome_module.NEURAL_INPUT_COUNT do
    candidates[#candidates+1] = inputIndex
  end
  for memoryIndex = 1, genome_module.MEMORY_INPUT_COUNT do
    candidates[#candidates+1] = genome_module.memoryInputNode(memoryIndex)
  end
  if not inputOnly then
    for _, gene in ipairs(gen.genes) do
      if isHiddenNode(gene.sourceNode) then candidates[#candidates+1]=gene.sourceNode end
      if isHiddenNode(gene.targetNode) then candidates[#candidates+1]=gene.targetNode end
    end
  end
  return candidates[math.random(#candidates)]
end

-- Add a connection to the genome
local function mutateAddConnection(gen, populationState, biasOnly)
  local sourceNode = biasOnly and genome_module.NEURAL_INPUT_COUNT or chooseRandomNode(gen,populationState,false)
  local targetNode = genome_module.OUTPUT_NODE_OFFSET + math.random(genome_module.NETWORK_OUTPUT_COUNT)
  if not biasOnly then
    local hiddenNodes = {}
    for _, gene in ipairs(gen.genes) do
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
    or hasConnection(gen,sourceNode,targetNode) then return end
  table.insert(gen.genes,createGene(sourceNode,targetNode,math.random()*4-2,
    getInnovationNumber(populationState,sourceNode,targetNode)))
end

-- Add a hidden node by splitting an existing connection
local function mutateAddNode(gen, populationState)
  local enabledGenes = {}
  for _, gene in ipairs(gen.genes) do
    local existingNode=populationState.splitHistory[gene.innovation]
    if gene.enabled and (not existingNode
      or not hasConnection(gen,gene.sourceNode,existingNode)) then
      enabledGenes[#enabledGenes+1] = gene
    end
  end
  if #enabledGenes == 0 then return end
  local splitGene = enabledGenes[math.random(#enabledGenes)]
  local newHiddenNode=populationState.splitHistory[splitGene.innovation]
  if not newHiddenNode then
    newHiddenNode=(populationState.nextHiddenNode or genome_module.NEURAL_INPUT_COUNT)+1
    if newHiddenNode>=genome_module.OUTPUT_NODE_OFFSET then return end
    populationState.nextHiddenNode=newHiddenNode
    populationState.splitHistory[splitGene.innovation]=newHiddenNode
  end
  splitGene.enabled=false
  gen.highestHiddenNode=math.max(gen.highestHiddenNode,newHiddenNode)
  table.insert(gen.genes,createGene(splitGene.sourceNode,newHiddenNode,1,
    getInnovationNumber(populationState,splitGene.sourceNode,newHiddenNode)))
  table.insert(gen.genes,createGene(newHiddenNode,splitGene.targetNode,splitGene.weight,
    getInnovationNumber(populationState,newHiddenNode,splitGene.targetNode)))
end

-- Repeat mutation by probabilistic rate
local function repeatMutationByRate(rate, callback)
  while rate > 0 do
    if math.random() < math.min(1,rate) then callback() end
    rate = rate - 1
  end
end

-- Mutate genome: adjust weights, add connections, add nodes, enable/disable genes
function mutation.mutate(gen, populationState)
  for mutationName, mutationRate in pairs(gen.mutationRates) do
    if mutationName ~= "step" then
      gen.mutationRates[mutationName] = mutationRate
        * (math.random(2) == 1 and 0.95 or 1.05263)
    end
  end
  if math.random() < gen.mutationRates.connections then
    for _, gene in ipairs(gen.genes) do
      if math.random() < 0.9 then
        gene.weight = gene.weight + (math.random()*2-1)*gen.mutationRates.step
      else
        gene.weight = math.random()*4-2
      end
    end
  end
  repeatMutationByRate(gen.mutationRates.link,function() mutateAddConnection(gen,populationState,false) end)
  repeatMutationByRate(gen.mutationRates.bias,function() mutateAddConnection(gen,populationState,true) end)
  repeatMutationByRate(gen.mutationRates.node,function() mutateAddNode(gen,populationState) end)
  repeatMutationByRate(gen.mutationRates.enable,function()
    local disabledGenes = {}
    for _, gene in ipairs(gen.genes) do
      if not gene.enabled then disabledGenes[#disabledGenes+1] = gene end
    end
    if #disabledGenes > 0 then disabledGenes[math.random(#disabledGenes)].enabled=true end
  end)
  repeatMutationByRate(gen.mutationRates.disable,function()
    local enabledGenes = {}
    for _, gene in ipairs(gen.genes) do
      if gene.enabled then enabledGenes[#enabledGenes+1] = gene end
    end
    if #enabledGenes > 0 then enabledGenes[math.random(#enabledGenes)].enabled=false end
  end)
end

return mutation
