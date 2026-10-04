local function install(AI,C)
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local GLOBAL_INPUT_COUNT=C.GLOBAL_INPUT_COUNT
local OBSERVATION_INPUT_COUNT=C.OBSERVATION_INPUT_COUNT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local ACTION_COUNT=C.ACTION_COUNT
local DURATION_OUTPUT_COUNT=C.DURATION_OUTPUT_COUNT
local NETWORK_OUTPUT_COUNT=C.NETWORK_OUTPUT_COUNT
local OUTPUT_NODE_OFFSET=C.OUTPUT_NODE_OFFSET
local MEMORY_INPUT_OFFSET=C.MEMORY_INPUT_OFFSET
local MEMORY_INPUT_COUNT=C.MEMORY_INPUT_COUNT
local clamp=C.clamp
local function memoryNode(index) return MEMORY_INPUT_OFFSET + index end
local function isMemoryInput(nodeId)
  return nodeId >= MEMORY_INPUT_OFFSET and nodeId < MEMORY_INPUT_OFFSET + MEMORY_INPUT_COUNT
end
local function isHiddenNode(nodeId)
  return nodeId > NEURAL_INPUT_COUNT and nodeId < MEMORY_INPUT_OFFSET
end

function AI.inputCount() return NEURAL_INPUT_COUNT end
function AI.outputNode(index) return OUTPUT_NODE_OFFSET + index end
function AI.memoryInputNode(index) return memoryNode(index) end

local function sigmoid(value)
  value = clamp(value,-60,60)
  return 2/(1+math.exp(-4.9*value))-1
end

-- Follow enabled connections when evaluating each output. A split can add a
-- newer hidden node before an older hidden target, so numeric order is unsafe.
function AI.evaluateGenome(genome, inputValues, memoryValues)
  local nodeValues, incomingConnections = {}, {}
  for inputIndex = 1, OBSERVATION_INPUT_COUNT do
    nodeValues[inputIndex] = inputValues[inputIndex] or 0
  end
  nodeValues[NEURAL_INPUT_COUNT] = 1
  for memoryIndex = 1, MEMORY_INPUT_COUNT do
    nodeValues[memoryNode(memoryIndex)] = memoryValues and memoryValues[memoryIndex] or 0
  end
  for _, gene in ipairs(genome.genes) do
    if gene.enabled then
      incomingConnections[gene.targetNode] = incomingConnections[gene.targetNode] or {}
      incomingConnections[gene.targetNode][#incomingConnections[gene.targetNode]+1] = gene
    end
  end
  local evaluating={}
  local function evaluateNode(nodeId)
    if nodeValues[nodeId]~=nil then return nodeValues[nodeId] end
    if nodeId<=NEURAL_INPUT_COUNT or evaluating[nodeId] then return 0 end
    evaluating[nodeId]=true
    local nodeConnections = incomingConnections[nodeId]
    if nodeConnections then
      local weightedSum = 0
      for _, gene in ipairs(nodeConnections) do
        weightedSum = weightedSum + evaluateNode(gene.sourceNode)*gene.weight
      end
      nodeValues[nodeId] = sigmoid(weightedSum)
    else
      nodeValues[nodeId]=0
    end
    evaluating[nodeId]=nil
    return nodeValues[nodeId]
  end
  local actionScores, durationScores = {}, {}
  for actionIndex = 1, ACTION_COUNT do
    actionScores[actionIndex] = evaluateNode(OUTPUT_NODE_OFFSET+actionIndex)
  end
  for durationIndex = 1, DURATION_OUTPUT_COUNT do
    durationScores[durationIndex] = evaluateNode(OUTPUT_NODE_OFFSET+ACTION_COUNT+durationIndex)
  end
  return actionScores,nodeValues,durationScores
end

local function createEmptyGenome()
  return {genes={},fitness=0,adjustedFitness=0,highestHiddenNode=NEURAL_INPUT_COUNT,
    mutationRates={connections=0.8,link=1.0,bias=0.4,node=0.12,enable=0.2,disable=0.2,step=0.1}}
end

local function cloneGenome(genome)
  local genomeCopy={genes={},fitness=genome.fitness or 0,adjustedFitness=0,
    highestHiddenNode=genome.highestHiddenNode,mutationRates={}}
  for mutationName, mutationRate in pairs(genome.mutationRates) do
    genomeCopy.mutationRates[mutationName] = mutationRate
  end
  for _, gene in ipairs(genome.genes) do
    genomeCopy.genes[#genomeCopy.genes+1]={sourceNode=gene.sourceNode,targetNode=gene.targetNode,weight=gene.weight,
      enabled=gene.enabled,innovation=gene.innovation}
  end
  return genomeCopy
end

-- The signature describes the active policy. Disabled genes do not change
-- current behavior, and rounding avoids treating serialization noise as novelty.
local function genomeSignature(genome)
  local activeGenes={}
  for _,gene in ipairs(genome.genes or {}) do
    if gene.enabled then
      activeGenes[#activeGenes+1]=table.concat({gene.sourceNode,gene.targetNode,
        string.format("%.4f",gene.weight)},":")
    end
  end
  table.sort(activeGenes)
  return table.concat(activeGenes,"|")
end

function AI.policySignature(genome)
  return genomeSignature(genome)
end
AI.memoryNode=memoryNode
AI.isMemoryInput=isMemoryInput
AI.isHiddenNode=isHiddenNode
AI.createEmptyGenome=createEmptyGenome
AI.cloneGenome=cloneGenome
AI.genomeSignature=genomeSignature
end

return install
