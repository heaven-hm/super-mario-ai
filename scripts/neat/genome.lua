-- NEAT genome representation: structure, initialization, cloning, and evaluation

local genome = {}

-- Neural network architecture constants
genome.SENSOR_RADIUS_TILES = 6
genome.GRID_WIDTH = genome.SENSOR_RADIUS_TILES * 2 + 1
genome.GRID_INPUT_COUNT = genome.GRID_WIDTH * genome.GRID_WIDTH
genome.GLOBAL_INPUT_COUNT = 15
genome.OBSERVATION_INPUT_COUNT = genome.GRID_INPUT_COUNT + genome.GLOBAL_INPUT_COUNT
genome.NEURAL_INPUT_COUNT = genome.OBSERVATION_INPUT_COUNT + 1 -- bias node
genome.ACTION_COUNT = 6
genome.DURATION_OUTPUT_COUNT = 4
genome.NETWORK_OUTPUT_COUNT = genome.ACTION_COUNT + genome.DURATION_OUTPUT_COUNT
genome.OUTPUT_NODE_OFFSET = 1000000
genome.MEMORY_INPUT_OFFSET = 900000
genome.MEMORY_FEATURE_COUNT = genome.GLOBAL_INPUT_COUNT
genome.MEMORY_LAGS = {1, 4}
genome.MEMORY_INPUT_COUNT = genome.MEMORY_FEATURE_COUNT * #genome.MEMORY_LAGS

local function clamp(value, low, high)
  return math.max(low, math.min(high, value))
end

local function sigmoid(value)
  value = clamp(value,-60,60)
  return 2/(1+math.exp(-4.9*value))-1
end

local function memoryNode(index)
  return genome.MEMORY_INPUT_OFFSET + index
end

local function isMemoryInput(nodeId)
  return nodeId >= genome.MEMORY_INPUT_OFFSET and nodeId < genome.MEMORY_INPUT_OFFSET + genome.MEMORY_INPUT_COUNT
end

local function isHiddenNode(nodeId)
  return nodeId > genome.NEURAL_INPUT_COUNT and nodeId < genome.MEMORY_INPUT_OFFSET
end

function genome.inputCount()
  return genome.NEURAL_INPUT_COUNT
end

function genome.outputNode(index)
  return genome.OUTPUT_NODE_OFFSET + index
end

function genome.memoryInputNode(index)
  return memoryNode(index)
end

function genome.sensorIndex(horizontalOffset, verticalOffset)
  local columnIndex = math.floor((horizontalOffset + genome.SENSOR_RADIUS_TILES * 16) / 16)
  local rowIndex = math.floor((verticalOffset + genome.SENSOR_RADIUS_TILES * 16) / 16)
  return rowIndex * genome.GRID_WIDTH + columnIndex + 1
end

-- Create a new empty genome
function genome.create()
  return {genes={},fitness=0,adjustedFitness=0,highestHiddenNode=genome.NEURAL_INPUT_COUNT,
    mutationRates={connections=0.8,link=1.0,bias=0.4,node=0.12,enable=0.2,disable=0.2,step=0.1}}
end

-- Clone a genome
function genome.clone(original)
  local genomeCopy={genes={},fitness=original.fitness or 0,adjustedFitness=0,
    highestHiddenNode=original.highestHiddenNode,mutationRates={}}
  for mutationName, mutationRate in pairs(original.mutationRates) do
    genomeCopy.mutationRates[mutationName] = mutationRate
  end
  for _, gene in ipairs(original.genes) do
    genomeCopy.genes[#genomeCopy.genes+1]={sourceNode=gene.sourceNode,targetNode=gene.targetNode,weight=gene.weight,
      enabled=gene.enabled,innovation=gene.innovation}
  end
  return genomeCopy
end

-- Evaluate genome on inputs and return action/duration scores
function genome.evaluate(gen, inputValues, memoryValues)
  local nodeValues, incomingConnections = {}, {}
  for inputIndex = 1, genome.OBSERVATION_INPUT_COUNT do
    nodeValues[inputIndex] = inputValues[inputIndex] or 0
  end
  nodeValues[genome.NEURAL_INPUT_COUNT] = 1
  for memoryIndex = 1, genome.MEMORY_INPUT_COUNT do
    nodeValues[memoryNode(memoryIndex)] = memoryValues and memoryValues[memoryIndex] or 0
  end
  for _, gene in ipairs(gen.genes) do
    if gene.enabled then
      incomingConnections[gene.targetNode] = incomingConnections[gene.targetNode] or {}
      incomingConnections[gene.targetNode][#incomingConnections[gene.targetNode]+1] = gene
    end
  end
  local evaluating={}
  local function evaluateNode(nodeId)
    if nodeValues[nodeId]~=nil then return nodeValues[nodeId] end
    if nodeId<=genome.NEURAL_INPUT_COUNT or evaluating[nodeId] then return 0 end
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
  for actionIndex = 1, genome.ACTION_COUNT do
    actionScores[actionIndex] = evaluateNode(genome.OUTPUT_NODE_OFFSET+actionIndex)
  end
  for durationIndex = 1, genome.DURATION_OUTPUT_COUNT do
    durationScores[durationIndex] = evaluateNode(genome.OUTPUT_NODE_OFFSET+genome.ACTION_COUNT+durationIndex)
  end
  return actionScores,nodeValues,durationScores
end

-- Get signature of active policy (for novelty/diversity tracking)
function genome.signature(gen)
  local activeGenes={}
  for _,gene in ipairs(gen.genes or {}) do
    if gene.enabled then
      activeGenes[#activeGenes+1]=table.concat({gene.sourceNode,gene.targetNode,
        string.format("%.4f",gene.weight)},":")
    end
  end
  table.sort(activeGenes)
  return table.concat(activeGenes,"|")
end

return genome
