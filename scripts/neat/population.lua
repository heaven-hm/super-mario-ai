local function install(AI,C)
local ACTION_COUNT=C.ACTION_COUNT
local GRID_INPUT_COUNT=C.GRID_INPUT_COUNT
local NEURAL_INPUT_COUNT=C.NEURAL_INPUT_COUNT
local OUTPUT_NODE_OFFSET=C.OUTPUT_NODE_OFFSET
local DEFAULT_POPULATION_SIZE=C.DEFAULT_POPULATION_SIZE
local createEmptyGenome=AI.createEmptyGenome
local createGene=AI.createGene
local getInnovationNumber=AI.getInnovationNumber
local cloneGenome=AI.cloneGenome
function AI.newGenome(populationState)
  local genome=createEmptyGenome()
  -- Sparse initial networks keep the first generation diverse and inexpensive.
  for actionIndex = 1, ACTION_COUNT do
    local sourceNode = math.random(NEURAL_INPUT_COUNT)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,
      (math.random()*2-1)*0.5,getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  return genome
end

local function createSeededGenome(populationState)
  local genome=createEmptyGenome()
  local enemyHorizontalOffsetInput = GRID_INPUT_COUNT+6
  local itemHorizontalOffsetInput = GRID_INPUT_COUNT+11
  local gapAheadInput = GRID_INPUT_COUNT+14
  local enemyContactInput = GRID_INPUT_COUNT+15
  local function connect(sourceNode, actionIndex, weight)
    local targetNode = OUTPUT_NODE_OFFSET + actionIndex
    table.insert(genome.genes,createGene(sourceNode,targetNode,weight,
      getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  -- Start from sensible SMB1 play: run on clear ground, jump for an enemy or
  -- pit, and turn back toward a visible reward. Evolution can change all links.
  connect(NEURAL_INPUT_COUNT,1,0.55)
  connect(NEURAL_INPUT_COUNT,2,-0.28)
  connect(enemyHorizontalOffsetInput,2,3.4)
  connect(gapAheadInput,2,3.0)
  connect(itemHorizontalOffsetInput,3,-1.2)
  connect(enemyHorizontalOffsetInput,3,-2.0)
  connect(NEURAL_INPUT_COUNT,5,-0.2)
  connect(enemyContactInput,5,1.5)
  return genome
end

function AI.newPopulation(populationSize)
  local populationState = {generation=1,nextInnovation=ACTION_COUNT,innovations={},genomes={},species={},
    bestFitness=0,population=populationSize or DEFAULT_POPULATION_SIZE,nextGenomeIndex=1,
    nextHiddenNode=NEURAL_INPUT_COUNT,splitHistory={},behaviorArchive={},
    episodeHistory={},topPerformers={},experienceMemory={},legacyLogImported=false}
  for genomeIndex = 1, populationState.population do
    local genome
    if genomeIndex == 1 then
      genome = createSeededGenome(populationState)
    else
      genome = cloneGenome(populationState.genomes[1])
      AI.mutate(genome,populationState)
    end
    populationState.genomes[#populationState.genomes+1]=genome
  end
  AI.assignSpecies(populationState)
  return populationState
end
end

return install
