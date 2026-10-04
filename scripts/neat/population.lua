-- NEAT population initialization and management

local population = {}

local genome = require("scripts.neat.genome")
local mutation = require("scripts.neat.mutation")
local species = require("scripts.neat.species")

local DEFAULT_POPULATION_SIZE = 300

-- Helper to get innovation number
local function getInnovationNumber(populationState, sourceNode, targetNode)
  local connectionIdentifier = tostring(sourceNode)..":"..tostring(targetNode)
  if not populationState.innovations[connectionIdentifier] then
    populationState.nextInnovation = populationState.nextInnovation + 1
    populationState.innovations[connectionIdentifier] = populationState.nextInnovation
  end
  return populationState.innovations[connectionIdentifier]
end

-- Create a gene
local function createGene(sourceNode, targetNode, weight, innovationNumber)
  return {sourceNode=sourceNode,targetNode=targetNode,weight=weight,
    enabled=true,innovation=innovationNumber}
end

-- Create a seeded genome with sensible initial connections
local function createSeededGenome(populationState)
  local gen=genome.create()
  local enemyHorizontalOffsetInput = genome.GRID_INPUT_COUNT+6
  local itemHorizontalOffsetInput = genome.GRID_INPUT_COUNT+11
  local gapAheadInput = genome.GRID_INPUT_COUNT+14
  local enemyContactInput = genome.GRID_INPUT_COUNT+15
  local function connect(sourceNode, actionIndex, weight)
    local targetNode = genome.OUTPUT_NODE_OFFSET + actionIndex
    table.insert(gen.genes,createGene(sourceNode,targetNode,weight,
      getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  -- Sensible initial policy: run on clear ground, jump for enemy/pit, retreat for reward
  connect(genome.NEURAL_INPUT_COUNT,1,0.55)
  connect(genome.NEURAL_INPUT_COUNT,2,-0.28)
  connect(enemyHorizontalOffsetInput,2,3.4)
  connect(gapAheadInput,2,3.0)
  connect(itemHorizontalOffsetInput,3,-1.2)
  connect(enemyHorizontalOffsetInput,3,-2.0)
  connect(genome.NEURAL_INPUT_COUNT,5,-0.2)
  connect(enemyContactInput,5,1.5)
  return gen
end

-- Create a new random genome with sparse initial connections
function population.newGenome(populationState)
  local gen=genome.create()
  for actionIndex = 1, genome.ACTION_COUNT do
    local sourceNode = math.random(genome.NEURAL_INPUT_COUNT)
    local targetNode = genome.OUTPUT_NODE_OFFSET + actionIndex
    table.insert(gen.genes,createGene(sourceNode,targetNode,
      (math.random()*2-1)*0.5,getInnovationNumber(populationState,sourceNode,targetNode)))
  end
  return gen
end

-- Create a new population
function population.create(populationSize)
  local populationState = {generation=1,nextInnovation=genome.ACTION_COUNT,innovations={},genomes={},species={},
    bestFitness=0,population=populationSize or DEFAULT_POPULATION_SIZE,nextGenomeIndex=1,
    nextHiddenNode=genome.NEURAL_INPUT_COUNT,splitHistory={},behaviorArchive={},
    episodeHistory={},topPerformers={},experienceMemory={},legacyLogImported=false}
  for genomeIndex = 1, populationState.population do
    local gen
    if genomeIndex == 1 then
      gen = createSeededGenome(populationState)
    else
      gen = genome.clone(populationState.genomes[1])
      mutation.mutate(gen,populationState)
    end
    populationState.genomes[#populationState.genomes+1]=gen
  end
  species.assign(populationState)
  return populationState
end

return population
