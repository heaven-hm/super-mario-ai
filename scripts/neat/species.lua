-- NEAT speciation, reproduction, and generational evolution

local species = {}

local genome_module = require("scripts.neat.genome")
local mutation_module = require("scripts.neat.mutation")

local SPECIES_DISTANCE_THRESHOLD = 1.0
local MAX_STALE_GENERATIONS = 15

-- Calculate genetic distance between two genomes
local function calculateGenomeDistance(firstGenome, secondGenome)
  local secondGenesByInnovation = {}
  for _, gene in ipairs(secondGenome.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  local matchingGeneCount, totalWeightDifference, unmatchedGeneCount = 0, 0, 0
  for _, gene in ipairs(firstGenome.genes) do
    local matchingGene = secondGenesByInnovation[gene.innovation]
    if matchingGene then
      matchingGeneCount = matchingGeneCount+1
      totalWeightDifference = totalWeightDifference+math.abs(gene.weight-matchingGene.weight)
    else
      unmatchedGeneCount = unmatchedGeneCount+1
    end
  end
  unmatchedGeneCount = unmatchedGeneCount
    + math.max(0,#secondGenome.genes-matchingGeneCount)
  local genomeSize = math.max(1,#firstGenome.genes,#secondGenome.genes)
  local averageWeightDifference = matchingGeneCount > 0
    and totalWeightDifference/matchingGeneCount or 0
  return 2*unmatchedGeneCount/genomeSize + 0.4*averageWeightDifference
end

-- Assign each genome to a species based on genetic distance
function species.assign(populationState)
  local previousSpecies = populationState.species or {}
  local currentSpecies = {}
  for _, gen in ipairs(populationState.genomes) do
    local matchingSpecies
    for _, speciesGroup in ipairs(currentSpecies) do
      if calculateGenomeDistance(gen,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
        matchingSpecies=speciesGroup
        break
      end
    end
    if not matchingSpecies then
      local previousMatch
      for _, speciesGroup in ipairs(previousSpecies) do
        if calculateGenomeDistance(gen,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
          previousMatch=speciesGroup
          break
        end
      end
      matchingSpecies={id=previousMatch and previousMatch.id or (#currentSpecies+1),genomes={},
        topFitness=previousMatch and previousMatch.topFitness or 0,
        staleness=previousMatch and previousMatch.staleness or 0,
        representative=genome_module.clone(gen)}
      currentSpecies[#currentSpecies+1]=matchingSpecies
    end
    matchingSpecies.genomes[#matchingSpecies.genomes+1]=gen
    gen.species=matchingSpecies.id
  end
  populationState.species=currentSpecies
end

-- Rank genomes and species by fitness
local function rankSpecies(populationState)
  table.sort(populationState.genomes,function(firstGenome,secondGenome)
    return firstGenome.fitness > secondGenome.fitness
  end)
  for rank, gen in ipairs(populationState.genomes) do
    gen.globalRank = #populationState.genomes-rank+1
  end
  local champion = populationState.genomes[1]
  populationState.bestFitness = math.max(populationState.bestFitness or 0,
    champion and champion.fitness or 0)
  for _, speciesGroup in ipairs(populationState.species) do
    table.sort(speciesGroup.genomes,function(firstGenome,secondGenome)
      return firstGenome.fitness > secondGenome.fitness
    end)
    local topFitness = speciesGroup.genomes[1] and speciesGroup.genomes[1].fitness or 0
    if topFitness > speciesGroup.topFitness then
      speciesGroup.topFitness = topFitness
      speciesGroup.staleness = 0
    else
      speciesGroup.staleness = (speciesGroup.staleness or 0)+1
    end
    local adjustedFitnessTotal = 0
    for _, gen in ipairs(speciesGroup.genomes) do
      gen.adjustedFitness = gen.globalRank/math.max(1,#speciesGroup.genomes)
      adjustedFitnessTotal = adjustedFitnessTotal+gen.adjustedFitness
    end
    speciesGroup.averageFitness = adjustedFitnessTotal
  end
end

-- Choose a species for breeding weighted by fitness
local function chooseSpeciesForBreeding(speciesGroups)
  local totalFitness = 0
  for _, speciesGroup in ipairs(speciesGroups) do
    totalFitness = totalFitness + math.max(0,speciesGroup.averageFitness or 0)
  end
  if totalFitness <= 0 then return speciesGroups[math.random(#speciesGroups)] end
  local selectionPoint = math.random()*totalFitness
  for _, speciesGroup in ipairs(speciesGroups) do
    selectionPoint = selectionPoint-math.max(0,speciesGroup.averageFitness or 0)
    if selectionPoint <= 0 then return speciesGroup end
  end
  return speciesGroups[#speciesGroups]
end

-- Crossover two genomes
local function crossover(first,second)
  if second.fitness > first.fitness then first, second = second, first end
  local child = genome_module.create()
  local secondGenesByInnovation = {}
  for _, gene in ipairs(second.genes) do
    secondGenesByInnovation[gene.innovation] = gene
  end
  for _, firstGene in ipairs(first.genes) do
    local matchingGene = secondGenesByInnovation[firstGene.innovation]
    local selectedGene = matchingGene and math.random(2) == 1 and matchingGene or firstGene
    local childGene = {sourceNode=selectedGene.sourceNode,targetNode=selectedGene.targetNode,
      weight=selectedGene.weight,enabled=selectedGene.enabled,innovation=selectedGene.innovation}
    if matchingGene and ((not firstGene.enabled) or (not matchingGene.enabled)) and math.random() < 0.75 then
      childGene.enabled = false
    end
    child.genes[#child.genes+1] = childGene
  end
  child.highestHiddenNode = math.max(first.highestHiddenNode,second.highestHiddenNode)
  for mutationName, mutationRate in pairs(first.mutationRates) do
    child.mutationRates[mutationName] = mutationRate
  end
  return child
end

-- Breed a child from a species
local function breedChild(group,populationState)
  local speciesMembers = group.genomes
  if #speciesMembers == 1 then
    local child = genome_module.clone(speciesMembers[1])
    child.fitness, child.adjustedFitness = 0, 0
    mutation_module.mutate(child,populationState)
    return child
  end
  local function selectParent()
    local first = speciesMembers[math.random(#speciesMembers)]
    local second = speciesMembers[math.random(#speciesMembers)]
    return first.fitness >= second.fitness and first or second
  end
  local firstParent = selectParent()
  local secondParent = selectParent()
  local child = math.random() < 0.75
    and crossover(firstParent,secondParent) or genome_module.clone(firstParent)
  child.fitness, child.adjustedFitness = 0, 0
  mutation_module.mutate(child,populationState)
  return child
end

-- Generate next generation via reproduction and mutation
function species.nextGeneration(populationState)
  rankSpecies(populationState)
  local champion = populationState.genomes[1]
  local historicChampion=populationState.topPerformers and populationState.topPerformers[1]
  if historicChampion and historicChampion.genome
    and (historicChampion.fitness or 0)>(champion.fitness or 0) then
    champion=genome_module.clone(historicChampion.genome)
    champion.fitness=historicChampion.fitness
  end
  local survivingSpecies = {}
  for _, speciesGroup in ipairs(populationState.species) do
    if speciesGroup.staleness < MAX_STALE_GENERATIONS or speciesGroup.genomes[1] == champion then
      local survivorCount = math.max(1,math.ceil(#speciesGroup.genomes/2))
      for memberIndex = #speciesGroup.genomes, survivorCount+1, -1 do
        speciesGroup.genomes[memberIndex] = nil
      end
      survivingSpecies[#survivingSpecies+1] = speciesGroup
    end
  end
  if #survivingSpecies == 0 then
    survivingSpecies = {{id=1,genomes={champion},topFitness=champion.fitness,staleness=0,
      representative=genome_module.clone(champion),averageFitness=1}}
  end
  local nextPopulation = {generation=populationState.generation+1,
    nextInnovation=populationState.nextInnovation,innovations=populationState.innovations,
    nextHiddenNode=populationState.nextHiddenNode,splitHistory=populationState.splitHistory,
    species=survivingSpecies,genomes={genome_module.clone(champion)},
    bestFitness=populationState.bestFitness,population=populationState.population,
    nextGenomeIndex=1,behaviorArchive=populationState.behaviorArchive or {},
    episodeHistory=populationState.episodeHistory or {},topPerformers=populationState.topPerformers or {},
    experienceMemory=populationState.experienceMemory or {},
    legacyLogImported=populationState.legacyLogImported or false}
  local targetPopulationSize = populationState.population or 300
  local activePolicies={[genome_module.signature(champion)]=true}
  while #nextPopulation.genomes < targetPopulationSize do
    local speciesGroup = chooseSpeciesForBreeding(survivingSpecies)
    local child=breedChild(speciesGroup,nextPopulation)
    local signature=genome_module.signature(child)
    local retryCount=0
    while activePolicies[signature] and retryCount<32 do
      child=breedChild(speciesGroup,nextPopulation)
      signature=genome_module.signature(child)
      retryCount=retryCount+1
    end
    if activePolicies[signature] then
      retryCount=0
      repeat
        child=genome_module.create()
        mutation_module.mutate(child,nextPopulation)
        signature=genome_module.signature(child)
        retryCount=retryCount+1
      until not activePolicies[signature] or retryCount>=128
    end
    assert(not activePolicies[signature],"unable to generate a distinct NEAT policy")
    activePolicies[signature]=true
    nextPopulation.genomes[#nextPopulation.genomes+1]=child
  end
  species.assign(nextPopulation)
  return nextPopulation
end

return species
