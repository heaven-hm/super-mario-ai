local function install(AI,C)
local DEFAULT_POPULATION_SIZE=C.DEFAULT_POPULATION_SIZE
local SPECIES_DISTANCE_THRESHOLD=C.SPECIES_DISTANCE_THRESHOLD
local MAX_STALE_GENERATIONS=C.MAX_STALE_GENERATIONS
local cloneGenome=AI.cloneGenome
local genomeSignature=AI.genomeSignature
local createEmptyGenome=AI.createEmptyGenome
local createGene=AI.createGene
local getInnovationNumber=AI.getInnovationNumber
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

local function assignSpecies(populationState)
  local previousSpecies = populationState.species or {}
  local currentSpecies = {}
  for _, genome in ipairs(populationState.genomes) do
    local matchingSpecies
    for _, speciesGroup in ipairs(currentSpecies) do
      if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
        matchingSpecies=speciesGroup
        break
      end
    end
    if not matchingSpecies then
      local previousMatch
      for _, speciesGroup in ipairs(previousSpecies) do
        if calculateGenomeDistance(genome,speciesGroup.representative)<SPECIES_DISTANCE_THRESHOLD then
          previousMatch=speciesGroup
          break
        end
      end
      matchingSpecies={id=previousMatch and previousMatch.id or (#currentSpecies+1),genomes={},
        topFitness=previousMatch and previousMatch.topFitness or 0,
        staleness=previousMatch and previousMatch.staleness or 0,
        representative=cloneGenome(genome)}
      currentSpecies[#currentSpecies+1]=matchingSpecies
    end
    matchingSpecies.genomes[#matchingSpecies.genomes+1]=genome
    genome.species=matchingSpecies.id
  end
  populationState.species=currentSpecies
end

-- Create one seeded SMB1 controller, then mutate clones until the requested
-- population size is reached. The first genome provides a useful prior; NEAT
-- is free to replace that prior during later generations.

local function crossover(first,second)
  if second.fitness > first.fitness then first, second = second, first end
  local child = createEmptyGenome()
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

local function rankSpecies(populationState)
  table.sort(populationState.genomes,function(firstGenome,secondGenome)
    return firstGenome.fitness > secondGenome.fitness
  end)
  for rank, genome in ipairs(populationState.genomes) do
    genome.globalRank = #populationState.genomes-rank+1
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
    for _, genome in ipairs(speciesGroup.genomes) do
      genome.adjustedFitness = genome.globalRank/math.max(1,#speciesGroup.genomes)
      adjustedFitnessTotal = adjustedFitnessTotal+genome.adjustedFitness
    end
    -- Adjusted fitness already divides each rank by species size. Summing it
    -- gives the species one fair breeding weight; averaging would divide by
    -- species size twice and overproduce one-member species.
    speciesGroup.averageFitness = adjustedFitnessTotal
  end
end

local function chooseSpeciesForBreeding(species)
  local totalFitness = 0
  for _, speciesGroup in ipairs(species) do
    totalFitness = totalFitness + math.max(0,speciesGroup.averageFitness or 0)
  end
  if totalFitness <= 0 then return species[math.random(#species)] end
  local selectionPoint = math.random()*totalFitness
  for _, speciesGroup in ipairs(species) do
    selectionPoint = selectionPoint-math.max(0,speciesGroup.averageFitness or 0)
    if selectionPoint <= 0 then return speciesGroup end
  end
  return species[#species]
end

local function breedChild(group,populationState)
  local speciesMembers = group.genomes
  if #speciesMembers == 1 then
    local child = cloneGenome(speciesMembers[1])
    child.fitness, child.adjustedFitness = 0, 0
    -- A newly formed species must keep exploring rather than make exact copies.
    AI.mutate(child,populationState)
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
    and crossover(firstParent,secondParent) or cloneGenome(firstParent)
  child.fitness, child.adjustedFitness = 0, 0
  AI.mutate(child,populationState)
  return child
end

function AI.nextGeneration(populationState)
  rankSpecies(populationState)
  local champion = populationState.genomes[1]
  local historicChampion=populationState.topPerformers and populationState.topPerformers[1]
  if historicChampion and historicChampion.genome
    and (historicChampion.fitness or 0)>(champion.fitness or 0) then
    champion=cloneGenome(historicChampion.genome)
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
      representative=cloneGenome(champion),averageFitness=1}}
  end
  local nextPopulation = {generation=populationState.generation+1,
    nextInnovation=populationState.nextInnovation,innovations=populationState.innovations,
    nextHiddenNode=populationState.nextHiddenNode,splitHistory=populationState.splitHistory,
    species=survivingSpecies,genomes={cloneGenome(champion)},
    bestFitness=populationState.bestFitness,population=populationState.population,
    nextGenomeIndex=1,behaviorArchive=populationState.behaviorArchive or {},
    episodeHistory=populationState.episodeHistory or {},topPerformers=populationState.topPerformers or {},
    experienceMemory=populationState.experienceMemory or {},
    legacyLogImported=populationState.legacyLogImported or false}
  local targetPopulationSize = populationState.population or DEFAULT_POPULATION_SIZE
  local activePolicies={[genomeSignature(champion)]=true}
  while #nextPopulation.genomes < targetPopulationSize do
    local speciesGroup = chooseSpeciesForBreeding(survivingSpecies)
    local child=breedChild(speciesGroup,nextPopulation)
    local signature=genomeSignature(child)
    local retryCount=0
    while activePolicies[signature] and retryCount<32 do
      child=breedChild(speciesGroup,nextPopulation)
      signature=genomeSignature(child)
      retryCount=retryCount+1
    end
    -- A fresh sparse genome is a final escape from a clone-heavy species.
    if activePolicies[signature] then
      retryCount=0
      repeat
        child=AI.newGenome(nextPopulation)
        AI.mutate(child,nextPopulation)
        signature=genomeSignature(child)
        retryCount=retryCount+1
      until not activePolicies[signature] or retryCount>=128
    end
    assert(not activePolicies[signature],"unable to generate a distinct NEAT policy")
    activePolicies[signature]=true
    nextPopulation.genomes[#nextPopulation.genomes+1]=child
  end
  assignSpecies(nextPopulation)
  return nextPopulation
end
AI.assignSpecies=assignSpecies
AI.calculateGenomeDistance=calculateGenomeDistance
end

return install
