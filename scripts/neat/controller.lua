-- NEAT controller: decision-making, episode management, and action selection

local controller = {}

local genome_module = require("scripts.neat.genome")
local observation_module = require("scripts.neat.observation")

-- Action definitions
controller.ACTION_OPTIONS = {
  {name="run", right=true, B=true},
  {name="jump_run", right=true, B=true, A=true},
  {name="retreat", left=true, B=true},
  {name="brake"},
  {name="jump_place", A=true},
  {name="walk", right=true},
}

controller.ACTION_HOLD_FRAMES = {1, 2, 4, 6}
controller.INITIAL_PROGRESS_DEADLINE_FRAMES = 180
controller.MIN_INITIAL_PROGRESS_PIXELS = 16

-- Create new AI state for an episode
function controller.createState(populationState)
  return {
    populationState = populationState,
    runner = {genome = populationState.genomes[1]},
    genomeIndex = 1,
    episodeActive = false,
    episodeFrames = 0,
    episodeDecisions = 0,
    episodeReward = 0,
    startWorldX = 0,
    furthestWorldX = 0,
    frames = 0,
    championMode = false,
    episodeStartTime = os and os.time and os.time() or 0
  }
end

-- Begin an episode
function controller.beginEpisode(aiState, state)
  aiState.episodeActive = true
  aiState.episodeFrames = 0
  aiState.episodeDecisions = 0
  aiState.episodeReward = 0
  aiState.startWorldX = state.worldX or 0
  aiState.furthestWorldX = state.worldX or 0
  aiState.episodeStartTime = os and os.time and os.time() or 0
end

-- Determine if training start is valid
function controller.isValidTrainingStart(state, maxX)
  return state.worldX and state.worldX <= (maxX or 128)
end

-- Determine if training start is unsafe
function controller.isUnsafeTrainingStart(aiState, state)
  return false
end

-- Abandon current episode
function controller.abandonEpisode(aiState)
  aiState.episodeActive = false
end

-- Make a decision based on genome and observation
function controller.decide(aiState, state, evaluateGenome, buildObservationInputs)
  if not aiState.episodeActive then
    return {name="brake"}
  end
  
  local genome = aiState.runner.genome
  local inputs = buildObservationInputs(state)
  local actionScores, nodeValues, durationScores = evaluateGenome(genome, inputs)
  
  local maxScore = -math.huge
  local bestActionIndex = 1
  for i = 1, #actionScores do
    if actionScores[i] > maxScore then
      maxScore = actionScores[i]
      bestActionIndex = i
    end
  end
  
  local maxDuration = -math.huge
  local bestDurationIndex = 1
  for i = 1, #durationScores do
    if durationScores[i] > maxDuration then
      maxDuration = durationScores[i]
      bestDurationIndex = i
    end
  end
  
  local actionOption = controller.ACTION_OPTIONS[bestActionIndex] or controller.ACTION_OPTIONS[1]
  local action = {name = actionOption.name, buttonIndex = bestActionIndex, reason = "network decision"}
  for key, value in pairs(actionOption) do
    if key ~= "name" then
      action[key] = value
    end
  end
  
  aiState.episodeDecisions = (aiState.episodeDecisions or 0) + 1
  return action
end

-- Determine episode stop reason
function controller.episodeStopReason(aiState, state)
  if aiState.episodeFrames > 600 then
    return "timeout"
  end
  if state.phase == "death" then
    return "death"
  end
  if state.phase == "victory" then
    return "victory"
  end
  if aiState.episodeFrames > controller.INITIAL_PROGRESS_DEADLINE_FRAMES 
    and (aiState.furthestWorldX or 0) < controller.MIN_INITIAL_PROGRESS_PIXELS then
    return "stuck"
  end
  return nil
end

-- Finish an episode and compute fitness
function controller.finishEpisode(aiState, state, forced_reason)
  if not aiState.episodeActive then return 0 end
  
  local reason = forced_reason or controller.episodeStopReason(aiState, state)
  local fitness = aiState.furthestWorldX or 0
  
  if reason == "victory" then
    fitness = fitness + 1000
  elseif reason == "death" or reason == "stuck" then
    fitness = math.max(0, fitness - 100)
  end
  
  aiState.runner.genome.fitness = fitness
  aiState.episodeActive = false
  return fitness
end

-- Get best genome index
function controller.bestGenomeIndex(populationState)
  local bestIndex = 1
  local bestFitness = 0
  for i, genome in ipairs(populationState.genomes) do
    if (genome.fitness or 0) > bestFitness then
      bestIndex = i
      bestFitness = genome.fitness or 0
    end
  end
  return bestIndex
end

-- Get best performer
function controller.bestPerformer(populationState)
  local bestIndex = controller.bestGenomeIndex(populationState)
  return {fitness = populationState.genomes[bestIndex].fitness or 0,
          genome = populationState.genomes[bestIndex]}
end

return controller
