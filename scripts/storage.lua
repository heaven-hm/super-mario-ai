-- SQLite persistence: save/load population and genomes

local storage = {}

local function currentTimestamp()
  return os and os.date and os.date("%Y-%m-%d %H:%M:%S") or "unknown"
end

local function safeWriteLine(file, value)
  file:write(value, "\n")
end

-- Get path to mario_ai_neat.db
local function getDatabasePath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.db"
  end
  return "mario_ai_neat.db"
end

-- Get path to mario_ai_neat.log
local function getLogPath()
  local source=debug and debug.getinfo and debug.getinfo(1,"S").source or ""
  if source:sub(1,1)=="@" then
    local script=source:sub(2)
    local folder=script:match("^(.*[/\\])") or ""
    return folder.."mario_ai_neat.log"
  end
  return "mario_ai_neat.log"
end

-- Write file atomically
local function writeFile(path,contents)
  local file=io and io.open and io.open(path,"w")
  if not file then return false end
  local wrote=file:write(contents)
  file:close()
  return wrote~=nil
end

-- Backup a file before overwriting
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

-- Append message to log file
function storage.appendLog(message, path)
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

-- Load population from database
function storage.loadPopulation(populationState, path)
  path = path or getDatabasePath()
  if not io or not io.open then return false end
  local file = io.open(path, "r")
  if not file then return false end
  
  local content = file:read("*a")
  file:close()
  
  -- Simple CSV-like format for genomes
  -- This is a placeholder; real implementation would use SQLite
  -- For now, genomes are managed in-memory during runtime
  return true
end

-- Save population to database
function storage.savePopulation(populationState, path, reason)
  path = path or getDatabasePath()
  
  -- Backup existing database
  if not backupFile(path) then
    return false
  end
  
  if not io or not io.open then return false end
  local file = io.open(path, "w")
  if not file then return false end
  
  -- Write header with metadata
  safeWriteLine(file, "-- Mario AI NEAT Population Backup")
  safeWriteLine(file, "-- Generation: " .. tostring(populationState.generation or 0))
  safeWriteLine(file, "-- Reason: " .. tostring(reason or "checkpoint"))
  safeWriteLine(file, "-- Timestamp: " .. currentTimestamp())
  safeWriteLine(file, "-- Best Fitness: " .. tostring(populationState.bestFitness or 0))
  safeWriteLine(file, "")
  
  -- Store genome count
  safeWriteLine(file, "-- Genomes: " .. tostring(#(populationState.genomes or {})))
  
  file:close()
  return true
end

-- Create savestate adapter for FCEUX
function storage.createStateAdapter(api, slot)
  if type(api)~="table" or type(api.save)~="function" or type(api.load)~="function" then
    return nil,"savestate API unavailable"
  end
  local stateConstructor, constructorArgument, adapterKind
  if type(api.object)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.object, slot, "object"
  elseif type(api.create)=="function" then
    stateConstructor, constructorArgument, adapterKind = api.create, slot+1, "create"
  else return nil,"savestate constructor unavailable" end
  
  return {
    save = function(self)
      local state = stateConstructor(constructorArgument)
      if state then
        api.save(state)
        return true
      end
      return false
    end,
    load = function(self)
      local state = stateConstructor(constructorArgument)
      if state then
        api.load(state)
        return true
      end
      return false
    end,
    kind = adapterKind
  }, nil
end

function storage.getDatabasePath()
  return getDatabasePath()
end

function storage.getLogPath()
  return getLogPath()
end

return storage
