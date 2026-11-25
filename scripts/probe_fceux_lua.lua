-- Minimal FCEUX Lua liveness probe used while diagnosing Issue #3.
--
--   fceux --loadlua scripts/probe_fceux_lua.lua /path/to/SuperMarioBros.nes
--
-- Writes a line every 30 emulated frames to the path in PROBE_OUTPUT. If the
-- file never gets a second line, FCEUX is loading the Lua script but its
-- emu.frameadvance() coroutine is not being resumed, so the project's
-- python/fceux_bridge harness cannot produce observations.

local PROBE_OUTPUT = os.getenv("FCEUX_LUA_PROBE_OUTPUT")
  or "/tmp/fceux-lua-probe.txt"

local handle = io.open(PROBE_OUTPUT, "w")
if not handle then return end
handle:write(string.format("start paused=%s emulating=%s\n",
  tostring(emu.paused()), tostring(emu.emulating())))
handle:close()

local frames = 0
while true do
  frames = frames + 1
  if frames % 30 == 0 then
    handle = io.open(PROBE_OUTPUT, "a")
    if handle then
      handle:write(string.format("frames=%d paused=%s\n", frames, tostring(emu.paused())))
      handle:close()
    end
  end
  emu.frameadvance()
end
