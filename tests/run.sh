#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
LUA_BIN="${LUA_BIN:-lua5.1}"
command -v "$LUA_BIN" >/dev/null 2>&1 || LUA_BIN=luajit
"$LUA_BIN" tests/run.lua
"$LUA_BIN" -e "assert(loadfile('mario_ai_neat.lua'))"

# Keep emulator loop tests away from the live training database in this folder.
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/mario-ai-tests.XXXXXX")
trap 'rm -rf "$test_directory"' EXIT
cp mario_ai_neat.lua tests/integration.lua tests/recovery.lua tests/level_flow.lua tests/champion_flow.lua "$test_directory/" && mkdir -p "$test_directory/scripts" && cp -R scripts/smb1 scripts/neat scripts/qmemory.lua scripts/hud.lua scripts/storage.lua "$test_directory/scripts/"
cd "$test_directory"
"$LUA_BIN" integration.lua
"$LUA_BIN" recovery.lua
"$LUA_BIN" level_flow.lua
"$LUA_BIN" champion_flow.lua
