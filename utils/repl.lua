require "trepl"

function debugRepl(restoreGlobals)
  restoreGlobals = restoreGlobals or false

  -- optionally make a shallow copy of _G
  local oldG = {}
  if restoreGlobals then
    for k, v in pairs(_G) do
      oldG[k] = v
    end
  end

  -- copy upvalues to _G
  local i = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local k, v = debug.getupvalue(func, i)
    if k ~= nil then
      _G[k] = v
    else
      break
    end
    i = i + 1
  end

  -- copy locals to _G
  local i = 1
  while true do
    local k, v = debug.getlocal(2, i)
    if k ~= nil then
      _G[k] = v
    else
      break
    end
    i = i + 1
  end

  repl()

  if restoreGlobals then
    _G = oldG
  end
end
