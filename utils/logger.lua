--
-- M.lua
--
-- Copyright (c) 2016 rxi
--
-- This library is free software; you can redistribute it and/or modify it
-- under the terms of the MIT license. See LICENSE for details.
--

local M = { _version = "0.1.0" }

M.usecolor = true
M.outfile = nil
M.level = "trace"
M.logToConsole = true


local modes = {
  { name = "trace", color = "\27[34m", },
  { name = "debug", color = "\27[36m", },
  { name = "info",  color = "\27[32m", },
  { name = "warn",  color = "\27[33m", },
  { name = "error", color = "\27[31m", },
  { name = "fatal", color = "\27[30m", },
}


local levels = {}
for i, v in ipairs(modes) do
  levels[v.name] = i
end


local round = function(x, increment)
  increment = increment or 1
  x = x / increment
  return (x > 0 and math.floor(x + .5) or math.ceil(x - .5)) * increment
end

local table_to_str

-- taken from http://lua-users.org/wiki/TableUtils
local val_to_str = function( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
     if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return " .. v .. "
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table_to_str( v ) or
      tostring( v )
  end
end

-- taken from http://lua-users.org/wiki/TableUtils
local key_to_str = function(k)
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. val_to_str( k ) .. "]"
  end
end
local spairs = function (t, order)
   -- collect the keys
   local keys = {}
   for k in pairs(t) do keys[#keys+1] = k end

   -- if order function given, sort by it by passing the table and keys a, b,
   -- otherwise just sort the keys
   if order then
      table.sort(keys, function(a,b) return order(t, a, b) end)
   else
      table.sort(keys)
   end

   -- return the iterator function
   local i = 0
   return function()
      i = i + 1
      if keys[i] then
         return keys[i], t[keys[i]]
      end
   end
end

-- taken from http://lua-users.org/wiki/TableUtils
table_to_str = function(tbl)
  local result, done = {}, {}
  -- for k, v in spairs(tbl) do
  --   table.insert( result, val_to_str(v) )
  --   done[ k ] = true
  -- end
  for k, v in spairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
      '\t'..modes[1].color.. key_to_str( k ) .. " \27[0m : " .. val_to_str( v ) )
    end
  end
  return "{\n" .. table.concat( result, ", \n" ) .. "}"
end



-- local _tostring = tostring

local printstr = function(...)

  local t = {}
  for i = 1, select('#', ...) do
    local x = select(i, ...)
    if type(x) == "number" then
      x = round(x, .01)
    elseif type(x) == "table" then
      x = table_to_str(x)
    end
    t[#t + 1] = tostring(x)
  end
  return table.concat(t, " ")
end


for i, x in ipairs(modes) do
  local nameupper = x.name:upper()

  M[x.name] = function(...)
    
    if i >= levels[M.level] and M.outfile then    
       local msg = {...}
       if #msg > 1 then
          msg = msg[1] .. ' in '.. string.format('%d', msg[2]*1e3) ..'ms'
       else
          msg = ...
       end


       local info = debug.getinfo(2, "Sl")
       local lineinfo = info.short_src .. ":" .. info.currentline

      local fp = io.open(M.outfile, "a")
      local str = string.format("[%-6s%s]%s: %s\n",
                                nameupper, os.date(), lineinfo, printstr(msg))
      fp:write(str)
      fp:close()
    end

    -- Output to console
    if i >= levels[M.level] and M.logToConsole then
      local msg = {...}
      if #msg > 1 then
         msg = msg[1] .. ' in '.. string.format('%d', msg[2]*1e3) ..'ms'
      else
         msg = ...
      end


      local info = debug.getinfo(2, "Sl")
      local lineinfo = info.short_src .. ":" .. info.currentline

      print(string.format("%s[%-6s%s]%s: %s %s %s",
                        M.usecolor and x.color or "",
                        nameupper,
                        os.date("%H:%M:%S"),
                        lineinfo,
                        M.usecolor and "\27[1;30m" or "",
                        printstr(msg),
                        M.usecolor and "\27[0m" or ""
                        ))
    end


  end
end


local init = function(opt)
   paths.mkdir(opt.log_dir..'/'..opt.version..'/')
   if opt.log_to_file then 
      M.outfile=opt.log_dir .. '/' .. opt.version ..'/'.. 'log_'.. os.date("_%Y%m%d_%H%M%S")..'.log'
   end
   M.logToConsole = opt.log_to_console
   M.level = modes[opt.log_level].name
end

M.init = init

return M
