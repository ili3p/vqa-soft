local M = {}

M.len = function(tbl)
   local c = 0

   for _,_ in pairs(tbl) do
      c = c + 1
   end

   return c
end
M.set = function(hash, key, obj)
   tds = require'tds'
   hash[key] = tds.Hash()
   for k, v in pairs(obj) do
      if type(v) == 'table' then
         M.set(hash[key], k, v)
      else
         hash[key][k] = v
      end
   end
end
M.keys = function(tbl)
    local elems = {}
    for k, _ in pairs(tbl) do
        table.insert(elems, k)
    end

    return elems
end

M.file_as_string = function(fn)
    local f = io.open(fn, 'r')
    local str = f:read()
    f:close()

    return str
end

M.range = function(n)
   local r = {}
   for i=1,n do
      table.insert(r,i)
   end

   return r
end

M.hash2tbl = function(hash)
   local tbl = {}
   for k,v in pairs(hash) do
      tbl[k] = v
   end
   
   return tbl
end

M.tableinvert = function(tbl)
   local inv = {}
   for k,v in pairs(tbl) do
      if inv[v] then print('Error:','Values are not unique ('..v..')') end
      inv[v] = k
   end
   return inv
end

M.word_ids_to_word = function(id2word, wordTens)
   local words = {}
   for i=1,wordTens:size(1) do
      if wordTens[i] ~=  0 then
         table.insert(words, id2word[wordTens[i]])
         table.insert(words, ' ')
      end
   end

   return table.concat(words)
end


M.spairs = function (t, order)
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

M.printsorted = function(t)
   for k,v in M.spairs(t) do
      -- TODO
      print(' ',k,v)
   end
end


M.plotter = function(plot,title, legend)
   -- local colors = {'#FF0000','#00FF00','#0000FF'}
   local plot = plot
   local title = title
   local legend = legend
   local inv = M.tableinvert(legend)
   local n = #legend
   local ybuff = torch.DoubleTensor(2, n)
   local xbuff = torch.DoubleTensor(1, 2)
   local t = 0
   local id 
   return {
      plot = function(name, x, y)
         if id then
            plot:updateTrace{
               win      = id, 
               name     = name,
               X        = torch.DoubleTensor{x},
               Y        = torch.DoubleTensor{y},
               append   = true,
               options  = {
                  title = 'Updated at: ' .. os.date(" %H:%M:%S"),
               },
            }
         else
            t = t + 1
            ybuff[math.ceil(t/n)][inv[name]] = y
            xbuff[1][math.ceil(t/n)] = x
            if t == 2*n then
               id = plot:line{
                  -- X        = xbuff:view(1,2):repeatTensor(2,1),
                  Y        = ybuff:view(2,n),
                  options  = {
                     legend      = legend, 
                     title       = title, 
                     -- markercolor = colors,
                  },
               }
               t = nil
               xbuff = nil
               ybuff = nil
            end
         end
      end
   }
end
         


M.load_json = function(fn)
   local cjson = require'cjson'

   local file = io.open(fn, 'r')
   local jsondata = cjson.decode(file:read())
   file:close()

   return jsondata
end

M.json_to_hash2 = function(fn )
   local tds = require'tds'

   local jsondata = M.load_json(fn)

   local hash = tds.Hash() 
   for k,v in pairs(jsondata) do
      hash[k] = tds.Hash()
      for i=1,#v do
         hash[k][i] = v[i]
      end
   end

   return hash
end
   
M.json_to_hash = function(fn, key, _hash)
   local tds = require'tds'

   local jsondata = M.load_json(fn)[key]

   local hash = _hash and _hash or tds.Hash() 
   for i=1,#jsondata do
      local index = #hash+1
      hash[index] = tds.Hash()
      for k,v in pairs(jsondata[i]) do
         hash[index][k] = v
      end
   end

   return hash
end

M.preprocess_string = function(str)
    str = str:lower():gsub('"', ''):gsub('\'s',''):gsub('@',' ')
    str = str:gsub('%w+%.com', 'url'):gsub('&',' ')
    str = str:gsub('%(',''):gsub('%)',''):gsub('\'',''):gsub('#',' ')
    str = str:gsub('!',' '):gsub(',',' '):gsub('-',' '):gsub('`',' ')
    str = str:gsub('%$', 'dollar '):gsub('/',' '):gsub('%.',' '):gsub(';',' ')
    str = str:gsub('>',' '):gsub(':',' ')
    str = str:gsub('_',' '):gsub('%*',''):gsub('?',' ')
    str = str:gsub('%d+',' digit ')
    str = str:gsub(' +',' ')

    return str
end

return M
