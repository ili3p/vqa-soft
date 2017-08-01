local bsz, epoch_size, que, ans, img, permutation, p, loaded 
local logger, img_dir

local _init = function(_bsz, _epoch_size, _logger, que_len, _img_dir)
   bsz = _bsz
   epoch_size = _epoch_size
   que = torch.FloatTensor(bsz, que_len)
   img = torch.FloatTensor(bsz, 2048, 14, 14)
   ans = torch.FloatTensor(bsz, 11)
   soft = torch.FloatTensor(bsz, 10, 2)
   loaded = {}
   logger = _logger
   img_dir = _img_dir
end

local _dowork = function(questions, mapping, answers, soft_answers, cache, itensor, anstypes, iter)

   local t = sys.clock()
   collectgarbage()
   logger.trace('collectgarbage',(sys.clock()-t))

   if not answers then
      ans = ans:zero()
   end

   loaded = {}
   que_types = {}
   for i=1, bsz do
      local index = (i + bsz*((iter-1)%epoch_size))
      if index > questions:size(1) then
         break
      end

      que[i] = questions[index]

      img[i] = cache[mapping[index]] and itensor[cache[mapping[index]]] or torch.load(img_dir .. mapping[index]):decompress()

      if answers then
         ans[i] = answers[index]
      end

      if soft_answers then
         soft[i] = soft_answers[index]
      end

      if anstypes then
         que_types[i] = anstypes[index]
      end

      loaded[i] = mapping[index]
   end

   if #loaded == 0 then return -1 end

   -- not enough for full batch
   if #loaded < bsz then
      return iter, que[{{1, #loaded}}]:clone(), img[{{1, #loaded}}]:clone(), ans[{{1, #loaded}}]:clone(), soft[{{1,#loaded}}]:clone(), tds.Hash(loaded), tds.Hash(que_types)
   end

   logger.trace('_dowork',(sys.clock()-t))
   return iter, que:clone(), img:clone(), ans:clone(), soft:clone(), tds.Hash(loaded), tds.Hash(que_types)
end

return {
   init = _init,
   dowork = _dowork,
}
