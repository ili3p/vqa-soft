local bsz, epoch_size, que, ans, img, loaded, ans_aug 
local logger, img_dir

local _init = function(_bsz, _epoch_size, _logger, que_len, _img_dir, _soft_ans, _ans_aug)
   bsz = _bsz
   epoch_size = _epoch_size
   ans_aug = _ans_aug
   que = torch.FloatTensor(bsz, que_len)
   img = torch.FloatTensor(bsz, 2048, 14, 14)
   if _soft_ans then
      if ans_aug then
         ans = torch.FloatTensor(bsz, 1, 2)
      else
         ans = torch.FloatTensor(bsz, 10, 2)
      end
   else
      ans = torch.FloatTensor(bsz, 11)
   end

   loaded = {}
   logger = _logger
   img_dir = _img_dir
end

local _dowork = function(questions, mapping, answers, cache, itensor, permutation, iter)

   local t = sys.clock()
   collectgarbage()
   logger.trace('collectgarbage',(sys.clock()-t))

   assert(answers)

   loaded = {}
   for i=1, bsz do
      local index = permutation[(i + bsz*((iter-1)%epoch_size))]

      img[i] = cache[mapping[index]] and itensor[cache[mapping[index]]] or torch.load(img_dir .. mapping[index]):decompress()

      que[i] = questions[index]
      if ans_aug then 
         local k = 1
         for i=2, 10 do 
            if answers[index][i][1] == 0 then
               k = i-1
               break
            end
         end
         print(k)
         k = torch.random(k)
         ans[i][1] = answers[index][k]
      else
         ans[i] = answers[index]
      end
      loaded[i] = mapping[index]
   end

   logger.trace('_dowork',(sys.clock()-t))
   return iter, que:clone(), img:clone(), ans:clone(), tds.Hash(loaded) 
end

return {
   init = _init,
   dowork = _dowork,
}
