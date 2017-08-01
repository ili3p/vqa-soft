local data = require'./datapreprocess.lua'
-- data private variables
local _lookuptable
-- train_tid2img maps a question tensor index to an image filename
local train_questions, train_tid2img, train_answers
local cache, itensor, storage
local val_cache, val_itensor, val_storage

-- for debugging purposes
local vocab, train_tid2qid, test_tid2qid
-- test_answers only not nil when test is val set
local test_questions, test_tid2img, soft_test_answers, test_answers, test_tid2anstype

local threads = require'threads'
threads.Threads.serialization('threads.sharedserialize')
local valbatch
local permutation, epoch_size, val_epoch_size
local opt, logger

local buffer = {}
local in_buffer = tds.Vec()
local running = tds.Vec()
local last_iter = 0
local epoch = 0

local val_buffer = {}
local val_in_buffer = tds.Vec()
local val_running = tds.Vec()
local val_last_iter = 0

local _addwork = function(iter)

   local work = function(iter, in_buffer, running)
      if not running[iter%opt.buffer_size + 1] and not in_buffer[iter%opt.buffer_size + 1] then
         running[iter%opt.buffer_size + 1] = true
         return worker.dowork(train_questions, train_tid2img, train_answers, cache, itensor, permutation, iter)
      else -- buffer is full, must wait
         logger.warn('Train buffer is full at iteration ' .. iter)
         return -1
      end
   end
   local endcallback = function(iter, ...) 
      if iter ~= -1 then
         assert(not buffer[iter%opt.buffer_size + 1])
         buffer[iter%opt.buffer_size + 1] = {...} 
         in_buffer[iter%opt.buffer_size + 1] = true
         running[iter%opt.buffer_size + 1] = false
      end
   end

   last_iter = math.max(iter, last_iter)

   pool:addjob(work, endcallback, iter, in_buffer, running)
end

local _getbatch = function(iter)
   local t = sys.clock()

   if iter%epoch_size == 0 then
      while pool:hasjob() do
         pool:dojob()
      end
      permutation = torch.randperm(permutation:size(1))
      epoch = epoch + 1
      for i=1,opt.dataworkers-1 do 
         _addwork(iter + i)
      end
   end
   if (last_iter-(epoch*epoch_size)) + 1 <= epoch_size then
      _addwork(last_iter + 1)
   end

   local ind = iter%opt.buffer_size + 1
   local c = 0
   while not buffer[ind] do -- wait for this specific batch
      pool:dojob()
      c = c + 1
      if c > 10 then
         logger.warn('Stuck')
      end
   end
   if c > 1 then
     logger.trace('Got batch after '..c..' tries.')
   end
   local batch = buffer[ind]
   buffer[ind] = nil
   in_buffer[ind] = false
   logger.trace('dojob', (sys.clock()-t))
   t = sys.clock()
   logger.trace('_addwork', (sys.clock()-t))
   return table.unpack(batch)
end

local _addvalwork = function(iter)
   local work = function(iter, val_in_buffer, val_running) 
      if not val_running[iter%opt.val_buffer_size + 1] and not val_in_buffer[iter%opt.val_buffer_size + 1] then
         val_running[iter%opt.val_buffer_size + 1] = true
         return valworker.dowork(test_questions, test_tid2img, test_answers, soft_test_answers, val_cache, val_itensor, test_tid2anstype, iter)
      else
         logger.warn('Val buffer is full at iteration '.. iter)
         return -1
      end
   end
   local endcallback = function(iter, ...) 
      if iter ~= -1 then
         assert(not val_buffer[iter%opt.val_buffer_size +1])
         val_buffer[iter%opt.val_buffer_size + 1] = {...}
         val_in_buffer[iter%opt.val_buffer_size + 1] = true
         val_running[iter%opt.val_buffer_size + 1] = false 
      end
   end

   val_last_iter = math.max(iter, val_last_iter)

   valpool:addjob(work, endcallback, iter, val_in_buffer, val_running)
end

local _getvalbatch = function(iter)
   local t = sys.clock()

   _addvalwork(val_last_iter + 1)

   local ind = iter%opt.val_buffer_size + 1
   local c = 0
   while not val_buffer[ind] do -- wait for this specific batch
      valpool:dojob()
      c = c + 1
      if c > 10 then
         logger.warn('Stuck at val')
      end
   end
   if c > 1 then
     logger.trace('Got batch after '..c..' tries.')
   end
   local batch = val_buffer[ind]
   val_buffer[ind] = nil
   val_in_buffer[ind] = false
   logger.trace('dojob', (sys.clock()-t))
   t = sys.clock()
   logger.trace('_addwork', (sys.clock()-t))
   return table.unpack(batch)
end

local _setvalworkers = function()
   for i=1, opt.dataworkers do 
      _addvalwork(i)
   end
end

local _getvocab = function()
   return vocab
end

local _getlookup = function()
   return _lookuptable
end

local _init = function(_opt, _logger)
   opt = _opt 
   logger = _logger
   paths.mkdir('./data/')

   -- not used, better to leave it to the OS to do RAM caching 
   cache = {} 
   val_cache = {}

   local t = sys.clock()
   local q_data = data.get_qdata()
   local answers, soft_answers = data.get_answers()
   _lookuptable = data.get_lookup()

   data.clean()

   train_questions = q_data.train_questions
   train_tid2img = q_data.train_tid2img
   train_tid2qid = q_data.train_tid2qid

   train_answers = (opt.criterion:find('Soft') and soft_answers or answers)['train_answers']

   vocab = q_data.vocab

   test_questions = q_data.test_questions
   test_tid2img = q_data.test_tid2img
   test_tid2qid = q_data.test_tid2qid
   test_tid2anstype = q_data.test_tid2anstype

   test_answers = answers['test_answers']
   soft_test_answers = soft_answers['test_answers']

   logger.info('Vocab size '..#vocab)
   logger.info('Train questions '..train_questions:size(1))
   logger.info('Test questions ' ..test_questions:size(1))

   permutation = torch.randperm(train_questions:size(1))
   epoch_size = math.floor(train_questions:size(1)/opt.batch_size)
   val_epoch_size = math.ceil(test_questions:size(1)/opt.val_batch_size)

   opt.ans_aug = opt.criterion:find('Soft') and opt.ans_aug or false

   pool = threads.Threads(opt.dataworkers, 
            function(threadid) 
               require'sys' 
               require'torch' 
               require'xlua'
               require'cunn'
               require'torchzlib'
               tds = require'tds'
            end, 
            function() 
               torch.manualSeed(opt.rnd_seed + __threadid)
               cutorch.manualSeed(opt.rnd_seed + __threadid)
               torch.setnumthreads(1)
               worker = paths.dofile('./dataworker.lua') 
               worker.init(opt.batch_size, epoch_size, logger, opt.que_len, opt.img_dir, opt.criterion:find('Soft'), opt.ans_aug)
            end)

   valpool = threads.Threads(opt.val_dataworkers, 
            function(threadid) 
               require'sys' 
               require'torch' 
               require'xlua'
               require'cunn'
               require'torchzlib'
               tds = require'tds'
            end, 
            function() 
               torch.manualSeed(opt.rnd_seed + __threadid)
               cutorch.manualSeed(opt.rnd_seed + __threadid)
               torch.setnumthreads(1)
               valworker = paths.dofile('./valdataworker.lua') 
               valworker.init(opt.val_batch_size, val_epoch_size, logger, opt.que_len, opt.img_dir)
            end)

   for iter=1, opt.dataworkers do 
      _addwork(iter) 
   end
   for iter=1, opt.val_dataworkers do
      _addvalwork(iter)
   end

   logger.trace('init', (sys.clock()-t))

   opt.word2vec = '../../utils/glove.hash'
   ltbl = data.get_lookup()

   collectgarbage()
   return epoch_size, val_epoch_size, test_questions:size(1)
end

return {
   init = _init,
   getbatch = _getbatch,
   getvalbatch = _getvalbatch,
   setvalworkers = _setvalworkers,
   getlookup = _getlookup,
   getvocab = _getvocab,
   test_tid2qid = test_tid2qid,
   train_tid2qid = train_tid2qid,
   answer_data = data.answer_data,
}
