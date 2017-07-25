-- debugger is a global variable so it can be accessed from everywhere
_, debugger = pcall(require,'fb.debugger') 
local tds = require'tds'
local util = require'../util.lua'
local logger = require'../logger.lua'

local opt = {
   train_que   = '../../vqa2_data/v2_OpenEnded_mscoco_train2014_questions.json',
   val_que     = '../../vqa2_data/v2_OpenEnded_mscoco_val2014_questions.json',
   testdev_que = '../../vqa2_data/v2_OpenEnded_mscoco_test-dev2015_questions.json',
   test_que    = '../../vqa2_data/v2_OpenEnded_mscoco_test2015_questions.json',

   word2vec    = './wiki.en.vec',
   vec_len     = 300,
   nvec        = 2519371,

   outhash     = '../fastText.hash'
}

for k, v in pairs(opt) do 
   opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end
print(opt)



local build_vocab = function()
   logger.info('Building vocabulary out of all question words...')
   local vocab = tds.Hash()
   local jsons = {opt.train_que, opt.val_que, opt.testdev_que, opt.test_que}

   for i=1, #jsons do 
      logger.info('Processing '..jsons[i])
      local que_hash = util.json_to_hash(jsons[i], 'questions')
      for j=1, #que_hash do
         if j%100 == 0 then xlua.progress(j,#que_hash) end
         local question = que_hash[j]['question']
         local words = util.preprocess_string(question):split(' ') 
         for k=1, #words do
            if not vocab[words[k]] then
               vocab[words[k]] = #vocab + 1
            end
         end
      end
      xlua.progress(#que_hash, #que_hash)
   end

   logger.info('Done. Vocabulary size is '..#vocab)

   return vocab
end

vocab = build_vocab()
local lookup = torch.FloatTensor(#vocab, opt.vec_len)

local UNK
local done = 0
local buffsz = 2^13 -- == 8k
local f = io.input(opt.word2vec)
logger.info('Extracting word2vec vectors')
while true do -- breaks when no more lines
    local lines, leftover = f:read(buffsz, '*line')
    if not lines then break end  -- no more lines
    if leftover then lines = lines .. leftover .. '\n' end -- join the leftover
    lines = lines:split('\n')

    for i=1, #lines do
        if done % 1000 == 0 then xlua.progress(done, opt.nvec) end
        local line = lines[i]:split(' ')
        local word = line[1]
        table.remove(line, 1) -- remove the word
        if word == 'unk' then 
            UNK = torch.FloatTensor(line) 
        else
            local index = vocab[word]
            if index then
               vocab[word] = torch.FloatTensor(line)
            end
        end
        done = done + 1
    end
end
xlua.progress(done, opt.nvec)
f:close()

local unks = 0
for word,index in pairs(vocab) do
   if type(index) == 'number' then
      logger.debug('No word2vec vector for ' .. word)
      vocab[word] = UNK
      unks = unks + 1
   end
end

vocab['UNK'] = UNK

torch.save(opt.outhash, vocab)
logger.info('Words in vocab '.. #vocab)
logger.info('Words set to UNK vector '.. unks)
