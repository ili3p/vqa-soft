local q_data, answer_data, vocab, answer_tensors
-- shorthand
local strf = string.format

local stop_words = { 'the', 'this', 'these', 'there', 'that', 'any', 'you', 
'in', 'some', 'such', 'being', 'will', 'of', 'for', 'a', 'an', 'to', 'as'}
for i=1,#stop_words do stop_words[stop_words[i]] = true end

-- the preprocessing of questions, override it if needed
local preprocess_string = function(str)
   local words = util.preprocess_string(str):split(' ')

   local new_words = {}
   for i=1, #words do
      if not stop_words[words[i]] then
         table.insert(new_words, words[i])
      end
   end

   local new_str = table.concat(new_words, ' ')
   return new_str 
end

local process_answers = function()

   local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix ..'_type_'..opt.ans_type
   suffix = suffix ..'_'..opt.answer_count..'.json'

   local qid2ans_fn = opt.qid2ans..suffix

   if not paths.filep(qid2ans_fn) then
      logger.info('qid2ans table not found in '..qid2ans_fn)

      local args = {}
      args[1] = '--input_train '          .. opt.ans_train
      args[2] = '--input_val '            .. opt.ans_val
      args[3] = '--output_qid2ans '       .. qid2ans_fn
      args[4] = '--output_qid2type '      .. opt.qid2type..suffix
      args[5] = '--output_qid2anstype '   .. opt.qid2anstype..suffix
      args[6] = '--output_ans_id2str '    .. opt.ans_id2str..suffix
      args[7] = '--answer_count '         .. opt.answer_count
      args[8] = '--ans_type '             .. opt.ans_type:gsub('-','/')

      if opt.train_on_val then
         args[9] = '--train_on_val True'
      end
      -- ugly but cjson cannot read big json files
      -- and I have no time to make lua json lib 
      -- that will use tds.Hash() instead of lua table
      -- after all I do not use lua anyway 
      logger.info('Preprocessing answer data...')
      logger.info('Executing: '.. ('python ./process_answers.py '..table.concat(args, ' ')))
      os.execute('python ./process_answers.py '..table.concat(args, ' ')) 
      logger.info('Done. qid2ans table in  '..qid2ans_fn)
   else
      logger.info('Found qid2ans table in '..qid2ans_fn)
   end
   local qid2ans  = util.load_json(qid2ans_fn)

   return qid2ans
end

local process_questions = function()

   local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix .. '_'..(opt.left_aligned and 'L' or 'R')
   suffix = suffix ..'_a'..opt.answer_count
   suffix = suffix ..'_f'..opt.word_freq
   suffix = suffix ..'_type_'..opt.ans_type
   suffix = suffix .. '_l'..opt.que_len ..'.t7'

   local output = {}

   if not paths.filep(opt.train_questions .. suffix) then
      logger.info(strf('File %s not found', opt.train_questions..suffix))
      answer_data = answer_data and answer_data or process_answers(opt)

      local train_jsons = {opt.que_train}
      local test_jsons = {}
   
      if opt.train_on_val then
         table.insert(train_jsons, opt.que_val)
         table.insert(test_jsons, opt.que_test)
      else
         table.insert(test_jsons, opt.que_val)
      end
      
      logger.info('Loading train questions...')
      local train_hash = tds.Hash() 
      -- use set_limit for separating train from val sets
      local set_limit
      for i=1, #train_jsons do 
         logger.info('Reading '..train_jsons[i])
         train_hash = util.json_to_hash(train_jsons[i], 'questions', train_hash)
         set_limit = set_limit and set_limit or  #train_hash
      end
      logger.info(strf('Done. Found %d train questions.', #train_hash))
   
      vocab = tds.Hash()

      -- large enough to fit all unique words in the whole dataset
      local word_counts = torch.IntTensor(20000,1):fill(0)
     
      logger.info('Building vocab on training set...')
      for i=1, #train_hash do
         if i%100 == 0 then xlua.progress(i, #train_hash) end
   
         local qid = train_hash[i]['question_id']
         if answer_data[tostring(qid)] then
            local question = train_hash[i]['question']
            local words = preprocess_string(question):split(' ') 
   
            for j=1, math.min(#words, opt.que_len) do
               if not vocab[words[j]] then
                  vocab[words[j]] = #vocab + 1
               end
               word_counts[vocab[words[j]]]:add(1)
            end
         end
      end
      xlua.progress(#train_hash, #train_hash)
      vocab['UNK'] = #vocab + 1
      logger.info('Whole vocabulary size:'..#vocab)


      logger.info('Removing less frequent words from vocab...')
      -- remove less frequent words
      local inv_vocab = util.tableinvert(vocab)
      local freqs, indexes = word_counts:sort(1, true)
      local i = 0
      local c = 0
      while true do
         if i%100 == 0 then xlua.progress(i,freqs:size(1)) end

         i = i + 1
         local freq = freqs[i][1]

         if freq == 0 then
            -- we reached the end of the vocab
            break
         end

         if freq <  opt.word_freq then 

            -- delete the word from vocab
            local word = inv_vocab[indexes[i][1]]
            vocab[word] = nil
            c = c+1
         end
      end
      -- make word indexes sequential 
      local tmp = tds.Hash()
      for k,_ in pairs(vocab) do
         tmp[k] = #tmp + 1
      end
      vocab = tmp
      tmp = nil
      xlua.progress(freqs:size(1),freqs:size(1))
      logger.info(strf('Removed %d (%.2f%%) words that appear less than %d times.',c,100*c/(#vocab+c),opt.word_freq))
      logger.info(strf('Vocab size %d (%.2f%%):',#vocab, 100*#vocab/(#vocab+c)))


      -- each row contains right-aligned, zero-padded, question word vocab index
      -- change to left-aligned when MLP language model
      -- allocate #train_hash rows and then reduce the size,
      -- because not all questions have an answer in the answer set
      local train_questions = torch.IntTensor(#train_hash, opt.que_len):fill(0)
        
      -- mapping from tensor row index to question id
      -- needed to connect the question tensor and answer tensor
      local train_tid2qid = tds.Hash()

      -- mapping from tensor row to image filename
      -- needed to connect the question tensor and image tensor
      local train_tid2img = tds.Hash()

      logger.info('Processing train questions...')
      local tid = 0
      local unks = 0
      local wc = 0
      local q_unks = 0
      local set = 'train2014'
      for i=1, #train_hash do
         if i%100 == 0 then xlua.progress(i, #train_hash) end

         -- if there are more quetions than in train set or is augmented train set
         -- we are using val set
         if i > set_limit or train_hash[i]['is_val'] then set = 'val2014' end
   
         local qid = train_hash[i]['question_id']
         if answer_data[tostring(qid)] then
            tid = tid + 1
            train_tid2qid[tid] = qid
            local img_fn = strf('%s/COCO_%s_%012d.t7z',set,set,train_hash[i]['image_id'])
            train_tid2img[tid] = img_fn
            if not paths.filep(opt.img_dir .. img_fn) then
               debugger.enter()
            end
   
            local question = train_hash[i]['question']
            local words = preprocess_string(question):split(' ') 
            local offset = opt.left_aligned and 0 or math.max(opt.que_len - #words, 0)
            local has_unk = false
   
            for j=1, math.min(#words, opt.que_len) do
               local index = vocab[words[j]]
               if not index then 
                  index = vocab['UNK']
                  unks = unks + 1
                  has_unk = true
               end
               wc = wc + 1
               train_questions[tid][offset + j] =  index
            end
            if has_unk then q_unks = q_unks+1 end
         end
      end
      xlua.progress(#train_hash, #train_hash)
      train_questions = train_questions[{{1, tid}}]
      logger.info(strf('Unknown train words %d (%.2f%%), all train words %d.', unks, 100*unks/wc, wc))
      logger.info(strf('Questions with at least one unknown word %d (%.2f%%).', q_unks, 100*q_unks/train_questions:size(1)))
      logger.info(strf('Train set size [%dx%d].',train_questions:size(1), train_questions:size(2)))
      logger.info(strf('Train set is %.2f%%.',100*train_questions:size(1)/#train_hash))
   
      logger.info('Loading test questions...')
      local test_hash = tds.Hash() 
      for i=1, #test_jsons do 
         logger.info('Reading '..test_jsons[i])
         test_hash = util.json_to_hash(test_jsons[i], 'questions', test_hash)
      end
      logger.info(strf('Done. Found %d test questions.', #test_hash))
   
      logger.info('Processing test questions...')
      local test_questions = torch.IntTensor(#test_hash, opt.que_len):fill(0)
      local test_tid2qid = tds.Hash()
      local test_tid2img = tds.Hash()
      unks = 0
      wc = 0
      tid = 0
      q_unks = 0
      set = opt.train_on_val and 'test2015' or 'val2014'
      for i=1, #test_hash do
         if i%100 == 0 then xlua.progress(i,#test_hash) end

         local qid = test_hash[i]['question_id']
         -- if not training on val then we need questions to have answer data
         -- or train on val
       
         if (not opt.train_on_val and answer_data[tostring(qid)]) or opt.train_on_val then 
            tid = tid + 1
            test_tid2qid[tid] = qid
            local img_fn = strf('%s/COCO_%s_%012d.t7z',set,set,test_hash[i]['image_id'])
            test_tid2img[tid] = img_fn
            if not paths.filep(opt.img_dir .. img_fn) then
               debugger.enter()
            end
   
            local question = test_hash[i]['question']
            local words = preprocess_string(question):split(' ') 
            local offset = opt.left_aligned and 0 or math.max(opt.que_len - #words, 0)
            local has_unk = false
   
            for j=1, math.min(#words, opt.que_len) do
               local index = vocab[words[j]]
               if not index then 
                  index = vocab['UNK']
                  unks = unks + 1
                  has_unk = true
               end
               wc = wc + 1
               test_questions[tid][offset + j] = index
            end
         end
      end
      xlua.progress(#test_hash, #test_hash)
      if tid > 0 then
         test_questions = test_questions[{{1,tid}}]
      end
      logger.info(strf('Unknown test words %d (%.2f%%), all test words %d.', unks, 100*unks/wc, wc))
      logger.info(strf('Questions with at least one unknown word %d (%.2f%%).', q_unks, 100*q_unks/test_questions:size(1)))
      logger.info(strf('Test set size [%dx%d].',test_questions:size(1), test_questions:size(2)))
      logger.info(strf('Test set is %.2f%%.',100*test_questions:size(1)/#test_hash))

      local test_tid2anstype

      logger.info('Prepring answer types for test questions')
      if not opt.train_on_val then

         test_tid2anstype = tds.Hash()
         local qid2tid = util.tableinvert(test_tid2qid)
         local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
         suffix = suffix ..'_type_'..opt.ans_type
         suffix = suffix ..'_'..opt.answer_count..'.json'
         local tmp = util.load_json(opt.qid2anstype .. suffix)

         for qid,anstype in pairs(tmp) do
            local tid = qid2tid[tonumber(qid)]
            if tid then -- tid is nil if qid is from train set
               test_tid2anstype[tid] = anstype
            end
         end
      end
      logger.info('Done.')


    
      logger.info('Saving files with suffix '..suffix..'.')

      torch.save(opt.train_questions  .. suffix, train_questions)
      torch.save(opt.train_tid2qid    .. suffix, train_tid2qid)
      torch.save(opt.train_tid2img    .. suffix, train_tid2img)
      torch.save(opt.test_questions   .. suffix, test_questions)
      torch.save(opt.test_tid2qid     .. suffix, test_tid2qid)
      torch.save(opt.test_tid2img     .. suffix, test_tid2img)
      torch.save(opt.test_tid2anstype .. suffix, test_tid2anstype)
      torch.save(opt.vocab            .. suffix, vocab)

      output = {
       train_questions  = train_questions,
       train_tid2qid    = train_tid2qid,
       train_tid2img    = train_tid2img,
       test_questions   = test_questions,
       test_tid2qid     = test_tid2qid,
       test_tid2img     = test_tid2img,
       test_tid2anstype = test_tid2anstype,
       vocab            = vocab,
      }

   else
      logger.info(strf('File %s found', opt.train_questions..suffix))
      logger.info('Loading files with suffix '..suffix..'.')
      output = {
       train_questions  = torch.load(opt.train_questions  .. suffix),
       train_tid2qid    = torch.load(opt.train_tid2qid    .. suffix),
       train_tid2img    = torch.load(opt.train_tid2img    .. suffix),
       test_questions   = torch.load(opt.test_questions   .. suffix),
       test_tid2qid     = torch.load(opt.test_tid2qid     .. suffix),
       test_tid2img     = torch.load(opt.test_tid2img     .. suffix),
       test_tid2anstype = torch.load(opt.test_tid2anstype .. suffix),
       vocab            = torch.load(opt.vocab            .. suffix),
      }
   end

   logger.info('All done with questions.')

   return output
end

local make_answer_tensor = function(question_tensor, tid2qid, answer_map)
   -- make answer tensors
   local n_que = question_tensor:size(1)
   local answer_tensor = torch.IntTensor(n_que, 11) 
   -- 11 since the  first is the MC answer and the rest are all 10 answers
   -- but be ware that it may contain -1 
   -- meaning the answer is not in the answer set
   for i=1, n_que do
      if i%100 == 0 then xlua.progress(i,n_que) end

      answer_tensor[i] = torch.IntTensor(answer_map[tostring(tid2qid[i])])
   end
   xlua.progress(n_que, n_que)

   return answer_tensor
end

local make_answers = function()
   -- if either answer or question data change we need to update these tensors
   local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix .. '_'..(opt.left_aligned and 'L' or 'R')
   suffix = suffix ..'_'..opt.answer_count
   suffix = suffix ..'_'..opt.ans_type
   suffix = suffix .. '_'..opt.que_len ..'.t7'

   local train_answers, test_answers

   if not paths.filep(opt.train_answers .. suffix) then
      answer_data = answer_data and answer_data or process_answers()
      q_data = q_data and q_data or process_questions()
      logger.info(strf('File %s not found.',opt.train_answers .. suffix))
      logger.info('Making answer tensor.')
      train_answers = make_answer_tensor(q_data.train_questions, q_data.train_tid2qid, answer_data)
      torch.save(opt.train_answers .. suffix, train_answers)

      -- if we are not training on val 
      -- then we have answers for test questions as well
      if not opt.train_on_val then
         logger.info('Making test answer tensor.')
         test_answers = make_answer_tensor(q_data.test_questions, q_data.test_tid2qid, answer_data)
         torch.save(opt.test_answers .. suffix, test_answers)
      end
      logger.info('Done.')
   else
      train_answers = torch.load(opt.train_answers .. suffix)
      if not opt.train_on_val then
         test_answers = torch.load(opt.test_answers .. suffix)
      end
   end

   return {train_answers = train_answers, test_answers = test_answers}
end

local make_soft = function(ans) 
   local soft_ans = torch.FloatTensor(ans:size(1), 10, 2):zero()

   for i=1, ans:size(1) do 
      local counts = {}
      for j=2,11 do
         counts[ans[i][j]] = counts[ans[i][j]] and counts[ans[i][j]] or 0 
         counts[ans[i][j]] = counts[ans[i][j]] + 1
      end
      local k = 1
      for id,v in pairs(counts) do -- unique answers with probabilities 
         if id ~= -1 then
            soft_ans[i][k][1] = id  -- add class index
            soft_ans[i][k][2] = v/10 
            k = k + 1
         end
      end
   end

   return soft_ans
end

local make_soft_answers = function(answer_tensors)
   local suffix = '_soft2_' 
   suffix = suffix .. '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix .. '_'..(opt.left_aligned and 'L' or 'R')
   suffix = suffix ..'_'..opt.answer_count
   suffix = suffix ..'_'..opt.ans_type
   suffix = suffix .. '_'..opt.que_len ..'.t7'

   local train_answers, test_answers

   if not paths.filep(opt.train_answers .. suffix) then
      logger.info(strf('File %s not found.',opt.train_answers .. suffix))
      logger.info('Making soft answer tensor.')
      train_answers = make_soft(answer_tensors.train_answers)
      torch.save(opt.train_answers .. suffix, train_answers)
      if not opt.train_on_val then
         logger.info('Making soft test answer tensor.')
         test_answers = make_soft(answer_tensors.test_answers)
         torch.save(opt.test_answers .. suffix, test_answers)
      end
   else
      train_answers = torch.load(opt.train_answers .. suffix)
      if not opt.train_on_val then
         test_answers = torch.load(opt.test_answers .. suffix)
      end
   end

   return {train_answers = train_answers, test_answers = test_answers}
end



local make_lookuptable = function()

   local w2v_suf = opt.word2vec:split('/')
   w2v_suf = w2v_suf[#w2v_suf]:split('%.')[1]

   local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix .. '_'..(opt.left_aligned and 'L' or 'R')
   suffix = suffix ..'_a'..opt.answer_count
   suffix = suffix ..'_f'..opt.word_freq
   suffix = suffix ..'_w'..w2v_suf
   suffix = suffix .. '_l'..opt.que_len ..'.t7'
   local ltbl


   if not paths.filep(opt.lookuptable .. suffix) then

      vocab = vocab and  vocab or process_questions()['vocab']
      logger.info(strf('File %s not found.',opt.lookuptable .. suffix))
      logger.info('Making new lookuptable using '..opt.word2vec)
      ltbl = nn.LookupTableMaskZero(#vocab, opt.vec_len)
      ltbl.weight[1]:zero() -- for zero padding, i.e. no word
      local word2vec = torch.load(opt.word2vec)

      for k,v in pairs(vocab) do
         ltbl.weight[1+v]:copy(word2vec[k])
      end
      torch.save(opt.lookuptable .. suffix, ltbl)
   else
      logger.info(strf('File %s found, loading...',opt.lookuptable .. suffix))
      ltbl = torch.load(opt.lookuptable .. suffix)
   end

   return ltbl
end

local get_qdata = function()
   q_data = q_data and q_data or process_questions()
   return q_data
end

local get_answers = function()
   answer_tensors = answer_tensors and answer_tensors or make_answers()
   soft_answers = soft_answers and soft_answers or make_soft_answers(answer_tensors)
   return answer_tensors, soft_answers
end

local get_lookup = function()
  return make_lookuptable() 
end

return {
   get_qdata = get_qdata,
   get_answers = get_answers,
   get_lookup = get_lookup,
   clean = function() q_data=nil; answer_data=nil; vocab=nil; answer_tensors=nil end
}
