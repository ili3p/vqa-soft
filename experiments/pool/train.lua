-- debugger is a global variable so it can be accessed from everywhere
_, debugger = pcall(require,'fb.debugger') 
require'nn'
require'rnn'
require'cunn'
require'cudnn'
require'optim'
require'pretty-nn'
cudnn.benchmark = true
cudnn.fastest = true 
tds = require'tds'
logger = require'../../utils/logger.lua'
util = require'../../utils/util.lua'
optnet = require'optnet'
threads = require'threads'
threads.Threads.serialization('threads.sharedserialize')

strf = string.format

opt = require('./opt.lua')
for k, v in pairs(opt) do 
   if type(v) == 'boolean' then
      opt[k] = os.getenv(k) == nil and opt[k] or os.getenv(k) == 'true'
   elseif type(v) ~= 'table' then
      opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
   end
end
print(opt)
logger.init(opt)
logger.trace(xlua.table2string(opt, true))

local experiment_id = paths.basename(paths.cwd())
torch.manualSeed(opt.rnd_seed)
cutorch.manualSeed(opt.rnd_seed)

dataloader = require('./dataloader.lua')
local nepo, val_nepo, val_num_que = dataloader.init(opt, logger)

logger.info('Loading and initializing model from:'..opt.model)
local vqa = require(opt.model)

local model 
if opt.checkpoint:len() == 0 then
   model = vqa.new(opt, dataloader.getlookup())
   model = model:add(nn.LogSoftMax())
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:normal()
         v.bias:zero()
      end
   end
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   model = model:cuda()
   local sample = {
      torch.zeros(opt.batch_size,opt.que_len):type('torch.CudaTensor'), 
      torch.zeros(opt.batch_size,opt.size_image, 14, 14):type('torch.CudaTensor')
   }
   optnet.optimizeMemory(model, sample, {inplace = false, mode = 'training'})  
   sample = nil
else
   model = torch.load(opt.checkpoint):cuda()
   logger.info('Preloaded from '..opt.checkpoint)
end

local start_iter = tonumber(opt.start_iter)

if opt.num_gpus > 1 then
   net = nn.DataParallelTable(1, true, true)
   net:threads(function() 
      require 'cudnn'  
      require 'rnn'
      cudnn.benchmark = true
      cudnn.fastest = true 
   end)   
   net:add(model, util.range(opt.num_gpus))
else
   net = model
end
net = net:cuda()
local criterion = nn[opt.criterion]():cuda()
local val_criterion = criterion:clone():cuda()

logger.info('Network:\n')
logger.info(tostring(net))
logger.info('Epoch in '..nepo..' iterations')

local iter = tonumber(opt.start_iter) - 1
local val_iter = 1

local weights, dw = net:getParameters()

local que, img, ans, img_map, is_val, outputs, prob, soft, ans_type

local val_predictions = torch.IntTensor(val_num_que):zero()
local val_groundtruth = torch.IntTensor(val_num_que, 10):zero()
local eval_pool = threads.Threads(1, function() require'torch' require'xlua' tds = require'tds' end)

local train = function(x)
   if opt.repl then debugger.enter() end

   net:training()
   net:zeroGradParameters()
   dw:zero()
   if x ~= weights then
      weights:copy(x)
   end

   que, img, ans, img_map = dataloader.getbatch(iter-opt.start_iter+1)

   que = que:cuda()
   img = img:cuda()
   ans = opt.criterion:find('Soft') and ans:cuda() or ans[{{},{1}}]:cuda()

   outputs = net:forward({que, img})

   local loss = criterion:forward(outputs, ans)

   local dloss = criterion:backward(outputs, ans)

   net:backward({que, img}, dloss)


   if opt.gradclip > 0 then
      net:gradParamClip(opt.gradclip)
   end

   return loss, dw 
end

local calculate_score

local eval = function(dt, iter) 
   net:evaluate()
   if opt.repl then debugger.enter() end

   local val_answertypes = tds.Hash()
   local st = 1
   local correct = 0

   if iter == nepo then
      xlua.log('Evaluating on val...', 1)
   end
   for i=1, val_nepo do
      xlua.progress(i, val_nepo, 2)
      que, img, ans, soft, img_map, ans_type = dataloader.getvalbatch(val_iter)
      val_iter = val_iter + 1

      -- batch normalization of batch of 1 is impossible
      if que:size(1) == 1 then
         que = que:repeatTensor(2,1)
         ans = ans:repeatTensor(2,1)
         if img:size():size() == 2 then
            img = img:repeatTensor(2,1) 
         else
            img:repeatTensor(2,1,1,1)
         end
      end

      outputs = net:forward({que:cuda(), img:cuda()})
      local _, pred = outputs:max(2)


      val_predictions[{{st,st+pred:size(1)-1}}]:copy(pred:squeeze():int())
      val_groundtruth[{{st,st+pred:size(1)-1}, {}}]:copy(ans[{{},{2,11}}]:int())
      for k=1,pred:size(1) do 
         val_answertypes[st+k-1] = ans_type[k]
      end
      st = st + pred:size(1)
   end

   calculate_score(dt, iter, val_predictions:clone(), val_groundtruth:clone(), val_answertypes)
end

calculate_score = function(dt, iter, pred, ans, ans_type)

   local work = function(iter, pred, ans, ans_type)
   
      local accQA = {}
      local accAnsType = {}
      for i=1, pred:size(1) do 
         local gtAcc = {}
         for j=1, 10 do
            gtAnsDatum = ans[i][j]
            local otherGTAns = {}
            for k=1, 10 do
               if k ~= j then
                  table.insert(otherGTAns, ans[i][k])
               end
            end
            local matchingAns = {}
            for _,v in pairs(otherGTAns) do
               if v == pred[i] then
                  table.insert(matchingAns, v)
               end
            end
            local acc = math.min(1, #matchingAns/3)
            table.insert(gtAcc, acc)
         end
      
         local s = 0
         for _,v in pairs(gtAcc) do
            s = s + v
         end
         avgGTAcc = s/#gtAcc
      
         table.insert(accQA, avgGTAcc)
      
         accAnsType[ans_type[i]] =  accAnsType[ans_type[i]] and  accAnsType[ans_type[i]] or {} 
         table.insert(accAnsType[ans_type[i]], avgGTAcc)
      end
      
      local s = 0
      for _,v in pairs(accQA) do
         s = s + v
      end
      local type_acc = {}
      for ansType,arr in pairs(accAnsType) do
         type_acc[ansType] = type_acc[ansType] and type_acc[ansType] or 0
      
         local s = 0
         for _,v in pairs(arr) do
            s = s + v
         end
         type_acc[ansType] = type_acc[ansType] + s/(#arr/.9246)
      end

      local overall = 100*s/(#accQA/.9246)


      return iter, overall, type_acc 
   end

   local after_work = function(iter, overall, type_acc)
      strf = string.format

      paths.mkdir(opt.log_dir .. opt.version .. '/')

      local acc_log = io.open(opt.log_dir .. opt.version .. '/acc_'..dt..'.csv', 'a')
      acc_log:write(os.date('%Y-%m-%d_%H:%M:%S') .. '\t')
      acc_log:write(iter .. '\t')
      acc_log:write((iter/nepo) .. '\t')
      acc_log:write(strf('%.2f\t', overall))
      if opt.ans_type == 'all' then
         acc_log:write(strf('%.2f\t', 100*type_acc['yes/no']))
         acc_log:write(strf('%.2f\t', 100*type_acc['number']))
         acc_log:write(strf('%.2f\n', 100*type_acc['other']))
      else
         acc_log:write(strf('%.2f\n', 100*type_acc[opt.ans_type]))
      end
      acc_log:close()
      xlua.log(strf('Overall val accuracy at epoch %d %.2f%% ',iter/nepo,overall), 1)
   end

   eval_pool:addjob(work, after_work, iter, pred, ans, ans_type)
end

local get_val_loss = function()
   que, img, ans, soft = dataloader.getvalbatch(val_iter)
   val_iter = val_iter + 1

   net:evaluate()
   outputs = net:forward({que:cuda(), img:cuda()})
   net:training()

   ans = opt.criterion:find('Soft') and soft:cuda() or ans[{{},{1}}]:cuda()

   return val_criterion:forward(outputs, ans), {que,img,ans}
end

local log_losses = function(dt, iter, train_loss, val_loss)
   paths.mkdir(opt.log_dir .. opt.version .. '/')

   local loss_log = io.open(opt.log_dir .. opt.version .. '/losses_'..dt..'.csv', 'a')
   loss_log:write(os.date('%Y%m%d_%H:%M:%S')..'\t')
   loss_log:write(iter .. '\t')
   loss_log:write(strf('%d',(iter/nepo)) .. '\t')
   loss_log:write(strf('%.2f', train_loss) .. '\t')
   loss_log:write(strf('%.2f', val_loss) .. '\n')
   loss_log:close()
end

local function deep_copy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deep_copy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

local make_checkpoint = function(iter, optim_config)
   logger.info('Saving weights to '..opt.save_dir)
   paths.mkdir(opt.save_dir)
   local fn = {experiment_id}
   table.insert(fn, iter)
   table.insert(fn, (iter/nepo)..'_epoch')
   table.insert(fn,'('.. opt.version .. ')' )
   table.insert(fn, os.date('_%Y%m%d_%H%M%S')..'.t7')
   fn = table.concat(fn, '_')

   local copy = net:clone():float()
   if torch.type(copy) == 'nn.DataParallelTable' then
      copy = copy:get(1)
   end

   torch.save(opt.save_dir .. fn, deep_copy(copy):float():clearState())
   torch.save(opt.save_dir .. fn .. '_optim', optim_config)
end

local optim_config 
if opt.checkpoint:len() == 0 then
   optim_config = {
      learningRate = opt.learning_rate,
      momentum = opt.momentum,
      state = {},
   }
else
   optim_config = torch.load(opt.checkpoint .. '_optim', optim_config)
end

local val_loss = 0
local dt = os.date('%Y%m%d_%H%M%S')
local loss_avg = 0

if opt.repl then
   local vocab = dataloader.getvocab()
   local inv = util.tableinvert(vocab)
   local suffix = '_'..(opt.train_on_val and 'withVAL' or 'noVAL')
   suffix = suffix ..'_type_'..opt.ans_type
   suffix = suffix ..'_'..opt.answer_count..'.json'
   local ansmap = util.load_json(opt.ans_id2str..suffix)
   que, img, ans, img_map = dataloader.getbatch(1)
   local visdom = require'visdom'
   local plot = visdom{server = opt.plot_server, port = opt.plot_port}

   local show_imgs = function(que, img, ans, img_map, index)
      for i=1, 10 do 
         local imgfn = '/temp/ilija/fast/ms_coco_images/'.. img_map[i]:split('%.')[1]..'.jpg'
         local a = ansmap[tostring(ans[i][1][1])]
         plot:image{
            img      = image.load(imgfn),
            options  = {
               title   = imgfn .. '   ' .. util.word_ids_to_word(inv, que[i]),
               caption = index..' INDEX '.. (a and a or ' is test set')
            }
         }
      end
   end

   -- show_imgs(que,img,ans,img_map,1)
   debugger.enter()
end

paths.mkdir(opt.log_dir .. opt.version .. '/')


while iter < opt.max_iter do

   iter = iter + 1

   if iter%20 == 0 then
      collectgarbage()
   end

   if iter/nepo == 30 then
      optim_config.learningRate = opt.learning_rate/10
   elseif iter/nepo == 60 then
      optim_config.learningRate = opt.learning_rate/100
   end
   
   local _, loss = optim.adam(train, weights, optim_config, optim_config.state) 

   loss = loss[1]
   loss_avg = loss_avg ~= 0 and loss_avg*.95+loss*.05 or loss


   xlua.log(strf('"%s" Epoch %.2f%% Training Loss: %.2f', opt.version, 100.0*iter/nepo, loss_avg), 3)
   xlua.progress(iter, opt.max_iter, 4) 

   if iter/nepo > 1 and iter%(nepo/3) == 0 and eval_pool:hasjob() then
      eval_pool:synchronize()
   end

   if iter/nepo >= opt.eval_after and iter%nepo == 0 then
      if not opt.train_on_val and opt.eval then
         eval(dt, iter)
      end
      if iter/nepo >= opt.save_after then
         make_checkpoint(iter, optim_config)
      end
   end
end
