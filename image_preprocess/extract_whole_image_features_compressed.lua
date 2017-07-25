require 'nn'
require 'cudnn'
cudnn.benchmark = true 
cudnn.fastest = true 
require 'cunn'
require 'image'
require 'torchzlib'

local cjson = require('cjson') 
local t = require './transforms.lua'
local tds = require 'tds'
local threads = require 'threads'
optnet = require'optnet'
util = require '../utils/util.lua'
threads.Threads.serialization('threads.sharedserialize')

opt = {
   input           = './train_val_unique_img_fn.json' 

   -- folder containing train2014, val2014, and test2015 directories 
   imgroot         = '/temp/ilija/fast/ms_coco_images/',   
   model           = '../resnet_models/resnet-152.t7',
   outdir          = '../resnet_features/trainval2014/', 
   imgsize         = 448,
   region          =  14,
   bsz             = 50,
   nthreads        =  10, 
   num_gpus        =   3,
   rnd_seed        = 139,
   compress_factor =   0,
}
for k, v in pairs(opt) do 
   opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end
print(opt)
paths.mkdir(opt.outdir)

torch.manualSeed(opt.rnd_seed)
cutorch.manualSeed(opt.rnd_seed)
torch.setnumthreads(opt.nthreads)

print('Reading file:', opt.input)
local file = io.open(opt.input, 'r')
local text = file:read()
file:close()
jsondata = cjson.decode(text)

local done = tds.Hash()
for line in io.lines('all.txt') do
   done[line:split('%.')[1] .. '.jpg'] = true
end

local imgs = tds.Vec()
for k, v in pairs(jsondata) do
   -- if not done[k] then
       imgs:insert(k)
    -- end
end
print('Total images:', #imgs)
local meanstd = {
     mean = {
    0.48462227599918,
    0.45624044862054,
    0.40588363755159,
  },
  std = {
    0.22889466674951,
    0.22446679341259,
    0.22495548344775,
  }
}

local transform = t.Compose{
   -- t.Scale(opt.imgsize),
   t.ScaleEqual(opt.imgsize),
   -- t.ColorNormalize(meanstd),
   t.CenterCrop(opt.imgsize),
}

img = image.load(opt.imgroot .. 'train2014/COCO_train2014_000000009064.jpg')
print(img:mean())
img = transform(img)
print(img:mean())
image.save('img.jpg', img)

local initfn = function()
    require 'torch'
    require 'image'
    local tds = require 'tds'
    torch.setnumthreads(1)
end

local nimgs = #imgs

print(nimgs)

local model = torch.load(opt.model)

model:remove(11)
model:remove(10)
model:remove(9)
model.modules[8].modules[3]:remove(3)
model = model:cuda()
print(model)

local sample = torch.zeros(2,3, opt.imgsize, opt.imgsize):cuda()
optnet.optimizeMemory(model, sample, {mode='inference', inplace=true, reuseBuffers=true, removeGradParams=true})
sample = nil

if opt.num_gpus > 1 then
   net = nn.DataParallelTable(1) 
      :add(model, util.range(opt.num_gpus))
      :threads(function()
         local cudnn = require 'cudnn'
         cudnn.fastest = true 
         cudnn.benchmark = true
      end)
else
   net = model 
end
net = net:cuda()
net:evaluate()

local compress_pool = threads.Threads(4, function() torch.setnumthreads(1) require'torch' require'torchzlib' end)
-- a thread job
add_compress_job = function(out, batchmap, compress_factor, outdir)
    
    -- executed on the thread's thread
    local loadimgs = function (out, batchmap, compress_factor, outdir)
       for k,v in pairs(batchmap) do
          local ct = torch.CompressedTensor(out[k], compress_factor)
          torch.save(outdir..v..'.t7z', ct)
       end
    end
    
    compress_pool:addjob(loadimgs, nil, out, batchmap, compress_factor, outdir)
end

-- use pool to load images from disk, otherwise 70% of time is spent there
local pool = threads.Threads(opt.nthreads, initfn)
-- a thread job
addjob = function()
    
    -- executed on the thread's thread
    local loadimgs = function (imgs, opt, transform, batch_index)
        collectgarbage()
        local batch = torch.FloatTensor(opt.bsz, 3, opt.imgsize, opt.imgsize)
        local batchmap = {}
        local thind = 1
        for ind=batch_index, batch_index + opt.bsz - 1 do 
            if not imgs[ind] then break end
            local dir
            if imgs[ind]:find('test') then
               dir = 'test2015/'
            else
               dir = (imgs[ind]:match('train') and '/train' or '/val') 
               dir = dir .. '2014/'
            end
            local img = image.load(opt.imgroot .. dir  .. imgs[ind], 3, 'float')
            img = transform(img)
            batch[thind]:copy(img)
            batchmap[thind] = imgs[ind]
            thind = thind + 1
        end

        if #batchmap == 0 then
           return nil
        else 
           return batch[{{1,#batchmap}}], batchmap
        end
    end
    
    -- executed on the main thread
    local endcallback = function(batch, batchmap)

       if batch then 
          local out = net:forward(batch:cuda()):float()

          add_compress_job(out:clone(), batchmap, opt.compress_factor, opt.outdir)

          completed = completed + #batchmap
          xlua.progress(completed, nimgs)
          
           -- -- add new jobs 
           if batch_index < nimgs then
               addjob()
               batch_index = batch_index + opt.bsz
           end
       end
    end
    
    pool:addjob(loadimgs, endcallback, imgs, opt, transform, batch_index)
end

batch_index = 1
completed = 0

while batch_index < nimgs do 

    if batch_index < nimgs then
      for k=1, opt.nthreads do
          addjob()
          batch_index = batch_index + opt.bsz
      end
   end

    while completed < nimgs do
        pool:dojob()
    end
end

print('Still writing')
compress_pool:synchronize()
