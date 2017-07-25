require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
require 'hdf5'
require 'sys'

cudnn.benchmark = true 
cudnn.fastest = true 

debugger = require'fb.debugger'
local cjson = require('cjson') 
local t = require 'transforms'
local tds = require 'tds'
local optnet = require 'optnet'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

opt = {
   input       = '/temp/ilija/VQA/datasets/vqa/train_val_unique_img_fn.json',  
   imgroot     = '/temp/ilija/fast/ms_coco_images/',
   set         = 'trainval', -- train, trainval, val, testdev, test
   model       = '../resnet_models/resnext_101_64x4d.t7',
   outdir      = '../resnet_features/',
   imgsize     = 448,
   region      =  14,
   bsz         = 100,
   nthreads    =  10, 
   rnd_seed    = 139,
   num_gpus    =   1,
}
for k, v in pairs(opt) do 
   opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] 
end
opt.outfilename = opt.outdir .. 'avg_region_ResNeXt50_'..opt.imgsize..'_'..opt.set..'.t7'
opt.mappingfn =  opt.outdir .. 'map_avg_region_ResNeXt50_'..opt.imgsize..'_'..opt.set..'.hash'
print(opt)

torch.manualSeed(opt.rnd_seed)
cutorch.manualSeed(opt.rnd_seed)
torch.setnumthreads(opt.nthreads)

print('Reading file:', opt.input)
local file = io.open(opt.input, 'r')
local text = file:read()
file:close()
jsondata = cjson.decode(text)

local imgs = tds.Vec()

for k, v in pairs(jsondata) do
    imgs:insert(k)
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
   t.Scale(opt.imgsize),
   t.ColorNormalize(meanstd),
   t.CenterCrop(opt.imgsize),
}

local initfn = function()
    require 'torch'
    require 'image'
    local tds = require 'tds'
end

local nimgs = #imgs
features = torch.FloatTensor(nimgs, 2048)
mapping = tds.Hash()

local model = torch.load(opt.model)
-- -- remove ReLUs
-- for i=1,3 do
--    model.modules[8].modules[i]:remove(3) 
-- end
-- adapt the model to larger image
model.modules[9] = cudnn.SpatialAveragePooling(opt.region,opt.region,1,1) 
model:remove(11)
print(model)
model = model:cuda()

local sampleInput = torch.zeros(4,3,opt.imgsize,opt.imgsize):type('torch.CudaTensor')
optnet.optimizeMemory(model, sampleInput, {mode='inference', inplace=true, reuseBuffers=true, removeGradParams=true})


if opt.num_gpus > 1 then
   net = nn.DataParallelTable(1) 
      :add(model, {1,2})
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


index = 1

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
            if opt.set:find('test') then
               dir = '/test2015/'
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

        if #batchmap > 0 then
           return batch[{{1,#batchmap}}], batchmap
        else 
           return nil
        end
    end
    
    -- executed on the main thread
    local endcallback = function(batch, batchmap)
       
       if batch then
           local out = net:forward(batch:cuda())
           features[{{index, index + out:size(1) - 1}}]:copy(out)

           for k,v in pairs(batchmap) do
              mapping[v] = index
              index = index + 1
           end

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

while index < nimgs do 

    if batch_index < nimgs then
      for k=1, opt.nthreads do
          addjob()
          batch_index = batch_index + opt.bsz
      end
   end

    while index < nimgs do
        pool:dojob()
        collectgarbage()
        xlua.progress(index, nimgs)
    end
end

print('saving')
torch.save(opt.mappingfn, mapping)
torch.save(opt.outfilename, features)
