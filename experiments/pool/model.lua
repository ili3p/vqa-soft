local nninit = require 'nninit'
local M = {}

M.new = function(opt, ltbl)


   local lang = nn.Sequential()
      :add(ltbl)
      :add(nn.Dropout(opt.dropout, true))
      :add(cudnn[opt.activation](true)) 
      :add(cudnn.LSTM(opt.vec_len, opt.size_rnn, opt.size_rnn_layer, true)
         :init('weight', nninit.uniform, -0.08, 0.08)) 
      :add(nn.Select(2, -1))
      :add(cudnn.BatchNormalization(opt.size_rnn))

   local vision = nn.Sequential()
      :add(nn.Dropout(opt.dropout, true))
      :add(cudnn.SpatialConvolution(opt.size_image, opt.size_common, 1, 1)
         :init('weight', nninit.xavier, {dist='normal', gain=opt.img_activation:lower()}))
      :add(cudnn[opt.img_activation](true))

   local attention = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.Dropout(opt.dropout, true))
            :add(nn.Linear(opt.size_rnn, opt.size_common)
            :init('weight', nninit.xavier, {dist='normal', gain=opt.activation:lower()}))
            :add(cudnn[opt.activation](true))
            :add(nn.Replicate(14*14, 3))  
            :add(nn.Reshape(opt.size_common, 14, 14, true))
            )
         :add(vision))
      :add(nn.CMulTable())
      :add(cudnn.SpatialConvolution(opt.size_common, opt.glimpse, 1, 1)
         :init('weight', nninit.xavier, {dist='normal', gain=opt.img_activation:lower()}))
      :add(nn.View(opt.glimpse, 14*14))
      :add(nn.SplitTable(2)) -- split the attentions in separate tables
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.SoftMax())
         )
         :add(nn.Sequential()
            :add(nn.SoftMax())
         )
      )

   local att_applier = nn.Sequential()
            :add(nn.NarrowTable(2, 2))
            :add(nn.ConcatTable()
               :add(nn.Sequential()
                  :add(nn.ParallelTable()
                     :add(nn.Identity())
                     :add(nn.Sequential()
                        :add(nn.SelectTable(1))
                        :add(nn.View(14*14, 1))
                     )
                  )
                  :add(nn.MM())
                  :add(nn.Squeeze())
                  :add(nn.Dropout(opt.dropout, true))
                  :add(nn.Linear(opt.size_image, opt.size_multi)
                  :init('weight', nninit.xavier, {dist='uniform', gain=opt.activation:lower()}))
                  :add(cudnn[opt.activation](true))
               )
               :add(nn.Sequential()
                  :add(nn.ParallelTable()
                     :add(nn.Identity())
                     :add(nn.Sequential()
                        :add(nn.SelectTable(2))
                        :add(nn.View(14*14, 1))
                     )
                  )
                  :add(nn.MM())
                  :add(nn.Squeeze())
                  :add(nn.Dropout(opt.dropout, true))
                  :add(nn.Linear(opt.size_image, opt.size_multi)
                  :init('weight', nninit.xavier, {dist='normal', gain=opt.activation:lower()}))
                  :add(cudnn[opt.activation](true))
               )
            )
            :add(nn.JoinTable(2))

   local model = nn.Sequential()
      :add(nn.ParallelTable()
         :add(lang) -- from question words to LSTM
         :add(nn.Identity()) -- no change to the image yet
      )
      :add(nn.ConcatTable() -- feed {que, img} to all members
         :add(nn.SelectTable(1)) -- pass only the question to be later used
         :add(nn.Sequential()
            :add(nn.SelectTable(2)) -- pass only the image to be later used
            :add(nn.Reshape(opt.size_image, 14*14, true))
         )
         -- calculate the attention
         :add(attention) -- input {que,img}, output: {att_w1,att_w2,...att_wN} 
      ) -- the output is {que, img, {att_w1,att_w2, ...}} 
      :add(nn.ConcatTable()
         :add(nn.Sequential() -- transform the question before multiplication
            :add(nn.SelectTable(1)) -- select just the que
            :add(nn.Dropout(opt.dropout, true))
            :add(nn.Linear(opt.size_rnn, opt.glimpse*opt.size_multi) -- TODO think what to do with multiple glimpse
               :init('weight', nninit.xavier, {dist='normal', gain=opt.activation:lower()}))
            :add(cudnn.BatchNormalization(opt.glimpse*opt.size_multi))
            :add(cudnn[opt.activation](true))
         ) 
         :add(att_applier)  -- apply the attention to the image
      ) -- the output now is {que, att_img}
      -- multiply or concat the two and pass them to the classifier
      :add(nn.CMulTable())  -- final multiplicaiton of the question and attended image
      -- classifier starts
      :add(nn.Dropout(opt.dropout, true))
      :add(nn.Linear((opt.glimpse*opt.size_multi), opt.glimpse*opt.size_multi)
         :init('weight', nninit.xavier, {dist='normal', gain=opt.activation:lower()}))-- 'cls output'
      :add(cudnn.BatchNormalization(opt.size_multi*opt.glimpse))
      :add(cudnn[opt.activation](true))
      :add(nn.Dropout(opt.dropout, true))
      :add(nn.Linear(opt.size_multi*opt.glimpse, opt.answer_count)
      :init('weight', nninit.xavier, {dist='normal', gain='linear'}))


   return model
end

return M
