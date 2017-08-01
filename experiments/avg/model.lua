local nninit = require 'nninit'
local M = {}

M.new = function(opt, ltbl)

   local model = nn.Sequential()
         :add(nn.ParallelTable()
            :add(nn.Sequential()
               :add(ltbl)
               :add(nn.Dropout(opt.dropout, true))
               :add(cudnn[opt.activation](true)) 
               -- :add(nn.SeqGRU(opt.vec_len, opt.size_rnn)
               :add(cudnn.GRU(opt.vec_len, opt.size_rnn, opt.size_rnn_layer, true)
                  :init('weight', nninit.uniform, -0.08, 0.08)) 
               :add(nn.Select(2, -1))
               :add(cudnn.BatchNormalization(opt.size_rnn))
            )
            :add(nn.Sequential()
               :add(cudnn.SpatialAveragePooling(14,14,1,1))
               :add(nn.View(opt.size_image))
               :add(cudnn.BatchNormalization(opt.size_image))
               :add(nn.Dropout(opt.dropout, true))
               :add(nn.Linear(opt.size_image, opt.size_rnn*2)
               :init('weight', nninit.xavier, {dist='normal', gain=opt.img_activation:lower()}))
               :add(cudnn.BatchNormalization(opt.size_rnn*2))
               :add(cudnn[opt.img_activation](true))
               :add(nn.Dropout(opt.dropout, true))
               :add(nn.Linear(opt.size_rnn*2, opt.size_rnn)
               :init('weight', nninit.xavier, {dist='normal', gain=opt.img_activation:lower()}))
               :add(cudnn.BatchNormalization(opt.size_rnn))
               :add(cudnn[opt.img_activation](true))
            )
         )
         :add(nn.CMulTable())
         :add(cudnn.BatchNormalization(opt.size_rnn))
         :add(nn.Dropout(opt.dropout, true))
         :add(nn.Linear(opt.size_rnn, opt.size_classifier)
         :init('weight', nninit.xavier, {dist='normal', gain=opt.activation:lower()}))
         :add(cudnn.BatchNormalization(opt.size_classifier))
         :add(cudnn[opt.activation](true))  
         :add(nn.Dropout(opt.dropout, true))
         :add(nn.Linear(opt.size_classifier, opt.answer_count)
         :init('weight', nninit.xavier, {dist='normal', gain='linear'}))

     -- model.modules[1].modules[1].modules[4]:maskZero(1)

   return model
end

return M
