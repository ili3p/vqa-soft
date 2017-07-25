return {
   -- model loading and saving
   model          = './model.lua',
   eval           = true,
   log_loss_every = 350,
   save_dir       = './checkpoints/', 
   save_after     = 15, -- in iterations
   eval_after     = 1, -- in epochs 
   criterion      = 'SoftClassNLLCriterion',
   checkpoint     = '',
   start_iter     = 1,
   -- model definitions
   size_multi      = 1500,
   size_common     = 1500,
   size_classifier = 3000,
   size_image      = 2048,
   size_rnn        = 2400,
   size_rnn_layer  = 1,
   -- training related
   max_iter       = 432000, -- about 70 epochs
   learning_rate  = 1e-4,
   momentum       = 0.9,
   batch_size     = 64,
   val_batch_size = 96,
   gradclip       = 0, -- 0 means disabled
   dropout        = 0.5,
   rnn_dropout    = 0.0,
   activation     = 'Tanh',
   img_activation = 'Tanh',
   glimpse        = 2,
   showprogress  = true,
   repl          = false,
   -- logging related
   log_dir        = './logs/',
   log_level      = 3, -- 1:trace, 2:debug, 3:info, 4:warn, 5:error, 6:fatal
   log_to_console = true,
   log_to_file    = true,
   version = 'vanila',
   -- plot related
   showplot      = false,
   plot_every    = 1000,
   plot_server   = 'http://localhost',
   plot_port     = 8097,
   -- misc
   num_gpus = 1,
   rnd_seed = 139, 

   -- data related
   
   -- number of dataloading threads
   dataworkers = 4,
   buffer_size = 8,

   val_dataworkers = 2,
   val_buffer_size = 8,

   img_dir = '../../resnet_features/',

   que_train   = '../../vqa2_data/v2_OpenEnded_mscoco_train2014_questions.json',
   que_val     = '../../vqa2_data/v2_OpenEnded_mscoco_val2014_questions.json',
   -- change this to point to test std json when needed
   que_test    = '../../vqa2_data/v2_OpenEnded_mscoco_test-dev2015_questions.json',

   ans_train    = '../../vqa2_data/v2_mscoco_train2014_annotations.json',
   ans_val      = '../../vqa2_data/v2_mscoco_val2014_annotations.json',


   -- train on trainval or just train
   train_on_val = false,
   -- left aligned questions for MLP and right for RNN language model
   left_aligned = false,

   -- at least how many times should the word appear in train set to be in vocab
   word_freq = 1,
   que_len = 10,
   ans_type = 'all', -- yes-no number other 
   ans_aug = false,

   -- outputs file prefix, the full filename depends on the options
   train_questions    = './data/train_questions',
   train_tid2qid      = './data/train_tid2qid',
   train_tid2img      = './data/train_tid2img',

   test_questions     = './data/test_questions',
   test_tid2qid       = './data/test_tid2qid',
   test_tid2img       = './data/test_tid2img',
   test_tid2anstype   = './data/test_tid2anstype', 

   vocab          = './data/vocab', 

   lookuptable    = './data/lookup',

   -- which word2vec to use
   word2vec = '../../utils/fastText.hash',
   vec_len  = 300,

   answer_count = 3000,

   -- outputs file prefix, the full filename depends on the options
   qid2ans      = './data/qid2ans',
   qid2type     = './data/qid2type',
   qid2anstype  = './data/qid2anstype',
   ans_id2str   = './data/ans_id2str',

   -- outputs file prefix, the full filename depends on the options
   train_answers = './data/train_answers',
   test_answers = './data/test_answers',
}

