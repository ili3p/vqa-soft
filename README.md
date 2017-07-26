Accompanying code for "[A Simple Loss Function for Improving the Convergence and Accuracy of Visual Question Answering Models](paper.pdf)" CVPR 2017 VQA workshop paper.

The repo contains code for reproducing the paper's experiments and efficient GPU implementation of the proposed loss function for torch, pytorch, and caffe. 

### Requirements

To run the experiments you would first need to install torch from [https://github.com/torch/distro/](https://github.com/torch/distro/). We used torch version from commit `5c1d3cfda8101123628a45e70435d545ae1bc771` but later versions probably would work too.

After installing torch you will need to install the following useful lua libraries:

C data structures for torch [https://github.com/torch/tds](https://github.com/torch/tds), so we can allocate data in C memory space instead of lua's and thus avoid lua's memory limit and garbage collection.

`luarocks install tds`

RNN lib for torch [https://github.com/Element-Research/rnn](https://github.com/Element-Research/rnn) for mask zero lookuptable and other useful modules.

`luarocks install rnn`

threads for lua [https://github.com/torch/threads](https://github.com/torch/threads) for multi-threaded code.

`luarocks install threads`

The following libraries are required but you can modify the code and still run the experiments. However we recommend installing them anyway.

fb-debugger a source-level debugger for lua

Follow the install instructions at [https://github.com/facebook/fblualib/blob/master/fblualib/debugger/README.md](https://github.com/facebook/fblualib/blob/master/fblualib/debugger/README.md).

OptNet - Reducing memory usage in torch neural nets [https://github.com/fmassa/optimize-net](https://github.com/fmassa/optimize-net).

`luarocks install optnet`

Visdom for visualization [https://github.com/facebookresearch/visdom](https://github.com/facebookresearch/visdom).

`pip install visdom`

`luarocks install visdom`
