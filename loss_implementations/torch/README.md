
Follow these steps to add the loss function to your torch installation.

1. add the lua file in extra/nn/
2. edit extra/nn/init.lua and add it there
4. add the cu file to extra/cunn/lib/THCUNN/
5. add the cu file to extra/cunn/lib/THCUNN/generic/
3. edit extra/cunn/lib/THCUNN/generic/THCUNN.h and add it there
6. cd extra/nn/ ; luarocks make rocks/nn-scm-1.rockspec 
7. cd extra/cunn/ ; luarocks make rocks/cunn-scm-1.rockspec

We used torch commit 5c1d3cfda8101123628a45e70435d545ae1bc771
It is very likely that you will have to modify the code if you are using different commit. 
