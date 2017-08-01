I'm not familiar with pytorch framework architecture and I probably have modified more files than I needed. 
I have used version v0.1.12 commit `ccd5f4dbfcf8ba4d5903a5b57f0200742833dd54`. 

`git status` says I have modified the following files: 

Note that I didn't add new loss function as in torch, but just modified the standard NLL criterion to behave like soft NLL. You can rename the loss function if you want to keep both criterions.

```
On branch master
Your branch and 'origin/master' have diverged,
and have 3 and 486 different commits each, respectively.
  (use "git pull" to merge the remote branch into yours)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   torch/lib/THC/generic/THCTensor.c
	modified:   torch/lib/THCUNN/ClassNLLCriterion.cu
	modified:   torch/lib/THCUNN/generic/ClassNLLCriterion.cu
	modified:   torch/lib/THCUNN/generic/THCUNN.h
	modified:   torch/lib/THNN/generic/THNN.h
	modified:   torch/lib/THNN/init.c
	modified:   torch/nn/_functions/thnn/auto.py
	modified:   torch/nn/functional.py
	modified:   torch/nn/modules/__init__.py
	modified:   torch/nn/modules/loss.py
	modified:   torch/utils/serialization/read_lua_file.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	torch/csrc/generated/
	torch/lib/THCUNN/SoftClassNLLCriterion.cu
	torch/lib/THCUNN/generic/SoftClassNLLCriterion.cu
	torch/lib/THNN/generic/SoftClassNLLCriterion.c

no changes added to commit (use "git add" and/or "git commit -a")
```

You can find the modified and the new files in the respective folders. The new files are actually not needed since in the end I just modified the normal ClassNLLCriterion to behave as SoftClassNLLCriterion. 
