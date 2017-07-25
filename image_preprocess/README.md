Follow the instructions at: [https://github.com/akirafukui/vqa-mcb/tree/master/preprocess](https://github.com/akirafukui/vqa-mcb/tree/master/preprocess) to obtain image tensor representations. Use the default configuration which should give you `2048x14x14` dimensional tensor for each image.  
Modify the file `extract_resnet.py` at line 70 to save the tensors as uncompress numpy array or run the script `uncompress.py` located in this directory to uncompress the arrays. 
Then run `convert_to_torch.lua` to convert the numpy tensors to compressed torch tensors. 

Why do we obtain the image tensors in this convoluted way? Well, for some reason the pre-trained ResNet-152 caffe model produces image features that are more sparse and thus when compressed take up about three times less space then the pre-trained ResNet-152 torch model.
 The train+val features obtained from caffe model are 28.3GB and can be easily cached in 64GB RAM memory which brings the dataloading down to 1ms. 
On the other hand, the same features obtained from the torch model are 83GB and cannot be fully cached in RAM, so the model often needs to read them from disk which makes the dataloading a performance bottleneck.

If you still want to use the torch features for some reason, then you can use the `extract_whole_image_features_compressed.lua` file to do so. 
The only advantage is that the torch code is optimized, multi-threaded code that can use 3 GPUs to extract all features in less than one hour. 
On the other hand, the caffe code on 3 GPUs takes about 15 hours. 


In the end you should have `resnet_features` directory under the main `vqa-soft` directory, i.e.:
```
vqa-soft/resnet_features/
|-- test2015/
|-- train2014/
`-- val2014/

```

