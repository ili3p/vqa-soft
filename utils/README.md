This directory contains utility code. The most important are the `fastText.hash` and `glove.hash` which are `tds.Hash` mapping each word in the whole vqa2 dataset to a [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) and to a [glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) vector.


Folders `extract_fastText` and `extract_glove` contain the scripts for generating these hashes from the raw vector files (downloaded from the links above).

This directory should contain:

```
../utils/
|-- extract_fastText
|   |-- vocab_to_fastText.lua
|   `-- wiki.en.vec
|-- extract_glove
|   |-- glove.840B.300d.txt
|   `-- vocab_to_glove.lua
|-- fastText.hash
|-- glove.hash
|-- logger.lua
|-- README.md
|-- repl.lua
`-- util.lua

2 directories, 10 files

```

Note that you can omit the `extract*` folders if using the pregenerated word maps.
