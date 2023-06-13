# SoundStorm: Efficient Parallel Audio Generation 

**Work In Progress ...**

SoundStorm is a model for efficient, non-autoregressive audio generation. SoundStorm receives as input the semantic tokens of
AudioLM, and relies on bidirectional attention and confidence-based parallel decoding to generate the tokens of a neural audio codec.

![](arch.png)

## Pre-processing and Training Scripts:

**Filelist :**
`train.txt` or `valid.txt` looks like this:
```
audiofilename1.wav|
audiofilename2.wav|
audiofilename3.wav
...
```

### Start Training:
```
python train.py --path ./data --train train.txt --ratio 2
```
**Semantic token path:** `./data/semantic_code`

**Acoustic token path:** `./data/codec_code`

**ratio:** acoustic_frame_rate / semantic_frame_rate  # Add should be a integer






## References :

* MAskGIT code : https://github.com/dome272/MaskGIT-pytorch

