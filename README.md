# Neural Machine Translation based on Transformer

**Major Reference**:

Vaswani, Ashish, et al. "Attention is all you need." 
Advances in Neural Information Processing Systems. 2017.
(http://papers.nips.cc/paper/7181-attention-is-all-you-need)

**Requirements**: <br>
  Python 3.6  <br>
  pytorch >= 0.4 <br>
  torchtext >= 0.3 (installed from the source) <br>
  tqdm <br>
  tensorflow-cpu, tensorbaordX <br> (if need to use tensorboard for visualization)

**Dataset** <br>
We provided the proprocessed (BPE) parallel corpora at <br>
https://www.dropbox.com/sh/p5b6m14is8hd4rn/AAAW5M6ddaiwjd5DbugNPmdEa?dl=0

You can simply download all the datasets and put them to your <DATA_DIR>.<br>

*currently included* <br>
WMT16 RO-EN <br>


**Basic functions** <br>
*build vocabulary*
```shell
python ez_run.py \
                --prefix [time]  \
                --mode data \
                --data_prefix <DATA_DIR> \
                --dataset "wmt16" \
                --src "ro" \
                --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --char # (optional) if use, build the character-level vocabulary.
```

*train* <br>
train the NMT model with basic Transformer <br>
Due to pytorch limitation, the multi-GPU version is still under constration.
In order to achieve large batch size on single GPU, we used a trick to perform multiple passes (--inter_size) before one update to the parametrs which, however, hurts the training efficiency.

```shell
python ez_run.py \
                --prefix [time]  \
                --gpu  <GPU_ID> \
                --mode train \
                --data_prefix <DATA_DIR> \
                --dataset "wmt16" \
                --src "ro" \
                --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --workspace_prefix <MODEL_DIR> \  # where you save models, states, and logs.
                --params "t2t-base" \  # d_model=512, d_ff=2048, n_h=8, n_l=6, warmup=16000
                --eval_every 500 \
                --batch_size 1200 \    # the actual batch-size=1200*3=3600
                --inter_size 3 \      
                --label_smooth 0.1 \   # (optional) if > 0.1, use label-smoothing when training
                --share_embeddings \   # (optional) if use, source and target share the same vocabulary.
                --char \               # (optional) if use, train the character-level instead of bpe-level. 
                --causal_enc \         # (optional) if use, encoder uses causal attention (unidirectional)
                --encoder_lm \         # (optional) if use, additional LM loss over encoder (requires "--causal_enc")
                --tensorboard \
                --debug                # (optional) if use, no saving tensorboard.
```


# TODO
  Add more description about data processing/training/testing
  The torchtext requirements can be removed if possible.
