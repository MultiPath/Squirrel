# Neural Machine Translation based on Transformer

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


**Basic function**
python ez_run.py \
                --prefix [time] \  
                --gpu  <CUDA> \
                --mode <MODE> \
                --data_prefix <DATA_DIR> \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --workspace_prefix <MODEL_DIR> \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1200 \
                --inter_size 3 \
                --beam 5 \
                --alpha 1 \
                --share_embeddings \
                --tensorboard \
                --debug \

<MODE>: "data"  # build vocabulary <br>
<MODE>: "train" # train the NMT model with t2t-base setup. batch_size = 1200 x 3 <br>
<MODE>: "test"  # decoding on the dev/test set using beam-search. beam_size = 5, alpha = 1 <br>

# TODO
  Add more description about data processing/training/testing
  The torchtext requirements can be removed if possible.
