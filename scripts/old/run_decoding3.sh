python run.py --prefix [time] --gpu 0 \
              --level subword \
              --batchsize 2048 \
              --beam_size 1 \
              --alpha 0.6 \
              --fast --use_alignment --positional_attention --diag \
              --use_mask \
              --mode test \
              --load_from 09.30_00.39.iwslt_subword_fast_278_507_5_2_0.079_746___ \
              #--load_from 10.06_16.13.iwslt_subword_fast_278_507_5_2_0.079_746___ \
              #--load_from 09.30_04.37.iwslt2_subword_fast_278_507_5_2_0.079_746___ \
              # --load_from 10.07_20.59.iwslt_subword_fast_278_507_5_2_0.079_746___ \

