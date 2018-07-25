python run.py   --prefix [time] --gpu 4 \
                --dataset wmt16-roen --language roen --load_vocab \
                --level subword \
                --load_from 10.22_01.24.wmt16-roen_subword_fast_512_512_6_8_0.100_16000___ \
                --teacher 10.17_21.19.wmt16-enro_subword_512_512_6_8_0.100_16000___  \
                --use_mask  --diag --positional_attention \
                --use_wo --share_embeddings \
                --preordering --use_posterior_order --fertility_only \
                --remove_eos \
                --fertility_mode argmax \
                --mode test \
                --batchsize 20000 \
                --fast \
                --test_set test \