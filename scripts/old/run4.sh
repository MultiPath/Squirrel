python run_iterative.py --prefix [time] --gpu 6 --eval-every 500 --fast --tensorboard \
                                    --level subword \
                                    --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                                    --share_encoder --finetune_encoder \
                                    --use_mask --diag \
                                    --use_alignment \
                                    --iterations 3 \
                                    --debug
                                    #--load_from 09.27_17.51.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                                    #--debug
