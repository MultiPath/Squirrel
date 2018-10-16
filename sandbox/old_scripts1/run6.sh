python run.py --prefix [time] --gpu 5 --eval-every 500 --fast --tensorboard \
                                    --level subword \
                                    --use_mask --diag --positional_attention \
                                    --use_alignment \
                                    --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                                    --share_encoder --finetune_encoder \
                                    --seq_dist \
                                    --noisy \
                                    # --debug \
                                    # --disable_lr_schedule \
                                    # --debug \
                                    # --no_source
                                    # --positional_attention \
                                    #--attention_discrimination --debug
                                    #--load_from 09.27_17.51.iwslt_subword_fast_278_507_5_2_0.079_746___ \
