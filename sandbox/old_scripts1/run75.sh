python run.py --prefix [time] --gpu 1 --eval-every 100 --fast --tensorboard \
                                    --level subword \
                                    --use_mask --diag --positional_attention \
                                    --use_alignment \
                                    --load_from 09.30_04.37.iwslt2_subword_fast_278_507_5_2_0.079_746___ \
                                    --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                                    --dataset iwslt2 \
                                    --disable_lr_schedule \
                                    --temperature 1 \
                                    --debug \
                                    --teacher_discriminator --st \
                                    #--reverse_kl \
                                    #--st \
                                    #--entropy \
                                    #--teacher_discriminator --st \
                                    # --debug \
                                    #--share_encoder --finetune_encoder \
                                    #--disable_lr_schedule \
                                    # --debug \
                                    # --no_source
                                    # --positional_attention \
                                    #--attention_discrimination --debug
                                    #--load_from 09.27_17.51.iwslt_subword_fast_278_507_5_2_0.079_746___ \
