#Demo code for training

# g_blocks: number of layers in the top and middle sub-networks
# m_blocks: number of layers in the bottom sub-networks
# n_feats : number of feature map channels in the bottom sub-networks
# ext     : how to load training data 
# testbin : how to load testing data (for fair comparison with previous approaches, we use matlab to generate noisy images with random seed 0, and load the noisy matrix at test time)


python main.py --model sgndn3 --g_blocks 2 --m_blocks 2 --act relu --noise_sigma 50 --n_feats 32 -- lr 1e-4 --gamma 0.1 -- lr_decay 100 --test_every 5000 --patch 512 --batch 4 --ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68 --testbin True --save ./Checkpoints/sgndn3_RELU_Sigma50_g2_m2_f32_p512_b4





# Demo code for testing
python main.py --test_only --model sgndn3 --g_blocks 2 --m_blocks 2 --act relu --noise_sigma 50 --n_feats 32 -- lr 1e-4 --gamma 0.1 -- lr_decay 100 --test_every 5000 --patch 512 --batch 4 --ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68 --testbin True --test_only --pre_train ./Checkpoints/sgndn3_RELU_Sigma50_g2_m2_f32_p512_b4/model/model_best.pt
