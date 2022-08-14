#!/bin/bash

# Train language model
python run_lm.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU \
    --n_layers 2 --hidden_size 300  --embedding_dim 100 --epochs 1 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --padding_loss 0.0 --eos_loss 1.0 --verbose_level 1 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name lm

# Dual supervised learning without structure matching
python run_dsl.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name dsl-l0.1 --lm_model_dir ../data/model_slt/lm --lm2_model_dir ../data/model_slt/lm2 \

# Dual learning with structure matching
python main.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --lmbda1 0.5 --lmbda2 0.6 --lmbda3 0.2 --threshold_omega 0.3 --threshold_sigma 0.5 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name text-text-SMDL --nlg1_st 1 --nlg2_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model text-text
