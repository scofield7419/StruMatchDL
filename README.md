# StruMatchDL

**The pytroch implementation of the ICML 2022 paper [Matching Structure for Dual Learning](https://proceedings.mlr.press/v162/fei22a.html)**

------------

# Overview

>  Many natural language processing (NLP) tasks appear in dual forms, which are generally solved by dual learning technique that models the dualities between the coupled tasks. In this work, we propose to further enhance dual learning with structure matching that explicitly builds structural connections in between. Starting with the dual text↔text generation, we perform duallysyntactic structure co-echoing of the region of interest (RoI) between the task pair, together with a syntax cross-reconstruction at the decoding side. We next extend the idea to a text↔non-text setup, making alignment between the syntactic-semantic structure. Over 2*14 tasks covering 5 dual learning scenarios, the proposed structure matching method shows its significant effectiveness in enhancing existing dual learning. Our method can retrieve the key RoIs that are highly crucial to the
task performance. Besides NLP tasks, it is also revealed that our approach has great potential in facilitating more non-text↔non-text scenarios.



--------------

# Requirements

```bash
conda create --name dual-scsg python=3.8
pip install -r requirements.txt
```



# Datasets


Step 1. Prepare the relevant dataset, and put it under data/[name]/.
    - WMT14(EN-DE)
    - WMT14(EN-FR)
    - ParaNMT
    - QUORA

Step 2. Obtain the parse trees for sentences via CoreNLP.

Step 3. Run the system.



# Usage


## Parsing sentences

- Downloading the CoreNLP:

```bash
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
```

- Deploy CoreNLP service

```bash
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8083 -timeout 15000 > 1.log 2>&1 &
```

- Parsing sentences for constituency trees

```bash
python data/parsing.py
```



## Train language model
If two coupled tasks are with OOD texts (e.g., NMT), please train two separate LM.

```bash
python run_lm.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU \
    --n_layers 2 --hidden_size 300  --embedding_dim 100 --epochs 1 --batch_size 32 --optimizer Adam \
    --learning_rate 0.001 --padding_loss 0.0 --eos_loss 1.0 --verbose_level 1 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name lm
```

## Dual supervised learning for Bakbone systems, without structure matching


```
python run_dsl.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 \
    --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 \
    --hidden_size 200  --embedding_dim 50 --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam \
    --learning_rate 0.001 --teacher_forcing_ratio 0.9 --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 \
    --is_load 0 --train 1 --dir_name dsl-l0.1 --lm_model_dir ../data/model_slt/lm --lm2_model_dir ../data/model_slt/lm2 \
```

## Dual learning with structure matching

```
python main.py \
    --data_dir ../data/ --dataset [name] --fold_attr 1 --vocab_size 500 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 \
    --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --cell GRU --n_layers 1 --hidden_size 200 \
    --lmbda1 0.5 --lmbda2 0.6 --lmbda3 0.2 --threshold_omega 0.3 --threshold_sigma 0.5 \
    --embedding_dim 50  --bidirectional 1  --epochs 10 --batch_size 64 --optimizer Adam --learning_rate 0.001 --teacher_forcing_ratio 0.9 \
    --tf_decay_rate 1.0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --save_epochs 1 --is_load 0 --train 1 \
    --dir_name text-text-SMDL --nlg1_st 1 --nlg2_st 1 --primal_supervised 1 --dual_supervised 1 \
    --schedule joint --supervised 1 --model text-text
```


# Citation

```
@inproceedings{MSDual2022ICML,
  author    = {Hao Fei and
               Shengqiong Wu and
               Yafeng Ren and
               Meishan Zhang},
  title     = {Matching Structure for Dual Learning},
  booktitle = {Proceedings of the International Conference on Machine Learning, {ICML}},
  pages     = {6373--6391},
  year      = {2022},
}
```



# License

The code is released under Apache License 2.0 for Noncommercial use only. 
