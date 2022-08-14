# Matching Structure for Dual Learning



### Requirements
* Python >= 3.6
* torch >= 0.4.1
* Other required packages in `requirements.txt`

### Steps

```
# Prepare the relevant dataset, and put it under data/[name]/.
    * WMT14(EN-DE)
    * WMT14(EN-FR)
    * ParaNMT
    * QUORA

# Obtain the parse trees for sentences via CoreNLP.

# Run the models.
```


### Usage


```
# Parsing sentences

## Downloading the CoreNLP:
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip

## Deploy CoreNLP service
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8083 -timeout 15000 > 1.log 2>&1 &

## Parsing sentences for constituency trees
python data/parsing.py

```


```
# Train language model
## If two coupled tasks are with OOD texts (e.g., NMT), please train two separate LM.

python run_lm.py 

```

```
# Dual learning Baseline
python run_dsl.py

```

```
# Dual learning with structural matching
python main.py

```

### Cite

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