This repository contains source code for the paper 
"Language Model Prior for Low-Resource Neural Machine Translation" 
([Paper](https://arxiv.org/abs/2004.14928))


# Introduction 
In this work, we use a  language model (LM) trained
on target-side monolingual corpora as a weakly
informative prior. We add a regularization term,
which drives the output distributions of the translation model (TM) to
be probable under the distributions of the LM.

# Prerequisites

 
### Install Requirements

**Create Environment (Optional)**: Ideally, you should create an environment 
for the project.

```
conda create -n lmprior python=3
conda activate lmprior
```

Install PyTorch `1.4` ([guide](https://pytorch.org/get-started/previous-versions/#v140)) 
with the desired Cuda version if you want to use the GPU:
```shell
# CUDA 10.1
pip install torch==1.4.0 torchvision==0.5.0
```
and then the rest of the requirements:
```
pip install -r requirements.txt
```

### Download Data

**1. Parallel data**: 
You can download the preprocessed data, the truecase models and the pretrained sentencepiece models from this link:
http://data.statmt.org/cbaziotis/projects/lm-prior/parallel. 
Put the `wmt_ende` and `wmt_entr` folders in the `datasets/mt/` directory. 

To prepare the data on your own: 
 1. run `datasets/mt/download_data.sh`
 2. run `datasets/mt/preprocess_parallel.sh`
 
 
**2. Monolingual data**: 
You can download the preprocessed data from this link:
http://data.statmt.org/cbaziotis/projects/lm-prior/mono and then put the files in 
the `datasets/mono/priors/` directory.



### Training

##### Run Visdom server
We use Visdom for visualizing the training progress. Therefore, first open a terminal and run the visdom server:
```shell script
> visdom
``` 

Once you start training a model, open the visdom dashboard in your browser 
(by default in `http://localhost:8097/`) and select a model to view its statistics,
or multiple models to compare them.

Read more about visdom here: https://github.com/facebookresearch/visdom#usage

##### How to train a model
Every model requires a base configuration stored in a `.yaml` file. 
All model configurations are stored in the `configs/` directory. 
When you run an experiment you need to pass the base config to the corresponding 
python script and optionally override the parameters in the config file.

For example, you can train a LM on a small test corpus like this:
```shell script
/models$  python sent_lm.py --config ../configs/prototype.rnn_lm_en.yaml
```

To override one of the parameters in the config, you don't have to create a new one,
just pass the parameter-value pair like that:
```shell script
/models$  python sent_lm.py --config ../configs/prototype.rnn_lm_en.yaml  model.emb_size=256
```
For nested parameters, separate the names with `.`.

For every model that is trained, all its data, 
including the checkpoint, outputs and its training progress,
are saved in the `experiments/` directory, under `experiments/CONFIG_NAME/START_DATETIME`.
For instance, a model trained with the command above will be saved under:
`experiments/prototype.rnn_lm_en/20-10-21_16:03:22`.

> Verify that the model is training by opening visdom and selecting the model from the search bar.

#### 1. Train a language model (LM)
To train an LM you need to run  `models/sent_lm.py` using the desired config.
For example, to train an English Transformer-based LM on the 3M NewsCrawl data, 
same as in the paper, use the config `configs/transformer/prior.lm_news_en_trans.yaml`
and (optionally) pass any parameters to override those in the config file:

```shell
python mono_sent.py --config ../configs/transformer/prior.lm_news_en_trans.yaml  \
  --device cuda  --name prior.lm_news_en_3M_trans_big \ 
  batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3
```

**Reproducibility**: You can find the exact commands 
that were used for training the LMs used in the paper
in `configs/transformer/experiments_priors.sh`.


**Sanity check**
Verify that the model is training correctly by looking at the loss and model outputs (samples) in visdom.
You can test that everything is working correctly by trying first with a small model.
You should start to see reasonable sentences after a while.

#### 2. Train a translation model (TM)

To train a TM you need to run  `models/nmt_prior.py` using the desired config.
For the Transformer-based experiments, check the config files in `configs/transformer/`.


##### Train a standard TM

To train a standard TM for `de->en` with a transformer architecture run:
```shell
/models$  nmt_prior.py --config ../../configs/transformer/trans.deen_base.yaml --name final.trans.deen_base
```
All the model outputs will be saved in `experiments/trans.deen_base/START_DATETIME/`,
including the checkpoint of the model that has achieved the best score in the dev set.

**Evaluation**: To evaluate a pretrained model, run:
```shell

# translate the preprocessed input file (DE)
python models/translate.py --src datasets/mt/wmt_ende/test.de.pp \ 
  --out test.en.pp.hyps \
  --cp experiments/trans.deen_base/20-10-21_21:41:08/trans.deen_base_best.pt \ 
  --beam_size 5 --device cuda

# compare the raw detokenized hypothesis file (EN'), with the raw test set (EN)
cat test.en.pp.hyps | sacrebleu datasets/mt/wmt_ende/test.en
```


**Reproducibility**: In the following files, you will find all the commands for
reproducing the experiments in the paper:

 - `configs/transformer/experiments_nmt.sh` contains the commands for the main NMT experiments.
 - `configs/transformer/experiments_nmt_subsample_deen.sh` contains the commands 
   for the NMT experiments on various scales of the `en->de` parallel data.
 - `configs/transformer/experiments_sensitivity.sh` contains the commands 
   for the sensitivity analysis.



### Analysis

To view more information about the analysis done in the paper go to: 
http://data.statmt.org/cbaziotis/projects/lm-prior/analysis