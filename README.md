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

Install PyTorch `1.4` with the desired Cuda version if you want to use the GPU
and then the rest of the requirements:

```
pip install -r requirements.txt
```

### Download Data

**1. Parallel data**: You can download the preprocessed data, the truecase models and the pretrained sentencepiece models from this link:
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

##### 1. Run Visdom server
We use Visdom for visualizing the training progress. Therefore, first open a terminal and run the visdom server:
```shell script
> visdom
``` 
Read more about visdom here: https://github.com/facebookresearch/visdom#usage

##### 2. Train a Translation or Language Model
Every model has a base configuration stored in a `.yaml` file. 
All model configurations are stored in the `configs/` directory. 
When you run an experiment you need to pass the base config to the corresponding 
python script and optionally override the parameters in the config file.

For example, you can train a LM on a small test corpus like this:
```shell script
/models$  python sent_lm.py --config ../configs/prototype.rnn_lm_en.yaml model.emb_size=256
```

**Train a LM**: Run `models/sent_lm.py` using the desired config