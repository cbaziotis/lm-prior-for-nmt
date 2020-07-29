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

## Data

##### Prepare data on your own
To prepare the parallel data on your own: 
 1. run `bash datasets/mt/download_data.sh`
 2. run `bash datasets/mt/preprocess_parallel.sh`
 
 
##### Download preprocessed data
You can download the preprocessed data (monolingual and parallel), the truecase models 
and the pretrained sentencepiece models, by running:

```shell script
bash download_preprocessed_data.sh
```

 - The **parallel data** will be placed under`datasets/parallel/wmt_ende` and `datasets/parallel/wmt_entr`
 - The **monolingual data** will be placed under`datasets/mono/priors/`


## Training

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
lm-prior-for-nmt/models$  python sent_lm.py --config ../configs/prototype.rnn_lm_en.yaml model.emb_size=256
```
All the outputs of the models and its training progress will be saved in the `experiments/` directory.

 - **Train a LM**: Run `models/sent_lm.py` using the desired config.
 - **Train a TM**: Run `models/nmt_prior.py` using the desired config.

Take a look at the config files in `configs/transformer/` for details.

##### Download Pretrained LMs

To download all the checkpoints of the LM-priors used in the experiments, run:

```shell script
bash download_lm_checkpoints.sh
```

## Analysis Web-tool

To view more information about the analysis done in the paper go to: 
http://data.statmt.org/cbaziotis/projects/lm-prior/analysis