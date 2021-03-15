# FAIRSEQ Plug-in

This page contains the documentation for
the [fairseq](https://github.com/pytorch/fairseq) [plug-in](https://fairseq.readthedocs.io/en/latest/overview.html)
of the paper "Language Model Prior for Low-Resource Neural Machine Translation"
([Paper](https://arxiv.org/abs/2004.14928))

To be able to use the plug-in with your fairseq installation simply copy the
contents of
`fairseq_extension/user` into your `user` directory and don't forget to update
its `__init__.py`. In practice, all you really need is
the `fairseq_extension/user/lm_prior/lm_prior.py` file. If you are not familiar
with fairseq's plug-in system
read [this](https://fairseq.readthedocs.io/en/latest/overview.html) first.


Here is an example of how to train a model with fairseq using the plugin:

```shell
DATA=fairseq_extension/data-bin/parallel.en_de
USER=fairseq_extension/user
LM_CHKP=fairseq_extension/checkpoints/lm.en.3M/checkpoint_best.pt

fairseq-train  $DATA \
  --user-dir  $USER \
  --task translation \
  --source-lang de --target-lang en \
  --arch transformer_mt_lm \ 
  --lm-checkpoint $LM_CHKP \
  --criterion cross_entropy_prior \ 
  --prior-lambda 0.5 \
  --prior-tau 2 \
  --label-smoothing 0 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-06 \
  --warmup-updates 8000 --lr 0.0005 --lr-scheduler inverse_sqrt \
  --weight-decay 0.01 --clip-norm 0.0 \
  --max-source-positions 256 --max-target-positions 256 --max-tokens 12000 \
  --save-dir /home/christos/PycharmProjects/lmprior/fairseq_extension/checkpoints/nmt.deen.prior.3M_ls=0.0_tau=2_lambda=0.3_seed=1 \
  --tensorboard-logdir /home/christos/PycharmProjects/lmprior/fairseq_extension/tensorboard/nmt.deen.prior.3M_ls=0.0_tau=2_lambda=0.3_seed=1 \
  --seed 1

```
These parameters are **fixed**:

- `--arch`=`transformer_mt_lm`
- `--criterion`=`cross_entropy_prior`

There are 3 parameters that you can change:

- `--prior-lambda`: this controls the weight applied to the auxiliary regularization term. It is recommended to use values in the range `[0.1-0.5]` (In the paper we set `λ=0.5`).
- `--prior-tau`: this is the temperature parameter applied to the KL term. It is recommended to use values in the range `[1-5]` (In the paper we set `τ=2`).
- `--label-smoothing`:  If you want to also apply label-smoothing the target distribution, specify the value of the smoothing parameter.


**Disclaimer**: This is a re-implementation of the original code in fairseq and
the goal is to enable others to try the method in a widely adopted framework. As
the method is implemented in a different framework there may be some small
differences in the final results. However, I expect that your results will
generally agree with the findings of the paper.

# Prerequisites

### 1. Create Conda Environment

```
conda create -n fairseq-lm-prior python=3
conda activate fairseq-lm-prior
```

### 2. Install Pytorch and Fairseq

Install PyTorch (`1.8`) with the desired Cuda version:

```shell
# CUDA 10.2
pip install torch torchvision torchaudio

```

and then install the current version
of [fairseq](https://github.com/pytorch/fairseq#requirements-and-installation) (
latest [commit](https://github.com/pytorch/fairseq/commit/965240c784910895b05e66d7ef7e15321050b414)
at the time of writing) rest of the requirements:

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### 3. Download Data

You can run `fairseq_extension/download_data.sh` which will store the parallel
and monolingual data used in the paper in the `fairseq_extension/data/`
directory.

```text
data
├── mono
│   ├── news.30M.de.train
│   ├── news.30M.en.train
│   ├── news.3M.de.train
│   ├── news.3M.en.train
│   ├── news.3M.tr.train
│   ├── news.de.2014-2017
│   ├── news.de.val
│   ├── news.en.2014-2017
│   ├── news.en.val
│   ├── news.tr
│   └── news.tr.val
├── parallel_en_de
│   ├── dev.de
│   ├── dev.en
│   ├── test.de
│   ├── test.en
│   ├── train.de
│   └── train.en
└── parallel_en_tr
    ├── dev.en
    ├── dev.tr
    ├── test.en
    ├── test.tr
    ├── train.en
    └── train.tr
```

### 4. Prepare Training Data

To preprocess and binarize the (monolingual and parallel) training data
run  `fairseq_extension/prepare_data.sh`.

**Important**: Unlike in the paper, for simplicity here we simply tokenize the
raw data with sentencepiece, and omit any further preprocessing steps.

Running `fairseq_extension/prepare_data.sh` will create the following outputs:

- A `fairseq_extension/tok/` directory, which will contain the tokenized data.
- A `fairseq_extension/vocab/` directory, which will contain the sentencepiece
  models.
- A `fairseq_extension/data-bin/` directory, which will contain the binarized
  datasets, which will be used for training with fairseq.

# Training

### Start Tensorboard server (optional)

The training progress (statistics, losses etc.) of each model will be saved
under `fairseq_extension/tensorboard/`. To visualize the training process run:

```shell script
tensorboard --logdir fairseq_extension/tensorboard --port 8002
``` 

and open `http://localhost:8002/` in your browser.

### (SLURM) Experiment Launchers

To train the language and translation model you can use the **SLURM** experiment
launcher scripts:

```text
fairseq_extension/experiments/
├── eval-translation.sh
├── slurm_train_lm.sh       # Launch LM training experiments
├── slurm_train_mt.sh       # Launch MT training experiments
```

These scripts will automatically generate the training scripts and place the
under `fairseq_extension/experiments/lm`
and `fairseq_extension/experiments/translation`, respectively.

**If you don't want to use SLURM**, you can inspect and run any of those scripts
yourself. For instance, after
running `fairseq_extension/experiments/slurm_train_mt.sh`, you can edit and
run `fairseq_extension/experiments/translation/nmt.deen.prior.3M_ls=0.0_tau=2_lambda=0.5_seed=1.sh`
which contains the code for training an de->en translation model with an (en) LM
trained on 3M data.

**If you do use SLURM**, please update the header of `slurm_train_lm.sh`
and `slurm_train_mt.sh` based on your server settings.

```shell
#!/bin/bash
############################################################################
# SLURM SETTINGS - Update these parameters based on your setup/server
############################################################################
CONDA_ENV="fairseq-lm-prior"  # This is the name of the project's conda environment
ACCOUNT="Project123-GPU"      # Your slurm account.
TIME="35:59:59"               # The duration of each slurm job. E.g.
ARRAY="1-4%1"                 # How many times to repeat the slurm job."1-2%1"
MODE="train"                  # The job mode (NOT slurm). 1) "train" means that you want to
                              # first train and then eval the trained model, while
                              # 2) "eval" just evaluates it an already trained model.
```

### TL;DR: Training Steps

1. Create and launch the LM training experiments using `slurm_train_lm.sh` (or
   on the specific experiments under `fairseq_extension/experiments/lm`).
2. Create and launch the MT training experiments using `slurm_train_mt.sh` (or
   on the specific experiments
   under `fairseq_extension/experiments/translation`). It assumes that you have
   already trained the corresponding LM priors.

The checkpoints of each model will be saved
under `fairseq_extension/checkpoints/`. Also, after the training of a
translation model finishes, it will be automatically evaluated and its outputs
will be saved in the same directory as its checkpoints.

# Reference

```
@inproceedings{baziotis-etal-2020-language,
    title = "Language Model Prior for Low-Resource Neural Machine Translation",
    author = "Baziotis, Christos  and
      Haddow, Barry  and
      Birch, Alexandra",
    booktitle = "Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.615",
    doi = "10.18653/v1/2020.emnlp-main.615",
    pages = "7622--7634"
}
```