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

############################################################################
# DIRECTORIES
############################################################################
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR=$(readlink -f "$CURRENT_DIR/..")
TB_DIR="$PROJECT_DIR/tensorboard"
EXP_LAUNCH_DIR="$CURRENT_DIR/translation"

mkdir -p $EXP_LAUNCH_DIR
mkdir -p $EXP_LAUNCH_DIR/log

############################################################################
# Job Generator
############################################################################

TOTAL_UPDATES=80000   # Total number of training steps
WARMUP_UPDATES=8000   # Warmup the learning rate over this many updates
MAX_TOKENS=12000       # Warmup the learning rate over this many updates
PEAK_LR=0.0005        # Peak learning rate, adjust as needed
WEIGHT_DECAY=0.01
CLIP_NORM=0.0
EPS=1e-06
BETA='(0.9, 0.999)'
UPDATE_FREQ=1         # Increase the batch size X
N_GPU=2

generate_job() {
  EXP_NAME=${1}
  DATA=$(readlink -f $PROJECT_DIR/data-bin/${2})
  SRC_LANG=${3}
  TRG_LANG=${4}
  HPARAMS=${5}
  SEED=${6}

  EXP_NAME="${EXP_NAME}_seed=${SEED}"
  SAVE_DIR=$PROJECT_DIR/checkpoints/$EXP_NAME
  FILE="$EXP_LAUNCH_DIR/${EXP_NAME}.sh"

  echo "Experiment: ${EXP_NAME}"
  echo

  if [ "${MODE}" == 'train' ]; then
    cat <<END >$FILE
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH --job-name=${EXP_NAME}
#SBATCH --output=$EXP_LAUNCH_DIR/log/${EXP_NAME}.%j.out
#SBATCH --error=$EXP_LAUNCH_DIR/log/${EXP_NAME}.%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:$N_GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=$TIME
#SBATCH --partition=pascal
#SBATCH --array=$ARRAY

# prepare for experiment - load necessary modules etc.
source $HOME/.bashrc
conda activate $CONDA_ENV

export PYTHONPATH=$PROJECT_DIR/../:\$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH

fairseq-train $DATA \\
  --user-dir $PROJECT_DIR/user \\
  --task translation \\
  --source-lang $SRC_LANG --target-lang $TRG_LANG \\
  $HPARAMS \\
  --optimizer adam --adam-betas '$BETA' --adam-eps $EPS \\
  --warmup-updates $WARMUP_UPDATES \\
  --max-update $TOTAL_UPDATES \\
  --lr $PEAK_LR --lr-scheduler inverse_sqrt \\
  --weight-decay $WEIGHT_DECAY --clip-norm $CLIP_NORM \\
  --max-source-positions 200 \\
  --max-target-positions 200 \\
  --max-tokens $MAX_TOKENS \\
  --update-freq $UPDATE_FREQ  \\
  --save-dir $SAVE_DIR \\
  --tensorboard-logdir $TB_DIR/$EXP_NAME \\
  --log-interval 100  --log-format tqdm \\
  --save-interval-updates 5000 \\
  --keep-interval-updates 0 \\
  --validate-interval 1000 \\
  --no-epoch-checkpoints \\
  --eval-bleu \\
  --eval-bleu-args '{"beam": 5}' \\
  --eval-bleu-detok space \\
  --eval-bleu-remove-bpe sentencepiece \\
  --eval-bleu-print-samples \\
  --best-checkpoint-metric bleu \\
  --maximize-best-checkpoint-metric \\
  --seed $SEED

scancel \$SLURM_ARRAY_JOB_ID

END
  fi

  cat <<END >>$FILE

for split in valid test; do
  fairseq-generate $PARA_DATA \\
    --user-dir $PROJECT_DIR/user \\
    --source-lang $SRC_LANG --target-lang $TRG_LANG \\
    --gen-subset \$split \\
    --path $SAVE_DIR/checkpoint_best.pt \\
    --results-path $SAVE_DIR \\
    --beam 5 --remove-bpe sentencepiece --sacrebleu
done

$PROJECT_DIR/experiments/eval-translation.sh $SAVE_DIR $SRC_LANG $TRG_LANG

scancel \${SLURM_ARRAY_JOB_ID}

END

  sbatch $FILE
}


generate_job nmt.deen.base parallel.en_de de en "--arch paper_transformer_mt --criterion cross_entropy" 1
generate_job nmt.deen.ls parallel.en_de de en "--arch paper_transformer_mt --criterion label_smoothed_cross_entropy --label-smoothing 0.1" 1
generate_job nmt.deen.prior.3M_ls=0.0_tau=2_lambda=0.5 parallel.en_de de en "--arch paper_transformer_mt_lm --lm-checkpoint $PROJECT_DIR/checkpoints/lm.en.3M/checkpoint_best.pt --criterion cross_entropy_prior --prior-lambda 0.5 --prior-tau 2 --label-smoothing 0" 1
