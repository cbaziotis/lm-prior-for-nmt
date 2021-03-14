#!/bin/bash

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
# SLURM SETTINGS
############################################################################
CONDA_ENV="fairseq-lm-prior"

ACCOUNT="T2-CS119-GPU"
TIME="35:59:59"
ARRAY="1-2%1"
MODE='train'

for i in "$@"; do
  if [[ $i == "low" ]]; then
    ACCOUNT="T2-CS055-SL4-GPU"
    TIME="11:59:59"
    ARRAY="1-6%1"
  elif [[ $i == "eval" ]]; then
    MODE='eval'
  fi
done


############################################################################
# Job Generator
############################################################################

TOTAL_UPDATES=50000  # Total number of training steps
WARMUP_UPDATES=8000   # Warmup the learning rate over this many updates
MAX_TOKENS=12000      # Warmup the learning rate over this many updates
PEAK_LR=0.0002        # Peak learning rate, adjust as needed
WEIGHT_DECAY=0.01
CLIP_NORM=0.0
EPS=1e-06
BETA='(0.9, 0.999)'
LS=0.1                # Label smoothing
UPDATE_FREQ=1         # Increase the batch size X
N_GPU=2

generate_job() {
  EXP_NAME=${1}
  DATA=$(readlink -f $PROJECT_DIR/data-bin/${2})
  SRC_LANG=${3}
  TRG_LANG=${4}
  SEED=${5}

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
  --arch paper_transformer_mt \\
  --criterion label_smoothed_cross_entropy --label-smoothing $LS \\
  --optimizer adam --adam-betas '$BETA' --adam-eps $EPS \\
  --warmup-updates $WARMUP_UPDATES \\
  --max-update $TOTAL_UPDATES \\
  --lr $PEAK_LR --lr-scheduler inverse_sqrt \\
  --weight-decay $WEIGHT_DECAY --clip-norm $CLIP_NORM \\
  --max-source-positions 256 \\
  --max-target-positions 256 \\
  --max-tokens $MAX_TOKENS \\
  --update-freq $UPDATE_FREQ  \\
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

fairseq-generate $TEST_DATA \\
  --user-dir $PROJECT_DIR/user \\
  --source-lang $SRC_LANG --target-lang $TRG_LANG \\
  --path $SAVE_DIR/checkpoint_best.pt \\
  --results-path $SAVE_DIR/newstest2019 \\
  --beam 5 --remove-bpe sentencepiece --sacrebleu

$PROJECT_DIR/experiments/eval-translation.sh $SAVE_DIR $SRC_LANG $TRG_LANG

scancel \${SLURM_ARRAY_JOB_ID}

END

  sbatch $FILE
}


generate_job nmt.en_de.ls parallel.en_de en de 1