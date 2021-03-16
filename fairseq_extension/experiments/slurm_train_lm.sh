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
EXP_LAUNCH_DIR="$CURRENT_DIR/lm"

mkdir -p $EXP_LAUNCH_DIR
mkdir -p $EXP_LAUNCH_DIR/log

############################################################################
# Job Generator
############################################################################

TOTAL_UPDATES=100000 # Total number of training steps
WARMUP_UPDATES=16000 # Warmup the learning rate over this many updates
MAX_TOKENS=16000      # Warmup the learning rate over this many updates
PEAK_LR=0.0005       # Peak learning rate, adjust as needed
WEIGHT_DECAY=0.01
CLIP_NORM=0.0
EPS=1e-06
BETA='(0.9, 0.999)'
UPDATE_FREQ=1 # Increase the batch size X
N_GPU=4

generate_job() {
  EXP_NAME=${1}
  DATA=$(readlink -f $PROJECT_DIR/data-bin/${2})
  SAVE_DIR=$PROJECT_DIR/checkpoints/$EXP_NAME
  FILE="$EXP_LAUNCH_DIR/${EXP_NAME}.sh"

  echo "Experiment: ${EXP_NAME}"
  echo

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
  --task language_modeling \\
  --arch paper_transformer_lm \\
  --sample-break-mode eos \\
  --criterion cross_entropy \\
  --optimizer adam --adam-betas '$BETA' --adam-eps $EPS \\
  --warmup-updates $WARMUP_UPDATES \\
  --max-update $TOTAL_UPDATES \\
  --lr $PEAK_LR --lr-scheduler inverse_sqrt \\
  --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ  \\
  --weight-decay $WEIGHT_DECAY --clip-norm $CLIP_NORM \\
  --tokens-per-sample 256 \\
  --save-dir $SAVE_DIR \\
  --tensorboard-logdir $TB_DIR/$EXP_NAME \\
  --log-interval 100 --log-format tqdm \\
  --no-epoch-checkpoints \\
  --save-interval-updates 20000 \\
  --keep-interval-updates 0 \\
  --patience 10 \\
  --skip-invalid-size-inputs-valid-test \\
  --ddp-backend no_c10d

scancel \$SLURM_ARRAY_JOB_ID

END

  sbatch $FILE
}

generate_job lm.en.3M  mono.en.3M
generate_job lm.en.30M mono.en.30M

generate_job lm.de.3M  mono.de.3M
generate_job lm.de.30M mono.de.30M

generate_job lm.tr.3M  mono.tr.3M
