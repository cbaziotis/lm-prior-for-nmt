import os
import sys

from fairseq_cli.train import cli_main

from fairseq_extension.sys_config import PATH_CP, PATH_TB, PATH_BIN, PATH_USER

dataset = 'parallel.en_de'
experiment = 'ende_prior_3'

sys.argv.extend([os.path.join(PATH_BIN, dataset)])
sys.argv.extend(['--user-dir', PATH_USER])

sys.argv.extend(['--task', 'translation'])
sys.argv.extend(['--source-lang', 'de'])
sys.argv.extend(['--target-lang', 'en'])
sys.argv.extend(['--arch', 'paper_transformer_mt_lm'])
sys.argv.extend(['--lm-checkpoint', os.path.join(PATH_CP,
                                                 'mono.en.3M/lm.en.3M/checkpoint_best.pt')])

sys.argv.extend(['--optimizer', 'adam'])
sys.argv.extend(['--lr', '5e-4'])
sys.argv.extend(['--lr-scheduler', 'inverse_sqrt'])
sys.argv.extend(['--clip-norm', '0.1'])
sys.argv.extend(['--warmup-updates', '4000'])
sys.argv.extend(['--weight-decay', '0.0001'])

sys.argv.extend(['--criterion', 'cross_entropy_prior'])
sys.argv.extend(['--max-tokens', '2000'])

# LOGGING
sys.argv.extend(['--save-dir', os.path.join(PATH_CP, dataset, experiment)])
sys.argv.extend(['--tensorboard-logdir', os.path.join(PATH_TB, experiment)])
sys.argv.extend(['--log-interval', '10'])
sys.argv.extend(['--log-format', 'tqdm'])
sys.argv.extend(['--no-epoch-checkpoints'])
sys.argv.extend(['--num-workers', '0'])
sys.argv.extend(['--save-interval-updates', '20'])

# EVALUATION
# sys.argv.extend(['--eval-bleu'])
# sys.argv.extend(['--eval-bleu-args', '{"beam": 5}'])
# sys.argv.extend(['--eval-bleu-detok', 'space'])
# sys.argv.extend(['--eval-bleu-remove-bpe', 'sentencepiece'])
# sys.argv.extend(['--eval-bleu-print-samples'])
# sys.argv.extend(['--best-checkpoint-metric', 'bleu'])
# sys.argv.extend(['--maximize-best-checkpoint-metric'])

cli_main()
