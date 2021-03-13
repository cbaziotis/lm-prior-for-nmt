#!/bin/bash

# main paths
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MAIN_PATH=$(readlink -f "$DIR")
DATA_PATH=$MAIN_PATH/data
VOCAB_PATH=$MAIN_PATH/vocab
BIN_PATH=$MAIN_PATH/data-bin
TOK_PATH=$MAIN_PATH/tok

mkdir -p $DATA_PATH
mkdir -p $BIN_PATH
mkdir -p $VOCAB_PATH
mkdir -p $TOK_PATH

VOCAB_SIZE=16000
SPM_COVERAGE=0.9999

for lang in en de tr; do

  if [ "$lang" == "en" ]; then
    SPM_DATA="$DATA_PATH/en_de/train.en,$DATA_PATH/en_tr/train.en"

  elif [ "$lang" == "de" ]; then
    SPM_DATA="$DATA_PATH/en_de/train.de"

  elif [ "$lang" == "tr" ]; then
    SPM_DATA="$DATA_PATH/en_tr/train.tr"
  fi

  SPM_PREFIX=$VOCAB_PATH/$lang

  #---------------------------------------------------------------------------
  # Train the sentencepiece model (SPM_PREFIX)
  #---------------------------------------------------------------------------
  if [ ! -f "$SPM_PREFIX.model" ]; then
    echo "Training $SPM_PREFIX..."
    spm_train --input=$SPM_DATA \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=$SPM_COVERAGE \
      --max_sentence_length=256 \
      --model_prefix=$SPM_PREFIX \
      --model_type=unigram

    # convert SPM_PREFIX vocab to fairseq dict for later use
    cut -f1 $SPM_PREFIX.vocab | tail -n +4 | sed "s/$/ 100/g" >$SPM_PREFIX.dict.txt

  else
    echo "$SPM_PREFIX.model already trained."
  fi

done

prepare_parallel() {
  L1=${1}
  L2=${2}
  mkdir -p $TOK_PATH/parallel.${L1}_${L2}

  echo "Tokenizing the parallel data"
  for split in train dev test; do

    raw=$DATA_PATH/${L1}_${L2}/${split}
    tokenized=$TOK_PATH/parallel.${L1}_${L2}/${split}

    spm_encode --model=$VOCAB_PATH/${L1}.model --output_format=piece < ${raw}.${L1} > ${tokenized}.${L1}
    spm_encode --model=$VOCAB_PATH/${L2}.model --output_format=piece < ${raw}.${L2} > ${tokenized}.${L2}

  done


  # Path with binarized data
  PROC_PATH=$BIN_PATH/parallel.${L1}_${L2}

  fairseq-preprocess \
    --trainpref $TOK_PATH/parallel.${L1}_${L2}/train \
    --validpref $TOK_PATH/parallel.${L1}_${L2}/dev \
    --testpref $TOK_PATH/parallel.${L1}_${L2}/test \
    --destdir $PROC_PATH \
    --srcdict $VOCAB_PATH/${L1}.dict.txt \
    --tgtdict $VOCAB_PATH/${L2}.dict.txt \
    --source-lang $L1 \
    --target-lang $L2 \
    --bpe sentencepiece \
    --workers 10

  # copy SPM data for logging purposes
  for lang in "${L1}" "${L2}"; do
    cp $VOCAB_PATH/$lang.model $PROC_PATH/$lang.spm.model
    cp $VOCAB_PATH/$lang.vocab $PROC_PATH/$lang.spm.vocab
  done

}

prepare_mono() {
  LANG=${1}
  SIZE=${2}

  mkdir -p $TOK_PATH/mono.${LANG}

  TRAIN=$DATA_PATH/$LANG/news.$SIZE.$LANG.train
  VALID=$DATA_PATH/$LANG/news.$LANG.val

  SPM_PREFIX=$VOCAB_PATH/$LANG

  echo "Tokenizing the monolingual data using the pretrained SPM:'$SPM_PREFIX.model'"
  spm_encode --model=$SPM_PREFIX.model --output_format=piece < $TRAIN > $TOK_PATH/mono.${LANG}/$LANG.$SIZE.train
  spm_encode --model=$SPM_PREFIX.model --output_format=piece < $VALID > $TOK_PATH/mono.${LANG}/$LANG.val

  # Path with binarized data
  PROC_PATH=$BIN_PATH/mono.$LANG.$SIZE

  fairseq-preprocess --only-source \
    --trainpref $TOK_PATH/mono.${LANG}/$LANG.$SIZE.train \
    --validpref $TOK_PATH/mono.${LANG}/$LANG.val \
    --destdir $PROC_PATH \
    --srcdict $SPM_PREFIX.dict.txt \
    --bpe sentencepiece \
    --workers 8

  # copy SPM data for logging purposes
  cp $SPM_PREFIX.model $PROC_PATH/spm.model
  cp $SPM_PREFIX.vocab $PROC_PATH/spm.vocab


}

#prepare_parallel en de
#prepare_parallel en tr
prepare_mono en 3M

