#!/bin/bash

SAVE_DIR=$1
SRC=$2
TRG=$3

cd $SAVE_DIR

mv $SAVE_DIR/generate-valid.txt $SAVE_DIR/generate-valid-${SRC}${TRG}.txt
mv $SAVE_DIR/generate-test.txt $SAVE_DIR/generate-test-${SRC}${TRG}.txt

for split in valid-${SRC}${TRG} test-${SRC}${TRG}; do
  cat generate-$split.txt | grep -P "^H" | sort -V | cut -f 3- | sed "s/\[${TRG}\]//g" >$split.hyp
  cat generate-$split.txt | grep -P "^T" | sort -V | cut -f 2- | sed "s/\[${TRG}\]//g" >$split.ref
  cat generate-$split.txt | grep -P "^S" | sort -V | cut -f 2- | sed "s/\[${SRC}\]//g" >$split.src

  # Normal HYP-REF evaluation
  cat $split.hyp | sacrebleu $split.ref -m bleu chrf ter > $split.scores
  cat $split.hyp | sacrebleu $split.ref -b -m bleu       > $split.bleu
  cat $split.hyp | sacrebleu $split.ref -b -m chrf       > $split.chrf
  cat $split.hyp | sacrebleu $split.ref -b -m ter        > $split.ter

done