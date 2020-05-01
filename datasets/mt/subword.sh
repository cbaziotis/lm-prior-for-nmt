#!/bin/bash

vocab=$1

en=./wmt_ende/train.en.pp,./wmt_entr/train.en.pp
tr=./wmt_entr/train.tr.pp
de=./wmt_ende/train.de.pp

spm_train_path=~/workspace/libs/sentencepiece/build/src/spm_train

$spm_train_path --input=$en --character_coverage=1 --model_prefix=en.$vocab --model_type=unigram --vocab_size=$vocab --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --bos_piece '<sos>' --eos_piece '<eos>'
$spm_train_path --input=$tr --character_coverage=1 --model_prefix=tr.$vocab --model_type=unigram --vocab_size=$vocab --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --bos_piece '<sos>' --eos_piece '<eos>'
$spm_train_path --input=$de --character_coverage=1 --model_prefix=de.$vocab --model_type=unigram --vocab_size=$vocab --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --bos_piece '<sos>' --eos_piece '<eos>'

cp en.$vocab.* ./wmt_ende/
cp en.$vocab.* ./wmt_entr/
rm en.$vocab.*
mv de.$vocab.* ./wmt_ende/
mv tr.$vocab.* ./wmt_entr/
