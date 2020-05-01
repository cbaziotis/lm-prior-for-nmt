#!/bin/bash

#####################################################################################
tools=../../tools
moses_scripts=$tools/moses_scripts/scripts
ende_path=./wmt_ende/
entr_path=./wmt_entr/

en_mono=../mono/priors/news.en.2014-2017
de_mono=../mono/priors/news.de.2014-2017
tr_mono=../mono/priors/news.tr

#####################################################################################
echo "Normalizing punctuation..."
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $en_mono > $en_mono.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l de < $de_mono > $de_mono.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l tr < $tr_mono > $tr_mono.norm

echo "Truecasing..."
$moses_scripts/recaser/truecase.perl < $en_mono.norm > $en_mono.pp -model $ende_path/truecase-model.en
$moses_scripts/recaser/truecase.perl < $de_mono.norm > $de_mono.pp -model $ende_path/truecase-model.de
$moses_scripts/recaser/truecase.perl < $tr_mono.norm > $tr_mono.pp -model $entr_path/truecase-model.tr

rm $en_mono.norm $de_mono.norm $tr_mono.norm

echo "done!"
