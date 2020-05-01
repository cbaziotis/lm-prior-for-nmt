#!/bin/bash

#####################################################################################
tools=../../tools
moses_scripts=$tools/moses_scripts/scripts
ende_path=./wmt_ende/
entr_path=./wmt_entr/


#####################################################################################

echo "Normalizing punctuation..."

#EN-DE
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $ende_path/train.en > $ende_path/train.en.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $ende_path/dev.en   > $ende_path/dev.en.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $ende_path/test.en  > $ende_path/test.en.norm

$moses_scripts/tokenizer/normalize-punctuation.perl -l de < $ende_path/train.de > $ende_path/train.de.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l de < $ende_path/dev.de   > $ende_path/dev.de.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l de < $ende_path/test.de  > $ende_path/test.de.norm

#EN-TR
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $entr_path/train.en > $entr_path/train.en.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $entr_path/dev.en   > $entr_path/dev.en.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l en < $entr_path/test.en  > $entr_path/test.en.norm

$moses_scripts/tokenizer/normalize-punctuation.perl -l tr < $entr_path/train.tr > $entr_path/train.tr.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l tr < $entr_path/dev.tr   > $entr_path/dev.tr.norm
$moses_scripts/tokenizer/normalize-punctuation.perl -l tr < $entr_path/test.tr  > $entr_path/test.tr.norm


echo "Truecasing..."

# learn truecase on the concatenation of the 2 English corpora
cat $ende_path/train.en.norm $entr_path/train.en.norm | $moses_scripts/recaser/train-truecaser.perl -model truecase-model.en -corpus -
cp truecase-model.en $ende_path/truecase-model.en
cp truecase-model.en $entr_path/truecase-model.en
rm truecase-model.en

#apply English truecaser to english side of the EN-DE parallel data
$moses_scripts/recaser/truecase.perl < $ende_path/train.en.norm > $ende_path/train.en.pp -model $ende_path/truecase-model.en
$moses_scripts/recaser/truecase.perl < $ende_path/dev.en.norm   > $ende_path/dev.en.pp   -model $ende_path/truecase-model.en
$moses_scripts/recaser/truecase.perl < $ende_path/test.en.norm  > $ende_path/test.en.pp  -model $ende_path/truecase-model.en

#apply English truecaser to english side of the EN-TR parallel data
$moses_scripts/recaser/truecase.perl < $entr_path/train.en.norm > $entr_path/train.en.pp -model $entr_path/truecase-model.en
$moses_scripts/recaser/truecase.perl < $entr_path/dev.en.norm   > $entr_path/dev.en.pp   -model $entr_path/truecase-model.en
$moses_scripts/recaser/truecase.perl < $entr_path/test.en.norm  > $entr_path/test.en.pp  -model $entr_path/truecase-model.en


# learn and apply truecaser on the Turkish
$moses_scripts/recaser/train-truecaser.perl -model $entr_path/truecase-model.tr -corpus $entr_path/train.tr.norm
$moses_scripts/recaser/truecase.perl < $entr_path/train.tr.norm > $entr_path/train.tr.pp -model $entr_path/truecase-model.tr
$moses_scripts/recaser/truecase.perl < $entr_path/dev.tr.norm   > $entr_path/dev.tr.pp   -model $entr_path/truecase-model.tr
$moses_scripts/recaser/truecase.perl < $entr_path/test.tr.norm  > $entr_path/test.tr.pp  -model $entr_path/truecase-model.tr

# learn and apply truecaser on the German
$moses_scripts/recaser/train-truecaser.perl -model $ende_path/truecase-model.de -corpus $ende_path/train.de.norm
$moses_scripts/recaser/truecase.perl < $ende_path/train.de.norm > $ende_path/train.de.pp -model $ende_path/truecase-model.de
$moses_scripts/recaser/truecase.perl < $ende_path/dev.de.norm   > $ende_path/dev.de.pp   -model $ende_path/truecase-model.de
$moses_scripts/recaser/truecase.perl < $ende_path/test.de.norm  > $ende_path/test.de.pp  -model $ende_path/truecase-model.de


rm $ende_path/*.norm
rm $entr_path/*.norm

echo "Filtering bad pairs..."
python $tools/data/clean_parallel.py -src_file $ende_path/train.en.pp -trg_file $ende_path/train.de.pp -max_length 60 -min_length 1 -length_ratio_threshold 1.5
mv $ende_path/train.en.pp.filtered $ende_path/train.en.pp
mv $ende_path/train.de.pp.filtered $ende_path/train.de.pp

python $tools/data/clean_parallel.py -src_file $entr_path/train.en.pp -trg_file $entr_path/train.tr.pp -max_length 60 -min_length 1 -length_ratio_threshold 1.5
mv $entr_path/train.en.pp.filtered $entr_path/train.en.pp
mv $entr_path/train.tr.pp.filtered $entr_path/train.tr.pp


echo "done!"
