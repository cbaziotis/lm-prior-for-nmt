#!/bin/bash

wget http://data.statmt.org/wmt18/translation-task/dev.tgz
tar zxvf dev.tgz

wget http://data.statmt.org/wmt18/translation-task/test.tgz
tar zxvf test.tgz

wget http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz
tar zxvf training-parallel-nc-v13.tgz
mv training-parallel-nc-v13 train

rm *.tgz



#------------------------------------------------------------------------------
# English - German
#------------------------------------------------------------------------------
mkdir wmt_ende

# train
cp train/news-commentary-v13.de-en.de wmt_ende/train.de
cp train/news-commentary-v13.de-en.en wmt_ende/train.en

# dev
perl input-from-sgm.perl < dev/newstest2017-deen-src.de.sgm > wmt_ende/dev.de
perl input-from-sgm.perl < dev/newstest2017-deen-ref.en.sgm > wmt_ende/dev.en

# test
perl input-from-sgm.perl < test/newstest2018-deen-src.de.sgm > wmt_ende/test.de
perl input-from-sgm.perl < test/newstest2018-deen-ref.en.sgm > wmt_ende/test.en



#------------------------------------------------------------------------------
# English - Turkish
#------------------------------------------------------------------------------
mkdir wmt_entr

wget http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-tr.txt.zip -O en-tr.zip
unzip en-tr.zip -d wmt_entr
rm en-tr.zip wmt_entr/LICENSE wmt_entr/README wmt_entr/SETIMES.en-tr.ids


# train
mv wmt_entr/SETIMES.en-tr.en wmt_entr/train.en
mv wmt_entr/SETIMES.en-tr.tr wmt_entr/train.tr

# dev
perl input-from-sgm.perl < dev/newstest2017-tren-src.tr.sgm > wmt_entr/dev.tr
perl input-from-sgm.perl < dev/newstest2017-tren-ref.en.sgm > wmt_entr/dev.en

# test
perl input-from-sgm.perl < test/newstest2018-tren-src.tr.sgm > wmt_entr/test.tr
perl input-from-sgm.perl < test/newstest2018-tren-ref.en.sgm > wmt_entr/test.en

rm -rf dev/ train/ test/