#!/bin/bash

#------------------------------------------------------------------------------------------------------
# Monolingual data
#------------------------------------------------------------------------------------------------------

# raw sub-samples of News Crawl data for each language
MONO_PATH=datasets/mono/priors
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.tr

# preprocessed subsamples of News Crawl data for each language
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017.pp
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017.pp
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.tr.pp

# preprocessed training-validation splits (for English and German there are 3M and 30M versions)
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017.pp.3M.train
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017.pp.30M.train
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017.pp.val

wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017.pp.3M.train
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017.pp.30M.train
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017.pp.val

wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.tr.pp.train
wget -P $MONO_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.tr.pp.valid

#------------------------------------------------------------------------------------------------------
# Parallel data
#------------------------------------------------------------------------------------------------------
PARA_ENDE_PATH=datasets/parallel/wmt_ende/
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/de.16000.model
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/de.16000.vocab
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/dev.de
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/dev.de.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/dev.en
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/dev.en.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/en.16000.model
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/en.16000.vocab
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/test.de
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/test.de.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/test.en
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/test.en.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/train.de
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/train.de.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/train.en
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/train.en.pp
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/truecase-model.de
wget -P $PARA_ENDE_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/truecase-model.en

PARA_ENTR_PATH=datasets/parallel/wmt_entr/
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/dev.en
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/dev.en.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/dev.tr
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/dev.tr.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/en.16000.model
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/en.16000.vocab
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/test.en
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/test.en.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/test.tr
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/test.tr.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/tr.16000.model
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/tr.16000.vocab
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/train.en
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/train.en.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/train.tr
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/train.tr.pp
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/truecase-model.en
wget -P $PARA_ENTR_PATH http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/truecase-model.tr
