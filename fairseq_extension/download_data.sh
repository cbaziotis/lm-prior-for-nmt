#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"



#---------------------------------------------------------------------------
# Download English-German Parallel Data
#---------------------------------------------------------------------------
mkdir -p $CURRENT_DIR/data/parallel_en_de
cd $CURRENT_DIR/data/parallel_en_de

for lang in en de; do
  for split in train dev test; do
    wget -c http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_ende/$split.$lang
  done
done

#---------------------------------------------------------------------------
# Download English-Turkish Parallel Data
#---------------------------------------------------------------------------
mkdir -p $CURRENT_DIR/data/parallel_en_tr
cd $CURRENT_DIR/data/parallel_en_tr

for lang in en de; do
  for split in train dev test; do
    wget -c http://data.statmt.org/cbaziotis/projects/lm-prior/parallel/wmt_entr/$split.$lang
  done
done

#---------------------------------------------------------------------------
# Download Monolingual Data
#---------------------------------------------------------------------------
mkdir -p $CURRENT_DIR/data/mono
cd $CURRENT_DIR/data/mono

wget -c http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.de.2014-2017
wget -c http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.en.2014-2017
wget -c http://data.statmt.org/cbaziotis/projects/lm-prior/mono/news.tr

echo "splitting German monolingual data..."
split -l 29990000 news.de.2014-2017
mv xaa news.30M.de.train
mv xab news.de.val
head -n 3000000 news.30M.de.train > news.3M.de.train

echo "splitting English monolingual data..."
split -l 29990000 news.en.2014-2017
mv xaa news.30M.en.train
mv xab news.en.val
head -n 3000000 news.30M.en.train > news.3M.en.train

echo "splitting Turkish monolingual data..."
split -l 2990000 news.tr
mv xaa news.3M.tr.train
mv xab news.tr.val