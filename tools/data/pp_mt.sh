#!/usr/bin/env bash

FILE=$1
LANG=$2

echo "Tokenizing..."
tokenizer.perl -l $2 < $1 > "$1.tok"

echo "Lowercasing..."
lowercase.perl < "$1.tok" > "$1.tok.lower"

echo "Extracting vocab..."
python extract_vocab.py -input "$1.tok.lower"