#!/usr/bin/env bash

FILE=$1
LANG=$2

echo "Splitting sentences..."
moses-sent-splitter -u $2 $1 "$1.sents"

echo "Removing titles..."
sed -i '/^\s*=/ d' "$1.sents"

echo "Remove empty lines..."
sed -i '/^\s*$/d' "$1.sents"