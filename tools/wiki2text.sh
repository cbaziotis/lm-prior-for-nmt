#!/usr/bin/env bash

FILE=$1
LANG=$2
OUT_DIR=$3
LEN=$4
OUT_FILE="$OUT_DIR/$LANG"wiki.txt

bunzip2 -c $FILE | ./wiki2text | grep -v '^=' > $OUT_FILE
moses-sent-splitter -u $LANG $OUT_FILE $OUT_FILE
sed -ri '/^.{,$LEN}$/d' $OUT_FILE