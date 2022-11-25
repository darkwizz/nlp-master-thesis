#!/bin/bash


LOG_FILE=$1
LOG_BASENAME=$( basename $LOG_FILE )
RES_DIR_PATH="$( dirname $LOG_FILE )/target-csv"

mkdir -p $RES_DIR_PATH

RES_FILE="$RES_DIR_PATH/$LOG_BASENAME.csv"
echo "eval_loss,epoch" >> $RES_FILE

for line in $( grep 'eval_loss' $LOG_FILE | tr -d ' ' | tr "'" "\"" )
do
    py_out=$( python -c "import json; grep_json = json.loads('$line'); print(grep_json['eval_loss'], grep_json['epoch'], sep=',')" )
    echo $py_out >> $RES_FILE
done