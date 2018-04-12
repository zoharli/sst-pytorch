#!/bin/bash
set -e 
options=""
for((i=1;i<=$#;i++))
do
	options="$options ${!i}"
done
python train.py $options
python test.py $options
python eval.py $options
exit 0
