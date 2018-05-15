#!/bin/bash
idx=0
while true
do
	options=$(python hyper_param.py --idx=$idx)
	if [ $? == 1 ]
	then
		time=$(date +%Y-%m-%d-%H:%M:%S)
		echo "=========>$time:Grid search finished." >>grid_search_${1}.txt
		exit 
	fi
	options="$options --train_id=$idx"
	time=$(date +%Y-%m-%d-%H:%M:%S)
	echo "==========>$time:Now training with hyper parameters:$options" >>grid_search_${1}.txt
	./run.sh $options >>grid_search_${1}.txt
	let "idx++"
done
