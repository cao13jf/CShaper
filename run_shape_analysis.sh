#!/usr/bin/env bash

echo "" > shape_analysis_fail.txt
for config_file in "./ConfigMemb/ShapeCnfigs"/*
do
    echo $config_file
    error=true
    {
        /bin/python3.6 shape_analysis.py $config_file
        } || {
         error=true
         }
#
    if [ $error ]
    then
        echo $config_file >> shape_analysis_fail.txt
    fi
done
