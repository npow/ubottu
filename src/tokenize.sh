#!/bin/bash

export CLASSPATH=.:../libs/commons-lang3-3.4.jar:../libs
echo $CLASSPATH
python tokenize_data.py
