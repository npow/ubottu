#!/bin/bash

export CLASSPATH=.:`pwd`/libs/commons-lang3-3.4.jar
echo $CLASSPATH
pypy tokenize.py
