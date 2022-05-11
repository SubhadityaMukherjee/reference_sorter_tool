#!/bin/bash
black "." && isort .
# create scripts if there are any notebook files
if [ ! -z $1 ]; then
        git add . && git commit -m $1 && git push
fi
