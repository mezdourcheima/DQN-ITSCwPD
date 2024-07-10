#!/usr/bin/bash

function run () {

ALGO="SotlBaseline"

python3 play.py -player $ALGO
#python3 play.py -player SotlBaseline  -log true -log_s 10 -log_dir "./logs/test/" > output.log 2>&1

}

cd ..

source venv/bin/activate

run

deactivate

exit
