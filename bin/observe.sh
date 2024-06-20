#!/usr/bin/bash

function run () {

SAVE="J7_TLS"

python3 observe.py -d save/$SAVE/DuelingDoubleDQNAgent_lr0.0001_model.pack

}

cd ..

source venv/bin/activate

run

deactivate

exit
