#!/bin/bash
PATH=$PATH:/home/j_kirtonwingate/.local/bin
echo "$PATH"

pip install -r requirements.txt
#conda install cudatoolkit=11.2
#pip install torch torchaudio torchvision --upgrade

python test_se.py --testset 'AVSEC' 
