#!/bin/bash
PATH=$PATH:/home/j_kirtonwingate/.local/bin
PATH=$PATH:/usr/bin
echo "$PATH"

#apt-get update && apt-get install libgl1
pip install -r requirements.txt
#apt-get install ffmpeg libavcodec-extra
pip install avconv
pip install ffmpeg
pip install ffmpeg-python
#yes | poetry run ffdl install --add-path
#conda install cudatoolkit=11.2
#pip install torch torchaudio torchvision --upgrade

python train.py --batch_size=1 --lr=1e-4 --mode=regen-joint-training --weighting_denoiser_to_score=0.5 --num_eval_files=1 --condition="post_denoiser" --nockpt
