import os
#import jax
os.environ["KERAS_BACKEND"] = "torch" # "torch"
device = "cuda:0"
from scipy import signal
SEED = 42
stft_size = 512
window_size = 512
window_shift = 128
sampling_rate = 16000
windows = signal.windows.hann
max_audio_length = 32640
max_video_length = 51
video_frame_size = (112, 112)

# paths
GRID_PATH =  "/Users/jasperkirton/Documents/COG-MHEAR/Grid/"
AVSEC_PATH_CLIPS = "/media/a_hussain_disk/data/avsec_challenge/"
AVSEC_PATH_FEAT = "/home/j_kirtonwingate/data/avsec/"
