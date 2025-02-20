from config import *
import logging
import random
from os.path import join, isfile

import cv2
import librosa
import numpy as np
import torchaudio
import torch
from decord import VideoReader
import pytorch_lightning as pl
#from pysepm_edit.pysepm import SNRseg, fwSNRseg
from decord import cpu
from scipy.io import wavfile
from torch.utils.data import Dataset
import json

from tqdm import tqdm

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]




class AVSEDataset(Dataset):
    def __init__(self, files_list, shuffle=False, seed=SEED, subsample=1,
                 clipped_batch=True, window="hann", sample_items=True, time_domain=False, a_only=False, spec_transform=None,
                 normalize="noisy"):
        super(AVSEDataset, self).__init__()
        self.time_domain = time_domain
        self.a_only = a_only
        self.clipped_batch = clipped_batch
        self.files_list = files_list
        self.shuffle = shuffle
        if self.shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.sample_items = sample_items
        self.spec_transform = spec_transform
        #self.normalize_audio = True
        self.normalize = normalize
        self.window = get_window(window, stft_size)
        self.windows = {}

    def __len__(self):
        return 1#len(self.files_list) # debug (1) or not

    def __getitem__(self, idx, raw=False):
        while True:
            try:
                data = {}
                if self.sample_items:
                    clean_file, noise_file, noisy_file, mp4_file, scene_id = random.sample(self.files_list, 1)[0]
                else:
                    clean_file, noise_file, noisy_file, mp4_file, scene_id = self.files_list[idx]
                    x = torchaudio.load(clean_file)[0]
                    #data["noisy_stft"] = self.get_stft(data["noisy"]).T
                    y = torchaudio.load(noisy_file)[0]
                    data["scene"] = scene_id
                if not self.a_only:
                    if raw:
                        data["clean_stft"], data["noisy_stft"], data["video_frames"] = self.get_data(clean_file, noise_file,
                                                                                                    noisy_file, mp4_file, a_only=False) #this is inefficient TODO
                        min_len = min(x.size(-1), y.size(-1))
                        x, y = x[..., : min_len], y[..., : min_len]
                        return x, y, data["video_frames"]
                    else:
                        data["clean_stft"], data["noisy_stft"], data["video_frames"] = self.get_data(clean_file, noise_file,
                                                                                                    noisy_file, mp4_file, a_only=False)
                        data["clean_stft"] = torch.Tensor(data["clean_stft"])
                        data["noisy_stft"] = torch.Tensor(data["noisy_stft"])
                        data["video_frames"] = torch.Tensor(data["video_frames"])
                        X, Y = self.spec_transform(data["clean_stft"]), self.spec_transform(data["noisy_stft"])
                        return X, Y, data["video_frames"]
                else:
                    data["clean_stft"], data["noisy_stft"] = self.get_data(clean_file, noise_file,
                                                        noisy_file, mp4_file, a_only=True)
                    data["clean_stft"] = torch.Tensor(data["clean_stft"])
                    data["noisy_stft"] = torch.Tensor(data["noisy_stft"])
            except Exception as e:
                logging.error("Error in loading data: {}".format(e))

    def load_wav(self, wav_path):
        audio = wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)
        #normfac = max(abs(audio))
        #if self.normalize_audio:
        #    audio = audio / normfac
        return audio

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def get_stft(self, audio):
        window = self._get_window(audio)
        return torch.stft(audio, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=window, return_complex=True,
                            center=True)[:256, :]

    def  get_audio_features(self, audio):
        return self.get_stft(torch.Tensor(audio))
        #return np.abs(self.get_stft(audio)).transpose(1, 0).astype(np.float32)

    def get_fwssnr(self, clean, noisy, fs=16000):
        return fwSNRseg(clean, noisy, fs) # arr, np.mean(arr)

    def get_snr_features(self, clean, noisy):
        return self.get_fwssnr(clean, noisy)

    def get_ntype(self, md, scene_id):
        d = next(item for item in md if item["scene"] == scene_id)
        n_t = d["interferer"]["type"]
        return n_t

    def get_data(self, clean_file, noise_file, noisy_file, mp4_file, a_only=False):
        if isfile(clean_file):
            clean = self.load_wav(clean_file)

        else:
            # clean file for test set is not available
            clean = np.zeros(noisy.shape)
        noisy = self.load_wav(noisy_file)
        if not a_only:
            vr = VideoReader(mp4_file, ctx=cpu(0))
            if self.clipped_batch:
                if clean.shape[0] > max_audio_length:
                    clip_idx = random.randint(0, clean.shape[0] - max_audio_length)
                    video_idx = int((clip_idx / 16000) * 25)
                    clean = clean[clip_idx:clip_idx + max_audio_length]
                    noisy = noisy[clip_idx:clip_idx + max_audio_length]
                else:
                    video_idx = -1
                    clean = np.pad(clean, pad_width=[0, max_audio_length - clean.shape[0]], mode="constant")
                    noisy = np.pad(noisy, pad_width=[0, max_audio_length - noisy.shape[0]], mode="constant")
                if len(vr) < max_video_length:
                    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
                else:
                    max_idx = min(video_idx + max_video_length, len(vr))
                    frames = vr.get_batch(list(range(video_idx, max_idx))).asnumpy()
                bg_frames = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]
                bg_frames = np.array([cv2.resize(bg_frames[i], video_frame_size) for i in range(len(bg_frames))]).astype(
                    np.float32)
                bg_frames /= 255.0
                if len(bg_frames) < max_video_length:
                    bg_frames = np.concatenate(
                        (bg_frames,
                         np.zeros((max_video_length - len(bg_frames), video_frame_size[0], video_frame_size[1])).astype(bg_frames.dtype)),
                        axis=0)
            else:
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
                bg_frames = np.array(
                    [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
                bg_frames = np.array([cv2.resize(bg_frames[i], video_frame_size) for i in range(len(bg_frames))]).astype(
                    np.float32)
                bg_frames /= 255.0
            if self.normalize == "noisy":
                normfac = np.max(np.abs(noisy))
            elif self.normalize == "clean":
                normfac = np.max(np.abs(clean))
            elif self.normalize == "not":
                normfac = 1.0
            x = clean / normfac
            y = noisy / normfac
            return self.get_audio_features(x), self.get_audio_features(y), bg_frames
        else:
            if self.clipped_batch:
                if clean.shape[0] > max_audio_length:
                    clip_idx = random.randint(0, clean.shape[0] - max_audio_length)
                    clean = clean[clip_idx:clip_idx + max_audio_length]
                    noisy = noisy[clip_idx:clip_idx + max_audio_length]
                else:
                    clean = np.pad(clean, pad_width=[0, max_audio_length - clean.shape[0]], mode="constant")
                    noisy = np.pad(noisy, pad_width=[0, max_audio_length - noisy.shape[0]], mode="constant")
            if self.time_domain:
                return clean, noisy
            return self.get_audio_features(clean)[..., np.newaxis], self.get_audio_features(noisy)[..., np.newaxis]


class AVSEChallengeDataModule(pl.LightningDataModule):
    def __init__(self, data_root=AVSEC_PATH_CLIPS, feat_root=AVSEC_PATH_FEAT, batch_size=1, time_domain=False, a_only=False,
                 gpu=True, window="hann", shuffle=False, **kwargs):
        super(AVSEChallengeDataModule, self).__init__()
        self.data_root = data_root
        self.feat_root = feat_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.time_domain = time_domain
        self.window = get_window(window, stft_size)
        self.windows = {}
        self.a_only = a_only
        self.gpu = gpu
        self.kwargs = kwargs
        self.spec_abs_exponent = 0.5
        self.spec_factor = 0.15

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset_batch = AVSEDataset(self.get_files_list(join(self.data_root, "train"), self.feat_root, "train"),
                                               time_domain=self.time_domain, a_only=self.a_only,  spec_transform=self.spec_fwd,
                                                   shuffle=self.shuffle)
            self.valid_dataset_batch = AVSEDataset(self.get_files_list(join(self.data_root, "train"), self.feat_root, "train"),
                                             time_domain=self.time_domain,a_only=self.a_only,  spec_transform=self.spec_fwd,
                                                   shuffle=self.shuffle)
            self.valid_set = AVSEDataset(self.get_files_list(join(self.data_root, "dev"), self.feat_root, "dev"),
                                       clipped_batch=True, sample_items=False, time_domain=self.time_domain, a_only=self.a_only,
                                         spec_transform=self.spec_fwd)
        # !TODO Uncomment this for test set
            #self.test_dataset = AVSEDataset(self.get_files_list(join(data_root, "eval"), test_set=True),
            #                    self.get_metadata(data_root, "eval"),sample_items=False,
            #                    clipped_batch=False, time_domain=time_domain)

    @staticmethod
    def get_files_list(data_root, feat_root, set, test_set=False):
        files_list = []
        with open(join(feat_root, ('metadata/scenes.' + set + '.json'))) as f:
            md = json.load(f)
        for file in os.listdir(join(data_root, "scenes")):
            if file.endswith("mixed.wav"):
                files = (join(data_root, "scenes", file.replace("mixed", "target")),
                         join(data_root, "scenes", file.replace("mixed", "interferer")),
                         join(data_root, "scenes", file),
                         join(feat_root, "lips", file.replace("_mixed.wav", "_silent.mp4")),
                         file.replace("_mixed.wav", "")
                         )
                scene_id = files[4]
                d = next(item for item in md if item["scene"] == scene_id)
                n_t = d["interferer"]["type"]
                snr = d["SNR"]
                #if n_t == "noise" and snr > 0:
                if not test_set:
                    if all([isfile(f) for f in files[:-1]]):
                        files_list.append(files)
                else:
                    files_list.append(files)
        return files_list

    @staticmethod
    def get_metadata(data_root, set):
        with open(join(data_root, ('metadata/scenes.' + set + '.json'))) as f:
            md = json.load(f)
        return md

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def get_stft(self, audio):
        window = self._get_window(audio)
        return torch.stft(audio, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=window, return_complex=True,
                            center=True)[:, :256, :]

    def istft(self, spec, length):
        window = self._get_window(spec)
        spec = torch.nn.functional.pad(spec, (0, 0, 0, 1), "constant", 0) # pad with 0s the last frequency dimension
        return torch.istft(spec, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=window,
                            center=True, length=length)

    def spec_fwd(self, spec):
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            #print("running fwd")
            spec = torch.abs(spec)**e * torch.exp(1j * torch.angle(spec)) # spec transform
        return spec * self.spec_factor

    def spec_back(self, spec):
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            #print("running back")
            spec = torch.abs(spec)**(1/e) * torch.exp(1j * torch.angle(spec))
        return spec


    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(self.train_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        assert len(self.valid_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(self.valid_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--format", type=str, default="avsec", choices=["voxceleb2_SE", 'voxceleb2_SS', "avsec"],
                            help="File paths follow the DNS data description.")
        parser.add_argument("--base_dir", type=str, default="/Users/jasperkirton/Documents/COG-MHEAR/AVSEC2/",
                            # "/mnt/scratch/datasets/new_avspeech" , "/mnt/work2/users/cyong/storm/new_avspeech"
                            help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, "
                                 "each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--use_sync_encoder", action="store_true",
                            help="enable this option when the denoiser is 'ncsnpp_crossatt_sync'")  # 이거 잘 안 먹히는듯? 어떻게 고쳐야할깜
        parser.add_argument("--batch_size", type=int, default=1, help="The batch size. 32 by default.")
        parser.add_argument("--n_fft", type=int, default=510,
                            help="Number of FFT bins. 510 by default.")  # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256,
                            help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="sqrthann",
                            help="The window function to use for the STFT. 'sqrthann' by default.")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15,
                            help="Factor to multiply complex STFT coefficients by.")  ##### In Simon's current impl, this is 0.15 !
        parser.add_argument("--spec_abs_exponent", type=float, default=1,
                            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). "
                                 "1 by default; set to values < 1 to bring out quieter features.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy",
                            help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")
        parser.add_argument("--shuffle", action="store_true", help="Shuffle data or not")

        return parser


if __name__ == '__main__':
    dm = AVSEChallengeDataModule(gpu=False)
    dm.setup(stage="fit")
    print(dm.valid_set.__getitem__(0,raw=False))

    #print(AVSEDataset.get_data(1))