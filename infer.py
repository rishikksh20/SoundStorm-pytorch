import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from SoundStorm import SoundStorm

DEVICE = "cuda:0"

CHKPT = "./checkpoints/transformer_epoch_100.pt"

path = None
if __name__ == '__main__':

    cond = "./data/semantic_code/26_495_000007_000003.npy"
    codes = "./data/codec_code/26_495_000007_000003.npy"

    semb = np.load(cond)  # [L/2]
    codes = np.load(codes)  # [L, number_of_quantizers]

    semb = np.repeat(semb, 2, axis=0)
    mel_len = min(semb.shape[0], codes.shape[0])

    if semb.shape[0] > mel_len:
        semb = semb[:mel_len]
    else:
        codes = codes[:mel_len]

    assert semb.shape[0] == codes.shape[0]
    prompt = torch.from_numpy(codes[:600, :]).T.unsqueeze(0).to(DEVICE)
    semb = torch.from_numpy(semb).unsqueeze(0).to(DEVICE)
    model = SoundStorm().to(DEVICE)

    chkpt = torch.load(CHKPT, map_location=DEVICE)
    model.load_state_dict(chkpt)
    model.eval()

    codec = model.generate(semb, prompt)        # [B, q, n]

    np.save("out.npy", codec.cpu().detach().numpy())