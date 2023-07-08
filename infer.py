import os
import numpy as np
from einops import rearrange
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from SoundStorm import SoundStorm
from encodec import EncodecModel
from scipy.io.wavfile import write

DEVICE = "cuda:0"

CHKPT = "./checkpoints/transformer_step_40000.pt"




encodec_ = EncodecModel.encodec_model_24khz()
encodec_.normalize = False
encodec_.set_target_bandwidth(6.0)
encodec_ = encodec_.cuda()
target_sample_hz = 24000
torch_args = True

path = None
if __name__ == '__main__':

    #cond = "./data/semantic_code/26_495_000007_000003.npy"
    #codes = "./data/codec_code/26_495_000007_000003.npy"
    cond = "../../../data/whisperspeech/whisperspeech/librilight/stoks/large/6454/a_christmas_miscellany_2018_1807_librivox_64kb_mp3/christmasmiscellany2018_02_various_64kb.stoks"
    codes = "../../../data/whisperspeech/whisperspeech/librilight/encodec-6kbps/large/6454/a_christmas_miscellany_2018_1807_librivox_64kb_mp3/christmasmiscellany2018_02_various_64kb.encodec"


    j = 2
    if torch_args:
        jA = j * 2250
        Stoks = torch.load(cond, map_location=DEVICE)[j]
        Atoks = torch.load(codes, map_location=DEVICE)[0, :, jA:jA + 2250]
        codes = F.pad(Atoks, (0, 2250 - Atoks.shape[-1]), value=1026)



        prompt = codes[:, :750].clone().detach().unsqueeze(0).cuda()
        semb = Stoks.unsqueeze(0)
        b, n = semb.shape
        semb = semb.reshape(b, n // 2, 2)
        semb = semb.repeat_interleave(2, -1)[:, :, :3]
        semb[:, :, 1] = 1025
        semb = semb.reshape(b, n // 2 * 3)
        assert semb.shape[-1] == codes.shape[-1]
        print("shape of semb", semb.shape)
        print("shape of codes", codes.shape)

        # codec = rearrange(codes.unsqueeze(0), 'b q n -> q b n')
        # emb = encodec_.quantizer.decode(codec)
        #
        # audio = encodec_.decoder(emb).squeeze()
        # print("shape of audio", audio.shape)
        # write("org.wav", target_sample_hz, audio.detach().cpu().numpy())
    else:
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
    model = SoundStorm(dim=768, heads=8, linear_units=3072, num_blocks=8).to(DEVICE)

    chkpt = torch.load(CHKPT, map_location=DEVICE)
    model.load_state_dict(chkpt['model'])
    model.eval()

    codec = model.generate(semb, prompt)        # [B, q, n]

    np.save("out.npy", codec.cpu().detach().numpy())

    codes = rearrange(codec, 'b q n -> q b n')
    emb = encodec_.quantizer.decode(codes)

    audio = encodec_.decoder(emb)

    write("out_conf2.wav", target_sample_hz, audio.detach().cpu().numpy())