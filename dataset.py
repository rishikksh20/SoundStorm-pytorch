import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def variable_random_window(semb, codes, ilens, min_frame=40):
    # Codes: [B, T, n_q]
    # Semb: [B, T]
    n_batch = len(semb)

    max_frame = min(x.size(1) for x in codes)

    frames_per_seg = random.randint(min_frame, max_frame) if max_frame > min_frame else max_frame
    n_q = codes.shape[-1]

    assert codes.shape[1] == semb.shape[1]



    new_codes = torch.zeros(
        (n_batch, frames_per_seg, n_q), dtype=codes.dtype, device=codes.device
    )
    new_semb = torch.zeros(
        (n_batch, frames_per_seg), dtype=semb.dtype, device=semb.device
    )

    for i in range(n_batch):
        start = random.randint(0, ilens[i] - frames_per_seg - 1)

        new_codes[i] = codes[i, start:start+frames_per_seg]
        new_semb[i] = semb[i, start:start+frames_per_seg]
    return new_semb, new_codes


def get_tts_dataset(path, batch_size, train_filelist, valid_filelist=None, ratio=1, valid=False):
    if valid:
        file_ = valid_filelist
        pin_mem = False
        num_workers = 0
        shuffle = False
    else:
        file_ = train_filelist
        pin_mem = True
        num_workers = 4
        shuffle = True
    train_dataset = TTSDataset(
        path, file_, ratio
    )

    train_set = DataLoader(
        train_dataset,
        collate_fn=collate_tts,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_mem,
    )
    return train_set


class TTSDataset(Dataset):
    def __init__(self, path, file_, ratio):
        self.path = path
        with open("{}".format(file_), encoding="utf-8") as f:
            self._metadata = [line.strip().split("|") for line in f]

        self.ratio = ratio

    def __getitem__(self, index):
        id = self._metadata[index][0].split(".")[0]

        semb = np.load(os.path.join(self.path, "semantic_code", f"{id}.npy"))  # [B, L/2]
        codes = np.load(os.path.join(self.path, "codec_code", f"{id}.npy"))   # [B, L, number_of_quantizers]

        semb = np.repeat(semb, self.ratio, axis=1)
        assert semb.shape[-1] == codes.shape[-2]

        mel_len = semb.shape[1]

        return (
            semb,
            id,
            mel_len,
            codes

        )  # codes [L, n_q]

    def __len__(self):
        return len(self._metadata)



def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode="constant")


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant")


def collate_tts(batch):
    olens = torch.from_numpy(np.array([y[2].shape[0] for y in batch])).long()
    ids = [x[1] for x in batch]

    semb = torch.from_numpy(np.array([y[0] for y in batch])).long()
    codes  = torch.from_numpy(np.array([y[-1] for y in batch])).long()
    # perform padding and conversion to tensor
    semb, codes = variable_random_window(semb, codes, olens)

    # scale spectrograms to -4 <--> 4
    # mels = (mels * 8.) - 4

    return (
        semb,
        codes,
        olens,
        ids
    )