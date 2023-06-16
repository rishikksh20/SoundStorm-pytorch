import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def variable_random_window(batch, ilens, frames=3000):
    # Codes: [B, T, n_q]
    # Semb: [B, T]
    n_batch = len(batch)

    max_frame = min(ilens)
    

    frames_per_seg = random.randint(frames, max_frame) if max_frame > frames else max_frame
    
    n_q = batch[0][3].shape[-1]




    new_codes = torch.zeros(
        (n_batch, frames_per_seg, n_q), dtype=ilens.dtype, device=ilens.device
    )
    new_semb = torch.zeros(
        (n_batch, frames_per_seg), dtype=ilens.dtype, device=ilens.device
    )

    for i in range(n_batch):
        start = random.randint(0, ilens[i] - frames_per_seg)

        new_codes[i] = torch.from_numpy(batch[i][3][start:start+frames_per_seg]).long()
        new_semb[i] = torch.from_numpy(batch[i][0][start:start+frames_per_seg]).long()
    return new_semb, new_codes


def get_tts_dataset(path, batch_size, train_filelist, ratio=2, valid_filelist=None, valid=False):
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
            self._metadata = [line.strip() for line in f]

        self.ratio = ratio

    def __getitem__(self, index):
        id = self._metadata[index].split(".")[0]

        semb = np.load(os.path.join(self.path, "semantic_code", f"{id}.npy"))  # [L/2]
        codes = np.load(os.path.join(self.path, "codec_code", f"{id}.npy"))   # [L, number_of_quantizers]
        
        semb = np.repeat(semb, self.ratio, axis=0)
        mel_len = min(semb.shape[0], codes.shape[0])
        
        if semb.shape[0] > mel_len:
            semb = semb[:mel_len]
        else:
            codes = codes[:mel_len]

        assert semb.shape[0] == codes.shape[0]
        
        
        return (
            semb,
            id,
            mel_len,
            codes

        )

    def __len__(self):
        return len(self._metadata)



def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode="constant")


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant")


def collate_tts(batch):
    olens = torch.from_numpy(np.array([y[2] for y in batch])).long()
    ids = [x[1] for x in batch]

    # perform padding and conversion to tensor
    semb, codes = variable_random_window(batch, olens)

    # scale spectrograms to -4 <--> 4
    # mels = (mels * 8.) - 4

    return (
        semb,
        codes,
        olens,
        ids
    )