import torch.nn.functional as F
import torch
import pandas as pd
import random
from pathlib import Path

from torch.utils.data import DataLoader


def load_librilight_data(encodec_path, stoks_path, speaker=None):
    speakers = []
    atoks = []
    stoks = []
    for path in Path(encodec_path).rglob('*.encodec'):
        speakers.append(path.parents[1].name)
        atoks.append(path)
        stoks.append(Path(stoks_path) / path.relative_to(encodec_path).with_suffix('.stoks'))
    data = pd.DataFrame(dict(atoks=atoks, stoks=stoks, speaker=speakers))
    if speaker: data = data[data['speaker'] == speaker]
    return data


class SADataset(torch.utils.data.Dataset):
    def __init__(self, data, speakers):
        self.data = data
        self.samples = [(i, j) for i, name in enumerate(data['stoks']) for j in
                        range(torch.load(name, map_location='cpu').shape[0])]
        self.speakers = speakers

    def __len__(self):
        return len(self.samples)

    def S_tokens(self):
        return len(self) * 1500

    def hours(self):
        return len(self) * 30 / 3600

    def __repr__(self):
        return f"Dataset: {len(self)} samples ({len(self.data['speaker'].unique())} speakers), {self.S_tokens()} Stokens, {self.hours():.1f} hours)"

    def __getitem__(self, idx):
        i, j = self.samples[idx]
        row = self.data.iloc[i]
        jA = j * 2250
        Stoks = torch.load(row['stoks'], map_location='cpu')[j]
        Atoks = torch.load(row['atoks'], map_location='cpu')[0, :, jA:jA + 2250]
        return Stoks, F.pad(Atoks, (0, 2250 - Atoks.shape[-1]), value=1026), torch.tensor(self.speakers[row['speaker']])


import math


def load_datasets(
        stoks_path: Path,  # semantic tokens path
        encodec_path: Path,  # encodec tokens path
        subsample: float = 1,  # use a fraction of the files
        val_split: float = 0.001,  # how much data to use for validation
        speaker: str = None  # only load a single speaker id
):
    data = load_librilight_data(encodec_path, stoks_path, speaker=speaker)

    speakers = {id: i for i, id in enumerate(data['speaker'].unique())}

    # select at most 4 frequent speakers from the dataset for the validation set
    # this way, even when subsampling, we avoid the problem of having someone
    # in the validation set that's not in the training set
    val_speakers = data.groupby('speaker').size().sort_values()[-4:].index
    Nval = math.ceil(val_split * len(data) / len(val_speakers))
    val_idxs = []
    for idx in val_speakers:
        val_idxs += list(data[data['speaker'] == idx][:Nval].index)

    train_idxs = list(set(data.index) - set(val_idxs))

    random.seed(0)
    random.shuffle(train_idxs)
    Ntrain = int(len(train_idxs) * subsample)

    val_data, train_data = data.loc[val_idxs], data.loc[train_idxs[:Ntrain]]

    return SADataset(train_data, speakers), SADataset(val_data, speakers)

def get_tts_dataset(semb_path, encodec_path, batch_size):
    s_train_ds, val_ds = load_datasets(semb_path, encodec_path,
                                       subsample=1.0, speaker='6454')

    pin_mem = True
    num_workers = 4
    shuffle = True


    train_set = DataLoader(
        s_train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_mem,
    )

    valid_set = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    return train_set, valid_set