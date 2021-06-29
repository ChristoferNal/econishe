import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

COLUMN_MAINS = 'grid'
COLUMN_DATE = 'localminute'


class PowerDataset(Dataset):
    """Power dataset."""

    def __init__(self, path, device,
                 start_date="2018-01-01", end_date="2018-02-16",
                 should_normalize=True, window_size=50):
        """
        Args:
            path (string): Path to the csv file.
            device(string): the desired device.
            start_date: the first date e.g. '2016-04-01'.
            end_date: the last date e.g. '2017-04-01'.
        """
        self.mmax = None
        self.path = path
        self.device = device
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size

        cols = [COLUMN_DATE, COLUMN_MAINS, device]
        data = pd.read_csv(path, usecols=cols)
        # data.resample('3S')
        # data = data.drop_duplicates(subset=device)
        if self.start_date and self.end_date:
            data = data[(data[COLUMN_DATE] >= self.start_date) & (data[COLUMN_DATE] <= self.end_date)]

        mainchunk = data[COLUMN_MAINS]
        meterchunk = data[self.device]

        # mainchunk, meterchunk = self._align_chunks(mainchunk, meterchunk)
        if should_normalize:
            mainchunk, meterchunk = self._normalize_chunks(mainchunk, meterchunk)
        mainchunk, meterchunk = self._replace_nans(mainchunk, meterchunk)
        mainchunk, meterchunk = self._apply_rolling_window(mainchunk.ravel(), meterchunk.ravel())
        self.mainchunk, self.meterchunk = torch.from_numpy(np.array(mainchunk)), torch.from_numpy(
            np.array(meterchunk))

    def _apply_rolling_window(self, mainchunk, meterchunk):
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk) - self.window_size + 1)[:, None]
        mainchunk = mainchunk[indexer]
        meterchunk = meterchunk[self.window_size - 1:]
        return mainchunk, meterchunk

    def _replace_nans(self, mainchunk, meterchunk):
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        return mainchunk, meterchunk

    def _normalize_chunks(self, mainchunk, meterchunk):
        if self.mmax is None:
            self.mmax = mainchunk.max()

        mainchunk = mainchunk / self.mmax
        meterchunk = meterchunk / self.mmax
        return mainchunk, meterchunk

    def _align_chunks(self, mainchunk, meterchunk):
        mainchunk = mainchunk[~mainchunk.index.duplicated()]
        meterchunk = meterchunk[~meterchunk.index.duplicated()]
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]
        return mainchunk, meterchunk

    def __len__(self):
        return len(self.mainchunk)

    def __getitem__(self, i):
        a, b = self.mainchunk[i].float(), self.meterchunk[i].float()
        # print(f"main {a}, meter {b}")
        return a, b

    def __mmax__(self):
        return self.mmax

# dataset = PowerDataset(path='../data/house_7901.csv', device='microwave1')
# train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
# for i, data in enumerate(train_loader):
#     return i, x, y
