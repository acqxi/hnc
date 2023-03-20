import functools
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import skimage.transform as sk_xfm
import torch
import torchvision.transforms as xfms
from torch.utils.data import Dataset


def RandomFlip3D(p=0.5, axis=0):
    """ fb=2, ud=0, rl=1 """
    return xfms.RandomApply([xfms.Lambda(lambda x: x.flip(axis))], p=p)


def Random90Rotate3D(p=0.5, k=1, axis=0):
    """ k=1,2,3; x=2, (preserved necessary) y=0, z=1 """
    return xfms.RandomApply([xfms.Lambda(lambda x: x.rot90(k, (1 if axis == 2 else 2, 1 if axis == 0 else 0)))], p=p)


def RawToTensor():
    return xfms.Lambda(lambda x: torch.from_numpy(x).float())


def RawAddUpperDim():
    return xfms.Lambda(lambda x: x.unsqueeze(0))


class VGHTC_HNC158(Dataset):
    def __init__(
            self,
            dataPath: Union[Path, str],
            xlsxName: str = 'vghtc_hnc158',
            train: bool = True,
            testGroup: int = 4,
            balance: bool = True,
            HU: Tuple[int, int] = (-340, 460),
            outputSize: Tuple[int, int, int] = None,
            preservedSize: Tuple[int, int, int] = None,
            invariantSize: Tuple[int, int, int] = None,
            outputMode: str = 'preserved',  # preserved, invariant, both
            transform=None):
        self.data_path = dataPath if isinstance(dataPath, Path) else Path(dataPath)
        self.xlsx_name = xlsxName
        self.train = train
        self.test_group = testGroup
        self.balance = balance
        self.hu = HU
        self.output_mode = outputMode
        self.preserved_size = preservedSize or outputSize or (18, 130, 130)
        self.invariant_size = invariantSize or outputSize or (32, 32, 32)
        self.transform = transform

        self.df = pd.read_excel(self.data_path / f'{self.xlsx_name}.xlsx', sheet_name='data', index_col=0)

        condition = self.df['group'] != self.test_group if self.train else self.df['group'] == self.test_group
        self.file_names = list(self.df[condition].index)

        lnm_count = self.df[condition].groupby('LNM')['LNM'].count()
        print(
            f'VGHTC_HNC158: {len(self.file_names)} of {len(self.df)} files loaded for t{"rain" if self.train else "est"} set.',
            f' test group: {self.test_group}',
            f'\n\tLNM ratio = pos:neg = {lnm_count["pos"]}:{lnm_count["neg"]} = {lnm_count["pos"]/lnm_count["neg"]:.2f}')

        if balance:
            self.file_names.extend(list(self.df[(condition) & (self.df['LNM'] == 'pos')].index))
            print(
                f'after balance, train set size up to: {len(self.file_names)}',
                f'\n\tLNM ratio = pos:neg = {lnm_count["pos"]*2}:{lnm_count["neg"]} = {lnm_count["pos"]*2/lnm_count["neg"]:.2f}'
            )

        if outputMode not in ['preserved', 'invariant', 'both']:
            raise ValueError(f'outputMode should be one of preserved, invariant, both, but got {outputMode}')
        elif outputMode == 'both':
            print(f'outputMode: {outputMode}, size: preserved: {self.preserved_size}, invariant: {self.invariant_size}')
        else:
            print(
                f'outputMode: {outputMode}, size: {self.preserved_size if outputMode == "preserved" else self.invariant_size}')

    def __len__(self):
        return len(self.file_names)

    def _add_transform(func: Callable):  # type: ignore
        @functools.wraps(func)
        def wrapper(self, img):
            if self.output_mode == 'both' and self.transform:
                return RawAddUpperDim()(RawToTensor()(func(self, np.array(img))))
            elif self.transform:
                return self.transform(func(self, img))
            else:
                return func(self, img)

        return wrapper

    @_add_transform
    def invariant(self, img):
        img = sk_xfm.resize(img, self.invariant_size, mode="constant", preserve_range=True)
        return img

    @_add_transform
    def preserved(self, img):
        new_img = np.zeros(self.preserved_size) - 1
        # put img in the center of empty
        dz, dx, dy = img.shape
        Z, X, Y = self.preserved_size
        startX, startY, startZ = (X - dx) // 2, (Y - dy) // 2, (Z - dz) // 2
        new_img[startZ:startZ + dz, startX:startX + dx, startY:startY + dy] = img
        return new_img

    def load_clip_norm(self, idx):
        img = np.load(self.data_path / f'{self.file_names[idx]}.npy')
        img = np.clip(img, self.hu[0] + 1000, self.hu[1] + 1000)
        # normalize
        img = 2 * (img - (self.hu[0] + 1000)) / (self.hu[1] - self.hu[0]) - 1
        return img

    def labels(self, idx):
        return (
            1 if self.df.loc[self.file_names[idx], 'LNM'] == 'pos' else 0,
            self.file_names[idx],
            self.df.loc[self.file_names[idx], 'spacing'],
        )

    def __getitem__(self, idx):
        img = self.load_clip_norm(idx)
        img_methods = {'preserved': self.preserved, 'invariant': self.invariant}

        if self.output_mode == 'both':
            if self.transform:
                img = self.transform(img)
            if len(img.shape) == 4:
                img = img[0]
            return self.preserved(img), self.invariant(img), *self.labels(idx)
        else:
            return img_methods[self.output_mode](img), *self.labels(idx)
