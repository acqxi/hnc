#%%
import glob
import os
import random as rd
import re
from operator import xor
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.transform as sk_xfmr
import torch
import torchvision.transforms as xfmr
from scipy.ndimage import rotate as sci_rotate
from torch.utils.data import DataLoader, Dataset

from util import plt_img_bar

#%%
TEST_LIST = [
    '152-20191018-LAP2', '141-20191107-LAP3', '366-20200214-LAP1', '244-20190828-LAP2', '146-20190520-LAP1',
    '109-20190614-LAP4', '100-20190725-LAP4', '363-20190429-LAP3', '109-20190614-LAP3', '353-20190318-LAP7',
    '237-20190606-LAP4', '227-20191021-LAP1', '338-20200115-LAP2', '134-20190414-LAP1', '029-20200817-LAP1',
    '326-20200304-LAP2', '152-20191018-LAP5', '178-20190225-LAP1', '299-20200803-LAP1', '263-20190211-LAP3',
    '213-20191220-LAP1', '219-20190107-LAP2', '202-20190911-LAP1', '130-20190425-LAP5', '240-20190622-LAP1',
    '370-20200722-LAP2', '352-20190225-LAP4', '117-20191028-LAP3', '281-20190227-LAP1', '372-20200722-LAP1',
    '248-20191113-LAP1', '160-20191115-LAP2', '156-20190807-LAP3', '121-20190522-LAP1', '194-20190708-LAP1',
    '263-20190211-LAP2', '186-20190401-LAP1', '122-20190523-LAP1', '186-20190401-LAP2', '220-20190222-LAP1',
    '187-20190428-LAP1', '287-20200217-LAP1', '149-20190603-LAP2', '056-20190509-LAP3', '117-20191028-LAP1',
    '326-20200304-LAP1', '152-20191018-LAP1', '123-20191106-LAP1', '234-20190327-LAP2', '171-20190131-LAP1',
    '189-20190523-LAP1', '353-20190318-LAP6', '181-20190227-LAP2', '056-20190509-LAP2', '245-20191004-LAP1',
    '158-20181227-LAP2', '219-20190107-LAP1', '149-20190603-LAP1', '325-20200427-LAP2', '280-20191121-LAP1',
    '236-20190613-LAP2', '222-20190329-LAP4', '166-20190801-LAP1', '152-20191018-LAP8', '194-20190708-LAP4',
    '234-20190327-LAP3', '219-20190107-LAP3', '141-20191107-LAP2', '152-20191018-LAP6', '183-20190301-LAP2',
    '217-20200422-LAP3', '078-20200103-LAP1', '195-20190710-LAP3', '194-20190708-LAP6', '029-20200817-LAP2',
    '300-20200907-LAP2', '181-20190227-LAP1', '263-20190211-LAP1', '366-20200214-LAP3', '001360306A-LAP2', '002553910A-LAP2',
    '002017449G-LAP2', '002719765F-LAP1', '002017449G-LAP1', '002579381B-LAP1', '001417900B-LAP2', '001669360A-LAP3',
    '002719765F-LAP1', '002715797H-LAP4', '002685520J-LAP1', '001669360A-LAP2', '002713008B-LAP1', '002700831H-LAP3',
    '001801440J-LAP2', '001818273F-LAP1', '002691966E-LAP1', '002677033C-LAP2', '002110208J-LAP3', '002700831H-LAP2',
    '000786930J-LAP2', '002110208J-LAP1', '001203959H-LAP1', '000786930J-LAP1', '001417900B-LAP1', '001801440J-LAP1',
    '002662157I-LAP1', '000474327F-LAP1', '002685520J-LAP2', '001417900B-LAP3', '000257462B-LAP1'
]


class SingleCenterTumorDataset( Dataset ):

    def __init__(
        self,
        dataDir: Union[ str, os.PathLike ],
        isTrain: bool,
        testList: List[ str ] = TEST_LIST,
        huCenter: Optional[ int ] = None,
        huRange: Optional[ int ] = None,
        xfmr: Optional[ xfmr.Compose ] = None,
        flip: bool = False,
        balance: bool = False,
        isExtend: bool = True,
        isOneHot: bool = False,
    ):
        self.origin_paths = glob.glob( os.path.join( dataDir, '*.nii.gz' ) )
        self.paths, self.nones, self.nms, self.enes = [], [], [], []
        self.xfmr = xfmr
        self.flip = flip
        self.balance = balance
        self.onehot = isOneHot
        self.ishu = True if huCenter is not None else False

        for path in self.origin_paths:
            _, file_name = os.path.split( path )
            if xor( isTrain, file_name[ :-11 ] in testList ):
                if not isExtend and len( file_name.split( '-' ) ) == 4:
                    continue
                self.paths += [ path ]
                if balance:
                    ttype = file_name[ -10:-7 ]
                    if ttype == '0-0':
                        self.nones += [ path ]
                    elif ttype == '1-0':
                        self.nms += [ path ]
                    elif ttype == '1-1':
                        self.enes += [ path ]

        self.times = ( 2 if flip else 1 ) * ( 3 if balance else 1 )
        self.nums = max( len( self.nones ), len( self.nms ), len( self.enes ) ) if balance else len( self.paths )

        print( f"load {dataDir} in {'train' if isTrain else 'test'} mode with {self.times} x {self.nums} data" )
        if balance:
            print( f"{len(self.nones)=}, {len(self.nms)=}, {len(self.enes)=}" )
            rd.shuffle( self.nones )
            rd.shuffle( self.nms )
            rd.shuffle( self.enes )
            self.b_type = 0
            self.b_idx = 0
            self.b_paths = [ self.nones, self.nms, self.enes ]

        if huCenter:
            self.hu_center, self.hu_range, self.hu1, self.hu2 = huCenter, huRange * 2, huCenter - huRange, huCenter + huRange
            print( f" config: hu? {huCenter} +- {huRange} ({self.hu1} -> {self.hu2})" )

    def __len__( self ):
        return self.times * self.nums

    def __getitem__( self, idx ):

        angle = 0
        path = self.paths[ idx % self.nums ]

        if self.balance:
            angle = self.b_idx * 180 / self.nums
            path = self.b_paths[ self.b_type ][ self.b_idx % len( self.b_paths[ self.b_type ] ) ]
            self.b_type = ( self.b_type + 1 ) % 3
            self.b_idx += 1

        if self.flip and ( self.b_idx if self.balance else idx ) > self.nums:
            angle += 180

        sitk_img = sitk.ReadImage( path )
        img = sitk.GetArrayFromImage( sitk_img )
        assert img.shape[ 0 ] <= 32 and img.shape[ 1 ] <= 118 and img.shape[ 2 ] <= 118, f"{img.shape=} too big @ {path}"

        if self.ishu:
            img = ( img.clip( self.hu1, self.hu2 ) - self.hu1 ) / self.hu_range
        else:
            img = ( img - np.min( img ) ) / ( np.max( img ) - np.min( img ) )

        img = sci_rotate( img, angle, axes=( 2, 1 ), order=0, reshape=False )
        assert img.shape[ 0 ] <= 32 and img.shape[ 1 ] <= 118 and img.shape[ 2 ] <= 118, f"{img.shape=} too big @ {path}"

        # newimg = np.zeros( ( 32, 118, 118 ) )
        # newimg[ 16 - len( img ) // 2:16 - len( img ) // 2 + len( img ) + 1 ] = img
        # img = newimg
        x1, y1, z1 = ( 32 - img.shape[ 0 ] ) // 2, ( 118 - img.shape[ 1 ] ) // 2, ( 118 - img.shape[ 2 ] ) // 2
        x2, y2, z2 = 32 - x1 - img.shape[ 0 ], 118 - y1 - img.shape[ 1 ], 118 - z1 - img.shape[ 2 ]
        assert img.shape[ 0 ] + x1 + x2 <= 32, f"{img.shape[0]} + {x1} + {x2} not <= 32 @ {path}"
        assert img.shape[ 1 ] + y1 + y2 <= 118, f"{img.shape[1]} + {y1} + {y2} not <= 118 @ {path}"
        assert img.shape[ 2 ] + z1 + z2 <= 118, f"{img.shape[2]} + {z1} + {z2} not <= 118 @ {path}"
        img = np.pad(
            img, [ ( max( 0, x1 ), max( 0, x2 ) ), ( max( 0, y1 ), max( 0, y2 ) ), ( max( 0, z1 ), max( 0, z2 ) ) ],
            mode='constant',
            constant_values=0 )
        img = np.expand_dims( img, axis=0 )
        assert img.shape == ( 1, 32, 118, 118 ), (
            f"expects (1,32,118,118) but got {img.shape} @ {path}",
            f"{x1=}, {x2=}, {y1=}, {y2=}, {z1=}, {z2=}, {sitk.GetArrayFromImage( sitk_img ).shape=}" )
        img: torch.Tensor = self.xfmr( torch.from_numpy( img ) )

        mn: int = int( path[ -10 ] )
        ene: int = int( path[ -8 ] )

        if self.onehot:
            flag = mn + ene  # 00 : 0, 10 : 1, 11 : 2
            flags = torch.zeros( 3, dtype=int )
            flags[ flag ] = 1
        else:
            flags = torch.from_numpy( np.array( [ mn, ene ] ) )

        return img.float().contiguous(), flags, path


# %%
if __name__ == '__main__':
    data_dir = os.path.join( 'data', 'test', 'crop' )
    batch_size = 16
    n_worker = 0

    img_xfmr = xfmr.Compose( [ xfmr.Normalize( [ 0.5 ], [ 0.5 ] ) ] )

    train_set = SingleCenterTumorDataset( dataDir=data_dir, isTrain=True, xfmr=img_xfmr, flip=True, balance=True )
    valid_set = SingleCenterTumorDataset( dataDir=data_dir, isTrain=False, xfmr=img_xfmr, flip=True, balance=True )

    train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker )
    valid_loader = DataLoader( valid_set, batch_size=batch_size, shuffle=True, num_workers=n_worker )

    dataloaders = { 'train': train_loader, 'valid': valid_loader}

# %%
if __name__ == '__main__':
    for idx, ( imgs, flags, names ) in enumerate( train_loader ):
        plt_img_bar(
            imgs[ :, 0, 16 ],
            title='',
            names=[ os.path.split( name )[ 1 ] for name in names ],
            cmps=plt.cm.bone,
            mode=2,
            showAxis=False )
        if idx > 5:
            break
        # print( idx, imgs.shape )
# %%
