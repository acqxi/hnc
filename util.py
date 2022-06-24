# %%
import glob
import os
import re
from math import sqrt
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import rotate as sci_rotate


# %%
def plt_img_bar(
        img_list: Union[ List[ np.ndarray ], np.ndarray ],
        cmps: Optional[ List[ matplotlib.colors.LinearSegmentedColormap ] ] = None,
        title: Optional[ str ] = None,
        names: List[ Optional[ str ] ] = None,
        showAxis: bool = False,
        mode: int = 0,
        save: bool = False,
        clip: Optional[ List[ int ] ] = None ):
    img_len = len( img_list )
    ARRANGE_MODE = [ [ 1, img_len ], [ 1, img_len ], [ int( sqrt( img_len ) ) + 1 ] * 2 ]
    FIGSIZE_MODE = [ ( 5 * img_len, 5 ), ( 5, 5 * img_len ), tuple( [ 4 * int( sqrt( img_len ) ) + 1 ] * 2 ) ]
    names = names or [ None for _ in range( img_len ) ]
    cmps = cmps if isinstance( cmps, list ) else [ cmps or None for _ in range( img_len ) ]
    fig = plt.figure( figsize=FIGSIZE_MODE[ mode ] )
    for idx, ( name, img, cmp ) in enumerate( zip( names, img_list, cmps ) ):

        plt.subplot( *ARRANGE_MODE[ mode ], idx + 1 )
        if clip is not None:
            im = plt.imshow( ( img.clip( clip[ 0 ], clip[ 1 ] ) - clip[ 0 ] ) / ( clip[ 1 ] - clip[ 0 ] ), cmap=cmp )
        else:
            im = plt.imshow( img, cmap=cmp )
        if name:
            plt.title( name )
        if not showAxis:
            plt.axis( 'off' )
        ax = plt.gca()
        divider = make_axes_locatable( ax )
        cax = divider.append_axes( "right", size="5%", pad=0.05 )
        plt.colorbar( im, cax=cax )
    plt.suptitle( title )
    plt.show()
    if save:
        fig.savefig( title + '.png', dpi=300, facecolor='w', edgecolor='none' )


# %%
if __name__ == '__main__':
    path = 'data/test/crop/378-20200706-LAP1-1-0.nii'
    img = sitk.ReadImage( path )
    imgAry = sitk.GetArrayFromImage( img )
    # imgAry = ( imgAry.clip( -120, 120 ) + 120 ) / 240
    # imgAry = sci_rotate( imgAry, 45, axes=( 2, 1 ), order=0)
    plt_img_bar(
        imgAry,
        title=os.path.split( path )[ 1 ],
        names=list( range( 1, len( imgAry )+1 ) ),
        cmps=plt.cm.bone,
        mode=2,
        showAxis=False )
# %%
if __name__ == '__main__':
    path = 'D:\\OneDrive\\0826-LAP-shareData\\20220101-中榮病人\\HNNLAP-378-20200706'
    img = [ sitk.GetArrayFromImage(sitk.ReadImage(p))[0] for p in glob.glob(os.path.join(path,'**\\CT*.dcm'),recursive=True)]
    plt_img_bar(img,
        cmps=plt.cm.bone,
        mode=2,
        showAxis=False )
# %%
