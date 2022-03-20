#%%
import glob
import os
from functools import reduce
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
# %cd root/HNC_ENE
# %%
filefolder = os.path.abspath( os.path.dirname( __file__ ) )
folder_paths = glob.glob( os.path.join( filefolder, 'rawdata', 'new_data', 'HNNLAP*' ) )
folder_paths
# %%
recordpath = os.path.join( filefolder, 'rawdata', 'new_data', 'contour.txt' )


def read_contour_txt( recordpath: str, sept: str = ' ' ):
    recorddata = {}
    tempdata = []
    with open( recordpath, 'r' ) as f:
        for line in f.readlines():
            words = [ word.replace( '\n', '' ) for word in line.split( sept ) if word and word != '\n' ]
            # print( words )
            if not words:
                continue
            if words[ 0 ].startswith( 'HNNLAP' ):
                tempdata = recorddata[ words[ 0 ].replace( 'CT', '' )[ -12: ] ] = {}
            elif words[ 0 ].startswith( 'LAP' ):
                tempdata[ words[ 0 ] ] = [ 1 if i == 'pos' else 0 for i in words[ 1: ] ]
    return recorddata


recorddata = read_contour_txt( recordpath, '\t' )
recorddata


# %%
class Tumor():

    def __init__( self, name ) -> None:
        self.name = name
        self.data = {}
        self.d = np.zeros( 3000, np.int32 )
        self.x_min = 999
        self.x_max = 0
        self.y_min = 999
        self.y_max = 0
        self.tumor3d = []
        self.tumor3dc = []
        self.tumor3dcc = np.zeros( ( 20, 118, 118 ) )

    def crop( self ):
        for loc, v in self.data.items():
            ( xmin, ymin ), ( xmax, ymax ) = v[ 'crop' ]
            self.x_min = min( self.x_min, xmin )
            self.x_max = max( self.x_max, xmax )
            self.y_min = min( self.y_min, ymin )
            self.y_max = max( self.y_max, ymax )

        for loc, v in self.data.items():
            # print( loc )
            v[ 'crop_tumor' ] = v[ 'uncrop_tumor' ][ self.y_min:self.y_max, self.x_min:self.x_max ]
            self.tumor3d.append( v[ 'uncrop_tumor' ] )
            self.tumor3dc.append( v[ 'crop_tumor' ] )

            # plt.figure( figsize=( 10, 10 ) )
            # plt.imshow( v[ 'crop_tumor' ].clip( 930, 930+255 ), cmap=plt.cm.gray )
            # plt.title( v[ 'crop' ] )
            # plt.colorbar()
            # plt.show()

        self.tumor3dc = np.array( self.tumor3dc )
        self.tumor3d = np.array( self.tumor3d )
        print( self.name, self.tumor3dc.shape )

        z, y, x = self.tumor3dc.shape
        if z > 32 or x > 118 or y > 118:
            raise ValueError( f"{self.name}'s shape {self.tumor3dc.shape} > ( 32, 118, 118 )" )
        zstart, ystart, xstart = 9 - z // 2, 58 - y // 2, 58 - x // 2
        self.tumor3dcc[ zstart:zstart + z, ystart:ystart + y, xstart:xstart + x ] = self.tumor3dc

    def show( self, voxel: np.ndarray ):
        plt.figure( figsize=( 12, 4 * ( voxel.shape[ 0 ] // 4 + 1 ) ) )
        for i, slice in enumerate( voxel ):
            plt.subplot( voxel.shape[ 0 ] // 4 + 1, 4, i + 1 )
            plt.imshow( slice, cmap=plt.cm.bone )
            plt.title( f"{self.name} = {voxel.shape}" )
            # plt.colorbar()
        plt.show()

    def save( self, voxel, name: Optional[ str ] = None, folderName: Optional[ str ] = None ):
        savePath = os.path.join( filefolder, 'rawdata', folderName or 'image_unname' )
        os.makedirs( savePath, exist_ok=True )
        out = sitk.GetImageFromArray( voxel )
        sitk.WriteImage( out, os.path.join( savePath, name + '.nii.gz' ) )

    def add( self, loc, maskImgAry: np.ndarray, ctImgAry: np.ndarray, cropPos: Tuple[ Tuple[ int ] ] ):
        if loc in self.data:
            raise IndexError( f"{loc} exist" )
        tumor_only = ctImgAry.copy()
        tumor_only[ maskImgAry == 0 ] = 0
        self.data[ loc ] = {
            'mask': maskImgAry,
            'ct': ctImgAry,
            'crop': cropPos,
            'uncrop_tumor': tumor_only,
        }
        # print( loc )
        # d = self.d
        # for i in range( tumor_only.shape[ 0 ] ):
        #     for j in range( tumor_only.shape[ 1 ] ):
        #         d[ tumor_only[ i, j ] ] += 1
        # for i, z in enumerate( d ):
        #     if z == 0:
        #         continue
        #     # print(i,':',z)
        # plt.figure( figsize=( 12, 6 ) )
        # plt.subplot(1,2,1)
        # plt.imshow( tumor_only.clip( 930, 930+255 ), cmap=plt.cm.bone )
        # plt.subplot(1,2,2)
        # plt.imshow( ctImgAry.clip( 930, 930+255 ), cmap=plt.cm.bone  )
        # plt.title( cropPos )
        # plt.show()
        # # 取出腫瘤
        # # 切出腫瘤


class ComputedTomography():

    def __init__( self, folderPath ) -> None:
        self.root = folderPath
        self.RTSS_path = glob.glob( os.path.join( folderPath, 'RS*' ) )[ 0 ]
        self.RTSS = pydicom.dcmread( self.RTSS_path )
        self.data = {}
        self.tumors: Dict[ str, Tumor ] = {}
        self.gen_tumors()

    def convert_mm_to_pixel( self, di_ipp, di_iop, di_ps, contourData ):
        matrix_im = np.array(
            [ [ di_iop[ 0 ] * di_ps[ 0 ], di_iop[ 3 ] * di_ps[ 1 ],
                np.finfo( np.float16 ).tiny, di_ipp[ 0 ] ],
              [ di_iop[ 1 ] * di_ps[ 0 ], di_iop[ 4 ] * di_ps[ 1 ],
                np.finfo( np.float16 ).tiny, di_ipp[ 1 ] ],
              [ di_iop[ 2 ] * di_ps[ 0 ], di_iop[ 5 ] * di_ps[ 1 ],
                np.finfo( np.float16 ).tiny, di_ipp[ 2 ] ], [ 0, 0, 0, 1 ] ] )
        inv_matrix_im = np.linalg.inv( matrix_im )
        contour_px = []

        for index, v in enumerate( [ contourData[ i:i + 3 ] for i in range( 0, len( contourData ), 3 ) ] ):
            v.append( 1 )
            i, j, trash, trash = [ int( np.around( i ) ) for i in inv_matrix_im.dot( np.array( v ) ) ]
            contour_px.append( ( i, j ) )

        return contour_px

    def coutours_img( self, contourData_px, shape, mode: int = 0 ):
        px = contourData_px
        img = Image.new( 'L', shape[ ::-1 ], 0 )
        if mode == 0:
            ImageDraw.Draw( img ).polygon( px, outline=1, fill=1 )
        elif mode == 1:
            ImageDraw.Draw( img ).polygon( px, outline=1, fill=2 )
            ImageDraw.Draw( img ).line( px, fill=1, width=6 )
        return img

    def get_edge( self, contourPixels: List[ Tuple[ int ] ] ):
        # contourPixels = np.array(contourPixels)
        # print(contourPixels)
        xs, ys = [ p[ 0 ] for p in contourPixels ], [ p[ 1 ] for p in contourPixels ]
        return ( min( xs ), min( ys ) ), ( max( xs ), max( ys ) )

    def gen_tumors( self ):
        for i in range( len( self.RTSS.StructureSetROISequence ) ):
            contour_name = self.RTSS.StructureSetROISequence[ i ].ROIName
            if contour_name != "BODY":
                #print(len(dicom_ds.ROIContourSequence),i)
                if contour_name in self.tumors:
                    raise IndexError( f"{contour_name} exist" )
                tumor = self.tumors[ contour_name ] = Tumor( contour_name )

                for seq in self.RTSS.ROIContourSequence[ i ].ContourSequence:
                    ct_id = seq.ContourImageSequence[ 0 ].ReferencedSOPInstanceUID
                    if ct_id not in self.data:
                        ct_path = f"CT.{seq.ContourImageSequence[ 0 ].ReferencedSOPInstanceUID}.dcm"
                        ct = pydicom.dcmread( os.path.join( self.root, ct_path ) )
                        self.data[ ct_id ] = {
                            'slice': ct.SliceLocation,
                            'loc': ct.InstanceNumber,
                            'spacing': ct.PixelSpacing,
                            'position': ct.ImagePositionPatient,
                            'orientation': ct.ImageOrientationPatient,
                            'ctpx': ct.pixel_array,
                        }
                    data = self.data[ ct_id ]

                    ary_mm = seq.ContourData
                    mask_poly = self.convert_mm_to_pixel( data[ 'position' ], data[ 'orientation' ], data[ 'spacing' ], ary_mm )
                    # print(ary_mm)
                    # print(mask_poly)
                    mask_img = self.coutours_img( mask_poly, data[ 'ctpx' ].shape )

                    tumor.add( data[ 'loc' ], np.array( mask_img ), data[ 'ctpx' ], self.get_edge( mask_poly ) )
            else:
                continue


# %%
# a = ComputedTomography( sorted( folder_paths )[ 203 ] )
# a.tumors[ 'LAP1' ].crop()
# a.tumors[ 'LAP1' ].show( a.tumors[ 'LAP1' ].tumor3d )
# a.tumors[ 'LAP1' ].show( a.tumors[ 'LAP1' ].tumor3dc )
# a.tumors[ 'LAP1' ].show( a.tumors[ 'LAP1' ].tumor3dcc )
# %%
for path in sorted( folder_paths ):
    print( path[ -12: ] )
    a = ComputedTomography( path )
    for tumor in a.tumors.values():
        if path[ -12: ] == '107-20180308' and tumor.name == 'LAP':
            tumor.name = 'LAP1'
        if path[ -12: ] in recorddata:
            if tumor.name == 'Tumor':
                print( 'tumor skip' )
                continue
            try:
                tumor.crop()
                pathology = ''.join( map( str, recorddata[ path[ -12: ] ][ tumor.name ][ 1: ] ) )
                tumor.save( tumor.tumor3dcc, f"{path[ -12: ]}_{tumor.name}_{pathology}", 'new_imagecc' )
                tumor.save( tumor.tumor3dc, f"{path[ -12: ]}_{tumor.name}_{pathology}", 'new_imagec' )
            except KeyError as e:
                print( f'{e} doesn\'t exist at {path[ -12: ]}' )
            except ValueError as e:
                print( f'{e} at {path[ -12: ]}' )

        else:
            print( f'{path[ -12: ]} doesn\'t exist' )
# %%
