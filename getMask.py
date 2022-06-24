# %%
import glob
from math import sqrt
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyclipper
import pydicom
import SimpleITK as sitk
from einops import rearrange
from PIL import Image, ImageDraw

FILE_ROOT = os.path.abspath( os.path.dirname( __file__ ) )


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
        self.tumor3d = []  # original
        self.tumor3dc = []  # crop
        self.tumor3dcc = np.zeros( ( 32, 118, 118 ) )  # center crop

    def crop( self ):
        for loc, v in self.data.items():
            ( xmin, ymin ), ( xmax, ymax ) = v[ 'crop' ]
            self.x_min = min( self.x_min, xmin )
            self.x_max = max( self.x_max, xmax )
            self.y_min = min( self.y_min, ymin )
            self.y_max = max( self.y_max, ymax )

        self.tumor3d = []  # original
        self.tumor3dc = []  # crop

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
        # print( self.name, self.tumor3dc.shape )

        try:
            z, y, x = self.tumor3dc.shape
        except ValueError as e:
            raise ValueError( f"{self.name}'s shape {self.tumor3dc.shape} => unknown error({e=}) " )
        if z > 32 or x > 118 or y > 118:
            raise ValueError( f"{self.name}'s shape {self.tumor3dc.shape} > ( 32, 118, 118 )" )
        zstart, ystart, xstart = 9 - z // 2, 58 - y // 2, 58 - x // 2
        self.tumor3dcc[ zstart:zstart + z, ystart:ystart + y, xstart:xstart + x ] = self.tumor3dc

    def show( self, slot: int = 1, mode=0, hu: int = -110 ):
        voxel = [ self.tumor3d, self.tumor3dc, self.tumor3dcc ][ slot ]
        if not isinstance( voxel, np.ndarray ):
            voxel = np.array( voxel )
        # plt.figure( figsize=( 12, 4 * ( voxel.shape[ 0 ] // 4 + 1 ) ) )
        fig = plt.figure()
        plt.suptitle( f"{self.name} = {voxel.shape}" )
        # print( voxel.shape )
        for i, slice in enumerate( voxel ):
            plt.subplot( voxel.shape[ 0 ] // 4 + 1, 4, i + 1 )
            px = plt.imshow( slice if mode == 0 else ( slice.clip( hu, hu + 255 ) - hu ), cmap=plt.cm.bone )
            fig.colorbar( px )
        plt.show()

    def save(
            self,
            name: str,
            slot: int = 1,
            folderName: Optional[ str ] = None,
            saveFolder: Optional[ str ] = None,
            hu: Optional[ int ] = None ):
        voxel = [ self.tumor3d, self.tumor3dc, self.tumor3dcc ][ slot ]
        savePath = os.path.join( saveFolder or os.path.join( FILE_ROOT, 'rawdata' ), folderName or 'image_unname' )
        os.makedirs( savePath, exist_ok=True )
        out = sitk.GetImageFromArray( voxel if hu is None else ( voxel.clip( hu, hu + 255 ) - hu ).astype( np.uint8 ) )
        sitk.WriteImage( out, os.path.join( savePath, name + '.nii.gz' ) )

    def add( self, loc, maskImgAry: np.ndarray, ctImgAry: np.ndarray, cropPos: Tuple[ Tuple[ int ] ], maskValue: int = 1 ):
        if loc in self.data:
            raise IndexError( f"{loc} exist" )
        tumor_only = ctImgAry.copy()
        tumor_only[ maskImgAry != maskValue ] = 0
        self.data[ loc ] = {
            'mask': maskImgAry,
            'ct': ctImgAry,
            'crop': cropPos,
            'uncrop_tumor': tumor_only,
        }


# %%
def equidistant_zoom_contour( contour: np.ndarray, margin ):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离，margin正数是外扩，负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    # 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
    pco.MiterLimit = 2
    contour = contour[ :, 0, : ]
    pco.AddPath( contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON )
    solution = pco.Execute( margin )
    solution = np.array( solution ).reshape( -1, 1, 2 ).astype( int )
    return solution


def convert_mm_to_pixel( di_ipp, di_iop, di_ps, contourData ):
    matrix_im = np.array( [ [ di_iop[ 0 ] * di_ps[ 0 ], di_iop[ 3 ] * di_ps[ 1 ],
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


def coutours_img( px, shape, mode: int = 0 ):
    if isinstance( px, np.ndarray ):
        px = px.tolist()
    if not isinstance( px, tuple ):
        px = [ ( x, y ) for x, y in px ]
    img = Image.new( 'L', shape[ ::-1 ], 0 )
    if mode == 0:
        ImageDraw.Draw( img ).polygon( px, outline=1, fill=1 )
    elif mode == 1:
        ImageDraw.Draw( img ).polygon( px, outline=1, fill=2 )
        ImageDraw.Draw( img ).line( px, fill=1, width=6 )
    return img


def get_edge( contourPixels: List[ Tuple[ int ] ] ):
    """
    It takes a list of pixel coordinates and returns the minimum and maximum x and y coordinates

    :param contourPixels: a list of tuples of the form (x, y)
    :type contourPixels: List[ Tuple[ int ] ]
    :return: The top left and bottom right coordinates of the bounding box.
    """

    # contourPixels = np.array(contourPixels)
    # print(contourPixels)
    xs, ys = [ p[ 0 ] for p in contourPixels ], [ p[ 1 ] for p in contourPixels ]
    return ( min( xs ), min( ys ) ), ( max( xs ), max( ys ) )


# %%
def get_tumors_from_dcm( RTSSpath: str, debug: bool = False, exten: int = 5, passTumor: bool = True ):
    folder = os.path.dirname( RTSSpath )
    RTSS = pydicom.dcmread( RTSSpath )
    cts = {}
    tumors = {}
    del_list = []
    for i in range( len( RTSS.StructureSetROISequence ) ):
        contour_name = RTSS.StructureSetROISequence[ i ].ROIName
        if passTumor and 'Tumor' in contour_name:
            continue
        if contour_name != "BODY":
            #print(len(dicom_ds.ROIContourSequence),i)
            if contour_name in tumors:
                raise IndexError( f"{contour_name} exist" )
            tumor = tumors[ contour_name ] = Tumor( contour_name )
            for seq in RTSS.ROIContourSequence[ i ].ContourSequence:
                ct_id = seq.ContourImageSequence[ 0 ].ReferencedSOPInstanceUID
                if ct_id not in cts:
                    ct_path = f"CT.{seq.ContourImageSequence[ 0 ].ReferencedSOPInstanceUID}.dcm"
                    ct = pydicom.dcmread( os.path.join( folder, ct_path ) )
                    cts[ ct_id ] = {
                        'slice': ct.SliceLocation,
                        'loc': ct.InstanceNumber,
                        'spacing': ct.PixelSpacing,
                        'position': ct.ImagePositionPatient,
                        'orientation': ct.ImageOrientationPatient,
                        'ctpx': ct.pixel_array,
                    }
                data = cts[ ct_id ]

                ary_mm = seq.ContourData
                mask_poly = convert_mm_to_pixel( data[ 'position' ], data[ 'orientation' ], data[ 'spacing' ], ary_mm )
                # print(ary_mm)

                mask_img = coutours_img( mask_poly, data[ 'ctpx' ].shape )

                mask_poly = np.array( mask_poly )
                mask_poly_edit = rearrange( np.array( mask_poly ), 'b (c l) -> b c l', c=1 )
                mask_poly_edit = equidistant_zoom_contour( mask_poly_edit, exten )
                mask_poly_edit = rearrange( mask_poly_edit, 'b c l -> b (c l)' )
                mask_poly_edit = np.append( mask_poly_edit, mask_poly_edit[ 0, None ], axis=0 )
                if debug:
                    print( mask_poly.shape, mask_poly_edit.shape )

                mask_img = coutours_img( mask_poly, data[ 'ctpx' ].shape )
                if debug:
                    plt.subplot( 1, 2, 1 )
                    plt.imshow( mask_img )
                    plt.plot( mask_poly[ :, 0 ], mask_poly[ :, 1 ] )
                    plt.subplot( 1, 2, 2 )
                mask_img_edit = coutours_img( mask_poly_edit, data[ 'ctpx' ].shape )
                if debug:
                    plt.imshow( mask_img_edit )
                    plt.plot( mask_poly_edit[ :, 0 ], mask_poly_edit[ :, 1 ] )
                    plt.show()

                mask_img = mask_img_edit
                mask_poly = mask_poly_edit
                try:
                    tumor.add( data[ 'loc' ], np.array( mask_img ), data[ 'ctpx' ], get_edge( mask_poly ) )
                except IndexError as e:
                    print( f"loc existed ({e}) in {contour_name} @ {RTSSpath.split( os.sep )[ -2 ]}" )
            try:
                tumor.crop()
            except ValueError as e:
                print( e, 'in', contour_name, '@', RTSSpath.split( os.sep )[ -2 ], 'del it' )
                del_list.append( contour_name )
    for contour_name in del_list:
        del tumors[ contour_name ]
    return tumors


# %%
if __name__ == '__main__':
    rs: Dict[ str, Tumor ] = get_tumors_from_dcm(
        RTSSpath=(
            'D:\\OneDrive\\0826-LAP-shareData\\20210128-中榮病人-121例\\dicom\\HNNLAP-031-20200902'
            '\\RS.1.2.246.352.205.4943630303210885154.604627541959089581.dcm' ) )
    for k, v in rs.items():
        v.show( slot=1, mode=1 )
        v.save(
            'test',
            slot=1,
            folderName='crop',
            saveFolder='./data/test',
            hu=-110,
        )
# %%


# %%
def diagonal_to_box( p1: Tuple[ int ], p2: Tuple[ int ] ) -> Tuple[ Tuple[ int ] ]:
    # print( p1, p2 )
    return ( p1[ 0 ], p1[ 0 ], p2[ 0 ], p2[ 0 ], p1[ 0 ] ), ( p1[ 1 ], p2[ 1 ], p2[ 1 ], p1[ 1 ], p1[ 1 ] )


def get_tumors_from_nrrd( nrrd: str, dcmsFolder: str, debug: bool = False, show: bool = False, exten: int = 5 ):
    img = sitk.ReadImage( nrrd )
    img_ary = sitk.GetArrayFromImage( img )
    # print( glob.glob( os.path.join( dcmsFolder, '**', '*.dcm' ), recursive=True ) )
    edge_z1, edge_z2 = list( map( int, img.GetMetaData( 'Segment0_Extent' ).split( ' ' ) ) )[ -2: ]  #[(x1,x2,y1,y2,z1,z2)]
    tumors = { f'LAP{k + 1}': Tumor( f'LAP{k + 1}' ) for k in range( np.max( img_ary ) ) }
    cts = {}
    print( dcmsFolder.split( os.sep )[ -1 ], ' with Lap: ', np.max( img_ary ) )
    for i in glob.glob( os.path.join( dcmsFolder, '**', '*.dcm' ), recursive=True ):
        if os.path.split( i )[ -1 ].startswith( 'RS' ):
            continue
        ct = sitk.ReadImage( i )
        # print(pydicom.dcmread(i))
        # print( i, ct.GetMetaData('0020|0013') ) # (0020, 0013) Instance Number
        cts[ int( ct.GetMetaData( '0020|0013' ) ) ] = {
            'slice': ct.GetMetaData( '0020|1041' ),  # (0020, 1041) Slice Location
            'loc': int( ct.GetMetaData( '0020|0013' ) ),  # (0020, 0013) Instance Number
            'spacing': ct.GetMetaData( '0028|0030' ),  # (0028, 0030) Pixel Spacing
            'position': ct.GetMetaData( '0020|0032' ),  # (0020, 0032) Image Position (Patient)
            'orientation': ct.GetMetaData( '0020|0037' ),  # (0020, 0037) Image Orientation (Patient)
            'ctpx': ( sitk.GetArrayFromImage( ct ) + 1024 ).clip( 0, 5000 ),  # (7fe0, 0010) Pixel Data
        }
    if debug:
        fig = plt.figure( figsize=( 15, 15 ) )
        plt.suptitle( nrrd )
        plt.axis( 'off' )
        for idx, slice in enumerate( img_ary.copy()[ min( edge_z2, edge_z1 ):max( edge_z2, edge_z1 ) + 1 ] ):
            plt.subplot(
                int( sqrt( abs( edge_z2 - edge_z1 ) * 2 ) ) + 1,
                int( sqrt( abs( edge_z2 - edge_z1 ) * 2 ) ) + 1, idx * 2 + 1 )
            px = plt.imshow( cts[ idx + min( edge_z2, edge_z1 ) ][ 'ctpx' ][ 0 ] )
            for i in ( set( slice.flatten() ) - { 0 } ):
                mask_lap = ( slice == i ).astype( np.uint8 )
                points, _ = cv2.findContours( mask_lap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
                plt.plot( *diagonal_to_box( *get_edge( points[ 0 ][ :, 0, : ] ) ), 'g' )
                plt.plot( points[ 0 ][ :, :, 0 ], points[ 0 ][ :, :, 1 ], 'r' )
            # fig.colorbar( px )

            plt.subplot(
                int( sqrt( abs( edge_z2 - edge_z1 ) * 2 ) ) + 1,
                int( sqrt( abs( edge_z2 - edge_z1 ) * 2 ) ) + 1, idx * 2 + 2 )
            plt.imshow( slice )

            for i in set( slice.flatten() ) - { 0 }:
                mask_lap = ( slice == i ).astype( np.uint8 )
                points, _ = cv2.findContours( mask_lap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
                mask_poly = points[ 0 ][ :, 0, : ]
                mask_poly_edit = points[ 0 ].copy()
                mask_poly_edit = equidistant_zoom_contour( mask_poly_edit, exten )
                mask_poly_edit = rearrange( mask_poly_edit, 'b c l -> b (c l)' )
                mask_poly_edit = np.append( mask_poly_edit, mask_poly_edit[ 0, None ], axis=0 )

                plt.plot( *diagonal_to_box( *get_edge( points[ 0 ][ :, 0, : ] ) ), 'g' )
                plt.plot( mask_poly[ :, 0 ], mask_poly[ :, 1 ], 'r' )
                plt.plot( mask_poly_edit[ :, 0 ], mask_poly_edit[ :, 1 ], 'b' )

                plt.title( set( slice.flatten() ) - { 0 } )
        return
    tumor_only_mask = img_ary.copy()[ min( edge_z2, edge_z1 ):max( edge_z2, edge_z1 ) + 1 ]
    for i in range( abs( edge_z2 - edge_z1 ) ):
        # print( f"{i=},{min( edge_z2, edge_z1 )=},{cts=}" )
        tumor_only_img = cts[ i + min( edge_z2, edge_z1 ) ][ 'ctpx' ][ 0 ].copy()
        # print(points[0][:,0,:])
        # print( set( tumor_only_mask[ i ].flatten() ) )
        for idx, lap in enumerate( set( tumor_only_mask[ i ].flatten() ) - { 0 } ):
            mask_lap = ( tumor_only_mask[ i ] == lap ).astype( np.uint8 )
            points, _ = cv2.findContours( mask_lap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
            if exten == 0:
                tumors[ f'LAP{lap}' ].add(
                    loc=i + min( edge_z2, edge_z1 ),
                    maskImgAry=mask_lap,
                    ctImgAry=tumor_only_img,
                    cropPos=get_edge( points[ 0 ][ :, 0, : ] ) )
            elif exten > 0:
                mask_poly_edit = points[ 0 ].copy()
                mask_poly_edit = equidistant_zoom_contour( mask_poly_edit, exten )
                mask_poly_edit = rearrange( mask_poly_edit, 'b c l -> b (c l)' )
                mask_poly_edit = np.append( mask_poly_edit, mask_poly_edit[ 0, None ], axis=0 )
                mask_img = coutours_img( mask_poly_edit, tumor_only_img.shape )
                # plt.subplot( 1, 2, 1 )
                # plt.plot( mask_poly_edit[ :, 0 ], mask_poly_edit[ :, 1 ], 'r' )
                # plt.imshow( mask_img )
                # plt.subplot( 1, 2, 2 )
                # plt.plot( mask_poly_edit[ :, 0 ], mask_poly_edit[ :, 1 ], 'r' )
                # plt.plot( *diagonal_to_box( *get_edge( mask_poly_edit ) ), 'g' )
                # plt.imshow( tumor_only_img )
                # plt.show()
                tumors[ f'LAP{lap}' ].add(
                    loc=i + min( edge_z2, edge_z1 ),
                    maskImgAry=np.array( mask_img ),
                    ctImgAry=tumor_only_img,
                    cropPos=get_edge( mask_poly_edit ) )
    del_list = []
    for k, tumor in tumors.items():
        try:
            tumor.crop()
        except ValueError as e:
            print( e, 'in', k, '@', dcmsFolder.split( os.sep )[ -1 ], 'del it' )
            del_list.append( k )
        if show:
            tumor.show( 1, mode=1 )
    for k in del_list:
        del tumors[ k ]
    return tumors


# %%
if __name__ == '__main__':
    nrrd_path = ( 'D:\\OneDrive\\0826-LAP-shareData\\20220301-HNC-2018\\001783328C-c\\001783328C_CTimage_20181205.nrrd' )
    dcms_folder = 'D:\\OneDrive\\0826-LAP-shareData\\20220301-HNC-2018\\001783328C-c'
    print( get_tumors_from_nrrd( nrrd_path, dcms_folder, exten=5, show=True ) )


# %%
def show_25_clip_dcm( dcmPath: str ):
    start_k, bonus = -250, 20
    ct = sitk.ReadImage( dcmPath )
    img_ary = sitk.GetArrayFromImage( ct )
    fig = plt.figure( figsize=( 15, 15 ) )
    for i in range( 25 ):
        start = start_k + i * bonus
        plt.subplot( 5, 5, i + 1 )
        plt.imshow( img_ary[ 0 ].clip( start, ( start ) + 255 ) - ( start ), cmap=plt.cm.bone )
        plt.title( f"{start} -> { ( start ) + 255}" )
    plt.show()


if __name__ == '__main__':
    show_25_clip_dcm(
        glob.glob( 'D:/OneDrive/0826-LAP-shareData\\20210128-中榮病人-121例\\dicom\\HNNLAP-031-20200902\\**/*.dcm',
                   recursive=True )[ 3 ] )
# %%
