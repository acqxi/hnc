# %%
import os
from glob import glob
from typing import Dict

import getMask
import lapAnno

SAVE_FOLDER = './data/test/'


# %%
def read_data1( dicomsFolder: str, saveFolder: str = SAVE_FOLDER, debug: bool = False, fewTest: bool = False ):
    RTSSs = glob( dicomsFolder + '**/RS.*.dcm', recursive=True )
    ANNOS = lapAnno.read_contour_txt( '../../../0826-LAP-shareData/20220101-中榮病人/contour.txt' )
    for i, RTSSpath in enumerate( RTSSs ):
        case_name = RTSSpath.split( os.sep )[ -2 ].replace( 'HNNLAP-', '' )
        if case_name in ANNOS:
            try:
                tumors: Dict[ str, getMask.Tumor ] = getMask.get_tumors_from_dcm( RTSSpath, debug=debug, exten=0 )
                for lap_name, tumor in tumors.items():
                    _, lnm, ene = ANNOS[ case_name ][ lap_name ]
                    if debug:
                        print( case_name, lap_name, lnm, ene )
                    tumor.save(
                        '-'.join( map( str, [ case_name, lap_name, lnm, ene ] ) ),
                        slot=1,
                        folderName='noEx_crop',
                        saveFolder=saveFolder,
                    )
                if fewTest and i > 5:
                    break
            except KeyError as e:
                print( case_name, 'haven\'t contour data for', e )
        elif debug:
            print( case_name, 'haven\'t contour data' )


read_data1( 'D:\\OneDrive\\0826-LAP-shareData\\20220101-中榮病人\\' )


# %%
def read_data2( dicomsFolder: str, saveFolder: str = SAVE_FOLDER, debug: bool = False, fewTest: bool = False ):
    RTSSs = glob( dicomsFolder + '**/RS.*.dcm', recursive=True )
    ANNOS = lapAnno.read_contour_execl( '../../../0826-LAP-shareData/20210128-中榮病人-121例/LAP_detail_0524.xlsx' )
    for i, RTSSpath in enumerate( RTSSs ):
        case_name = RTSSpath.split( os.sep )[ -2 ].replace( 'HNNLAP-', '' )
        if case_name in ANNOS:
            try:
                tumors: Dict[ str, getMask.Tumor ] = getMask.get_tumors_from_dcm( RTSSpath, debug=debug, exten=0 )
                for lap_name, tumor in tumors.items():
                    _, lnm, ene = ANNOS[ case_name ][ lap_name ]
                    if debug:
                        print( case_name, lap_name, lnm, ene )
                    tumor.save(
                        '-'.join( map( str, [ case_name, lap_name, lnm, ene ] ) ),
                        slot=1,
                        folderName='noEx_crop',
                        saveFolder=saveFolder,
                    )
                if fewTest and i > 5:
                    break
            except KeyError as e:
                print( case_name, 'haven\'t contour data for', e )
        elif debug:
            print( case_name, 'haven\'t contour data' )


read_data2( 'D:\\OneDrive\\0826-LAP-shareData\\20210128-中榮病人-121例\\dicom\\' )


# %%
def read_data2( dicomsFolder: str, saveFolder: str = SAVE_FOLDER, debug: bool = False, fewTest: bool = False ):
    CTs = glob( dicomsFolder + '**/*.dcm', recursive=True )
    FOLDERs = set( [ os.path.dirname( os.path.split( CTpath )[ 0 ] ) for CTpath in CTs ] )
    ANNOS = lapAnno.read_contour_execl_2018(
        '../../../0826-LAP-shareData/20220301-HNC-2018/2018 hnc.xls 術前CT日期.xls ~0408.xlsx' )
    for i, folderPath in enumerate( FOLDERs ):
        case_name = folderPath.split( os.sep )[ -1 ].replace( '-c', '' ).replace( '-C', '' )
        if debug:
            print( case_name, folderPath, '\n', glob( os.path.join( folderPath, '*.nrrd' ) )[ 0 ] )
            if not fewTest:
                return
        if case_name.split( '-' )[ 0 ] in ANNOS:
            nrrd_path = glob( os.path.join( folderPath, '*.nrrd' ) )[ 0 ]
            try:
                if debug:
                    print( nrrd_path, '\n', folderPath )
                tumors: Dict[ str, getMask.Tumor ] = getMask.get_tumors_from_nrrd( nrrd_path, folderPath, exten=0 )
                for lap_name, tumor in tumors.items():
                    _, lnm, ene = ANNOS[ case_name.split( '-' )[ 0 ] ][ lap_name ]
                    if debug:
                        print( case_name, folderPath, case_name.split( '-' )[ 0 ], lap_name, lnm, ene )
                        pass
                    tumor.save(
                        '-'.join( map( str, [ case_name, lap_name, lnm, ene ] ) ),
                        slot=1,
                        folderName='noEx_crop',
                        saveFolder=saveFolder,
                    )
                if fewTest and i > 5:
                    break
            except KeyError as e:
                print( case_name, 'haven\'t contour data for', e )
                raise
        elif debug:
            print( case_name, 'haven\'t contour data' )


read_data2( 'D:\\OneDrive\\0826-LAP-shareData\\20220301-HNC-2018\\' )
# %%
