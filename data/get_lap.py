import glob
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
import pyclipper
import pydicom
import skimage.transform as sk_xfmr
from data.utils import ExcelDataLoader
from einops import rearrange
from PIL import Image, ImageDraw

from .utils import ExcelDataLoader


def get_slice(ctPath: str, show: bool = False):
    ct = pydicom.dcmread(ctPath)
    img_ary = ct.pixel_array
    r_img_ary: np.ndarray = img_ary * ct.RescaleSlope + ct.RescaleIntercept
    if show:
        plt.imshow(r_img_ary, cmap=plt.cm.bone)
        plt.show()
    return {
        "slice": ct.SliceLocation,
        "loc": ct.InstanceNumber,
        "spacing": ct.PixelSpacing,
        "position": ct.ImagePositionPatient,
        "orientation": ct.ImageOrientationPatient,
        "ctpx": r_img_ary,
    }


def equidistant_zoom_contour(contour: np.ndarray, margin):
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 2
    contour = contour[:, 0, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    solution = np.array(solution).reshape(-1, 1, 2).astype(int)
    return solution


def convert_mm_to_pixel(di_ipp, di_iop, di_ps, contourData):
    # fmt: off
    matrix_im = np.array([ 
        [ di_iop[0] * di_ps[0], di_iop[3] * di_ps[1], np.finfo(np.float16).tiny, di_ipp[0] ],
        [ di_iop[1] * di_ps[0], di_iop[4] * di_ps[1], np.finfo(np.float16).tiny, di_ipp[1] ],
        [ di_iop[2] * di_ps[0], di_iop[5] * di_ps[1], np.finfo(np.float16).tiny, di_ipp[2] ],
        [ 0                   , 0                   , 0                        , 1         ],
    ]) 
    # fmt: on
    inv_matrix_im = np.linalg.inv(matrix_im)
    contour_px = []

    for index, v in enumerate([contourData[i : i + 3] for i in range(0, len(contourData), 3)]):
        v.append(1)
        i, j, trash, trash = [int(np.around(i)) for i in inv_matrix_im.dot(np.array(v))]
        contour_px.append((i, j))

    return contour_px


def coutours_img(px, shape, mode: int = 0):
    if isinstance(px, np.ndarray):
        px = px.tolist()
    if not isinstance(px, tuple):
        px = [(x, y) for x, y in px]
    img = Image.new("L", shape[::-1], 0)
    if mode == 0:
        ImageDraw.Draw(img).polygon(px, outline=1, fill=1)
    elif mode == 1:
        ImageDraw.Draw(img).polygon(px, outline=1, fill=2)
        ImageDraw.Draw(img).line(px, fill=1, width=6)
    return img


def slice_lap_mask(
    ctDict: Dict[str, any],
    tumorPointAryMM: np.ndarray,
    exten: int = 5,
    debug: bool = False,
):
    mask_poly = np.array(
        convert_mm_to_pixel(ctDict["position"], ctDict["orientation"], ctDict["spacing"], tumorPointAryMM)
    )
    mask_img = coutours_img(mask_poly, ctDict["ctpx"].shape)

    if exten > 0:
        mask_poly_ori = mask_poly.copy()
        mask_img_ori = np.array(mask_img)

        mask_poly = rearrange(np.array(mask_poly), "b (c l) -> b c l", c=1)
        mask_poly = equidistant_zoom_contour(mask_poly, exten)
        mask_poly = rearrange(mask_poly, "b c l -> b (c l)")
        mask_poly = np.append(mask_poly, mask_poly[0, None], axis=0)
        mask_img = coutours_img(mask_poly, ctDict["ctpx"].shape)

    if debug:
        plt.figure(figsize=(10, 10))
        debug_img = np.concatenate(
            [
                np.expand_dims(mask_img, axis=2),
                np.zeros((512, 512, 1)) if exten == 0 else np.expand_dims(mask_img_ori, axis=2),
                np.zeros((512, 512, 1)),
            ],
            axis=2,
        )
        plt.imshow(debug_img)
        plt.show()

    lapDict = {
        "maskpx": np.array(mask_img),
        "maskpoly": mask_poly,
        "loc": ctDict["loc"],
        "px": ctDict["ctpx"].copy(),
    }

    return lapDict


def get_laps_from_dcm(RTSSpath: str, debug: bool = False, exten: int = 5, skipTumor: bool = True):
    folder = os.path.dirname(RTSSpath)
    RTSS = pydicom.dcmread(RTSSpath)
    ct_full_list = [
        cts.ReferencedSOPInstanceUID
        for cts in RTSS.ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0]
        .ContourImageSequence
    ][::-1]
    laps = {}
    cts = {}
    for i in range(len(RTSS.StructureSetROISequence)):
        contour_name = RTSS.StructureSetROISequence[i].ROIName
        if skipTumor and "Tumor" in contour_name:
            continue
        if contour_name != "BODY":
            # print(len(dicom_ds.ROIContourSequence),i)
            if contour_name in laps:
                raise IndexError(f"{contour_name} exist")
            lap = laps[contour_name] = []
            for idx, cSeq in enumerate(RTSS.ROIContourSequence[i].ContourSequence):
                ct_id = cSeq.ContourImageSequence[0].ReferencedSOPInstanceUID
                if ct_id not in cts:
                    cts[ct_id] = get_slice(os.path.join(folder, f"CT.{ct_id}.dcm"), show=False)
                ctDict = cts[ct_id]
                lap.append(slice_lap_mask(ctDict, cSeq.ContourData, exten=exten, debug=False))
                # print(ctDict["loc"], ct_id, len(poly))
            # fmt: off
            if exten != 0:
                lap.insert(0, lap[0] | {
                    "px": get_slice( os.path.join(folder,f"CT.{ct_full_list[lap[0]['loc']]}.dcm"), show=False)["ctpx"]
                })
                lap.append(lap[-1] | {
                    "px": get_slice(os.path.join(folder, f"CT.{ct_full_list[lap[-1]['loc']-1]}.dcm"), show=False)["ctpx"]
                })
            # fmt: on
            if debug:
                print(contour_name)
                w, h = [(i // 2, (i + 1) // 2) for i in range(12) if (i // 2) * ((i + 1) // 2) >= len(lap)][0]
                plt.figure(figsize=(h * 3, w * 3))
                for idx, lp in enumerate(lap):
                    plt.subplot(w, h, idx + 1)
                    plt.imshow(lp["px"].clip(-160, 120), cmap=plt.cm.bone)
                    plt.plot(lp["maskpoly"][:, 0], lp["maskpoly"][:, 1], color="r")
                    plt.title(f"polyline:{lp['maskpoly'].shape}")
                plt.show()
    return laps


def only_lap(
    lapDictList: List[Dict[str, any]],
    mode: str = ["size-preserved", "size-scaled"][0],
    hu_cut: Optional[Tuple[int]] = (-160, 120),
    debug: bool = False,
):
    hu_cut = hu_cut or (-1000, 1000)
    lap_ary3d = np.zeros((len(lapDictList), 512, 512))
    lapDictList = sorted(lapDictList, key=lambda x: x["loc"])

    all_poly = np.concatenate([l["maskpoly"] for l in lapDictList])
    max_x = np.max(all_poly[:, 0])
    max_y = np.max(all_poly[:, 1])
    min_x = np.min(all_poly[:, 0])
    min_y = np.min(all_poly[:, 1])
    x, y, z = max_x - min_x, max_y - min_y, len(lapDictList)
    # print(max_x, max_y, min_x, min_y)
    if x > 132 or y > 132 or z > 20:
        raise IndexError("this will be droped! too big", (x, y, z))

    for idx, lapDict in enumerate(lapDictList):
        if lapDict["px"].shape != (512, 512):
            lapDict["px"] = lapDict["px"][:512, :512]
            lapDict["maskpx"] = lapDict["maskpx"][:512, :512]
            if debug:
                print(f"shape {lapDict['px'].shape} to left -> (512, 512)")
                plt.imshow(lapDict["px"].clip(-160, 120), cmap=plt.cm.bone)
                plt.plot(lapDict["maskpoly"][:, 0], lapDict["maskpoly"][:, 1], color="r")
                plt.title("[:512,:512]")
                plt.show()
        lapDict["px"][lapDict["maskpx"] == 0] = -1000
        lap_ary3d[idx] = lapDict["px"].clip(*hu_cut)
        # print(lapDict["maskpoly"].shape)
        if debug:
            plt.imshow(lap_ary3d[idx], cmap=plt.cm.bone)
            plt.plot((max_x, min_x, min_x, max_x, max_x), (max_y, max_y, min_y, min_y, max_y), color="y")
            plt.show()

    final_ary = None
    if mode == "size-scaled":
        final_ary = sk_xfmr.resize(lap_ary3d[:, min_y:max_y, min_x:max_x], (32, 32, 32), mode="constant")
    elif mode == "size-preserved":

        final_ary = np.full((20, 132, 132), hu_cut[0], dtype=int)
        final_ary[
            10 - z // 2 : 10 - z // 2 + z, 66 - y // 2 : 66 - y // 2 + y, 66 - x // 2 : 66 - x // 2 + x
        ] = lap_ary3d[:, min_y:max_y, min_x:max_x]
    if debug:
        plt.imshow(final_ary[len(final_ary) // 2], cmap=plt.cm.bone)
        plt.show()

    return final_ary, lap_ary3d, x, y, z


def get_all_laps(
    rtssRoot:str='./data/rawdata/',
    saveRoot:str='./data/laps/test/',
    excelPath:str='~/vghtc/classification158/data/rawdata/LAP.xlsx',
    exten:int=5,
    mode:str=["size-preserved", "size-scaled"][0],
):
    os.makedirs(saveRoot, exist_ok=True)
    nons, lnms, enes = 0, 0, 0

    rtss_paths = glob.glob(os.path.join(rtssRoot, '**', 'RS.*.dcm'), recursive=True)
    rtss_paths = sorted(rtss_paths, key=lambda x: int(re.findall(r"HNNLAP-(\d{2,3}-\d{8})", x)[0].replace("-", "")))
    exl = ExcelDataLoader(path=excelPath, indexName="病人編號", header=0)
    for rtss_path in rtss_paths:
        for lap_name, lap in get_laps_from_dcm(rtss_path, exten=5).items():
            try:
                ary, _, x, y, z = only_lap(lap, mode=mode)
                patient, date = re.findall(r"HNNLAP-(\d{2,3})-(\d{8})", rtss_path)[0]
                lnm, ene = exl[int(patient), int(date), lap_name, "Path"], exl[int(patient), int(date), lap_name, "ENE"]
                if len(lnm) == 0 or len(ene) == 0:
                    print(f"HNNLAP-{patient}-{date}", lap_name, "not found!")
                    continue
                if not isinstance(lnm, str) or not isinstance(ene, str):
                    print(f"HNNLAP-{patient}-{date}", lap_name, "more than 1 reference value")
                    continue
                # print(f"HNNLAP-{patient}-{date}", lap_name, f"(x, y, z) = ({x:3}, {y:3}, {z:3})", lnm, ene)
                if lnm == "neg":
                    np.save(os.path.join(saveRoot, f"{patient}_{date}_{lap_name}_0.npy"), ary)
                    nons += 1
                elif ene == "neg":
                    np.save(os.path.join(saveRoot, f"{patient}_{date}_{lap_name}_1.npy"), ary)
                    lnms += 1
                elif ene == "pos":
                    np.save(os.path.join(saveRoot, f"{patient}_{date}_{lap_name}_2.npy"), ary)
                    enes += 1
                else:
                    raise ValueError(f"HNNLAP-{patient}-{date}", lap_name, "unknown status")
            except IndexError as e:
                print(f"HNNLAP-{patient}-{date}", lap_name, e)
            except ValueError as e:
                print(f"HNNLAP-{patient}-{date}", lap_name, e)
                print(len(lnm))
    print(f"({nons=}) + ({lnms=}) + ({enes=}) = {nons+lnms+enes}")