from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from data.utils import equidistantZoomContour, readCT, conv_mm2px, show_debug_img


def crop3D(lapDict:dict, save_path:Union[str, Path], fileName:str='testFile', debug:bool=False):
    """
    > It takes a dictionary of slices, finds the largest bounding box that contains all the slices,
    crops all the slices to that bounding box, and then returns a 3D array of the cropped slices
    
    Args:
      lapDict (dict): the dictionary of the processed lap
      save_path (Union[str, Path]): The path to save the cropped 3D image.
      fileName (str): the name of the file to be saved. Defaults to testFile
      debug (bool): if True, will show the first 16 slices of the cropped 3D image. Defaults to False
    """
    # 3D crop all masked slices fit the contour
    # 1. find the max bbox
    mbbox = [999, 999, 0, 0]  # max bbox : x, y, w, h
    for loc, data in lapDict.items():
        bbox = data["bbox"]
        mbbox[0] = min(mbbox[0], bbox[0])
        mbbox[1] = min(mbbox[1], bbox[1])
        mbbox[2] = max(mbbox[2], bbox[0] + bbox[2])
        mbbox[3] = max(mbbox[3], bbox[1] + bbox[3])
    mbbox[2] -= mbbox[0]
    mbbox[3] -= mbbox[1]
    # print(f"max bbox: {mbbox}")

    # 2. make all slices mask apply
    for loc, data in lapDict.items():
        lapDict[loc]["masked_ctpx"] = data["ctpx"] * data["mask_ext"]

    # 3. crop all slices
    for loc, data in lapDict.items():
        data["croped_ctpx"] = data["masked_ctpx"][mbbox[1]:mbbox[1] + mbbox[3], mbbox[0]:mbbox[0] + mbbox[2]]

    # 4. make 3D image
    crop_3D = np.zeros((len(lapDict), mbbox[3], mbbox[2]), np.int16)
    min_loc = min(lapDict.keys())

    for loc, data in lapDict.items():
        crop_3D[loc - min_loc, :, :] = data["croped_ctpx"]
        if debug and loc - min_loc < 16:
            if loc - min_loc == 0:
                plt.figure(figsize=(10, 10))
            print(
                np.min(data["croped_ctpx"] * data["slope"] + data["intercept"]),
                np.max(data["croped_ctpx"] * data["slope"] + data["intercept"]),
            )
            plt.subplot(4, 4, loc - min_loc + 1)
            plt.imshow((data["croped_ctpx"] * data["slope"] + data["intercept"]).clip(-100, 155))
            plt.title(f"loc: {loc}")

    # lap["crop3D"] = crop_3D
        
    # 5. save the processed 3D image
    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / f"{fileName}.npy", crop_3D)

    return crop_3D


def getCropMaskLAP(rtss_path: Path, extend_size: int = 0, save_path: Path = None, pass_tumor: bool = True, debug: bool = False):
    """
    It reads the contour data from the RTSTRUCT file,
    and then uses the contour data to create a mask for each slice in the CT scan.

    The mask is then used to crop the CT scan to the area of interest.
    The cropped CT scan is then saved as a 3D numpy array.
    The function returns a dictionary of the 3D numpy arrays.
    The dictionary keys are the names of the contours.
    The dictionary values are the 3D numpy arrays.
    The 3D numpy arrays are of shape (number of slices, height, width).
    The height and width are the height and width of the bounding box of the contour.
    The bounding box is the smallest rectangle that contains the contour.
    The bounding box is calculated for each slice, and then the largest bounding box is used for all slices.

    Args:
      rtss_path (Path): Path, the path to the rtss file
      extend_size (int): the number of pixels to extend the contour by. Defaults to 0
      save_path (Path): The path to save the numpy array of the 3D image.
      pass_tumor (bool): If True, then the tumor contour will be ignored. Defaults to True
      debug (bool): If True, it will show the contour and the mask. Defaults to False
    """
    rtss = pydicom.dcmread(rtss_path)
    cts, laps, loc2id = {}, {}, {}  # type: ignore
    cis = rtss.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
    cis = cis.RTReferencedSeriesSequence[0].ContourImageSequence
    for ci in cis:
        ct_id = ci.ReferencedSOPInstanceUID
        if ct_id not in cts:
            ct_stem = f"CT.{ct_id}.dcm"
            cts[ct_id] = readCT(rtss_path.parent / ct_stem)
        loc2id[cts[ct_id]["loc"]] = ct_id

    for i in range(len(rtss.StructureSetROISequence)):
        contour_name = rtss.StructureSetROISequence[i].ROIName
        if contour_name in laps:
            print(f"Detect duplicate contour name: {contour_name}")
            continue
        if pass_tumor and "Tumor" in contour_name:
            continue
        if contour_name != "BODY":
            lap = laps[contour_name] = {"cts_masks": {}}
            for seq in rtss.ROIContourSequence[i].ContourSequence:
                ct_id = seq.ContourImageSequence[0].ReferencedSOPInstanceUID
                if ct_id not in cts:
                    ct_stem = f"CT.{ct_id}.dcm"
                    cts[ct_id] = readCT(rtss_path.parent / ct_stem)
                data = cts[ct_id]

                contour_points_mm = seq.ContourData
                contour_points_px = conv_mm2px(
                    data["position"],
                    data["orientation"],
                    data["spacing"],
                    contour_points_mm,
                )

                contour_points_px = np.array(contour_points_px)
                extend_contour_point = equidistantZoomContour(contour_points_px, extend_size)
                extend_contour_point = np.append(extend_contour_point, extend_contour_point[0:1], axis=0)

                mask_img = cv2.fillConvexPoly(np.zeros(data["ctpx"].shape, np.uint8), contour_points_px, 1)
                mask_img_extend = cv2.fillConvexPoly(np.zeros(data["ctpx"].shape, np.uint8), extend_contour_point, 1)

                lap["cts_masks"][data["loc"]] = {
                    "mask_ext": mask_img_extend,
                    "bbox": cv2.boundingRect(extend_contour_point),  # x, y, w, h
                    **data,
                }

                if debug:
                    show_debug_img(lap, data, contour_points_px, extend_contour_point, mask_img, mask_img_extend, rtss_path.parent.name, contour_name)

            if extend_size > 0:
                upper, lower = (
                    max(lap["cts_masks"].keys()) + 1,
                    min(lap["cts_masks"].keys()) - 1,
                )
                if upper <= len(cis):
                    lap["cts_masks"][upper] = {
                        "mask_ext": lap["cts_masks"][upper - 1]["mask_ext"],
                        "bbox": lap["cts_masks"][upper - 1]["bbox"],
                        **cts[loc2id[upper]],
                    }
                if lower >= 1:
                    lap["cts_masks"][lower] = {
                        "mask_ext": lap["cts_masks"][lower + 1]["mask_ext"],
                        "bbox": lap["cts_masks"][lower + 1]["bbox"],
                        **cts[loc2id[lower]],
                    }

            lap["instances_number"] = len(lap["cts_masks"])

            # print(f"Load {rtss_path.parent.name}'s {contour_name} done.")
            # print(sorted(laps[contour_name]['cts_masks'].keys()))

            lap["crop3D"] = crop3D(lap["cts_masks"],save_path=save_path,fileName=f'{rtss_path.parent.name}_{contour_name}',debug=debug)

    return laps

