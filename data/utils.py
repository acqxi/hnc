
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import pydicom


def conv_mm2px(di_ipp, di_iop, di_ps, contour_mm: list):
    """
    It converts a list of 3D points in millimeters to a list of 2D points in pixels.

    Args:
      di_ipp: Image Position Patient
      di_iop: Image Orientation Patient
      di_ps: pixel spacing
      contour_mm (list): list of x,y,z coordinates of the contour in mm

    Returns:
      The contour_px is being returned.
    """
    # yapf: disable
    matrix_im = [ [ di_iop[ 0 ] * di_ps[ 0 ], di_iop[ 3 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 0 ] ],
                  [ di_iop[ 1 ] * di_ps[ 0 ], di_iop[ 4 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 1 ] ],
                  [ di_iop[ 2 ] * di_ps[ 0 ], di_iop[ 5 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 2 ] ],
                  [ 0                       , 0                       , 0                          , 1           ] ]
    # yapf: enable
    inv_matrix_im = np.linalg.inv(np.array(matrix_im))
    mm_len = len(contour_mm)
    contour_mm_ary = np.concatenate([np.array(contour_mm).reshape(mm_len // 3, 3), np.ones((mm_len // 3, 1))], 1)
    contour_px = np.rint(np.dot(inv_matrix_im, contour_mm_ary.T).T)[:, 0:2].astype(int)

    return contour_px


def equidistantZoomContour(contour: np.ndarray, margin):
    """
    It takes a contour and a margin, and returns a contour that is the same shape as the original contour,
    but with the margin added to it.

    Args:
      contour (np.ndarray): the contour to be zoomed
      margin: the distance between the original contour and the new contour

    Returns:
      the contour of the image after the zoom.
    """
    pco = pyclipper.PyclipperOffset()
    # 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
    # pco.MiterLimit = 0
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    solution = np.array(solution).astype(int)[0]
    return solution


def readCT(ct_path):
    """
    It reads a CT scan from a file and returns a dictionary containing the slice location,
    the instance number, the pixel spacing, the image position, the image orientation,
    the rescale slope, the rescale intercept, and the pixel array.

    Args:
      ct_path: the path to the CT file

    Returns:
      A dictionary with the following keys:
        slice: the slice location
        loc: the instance number
        spacing: the pixel spacing
        position: the image position patient
        orientation: the image orientation patient
        slope: the rescale slope
        intercept: the rescale intercept
        ctpx: the pixel array
    """
    ct = pydicom.dcmread(ct_path)
    return {
        "slice": ct.SliceLocation,
        "loc": int(ct.InstanceNumber),
        "spacing": ct.PixelSpacing,
        "position": ct.ImagePositionPatient,
        "orientation": ct.ImageOrientationPatient,
        "slope": ct.RescaleSlope,
        "intercept": ct.RescaleIntercept,
        "ctpx": ct.pixel_array,
    }

def show_debug_img(lap, data, contourPointsPx, extendContourPoint, maskImg, maskImgExtend, patientName='TestPatient', contour_name='TestContour'):
    """
    > This function takes in a contour point, and extends it by a certain amount of pixels in the
    direction of the contour
    
    Args:
      lap: the laparoscopy data
      data: a dictionary containing the following keys:
      contourPointsPx: the contour points in pixel space
      extendContourPoint: the contour points that are extended to the edge of the bounding box
      maskImg: the mask image of the contour
      maskImgExtend: the mask image with the contour extended by the specified amount
      patientName: the name of the patient. Defaults to TestPatient
      contour_name: the name of the contour. Defaults to TestContour
    """
    plt.figure(figsize=(6, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(data["ctpx"], cmap=plt.cm.bone)
    # plot bbox
    b = lap["cts_masks"][data["loc"]]["bbox"]
    plt.plot(
        [b[0], b[0] + b[2], b[0] + b[2], b[0], b[0]],
        [b[1], b[1], b[1] + b[3], b[1] + b[3], b[1]],
        "b",
    )
    plt.plot(contourPointsPx[:, 0], contourPointsPx[:, 1], "r")
    plt.plot(extendContourPoint[:, 0], extendContourPoint[:, 1], "y")

    plt.subplot(1, 3, 2)
    plt.imshow(maskImg, cmap=plt.cm.bone)
    plt.subplot(1, 3, 3)
    plt.imshow(maskImgExtend, cmap=plt.cm.bone)

    plt.suptitle(f"{patientName}'s Contour {contour_name} at {data['loc']}" +
                    f" bbox: {lap['cts_masks'][data['loc']]['bbox']}")
    plt.tight_layout()
    plt.show()