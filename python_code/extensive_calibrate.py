import os
import sys
import numpy as np
import cv2
import glob
import argparse
import time
import math

# from image_utils import *

from generic_model import *
from fisheye_model import *
from omnidir_model import *

cv2.setUseOptimized(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration for Lens Distortion Correction")
    parser.add_argument(
        "-m=", action="store", nargs="?", default="generic", choices=["generic", "fisheye", "omnidir"],
        dest="calibrate_model", type=str, required=False, help="Specify the calibration model"
    )
    parser.add_argument(
        "-bw=", action="store", nargs="?", default=9, dest="board_width", type=int, required=False,
        help="Specify the number of horizontally crossing points in the chessboard"
    )
    parser.add_argument(
        "-bh=", action="store", nargs="?", default=6, dest="board_height", type=int, required=False,
        help="Specify the number of vertically crossing points in the chessboard"
    )
    parser.add_argument(
        "-sw=", action="store", nargs="?", default=50.0, dest="square_width", type=float, required=False,
        help="Specify the width of square in the chessboard"
    )
    parser.add_argument(
        "-sh=", action="store", nargs="?", default=50.0, dest="square_height", type=float, required=False,
        help="Specify the height of square in the chessboard"
    )
    parser.add_argument(
        "-o=", action="store", nargs="?", default="./params.xml", dest="output_filename", type=str,
        required=False, help="Specify the output filename to save param. of camera"
    )
    parser.add_argument(
        "-i=", action="store", nargs="?", default="./image_list.xml", dest="input_filename", type=str,
        required=False, help="Specify the input filename to load images"
    )
    parser.add_argument(
        "-os=", action="store", nargs="?", default=None, dest="undist_scale", type=float, required=False,
        help="Specify the scale of output image undistorted"
    )
    parser.add_argument(
        "-ow=", action="store", nargs="?", default=None, dest="undist_width", type=int, required=False,
        help="Specify the width of output image undistorted"
    )
    parser.add_argument(
        "-oh=", action="store", nargs="?", default=None, dest="undist_height", type=int, required=False,
        help="Specify the height of output image undistorted"
    )
    parser.add_argument(
        "-ox=", action="store", nargs="?", default=None, dest="undist_xoff", type=float, required=False,
        help="Specify the xoffset of output image undistorted"
    )
    parser.add_argument(
        "-oy=", action="store", nargs="?", default=None, dest="undist_yoff", type=float, required=False,
        help="Specify the yoffset of output image undistorted"
    )
    parser.add_argument(
        "-calibrate=", action="store", nargs="?", default=False, dest="need_calibrate", type=bool, required=False,
        help="Specify whether calibration do or not"
    )
    parser.add_argument(
        "-validate=", action="store", nargs="?", default=True, dest="need_validate", type=bool, required=False,
        help="Specify whether validation do or not"
    )
    args = parser.parse_args()

    #python3 extensive_calibrate.py -m= omnidir -bw= 9 -bh= 6 -sw= 50 -sh= 50 -o= omnidir_model.xml -i= omnidir_image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0 -oy= 0 -calibrate= true -validate= true
    #python3 extensive_calibrate.py -m= fisheye -bw= 9 -bh= 6 -sw= 50 -sh= 50 -o= fisheye_model.xml -i= omnidir_image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0 -oy= 0 -calibrate= true -validate= true
    #python3 extensive_calibrate.py -m= generic -bw= 9 -bh= 6 -sw= 50 -sh= 50 -o= generic_model.xml -i= image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0 -oy= 0 -calibrate= true -validate= true

    print("Calibration Model:", args.calibrate_model)
    calibrateImagesXML = args.input_filename
    calibrateModelXML = args.output_filename
    distortImagesXML = args.input_filename

    boardSize = (args.board_height, args.board_width)  # in the order of (h,w)
    squareSize = (args.square_height, args.square_width)  # in the order of (h,w)

    undistortSize = None if args.undist_height == 0 or args.undist_width == 0 else (args.undist_height, args.undist_width) # in the order of (h,w)
    undistortOffset = None if args.undist_yoff == 0 or args.undist_xoff == 0 else (args.undist_yoff, args.undist_xoff)  # in the order of (h,w)
    undistortScale = None if args.undist_scale == 0.0 else args.undist_scale

    print("calibrateImagesXML: ", calibrateImagesXML)
    print("calibrateModelXML: ", calibrateModelXML)
    print("distortImagesXML: ", distortImagesXML)

    print("boardSize: ", boardSize)
    print("squareSize: ", squareSize)

    print("undistortSize: ", undistortSize)
    print("undistortOffset: ", undistortOffset)
    print("undistortScale: ", undistortScale)

    if args.calibrate_model == "generic":
        calibrateFlags = cv2.CALIB_RATIONAL_MODEL  # + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_TILTED_MODEL
        # flags = cv2.CALIB_TILTED_MODEL
        # cv2.CALIB_FIX_ASPECT_RATIO
        # cv2.CALIB_FIX_FOCAL_LENGTH
        # cv2.CALIB_FIX_INTRINSIC
        # cv2.CALIB_FIX_K1 ~ 6
        # cv2.CALIB_FIX_PRINCIPAL_POINT
        # cv2.CALIB_FIX_S1_S2_S3_S4
        # cv2.CALIB_THIN_PRISM_MODEL
        # cv2.CALIB_TILTED_MODEL
        # cv2.CALIB_USE_INTRINSIC_GUESS
        # cv2.CALIB_USE_LU
        # cv2.CALIB_USE_QR
        # cv2.CALIB_ZERO_DISPARITY
        # cv2.CALIB_ZERO_TANGENT_DIST
        print("calibrateFlags: cv2.CALIB_RATIONAL_MODEL")

        model = GenericModel()
        if args.need_calibrate:
            model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

        if args.need_validate:
            image_list = model.readFileListFromXML(calibrateImagesXML)
            model.validateFromFiles(image_list, 0, undistortSize, undistortScale, undistortOffset, calibrateModelXML, True)

    elif args.calibrate_model == "fisheye":
        calibrateFlags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        # cv2.fisheye.CALIB_CHECK_COND
        # cv2.fisheye.CALIB_FIX_INTRINSIC
        # cv2.fisheye.CALIB_FIX_K1
        # cv2.fisheye.CALIB_FIX_K2
        # cv2.fisheye.CALIB_FIX_K3
        # cv2.fisheye.CALIB_FIX_K4
        # cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
        # cv2.fisheye.CALIB_FIX_SKEW
        # cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        # cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        print("calibrateFlags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW")

        model = FisheyeModel()
        if args.need_calibrate:
            model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

        if args.need_validate:
            image_list = model.readFileListFromXML(calibrateImagesXML)
            model.validateFromFiles(image_list, 0, undistortSize, undistortScale, undistortOffset, calibrateModelXML, True)

    elif args.calibrate_model == "omnidir":
        calibrateFlags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
        print("calibrateFlags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER")

        model = OmnidirModel()
        if args.need_calibrate:
            model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

        # undistortFlags = cv2.omnidir.RECTIFY_LONGLATI
        undistortFlags = cv2.omnidir.RECTIFY_CYLINDRICAL
        print("undistortFlags = cv2.omnidir.RECTIFY_CYLINDRICAL")

        if args.need_validate:
            image_list = model.readFileListFromXML(calibrateImagesXML)
            model.validateFromFiles(image_list, undistortFlags, undistortSize, undistortScale, undistortOffset,
                                    calibrateModelXML, True)

    cv2.destroyAllWindows()
    sys.exit(0)
    
