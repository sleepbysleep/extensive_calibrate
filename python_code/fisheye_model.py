import os
import sys
import cv2
import numpy as np
import time
import math

from base_model import *

# cv2.setUseOpenVX(True)
cv2.setUseOptimized(True)

DEBUG = True

class FisheyeModel(BaseModel):
    def __init__(self):
        # self.super().__init__()

        self.calibrateFlags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        self.boardSize = (6,9)
        self.squareSize = (50,50)
        # self.undistortFlags =
        self.undistortSize = None
        self.undistortOffset = None
        self.undistortScale = 1.0
        self.modelType = "fisheye"

    def computeReprojectionErrors(self, object_points_list:list, image_points_list:list, rvecs:list, tvecs:list,
                                  camera_matrix:np.ndarray, distort_coeffs:np.ndarray):
        total_points = 0
        _rms_error = 0
        image_reproj_errors = []
        for objpoints,imgpoints,rvec,tvec in zip(object_points_list,image_points_list,rvecs,tvecs):
            imgpoints2, _ = cv2.fisheye.projectPoints(objpoints, rvec, tvec, camera_matrix, distort_coeffs)
            error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2)
            _rms_error += error * error
            # print(imgpoints.shape, imgpoints2.shape)
            total_points += imgpoints.shape[0]
            image_reproj_errors.append(math.sqrt(error * error / imgpoints.shape[0]))
        image_reproj_errors = np.array(image_reproj_errors)
        _rms_error = math.sqrt(_rms_error / total_points)
        return _rms_error,image_reproj_errors

    def calibrate(self, object_points_list:list, image_points_list:list, image_size:tuple,
                  board_size:tuple, square_size:tuple, flags:int, image_list:list, model_xml_filename:str):
        avg_reproj_error, camera_matrix, distort_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            image_size=image_size[::-1],
            K=None, D=None, rvecs=None, tvecs=None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 0.0001)  # 1e-8)
        )
        print("=== Root Mean Square Error ===\n", avg_reproj_error)
        print("=== Camera Matrix ===\n", camera_matrix)
        print("=== Distance Coefficients ===\n", distort_coeffs)

        _avg_reproj_error,image_reproj_errors = self.computeReprojectionErrors(
            object_points_list,
            image_points_list,
            rvecs,
            tvecs,
            camera_matrix,
            distort_coeffs
        )
        if DEBUG: print("=== Recalculated RMS error ===\n", _avg_reproj_error)

        self.saveParamsToXML(
            model_type="fisheye",
            camera_matrix=camera_matrix,
            xi=np.zeros((1, 1)),
            distort_coeffs=distort_coeffs,
            flags=flags,
            avg_reproj_error=_avg_reproj_error,
            image_size=image_size,
            board_size=board_size,
            square_size=square_size,
            image_list=image_list,
            image_points_list=image_points_list,
            rvecs=rvecs,
            tvecs=tvecs,
            image_reproj_errors=image_reproj_errors,
            filename=model_xml_filename
        )

        self.loadParamsFromXML(model_xml_filename)

    def setUndistortParams(self, undistortFlags:int, undist_size:tuple, undist_scale:float, undist_offset:tuple):
        self.undistortSize = self.imageSize if undist_size is None else undist_size
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K=self.cameraMatrix,
            D=self.distortCoeffs,
            image_size=self.imageSize[::-1],
            R=None,
            P=None,
            balance=0,
            new_size=self.undistortSize[::-1],
            fov_scale=1.0
        )

        self.undistortScale = 1.0 if undist_scale is None else undist_scale
        new_camera_matrix[0,0] = new_camera_matrix[0,0] * self.undistortScale
        new_camera_matrix[1,1] = new_camera_matrix[1,1] * self.undistortScale

        if undist_offset is not None:
            self.undistortOffset = undist_offset
            new_camera_matrix[0,2] += self.undistortOffset[1]
            new_camera_matrix[1,2] += self.undistortOffset[0]
        else:
            self.undistortOffset = (0,0)

        # new_camera_matrix = np.array([[undist_size[1] * undist_scale, 0.0, undist_offset[1]],
        #                               [0.0, undist_size[0] * undist_scale, undist_offset[0]],
        #                               [0.0, 0.0, 1.0]])

        print("=== new_camera_matrix ===\n", new_camera_matrix)

        self.xMap,self.yMap = cv2.fisheye.initUndistortRectifyMap(
            K=self.cameraMatrix,
            D=self.distortCoeffs,
            R=None,
            P=new_camera_matrix,
            size=self.undistortSize[::-1],
            m1type=cv2.CV_32FC1
        )




if __name__ == '__main__':
    calibrateImagesXML = "omnidir_image_list.xml"
    calibrateModelXML = "fisheye_model.xml"
    distortImagesXML = "omnidir_image_list.xml"

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
    boardSize = (6, 9)  # in the order of (h,w)
    squareSize = (50, 50)  # in the order of (h,w)

    # undistortFlags = cv2.omnidir.RECTIFY_LONGLATI
    undistortFlags = cv2.omnidir.RECTIFY_CYLINDRICAL
    undistortSize = None  # in the order of (h,w)
    undistortScale = 1.0
    undistortOffset = (0,0)  # in the order of (h,w)

    model = FisheyeModel()
    # model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

    image_list = model.readFileListFromXML(calibrateImagesXML)
    model.validateFromFiles(image_list, 0, undistortSize, undistortScale, undistortOffset, calibrateModelXML, True)

    sys.exit(0)