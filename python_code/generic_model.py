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

class GenericModel(BaseModel):
    def __init__(self):
        # self.super().__init__()

        self.calibrateFlags = cv2.CALIB_RATIONAL_MODEL #+ cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5
        self.boardSize = (6,9)
        self.squareSize = (50,50)
        self.undistortSize = None
        self.undistortOffset = None
        self.undistortScale = 1.0
        self.modelType = "generic"

    def computeReprojectionErrors(self, object_points_list:list, image_points_list:list, rvecs:list, tvecs:list,
                                  camera_matrix:np.ndarray, distort_coeffs:np.ndarray):
        total_points = 0
        _rms_error = 0
        image_reproj_errors = []
        for objpoints,imgpoints,rvec,tvec in zip(object_points_list,image_points_list,rvecs,tvecs):
            imgpoints2, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, distort_coeffs)
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
        avg_reproj_error, camera_matrix, distort_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            imageSize=image_size[::-1],  # grayImage.shape[::-1],
            cameraMatrix=None, distCoeffs=None, rvecs=None, tvecs=None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 0.0001)  # 1e-8)
        )
        '''
        calib3d:
            The new rational distortion model:
                x' = x*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + <tangential_distortion for x>,
                y' = y*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + <tangential_distortion for y>
                has been introduced. It is useful for calibration of cameras with wide-angle lenses.
                Because of the increased number of parameters to optimize you need to supply more data to robustly estimate all of them.
                Or, simply initialize the distortion vectors with zeros and pass
                CV_CALIB_RATIONAL_MODEL + CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 or other such combinations to selectively enable or disable certain coefficients.
    
            rectification of trinocular camera setup, where all 3 heads are on the same line, is added. see samples/cpp/3calibration.cpp
        '''
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
            model_type="generic",
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

    def setUndistortParams(self, undist_flags:int, undist_size:tuple, undist_scale:float, undist_offset:tuple):
        self.undistortSize = self.imageSize if undist_size is None else undist_size

        new_camera_matrix,roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distortCoeffs,
            imageSize=self.imageSize[::-1],
            alpha=1,
            newImgSize=self.undistortSize[::-1],
            centerPrincipalPoint=True
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

        # new_camera_matrix = np.array([[self.undistortSize[1]*self.undistortScale, 0.0, self.undistortOffset[1]],
        #                               [0.0, self.undistortSize[0]*self.undistortScale, self.undistortOffset[0]],
        #                               [0.0, 0.0, 1.0]])
        print("=== new_camera_marix ===\n", new_camera_matrix)

        self.xMap,self.yMap = cv2.initUndistortRectifyMap(
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distortCoeffs,
            R=np.eye(3),
            newCameraMatrix=new_camera_matrix,
            size=self.undistortSize[::-1],
            m1type=cv2.CV_32FC1
        )

if __name__ == "__main__":
    calibrateImagesXML = "image_list.xml"
    calibrateModelXML = "generic_model.xml"
    distortImagesXML = "image_list.xml"

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
    boardSize = (6, 9)  # in the order of (h,w)
    squareSize = (50, 50)  # in the order of (h,w)

    undistortSize = None  # in the order of (h,w)
    undistortOffset = (0,50)  # in the order of (h,w)
    undistortScale = 1.35

    model = GenericModel()
    model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

    image_list = model.readFileListFromXML(calibrateImagesXML)
    model.validateFromFiles(image_list, 0, undistortSize, undistortScale, undistortOffset, calibrateModelXML, True)

    sys.exit(0)