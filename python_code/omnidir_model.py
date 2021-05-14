import os
import sys
import cv2
import numpy as np
import time
import math

# from utility import *
from base_model import *

# cv2.setUseOpenVX(True)
cv2.setUseOptimized(True)

DEBUG = True

class OmnidirModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.calibrateFlags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
        self.boardSize = (6, 9)
        self.squareSize = (50, 50)

        # undistortFlags = cv2.omnidir.RECTIFY_LONGLATI
        self.undistortFlags = cv2.omnidir.RECTIFY_CYLINDRICAL
        self.undistortSize = None
        self.undistortOffset = None
        self.undistortScale = 1.0
        self.modelType = "omnidir"

    def computeReprojectionErrors(self, object_points_list:list, image_points_list:list, rvecs:list, tvecs:list,
                                  idx:np.ndarray, camera_matrix:np.ndarray, xi:np.ndarray, distort_coeffs:np.ndarray):
        total_points = 0
        _rms_error = 0
        image_reproj_errors = []
        for i in idx[0,:]:
            #for objpoints,imgpoints,rvec,tvec in zip(object_points_list,image_points_list,rvecs,tvecs):
            objpoints,imgpoints = object_points_list[i],image_points_list[i]
            rvec,tvec = rvecs[i],tvecs[i]
            imgpoints2, _ = cv2.omnidir.projectPoints(objpoints, rvec, tvec, camera_matrix, xi[0,0], distort_coeffs)
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
        avg_reproj_error, camera_matrix, xi, distort_coeffs, rvecs, tvecs, idx = cv2.omnidir.calibrate(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            size=image_size[::-1],  # grayImage.shape[::-1],
            K=None, xi=None, D=None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 0.0001)  # 1e-8)
        )
        if DEBUG: print("used_indices:", idx)
        print("=== Root Mean Square Error ===\n", avg_reproj_error)
        print("=== Camera Matrix ===\n", camera_matrix)
        print("=== Xi ===\n", xi)
        print("=== Distortion Coefficients ===\n", distort_coeffs)

        _avg_reproj_error,image_reproj_errors = self.computeReprojectionErrors(
            object_points_list,
            image_points_list,
            rvecs,
            tvecs,
            idx,
            camera_matrix,
            xi,
            distort_coeffs
        )
        if DEBUG: print("=== Recalculated RMS error ===\n", _avg_reproj_error)

        self.saveParamsToXML(
            model_type="omnidir",
            camera_matrix=camera_matrix,
            xi=xi,
            distort_coeffs=distort_coeffs,
            flags=flags,
            avg_reproj_error=_avg_reproj_error,
            image_size=image_size,
            board_size=board_size,
            square_size=square_size,
            image_list=[image_list[i] for i in idx[0]],
            image_points_list=[image_points_list[i] for i in idx[0]],
            rvecs=[rvecs[i] for i in idx[0]],
            tvecs=[tvecs[i] for i in idx[0]],
            image_reproj_errors=image_reproj_errors,
            filename=model_xml_filename
        )

        self.loadParamsFromXML(model_xml_filename)

    def setUndistortParams(self, undist_flags:int, undist_size:tuple, undist_scale:float, undist_offset:tuple):
        self.undistortSize = self.imageSize if undist_size is None else undist_size
        self.undistortFlags = undist_flags

        if undist_flags == cv2.omnidir.RECTIFY_CYLINDRICAL or undist_flags == cv2.omnidir.RECTIFY_STEREOGRAPHIC or undist_flags == cv2.omnidir.RECTIFY_LONGLATI:
            self.undistortScale = 0.5 if undist_scale is None else undist_scale
            new_camera_matrix = np.array([[(self.undistortSize[1] / 3.1415) * self.undistortScale, 0.0, 0.0],
                                          [0.0, (self.undistortSize[0] / 3.1415) * self.undistortScale, self.undistortSize[0] / 4.0],
                                          [0.0, 0.0, 1.0]])
        elif undist_flags == cv2.omnidir.RECTIFY_PERSPECTIVE:
            self.undistortScale = 1.0 if undist_scale is None else undist_scale
            new_camera_matrix = np.array([[(self.undistortSize[1] / 4.0) * self.undistortScale, 0.0, self.undistortSize[1] / 2.0],
                                          [0.0, (self.undistortSize[0] / 4.0) * self.undistortScale, self.undistortSize[0] / 2.0],
                                          [0.0, 0.0, 1.0]])

        if undist_offset is not None:
            self.undistortOffset = undist_offset
            new_camera_matrix[0,2] += self.undistortOffset[1]
            new_camera_matrix[1,2] += self.undistortOffset[0]
        else:
            self.undistortOffset = (0,0)

        print("=== new_camera_marix ===\n", new_camera_matrix)

        self.xMap,self.yMap = cv2.omnidir.initUndistortRectifyMap(
            K=self.cameraMatrix,
            D=self.distortCoeffs,
            xi=self.xi,
            R=np.eye(3),
            P=new_camera_matrix,
            size=self.undistortSize[::-1],
            m1type=cv2.CV_32FC1,
            flags=self.undistortFlags
        )

if __name__ == "__main__":
    calibrateImagesXML = "omnidir_image_list.xml"
    calibrateModelXML = "omnidir_model.xml"
    distortImagesXML = "omnidir_image_list.xml"

    calibrateFlags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
    boardSize = (6, 9)  # in the order of (h,w)
    squareSize = (50, 50)  # in the order of (h,w)

    # undistortFlags = cv2.omnidir.RECTIFY_LONGLATI
    undistortFlags = cv2.omnidir.RECTIFY_CYLINDRICAL
    undistortSize = None  # in the order of (h,w)
    undistortOffset = None  # in the order of (h,w)
    undistortScale = None


    model = OmnidirModel()
    model.calibrateFromXML(calibrateImagesXML, boardSize, squareSize, calibrateFlags, calibrateModelXML, True)

    image_list = model.readFileListFromXML(calibrateImagesXML)
    model.validateFromFiles(image_list, undistortFlags, undistortSize, undistortScale, undistortOffset, calibrateModelXML, True)

    cv2.destroyAllWindows()
    sys.exit(0)

