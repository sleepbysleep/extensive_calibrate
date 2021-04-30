import os
import sys
import cv2
import numpy as np
import time
import math

# cv2.setUseOpenVX(True)
cv2.setUseOptimized(True)

DEBUG = True

class BaseModel:
    def __init__(self):
        # self.calibrateImagesXML = None
        # self.calibrateModelXML = None
        # self.distortImagesXML = None
        self.modelType = "Unknown"

        self.imageList = None
        self.objectPointsList = None
        self.imagePointsList = None
        self.detectedImageList = None
        self.imageSize = None

        self.calibrateFlags = 0
        self.boardSize = None
        self.squareSize = None

        self.cameraMatrix = None
        self.xi = None
        self.distortCoeffs = None
        self.avgReprojError = 0.0
        self.rVectors = []
        self.tVectors = []
        self.imageReprojErrors = None

        self.undistortFlags = 0
        self.undistortSize = None
        self.undistortOffset = None
        self.undistortScale = 1.0

        self.xMap = None
        self.yMap = None

    def readFileListFromXML(self, filename):
        '''
        Read the list of image filenames from XML file created by OpenCV's file storage.

        Parameters
        ----------
        filename: str
            readable XML filename which should have 'images' node enlisting image filenames.

        Returns
        -------
        images: list
        '''
        images = []
        file_storage = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        file_node = file_storage.getNode('images')
        for i in range(file_node.size()):
            images.append(file_node.at(i).string())

        file_storage.release()
        if DEBUG: print("images:\n", images)
        return images

    def createChessboardCorners(self, board_size: tuple = (6, 9), square_size: tuple = (50, 50)) -> np.ndarray:
        '''
        Create 3D points of [y, x, z] at crosses of the chessboard.

        Parameters
        ----------
        board_size: tuple
            (h,w) where h is the vertical number of crosses, and w is the horizontal number of crosses on the Chessboard.
        square_size: tuple
            (h,w) where h is the vertical length[mm] of square, and w is the horizontal length[mm] of square.

        Returns
        -------
        chessboard_corners: np.ndarray
            3D points of [y, x, z] at crosses of the chessboard.
        '''
        if DEBUG: print('board_size:', board_size)
        if DEBUG: print('square_size:', square_size)

        chessboard_corners = np.zeros((board_size[1] * board_size[0], 1, 3), np.float32)
        chessboard_corners[:, 0, :2] = \
            np.mgrid[0:board_size[1] * square_size[1]:square_size[1],
            0:board_size[0] * square_size[0]:square_size[0]].T.reshape(-1, 2)

        if DEBUG: print("chessboard_corners:\n", chessboard_corners)

        return chessboard_corners

    def detectChessboardCorners(self, image:np.ndarray, board_size:tuple, draw_corners:bool) -> (bool,np.array):
        found,image_points = cv2.findChessboardCorners(
            image,
            patternSize=board_size[::-1],
            corners=None,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  # cv2.CALIB_CB_FAST_CHECK +
                  0
        )

        if found:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(
                gray,
                corners=image_points,
                winSize=(11,11),
                zeroZone=(-1,-1),
                # criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            )

        if draw_corners:
            # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(image, board_size[::-1], image_points, found)
        return found,image_points

    def collectChessboardCorners(self, image_list: list, board_size: tuple = (6, 9), square_size: tuple = (50, 50),
                                show_result: bool = True):
        '''
        Gather couple of object points(a.k.a. ideal position), and image points of crosses on chessboard within image.

        Parameters
        ----------
        image_list: list
            List of image filenames.
        board_size: tuple
            (h,w) where h is the vertical number of crosses, and w is the horizontal number of crosses on the Chessboard.
        square_size: tuple
            (h,w) where h is the vertical length[mm] of square, and w is the horizontal length[mm] of square.
        show_result: bool
            Option for whether display the detections or not

        Returns
        -------
        chessboard_corners: np.ndarray
            3D points of [y, x, z] at crosses of the chessboard.
        '''

        object_points = self.createChessboardCorners(board_size, square_size)
        image_list_detected = []
        object_points_list = []
        image_points_list = []
        image_size = None
        for i, fname in enumerate(image_list):
            if DEBUG: print("Load image:", fname)
            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            if image_size is None:
                image_size = image.shape[:2]
            else:
                assert (image_size == image.shape[:2])

            found, image_points = self.detectChessboardCorners(image, board_size, show_result)

            if found:
                image_list_detected.append(fname)
                object_points_list.append(object_points)
                image_points_list.append(image_points)

            msg = f"{i + 1}/{len(image_list)}"
            if DEBUG:
                if found:
                    print(" Detecting crosses of chessboard complete - " + msg)
                else:
                    print(" Detecting crosses of chessboard failed! - " + msg)

            # Draw and display the corners
            if show_result:
                #cv2.drawChessboardCorners(image, board_size[::-1], image_points, found)

                text_size, base_line = cv2.getTextSize(msg, 1, 1, 1)
                cv2.putText(image, msg, (10 + text_size[1], 10 + 2 * base_line), 1, 1, (0, 0, 255))
                cv2.imshow('image', image)
                key = cv2.waitKey(500)
                if key == 27 or key == ord('q') or key == ord('Q'):
                    sys.exit(-1)
        return object_points_list, image_points_list, image_size, image_list_detected

    def saveParamsToXML(self, model_type: str, camera_matrix: np.ndarray, xi: np.ndarray, distort_coeffs: np.ndarray,
                        flags: int, avg_reproj_error: float,
                        image_size: tuple, board_size: tuple, square_size: tuple,
                        image_list: list, image_points_list: list, rvecs: list, tvecs: list,
                        image_reproj_errors: np.ndarray,
                        filename: str = "unknown_model.xml"):
        '''
        Save the parameters of camera calibration into2 OpenCV XML file.

        Parameters
        ----------
        filename: str
            writable XML filename which sould have every parameters related to camera calibration
        model_type: str
            supported the calibration model(i.e. generic, fisheye, and omnidir)
        image_size: tuple
            (h,w) specifying the geometric info. of image.
        board_size: tuple
            (h,w) w is the no. of horizontal crossing points, h is the no. of vertical crossing points) in chessboard chart
        square_size: tuple
            (h,w) in [mm] unit; specifying the geometric info. of unit rectangular in chessboard chart.
        flags: int
        camera_matrix: numpy.ndarray[numpy.float32]
        distort_coeffs: numpy.ndarray[numpy.float32]
        rvecs: list
        tvecs: list
        image_reproj_errors: list
        image_list: list
        image_points: list
        total_avg_err: float
        xi: numpy.ndarray[numpy.float32]
        alpha_value: float
        output_scale: float

        Returns
        -------
        None
        '''
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("calibration_time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        fs.write("calibration_model", model_type)
        fs.write("camera_matrix", camera_matrix)
        fs.write("xi", xi[0, 0])
        fs.write("distortion_coefficients", distort_coeffs)

        fs.write("flags", int(flags))
        fs.write("avg_reprojection_error", avg_reproj_error)

        fs.write("nframes", len(image_list))
        fs.write("image_width", int(image_size[1]))
        fs.write("image_height", int(image_size[0]))
        fs.write("board_width", int(board_size[1]))
        fs.write("board_height", int(board_size[0]))
        fs.write("square_width", int(square_size[1]))
        fs.write("square_height", int(square_size[0]))

        fs.write("used_images", image_list)
        # fs.write("<used_images>\n ")
        # # print("image list:")
        # for i,filename in enumerate(image_list):
        #     # print(filename)
        #     fs.write(" \"{}\"{}".format(filename.replace("\\", "/"), "" if i%2 == 0 else "\n "))
        # fs.write("</used_images>\n")

        image_point_array = np.zeros(shape=(len(image_points_list), len(image_points_list[0]), 2), dtype=np.float64)
        for i in range(len(image_points_list)):
            for j in range(len(image_points_list[i])):
                image_point_array[i, j, :] = image_points_list[i][j, 0, :]
        fs.write("image_points", image_point_array)

        if len(rvecs) > 0 and len(tvecs) > 0 and len(rvecs) == len(tvecs):
            merged = np.zeros(shape=(len(rvecs), 6), dtype=np.float64)
            for i in range(len(rvecs)):
                # merged[i,:] = rvecs[i].reshape(-1,3), tvecs[i].reshape(-1,3)
                merged[i, 0:3] = rvecs[i].reshape(-1, 3)
                merged[i, 3:6] = tvecs[i].reshape(-1, 3)
            fs.write("extrinsic_parameters", merged)

        fs.write("reprojection_errors", image_reproj_errors)

        fs.release()

    def loadParamsFromXML(self, filename: str = "omnidir_params.xml"):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.modelType = fs.getNode("calibration_model").string()
        self.cameraMatrix = fs.getNode("camera_matrix").mat()
        self.xi = np.array(fs.getNode("xi").real())
        self.distortCoeffs = fs.getNode("distortion_coefficients").mat()
        self.calibrateFlags = int(fs.getNode("flags").real())
        self.avgReprojError = fs.getNode("avg_reprojection_error").real()

        self.imageSize = (int(fs.getNode("image_height").real()), int(fs.getNode("image_width").real()))
        self.boardSize = (int(fs.getNode("board_height").real()), int(fs.getNode("board_width").real()))
        self.squareSize = (int(fs.getNode("square_height").real()), int(fs.getNode("square_width").real()))

        self.imageList = []
        for i in range(fs.getNode("used_images").size()):
            self.imageList.append(fs.getNode("used_images").at(i).string())

        image_points_array = fs.getNode("image_points").mat()
        self.imagePointsList = []
        for i in range(image_points_array.shape[0]):
            self.imagePointsList.append(image_points_array[i, :, :].reshape(-1, 1, 2))

        self.rVectors = []
        self.tVectors = []
        extrinsic_params = fs.getNode("extrinsic_parameters").mat()
        for i in range(extrinsic_params.shape[0]):
            self.rVectors.append(extrinsic_params[i, :3].reshape(3, 1))
            self.tVectors.append(extrinsic_params[i, 3:].reshape(3, 1))

        self.imageReprojErrors = fs.getNode("reprojection_errors").mat()

        fs.release()

    def calibrate(self, object_points_list:list, image_points_list:list, image_size:tuple,
                  board_size:tuple, square_size:tuple, flags:int, image_list:list, model_xml_filename:str):
        pass

    def calibrateFromFiles(self, image_list:list, board_size:tuple, square_size:tuple, flags:int, model_xml_filename:str, show_result:bool):
        object_points_list, image_points_list, image_size, image_list_detected = self.collectChessboardCorners(
            image_list=image_list,
            board_size=board_size,
            square_size=square_size,
            show_result=show_result
        )

        self.calibrate(
          object_points_list,
          image_points_list,
          image_size,
          board_size,
          square_size,
          flags,
          image_list_detected,
          model_xml_filename
        )

    def calibrateFromXML(self, image_xml_filename:str, board_size:tuple, square_size:tuple, flags:int, model_xml_filename:str, show_result:bool):
        image_list = self.readFileListFromXML(image_xml_filename)
        self.calibrateFromFiles(image_list, board_size, square_size, flags, model_xml_filename, show_result)

    def setUndistortParams(self, undist_flags:int, undist_size:tuple,  undist_scale:float, undist_offset:tuple):
        pass

    def undistortImage(self, distort_image:np.ndarray) -> np.ndarray:
        return cv2.remap(distort_image, self.xMap, self.yMap, cv2.INTER_LINEAR)

    def validateFromFiles(self, distort_image_list:list,
                           undist_flags:int, undist_size:tuple, undist_scale:float, undist_offset:tuple,
                           param_xml_filename:str, show_result:bool=True) -> list:
        self.loadParamsFromXML(param_xml_filename)
        print("=== Root Mean Square Error ===\n", self.avgReprojError)
        print("=== Camera Matrix ===\n", self.cameraMatrix)
        print("=== xi === \n", self.xi)
        print("=== Distance Coefficients ===\n", self.distortCoeffs)
        self.setUndistortParams(undist_flags, undist_size, undist_scale, undist_offset)

        undistort_image_list = []
        for filename in distort_image_list:
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            result = self.undistortImage(image)
            undistort_image_list.append(result)
            if show_result:
                cv2.imshow("testImage", image)
                cv2.imshow("resultImage", result)
                key = cv2.waitKey(500)
                if key == 27 or key == ord('q') or key == ord('Q'): break

        return undistort_image_list
