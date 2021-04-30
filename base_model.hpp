#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/opencv.hpp>

#define DEBUG true

class BaseModel
{
public:
  BaseModel() { }
  
  ~BaseModel() { }

  std::string calibrateImagesXML; // xml including the list of images for calibration
  std::string calibrateModelXML;  // xml including the parameters drived from/to calibration
  std::string distortImagesXML;	  // xml including the list of images for validation
  std::string modelType = "Unknown"; // the type of calibration: generic, fisheye, and omnidir

  std::vector<std::string> imageList; // the list of images for calibration
  std::vector<std::vector<cv::Point3f>> objectPointsList; // the list of realistic points in the 3D world 
  std::vector<std::vector<cv::Point2f>> imagePointsList; // the list of points detected in a chessboard image
  std::vector<std::string> detectedImageList; // the list of images used to calibration
  cv::Size imageSize;		// the size of image including chessboard exposure
  
  int calibrateFlags = 0;	// the flags for calibration
  cv::Size boardSize = cv::Size(9, 6); // horizontal, and vertical crossings in chessboard
  cv::Size squareSize = cv::Size(50, 50); // horizontal, and vertical length of chessboard's square

  cv::Mat cameraMatrix;		// transform matrix describing the optical geometry
  double xi;
  cv::Mat distortCoeffs;
  double avgReprojError;	// average error during calibration of refernce images
  std::vector<cv::Mat> rVectors;
  std::vector<cv::Mat> tVectors;
  std::vector<double> imageReprojErrors;
  
  int undistortFlags = 0;
  cv::Size undistortSize;
  cv::Point2f undistortOffset;
  double undistortScale = 1.0;

  cv::Mat xMap;
  cv::Mat yMap;

  std::vector<std::string> readFileListFromXML(const std::string&  filename) {
    std::vector<std::string> image_list;

    if (DEBUG) std::cout << "readFileListFromXML: " << filename << "\n";
  
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (fs.isOpened()) {
      cv::FileNode n = fs.getFirstTopLevelNode();
      if (n.type() == cv::FileNode::SEQ) {
	for (auto it = n.begin(); it != n.end(); ++it)
	  image_list.push_back(static_cast<std::string>(*it));
      }
    }
    
    return image_list;
  }
  
  bool detectChessboardCorners(
    cv::Mat& image,
    const cv::Size& board_size,
    std::vector<cv::Point2f>& image_points,
    bool draw_corners=true
  ) {
    bool found = cv::findChessboardCorners(
      image,
      board_size,
      image_points,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
    );

    if (found) {
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
      cv::cornerSubPix(
	gray,
	image_points,
	cv::Size(11,11),
	cv::Size(-1,-1),
	cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1)
      );
      
      // image_list_detected.push_back(image_list[i]);
      // object_points_list.push_back(object_points);
      // image_points_list.push_back(image_points);
      
      if (draw_corners)
	cv::drawChessboardCorners(image, board_size, cv::Mat(image_points), found);
    }
    
    return found;
  }

  void saveParamsToXML(
    const std::string& model_type,
    const cv::Mat& camera_matrix,
    const double xi,
    const cv::Mat& distort_coeffs,
    const int calibrate_flags,
    const double avg_reproj_error,
    const cv::Size& image_size,
    const cv::Size& board_size,
    const cv::Size& square_size,
    const std::vector<std::string>& image_list,
    const std::vector<std::vector<cv::Point2f>>& image_points_list,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const std::vector<double>& image_reproj_errors,
    const std::string& filename="unknown_model.xml"
  ) {
    //void saveParamsToXML(const std::string& filename="unknown_model.xml") {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    time_t tt;
    time(&tt);
    
    struct tm *t2 = localtime(&tt);
    
    char buf[1024];
    strftime(buf, sizeof(buf)-1, "%c", t2);

    fs << "calibration_time" << buf;
    fs << "calibration_model" << model_type;
    fs << "camera_matrix" << camera_matrix;
    fs << "xi" << xi;
    fs << "distortion_coefficients" << distort_coeffs;

    fs << "flags" << calibrate_flags;
    fs << "avg_reprojection_error" << avg_reproj_error;

    fs << "nframes" << (int)image_list.size();
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "board_width" << board_size.width;
    fs << "board_height" << board_size.height;
    fs << "square_width" << square_size.width;
    fs << "square_height" << square_size.height;

    fs << "used_images" << image_list;

    cv::Mat imagePtMat((int)image_points_list.size(),
		       (int)image_points_list[0].size(),
		       CV_64FC2);
    for (size_t i = 0; i < image_points_list.size(); ++i) {
      for (size_t j = 0; j < image_points_list[i].size(); ++j) {
	imagePtMat.at<cv::Vec2d>(i,j)[0] = image_points_list[i][j].x;
	imagePtMat.at<cv::Vec2d>(i,j)[1] = image_points_list[i][j].y;
      }
    }
    fs << "image_points" << imagePtMat;
  
    if (!rvecs.empty() and !tvecs.empty() and rvecs.size() == tvecs.size()) {
      assert(rvecs[0].type() == tvecs[0].type());
      
      cv::Mat merged((int)rvecs.size(), 6, rvecs[0].type());
      for (size_t i = 0; i < rvecs.size(); ++i) {
	cv::Mat r = merged(cv::Range(i, i+1), cv::Range(0,3));
	cv::Mat t = merged(cv::Range(i, i+1), cv::Range(3,6));

	assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
	assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
	//*.t() is MatExpr (not Mat) so we can use assignment operator
	r = rvecs[i].t();
	t = tvecs[i].t();
      }
      fs << "extrinsic_parameters" << merged;
    }

    cv::Mat m = cv::Mat(image_reproj_errors.size(), 1, CV_64F);
    std::memcpy(m.data, image_reproj_errors.data(), image_reproj_errors.size()*sizeof(double));
    fs << "reprojection_errors" << m;
  }

  void loadParamsFromXML(const std::string& filename="unknown_params.xml") {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    fs["calibration_model"] >> this->modelType;
    fs["camera_matrix"] >> this->cameraMatrix;
    fs["xi"] >> this->xi;
    fs["distortion_coefficients"] >> this->distortCoeffs;
    fs["flags"] >> this->calibrateFlags;
    fs["avg_reprojection_error"] >> this->avgReprojError;
    
    fs["image_width"] >> this->imageSize.width;
    fs["image_height"] >> this->imageSize.height;
    fs["board_width"] >> this->boardSize.width;
    fs["board_height"] >> this->boardSize.height;
    fs["square_width"] >> this->squareSize.width;
    fs["square_height"] >> this->squareSize.height;

    this->imageList.clear();
    cv::FileNode n = fs["used_images"];
    if (n.type() == cv::FileNode::SEQ) {
      for (auto it = n.begin(); it != n.end(); ++it) {
	this->imageList.push_back((std::string)*it);
      }
    }

    this->imagePointsList.clear();
    cv::Mat image_points_mat;
    fs["image_points"] >> image_points_mat;
    for (int i = 0; i < image_points_mat.rows; ++i) {
      std::vector<cv::Point2f> image_points;      
      for (int j = 0; j < image_points_mat.cols; ++j) {
	cv::Point2f pt(image_points_mat.at<cv::Vec2d>(i,j)[0],
		       image_points_mat.at<cv::Vec2d>(i,j)[1]);
	image_points.push_back(pt);
      }
      this->imagePointsList.push_back(image_points);
    }

    this->rVectors.clear();
    this->tVectors.clear();
    cv::Mat extrinsic_params;
    fs["extrinsic_parameters"] >> extrinsic_params;
    for (int i = 0; i < extrinsic_params.rows; ++i) {
      cv::Mat r = extrinsic_params(cv::Range(i,i+1), cv::Range(0,3));
      cv::Mat t = extrinsic_params(cv::Range(i,i+1), cv::Range(3,6));
      this->rVectors.push_back(r.t());
      this->tVectors.push_back(t.t());
    }

    cv::Mat reproj_errors;
    fs["reprojection_errors"] >> reproj_errors;
    for (int i = 0; i < reproj_errors.rows; ++i) {
      this->imageReprojErrors.push_back(reproj_errors.at<double>(i,0));
    }
  }

  virtual void calibrate(
    const std::vector<std::vector<cv::Point3f>>& object_points_list,
    const std::vector<std::vector<cv::Point2f>>& image_points_list,
    const cv::Size& image_size,
    const cv::Size& board_size,
    const cv::Size& square_size,
    const int flags,
    const std::vector<std::string>& detected_image_list,
    const std::string& model_filename
  ) { }

  void calibrateFromFiles(
    const std::vector<std::string>& image_list,
    const cv::Size& board_size,
    const cv::Size& square_size,
    const int flags,
    const std::string& model_filename,
    const bool show_result=true
  ) {
    // Gather object points, and image points regarding to Chessboard.
    std::vector<std::vector<cv::Point3f>> object_points_list;
    std::vector<std::vector<cv::Point2f>> image_points_list;
    std::vector<std::string> detected_image_list;
    cv::Size image_size;
    this->collectChessboardCorners(
      image_list,
      board_size,
      square_size,
      image_size,
      object_points_list,
      image_points_list,
      detected_image_list,
      show_result
    );

    this->calibrate(
      object_points_list,
      image_points_list,
      image_size,
      board_size,
      square_size,
      flags,
      detected_image_list,
      model_filename
    );
  }    

  void calibrateFromXML(
    const std::string& image_xml_filename,
    const cv::Size& board_size,
    const cv::Size& square_size,
    const int flags,
    const std::string& model_xml_filename,
    const bool show_result=true
  ) {
    // Read image list from XML file
    std::vector<std::string> image_list = this->readFileListFromXML(image_xml_filename);
    this->calibrateFromFiles(
      image_list,
      board_size,
      square_size,
      flags,
      model_xml_filename,
      show_result
    );
  }

  virtual void setUndistortParams(
    const int undist_flags,
    const cv::Size& undist_size,
    const double undist_scale,
    const cv::Point2f& undist_offset
  ) { }

  void undistortImage(const cv::Mat& distort_image, cv::Mat& undistort_image) {
    cv::remap(distort_image, undistort_image, this->xMap, this->yMap, cv::INTER_LINEAR);
  }

  std::vector<cv::Mat> validateFromFiles(
    const std::vector<std::string>& distort_image_list,
    const int undist_flags,
    const cv::Size& undist_size,
    const double undist_scale,
    const cv::Point2f& undist_offset,
    const std::string& model_filename,
    bool show_result=true
  ) {
    this->loadParamsFromXML(model_filename);
    std::cout << "=== Root Mean Square Error ===\n" << this->avgReprojError << "\n";
    std::cout << "=== Camera Matrix ===\n" << this->cameraMatrix << "\n";
    std::cout << "=== Xi ===\n" << this->xi << "\n";
    std::cout << "=== Distortion Coefficients ===\n" << this->distortCoeffs << "\n";

    this->setUndistortParams(undist_flags, undist_size, undist_scale, undist_offset);
    
    std::vector<cv::Mat> undistort_image_list;
    undistort_image_list.clear();
    for (auto filename:distort_image_list) {
      cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
      cv::Mat result;
      this->undistortImage(image, result);
      undistort_image_list.push_back(result);
      if (show_result) {
	// int new_rows = std::max(image.rows, result.rows);
	// int new_cols = std::max(image.cols, result.cols);
	// cv::Mat mosaic(new_rows, 2*new_cols, CV_8UC3);
	// mosaic(cv::Rect(0,0,image.cols,image.rows)) = image;
	// mosaic(cv::Rect(new_cols,0,result.cols,result.rows)) = result;
	// cv::imshow("Distort and Undistort Images", mosaic);
	cv::imshow("Distort Image", image);
	cv::imshow("Undistort Image", result);
	char key = (char)cv::waitKey(500);
	if (key == 27 || key == 'q' || key == 'Q') exit(-1);
      }
    }
    return undistort_image_list;
  }
  
protected:
  std::vector<cv::Point3f> createChessboardCorners(
      const cv::Size& board_size,
      const cv::Size& square_size
  ) {
    if (DEBUG) std::cout << "board_size: " << board_size << "\n";
    if (DEBUG) std::cout << "square_size: " << square_size << "\n";
    
    std::vector<cv::Point3f> corners;
    for (int i = 0; i < board_size.height; ++i) {
      for (int j = 0; j < board_size.width; ++j) {
	corners.push_back(
	  cv::Point3f(
	    float(j*square_size.width),
	    float(i*square_size.height),
	    0.0
	  )
	);
      }
    }

    return corners;
  }

  int collectChessboardCorners(
    const std::vector<std::string>& image_list,
    const cv::Size& board_size,
    const cv::Size& square_size,
    cv::Size& image_size,
    std::vector<std::vector<cv::Point3f>>& object_points_list,
    std::vector<std::vector<cv::Point2f>>& image_points_list,
    std::vector<std::string>& image_list_detected,
    bool show_result=true
  ) {
    std::vector<cv::Point3f> object_points = this->createChessboardCorners(board_size, square_size);

    int count = 0;
    image_list_detected.clear();
    object_points_list.clear();
    image_points_list.clear();
    for (size_t i = 0; i < image_list.size(); ++i) {
      if (DEBUG) std::cout << "Load image: " << image_list[i];
    
      cv::Mat image = cv::imread(image_list[i], cv::IMREAD_COLOR);
      if (image.empty()) {
	std::cerr << "Invalid image path: " + image_list[i] << std::endl;
	break;
      }

      if (image_size == cv::Size(0,0)) {
	image_size.width = image.cols;
	image_size.height = image.rows;
      }
      assert(image_size.width == image.cols and image_size.height == image.rows);
      
      std::vector<cv::Point2f> image_points;
      bool found = this->detectChessboardCorners(
	image,
	board_size,
	image_points,
	show_result
      );
      
      if (found) {
	image_list_detected.push_back(image_list[i]);
	object_points_list.push_back(object_points);
	image_points_list.push_back(image_points);
	++count;
      }

      std::string msg = cv::format("%d/%d", (int)(i+1), (int)image_list.size());
      if (DEBUG) {
	if (found) {
	  std::cout << " Detecting crosses of chessboard complete - " << msg << "\n";
	} else {
	  std::cout << " Detecting crosses of chessboard failed! - " << msg << "\n";
	}
      }

      if (show_result) {
	//cv::drawChessboardCorners(image, board_size, cv::Mat(image_points), found);

	int base_line = 0;
	cv::Size text_size = cv::getTextSize(msg, 1, 1, 1, &base_line);

	cv::putText(image, msg,
		    cv::Point(10 + text_size.width, 10 + 2 * base_line),
		    1, 1, cv::Scalar(0,0,255));
	cv::imshow("Image", image);
	char key = (char)cv::waitKey(500);
	if (key == 27 || key == 'q' || key == 'Q') {
	  exit(-1);
	}
      }
    }
    return count;
  }
  
private:
  
};
