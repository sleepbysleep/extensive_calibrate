#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/opencv.hpp>

#include "base_model.hpp"

class GenericModel:public BaseModel
{
public:
  GenericModel() {
    this->calibrateImagesXML = "image_list.xml";
    this->calibrateModelXML = "generic_model.xml";
    this->distortImagesXML = "image_list.xml";
    this->calibrateFlags =
      cv::CALIB_RATIONAL_MODEL |
      //cv::CALIB_FIX_K3 |
      //cv::CALIB_FIX_K4 |
      //cv::CALIB_FIX_K5 |
      0;
    this->boardSize = cv::Size(9,6);
    this->squareSize = cv::Size(50,50);
    this->undistortSize = cv::Size(0,0);
    this->undistortOffset = cv::Point2f(0,0);
    this->undistortScale  = 1.0;
    this->modelType = "generic";
  }
  
  ~GenericModel() {
    this->imageList.clear();
    this->objectPointsList.clear();
    this->imagePointsList.clear();
    this->detectedImageList.clear();
  }

  void calibrate(
    const std::vector<std::vector<cv::Point3f>>& object_points_list,
    const std::vector<std::vector<cv::Point2f>>& image_points_list,
    const cv::Size& image_size,
    const cv::Size& board_size,
    const cv::Size& square_size,
    const int flags,
    const std::vector<std::string>& detected_image_list,
    const std::string& model_filename
  ) {
    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double avg_reproj_error = cv::calibrateCamera(
      object_points_list,
      image_points_list,
      image_size,
      camera_matrix,
      distort_coeffs,
      rvecs,
      tvecs,
      flags,
      cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 200, 0.0001)
    );

    std::cout << "=== Root Mean Square Error ===\n" << avg_reproj_error << "\n";
    std::cout << "=== Camera Matrix ===\n" << camera_matrix << "\n";
    std::cout << "=== Distance Coefficients ===\n" << distort_coeffs << "\n";

    std::vector<double> image_reproj_errors;
    double _avg_reproj_error = this->computeReprojectionErrors(
      object_points_list,
      image_points_list,
      rvecs,
      tvecs,
      camera_matrix,
      distort_coeffs,
      image_reproj_errors
    );
    std::cout << "=== Recalculted Root Mean Square Error ===\n" << _avg_reproj_error << "\n";

    this->saveParamsToXML(
      "generic_model",
      camera_matrix,
      0,
      distort_coeffs,
      flags,
      avg_reproj_error,
      image_size,
      board_size,
      square_size,
      detected_image_list,
      image_points_list,
      rvecs,
      tvecs,
      image_reproj_errors,
      model_filename
    );
    
    this->loadParamsFromXML(model_filename);
  }
  
  void setUndistortParams(
    const int undist_flags,
    const cv::Size& undist_size,
    const double undist_scale,
    const cv::Point2f& undist_offset
  ) {
    cv::Mat new_camera_matrix;
    cv::Rect roi;
    
    this->undistortSize = (undist_size == cv::Size(0,0) ? this->imageSize : undist_size);
    new_camera_matrix = cv::getOptimalNewCameraMatrix(
      this->cameraMatrix,
      this->distortCoeffs,
      this->imageSize,
      1,
      this->undistortSize,
      &roi
    );
    this->undistortScale = (undist_scale == 0 ? 1.0 : undist_scale);
    new_camera_matrix.at<float>(0,0) *= this->undistortScale;
    new_camera_matrix.at<float>(1,1) *= this->undistortScale;
    
    this->undistortOffset = undist_offset;
    new_camera_matrix.at<float>(0,2) += this->undistortOffset.x;
    new_camera_matrix.at<float>(1,2) += this->undistortOffset.y;

    std::cout << "=== new_camera_matrix ===\n" << new_camera_matrix << "\n";

    cv::initUndistortRectifyMap(
      this->cameraMatrix,
      this->distortCoeffs,
      cv::noArray()/*cv::Mat()*/,
      new_camera_matrix,
      this->undistortSize,
      CV_32FC1,
      this->xMap,
      this->yMap
    );
  }

protected:
  double computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& object_points_list,
    const std::vector<std::vector<cv::Point2f>>& image_points_list,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& camera_matrix,
    const cv::Mat& distort_coeffs,
    std::vector<double>& image_reproj_errors
  ) {
    std::vector<cv::Point2f> image_points2;
    int total_points = 0;
    double total_error = 0;
    
    image_reproj_errors.resize(object_points_list.size());

    for (size_t i = 0; i < object_points_list.size(); ++i) {
      cv::projectPoints(
	cv::Mat(object_points_list[i]),
	rvecs[i],
	tvecs[i],
	camera_matrix,
	distort_coeffs,
	image_points2
      );
      
      double err = cv::norm(cv::Mat(image_points_list[i]), cv::Mat(image_points2), cv::NORM_L2);
      int n = (int)object_points_list[i].size();
      image_reproj_errors[i] = (double)std::sqrt(err*err/n);
      total_error += err*err;
      total_points += n;
    }

    return std::sqrt(total_error/total_points);
  }

private:
  
};

