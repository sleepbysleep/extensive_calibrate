#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include "base_model.hpp"

class OmnidirModel:public BaseModel
{
public:
  OmnidirModel() {
    this->calibrateFlags = cv::omnidir::CALIB_USE_GUESS + cv::omnidir::CALIB_FIX_SKEW + cv::omnidir::CALIB_FIX_CENTER;
    this->boardSize = cv::Size(9,6);
    this->squareSize = cv::Size(50,50);

    //this->undistortFlags = cv::omnidir::RECTIFY_LONGLATI;
    this->undistortFlags = cv::omnidir::RECTIFY_CYLINDRICAL;
    this->undistortSize = cv::Size(0,0);
    this->undistortOffset = cv::Point2f(0,0);
    this->undistortScale = 1.0;
    this->modelType = "omnidir";
  }

  ~OmnidirModel() {
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
    cv::Mat idx, xi, camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double avg_reproj_error = cv::omnidir::calibrate(
      object_points_list,
      image_points_list,
      image_size,
      camera_matrix,
      xi,
      distort_coeffs,
      rvecs,
      tvecs,
      flags,
      cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 200, 0.0001),
      idx
    );
    if (DEBUG) std::cout << "used_indices: \n" << idx << "\n";
    std::cout << "=== Root Mean Square Error ===\n" << avg_reproj_error << "\n";
    std::cout << "=== Camera Matrix ===\n" << camera_matrix << "\n";
    std::cout << "=== Distance Coefficients ===\n" << distort_coeffs << "\n";

    std::vector<double> image_reproj_errors;
    double _avg_reproj_error = this->computeReprojectionErrors(
      object_points_list,
      image_points_list,
      rvecs,
      tvecs,
      idx,
      camera_matrix,
      xi,
      distort_coeffs,
      image_reproj_errors
    );
    std::cout << "=== Recalculted Root Mean Square Error ===\n" << _avg_reproj_error << "\n";
    //std::vector<std::string> _image_list;

    this->saveParamsToXML(
      "omnidir_model",
      camera_matrix,
      xi.at<double>(0),
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
    const int undist_flags=0,
    const cv::Size& undist_size=cv::Size(0,0),
    const double undist_scale=0.9,
    const cv::Point2f& undist_offset=cv::Point2f(0,0)
  ) {
    this->undistortSize = (undist_size == cv::Size(0,0) ? this->imageSize : undist_size);    this->undistortFlags = undist_flags;

    cv::Mat new_camera_mat(3,3,CV_32F);
    if (undist_flags == cv::omnidir::RECTIFY_CYLINDRICAL or
	undist_flags == cv::omnidir::RECTIFY_STEREOGRAPHIC or
	undist_flags == cv::omnidir::RECTIFY_LONGLATI) {
      this->undistortScale = (undist_scale == 0 ? 0.5 : undist_scale);
      new_camera_mat.at<float>(0,0) = (this->undistortSize.width/3.1415)*this->undistortScale;
      new_camera_mat.at<float>(0,1) = 0.0; 
      new_camera_mat.at<float>(0,2) = 0.0;
      new_camera_mat.at<float>(1,0) = 0.0;
      new_camera_mat.at<float>(1,1) = (this->undistortSize.height/3.1415)*this->undistortScale;
      new_camera_mat.at<float>(1,2) = this->undistortSize.height/4.0;
      new_camera_mat.at<float>(2,0) = 0.0;
      new_camera_mat.at<float>(2,1) = 0.0;
      new_camera_mat.at<float>(2,2) = 1.0;
    } else if (undist_flags == cv::omnidir::RECTIFY_PERSPECTIVE) {
      this->undistortScale = (undist_scale == 0 ? 1.0 : undist_scale);
      new_camera_mat.at<float>(0,0) = (this->undistortSize.width/4.0)*this->undistortScale;
      new_camera_mat.at<float>(0,1) = 0.0;
      new_camera_mat.at<float>(0,2) = this->undistortSize.width/2.0;
      new_camera_mat.at<float>(1,0) = 0.0;
      new_camera_mat.at<float>(1,1) = (this->undistortSize.height/4.0)*this->undistortScale;
      new_camera_mat.at<float>(1,2) = this->undistortSize.height/2.0;
      new_camera_mat.at<float>(2,0) = 0.0;
      new_camera_mat.at<float>(2,1) = 0.0;
      new_camera_mat.at<float>(2,2) = 1.0;
    }

    this->undistortOffset = undist_offset;
    new_camera_mat.at<float>(0,2) += this->undistortOffset.x;
    new_camera_mat.at<float>(1,2) += this->undistortOffset.y;

    std::cout << "=== new_camera_matrix ===\n" << new_camera_mat << "\n";

    cv::omnidir::initUndistortRectifyMap(
      this->cameraMatrix,
      this->distortCoeffs,
      this->xi,
      cv::noArray(),
      new_camera_mat,
      this->imageSize,
      CV_32FC1,
      this->xMap,
      this->yMap,
      this->undistortFlags
    );
  }
  
protected:
  double computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& object_points_list,
    const std::vector<std::vector<cv::Point2f>>& image_points_list,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& idx,
    const cv::Mat& camera_matrix,
    const cv::Mat& xi,
    const cv::Mat& distort_coeffs,
    std::vector<double>& image_reproj_errors
  ) {
    std::vector<cv::Point2f> image_points2;
    int total_points = 0;
    double total_error = 0;
    
    image_reproj_errors.resize(object_points_list.size());
    for (int k = 0;  k < (int)idx.total(); ++k) {
      int i = idx.at<int>(k);
      cv::omnidir::projectPoints(
	object_points_list[i],
	image_points2,
	rvecs[i],
	tvecs[i],
	camera_matrix,
	xi.at<double>(0),
	distort_coeffs
      );
      
      double err = cv::norm(
	cv::Mat(image_points_list[i]),
	cv::Mat(image_points2),
	cv::NORM_L2
      );
      int n = (int)object_points_list[i].size();
      image_reproj_errors[i] = (double)std::sqrt(err*err/n);
      total_error += err*err;
      total_points += n;
    }

    return std::sqrt(total_error/total_points);
  }

private:
  
};


