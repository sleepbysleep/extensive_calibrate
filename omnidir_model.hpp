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




namespace omnidir_model {
  void calcChessboardCorners(const cv::Size &boardSize, const cv::Size2d &squareSize, cv::Mat& corners)
  {
    // corners has type of CV_64FC3
    corners.release();
    int n = boardSize.width * boardSize.height;
    corners.create(n, 1, CV_64FC3);
    cv::Vec3d *ptr = corners.ptr<cv::Vec3d>();
    for (int i = 0; i < boardSize.height; ++i) {
      for (int j = 0; j < boardSize.width; ++j) {
	ptr[i*boardSize.width + j] = cv::Vec3d(double(j * squareSize.width), double(i * squareSize.height), 0.0);
      }
    }
  }

  bool detectChessboardCorners(const std::vector<std::string>& list, std::vector<std::string>& list_detected,
			      std::vector<cv::Mat>& imagePoints, cv::Size boardSize, cv::Size& imageSize)
  {
    imagePoints.resize(0);
    list_detected.resize(0);
    int n_img = (int)list.size();
    cv::Mat img;
    for(int i = 0; i < n_img; ++i) {
      std::cout << list[i] << "... ";
      cv::Mat points;
      img = cv::imread(list[i], cv::IMREAD_GRAYSCALE);
      bool found = cv::findChessboardCorners( img, boardSize, points);
      if (found) {
	if (points.type() != CV_64FC2)
	  points.convertTo(points, CV_64FC2);
	imagePoints.push_back(points);
	list_detected.push_back(list[i]);
      }
      std::cout << (found ? "FOUND" : "NO") << std::endl;
    }
    if (!img.empty())
      imageSize = img.size();
    if (imagePoints.size() < 3)
      return false;
    else
      return true;
  }

  void saveCameraParams(const std::string & filename,
			const cv::Size& image_size,
			const cv::Size& board_size,
			const cv::Size2d& square_size,
			const double aspect_ratio, 
			int flags, const cv::Mat& camera_matrix,
			const cv::Mat& distortion_coeffs, const double xi,
			const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs,
			std::vector<std::string> detected_list, const cv::Mat& idx,
			const double rms, const std::vector<cv::Mat>& image_points)
  {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime(buf, sizeof(buf)-1, "%c", t2);

    fs << "calibration_time" << buf;
    fs << "calibration_model" << "omnidir";

    fs << "camera_matrix" << camera_matrix;
    fs << "xi" << xi;
    fs << "distortion_coefficients" << distortion_coeffs;


    // if (flags != 0) {
    //   sprintf( buf, "flags: %s%s%s%s%s%s%s%s%s",
    // 	       flags & cv::omnidir::CALIB_USE_GUESS ? "+use_intrinsic_guess" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_SKEW ? "+fix_skew" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_K1 ? "+fix_k1" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_K2 ? "+fix_k2" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_P1 ? "+fix_p1" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_P2 ? "+fix_p2" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_XI ? "+fix_xi" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_GAMMA ? "+fix_gamma" : "",
    // 	       flags & cv::omnidir::CALIB_FIX_CENTER ? "+fix_center" : "");
    //   fs.writeComment(buf, false);
    // }
    
    fs << "flags" << flags;
    fs << "avg_reprojection_error" << rms;


    if (!rvecs.empty())
      fs << "nFrames" << (int)rvecs.size();

    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    
    fs << "board_width" << board_size.width;
    fs << "board_height" << board_size.height;

    fs << "square_width" << (int)square_size.width;
    fs << "square_height" << (int)square_size.height;

    //fs << "aspect_ratio" << aspect_ratio;


    //cvWriteComment( *fs, "names of images that are acturally used in calibration", 0 );
    fs << "used_images" << "[";
    for (int i = 0;  i < (int)idx.total(); ++i) {
      fs << detected_list[(int)idx.at<int>(i)];
    }
    fs << "]";

    if (!rvecs.empty() && !tvecs.empty()) {
      cv::Mat rvec_tvec((int)rvecs.size(), 6, CV_64F);
      for (int i = 0; i < (int)rvecs.size(); ++i) {
	cv::Mat(rvecs[i]).reshape(1, 1).copyTo(rvec_tvec(cv::Rect(0, i, 3, 1)));
	cv::Mat(tvecs[i]).reshape(1, 1).copyTo(rvec_tvec(cv::Rect(3, i, 3, 1)));
      }
      //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
      fs << "extrinsic_parameters" << rvec_tvec;
    }

    if ( !image_points.empty() ) {
      cv::Mat imageMat((int)image_points.size(), (int)image_points[0].total(), CV_64FC2);
      for (int i = 0; i < (int)image_points.size(); ++i)
	{
	  cv::Mat r = imageMat.row(i).reshape(2, imageMat.cols);
	  cv::Mat imagei(image_points[i]);
	  imagei.copyTo(r);
	}
      fs << "image_points" << imageMat;
    }
  }  
};
