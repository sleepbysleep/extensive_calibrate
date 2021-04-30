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


namespace generic_model {
  double computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f> >& objectPoints,
    const std::vector<std::vector<cv::Point2f> >& imagePoints,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    std::vector<float>& perViewErrors 
  )
  {
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); i++ ) {
      cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
      err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
      int n = (int)objectPoints[i].size();
      perViewErrors[i] = (float)std::sqrt(err*err/n);
      totalErr += err*err;
      totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
  }

  enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

  void calcChessboardCorners(cv::Size boardSize, float squareSize,
			     std::vector<cv::Point3f>& corners, Pattern patternType = CHESSBOARD)
  {
    corners.resize(0);

    switch (patternType) {
    case CHESSBOARD:
    case CIRCLES_GRID:
      for( int i = 0; i < boardSize.height; i++ )
	for( int j = 0; j < boardSize.width; j++ )
	  corners.push_back(cv::Point3f(float(j*squareSize),
					float(i*squareSize), 0));
      break;

    case ASYMMETRIC_CIRCLES_GRID:
      for( int i = 0; i < boardSize.height; i++ )
	for( int j = 0; j < boardSize.width; j++ )
	  corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize),
					float(i*squareSize), 0));
      break;

    default:
      CV_Error(cv::Error::StsBadArg, "Unknown pattern type\n");
    }
  }

  bool runCalibration(std::vector<std::vector<cv::Point2f> > imagePoints,
		      cv::Size imageSize, cv::Size boardSize, Pattern patternType,
		      float squareSize, float aspectRatio,
		      int flags, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
		      std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
		      std::vector<float>& reprojErrs,
		      double& totalAvgErr)
  {
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    if (flags & cv::CALIB_FIX_ASPECT_RATIO)
      cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    generic_model::calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

/*
    for (int i = 0; i < objectPoints.size(); ++i) {
    int i = 0;
      for (int j = 0; j < objectPoints[i].size(); ++j) {
        std::cout << objectPoints[i][j] << " - " << imagePoints[i][j] << "\n";
      }
    }
*/


    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
				     distCoeffs, rvecs, tvecs,
				     cv::CALIB_RATIONAL_MODEL |
				     //cv::CALIB_FIX_K3 |
				     //cv::CALIB_FIX_K4 |
				     //cv::CALIB_FIX_K5 |
				     flags);

    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = generic_model::computeReprojectionErrors(objectPoints, imagePoints,
							   rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
  }

  void saveCameraParams(const std::string& filename,
			cv::Size imageSize, cv::Size boardSize,
			float squareSize, float aspectRatio, int flags,
			const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
			const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
			const std::vector<float>& reprojErrs,
			const std::vector<std::vector<cv::Point2f> >& imagePoints,
			double totalAvgErr )
  {
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
      fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & cv::CALIB_FIX_ASPECT_RATIO )
      fs << "aspectRatio" << aspectRatio;

    if( flags != 0 ) {
      sprintf( buf, "flags: %s%s%s%s",
	       flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
	       flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
	       flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
	       flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
      //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
      fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() ) {
      CV_Assert(rvecs[0].type() == tvecs[0].type());
      cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
      for( int i = 0; i < (int)rvecs.size(); i++ )
	{
	  cv::Mat r = bigmat(cv::Range(i, i+1), cv::Range(0,3));
	  cv::Mat t = bigmat(cv::Range(i, i+1), cv::Range(3,6));

	  CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
	  CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
	  //*.t() is MatExpr (not Mat) so we can use assignment operator
	  r = rvecs[i].t();
	  t = tvecs[i].t();
	}
      //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
      fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() ) {
      cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
      for( int i = 0; i < (int)imagePoints.size(); i++ )
	{
	  cv::Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
	  cv::Mat imgpti(imagePoints[i]);
	  imgpti.copyTo(r);
	}
      fs << "image_points" << imagePtMat;
    }
  }

  bool runAndSave(const std::string& outputFilename,
		  const std::vector<std::vector<cv::Point2f> >& imagePoints,
		  cv::Size imageSize, cv::Size boardSize, Pattern patternType, float squareSize,
		  float aspectRatio, int flags, cv::Mat& cameraMatrix,
		  cv::Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
  {
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = generic_model::runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
					    aspectRatio, flags, cameraMatrix, distCoeffs,
					    rvecs, tvecs, reprojErrs, totalAvgErr);

    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if( ok )
      generic_model::saveCameraParams(outputFilename, imageSize,
				      boardSize, squareSize, aspectRatio,
				      flags, cameraMatrix, distCoeffs,
				      writeExtrinsics ? rvecs : std::vector<cv::Mat>(),
				      writeExtrinsics ? tvecs : std::vector<cv::Mat>(),
				      writeExtrinsics ? reprojErrs : std::vector<float>(),
				      writePoints ? imagePoints : std::vector<std::vector<cv::Point2f> >(),
				      totalAvgErr );
    return ok;
  }
};
