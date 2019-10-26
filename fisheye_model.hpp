namespace fisheye_model {
  double computeReprojectionErrors(std::vector<std::vector<cv::Point3f>> const& objectPoints,
				   std::vector<std::vector<cv::Point2f>> const& imagePoints,
				   std::vector<cv::Mat> const& rvecs,
				   std::vector<cv::Mat> const& tvecs,
				   cv::Mat const& cameraMatrix,
				   cv::Mat const& distCoeffs,
				   std::vector<float> & perViewErrors )
  {
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ ) {

      cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);

      err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
      int n = (int)objectPoints[i].size();
      perViewErrors[i] = (float)std::sqrt(err*err/n);
      totalErr += err*err;
      totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
  }

  void calcChessboardCorners(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners)
  {
    corners.resize(0);

    for (int i = 0; i < boardSize.height; ++i) {
      for (int j = 0; j < boardSize.width; ++j) {
	corners.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));
      }
    }
  }

  bool runCalibration(std::vector<std::vector<cv::Point2f> > imagePoints,
		      cv::Size imageSize, cv::Size boardSize,
		      float squareSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
		      std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
		      std::vector<float>& reprojErrs,
		      double& totalAvgErr)
  {
    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    fisheye_model::calcChessboardCorners(boardSize, squareSize, objectPoints[0]);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix,
					distCoeffs, rvecs, tvecs,
					//cv::fisheye::CALIB_CHECK_COND |
					//cv::fisheye::CALIB_FIX_K2 |
					//cv::fisheye::CALIB_FIX_K3 |
					//cv::fisheye::CALIB_FIX_K4 |
					cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
					//cv::fisheye::CALIB_USE_INTRINSIC_GUESS |
					cv::fisheye::CALIB_FIX_SKEW |
					0);


    printf("RMS error reported by fisheye::calibrate: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = fisheye_model::computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
  }

  void saveCameraParams(const std::string& filename,
			cv::Size imageSize, cv::Size boardSize,
			float squareSize,
			const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
			const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
			const std::vector<float> &reprojErrs,
			const std::vector<std::string> &good_image_paths,
			const std::vector<std::vector<cv::Point2f> >& imagePoints,
			double totalAvgErr)
  {
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime(buf, sizeof(buf)-1, "%c", t2);

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
      fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;

    if (!reprojErrs.empty() && !good_image_paths.empty()) {

      fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);
    }

    CV_Assert(reprojErrs.size() == good_image_paths.size());
    for (size_t i = 0; i < good_image_paths.size(); ++i) {
      if (reprojErrs[i] > 5) {
	fs << ("image_with_high_reproj_error_" + std::to_string(i)) << good_image_paths[i];
      }
    }

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

    if (!imagePoints.empty()) {

      cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);

      for( int i = 0; i < (int)imagePoints.size(); i++ ) {
        cv::Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
        cv::Mat imgpti(imagePoints[i]);
        imgpti.copyTo(r);
      }
      fs << "image_points" << imagePtMat;
    }
  }

  bool runAndSave(const std::string& outputFilename,
		  const std::vector<std::vector<cv::Point2f> >& imagePoints,
		  const std::vector<std::string> &good_image_indices,
		  cv::Size imageSize, cv::Size boardSize, float squareSize,
		  cv::Mat& cameraMatrix,
		  cv::Mat& distCoeffs, bool writeExtrinsics, bool writePoints)
  {
    CV_Assert(good_image_indices.size() == imagePoints.size());

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = fisheye_model::runCalibration(imagePoints, imageSize, boardSize, squareSize, cameraMatrix, distCoeffs,
					    rvecs, tvecs, reprojErrs, totalAvgErr);

    printf("%s. avg reprojection error = %.2f\n", ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);

    if (ok) {
      fisheye_model::saveCameraParams(outputFilename, imageSize,
				      boardSize, squareSize,
				      cameraMatrix, distCoeffs,
				      writeExtrinsics ? rvecs : std::vector<cv::Mat>(),
				      writeExtrinsics ? tvecs : std::vector<cv::Mat>(),
				      writeExtrinsics ? reprojErrs : std::vector<float>(),
				      writeExtrinsics ? good_image_indices : std::vector<std::string>(),
				      writePoints ? imagePoints : std::vector<std::vector<cv::Point2f> >(),
				      totalAvgErr );
    }

    return ok;
  }
};
