
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

  bool detecChessboardCorners(const std::vector<std::string>& list, std::vector<std::string>& list_detected,
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

  void saveCameraParams(const std::string & filename, int flags, const cv::Mat& cameraMatrix,
			const cv::Mat& distCoeffs, const double xi, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs,
			std::vector<std::string> detec_list, const cv::Mat& idx, const double rms, const std::vector<cv::Mat>& imagePoints)
  {
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if ( !rvecs.empty())
      fs << "nFrames" << (int)rvecs.size();

    if ( flags != 0) {
      sprintf( buf, "flags: %s%s%s%s%s%s%s%s%s",
	       flags & cv::omnidir::CALIB_USE_GUESS ? "+use_intrinsic_guess" : "",
	       flags & cv::omnidir::CALIB_FIX_SKEW ? "+fix_skew" : "",
	       flags & cv::omnidir::CALIB_FIX_K1 ? "+fix_k1" : "",
	       flags & cv::omnidir::CALIB_FIX_K2 ? "+fix_k2" : "",
	       flags & cv::omnidir::CALIB_FIX_P1 ? "+fix_p1" : "",
	       flags & cv::omnidir::CALIB_FIX_P2 ? "+fix_p2" : "",
	       flags & cv::omnidir::CALIB_FIX_XI ? "+fix_xi" : "",
	       flags & cv::omnidir::CALIB_FIX_GAMMA ? "+fix_gamma" : "",
	       flags & cv::omnidir::CALIB_FIX_CENTER ? "+fix_center" : "");
      //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "xi" << xi;

    //cvWriteComment( *fs, "names of images that are acturally used in calibration", 0 );
    fs << "used_imgs" << "[";
    for (int i = 0;  i < (int)idx.total(); ++i)
      {
        fs << detec_list[(int)idx.at<int>(i)];
      }
    fs << "]";

    if ( !rvecs.empty() && !tvecs.empty() ) {
      cv::Mat rvec_tvec((int)rvecs.size(), 6, CV_64F);
      for (int i = 0; i < (int)rvecs.size(); ++i)
	{
	  cv::Mat(rvecs[i]).reshape(1, 1).copyTo(rvec_tvec(cv::Rect(0, i, 3, 1)));
	  cv::Mat(tvecs[i]).reshape(1, 1).copyTo(rvec_tvec(cv::Rect(3, i, 3, 1)));
	}
      //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
      fs << "extrinsic_parameters" << rvec_tvec;
    }

    fs << "rms" << rms;

    if ( !imagePoints.empty() ) {
      cv::Mat imageMat((int)imagePoints.size(), (int)imagePoints[0].total(), CV_64FC2);
      for (int i = 0; i < (int)imagePoints.size(); ++i)
	{
	  cv::Mat r = imageMat.row(i).reshape(2, imageMat.cols);
	  cv::Mat imagei(imagePoints[i]);
	  imagei.copyTo(r);
	}
      fs << "image_points" << imageMat;
    }
  }  
};
