#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include "generic_model.hpp"
#include "fisheye_model.hpp"
#include "omnidir_model.hpp"

static std::string calibrateModel;
static std::string inputFilename;
static std::string outputFilename;

static bool readStringList(const std::string& filename, std::vector<std::string>& l)
{
  l.resize(0);
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  cv::FileNode n = fs.getFirstTopLevelNode();
  if( n.type() != cv::FileNode::SEQ )
    return false;
  cv::FileNodeIterator it = n.begin(), it_end = n.end();
  for( ; it != it_end; ++it )
    l.push_back((std::string)*it);
  return true;
}

int main(int argc, char** argv)
{
  cv::CommandLineParser parser(argc, argv,
			       "{m|generic|calibration model}"
			       "{w||board width}"
			       "{h||board height}"
			       "{sw|1.0|square width}"
			       "{sh|1.0|square height}"
			       "{a|1.0|aspect ratio}"
			       "{o|out_camera_params.xml|output file}"
			       "{fa|false|fix aspect ratio}"
			       "{fz|false|fix zero tangent distance}"
			       "{fs|false|fix skew}"
			       "{fp|false|fix principal point at the center}"
			       "{su|false|show the undistorted}"
			       "{@input||input file - xml file with a list of the images, created with cpp-example-imagelist_creator tool}"
			       "{help||show help}");

  parser.about("This is a test for generic, fisheye, and ominidirectional camera calibration.\n"
	       "Example command line:\n"
	       "    ./calibrate -m=generic -w=6 -h=9 -a=1.0 -sw=80 -sh=80 -o=params.xml imagelist.xml\n");

  cv::Size board_size(parser.get<int>("w"), parser.get<int>("h"));
  cv::Size2d square_size(parser.get<double>("sw"), parser.get<double>("sh"));
  double aspect_ratio = parser.get<double>("a");
  
  calibrateModel = parser.get<std::string>("m");
  std::cout << calibrateModel << " calibration model\n";  

  int flags = 0;
  bool fixed_aspect = parser.get<bool>("fa");
  bool fixed_zero_tangent_dist = parser.get<bool>("fz");
  bool fixed_skew = parser.get<bool>("fs");
  bool fixed_principal = parser.get<bool>("fp");
  if (calibrateModel == "generic") {
    if (fixed_aspect) flags |= cv::CALIB_FIX_ASPECT_RATIO;
    if (fixed_zero_tangent_dist) flags |= cv::CALIB_ZERO_TANGENT_DIST;
    if (fixed_principal) flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
  } else if (calibrateModel == "omnidir") {
    if (fixed_skew) flags |= cv::omnidir::CALIB_FIX_SKEW;
    if (fixed_principal) flags |= cv::omnidir::CALIB_FIX_CENTER;
    //} else if (calibrateModel == "fisheye") {
  } else if (calibrateModel == "fisheye") {
  } else {
    assert(0);
  }

  std::string outputFilename = parser.get<std::string>("o");
  std::string inputFilename = parser.get<std::string>("@input");
  //std::string inputFilename = parser.get<std::string>("0");

  if (!parser.check()) { parser.printErrors(); return -1; }


  std::vector<std::string> image_list;
  if (!readStringList(inputFilename, image_list)) {
    std::cerr << "Can not read image_list" << std::endl;
    return -1;
  }

  if (image_list.empty()) {
    std::cerr << "Could not initialize capture" << std::endl;
    return -2;
  }

  assert(image_list.size() > 3);
  
  cv::Size image_size = cv::imread(image_list[0], 1).size();
  cv::namedWindow("Image View", 1);

  /////////////////////////////// Detection of points ////////////////////////  
  std::vector<std::vector<cv::Point2f>> image_points;
  std::vector<std::string> images_with_found_corners;

  for (size_t i = 0; i < image_list.size(); ++i) {
    cv::Mat view = cv::imread(image_list[i], 1);
    if (view.empty()) {
      std::cerr << "Invalid image path: " + image_list[i] << std::endl;
      break;
    }
    assert(image_size == view.size());
    
#if 1
    cv::Mat view_color;

    std::vector<cv::Mat> channels(3);
    cv::split(view, channels);
    cv::demosaicing(channels[0], view_color, cv::COLOR_BayerRG2BGR);
#else
    cv::Mat view_color = view;
#endif
    
    cv::Mat view_gray;
    cv::cvtColor(view_color, view_gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> pointbuf;
    bool found = cv::findChessboardCorners(view_color,
					   board_size,
					   pointbuf,
					   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
      cv::cornerSubPix(view_gray,
		       pointbuf,
		       cv::Size(11, 11),
		       cv::Size(-1,-1),
		       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
      
      image_points.push_back(pointbuf);
      cv::drawChessboardCorners(view_color, board_size, cv::Mat(pointbuf), found);
      images_with_found_corners.push_back(image_list[i]);
    } else {
      std::cout << "findChessboardCorners() failed at " << image_list[i] << ", " << i << std::endl;
    }

    std::string msg = "100/100";
    int base_line = 0;
    cv::Size text_size = cv::getTextSize(msg, 1, 1, 1, &base_line);
    cv::Point text_origin(view.cols - 2*text_size.width - 10, view.rows - 2*base_line - 10);
    msg = cv::format("%d/%d", (int)image_points.size(), image_list.size());
    cv::putText(view_color, msg, text_origin, 1, 1, cv::Scalar(0,0,255));

    cv::imshow("Image View", view_color);
    char key = (char)cv::waitKey(50);

    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }
  std::cout << "findChessboardCorners() is done!" << std::endl;

  /////////////////////////////// Calibration /////////////////////////////////
  cv::Mat camera_matrix, dist_coeffs, xi;
  if (calibrateModel == "generic") {
    generic_model::runAndSave(outputFilename, image_points,
			      image_size, board_size, generic_model::CHESSBOARD, square_size.width,
			      aspect_ratio, flags,
			      camera_matrix, dist_coeffs,
			      true, true);
  } else if (calibrateModel == "fisheye") {
    fisheye_model::runAndSave(outputFilename, image_points,
			      images_with_found_corners, image_size,
			      board_size, square_size.width,
			      camera_matrix, dist_coeffs,
			      true, true);
  } else if (calibrateModel == "omnidir") {
    std::vector<cv::Mat> imagePoints;
    for (int i = 0; i < image_points.size(); ++i) {
      cv::Mat points = cv::Mat(image_points[i]);
      if (points.type() != CV_64FC2)
	points.convertTo(points, CV_64FC2);
      imagePoints.push_back(points);
    }
    
    // calculate object coordinates
    std::vector<cv::Mat> objectPoints;
    cv::Mat object;
    omnidir_model::calcChessboardCorners(board_size, square_size, object);
    for(int i = 0; i < /*(int)detec_list.size()*/(int)images_with_found_corners.size(); ++i)
      objectPoints.push_back(object);
    
    // run calibration, some images are discarded in calibration process because they are failed
    // in initialization. Retained image indexes are in idx variable.
    cv::Mat idx;
    std::vector<cv::Vec3d> rvecs, tvecs;
    double _xi, rms;
    cv::TermCriteria criteria(3, 200, 1e-8);
    rms = cv::omnidir::calibrate(objectPoints, imagePoints, image_size,
				 camera_matrix, xi, dist_coeffs, rvecs, tvecs, flags, criteria, idx);
    _xi = xi.at<double>(0);
    std::cout << "Saving camera params to " << outputFilename << std::endl;
    omnidir_model::saveCameraParams(outputFilename, flags, camera_matrix, dist_coeffs, _xi,
				    rvecs, tvecs, /*detec_list*/images_with_found_corners, idx, rms, imagePoints);
  } else {
    assert(0);
  }

  std::cout << "Camera matrix : \n";
  cv::print(camera_matrix);
  std::cout << "\nDistortion coeffs : \n";
  cv::print(dist_coeffs);
  std::cout << '\n';

  ///////////////////////////////// Show results /////////////////////////////
  cv::Mat x_map, y_map;
  cv::Mat new_camera_matrix;
  cv::Rect roi = cv::Rect(0, 0, 0, 0);

  if (calibrateModel == "generic") {
    
    new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 0.0, image_size, &roi);
    
    std::cout << "New camera matrix: \n";
    cv::print(new_camera_matrix);
    std::cout << "\nROI:\n";
    std::cout << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << "\n";
    
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::noArray()/*cv::Mat()*/,
				new_camera_matrix,
				image_size, CV_32FC1, x_map, y_map);
  } else if (calibrateModel == "fisheye") {
    
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix,
							    dist_coeffs,
							    image_size,
 							    cv::noArray(),
							    new_camera_matrix,
							    0.5, image_size, 2);
    
    cv::fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::noArray(),
					 new_camera_matrix, image_size, CV_32FC1, x_map, y_map);
  } else if (calibrateModel == "omnidir") {
    //Testing
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    
    cv::Mat New_camera_mat(3,3,CV_32F);
    //New_camera_mat tries to get entire FOV,but it is losing some information at edges
    New_camera_mat.at<float>(0, 0) = 800;
    New_camera_mat.at<float>(0, 1) = 0; 
    New_camera_mat.at<float>(0, 2) = image_size.width/2;

    New_camera_mat.at<float>(1, 0) = 0;
    New_camera_mat.at<float>(1, 1) = 800;
    New_camera_mat.at<float>(1, 2) = image_size.height/2;

    New_camera_mat.at<float>(2, 0) = 0;
    New_camera_mat.at<float>(2, 1) = 0; 
    New_camera_mat.at<float>(2, 2) = 1;
    
    cv::omnidir::initUndistortRectifyMap(camera_matrix, dist_coeffs, xi,
					 R,
					 New_camera_mat,
					 image_size, CV_32FC1, x_map, y_map,
					 //cv::omnidir::RECTIFY_LONGLATI |
					 cv::omnidir::RECTIFY_PERSPECTIVE |
					 //cv::omnidir::RECTIFY_CYLINDRICAL |
					 0);
  }
  
  std::cout << "Calibration is done!\n";
  if (parser.get<bool>("su")) {
    for (int i = 0; i < (int)image_list.size(); ++i) {
      cv::Mat view = cv::imread(image_list[i], 1);
      if (view.empty()) {
	std::cerr << "Invalid image path: " + image_list[i] << std::endl;
	break;
      }
      assert(image_size == view.size());
    
#if 1
      cv::Mat view_color;

      std::vector<cv::Mat> channels(3);
      cv::split(view, channels);
      cv::demosaicing(channels[0], view_color, cv::COLOR_BayerRG2BGR);
#else
      cv::Mat view_color = view;
#endif
    
      cv::Mat view_undistort;
      cv::remap(view_color, view_undistort, x_map, y_map, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

      //cv::Mat view_roi;
      //if (roi.area() > 0) view_roi = view_undistort(roi);

      cv::imshow("Original View", view_color);
    
      std::string msg = "100/100";
      int base_line = 0;
      cv::Size text_size = cv::getTextSize(msg, 1, 1, 1, &base_line);
      cv::Point text_origin(view_undistort.cols - 2*text_size.width - 10, view_undistort.rows - 2*base_line - 10);
      msg = cv::format("%d/%d", i, image_list.size());
      cv::putText(view_undistort, msg, text_origin, 1, 1, cv::Scalar(0,0,255));
      cv::imshow("Image View", view_undistort);

      //if (!view_roi.empty()) cv::imshow("ROI View", view_roi);
    
      char c = (char)cv::waitKey();
      if (c == 27 || c == 'q' || c == 'Q') break;
    }
  }
  
  return 0;
}
