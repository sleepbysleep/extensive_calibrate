#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

using namespace cv;
using namespace std;

const char * usage =
  " Example command line for calibration from a list of stored images:\n"
  "   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
  " where image_list.xml is the standard OpenCV XML/YAML\n"
  " use imagelist_creator to create the xml or yaml list\n"
  " file consisting of the list of strings, e.g.:\n"
  " \n"
  "<?xml version=\"1.0\"?>\n"
  "<opencv_storage>\n"
  "<images>\n"
  "view000.png\n"
  "view001.png\n"
  "<!-- view002.png -->\n"
  "view003.png\n"
  "view010.png\n"
  "one_extra_view.jpg\n"
  "</images>\n"
  "</opencv_storage>\n";

static void help() {
  printf( "This is a fisheye camera calibration tool.\n"
          "Usage: calibration\n"
          "     -w=<board_width>         # the number of inner corners per one of board dimension\n"
          "     -h=<board_height>        # the number of inner corners per another board dimension\n"
          "     -o=<out_camera_params>   # the output filename for intrinsic [and extrinsic] parameters\n"
          "     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
          "     [-op]                    # write detected feature points\n"
          "     [-oe]                    # write extrinsic parameters\n"
          "     [-su]                    # show undistorted images after calibration\n"
          "     [input_data]             # input data: text file with a list of the images of the board\n"
          "\n" );
  printf("\n%s",usage);
}

static double computeReprojectionErrors( vector<vector<Point3f>> const& objectPoints,
                                         vector<vector<Point2f>> const& imagePoints,
                                         vector<Mat> const& rvecs,
                                         vector<Mat> const& tvecs,
                                         Mat const& cameraMatrix,
                                         Mat const& distCoeffs,
                                         vector<float> & perViewErrors )
{
  vector<Point2f> imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for( i = 0; i < (int)objectPoints.size(); i++ ) {

    cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);

    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
    int n = (int)objectPoints[i].size();
    perViewErrors[i] = (float)std::sqrt(err*err/n);
    totalErr += err*err;
    totalPoints += n;
  }

  return std::sqrt(totalErr/totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
  corners.resize(0);

  for (int i = 0; i < boardSize.height; ++i) {
    for (int j = 0; j < boardSize.width; ++j) {

      corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
    }
  }
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
                           Size imageSize, Size boardSize,
                           float squareSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<Mat>& rvecs, vector<Mat>& tvecs,
                           vector<float>& reprojErrs,
                           double& totalAvgErr)
{
  vector<vector<Point3f> > objectPoints(1);
  calcChessboardCorners(boardSize, squareSize, objectPoints[0]);

  objectPoints.resize(imagePoints.size(),objectPoints[0]);

  double rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix,
                          distCoeffs, rvecs, tvecs,
//			cv::fisheye::CALIB_CHECK_COND |
//			cv::fisheye::CALIB_FIX_K2 |
//			cv::fisheye::CALIB_FIX_K3 |
//			cv::fisheye::CALIB_FIX_K4 |
			cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
			cv::fisheye::CALIB_FIX_SKEW
			);


  printf("RMS error reported by fisheye::calibrate: %g\n", rms);

  bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

  totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

  return ok;
}

static void saveCameraParams(const string& filename,
                             Size imageSize, Size boardSize,
                             float squareSize,
                             const Mat& cameraMatrix, const Mat& distCoeffs,
                             const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                             const vector<float> &reprojErrs,
                             const vector<string> &good_image_paths,
                             const vector<vector<Point2f> >& imagePoints,
                             double totalAvgErr)
{
  FileStorage fs( filename, FileStorage::WRITE );

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

    fs << "per_view_reprojection_errors" << Mat(reprojErrs);
  }

  CV_Assert(reprojErrs.size() == good_image_paths.size());
  for (int i = 0; i < good_image_paths.size(); ++i) {
    if (reprojErrs[i] > 5) {
      fs << ("image_with_high_reproj_error_" + to_string(i)) << good_image_paths[i];
    }
  }

  if( !rvecs.empty() && !tvecs.empty() ) {

    CV_Assert(rvecs[0].type() == tvecs[0].type());
    Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
    for( int i = 0; i < (int)rvecs.size(); i++ )
    {
        Mat r = bigmat(Range(i, i+1), Range(0,3));
        Mat t = bigmat(Range(i, i+1), Range(3,6));

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

    Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);

    for( int i = 0; i < (int)imagePoints.size(); i++ ) {
        Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
        Mat imgpti(imagePoints[i]);
        imgpti.copyTo(r);
    }
    fs << "image_points" << imagePtMat;
  }
}

static bool readStringList(const string& filename, vector<string>& l)
{
  l.resize(0);
  FileStorage fs(filename, FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  FileNode n = fs.getFirstTopLevelNode();
  if( n.type() != FileNode::SEQ )
    return false;
  FileNodeIterator it = n.begin(), it_end = n.end();
  for( ; it != it_end; ++it )
    l.push_back((string)*it);
  return true;
}

static bool runAndSave(const string& outputFilename,
                       const vector<vector<Point2f> >& imagePoints,
                       const vector<string> &good_image_indices,
                       Size imageSize, Size boardSize, float squareSize,
                       Mat& cameraMatrix,
                       Mat& distCoeffs, bool writeExtrinsics, bool writePoints)
{
  CV_Assert(good_image_indices.size() == imagePoints.size());

  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;
  double totalAvgErr = 0;

  bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize, cameraMatrix, distCoeffs,
                 rvecs, tvecs, reprojErrs, totalAvgErr);

  printf("%s. avg reprojection error = %.2f\n", ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);

  if (ok) {
      saveCameraParams( outputFilename, imageSize,
                       boardSize, squareSize,
                       cameraMatrix, distCoeffs,
                       writeExtrinsics ? rvecs : vector<Mat>(),
                       writeExtrinsics ? tvecs : vector<Mat>(),
                       writeExtrinsics ? reprojErrs : vector<float>(),
                       writeExtrinsics ? good_image_indices : vector<string>(),
                       writePoints ? imagePoints : vector<vector<Point2f> >(),
                       totalAvgErr );
  }

  return ok;
}

int main(int argc, char** argv) { try {

  CommandLineParser parser(argc, argv,
                           "{help ||}{w||}{h||}{s|1|}{o|out_camera_data.yml|}"
                           "{op||}{oe||}{su||}"
                           "{@input_data|0|}");

  if (parser.has("help")) {
    help();
    return 0;
  }

  Size board_size(parser.get<int>("w"), parser.get<int>("h"));
  if (board_size.width <= 0) {
    return fprintf(stderr, "Invalid board width\n"), -1;
  }
  if (board_size.height <= 0) {
    return fprintf(stderr, "Invalid board height\n"), -1;
  }

  float square_size = parser.get<float>("s");
  if (square_size <= 0) {
    return fprintf(stderr, "Invalid board square width\n" ), -1;
  }

  bool write_points = parser.has("op");

  bool write_extrinsics = parser.has("oe");

  bool show_undistorted = parser.has("su");

  string output_filename = parser.get<string>("o");

  string input_filename = parser.get<string>("@input_data");

  if (input_filename.empty()) {
    throw runtime_error("Error in input filename");
  }

  if (!parser.check()) {
    help();
    parser.printErrors();
    return -1;
  }

  //

  vector<string> image_list;
  if (!readStringList(input_filename, image_list)) {
    throw std::runtime_error("Error reading image files list");
  }
  if (image_list.empty()) {
    return fprintf(stderr, "Could not initialize capture\n"), -2;
  }

  int nframes = (int)image_list.size();

  CV_Assert(nframes > 3);

  Size image_size = imread(image_list[0], 1).size();

  namedWindow("Image View", 1);

  vector<vector<Point2f>> image_points;
  vector<string> images_with_found_corners;

  for (int i = 0; i < nframes; ++i) {

    Mat view = imread(image_list[i], 1);

    if(view.empty()) {
      cerr << "Invalid image path: " + image_list[i] << endl;
    }

    CV_Assert(image_size == view.size());

//    std::cout << "view.channels(): " << view.channels() << '\n';

#if 1
    std::vector<Mat> channels(3);
    cv::Mat view_gray;

    cv::split(view, channels);
    cv::demosaicing(channels[0], view_gray, COLOR_BayerRG2GRAY);
#else
    Mat view_gray;
    cvtColor(view, view_gray, COLOR_BGR2GRAY);
#endif

//    view = view_color;
//    imshow("Image View", view_gray);
//    waitKey(1000);

    vector<Point2f> pointbuf;
    bool found = findChessboardCorners(view, board_size, pointbuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
      cornerSubPix(view_gray, pointbuf, Size(11, 11), Size(-1,-1), TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1));
    }

    if (found) {
      image_points.push_back(pointbuf);
      drawChessboardCorners(view, board_size, Mat(pointbuf), found);
      images_with_found_corners.push_back(image_list[i]);
    }

    string msg = "100/100";
    int baseLine = 0;
    Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
    Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);
    msg = format("%d/%d", (int)image_points.size(), nframes);
    putText(view, msg, textOrigin, 1, 1, Scalar(0,0,255));

    imshow("Image View", view);
    char key = (char)waitKey(50);

    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }

  Mat camera_matrix, dist_coeffs;

  runAndSave(output_filename, image_points,
             images_with_found_corners, image_size,
             board_size, square_size,
             camera_matrix, dist_coeffs,
             write_extrinsics, write_points);

  cout << "Camera matrix : \n";
  cv::print(camera_matrix);
  cout << "\n Distortion coeffs: \n";
  cv::print(dist_coeffs);
  cout << endl;


  if (show_undistorted) {

    CV_Assert(!camera_matrix.empty());

    Mat view, und_view, newCameraMatrix, mapx, mapy;

    fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_coeffs, image_size, noArray(), newCameraMatrix, 0.5, image_size, 2);

    fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, noArray(), newCameraMatrix, image_size, CV_32FC1, mapx, mapy);

    for (int i = 0; i < (int)image_list.size(); ++i) {

      view = imread(image_list[i], 1);

      if (view.empty()) {
        continue;
      }

      //fisheye::undistortImage(view, und_view, cameraMatrix, distCoeffs, newCameraMatrix);
      cv::remap(view, und_view, mapx, mapy, cv::INTER_CUBIC);

      cv::imshow("Image View", und_view);

      char c = (char)waitKey(50);
      if (c == 27 || c == 'q' || c == 'Q') {
        break;
      }
    }
  }

  return 0;

} catch (exception &e) { cerr << e.what() << endl; }}
