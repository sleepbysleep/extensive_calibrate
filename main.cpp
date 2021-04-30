#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include "base_model.hpp"
#include "generic_model.hpp"
#include "fisheye_model.hpp"
#include "omnidir_model.hpp"

int main(int argc, char** argv)
{
  cv::setUseOptimized(true);
  
  cv::CommandLineParser parser(
    argc,
    argv,
    "{m|generic|calibration model}"
    "{bw|9|board width}"
    "{bh|6|board height}"
    "{sw|50|square width}"
    "{sh|50|square height}"
    "{o|params.xml|output file}"
    "{i|image_list.xml|input file - xml file with a list of the images, created with cpp-example-imagelist_creator tool}"
    "{os|0.0|undistort scale}"
    "{ow|0|undistort width}"
    "{oh|0|undistort height}"
    "{ox|0.0|undistort xoffset}"
    "{oy|0.0|undistort yoffset}"
    "{validate|true|perform validation}"
    "{calibrate|true|perform calibration}"
    "{help||show help}"
  );

  parser.about(
    "This is a test for generic, fisheye, and ominidirectional camera calibration.\n"
    "Example command line:\n"
    "    ./extensive_calibrate -m=generic -bw=6 -bh=9 -sw=50 -sh=50 -o=params.xml -i=imagelist.xml -os=0.0 -ow=0 -oh=0 -ox=0 -oy=0 -calibate=true -validate=true\n"
  );


  std::string calibrateModel = parser.get<std::string>("m");
  std::string calibrateImagesXML = parser.get<std::string>("i");
  std::string calibrateModelXML = parser.get<std::string>("o");
  std::string distortImagesXML = parser.get<std::string>("i");

  cv::Size boardSize(parser.get<int>("bw"), parser.get<int>("bh"));
  cv::Size squareSize(parser.get<int>("sw"), parser.get<int>("sh"));

  cv::Size undistortSize(parser.get<int>("ow"), parser.get<int>("oh"));
  double undistortScale = parser.get<double>("os");
  cv::Point2f undistortOffset(parser.get<double>("ox"), parser.get<double>("oy"));

  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }

  std::cout << "Calibration Model: " << calibrateModel << "\n";
  std::cout << "calibrateImagesXML: " << calibrateImagesXML << "\n";
  std::cout << "calibrateModelXML: " << calibrateModelXML << "\n";
  std::cout << "distortImagesXML: " << distortImagesXML << "\n";
  std::cout << "boardSize: " << boardSize << "\n";
  std::cout << "squareSize: " << squareSize << "\n";
  std::cout << "undistortSize: " << undistortSize << "\n";
  std::cout << "undistortOffset: " << undistortOffset << "\n";
  std::cout << "undistortScale: " << undistortScale << "\n";

  if (calibrateModel == "generic") {
    GenericModel model;
    if (parser.get<bool>("calibrate")) {
      model.calibrateFromXML(
	calibrateImagesXML,
	boardSize,
	squareSize,
	cv::CALIB_RATIONAL_MODEL,
	calibrateModelXML,
	true
      );
    }

    if (parser.get<bool>("validate")) {
      std::vector<std::string> image_list = model.readFileListFromXML(calibrateImagesXML);

      std::vector<cv::Mat> results =  model.validateFromFiles(
	image_list,
	0,
	undistortSize,
	undistortScale,
	undistortOffset,
	calibrateModelXML,
	true
      );
    }
  } else if (calibrateModel == "fisheye") {
    FisheyeModel model;

    if (parser.get<bool>("calibrate")) {
      model.calibrateFromXML(
	calibrateImagesXML,
	boardSize,
	squareSize,
	cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_FIX_SKEW,
	calibrateModelXML,
	true
      );
    }

    if (parser.get<bool>("validate")) {
      std::vector<std::string> image_list = model.readFileListFromXML(calibrateImagesXML);
  
      std::vector<cv::Mat> results =  model.validateFromFiles(
	image_list,
	0,
	undistortSize,
	undistortScale,
	undistortOffset,
	calibrateModelXML,
	true
      );
    }
  } else if (calibrateModel == "omnidir") {
    OmnidirModel model;

    if (parser.get<bool>("calibrate")) {
      model.calibrateFromXML(
	calibrateImagesXML,
	boardSize,
	squareSize,
	cv::omnidir::CALIB_USE_GUESS + cv::omnidir::CALIB_FIX_SKEW + cv::omnidir::CALIB_FIX_CENTER,
	calibrateModelXML,
	true
      );
    }

    if (parser.get<bool>("validate")) {
      std::vector<std::string> image_list = model.readFileListFromXML(calibrateImagesXML);
  
      std::vector<cv::Mat> results =  model.validateFromFiles(
	image_list,
	cv::omnidir::RECTIFY_CYLINDRICAL,
	undistortSize,
	undistortScale,
	undistortOffset,
	calibrateModelXML,
	true
      );
    }
  } else {
    assert(0);
  }

  return 0;
}
