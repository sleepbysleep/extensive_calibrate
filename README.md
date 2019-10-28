# extensive_calibrate
Camera calibration c++ code based on OpenCV with generic, fisheye, and omnidirectional model

# Dependencies
<pre><code>  # apt install libopencv-dev libopencv-calib3d-dev libopencv-contrib-dev
  # apt install libboost-filesystem-dev</code></pre>

# Option
<pre><code>  -m (string): calibration model with "generic", "fisheye", and "omnidir"
  -w (integer): horizontal counts of chessboard crossing 
  -h (integer): vertical counts of chessboard crossing
  -sw (double): square width in chessboard
  -sh (double): square height in chessboard
  -a (double): aspect ratio of square in chessboard
  -o (string): filename for camera parameters including intrinsics, and extrinsics
  -fa (bool): cv::CALIB_FIX_ASPECT_RATIO for "generic" model
  -fz (bool): cv::CALIB_ZERO_TANGENT_DIST for "generic" model
  -fs (bool): cv::omnidir::CALIB_FIX_SKEW for "omnidir" model
  -fp (bool): cv::CALIB_FIX_PRINCIPAL_POINT or cv::omnidir::CALIB_FIX_CENTER for "generic" and "omnidir" mdoels
  -su: show the undistorted image
  (filename): at the end of commandline, xml file with a list of the images, created with imagelist_creator tool</code></pre>

# Usage
<pre><code>  $./imagelist_creator ./imagelist.xml ./test/*.png
    $./a.out -m=generic -w=7 -h=10 -a=1.0 -sw=80.0 -sh=80.0 -o=camera.xml -su -oe ./imagelist.xml</code></pre>
