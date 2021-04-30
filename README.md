# extensive_calibrate
Camera calibration c++ code based on OpenCV with generic, fisheye, and omnidirectional model. In addition, python code is implemented being totally compatible to c++ version.

# Dependencies
For C++ source code,
<pre><code>  # apt install libopencv-dev libopencv-calib3d-dev libopencv-contrib-dev
  # apt install libboost-filesystem-dev</code></pre>

For Python source code,
<pre><code>  # apt install python3-opencv</code></pre>

# Option
<pre><code>  -m (string): calibration model with "generic", "fisheye", and "omnidir"
  -bw (int): the number of horizontal crossings in chessboard 
  -bh (int): the number of vertical crossings in chessboard
  -sw (float): the width[mm] of unit squqre in chessboard
  -sh (float): the height[mm] of unit square in chessboard
  -o (string): xml filename including camera parameters like as intrinsics, and extrinsics
  -i (string): xml filename including image filenames for calibration and validation
  -os (float): the scale of output image that should be undistorted
  -ow (int): the width[pixels] of output image that should be undistorted
  -oh (int): the height[pixels] of output image that should be undistorted
  -ox (float): the x[pixels] translation of output image
  -oy (float): the y[pixels] translation of output image
  -calibrate (bool): the switch for calibration
  -validate (bool): the switch for validation</code></pre>

# Compile
<pre><code>  $ mkdir build
  $ cd build
  $ cmake ..
  $ cd ..</code></pre>

# Usage
For generic model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./image_list.xml ./images/*.jpg
  $./build/extensive_calibrate -m=generic -bw=9 -bh=6 -sw=50.0 -sh=50.0 -o=generic_model.xml -i=image_list.xml -os=0.0 -ow=0 -oh=0 -ox=0.0 -oy=0.0 -calibrate=true -validate=true </code></pre>

For fisheye model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./fisheye_image_list.xml ./omnidir_images/*.jpg
  $./build/extensive_calibrate -m=fisheye -bw=9 -bh=6 -sw=50.0 -sh=50.0 -o=fisheye_model.xml -i=fisheye_image_list.xml -os=0.0 -ow=0 -oh=0 -ox=0.0 -oy=0.0 -calibrate=true -validate=true </code></pre>

For omnidir model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./omnidir_image_list.xml ./omnidir_images/*.jpg
  $./build/extensive_calibrate -m=omnidir -bw=9 -bh=6 -sw=50.0 -sh=50.0 -o=omnidir_model.xml -i=omnidir_image_list.xml -os=0.0 -ow=0 -oh=0 -ox=0.0 -oy=0.0 -calibrate=true -validate=true </code></pre>

## In the case of Python version
For generic model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./image_list.xml ./images/*.jpg
  $python3 ./python_code/extensive_calibrate.py -m= generic -bw= 9 -bh= 6 -sw= 50.0 -sh= 50.0 -o= generic_model.xml -i= image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0.0 -oy= 0.0 -calibrate= true -validate= true </code></pre>

For fisheye model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./fisheye_image_list.xml ./omnidir_images/*.jpg
  $python3 ./python_code/extensive_calibrate.py -m= fisheye -bw= 9 -bh= 6 -sw= 50.0 -sh= 50.0 -o= fisheye_model.xml -i= fisheye_image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0.0 -oy= 0.0 -calibrate= true -validate= true </code></pre>

For omnidir model calibration, and then validation,
<pre><code>  $./build/imagelist_creator ./omnidir_image_list.xml ./omnidir_images/*.jpg
  $python3 ./python_code/extensive_calibrate.py -m= omnidir -bw= 9 -bh= 6 -sw= 50.0 -sh= 50.0 -o= omnidir_model.xml -i= omnidir_image_list.xml -os= 0.0 -ow= 0 -oh= 0 -ox= 0.0 -oy= 0.0 -calibrate= true -validate= true </code></pre>
