# extensive_calibrate
Camera calibration c++ code based on OpenCV with generic, fisheye, and omnidirectional model

# Dependencies
<pre><code>
	# apt install libopencv-dev libopencv-calib3d-dev libopencv-contrib-dev 
	# apt install libboost-filesystem-dev
</code></pre>

# Option
<pre><code>  cv::CommandLineParser parser(argc, argv,
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
			       "{@input||input file - xml file with a list of the images, created with imagelist_creator tool}"
			       "{help||show help}"</code></pre>

# Usage
<pre><code>
	$./imagelist_creator ./imagelist.xml ./test/*.png
	$./a.out -m=generic -w=7 -h=10 -a=1.0 -sw=80.0 -sh=80.0 -o=camera.xml -su -oe ./imagelist.xml
</code></pre>