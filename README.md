# 3DSMC_Project

This is the repository for the project work bundle adjustment project for the course 3D Scanning & Motion Capture for the WiSe 2023/24. Credits for this project do not belong to me alone, contact me for list of contributors. 

## Build
To build this project, create a build folder in the ```ba_project``` folder, then run ```cmake ..``` and ```make```. For that, make sure that you have installed OpenCV, Ceres, Glog, Eigen, FreeImage and PCL, and correctly linked in the ```CMakeLists.txt``` file. <br>
If you want to run it in a container, you can also use the Dockerfile in the root directory to build an image ```docker build -t <image_name> .```. Afterward you can run a container with the image ```docker run -it -v "<path to folder>:/usr/src/app" <container_name>``` and follow the steps from above to build and run the project. 


## Run
Run ```./ba_project --help``` to find out about all the options to run the project. For now, you can only load the FreiburgXYZ, the Replica and the FreiburgTeddy dataset. Make sure to provide the path to these datasets, and additionally specify which one you want to use through a command line argument.

## Example
The file ```combined_colored_cloud.py``` in the root directory shows the combined point cloud of a run with 'Standard' initialization and 'BA' for frame-to-frame pose estimation compared to the ground truth of the Replica dataset. 