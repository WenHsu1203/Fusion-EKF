# Introduction
1. Implementing a kalman filter in C++ to estimate the position and velocity of a moving object with noisy lidar and radar measurements. 
2. Obtaining _RMSE_ values that are lower than the tolerance requirement.

**What I have: Data from 2 Noisy Sensors**
* A LIDAR sensor measures the position in Cartesian coordinate `(x,y)`
* A RADAR sensor measures the position and veloctiy in polar coordinate `(rho, phi, d_rho)`

**What my GOALs are: Estimate the Position, Velocity and Direction**
* Assumption: CV(Constant Velocity) dynamic model
* LIDAR: `Linear` -> Kalman Filter || RADAR: `Non-Linear` -> Extended Kalman Filter
* The State vector of the system : `(px, py, vx, vy)`, so transform the polar coordinate to Cartesian if the data from RADAR
# Result

In the demo video, LIDAR measurements are `red circles`, RADAR measurements are `blue circles` with an arrow pointing in the direction of the observed angle, and estimation markers are `green triangles`, and px, py, vx, and vy _RMSE_ values are from Kalman Filter and those are within  __[0.11, 0.11, 0.52, 0.52]__.

[![video](http://img.youtube.com/vi/oUrYJXa3_FE/0.jpg)](http://www.youtube.com/watch?v=oUrYJXa3_FE "Extended Kalman Filter")
# Download
The Simulator can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases).

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Important Dependencies

* cmake >= 3.5
* All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* Linux: make is installed by default on most Linux distros
* Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
* Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
* Linux: gcc / g++ is installed by default on most Linux distros
* Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
* Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions
1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
* On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

