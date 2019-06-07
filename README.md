# KMeans Clustering in CUDA with OpenGL Visualization

This is our final project for *GPU Programming* course at *UC San Diego*.
The objective is to implement KMeans algorithm in CUDA.

## System Requirements
* Windows 7/8/8.1/10
* NVIDIA GPU with at least 5.2 CUDA compute capability 
* CUDA 9.1+

## Instructions

To compile and run the solution please use CMake. 
1. Choose the root directory of this repository as a Source folder, and choose any folder for the Build folder
2. Open classlab.sln (with Visual Studio), which is located in the Build folder you have chosen
3. Choose a Startup project of interest by using the right-click menu
4. Run the **release** version without debug

There are several projects of interest in this solution:
* **ece285kmeans-Images** holds the version of the program that performs PCA on MNIST dataset and then clusters 3D embedding with k-means algorithm
* **ece285kmeans-RandomPoints** holds the version of the program that performs k-means clustering on 3 randomly generated clusters
* **ece285kmeans-TriangleInequality** holds the implementation of the triangle inequality optimization, which forces the program to terminate if it is guaranteed that all points left for checking are further away than the last visited centroid.

## Visualization Controls
* Hold left-mouse key to rotate the clusters
* Use mouse wheel to zoom in and out
