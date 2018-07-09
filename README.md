# KMeans Clustering in CUDA

The objective is to implement KMeans algorithm in CUDA.

There are several branches of interest in this repository:
* **master** holds the version of the program that performs PCA on MNIST dataset and then clusters 3D embedding with k-means algorithm
* **randompointsnew** holds the version of the program that perform k-means clustering on 3 randomly generated clusters
* **implemented-paper-triangle-inequality...** holds the implementation of the triangle inequality optimization, which forces to terminate if it is guranteed that all other points of interestc are far away

To compile and run the solution please use CMake. 
1. Choose the root directory of this repository as a Source folder, and choose any folder for the Build folder
2. Open classlab.sln (with Visual Studio), which is located in the Build folder you chosen
3. Choose ece285kmeans as a Startup project by using the right-click menu
4. Build and run the release version without debug
