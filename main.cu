#include "dataset_read.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


int main() {

	// use std::vector::data to access the pointer for cudaMalloc
	thrust::host_vector<short> trainImage;
	thrust::host_vector<short> testImage;

	// Use absolute path to your data folder here.
	string absPath = "C:/Users/vgudavar/Desktop/ECE_285_GPU_Prog/Project/class_labs/Src/ece285kmeans";
	ReadMNIST(absPath + "/data/train-images.idx3-ubyte", 60000, 784, trainImage);
	ReadMNIST(absPath + "/data/t10k-images.idx3-ubyte", 10000, 784, testImage);
	

	thrust::host_vector<short> trainLabels;
	thrust::host_vector<short> testLabels;
	ReadLabels(absPath + "/data/train-labels.idx1-ubyte", 60000, trainLabels);
	ReadLabels(absPath + "/data/t10k-labels.idx1-ubyte", 10000, testLabels);

	return 0;
}