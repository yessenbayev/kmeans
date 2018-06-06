#include "dataset_read.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>


thrust::host_vector<float> getData(thrust::host_vector<float> trainImages, int idx, int size) {
	thrust::host_vector<float> tempVec;
	int start = idx*size;
	for (int i = start; i < start + size; i++) {
		tempVec.push_back(trainImages[start]);
	}
	return tempVec;
}

void initializeMeans(const thrust::host_vector<float> &trainImages, thrust::device_vector<float> &meansGPU, int trainSize, int k, int dim) {
	//initialize kmeans from the training set
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<> distr(0, trainSize - 1); // define the range
	set<int> generated;

	for (int i = 0; i < k;) {
		int index = distr(eng);
		if (generated.find(index) != generated.end()) {
			continue;
		}
		generated.insert(index);
		printf("Random Index Generated is : %d \n", index);
		thrust::host_vector<float> tempVec = getData(trainImages, index, dim);
		thrust::copy(tempVec.begin(), tempVec.end(), meansGPU.begin() + i*dim);
		i++;
	}
}


int main() {

	int trainSize = 60000;
	int testSize = 10000;
	int n_rows = 28;
	int n_cols = 28;
	int dim = n_rows*n_cols;
	int k = 10; // Number of Means to be used for clustering

	// use std::vector::data to access the pointer for cudaMalloc
	thrust::host_vector<float> trainImages;
	thrust::host_vector<float> testImages;

	// Use absolute path to your data folder here.
	string absPath = "C:/Users/vgudavar/Desktop/ECE_285_GPU_Prog/Project/class_labs/Src/ece285kmeans";
	ReadMNIST(absPath + "/data/train-images.idx3-ubyte", trainSize, 784, trainImages);
	ReadMNIST(absPath + "/data/t10k-images.idx3-ubyte", testSize, 784, testImages);
	

	thrust::host_vector<short> trainLabels;
	thrust::host_vector<short> testLabels;
	ReadLabels(absPath + "/data/train-labels.idx1-ubyte", trainSize, trainLabels);
	ReadLabels(absPath + "/data/t10k-labels.idx1-ubyte", testSize, testLabels);

	thrust::device_vector<float> trainImagesGPU = trainImages;
	thrust::device_vector<float> meansGPU(k*dim);
	initializeMeans(trainImages, meansGPU, trainSize, k, dim);
	thrust::device_vector<float> sumMeans(k*dim);
	thrust::device_vector<int> counts(k, 0);

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	for (int itr = 0; itr < number_of_iteratins; itr++) {

	}

	return 0;
}