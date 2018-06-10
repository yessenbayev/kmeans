#include "dataset_read.h"
#include "../common/common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>
#include "OpenGLEngine.hpp"


thrust::host_vector<float> getData(thrust::host_vector<float> trainImages, int idx, int size) {
	thrust::host_vector<float> tempVec;
	int start = idx*size;
	for (int i = start; i < start + size; i++) {
		tempVec.push_back(trainImages[i]);
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
		//printf("Random Index Generated is : %d \n", index);
		thrust::host_vector<float> tempVec = getData(trainImages, index, dim);
		thrust::copy(tempVec.begin(), tempVec.end(), meansGPU.begin() + i*dim);
		i++;
	}
}

__device__ float calcDistance(float* p1, float* p2, int len) {
	float dist = 0.0f;
	for (int i = 0; i < len; i++) {
		dist += (*(p2 + i) - *(p1 + i)) * (*(p2 + i) - *(p1 + i));
	}
	return dist;
}

__global__ void cluster_assignment(thrust::device_ptr<float> trainImagesGPU,
								   int trainSize, const thrust::device_ptr<float> meansGPU,
								   thrust::device_ptr<float> sumMeans, int k,
								   thrust::device_ptr<int> counts, int dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= trainSize) return;

		
	float *base_pointer = thrust::raw_pointer_cast(trainImagesGPU) + index*dim;

	float min_distance = FLT_MAX;
	int closest_cluster = -1;
	for (int clstr = 0; clstr < k; clstr++) {
		float* cluster_pointer = thrust::raw_pointer_cast(meansGPU) + clstr * dim;
		float euclid_dist = calcDistance(base_pointer, cluster_pointer, dim);
		if (euclid_dist < min_distance) {
			min_distance = euclid_dist;
			closest_cluster = clstr;
		}
	}
	
	for (int i = 0; i < dim; i++) {
		atomicAdd(thrust::raw_pointer_cast(sumMeans) + closest_cluster*dim + i, *(base_pointer + i));
	}
	atomicAdd(thrust::raw_pointer_cast(counts) + closest_cluster, 1);
}

__global__ void compute_means(thrust::device_ptr<float> means,
							  const thrust::device_ptr<float> sum_means,
							  const thrust::device_ptr<int> counts, int dim) {
	int cluster = threadIdx.x;
	int count = max(1, counts[cluster]);
	//printf(" The count for the cluster : %d is %d \n", cluster, count);
	for (int i = 0; i < dim; i++) {
		means[cluster*dim + i] = ((float)sum_means[cluster*dim + i] / (float)count);
	}
}


int main(int *argc, char **argv) {

	const int trainSize = 60000;
	const int testSize = 10000;
	const int n_rows = 28;
	const int n_cols = 28;
	const int dim = n_rows*n_cols;
	const int k = 10; // Number of Means to be used for clustering
	const int number_of_iterations = 100;

	// use std::vector::data to access the pointer for cudaMalloc
	thrust::host_vector<float> trainImages;
	thrust::host_vector<float> testImages;

	// Use absolute path to your data folder here.
	ReadMNIST("./data/train-images.idx3-ubyte", trainSize, dim, trainImages);
	ReadMNIST("./data/t10k-images.idx3-ubyte", testSize, dim , testImages);


	thrust::host_vector<short> trainLabels;
	thrust::host_vector<short> testLabels;
	ReadLabels("./data/train-labels.idx1-ubyte", trainSize, trainLabels);
	ReadLabels("./data/t10k-labels.idx1-ubyte", testSize, testLabels);

	
	thrust::device_vector<float> trainImagesGPU = trainImages;
	thrust::device_vector<float> meansGPU(k*dim);
	initializeMeans(trainImages, meansGPU, trainSize, k, dim);


	thrust::device_vector<float> sumMeans(k*dim);
	thrust::device_vector<int> counts(k);

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		thrust::fill(sumMeans.begin(), sumMeans.end(), 0);
		thrust::fill(counts.begin(), counts.end(), 0);
		cluster_assignment << <grid, block >> > (trainImagesGPU.data(), trainSize, meansGPU.data(), sumMeans.data(), k, counts.data(), dim);
		
		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU.data(), sumMeans.data(), counts.data(), dim);

		CHECK(cudaDeviceSynchronize());
	}
	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	printf("K-means are computed\n");



	//computing PCA by SVD with CuSolver

	//printf("Starting up graphics controller");
	//GraphicsController graphics;
	//graphics.initGL(argc, argv);
	//graphics.run();


	//CHECK(cudaDeviceReset());

	printf("Program completed executing\n");

	return 0;
}