#include "dataset_read.h"
#include "../common/common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>
#include <cuda_fp16.h>
#include "OpenGLEngine.hpp"

vector<short> getData(vector<short> trainImages, int idx, int size) {
	vector<short> tempVec;
	int start = idx*size;
	for (int i = start; i < start + size; i++) {
		tempVec.push_back(trainImages[i]);
	}
	return tempVec;
}

void initializeMeans(vector<short> &trainImages, half* meansGPU, int trainSize, int k, int dim) {
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
		vector<short> tempVec = getData(trainImages, index, dim);
		CHECK(cudaMemcpy(meansGPU + i*dim, tempVec.data(), dim*sizeof(short), cudaMemcpyHostToDevice));
		i++;
	}
}

__device__ float calcDistance(half* p1, half* p2,int c,int len) {
	float dist = 0.0f;
	for (int i = 0; i < len; i++) {
		float pp = (__half2float(p2[c*len+i]) - __half2float(*(p1 + i)));
		dist += pp*pp;
	}
	return dist;
}

__global__ void cluster_assignment(half* trainImagesGPU,
								   int trainSize, half* meansGPU,
								   int* sumMeans, int k,
								   int* counts, int dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= trainSize) return;

	// hard coded constant-->change
	__shared__ half cluster_centers[7840];

	for (int i = threadIdx.x; i < k*dim; i += blockDim.x) {
		cluster_centers[i] = meansGPU[i];
	}
	__syncthreads();
		
	half *base_pointer = trainImagesGPU + index*dim;

	float min_distance = FLT_MAX;
	int closest_cluster = -1;
	for (int clstr = 0; clstr < k; clstr++) {
		float euclid_dist = calcDistance(base_pointer,cluster_centers,clstr,dim);
		if (euclid_dist < min_distance) {
			min_distance = euclid_dist;
			closest_cluster = clstr;
		}
	}
	
	for (int i = 0; i < dim; i++) {
		atomicAdd(sumMeans + closest_cluster*dim + i, __half2uint_rn(*(base_pointer + i)));
	}
	atomicAdd(counts + closest_cluster, 1);
}

__global__ void compute_means(half* means,
							  int* sum_means,
							  int* counts, int dim) {
	int cluster = threadIdx.x;
	int count = max(1, counts[cluster]);
	printf(" The count for the cluster : %d is %d \n", cluster, count);
	for (int i = 0; i < dim; i++) {
		*(means + cluster*dim + i) = __float2half(((float)sum_means[cluster*dim + i] / (float)count));
	}
}


int main(int *argc, char **argv) {

	const int trainSize = 60000;
	const int testSize = 10000;
	const int n_rows = 28;
	const int n_cols = 28;
	const int dim = n_rows*n_cols;
	const int k = 10; // Number of Means to be used for clustering
	const int number_of_iterations = 5;

	// use std::vector::data to access the pointer for cudaMalloc
	vector<short> trainImages;
	vector<short> testImages;

	// Use absolute path to your data folder here.
	ReadMNIST("./data/train-images.idx3-ubyte", trainSize, dim, trainImages);
	ReadMNIST("./data/t10k-images.idx3-ubyte", testSize, dim , testImages);


	vector<short> trainLabels;
	vector<short> testLabels;
	ReadLabels("./data/train-labels.idx1-ubyte", trainSize, trainLabels);
	ReadLabels("./data/t10k-labels.idx1-ubyte", testSize, testLabels);

	half* trainImagesGPU;
	half* meansGPU;
	int* sumMeans;
	int* counts;
	CHECK(cudaMalloc(&trainImagesGPU, trainSize*dim*sizeof(short)));
	CHECK(cudaMemcpy(trainImagesGPU, trainImages.data(), trainSize*dim*sizeof(short), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&meansGPU, k*dim*sizeof(short)));
	initializeMeans(trainImages, meansGPU, trainSize, k, dim);

	CHECK(cudaMalloc(&sumMeans, k*dim * sizeof(int)));
	CHECK(cudaMalloc(&counts, k*sizeof(int)));

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		cudaMemset(sumMeans, 0, k*dim*sizeof(int));
		cudaMemset(counts, 0, k*sizeof(int));
		cluster_assignment << <grid, block >> > (trainImagesGPU, trainSize, meansGPU, sumMeans, k, counts, dim);
		
		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU, sumMeans, counts, dim);

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