#include "dataset_read.h"
#include "../common/common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>
#include <cuda_fp16.h>
#include "OpenGLEngine.hpp"

vector<float> ICDCPU;
vector<int> RIDCPU;


vector<float> getData(vector<float> trainImages, int idx, int size) {
	vector<float> tempVec;
	int start = idx*size;
	for (int i = start; i < start + size; i++) {
		tempVec.push_back(trainImages[i]);
	}
	return tempVec;
}

void initializeMeans(vector<float> &trainImages, float* meansCPU, float* meansGPU, int trainSize, int k, int dim) {
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
		vector<float> tempVec = getData(trainImages, index, dim);
		CHECK(cudaMemcpy(meansGPU + i*dim, tempVec.data(), dim*sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(meansCPU + i*dim, tempVec.data(), dim*sizeof(float), cudaMemcpyHostToHost));
		i++;
	}
}

__device__ float calcDistance(float* p1, half* p2,int c,int len) {
	float dist = 0.0f;
	for (int i = 0; i < len; i++) {
		float pp = (__half2float(p2[c*len+i]) - (*(p1 + i)));
		dist += pp*pp;
	}
	return dist;
}

__global__ void cluster_assignment(float* trainImagesGPU,
								   int trainSize, float* meansGPU,
								   int* assignments, float* ICD, int* RID,
								   float* sumMeans, int k,
								   float* counts, int dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= trainSize) return;

	// hard coded constant-->change
	__shared__ half cluster_centers[7840];

	for (int i = threadIdx.x; i < k*dim; i += blockDim.x) {
		cluster_centers[i] = __float2half(meansGPU[i]);
	}
	__syncthreads();

	float *base_pointer = trainImagesGPU + index*dim;

	int oldcenter = assignments[index];
	float oldDist;
	if (oldcenter != -1) {
		oldDist = calcDistance(base_pointer, cluster_centers, oldcenter, dim);
	}
	else {
		oldDist = FLT_MAX;
	}
	if (oldcenter != -1) {
		for (int l = 1; l < k; l++) {
			int clstr = RID[oldcenter*k + l];
			if (ICD[oldcenter*k+clstr] > 2 * oldDist) {
				break;
			}
			float euclid_dist = calcDistance(base_pointer, cluster_centers, clstr, dim);
			if (euclid_dist < oldDist) {
				oldDist = euclid_dist;
				oldcenter = clstr;
			}
		}
	}
	else {
		for (int clstr = 0; clstr < k; clstr++) {
			float euclid_dist = calcDistance(base_pointer, cluster_centers, clstr, dim);
			if (euclid_dist < oldDist) {
				oldDist = euclid_dist;
				oldcenter = clstr;
			}
		}
	}

	assignments[index] = oldcenter;
	
	for (int i = 0; i < dim; i++) {
		atomicAdd(sumMeans + oldcenter*dim + i, *(base_pointer + i));
	}
	atomicAdd(counts + oldcenter, 1.0);
}

__global__ void compute_means(float* means,
							  float* sum_means,
							  float* counts, int dim) {
	int cluster = threadIdx.x;
	float count = max(1.0f, counts[cluster]);
	//printf(" The count for the cluster : %d is %lf \n", cluster, count);
	for (int i = 0; i < dim; i++) {
		means[cluster*dim + i] = (sum_means[cluster*dim + i] / count);
	}
}

float calcDistanceClusters(float* meansCPU, int i, int j, int dim) {
	float dist = 0.0f;
	for (int u = 0; u < dim; u++) {
		dist += (meansCPU[i*dim + u] - meansCPU[j*dim + u])*(meansCPU[i*dim + u] - meansCPU[j*dim + u]);
	}
	return dist / (float)dim;
}

bool cmp(int i, int j) {
	return ICDCPU[i] < ICDCPU[j];
}

vector<float> preProcess(float* meansCPU, float* ICDGPU, int* RIDGPU, int k, int dim) {
	vector<float> ICDCPU;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			ICDCPU.push_back(calcDistanceClusters(meansCPU, i, j, dim));
		}
	}
	return ICDCPU;
}

vector<int> preProcessTwo(float* ICDGPU, int* RIDGPU, int k){
	vector<int> RIDCPU;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			RIDCPU.push_back(i*k + j);
		}
	}
	for (int i = 0; i < k; i++) {
		std::sort(RIDCPU.begin() + i*k, RIDCPU.begin() + i*k + k, cmp);
	}

	for (int i = 0; i < k*k; i++) {
		RIDCPU[i] = RIDCPU[i] % k;
	}

	CHECK(cudaMemcpy(ICDGPU, ICDCPU.data(), k*k * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(RIDGPU, RIDCPU.data(), k*k * sizeof(int), cudaMemcpyHostToDevice));
	return RIDCPU;
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
	vector<float> trainImages;
	vector<float> testImages;

	// Use absolute path to your data folder here.
	ReadMNIST("./data/train-images.idx3-ubyte", trainSize, dim, trainImages);
	ReadMNIST("./data/t10k-images.idx3-ubyte", testSize, dim , testImages);


	vector<short> trainLabels;
	vector<short> testLabels;
	ReadLabels("./data/train-labels.idx1-ubyte", trainSize, trainLabels);
	ReadLabels("./data/t10k-labels.idx1-ubyte", testSize, testLabels);

	float* trainImagesGPU;
	float* meansCPU;
	float* meansGPU;
	float* sumMeans;
	float* counts;
	float* ICDGPU;
	int* RIDGPU;
	int* assignments;
	CHECK(cudaMalloc(&trainImagesGPU, trainSize*dim*sizeof(float)));
	CHECK(cudaMalloc(&ICDGPU, k*k * sizeof(float)));
	CHECK(cudaMalloc(&assignments, trainSize * sizeof(int)));
	CHECK(cudaMalloc(&RIDGPU, k*k * sizeof(int)));
	CHECK(cudaMemcpy(trainImagesGPU, trainImages.data(), trainSize*dim*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&meansGPU, k*dim*sizeof(float)));
	meansCPU = (float*)malloc(k*dim * sizeof(float));
	initializeMeans(trainImages, meansCPU, meansGPU, trainSize, k, dim);
	
	ICDCPU = preProcess(meansCPU, ICDGPU, RIDGPU, k, dim);
	RIDCPU = preProcessTwo(ICDGPU, RIDGPU,k);

	CHECK(cudaMalloc(&sumMeans, k*dim * sizeof(float)));
	CHECK(cudaMalloc(&counts, k*sizeof(float)));
	cudaMemset(assignments, -1, trainSize * sizeof(int));

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		cudaMemset(sumMeans, 0, k*dim*sizeof(float));
		cudaMemset(counts, 0, k*sizeof(float));
		cluster_assignment << <grid, block >> > (trainImagesGPU, trainSize, meansGPU, assignments, ICDGPU, RIDGPU, sumMeans, k, counts, dim);
		
		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU, sumMeans, counts, dim);

		CHECK(cudaMemcpy(meansCPU, meansGPU, k*dim* sizeof(float), cudaMemcpyDeviceToHost));

		ICDCPU = preProcess(meansCPU, ICDGPU, RIDGPU, k, dim);
		RIDCPU = preProcessTwo(ICDGPU, RIDGPU, k);

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