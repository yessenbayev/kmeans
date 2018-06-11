#include "dataset_read.h"
#include "../common/common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>
#include <cuda_fp16.h>
#include "OpenGLEngine.hpp"


vector<float> getData(vector<float> trainImages, int idx, int size) {
	vector<float> tempVec;
	int start = idx*size;
	for (int i = start; i < start + size; i++) {
		tempVec.push_back(trainImages[i]);
	}
	return tempVec;
}

void initializeMeans(vector<float> &trainImages, float* meansGPU, int trainSize, int k, int dim) {
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
		atomicAdd(sumMeans + closest_cluster*dim + i, *(base_pointer + i));
	}
	atomicAdd(counts + closest_cluster, 1.0);
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

void read_Data_random(float *x1, float *y1, char* fname)
{
	string line;
	ifstream myfile(fname);
	long long num = 1000000;
	int i = 0;
	string::size_type sz;
	if (myfile.is_open())
	{
		while (getline(myfile, line, ','))
		{
			if (i <num)
				x1[i] = stof(line, &sz);
			else
				y1[i - num] = stof(line, &sz);
			i++;
			if (i == 2000000)
			{
				myfile.close();
				break;
			}
		}
		myfile.close();
	}


}

int main(int argc, char **argv) {

	long long num = 1000000;
	char* fname1 = "Data_random/point1.txt/point1.txt";
	char* fname2 = "Data_random/point2.txt/point2.txt";
	char* fname3 = "Data_random/point3.txt/point3.txt";
	float *x1, *y1, *x2, *y2, *x3, *y3;
	x1 = (float*)malloc((num) * sizeof(float));
	y1 = (float*)malloc((num) * sizeof(float));
	x2 = (float*)malloc((num) * sizeof(float));
	y2 = (float*)malloc((num) * sizeof(float));
	x3 = (float*)malloc((num) * sizeof(float));
	y3 = (float*)malloc((num) * sizeof(float));
	//read_Data_random(x2, y2,fname2);
	/*for (int i = 0; i < num; i++)
	{
		printf("X2 Value is %f\n", x2[i]);
		printf("Y2 Value is %f\n", y2[i]);
	}*/


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
	float* meansGPU;
	float* sumMeans;
	float* counts;
	CHECK(cudaMalloc(&trainImagesGPU, trainSize*dim*sizeof(float)));
	CHECK(cudaMemcpy(trainImagesGPU, trainImages.data(), trainSize*dim*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&meansGPU, k*dim*sizeof(float)));
	initializeMeans(trainImages, meansGPU, trainSize, k, dim);

	CHECK(cudaMalloc(&sumMeans, k*dim * sizeof(float)));
	CHECK(cudaMalloc(&counts, k*sizeof(float)));

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		cudaMemset(sumMeans, 0, k*dim*sizeof(float));
		cudaMemset(counts, 0, k*sizeof(float));
		cluster_assignment << <grid, block >> > (trainImagesGPU, trainSize, meansGPU, sumMeans, k, counts, dim);

		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU, sumMeans, counts, dim);

		CHECK(cudaDeviceSynchronize());
	}
	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	printf("K-means are computed\n");



	//computing PCA by SVD with CuSolver

	printf("Starting up graphics controller\n");
	GraphicsController graphics;
	graphics.initGL(&argc, argv);
	graphics.run();

	CHECK(cudaDeviceReset());

	printf("Program completed executing\n");
	return 0;
}
