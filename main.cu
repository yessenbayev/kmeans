#include "dataset_read.h"
#include "../common/common.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <set>
#include <random>
#include <cuda_fp16.h>
#include "OpenGLEngine.hpp"
#include <cusolverDn.h>

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
								   float* counts, int dim,int* labelGPU) {
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
	labelGPU[index] = closest_cluster;
	
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

void read_Data_random(vector<float> &x1, char* fname)
{
	string line;
	ifstream myfile(fname);
	long long num = 3000000;
	int i = 0;
	string::size_type sz;
	if (myfile.is_open())
	{
		while (getline(myfile, line, ','))
		{
			x1.push_back(stof(line, &sz));
			i++;
			if (i == num)
			{
				myfile.close();
				break;
			}
		}
		myfile.close();
	}
	

}

int main(int argc, char **argv) {
	CHECK(cudaDeviceReset());

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


	// use std::vector::data to access the pointer for cudaMalloc
	vector<float> trainImages;
	vector<float> testImages;

	// Use absolute path to your data folder here.
	ReadMNIST("./data/train-images.idx3-ubyte", trainSize, dim, trainImages);
	ReadMNIST("./data/t10k-images.idx3-ubyte", testSize, dim, testImages);


	vector<short> trainLabels;
	vector<short> testLabels;
	ReadLabels("./data/train-labels.idx1-ubyte", trainSize, trainLabels);
	ReadLabels("./data/t10k-labels.idx1-ubyte", testSize, testLabels);

	float* trainImagesGPU;
	float* meansGPU;
	int* labelGPU;
	int* labelCPU;
	float* sumMeans;
	float* counts;
	float* meansCPU;
	CHECK(cudaMalloc(&trainImagesGPU, trainSize*dim * sizeof(float)));
	CHECK(cudaMalloc(&labelGPU, trainSize * sizeof(int)));
	CHECK(cudaMemcpy(trainImagesGPU, trainImages.data(), trainSize*dim * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&meansGPU, k*dim * sizeof(float)));
	meansCPU = (float*)malloc(k*dim * sizeof(float));
	labelCPU = (int*)malloc(trainSize * sizeof(int));

	initializeMeans(trainImages, meansGPU, trainSize, k, dim);
	/*
	for (int i = 0; i < trainSize,; i+=3)
	{
	dataConrainer.push_back(std::make_tuple(x1[i], x1[i + 1], x1[i + 2]);

	}*/

	CHECK(cudaMalloc(&sumMeans, k*dim * sizeof(float)));
	CHECK(cudaMalloc(&counts, k * sizeof(float)));

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		cudaMemset(sumMeans, 0, k*dim * sizeof(float));
		cudaMemset(counts, 0, k * sizeof(float));
		cluster_assignment << <grid, block >> > (trainImagesGPU, trainSize, meansGPU, sumMeans, k, counts, dim, labelGPU);

		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU, sumMeans, counts, dim);

		CHECK(cudaDeviceSynchronize());
		//CHECK(cudaMemcpy(labelCPU, labelGPU, trainSize * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < trainSize; i++)
			assignmentContainer.push_back(labelCPU[i]);
	}

	CHECK(cudaMemcpy(meansCPU, meansGPU, k*dim * sizeof(float), cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < num; i++)
	assignmentContainer.push_back(labelCPU[i]);*/



	/*printf("first center is %f %f %f", meansCPU[0], meansCPU[1], meansCPU[2]);
	printf("second center is %f %f %f", meansCPU[3], meansCPU[4], meansCPU[5]);
	printf("third center is %f %f %f", meansCPU[6], meansCPU[7], meansCPU[8]);*/
	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	printf("K-means are computed\n");

	printf("\nCentering the Dataset Matrix\n");
	const int Nrows = trainSize;
	const int Ncols = dim;
	int low_dim = 3;

	CHECK(cudaMemcpy(trainImages.data(), trainImagesGPU,  trainSize*dim * sizeof(float), cudaMemcpyDeviceToHost));

	float* meansCPU2 = (float *)malloc(dim * sizeof(float));
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < trainSize; j++) {
			meansCPU2[i] += trainImages[i + j*dim];
		}
		meansCPU2[i] /= trainSize;
	}


	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < trainSize; j++) {
			trainImages[i + j*dim] -= meansCPU2[i];
		}
	}

	//trial of transpose
	for (int i = 0; i<low_dim; i++) {
		for (int j = 0; j<dim; j++) {
			trainImages[dim*i + j] = trainImages[j*dim + i];
			//h_W[dim*i + j] = h_U[dim*i + j] * h_S[dim*i + j];
			//printf("%2f\n", h_V[j*dim + i]);
		}
	}

	CHECK(cudaMemcpy(trainImagesGPU, trainImages.data(), trainSize*dim * sizeof(float), cudaMemcpyHostToDevice));



	CHECK(cudaDeviceSynchronize());



	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;
	CHECK(cudaMalloc(&devInfo, sizeof(int)));

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- host side SVD results space
	//float *h_U = (float *)malloc(Nrows * Nrows   * sizeof(float));
	float *h_V = (float *)malloc(Ncols * Ncols * sizeof(float));
	//float *h_S = (float *)malloc(min(Nrows, Ncols) * sizeof(float));

	// --- device side SVD workspace and matrices
	float *d_U;
	float *d_V;
	float *d_S;
	//CHECK(cudaMalloc(&d_U, Nrows * Nrows * sizeof(double)));
	CHECK(cudaMalloc(&d_V, Ncols * Ncols * sizeof(float)));
	CHECK(cudaMalloc(&d_S, min(Nrows, Ncols) * sizeof(float)));

	// --- CUDA SVD initialization
	cusolverDnSgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size);
	float *work;
	CHECK(cudaMalloc(&work, work_size * sizeof(float)));
	printf("\nCUDA SVD initialization\n");


	// --- CUDA SVD execution
	cusolverDnSgesvd(solver_handle, 'N', 'S', Nrows, Ncols, trainImagesGPU, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo);
	int devInfo_h = 0;
	CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) { printf("Unsuccessful SVD execution\n\n"); }
	printf("CUDA SVD execution\n");

	// --- Moving the results from device to host
	//CHECK(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_V, d_V, Ncols * Ncols * sizeof(float), cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(float), cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy(h_U, d_U, Nrows * Nrows * sizeof(float), cudaMemcpyDeviceToHost));

	cusolverDnDestroy(solver_handle);

	for (int i = 0; i < (Ncols*Ncols); i++)  printf("SVD index %d: %f\n", i, h_V[i]);

	//--host side projection matrix: Storing h_W as 2xdim instead of dimx2.
	float *h_W = (float*)malloc(dim * low_dim * sizeof(float));

	//printf("Projection Matrix\n");
	for (int i = 0; i<low_dim; i++) {
		for (int j = 0; j<dim; j++) {
			h_W[dim*i + j] = h_V[j*dim + i];
			//h_W[dim*i + j] = h_U[dim*i + j] * h_S[dim*i + j];
			//printf("%2f\n", h_V[j*dim + i]);
		}
	}

	//--device side projection matrix:
	//double *d_W;
	//CHECK(cudaMalloc(&d_W, dim*low_dim*sizeof(double)));
	//CHECK(cudaMemcpy(d_W, h_W, dim * low_dim * sizeof(double), cudaMemcpyHostToDevice));

	// --host side trainImages :
	//float *h_A = (float *)malloc(trainSize *dim * sizeof(float));
	//CHECK(cudaMemcpy(h_A, d_A, trainSize * dim * sizeof(float), cudaMemcpyDeviceToHost));

	//for (int i = 0; i < trainSize; i++)  printf("x:%f,y:%f,z:%f\n", h_transformedData[i * 3], h_transformedData[i * 3 + 1], h_transformedData[i * 3 + 2]);

	//Matrix Multiplication: h_transformedData = h_A*h_W.T;
	float* h_transformedData = (float *)malloc(trainSize *low_dim * sizeof(float));
	for (int i = 0; i < trainSize; i++) {
		for (int j = 0; j < low_dim; j++) {
			for (int k = 0; k < dim; k++) {
				h_transformedData[i*low_dim + j] += trainImages[i*dim + k] * h_V[j*dim + k];
			}
		}
	}
	printf("Matrix multiplication done");

	for (int i = 0; i < trainSize; i++)
	{
		dataContainer.push_back(std::make_tuple(h_transformedData[i * 3], h_transformedData[i * 3 + 1], h_transformedData[i * 3 + 2]));
		//printf("x:%f,y:%f,z:%f\n", std::get<0>(dataContainer[i]), std::get<1>(dataContainer[i]), std::get<2>(dataContainer[i]));
	}


	printf("Starting up graphics controller\n");
	GraphicsController graphics(1024,1024);
	graphics.initGL(&argc, argv);
	graphics.run();

	CHECK(cudaDeviceReset());

	printf("Program completed executing\n");

	return 0;
}