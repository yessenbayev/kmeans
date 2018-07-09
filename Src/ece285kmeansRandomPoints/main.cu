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
	std::cout << fname << " is being read" << std::endl;
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
	system("dir");

	int num = 3000000;
	char* fname1 = "Data_random2/point1.txt/point1.txt";
	char* fname2 = "Data_random2/point2.txt/point2.txt";
	char* fname3 = "Data_random2/point3.txt/point3.txt";
	printf("There\n");
	vector<float>x1;
	vector<float>x2;
	vector<float>x3;
	printf("...\n");
	read_Data_random(x1,fname1);
	read_Data_random(x2,fname2);
	read_Data_random(x3,fname3);
	//for (int i = 0; i < num; i++)
		//cout <<"Value is"<< x1[i]<<"\n";
	x1.insert(x1.end(), x2.begin(), x2.end());
	x1.insert(x1.end(), x3.begin(), x3.end());
	cout << "Value has been read";

	printf("There\n");
	// use std::vector::data to access the pointer for cudaMalloc
	vector<float> trainImages;
	vector<float> testImages;

	// Use absolute path to your data folder here.
	//ReadMNIST("./data/train-images.idx3-ubyte", trainSize, dim, trainImages);
	//ReadMNIST("./data/t10k-images.idx3-ubyte", testSize, dim , testImages);


	//vector<short> trainLabels;
	//vector<short> testLabels;
	//ReadLabels("./data/train-labels.idx1-ubyte", trainSize, trainLabels);
	//ReadLabels("./data/t10k-labels.idx1-ubyte", testSize, testLabels);

	float* trainImagesGPU;
	float* meansGPU;
	int* labelGPU;
	int* labelCPU;
	float* sumMeans;
	float* counts;
	float* meansCPU;
	CHECK(cudaMalloc(&trainImagesGPU, trainSize*dim*sizeof(float)));
	CHECK(cudaMalloc(&labelGPU, trainSize*sizeof(int)));
	CHECK(cudaMemcpy(trainImagesGPU, x1.data(), trainSize*dim*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&meansGPU, k*dim*sizeof(float)));
	meansCPU = (float*)malloc(k*dim * sizeof(float));
	labelCPU = (int*)malloc(trainSize * sizeof(int));
	printf("Yay\n");
	
	initializeMeans(x1, meansGPU, trainSize, k, dim);
	
	for (int i = 0; i < trainSize*3; i+=3)
	{
		dataContainer.push_back(std::make_tuple(x1[i], x1[i + 1], x1[i + 2]));
	
	}

	CHECK(cudaMalloc(&sumMeans, k*dim * sizeof(float)));
	CHECK(cudaMalloc(&counts, k*sizeof(float)));

	dim3 block(1024);
	dim3 grid((trainSize + block.x - 1) / block.x);

	clock_t start = clock();
	for (int itr = 0; itr < number_of_iterations; itr++) {
		cudaMemset(sumMeans, 0, k*dim*sizeof(float));
		cudaMemset(counts, 0, k*sizeof(float));
		cluster_assignment << <grid, block >> > (trainImagesGPU, trainSize, meansGPU, sumMeans, k, counts, dim,labelGPU);
		
		CHECK(cudaDeviceSynchronize());

		compute_means << <1, k >> > (meansGPU, sumMeans, counts, dim);

		CHECK(cudaDeviceSynchronize());
		//if (itr % 10 == 0)
		//{
			CHECK(cudaMemcpy(meansCPU, meansGPU, k*dim * sizeof(float), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(labelCPU, labelGPU, trainSize * sizeof(int), cudaMemcpyDeviceToHost));
			for (int i = 0; i < num; i++) {
				assignmentContainer.push_back(labelCPU[i]);
			}
			printf(" The assignmnet container size is : %lld \n", assignmentContainer.size());
			cout << " iteration is" << itr;
				
		//}
	}

	printf(" The assignmnet container size is : %lld \n", assignmentContainer.size());

	/*
	CHECK(cudaMemcpy(meansCPU, meansGPU, k*dim * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(labelCPU, labelGPU, trainSize * sizeof(int), cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < num; i++)
		assignmentContainer.push_back(labelCPU[i]);*/

		
	CHECK(cudaMemcpy(meansCPU, meansGPU, k*dim * sizeof(float), cudaMemcpyDeviceToHost));
	printf("first center is %f %f %f", meansCPU[0], meansCPU[1], meansCPU[2]);
	printf("second center is %f %f %f", meansCPU[3], meansCPU[4], meansCPU[5]);
	printf("third center is %f %f %f", meansCPU[6], meansCPU[7], meansCPU[8]);
	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	printf("K-means are computed\n");

	CHECK(cudaDeviceSynchronize());

	//computing PCA by SVD with CuSolver

	printf("Starting up graphics controller\n");
	GraphicsController graphics(1920,1080);
	graphics.initGL(&argc, argv);
	graphics.run();


	CHECK(cudaDeviceReset());

	printf("Program completed executing\n");

	return 0;
}