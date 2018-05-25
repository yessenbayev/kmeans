#include "dataset_read.h"

int main() {
	vector<vector<uint8_t>> trainImage;
	vector<vector<uint8_t>> testImage;
	ReadMNIST("data/train-images.idx3-ubyte", 60000, 784, trainImage);
	ReadMNIST("data/t10k-images.idx3-ubyte", 10000, 784, testImage);

	vector<uint8_t> trainLabels;
	vector<uint8_t> testLabels;
	ReadLabels("data/train-labels.idx1-ubyte", 60000, trainLabels);
	ReadLabels("data/t10k-labels.idx1-ubyte", 10000, testLabels);

	return 0;
}