#include "NeuralN.hpp"


int main() {
	const int input_size = 64;
	const int output_size = 32;
	NeuralN Net( 
		{ input_size, 32, output_size },
		{ NeuralN::SIGMOID, NeuralN::TAHN },
		true
	);
	std::cout << Net.paramsNumber() << " params\n";

	//trainig data reading 
	std::ifstream fin("train8.txt");
	int train_num;
	fin >> train_num;
	std::vector<std::vector<double>> train_data(train_num);
	std::vector<std::vector<double>> ans_data(train_num);

	for (int i = 0; i < train_num; i++)
	{
		train_data[i].resize(input_size);
		for (auto& j : train_data[i]) fin >> j;

		ans_data[i].resize(output_size);
		for (auto& j : ans_data[i]) fin >> j;

	}

	//training
	clock_t start = clock();
	Net.train(train_data, ans_data, 10000, 0.05);
	
	clock_t now = clock();
	std::cout << (double)(now - start) / CLOCKS_PER_SEC << " sec\n";


	mat test(1, input_size);
	std::ifstream test_data("test8.txt");
	for (int i = 0; i < input_size; ++i)
		test_data >> test(0, i);

	mat ans = Net.forward(test);
	for (int i = 0; i < 27; ++i)
		std::cout << "[ " << i << " ] = " << ans(0, i) << "\n";
	return 0;
}