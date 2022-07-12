#include "NeuralN.hpp"


int main() {
	int layer1_size = 64;
	NeuralN Net( 
		{ layer1_size, 32, 32 },
		{ NeuralN::SIGMOID, NeuralN::TAHN }
	);
	std::cout << Net.paramsNumber() << " params\n";

	clock_t start = clock();
	Net.trainFromFile("train8.txt", 10000, 0.05);
	
	clock_t now = clock();
	std::cout << (double)(now - start) / CLOCKS_PER_SEC << " sec\n";
	mat test(1, layer1_size);
	std::ifstream test_data("test8.txt");
	for (int i = 0; i < layer1_size; ++i)
		test_data >> test(0, i);

	mat ans = Net.forward(test);
	for (int i = 0; i < 27; ++i)
		std::cout << "[ " << i << " ] = " << ans(0, i) << "\n";
	return 0;
}