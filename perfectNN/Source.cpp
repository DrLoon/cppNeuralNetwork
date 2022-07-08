#include "NeuralN.hpp"


int main() {
	const NeuralN MyNet_static( 
		{ 10, 25, 10, 5, 3 }, 
		{ NeuralN::SIGMOID, NeuralN::SIGMOID, NeuralN::SIGMOID, NeuralN::SIGMOID } 
	);
	std::vector<double> a;
	std::vector<double> r = { 1, 1, 0.1, 0.2, 0.3, -0.6, -0.7, 1, 0.534, 0.123 };
	double s = 0;

	clock_t start = clock();
	for (int i = 0; i < 10000; ++i) {
		//r[0] += 0.00000001;
		a = MyNet_static.forward(r);
		s += a[0] + a[1] + a[2];
	}
	
	clock_t now = clock();

	std::cout << "Elapsed time in milliseconds: "
		<< (double)(now - start) / CLOCKS_PER_SEC
		<< " sec" << std::endl;
	std::cout << 10000000.0 / ((double)(now - start) / CLOCKS_PER_SEC / 100000) << std::endl;

	std::cout << a[0] << " " << a[1] << " " << a[2] << " " << s;
	return 0;
}