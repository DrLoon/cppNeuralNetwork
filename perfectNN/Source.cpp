#include "NeuralN.hpp"

#include <chrono>



int main() {
	const NeuralN MyNet_static( { 10, 25, 10, 5, 3 }, { NeuralN::SIGMOID, NeuralN::SIGMOID, NeuralN::SIGMOID, NeuralN::SIGMOID } );
	std::vector<double> a;
	std::vector<double> r = { 1,1,0.1,0.2,0.3,-0.6,-0.7,1,0.534,0.123 };
	double s = 0;

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < 10000; ++i) {
		//r[0] += 0.00000001;
		a = MyNet_static.forward(r);
		s += a[0] + a[1] + a[2];
	}
	auto end = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in milliseconds: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " ms" << std::endl;
	std::cout << 10000000.0 / ((double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000) << std::endl;

	std::cout << a[0] << " " << a[1] << " " << a[2] << " " << s;
	return 0;
}