#pragma once
#include <armadillo>

enum class activation_type { SIGMOID, RELU, SOFTMAX, TAHN };

class act_fun final {
	using mat = arma::mat;
private:
	act_fun() {};
public:
	static void relu(mat& input) {
		for (auto& i : input) {
			i = std::max(0.0, i);
		}
	}
	static void sigmoid(mat& input) {
		for (auto& i : input) {
			i = 1.0 / (exp(-i) + 1.0);
		}
	}
	static void tahn(mat& input) {
		for (auto& i : input) {
			i = 2.0 / (1 + exp(-2.0 * i)) - 1.0;
		}
	}
	static void softmax(mat& input) {
		double s = 0;
		double M = input.max();
		for (auto& i : input) {
			i -= M;
			s += exp(i);
		}
		for (auto& i : input) {
			i = exp(i) / s;
		}
	}
};