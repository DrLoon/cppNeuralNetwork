#pragma once
#include <vector>
#include <initializer_list>

class NeuralN;
#include <armadillo>
#include "act_fun.hpp"

class NeuralN {
	using mat = arma::mat;
	template<typename T> using vector = std::vector<T>;
	using enum activation_type;
public:

	NeuralN(std::initializer_list<int> _layers_size, std::initializer_list<activation_type> _layers_activation)
		: layers_size(_layers_size),
		layers_activation(_layers_activation)
	{
		if (layers_size.size() - 1 != layers_activation.size()) throw "bad input";

		layers.resize(layers_size.size() - 1);
		for (size_t i = 0; i < layers.size(); ++i) {
			layers[i] = arma::randu<mat>(layers_size[i], layers_size[i + 1]);
		}

		biases.resize(layers_size.size() - 1);
		for (size_t i = 0; i < layers.size(); ++i) {
			biases[i] = arma::randu<mat>(1, layers_size[i + 1]);
		}
	}

	vector<double> forward(vector<double> in_data) const {
		if (in_data.size() != layers_size[0]) throw "bad input";

		mat in_layer(1, layers_size[0]);
		for (int i = 0; i < layers_size[0]; ++i)
			in_layer(0, i) = in_data[i];

		for (int i = 0; i < layers.size(); ++i) {
			in_layer = in_layer * layers[i] + biases[i];
			switch (layers_activation[i]) {
			case SIGMOID:
				act_fun::sigmoid(in_layer);
				//temp = 1 / (exp(-temp) + 1);
				break;
			case RELU:
				act_fun::relu(in_layer);
				//temp = max(temp, mat(temp.n_rows, temp.n_cols, fill::zeros));
				break;
			case SOFTMAX:
				act_fun::softmax(in_layer);
				break;
			case TAHN:
				act_fun::tahn(in_layer);
				break;
			}
		}
		return arma::conv_to< vector<double> >::from(in_layer);
	}

	void read_weitghs(std::istream& gin) {
		for (auto& i : layers)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					gin >> i(j, k);


		for (auto& i : biases)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					gin >> i(j, k);

	}

	void read_weitghs(const vector<double>& in_data) {
		int counter = 0;
		for (auto& i : layers)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					i(j, k) = in_data[counter++];

		for (auto& i : biases)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					i(j, k) = in_data[counter++];
	}

	void write_weitghs(const std::string& filePath) const {
		std::ofstream file(filePath);
		for (auto& i : layers)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					file << i(j, k) << " ";

		for (auto& i : biases)
			for (int j = 0; j < i.n_rows; ++j)
				for (int k = 0; k < i.n_cols; ++k)
					file << i(j, k) << " ";

		file.close();
	}

	int paramsNumber() const {
		int res = 0;
		for (int i = 0; i < layers_size.size() - 1; ++i) {
			res += layers_size[i] * layers_size[i + 1];
		}
		for (int i = 1; i < layers_size.size(); ++i) {
			res += layers_size[i];
		}
		return res;
	}
private:
	const vector<int> layers_size;
	const vector<activation_type> layers_activation;

	vector<mat> layers;
	vector<mat> biases;

};
