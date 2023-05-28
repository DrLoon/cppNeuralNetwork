#pragma once
#include <vector>
#include <initializer_list>
#include <algorithm>

class NeuralN;
#include <armadillo>

namespace {

	using mat = arma::mat;

	void relu(mat& input) {
		for (auto& i : input) {
			i = std::max(0.0, i);
		}
	}
	void sigmoid(mat& input) {
		for (auto& i : input) {
			i = 1.0 / (exp(-i) + 1.0);
		}
	}
	double sigmoid(double input) {
		return 1.0 / (exp(-input) + 1.0);
	}
	void tahn(mat& input) {
		for (auto& i : input) {
			i = 2.0 / (1 + exp(-2.0 * i)) - 1.0;
		}
	}
	void softmax(mat& input) {
		double s = 0;
		double M = input.max();
		for (auto& i : input) {
			i -= M;
			s += exp(i);
		}
		for(auto& i : input) {
			i = exp(i) / s;
		}
	}
}

class NeuralN {
	using mat = arma::mat;
	template<typename T> using vector = std::vector<T>;
public:
	enum activation_type { SIGMOID, RELU, SOFTMAX, TAHN };

	NeuralN(std::initializer_list<int> _layers_size, std::initializer_list<activation_type> _layers_activation, bool _is_bias = true)
		: layers_size(_layers_size),
		  layers_activation(_layers_activation), 
		  is_bias(_is_bias)
	{
		if (layers_size.size() - 1 != layers_activation.size()) throw "bad input";

		layers.resize(layers_size.size() - 1);
		for (size_t i = 0; i < layers.size(); ++i) {
			layers[i] = arma::randu<mat>(layers_size[i], layers_size[i + 1]);
		}

		results.resize(layers_size.size());
		for (size_t i = 0; i < layers.size(); ++i) {
			results[i] = arma::randu<mat>(1, layers_size[i]);
		}

		results_before = results;

		biases.resize(layers_size.size() - 1);
		for (size_t i = 0; i < layers.size(); ++i) {
			biases[i] = arma::randu<mat>(1, layers_size[i + 1]);
		}
	}

	mat mat_from_vec(const vector<double>& vec) const{
		mat res(1, vec.size());
		for (int i = 0; i < vec.size(); ++i)
			res(0, i) = vec[i];
		return res;
	}

	vector<double> forward(const vector<double>& in_data) const {
		if (in_data.size() != layers_size[0]) throw "bad input";

		mat in_layer = std::move(mat_from_vec(in_data));

		for (int i = 0; i < layers.size(); ++i) {
			in_layer = in_layer * layers[i] + biases[i];
			switch (layers_activation[i]) {
			case SIGMOID:
				sigmoid(in_layer);
				break;
			case RELU:
				relu(in_layer);
				break;
			case SOFTMAX:
				softmax(in_layer);
				break;
			case TAHN:
				tahn(in_layer);
				break;
			}
		}
		return arma::conv_to< vector<double> >::from(in_layer);
	}
	mat forward(mat in_layer){
		if (in_layer.n_cols != layers_size[0]) throw "bad input";

		results[0] = in_layer;
		results_before[0] = results[0];
		for (int i = 0; i < layers.size(); ++i) {
			results[i + 1] = results[i] * layers[i];
			if (is_bias) results[i + 1] += biases[i];
			results_before[i + 1] = results[i + 1];
			switch (layers_activation[i]) {
			case SIGMOID:
				sigmoid(results[i + 1]);
				break;
			case RELU:
				relu(results[i + 1]);
				break;
			case SOFTMAX:
				softmax(results[i + 1]);
				break;
			case TAHN:
				tahn(results[i + 1]);
				break;
			}
		}
		return results.end()[-1];
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

		if (is_bias) 
		{
			for (int i = 1; i < layers_size.size(); ++i) {
				res += layers_size[i];
			}
		}
		return res;
	}
	void backtracking(mat inputs, mat answers, double alpha = 0.05) {
		if (inputs.n_cols != layers_size[0]) std::cout << "(backtracking err)";
		if (answers.n_cols != layers_size.end()[-1]) std::cout << "(backtracking err)";

		mat result = forward(inputs);
		vector<mat> err = results;
		err.end()[-1] = answers - result;

		for (int i = layers_size.size() - 1; i > 0; i--)
		{
			err[i - 1] = err[i] * layers[i - 1].t();
		}

		std::vector<mat> dWeits = layers;
		mat E;

		for (int i = 0; i < layers_size.size() - 1; i++) {
			E = results[i + 1];
			for (int j = 0; j < layers_size[i + 1]; j++)
			{
				//E(0, j) = err[i + 1](0,j) * this->gradient(results[i + 1](0, j), i);
				E(0, j) = err[i + 1](0, j) * this->gradient(results_before[i + 1](0, j), i);
			}

			dWeits[i] = (results[i].t() * E);
			layers[i] = layers[i] + dWeits[i] * alpha;
			if (is_bias) biases[i] = biases[i] + E * alpha;
		}
	}
	
	
	void train(std::vector<std::vector<double>> _x, std::vector<std::vector<double>> _y, int epochs, double _alpha = 0.05) {
		if (_x.size() != _y.size()) std::cout << "bad train input\n";

		int train_size = _x.size();

		std::vector<mat> x;
		std::vector<mat> y;
		for (auto& i : _x) x.push_back(std::move(mat_from_vec(i)));
		for (auto& i : _y) y.push_back(std::move(mat_from_vec(i)));


		for (int epoch = 0; epoch < epochs; ++epoch) {
			double err = 0;
			for (int sample = 0; sample < train_size; ++sample)
			{
				backtracking(x[sample], y[sample], _alpha);
				auto ans = this->forward(x[sample]);
				for (int k = 0; k < 32; ++k)
					//err += -log(1 - abs(ans.data[0][k] - ans_data[j].data[0][k]) + 0.001);
					err += abs(ans(0, k) - y[sample](0, k));
			}
			std::cout << epoch << " - " << err << "\n";
		}
	}
	double gradient(double x, double layer_num) {
		switch (layers_activation[layer_num]) {
		case SIGMOID:
			return sigmoid(x) * (1 - sigmoid(x));
			break;
		case RELU:
			return std::max(0.0, x);
			break;
		case TAHN:
			return 4 * sigmoid(2 * x) * (1 - sigmoid(2 * x));
			break;
		}
		return 0;
	}
	double update_weight() {
		//double beta_1 = 0.9;
		//double beta_2 = 0.999;
		//double gradient;
		//double m = beta_1 * old_m + (1 - beta_1) * gradient;
		//double s = beta_2 * old_s + (1 - beta_2) * gradient * gradient;
		//double cool_m = m / (1 - pow(beta_1, itaration));
		//double cool_s = s / (1 - pow(beta_2, itaration));
		//return cool_m * (sqrt(cool_s) + 1);
		return 0;
	}

private:
	const vector<int> layers_size;
	const vector<activation_type> layers_activation;

	vector<mat> layers;
	vector<mat> biases;
	vector<mat> results;
	vector<mat> results_before;

	bool is_bias = false;
};
