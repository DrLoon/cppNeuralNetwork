#pragma once
#include <armadillo>
#include "act_fun.hpp"

class Dense {
	template<typename T> using vector = std::vector<T>;
	using mat = arma::mat;
	using enum activation_type;
public:
	Dense(const int _n_number, const activation_type _f_activation) : n_number(_n_number), f_activation(_f_activation)
	{
		if (_n_number <= 0)
			throw "neurals number must be more than zero";
	}

	const int n_number;
	const activation_type f_activation;
};