#pragma once

#include "Value.h"

#include <vector>
#include <memory>
#include <ctime>

class Neuron : public std::enable_shared_from_this<Neuron>
{
public:
	Neuron();
	Neuron(int n_in);

	std::shared_ptr<Value> call(std::vector<std::shared_ptr<Value>>& x);

	std::vector<std::shared_ptr<Value>> w;
	std::shared_ptr<Value> b;
};

class Layer : public std::enable_shared_from_this<Layer>
{
public:
	Layer();
	Layer(int n_in, int n_out);

	std::vector<std::shared_ptr<Value>> call(std::vector<std::shared_ptr<Value>>& x);

	std::vector<std::shared_ptr<Neuron>> neurons;
};

class MLP
{
public:
	MLP();
	MLP(std::vector<int>& layers);

	std::vector<std::shared_ptr<Value>> call(std::vector<std::shared_ptr<Value>>& x);
	std::vector<std::shared_ptr<Value>> get_params();

	void reset();

	std::vector<std::shared_ptr<Layer>> layers;
};



