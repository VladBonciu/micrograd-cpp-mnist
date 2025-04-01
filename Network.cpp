#include "Network.h"
#include "omp.h"

std::shared_ptr<Value> rand_val(int n_in)
{
	//std::cout << (double)rand() / (double)RAND_MAX << "\n";
	//return std::make_shared<Value>(((double)rand() / (double)RAND_MAX) * 2 * (1/sqrt(n_in)) - (1 / sqrt(n_in)));
	return std::make_shared<Value>(((double)rand() / (double)RAND_MAX) * (1 / sqrt(n_in)));
}

Neuron::Neuron()
{
	std::cout << "initialized empty neuron\n";
}

Neuron::Neuron(int n_in)
{
	w = std::vector<std::shared_ptr<Value>>(n_in);
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = rand_val(n_in);
	}
	b = rand_val(n_in);
}

std::shared_ptr<Value> Neuron::call(std::vector<std::shared_ptr<Value>>& x)
{
	std::shared_ptr<Value> out = std::make_shared<Value>(0.0);

	for (int i = 0; i < w.size(); i++)
	{
		out = *(*(w[i]) * x[i]) + out;
	}

	out = *b + out;

	//out->print();

	//out = out->tanh();
	//out = out->sigm();
	out = out->relu();
	//out = out->leaky_relu();
	//out = out->elu();//doesnt work

	//out->print();

	return out;
}

Layer::Layer()
{
	std::cout << "initialized empty layer\n";
}

Layer::Layer(int n_in, int n_out)
{
	neurons = std::vector<std::shared_ptr<Neuron>>(n_out);
	for (int i = 0; i < n_out; i++)
	{
		neurons[i] = std::make_shared<Neuron>(n_in);
	}
}

std::vector<std::shared_ptr<Value>> Layer::call(std::vector<std::shared_ptr<Value>>& x)
{
	std::vector<std::shared_ptr<Value>> outs = std::vector<std::shared_ptr<Value>>(neurons.size());
	//#pragma omp parallel for
	for (int i = 0;i < neurons.size(); i++)
	{
		outs[i] = neurons[i]->call(x);
	}
	return outs;
}

MLP::MLP()
{
	std::cout << "initialized empty MLP\n";
}

MLP::MLP(std::vector<int>& _layers)
{
	layers = std::vector<std::shared_ptr<Layer>>(_layers.size() - 1);
	for (int i = 0; i < _layers.size() - 1; i++)
	{
		layers[i] = std::make_shared<Layer>(_layers[i], _layers[i + 1]);
	}
}

std::vector<std::shared_ptr<Value>> MLP::call(std::vector<std::shared_ptr<Value>>& x)
{
	
	std::vector<std::shared_ptr<Value>> outs = x;
	for (int i = 0; i < layers.size(); i++)
	{
		outs = layers[i]->call(outs);
	}
	return outs;
}

std::vector<std::shared_ptr<Value>> MLP::get_params()
{
	std::vector<std::shared_ptr<Value>> wb ;
	for (int i = 0; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i]->neurons.size(); j++)
		{
			for (int k = 0; k < layers[i]->neurons[j]->w.size();k++)
			{
				wb.push_back(layers[i]->neurons[j]->w[k]);
			}
			wb.push_back(layers[i]->neurons[j]->b);
		}
	}

	return wb;
}


// Ensure proper cleanup
void MLP::reset()
{
	for (auto& layer : layers)
	{
		for (auto& neuron : layer->neurons)
		{
			// Explicitly reset shared_ptr of neuron weights/biases if needed.
			for (auto& weight : neuron->w)
			{
				weight.reset();  // Releases memory
			}
			neuron->b.reset();  // Releases memory
		}
	}
	layers.clear();  // Clears the layers and neurons
}