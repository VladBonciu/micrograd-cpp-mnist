#include "Value.h"

#include "omp.h"

double _tanh(double x)
{
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

double _sigm(double x)
{
	return exp(x) / (exp(x) + 1);
}

Value::Value()
{
	this->data = 0;
	this->grad = 0;
	//this->label = 'X'; //uninitialized
}

Value::Value(double _data)
{
	this->data = _data;
	this->grad = 0;
	//this->label = _label;
}

Value::Value(double _data, std::vector<std::weak_ptr<Value>> _children)
{
	this->data = _data;
	this->grad = 0;
	this->prev = std::move(_children);
	//this->label = _label;
}

Value& Value::operator=(std::shared_ptr<Value>& src)
{
	return *src;
}

std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value>& src)
{
	double new_data = this->data + src->data;
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this(), src});
	new_value->_backward = [self = weak_from_this(), src, new_value]()
		{
			if (auto self_locked = self.lock())
			{
				src->grad += 1.0 * new_value.get()->grad;
				self_locked->grad += 1.0 * new_value.get()->grad;
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value>& src)
{
	double new_data = this->data * src->data;
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this(), src});
	new_value->_backward = [self = weak_from_this(), src, new_value]()
		{
			if (auto self_locked = self.lock())
			{
				src->grad += self_locked->data * new_value.get()->grad;
				self_locked->grad += src->data * new_value.get()->grad;
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::operator^(std::shared_ptr<Value>& src)
{
	double new_data = (double)powl(this->data, (*src).data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this(), src});
	new_value->_backward = [self = weak_from_this(), src, new_value]()
		{
			if (auto self_locked = self.lock())
			{
				if (self_locked->data <= 0)
				{
					//if x ^ n and x = 0 the derivative is 0
					//or x is < 0 which can get complex values
					self_locked->grad += 0;
					src->grad += 0;
				}
				else
				{
					self_locked->grad += src->data * pow(self_locked->data, src->data - 1) * new_value.get()->grad;
					src->grad += (double)logl(self_locked->data) * pow(self_locked->data, src->data) * new_value.get()->grad;
				}
			}
			
		};
	return new_value;
}

std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value>& src)
{
	std::shared_ptr<Value> negative = std::make_shared<Value>(-1);
	std::shared_ptr<Value> negated_obj = *negative * src;
	return *shared_from_this() + (negated_obj);
}

std::shared_ptr<Value> Value::operator/(std::shared_ptr<Value>& src)
{
	std::shared_ptr<Value> negative = std::make_shared<Value>(-1);
	std::shared_ptr<Value> inverse = *src ^ negative;
	return *shared_from_this() * (inverse);
}

std::shared_ptr<Value> Value::tanh()
{
	double new_data = _tanh(this->data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value]()
		{
			if (auto self_locked = self.lock())
			{
				self_locked->grad += (1.0 - pow(_tanh(self_locked->data), 2.0)) * new_value.get()->grad;
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::sigm()
{
	double new_data = _sigm(this->data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value]()
		{
			if (auto self_locked = self.lock())
			{
				//turns out sigm(x)' = sigm(x) * (1-sigm(x))
				self_locked->grad += ((1.0 - _sigm(self_locked->data)) * _sigm(self_locked->data)) * new_value.get()->grad;
			}
			
		};
	return new_value;
}

std::shared_ptr<Value> Value::log()
{
	double new_data = (double)logl(this->data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value]()
		{
			if (auto self_locked = self.lock())
			{
				//turns out sigm(x)' = sigm(x) * (1-sigm(x))
				self_locked->grad += (1 / self_locked->data)  * new_value.get()->grad;
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::exp()
{

	double new_data = (double)expl(this->data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value]()
		{
			if (auto self_locked = self.lock())
			{
				//turns out sigm(x)' = sigm(x) * (1-sigm(x))
				self_locked->grad += self_locked->data * (double)expl(self_locked->data-1) * new_value.get()->grad;
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::relu()
{

	double new_data = std::fmaxl(this->data, 0.0);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value]()
		{
			if (auto self_locked = self.lock())
			{
				if (self_locked->data == 0)
				{
					self_locked->grad += 0 * new_value.get()->grad;
				}
				else
				{
					self_locked->grad += 1 * new_value.get()->grad;
				}
				
			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::leaky_relu()
{
	double alpha = 0.01;
	double new_data = std::fmaxl(this->data, alpha * this->data);
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value, alpha]()
		{
			if (auto self_locked = self.lock())
			{
				if (self_locked->data == 0)
				{
					self_locked->grad += 0 * new_value.get()->grad;
				}
				else if (self_locked->data < 0)
				{
					self_locked->grad += alpha * new_value.get()->grad;
				}
				else
				{
					self_locked->grad += 1 * new_value.get()->grad;
				}

			}
		};
	return new_value;
}

std::shared_ptr<Value> Value::elu()//WARINIG DOES NOT WORK
{
	double alpha = 0.01; 
	double new_data = std::fmaxl(this->data, alpha * (expl(this->data)-1));
	std::shared_ptr<Value> new_value = std::make_shared<Value>(new_data, std::vector<std::weak_ptr<Value>>{weak_from_this()});
	new_value->_backward = [self = weak_from_this(), new_value, alpha]()
		{
			if (auto self_locked = self.lock())
			{
				if (self_locked->data <= 0)
				{
					self_locked->grad += alpha * expl(self_locked->data) * new_value.get()->grad;
				}
				else
				{
					self_locked->grad += 1 * new_value.get()->grad;
				}

			}
		};
	return new_value;
}

void Value::backward()
{
	std::vector<std::shared_ptr<Value>> topo;
	std::set<std::shared_ptr<Value>> visited;
	std::stack<std::shared_ptr<Value>> stack;

	//std::function <void(std::shared_ptr<Value>)> topo_sort = [&topo ,&visited, &topo_sort](std::shared_ptr<Value> val)
	//	{
	//		if (visited.find(val) == visited.end())
	//		{
	//			visited.insert(val);
	//			for (int i = 0; i < val->prev.size(); i++)
	//			{
	//				if (val->prev[i] != nullptr)
	//				{
	//					topo_sort(val->prev[i]);
	//				}
	//			}
	//			topo.push_back(val);
	//		}
	//	};

	// Use an explicit stack to avoid stack overflow (instead of recursive topological sort)
	stack.push(shared_from_this());

	while (!stack.empty()) {
		auto current = stack.top();
		stack.pop();

		if (visited.find(current) == visited.end()) {
			visited.insert(current);


			if (current->prev.size() > 0)
			{
				// Push all previous nodes to the stack
				for (int i = 0; i < current->prev.size(); i++) 
				{
					auto  prev_node = current->prev[i].lock();
					if (prev_node && visited.find(prev_node) == visited.end()) {
						stack.push(prev_node);
					}
				}
			}

			// After visiting all dependencies, add to the topo vector
			topo.push_back(current);
		}
	}

	this->grad = 1.0;

	visited.clear();

	for (auto& val : topo) {
		if (val->prev.size() > 0)
		{
			val->_backward();// Perform backward operation on each element in topological order
			//val->_backward = []() {};// Erases backward functionality so that there are no cyclic references which cause memory leaks (we will later destroy the values as well)
		}
	}

	//#pragma omp parallel for
	for (int i = 0; i < topo.size();i++) {
		if (topo[i]->prev.size() > 0)
		{
			topo[i]->_backward = [](){};// Erases backward functionality so that there are no cyclic references which cause memory leaks (we will later destroy the values as well)
		}
	}

	topo.erase(topo.begin(), topo.end());
	topo.shrink_to_fit();
}

void Value::print()
{
	//if (this->label.length() > 50)
	{
		std::cout << "Value(" << this->data << ", " << this->grad << ", ...)\n";
	}
	//else
	//{
	//	std::cout << "Value(" << this->data << ", " << this->grad  << ")\n";
	//}
}

void Value::erase()
{
	std::vector<std::shared_ptr<Value>> topo;
	std::set<std::shared_ptr<Value>> visited;
	std::stack<std::shared_ptr<Value>> stack;

	// Use an explicit stack to avoid stack overflow (instead of recursive topological sort)
	stack.push(shared_from_this());

	while (!stack.empty()) {
		auto current = stack.top();
		stack.pop();

		if (visited.find(current) == visited.end()) {
			visited.insert(current);


			if (current->prev.size() > 0)
			{
				// Push all previous nodes to the stack
				for (int i = 0; i < current->prev.size(); i++)
				{
					auto  prev_node = current->prev[i].lock();
					if (prev_node && visited.find(prev_node) == visited.end()) {
						stack.push(prev_node);
					}
				}
			}

			// After visiting all dependencies, add to the topo vector
			topo.push_back(current);
		}
	}

	this->grad = 1.0;

	visited.clear();

	for (auto& val : topo) {
		if (val->prev.size() > 0)
		{
			val->_backward = []() {};// Erases backward functionality so that there are no cyclic references which cause memory leaks (we will later destroy the values as well)
		}
	}
}
