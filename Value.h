#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <set>
#include <stack>

double _tanh(double x);
double _sigm(double x);
class Value : public std::enable_shared_from_this<Value>
{
public:

	double data;
	double grad;
	//std::string label;
	std::vector<std::weak_ptr<Value>> prev;
	std::function<void()> _backward;

	Value();
	Value(double _data);
	Value(double _data, std::vector<std::weak_ptr<Value>> _children);

	//~Value();

	void print();

	void erase();

	//Value& operator=(const Value& src);
	Value& operator=(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> operator+(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> operator*(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> operator^(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> operator-(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> operator/(std::shared_ptr<Value>& src);
	std::shared_ptr<Value> tanh();
	std::shared_ptr<Value> sigm();

	std::shared_ptr<Value> log();

	std::shared_ptr<Value> exp();

	std::shared_ptr<Value> relu();

	std::shared_ptr<Value> leaky_relu();

	std::shared_ptr<Value> elu();

	void backward();

	//friend void topo_sort(Value* v, std::vector<std::shared_ptr<Value>>& visited, std::vector<std::shared_ptr<Value>>(&topo));
};

