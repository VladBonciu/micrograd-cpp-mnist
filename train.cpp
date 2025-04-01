#include "train.h"
#include "omp.h"

std::vector<std::shared_ptr<Value>>  convert_values_array(std::vector<double>& x)
{
	std::vector<std::shared_ptr<Value>> new_values;
	new_values.reserve(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		new_values.push_back(std::make_shared<Value>(x[i]));
	}

	return new_values;
}

std::vector<std::vector<std::shared_ptr<Value>>> convert_values_matrix(std::vector<std::vector<double>>& x)
{
	std::vector<std::vector<std::shared_ptr<Value>>> new_values = std::vector<std::vector<std::shared_ptr<Value>>>(x.size());

	for (int i = 0; i < x.size(); i++)
	{
		new_values[i].reserve(x[i].size());
		for (int j = 0; j < x[i].size(); j++)
		{
			new_values[i].push_back(std::make_shared<Value>(x[i][j]));
		}

	}

	return new_values;
}

void save_weights(const char* file_path, std::vector<std::shared_ptr<Value>> wb, int _batch, int _epoch)
{
	std::fstream fout;
	fout.open(file_path, std::ios::out);

	if (!fout.is_open())
	{
		std::cout << "Error: file not opened!";
		return;
	}

	fout << _epoch << "\n";
	fout << _batch << "\n";

	for (int i = 0; i < wb.size();i++)
	{
		fout << wb[i]->data << " ";
	}

	fout.close();

	std::cout << "Succesfully saved weights and biases to " << file_path << "\n";
}

void load_weights(const char* file_path, std::vector<std::shared_ptr<Value>>& wb, int& _batch , int &_epoch)
{
	std::fstream fin;
	fin.open(file_path, std::ios::in);

	if (!fin.is_open())
	{
		std::cout << "Error: file not opened!";
		return;
	}

	fin >> _epoch;
	fin >> _batch;

	for (int i = 0; i < wb.size();i++)
	{
		fin >> wb[i]->data;
	}

	fin.close();

	std::cout << "Succesfully loaded weights and biases from " << file_path << "\n";
}

void softmax(std::vector<std::shared_ptr<Value>>& x)
{
	///*calculating sum(e^ypred) for  softmax function*/
	auto sum_of_exp = std::make_shared<Value>(0.000000001);

	for (int z = 0; z < x.size();z++)
	{
		auto exp = x[z]->exp();
		sum_of_exp = *sum_of_exp + exp;
	}

	for (int z = 0; z < x.size();z++)
	{
		auto exp = x[z]->exp();
		x[z] = *exp / sum_of_exp;
	}
	
}

void softmax_stable(std::vector<std::shared_ptr<Value>>& x)
{
	///*calculating sum(e^ypred) for  softmax function*/
	auto sum_of_exp = std::make_shared<Value>(0.000000001);
	auto max = std::make_shared<Value>(0);

	for (int z = 0; z < x.size();z++)
	{
		if (x[z]->data > max->data)
		{
			max = x[z];
		}
	}

	for (int z = 0; z < x.size();z++)
	{
		auto exp = (*x[z] - max)->exp();
		sum_of_exp = *sum_of_exp + exp;
	}

	for (int z = 0; z < x.size();z++)
	{
		auto exp = (*x[z] - max)->exp();
		x[z] = *exp / sum_of_exp;
	}

	//for (int z = 0; z < x.size();z++)
	//{
	//	x[z]->_backward = [=]()
	//		{
	//			for (int i = 0; i < x.size(); i++)
	//			{
	//				if (i == z)
	//				{
	//					x[i]->grad += x[i]->data * (1 - x[z]->data);
	//				}
	//				else
	//				{
	//					x[i]->grad += x[i]->data * (0 - x[z]->data);
	//				}
	//			}
	//		};
	//}

}

void sigmoid_output(std::vector<std::shared_ptr<Value>>& x)
{
	for (int z = 0; z < x.size();z++)
	{
		x[z] = x[z]->sigm();
	}
}

void tanh_output(std::vector<std::shared_ptr<Value>>& x)
{
	for (int z = 0; z < x.size();z++)
	{
		x[z] = x[z]->tanh();
	}
}

std::shared_ptr<Value> MULTICLASS_CROSS_ENTROPY(std::vector<std::shared_ptr<Value>>& y, std::vector<std::shared_ptr<Value>>& p)
{
	auto sum_yp = std::make_shared<Value>(0);

	for (int z = 0; z < p.size();z++)
	{
		auto log = (p[z]->log());
		auto a = *y[z] * log;
		sum_yp = *sum_yp + a;
	}

	sum_yp->_backward = [&]()
		{
			for (int z = 0; z < p.size();z++)
				p[z]->grad += (*p[z] + y[z])->data * sum_yp->grad;
		};

	return sum_yp;

	//for (int z = 0; z < p.size();z++)
	//{
	//	p[z]->_backward = [=]() {p[z]->grad += *(*p[z] - y[z]) };
	//}

	//for (int z = 0; z < p.size();z++)
	//{
	//	p[z]->_backward = []() {};

	//}
}

std::shared_ptr<Value> BINARY_CROSS_ENTROPY(std::vector<std::shared_ptr<Value>>& y, std::vector<std::shared_ptr<Value>>& p)
{
	auto sum_yp = std::make_shared<Value>(0);
	auto one = std::make_shared<Value>(1.0);
	auto n = std::make_shared<Value>(p.size());

	for (int j = 0; j < p.size(); j++)
	{
		auto ln1 = p[j]->log();
		auto ln2 = (*one - p[j])->log();
		auto diff1 = *y[j] * ln1;
		auto diff2 = *(*one - y[j]) * ln2;
		auto loss = *diff1 + diff2;
		sum_yp = *sum_yp + loss;
	}

	sum_yp = *sum_yp / n;

	sum_yp->_backward = [&]()
	{
		for (int z = 0; z < p.size();z++)
			p[z]->grad += (p[z]->data - y[z]->data) * sum_yp->grad;
	};

	return sum_yp;

	//for (int z = 0; z < p.size();z++)
	//{
	//	p[z]->_backward = [=]() {p[z]->grad += *(*p[z] - y[z]) };
	//}

	//for (int z = 0; z < p.size();z++)
	//{
	//	p[z]->_backward = []() {};

	//}
}

int main()
{
	//srand((unsigned int)time(0)); //for randomness on the initial neuron weights and biases values

	using namespace std::chrono;
	auto start = high_resolution_clock::now();

	std::vector<std::vector<std::string>> data = read_csv("mnist_test.csv");
	std::vector<std::vector<double>> p_data = convert_string_to_double_matrix(data);

	data.erase(data.begin(), data.end());//clean data from memory to free up space
	data.shrink_to_fit();

	int batches = read_csv_input_count("mnist_test.csv");

	int starting_batch = 0;
	int starting_epoch = 0;

	//system("pause");

	/*neural net*/
	std::vector<int> nn_layers = std::vector<int>{ N_INPUT, 16, 16, N_OUTPUT };
	MLP nn = MLP(nn_layers);

	/*weights and biases of the network*/
	std::vector<std::shared_ptr<Value>> wb = nn.get_params();

	/*training set*/
	std::vector<std::vector<double>> xi = std::vector<std::vector<double>>(BATCH_SIZE);

	/*desired output for each item of the training set*/
	std::vector<std::vector<double>> yi = std::vector<std::vector<double>>(BATCH_SIZE);

	for (int i = 0; i < BATCH_SIZE; i++)
	{
		xi[i] = std::vector<double>(p_data[0].size() - 1);
		xi[i].assign(p_data[0].begin() + 1, p_data[0].end());
		for (int k = 0; k < xi[i].size(); k++)
		{
			xi[i][k] = xi[i][k] / 255;
		}
		yi[i] = std::vector<double>{ -1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
		yi[i][(int)p_data[0][0]] = 1.0;
	}

	/*converted values*/
	std::vector<std::vector<std::shared_ptr<Value>>> xs = convert_values_matrix(xi);
	std::vector<std::vector<std::shared_ptr<Value>>> ys = convert_values_matrix(yi);

	/*actual output*/
	std::vector<std::shared_ptr<Value>> yt = std::vector<std::shared_ptr<Value>>(ys[0].size());

	std::vector<double> wb_data = std::vector<double>(wb.size());

	double learning_rate = 0.01;
	double clip_value = 1.0;

	double momentum = 1.0;
	double momentum_coef = 0.9;

	auto total_loss = std::make_shared<Value>(0.0);
	auto one = std::make_shared<Value>(1.0);
	auto min_one = std::make_shared<Value>(-1.0);
	auto out_size = std::make_shared<Value>(BATCH_SIZE);
	double min_loss = 1000;

	/*get weights and biases of the network*/
	wb = nn.get_params();

	/*load weights in ordder to train the model in multiple sessions*/
	//load_weights("wb.txt", wb, starting_batch, starting_epoch);

	printf_s("starting training with: BATCH_SIZE - %d | STEP_COUNT - %d | EPOCH_SIZE - %d | lr - %f | batches - %d\n", BATCH_SIZE, N_STEPS, N_EPOCHS , learning_rate, batches/ BATCH_SIZE);

	/*training in mini-batches over a few epochs*/
	//#pragma omp parallel for
	for (int e = starting_epoch; e < N_EPOCHS; e++)
	{
		printf_s("epoch %d / %d :\n", e, N_EPOCHS);

		if (e > 200)
		{
			printf_s("new learning rate: %f\n", learning_rate);
			learning_rate = 0.00002;
		}
		if (e > 230)
		{
			printf_s("new learning rate: %f\n", learning_rate);
			learning_rate = 0.000001;
		}

		//#pragma omp parallel for
		for (int b = starting_batch; b < batches / BATCH_SIZE ;b++)
		{

			//printf_s("batch %d / %d :\n", b, batches / BATCH_SIZE);

			//#pragma omp parallel for
			for (int i = 0; i < BATCH_SIZE; i++)
			{
				int j = i + b * BATCH_SIZE;
				xi[i] = std::vector<double>(p_data[j].size() - 1, { 0.0 });
				xi[i].assign(p_data[j].begin() + 1, p_data[j].end());
				for (int k = 0; k < xi[i].size(); k++)
				{
					xi[i][k] = xi[i][k] / 255;
				}
				//yi[i] = std::vector<double>{ -1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
				yi[i] = std::vector<double>{ 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
				yi[i][(int)p_data[j][0]] = 1.0;
				//printf_s("%d\n", i);
			}

			// Reuse memory for xs and ys instead of allocating new ones
			//for (auto& row : xs)
			//	row.clear();
			//xs.clear();
			xs = convert_values_matrix(xi);

			//for (auto& row : ys)
			//	row.clear();
			//ys.clear();
			ys = convert_values_matrix(yi);


			/*training process*/

			total_loss->data = 0;
			total_loss->grad = 0;


			for (int i = 0; i < xs.size(); i++)
			{
				yt = nn.call(xs[i]);

				//sigmoid_output(yt);
				//tanh_output(yt);
				//softmax_stable(yt);
				softmax(yt);

				for (int j = 0; j < N_OUTPUT; j++)
				{
					/*MSE LOSS*/
					//auto diff = *yt[j] - ys[i][j];
					//auto loss = *diff * diff;
					//total_loss = *total_loss + loss;

					/*BINARY CROSS ENTROPY*/
					auto ln1 = yt[j]->log();
					auto ln2 = (*one - yt[j])->log();
					auto diff1 = *ys[i][j] * ln1;
					auto diff2 = *(*one - ys[i][j]) * ln2;
					auto loss = *diff1 + diff2;
					total_loss = *total_loss - loss;

					///*MULTICLASS CROSS ENTROPY*/
					//auto ln1 = yt[j]->log();
					//auto diff1 = *ys[i][j] * ln1;
					//total_loss = *total_loss - diff1;

					/*auto a = MULTICLASS_CROSS_ENTROPY(ys[i], yt);
					total_loss = *total_loss - a;*/

					//auto ln1 = (*(*yt[j] / ys[i][j]) + one)->log();
					////auto loss = *ys[i][j] + ln1;
					//total_loss = *total_loss - ln1;

				}

				//auto a = MULTICLASS_CROSS_ENTROPY(ys[i], yt);
				//total_loss = *total_loss - a;
			}


			//total_loss = *total_loss / out_size;
			//total_loss = *total_loss * min_one;

			//for (int i = 0; i < yt.size();i++)
			//{
			//	printf_s("guess: %.2f | label: %.2f | %d\n", yt[i]->data, ys[ys.size() - 1][i]->data, i);
			//}

			//std::cout << "loss : " << total_loss->data << " \t| it : " << k << "\n";

			if(min_loss <= total_loss->data)
			{
				//printf_s("loss: %7.5f | it: %d\n", total_loss->data, b* N_STEPS);
			}
			else
			{
				min_loss = total_loss->data;
				printf_s("loss: %7.5f | it: %d -----NEW-LOW-----\n", total_loss->data,  b * N_STEPS);
			}
			
			//std::cout << nn.layers[0]->neurons[0]->w[0]->grad << "\n";

			//#pragma omp parallel for
			for (int i = 0; i < wb.size(); i++)
			{
				wb[i]->grad = 0.0;
			}
			//printf_s("w/b %d\n", wb[0].use_count());

			//for (int i = 0; i < wb[0].use_count(); i++)
			//{
			//	printf_s("w/b %d\n", wb[0].owner_before());
			//}

			/*backward pass*/
			total_loss->backward();

			//std::cout << nn.layers[0]->neurons[0]->w[0]->grad << "\n";

			//#pragma omp parallel for
			for (int i = 0; i < wb.size(); i++)
			{
				if (std::isnan(wb[i]->grad)) {
					std::cout << "NaN gradient detected!" << i << " " << momentum << std::endl;
					return -1;
				}
				wb[i]->grad = std::max(std::min(wb[i]->grad, clip_value), -clip_value);
					
				wb[i]->data += -learning_rate * wb[i]->grad;

				//momentum = momentum_coef * momentum + (1- momentum_coef) * wb[i]->grad;
				//wb[i]->data = wb[i]->data + -learning_rate * momentum;
					

			}

			//std::cout << nn.layers[0]->neurons[0]->w[0]->grad << "\n";

			//for (int i = 0; i < wb_data.size(); i++)
			//{
			//	wb_data[i] = wb[i]->data;
			//}

			//nn.reset();

			//nn = MLP(nn_layers);

			//wb = nn.get_params();

			////#pragma omp parallel for
			//for (int i = 0; i < wb_data.size(); i++)
			//{
			//	wb[i]->data = wb_data[i];
			//}

			if (b % 100 == 0)
			{
				/*saving current weights every batch*/
				save_weights("wb.txt", wb, b + 1, e);


				printf_s("loss: %7.5f | it: %d | momentum: %f\n", total_loss->data, b * N_STEPS, momentum);

				for (int i = 0; i < yt.size();i++)
				{
					printf_s("guess: %.2f | label: %.2f | %d\n", yt[i]->data, ys[ys.size() - 1][i]->data, i);
				}
			}
		}
		starting_batch = 0;
		/*and every epoch*/
		save_weights("wb.txt", wb, 0, e+1);


	}
	

	



	///*testing set*/
	//std::vector<std::vector<double>> t_xi = std::vector<std::vector<double>>
	//{ {1, 2, 3},
	//	{4, 5, 6},
	//	{7, 8, 9},
	//	{1, 1 ,2} };

	///*desired output for each item of the testing set*/
	//std::vector<std::vector<double>> t_yi = std::vector<std::vector<double>>
	//{ {1, 0 ,1},
	//	{0, 1 ,0},
	//	{1, 0 ,1},
	//	{1, 1 ,0} };

	///*converted values*/
	//std::vector<std::vector<std::shared_ptr<Value>>> t_xs = convert_values_matrix(xi);
	//std::vector<std::vector<std::shared_ptr<Value>>> t_ys = convert_values_matrix(yi);

	///*actual output*/
	//std::vector<std::shared_ptr<Value>> t_res = std::vector<std::shared_ptr<Value>>(t_ys[0].size());

	//double precision = 0;
	//double t_poz = 0;
	//double t_neg = 0;
	//double error_margin = 0.15;

	///*testing process*/
	//for (int i = 0; i < t_xs.size();i++)
	//{
	//	t_res = nn.call(t_xs[i]);

	//	auto total_diff = std::make_shared<Value>(0.0, "total_diff");

	//	for (int j = 0; j < t_res.size(); j++)
	//	{
	//		auto diff = *(t_ys[i][j]) - t_res[j];
	//		total_diff = *total_diff + diff;

	//		if (abs(diff->data) < error_margin)
	//		{
	//			t_poz++;
	//		}
	//		else
	//		{
	//			t_neg++;
	//		}

	//		printf_s("%f\n", diff->data);
	//	}

	//	//if (abs(total_diff->data) < error_margin)
	//	//{
	//	//	t_poz++;
	//	//}
	//	//else
	//	//{
	//	//	t_neg++;
	//	//}

	//	//printf_s("total diff: %f\n", total_diff->data);
	//}

	//precision = t_poz / (t_poz + t_neg);

	//printf_s("\nprecision: %f\n", precision);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	//printf_s("%Id", duration.count());
	std::cout << "execution time: " << duration.count() << " microseconds\n";

	system("pause");

	return 0;
}
