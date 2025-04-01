#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>

std::vector<std::vector<std::string>> read_csv(const char* file_path);
std::vector<std::vector<double>> convert_string_to_double_matrix(std::vector<std::vector<std::string>> in);
int read_csv_input_count(const char* file_path);