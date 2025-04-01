#include "BatchReadCSV.h"

std::vector<std::vector<std::string>> read_csv(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return std::vector<std::vector<std::string>>(0);
    }

    int row_count = 0;
    int col_count = 0;

    std::vector<std::vector<std::string>> out;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        std::vector<std::string> row;

        row.clear();

        // used for breaking words
        std::stringstream s(line);

        row_count++;

        while (std::getline(s, word, ','))
        {
            // add all the column data
            // of a row to a vector
            row.push_back(word);
        }

        out.push_back(row);
    }

    fin.close();

    std::cout << "Succesfully read .csv data.\n";

    return out;
}

int read_csv_input_count(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return -1;
    }

    int row_count = 0;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        row_count++;
    }

    fin.close();

    row_count--;

    std::cout << "Elements in the .csv file:" << row_count << "\n";

    return row_count;
}

std::vector<std::vector<std::string>> read_csv_batch(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return std::vector<std::vector<std::string>>(0);
    }

    int row_count = 0;
    int col_count = 0;

    std::vector<std::vector<std::string>> out;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        std::vector<std::string> row;

        row.clear();

        // used for breaking words
        std::stringstream s(line);

        row_count++;

        while (std::getline(s, word, ','))
        {
            // add all the column data
            // of a row to a vector
            row.push_back(word);
        }

        out.push_back(row);
    }

    fin.close();

    std::cout << "Succesfully read .csv data.\n";

    return out;
}

std::vector<std::vector<double>> convert_string_to_double_matrix(std::vector<std::vector<std::string>> in)
{
    std::vector<std::vector<double>> out = std::vector<std::vector<double>>(in.size() - 1);
    for (int i = 0; i < (in.size() - 1);i++)
    {
        out[i] = std::vector<double>(in[i + 1].size());
    }

    for (int i = 1; i < in.size(); i++)
    {
        for (int j = 0; j < in[i].size(); j++)
        {
            out[i - 1][j] = std::stod(in[i][j]);
            //out[i][j] = 
        }
    }

    std::cout << "Succesfully converted string data to double.\n";

    return out;
}



//int main(void)
//{
//    std::vector<std::vector<std::string>> data = read_csv("mnist_test.csv");
//    std::vector<std::vector<double>> p_data = convert_string_to_double_matrix(data);
//    
//    for (int i = 0; i < p_data.size();i++)
//    {
//        std::cout << p_data[i][0] << "\n";
//    }
//
//    std::cout << p_data.size() << "\n";
//
//    return 0;
//}