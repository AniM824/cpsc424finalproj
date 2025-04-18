#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"
#include <fstream>
#include <cstdint>
#include <vector>
#include <iostream>

struct eigenpair {
    double eigenvalue;
    parlay::sequence<double> eigenvector;
};

// endian swap for big-endian MNIST format
uint32_t read_uint32(std::ifstream &f) {
    uint32_t value;
    f.read(reinterpret_cast<char*>(&value), 4);
    return __builtin_bswap32(value);
}

parlay::sequence<parlay::sequence<double>> load_mnist_images_parlay(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Can't open MNIST file.");

    uint32_t magic = read_uint32(file);        // should be 2051
    uint32_t num_images = read_uint32(file);
    uint32_t num_rows = read_uint32(file);
    uint32_t num_cols = read_uint32(file);

    size_t image_size = num_rows * num_cols;
    size_t total_size = num_images * image_size;
    std::vector<unsigned char> buffer(total_size);
    file.read(reinterpret_cast<char*>(buffer.data()), total_size);

    auto images = parlay::tabulate(num_images, [&](size_t i) {
        return parlay::tabulate(image_size, [&](size_t j) {
            return static_cast<double>(buffer[i * image_size + j]) / 255.0;
        });
    });

    return images;
}

parlay::sequence<parlay::sequence<double>> parlay_transpose(const parlay::sequence<parlay::sequence<double>> &A) {
    size_t rows = A.size();
    size_t cols = A[0].size();

    return parlay::tabulate(cols, [&](size_t j) {
        return parlay::tabulate(rows, [&](size_t i) {
            return A[i][j];
        });
    });
}

double dot_product(const parlay::sequence<double> &a, const parlay::sequence<double> &b) {
    return parlay::reduce(parlay::tabulate(a.size(), [&](size_t i) {
        return a[i] * b[i];
    }));
}

parlay::sequence<parlay::sequence<double>> parlay_matrix_multiply(
    const parlay::sequence<parlay::sequence<double>> &A,
    const parlay::sequence<parlay::sequence<double>> &B)
{
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    auto B_T = parlay_transpose(B);

    return parlay::tabulate(m, [&](size_t i) {
        return parlay::tabulate(n, [&](size_t j) {
            return dot_product(A[i], B_T[j]);
        });
    });
}

parlay::sequence<eigenpair> sort_eigenpairs(const parlay::sequence<eigenpair>& data, int k) {
    auto comp = [&data](int i, int j) {
        return (data[i].eigenvalue > data[j].eigenvalue);
    };
    parlay::sequence<eigenpair> sorted_data(data.begin(), data.end());
    parlay::sort_inplace(sorted_data, comp);
    return sorted_data.cut(k);
}

parlay::sequence<parlay::sequence<double>> get_proj_matrix(const parlay::sequence<eigenpair>& top_eigenpairs) {
    size_t n = top_eigenpairs.size();
    size_t d = top_eigenpairs[0].eigenvector.size();
    
    parlay::sequence<parlay::sequence<double>> pca_matrix(n, parlay::sequence<double>(d));
    
    parlay::parallel_for(0, n, [&](size_t i) {
        const auto& eigenvector = top_eigenpairs[i].eigenvector;
        parlay::parallel_for(0, d, [&](size_t j) {
            pca_matrix[j][i] = eigenvector[j];
        });
    });
    
    return pca_matrix;
}

parlay::sequence<parlay::sequence<double>> get_projections(
    const parlay::sequence<parlay::sequence<double>>& proj_matrix, 
    const parlay::sequence<parlay::sequence<double>>& data) 
{
    return parlay_matrix_multiply(proj_matrix, data);
}

int main() {
    std::string filename = "train-images-idx3-ubyte";
    parlay::sequence<parlay::sequence<double>> U = load_mnist_images_parlay(filename);

    auto U_t = parlay_transpose(U);
    auto UTU = parlay_matrix_multiply(U_t, U);

    std::cout << "Loaded " << UTU.size() << " images, each with "
                << UTU[0].size() << " pixels." << std::endl;

    return 0;
}