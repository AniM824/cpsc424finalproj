#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"
#include <fstream>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <utility>

using Vector = parlay::sequence<double>;
using Matrix = parlay::sequence<Vector>;

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

Matrix load_mnist_images_parlay(const std::string &filename) {
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

Matrix parlay_transpose(const Matrix &A) {
    size_t rows = A.size();
    size_t cols = A[0].size();

    return parlay::tabulate(cols, [&](size_t j) {
        return parlay::tabulate(rows, [&](size_t i) {
            return A[i][j];
        });
    });
}

double parlay_dot_product(const Vector &a, const Vector &b) {
    return parlay::reduce(parlay::tabulate(a.size(), [&](size_t i) {
        return a[i] * b[i];
    }));
}

Matrix parlay_matrix_multiply(
    const parlay::sequence<parlay::sequence<double>> &A,
    const parlay::sequence<parlay::sequence<double>> &B)
{
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    auto B_T = parlay_transpose(B);

    return parlay::tabulate(m, [&](size_t i) {
        return parlay::tabulate(n, [&](size_t j) {
            return parlay_dot_product(A[i], B_T[j]);
        });
    });
}

Matrix random_matrix_generator(size_t rows, size_t cols) {
    std::default_random_engine generator;
    std::normal_distribution<double> gaussian(0.0,1.0);
    return parlay::tabulate(rows, [&](size_t i) {
        return parlay::tabulate(cols, [&](size_t j) {
            return gaussian(generator);
        });
    });
}

void normalize(Vector &vec) {
    double sum_squares = parlay::reduce(parlay::map(vec, [](double x) { return x * x; }));
    double norm = std::sqrt(sum_squares);
    if (norm > 0.0) {
        parlay::parallel_for(0, vec.size(), [&](size_t i) {
            vec[i] /= norm;
        });
    }
}

Matrix gram_schmidt(const Matrix &Y_cols) {
    size_t k = Y_cols.size();
    size_t d = Y_cols[0].size();
    Matrix Q(k, Vector(d, 0.0));

    for (size_t i = 0; i < k; ++i) {
        Vector qi = Y_cols[i];

        if (i > 0) {
            auto projections = parlay::tabulate(i, [&](size_t j) {
                double dot = parlay_dot_product(qi, Q[j]);
                return parlay::tabulate(d, [&](size_t l) {
                    return dot * Q[j][l];
                });
            });

            auto projections_T = parlay_transpose(projections);  // d × i
            Vector total_projection = parlay::tabulate(d, [&](size_t l) {
                return parlay::reduce(projections_T[l], parlay::addm<double>());
            });

            parlay::parallel_for(0, d, [&](size_t l) {
                qi[l] -= total_projection[l];
            });
        }

        normalize(qi);
        Q[i] = std::move(qi);
    }

    return Q;
}

Matrix identity_matrix(size_t n) {
  return parlay::tabulate(n, [&](size_t i) {
    return parlay::tabulate(n, [&](size_t j) {
      return (i == j) ? 1.0 : 0.0;
    });
  });
}

std::pair<Matrix, Matrix> qr_decomposition_givens(const Matrix &A) {
  size_t n = A.size();
  Matrix R = A;
  Matrix Q = identity_matrix(n);

  for (size_t j = 0; j < n - 1; ++j) {
    for (size_t i = n - 1; i > j; --i) {
      double a = R[i - 1][j];
      double b = R[i][j];

      if (std::abs(b) < 1e-10) continue;

      double r = std::hypot(a, b);
      double c = a / r;
      double s = -b / r;

      // === Parallel rotation on R (rows i-1 and i) ===
      parlay::parallel_for(0, n, [&](size_t k) {
        double R_ik1 = R[i - 1][k];
        double R_ik2 = R[i][k];
        R[i - 1][k] = c * R_ik1 - s * R_ik2;
        R[i][k]     = s * R_ik1 + c * R_ik2;
      });

      // === Parallel rotation on Q (accumulate transforms) ===
      parlay::parallel_for(0, n, [&](size_t k) {
        double Q_ik1 = Q[k][i - 1];
        double Q_ik2 = Q[k][i];
        Q[k][i - 1] = c * Q_ik1 - s * Q_ik2;
        Q[k][i]     = s * Q_ik1 + c * Q_ik2;
      });
    }
  }

  return {Q, R};
}

std::pair<Matrix, Vector> eigen_decompose_small(const Matrix &B) {
  const size_t max_iters = 100;
  size_t n = B.size();
  Matrix A = B;
  Matrix V = identity_matrix(n);

  for (size_t iter = 0; iter < max_iters; ++iter) {
    auto [Q, R] = qr_decomposition_givens(A);
    A = parlay_matrix_multiply(R, Q);
    V = parlay_matrix_multiply(V, Q);

    double off_diag_sum = parlay::reduce(parlay::tabulate(n * n, [&](size_t idx) {
        size_t i = idx / n;
        size_t j = idx % n;
        return (i != j) ? A[i][j] * A[i][j] : 0.0;
    }));

    if(std::sqrt(off_diag_sum) < 1e-10){
        break;
    }
  }

  Vector eigenvalues = parlay::tabulate(n, [&](size_t i) {
    return A[i][i];
  });

  return {V, eigenvalues};
}

std::pair<Matrix, Vector> randomized_block_power_iteration(const Matrix& A, int k, int q = 2){
  size_t d = A.size();

  Matrix Omega = random_matrix_generator(d, k);
  Matrix Y = parlay_matrix_multiply(A, Omega);

  for (int i = 0; i < q; ++i) {
    Y = parlay_matrix_multiply(A, Y);
    Y = parlay_transpose(Y);
    Y = gram_schmidt(Y);
    Y = parlay_transpose(Y);
  }

  Matrix Q = parlay_transpose(Y);
  Matrix B = parlay_matrix_multiply(parlay_transpose(Q), parlay_matrix_multiply(A, Q));
  auto [V, eigenvalues] = eigen_decompose_small(B);
  Matrix eigenvectors = parlay_matrix_multiply(Q, V);

  return {eigenvectors, eigenvalues};
}

parlay::sequence<eigenpair> sort_eigenpairs(const parlay::sequence<eigenpair>& data) {
    parlay::sequence<eigenpair> sorted_data(data.begin(), data.end());
    parlay::sort_inplace(sorted_data, [&](const eigenpair &a, const eigenpair &b) {
        return a.eigenvalue > b.eigenvalue;
    });

    return sorted_data;
}

parlay::sequence<parlay::sequence<double>> get_proj_matrix(const parlay::sequence<eigenpair>& top_eigenpairs) {
    size_t k = top_eigenpairs.size();
    size_t d = top_eigenpairs[0].eigenvector.size();
    
    parlay::sequence<parlay::sequence<double>> pca_matrix(d, parlay::sequence<double>(k));
    
    parlay::parallel_for(0, k, [&](size_t i) {
        const auto& eigenvector = top_eigenpairs[i].eigenvector;
        parlay::parallel_for(0, d, [&](size_t j) {
            pca_matrix[j][i] = eigenvector[j];
        });
    });
    
    return pca_matrix;
}

void mean_center(Matrix &data) {
    size_t n = data.size();
    size_t d = data[0].size();

    Vector mean = parlay::tabulate(d, [&](size_t j) {
        return parlay::reduce(parlay::tabulate(n, [&](size_t i) {
            return data[i][j];
        })) / static_cast<double>(n);
    });

    parlay::parallel_for(0, n, [&](size_t i) {
        parlay::parallel_for(0, d, [&](size_t j) {
            data[i][j] -= mean[j];
        });
    });
}


int main() {
    std::string filename = "train-images-idx3-ubyte";
    Matrix U = load_mnist_images_parlay(filename);
    mean_center(U);

    auto U_t = parlay_transpose(U);
    auto UTU = parlay_matrix_multiply(U_t, U);

    std::cout << "Loaded " << U.size() << " images, each with "
                << U[0].size() << " pixels." << std::endl;



    int k = 1;
    auto result = randomized_block_power_iteration(UTU, k);
    const Matrix& eigenvectors = result.first;
    const Vector& eigenvalues = result.second;

    parlay::sequence<eigenpair> eigenpairs = parlay::tabulate(k, [&](size_t i) {
        return eigenpair{eigenvalues[i], eigenvectors[i]};
    });

    auto top_k = sort_eigenpairs(eigenpairs);

    auto proj_matrix = get_proj_matrix(top_k);

    auto projected = parlay_matrix_multiply(U, proj_matrix);

    // std::cout << "Proj matrix size" << proj_matrix.size() << " × " << proj_matrix[0].size() << std::endl;
    // std::cout << "U size: " << U.size() << " × " << U[0].size() << std::endl;
    std::cout << "Projected matrix size: " << projected.size() << " × " << projected[0].size() << std::endl;
    // int i = 0;
    // for (const auto& row : projected) {
    //     for (double val : row) {
    //         std::cout << std::fixed << std::setprecision(2) << val << " ";
    //     }
    //     i += 1;
    // std::cout << "\n";
}


    return 0;
}