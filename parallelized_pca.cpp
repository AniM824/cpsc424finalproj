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
#include <chrono>
#include <Accelerate/Accelerate.h>


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

// Loads the labels file and returns a vector of size numItems
std::vector<uint8_t> load_mnist_labels(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    // 1) Magic number (should be 0x00000801)
    uint32_t magic = read_uint32(in);
    if (magic != 0x00000801)
        throw std::runtime_error("Invalid MNIST label file magic: 0x"
            + std::to_string(magic));

    // 2) Number of labels
    uint32_t numItems = read_uint32(in);

    // 3) Read all labels (one byte each)
    std::vector<uint8_t> labels(numItems);
    if (!in.read(reinterpret_cast<char*>(labels.data()), numItems))
        throw std::runtime_error("Failed to read label data");

    return labels;
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

double blas_dot_product(const Vector &a, const Vector &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    std::vector<double> a_flat(a.begin(), a.end());
    std::vector<double> b_flat(b.begin(), b.end());
    
    return cblas_ddot(a.size(), a_flat.data(), 1, b_flat.data(), 1);
}

Matrix blas_matrix_multiply(const Matrix& A, const Matrix& B) {
    size_t m = A.size();   
    size_t k = A[0].size();  
    size_t n = B[0].size();  

    // Flattening matrices more efficiently
    std::vector<double> A_flat(m * k);
    std::vector<double> B_flat(k * n);

    // Optimize A matrix flattening - direct assignment instead of std::copy
    parlay::parallel_for(0, m, [&](size_t i) {
        for (size_t j = 0; j < k; ++j) {
            A_flat[i * k + j] = A[i][j];
        }
    });

    // Fix B matrix flattening - was incorrectly accessing data
    parlay::parallel_for(0, k, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            B_flat[i * n + j] = B[i][j];
        }
    });

    std::vector<double> C_flat(m * n, 0.0);

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        1.0,
        A_flat.data(), k,
        B_flat.data(), n,
        0.0,
        C_flat.data(), n
    );

    // More efficient result conversion with better parallelism
    return parlay::tabulate(m, [&](size_t i) {
        return parlay::tabulate(n, [&](size_t j) {
            return C_flat[i * n + j];
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
                double dot = blas_dot_product(qi, Q[j]);
                return parlay::tabulate(d, [&](size_t l) {
                    return dot * Q[j][l];
                });
            });

            auto projections_T = parlay_transpose(projections);
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

      parlay::parallel_for(0, n, [&](size_t k) {
        double R_ik1 = R[i - 1][k];
        double R_ik2 = R[i][k];
        R[i - 1][k] = c * R_ik1 - s * R_ik2;
        R[i][k] = s * R_ik1 + c * R_ik2;
      });

      parlay::parallel_for(0, n, [&](size_t k) {
        double Q_ik1 = Q[k][i - 1];
        double Q_ik2 = Q[k][i];
        Q[k][i - 1] = c * Q_ik1 - s * Q_ik2;
        Q[k][i] = s * Q_ik1 + c * Q_ik2;
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
    A = blas_matrix_multiply(R, Q);
    V = blas_matrix_multiply(V, Q);

    double off_diag_sum = parlay::reduce(parlay::tabulate(n * n, [&](size_t idx) {
        size_t i = idx / n;
        size_t j = idx % n;
        if (i != j) {
            return A[i][j] * A[i][j];
        } 
        else {
            return 0.0;
        }
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

Matrix extract_columns(const Matrix &A, size_t num_cols) {
    size_t rows = A.size();
    return parlay::tabulate(rows, [&](size_t i) {
        return parlay::tabulate(num_cols, [&](size_t j) {
            return A[i][j];
        });
    });
}

std::pair<Matrix, Vector> randomized_block_power_iteration(const Matrix& A, int k, int q = 100, const Matrix& Omega = Matrix()) {
    size_t d = A.size();

    Matrix omega = Omega.size() > 0 ? Omega : random_matrix_generator(d, k);
    Matrix Y = blas_matrix_multiply(A, omega);

    for (int i = 0; i < q; ++i) {
        Y = blas_matrix_multiply(A, Y);
        Y = parlay_transpose(Y);
        Y = gram_schmidt(Y);
        Y = parlay_transpose(Y);
    }

    Matrix Q = parlay_transpose(Y);
    Matrix A_trunc = extract_columns(A, k);
    Matrix B = blas_matrix_multiply(parlay_transpose(Q), blas_matrix_multiply(A_trunc, Q));
    auto [V, eigenvalues] = eigen_decompose_small(B);
    Matrix eigenvectors = blas_matrix_multiply(Q, V);

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

void mean_center(Matrix& data) {
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

void comparison_test(const Matrix& U, const std::vector<uint8_t>& labels, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto U_t = parlay_transpose(U);
    auto UTU = blas_matrix_multiply(U_t, U);

    auto result = randomized_block_power_iteration(UTU, k);
    const Matrix& eigenvectors = result.first;
    const Vector& eigenvalues = result.second;

    parlay::sequence<eigenpair> eigenpairs = parlay::tabulate(k, [&](size_t i) {
        return eigenpair{eigenvalues[i], eigenvectors[i]};
    });

    auto top_k = sort_eigenpairs(eigenpairs);
    auto proj_matrix = get_proj_matrix(top_k);
    auto projected = blas_matrix_multiply(U, proj_matrix);

    // end timing
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Projection block took " 
              << elapsed.count() << " seconds\n";
    
    size_t n = projected.size();
    std::cout << "Projected matrix size: " << n << " × " << projected[0].size() << std::endl;

    std::ofstream out("pca_cpp.csv");
    out << std::fixed << std::setprecision(6);
    for (size_t j = 0; j < k; ++j) {
      out << "PC" << (j+1) << ',';
    }
    out << "label\n";

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        out << projected[i][j] << ',';
      }
      out << int(labels[i]) << '\n';
    }
    out.close();
    std::cout << "Wrote " << n << " rows (+" << k 
              << " PCs + 1 label) to pca_cpp.csv\n";
}

void detailed_time_test(const Matrix& U, const std::vector<uint8_t>& labels, int k, int num_iterations) {
    auto start_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    std::cout << "" << std::endl;
    
    // Step 1: Transpose
    auto t1_start = std::chrono::high_resolution_clock::now();
    auto U_t = parlay_transpose(U);
    auto t1_end = std::chrono::high_resolution_clock::now();
    elapsed = t1_end - t1_start;
    std::cout << "Transpose operation took " << elapsed.count() << " seconds\n";
    
    // Step 2: Matrix multiplication
    auto t2_start = std::chrono::high_resolution_clock::now();
    auto UTU = blas_matrix_multiply(U_t, U);
    auto t2_end = std::chrono::high_resolution_clock::now();
    elapsed = t2_end - t2_start;
    std::cout << "Matrix multiplication took " << elapsed.count() << " seconds\n";
    
    // Step 3: Power iteration
    auto t3_start = std::chrono::high_resolution_clock::now();
    auto result = randomized_block_power_iteration(UTU, k, num_iterations);
    auto t3_end = std::chrono::high_resolution_clock::now();
    elapsed = t3_end - t3_start;
    std::cout << "Randomized block power iteration with " << num_iterations << " iterations took " << elapsed.count() << " seconds\n";
    
    const Matrix& eigenvectors = result.first;
    const Vector& eigenvalues = result.second;

    // Step 4: Creating eigenpairs
    auto t4_start = std::chrono::high_resolution_clock::now();
    parlay::sequence<eigenpair> eigenpairs = parlay::tabulate(k, [&](size_t i) {
        return eigenpair{eigenvalues[i], eigenvectors[i]};
    });
    auto t4_end = std::chrono::high_resolution_clock::now();
    elapsed = t4_end - t4_start;
    std::cout << "Creating eigenpairs took " << elapsed.count() << " seconds\n";
    
    // Step 5: Sorting eigenpairs
    auto t5_start = std::chrono::high_resolution_clock::now();
    auto top_k = sort_eigenpairs(eigenpairs);
    auto t5_end = std::chrono::high_resolution_clock::now();
    elapsed = t5_end - t5_start;
    std::cout << "Sorting eigenpairs took " << elapsed.count() << " seconds\n";
    
    // Step 6: Getting projection matrix
    auto t6_start = std::chrono::high_resolution_clock::now();
    auto proj_matrix = get_proj_matrix(top_k);
    auto t6_end = std::chrono::high_resolution_clock::now();
    elapsed = t6_end - t6_start;
    std::cout << "Getting projection matrix took " << elapsed.count() << " seconds\n";
    
    // Step 7: Final projection
    auto t7_start = std::chrono::high_resolution_clock::now();
    auto projected = blas_matrix_multiply(U, proj_matrix);
    auto t7_end = std::chrono::high_resolution_clock::now();
    elapsed = t7_end - t7_start;
    std::cout << "Final projection took " << elapsed.count() << " seconds\n";
    
    auto end_total = std::chrono::high_resolution_clock::now();
    elapsed = end_total - start_total;
    std::cout << "Total PCA computation time: " << elapsed.count() << " seconds\n";

    // Output results to CSV
    size_t n = projected.size();
    std::ofstream out("pca_cpp.csv");
    out << std::fixed << std::setprecision(6);
    for (size_t j = 0; j < k; ++j) {
      out << "PC" << (j+1) << ',';
    }
    out << "label\n";

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        out << projected[i][j] << ',';
      }
      out << int(labels[i]) << '\n';
    }
    out.close();
    std::cout << "Wrote " << n << " rows (+" << k 
              << " PCs + 1 label) to pca_cpp.csv\n";

    std::cout << "" << std::endl;
}

void iterations_test(const Matrix& U, const std::vector<uint8_t>& labels, int k) {
    std::vector<int> iterations = {2, 5, 10, 50};
    
    // Generate the random matrix once and reuse it for all iterations
    size_t d = U.size();
    Matrix Omega = random_matrix_generator(d, k);
    
    for (int q : iterations) {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto U_t = parlay_transpose(U);
        auto UTU = blas_matrix_multiply(U_t, U);

        // Use the same Omega for all iterations
        auto result = randomized_block_power_iteration(UTU, k, q, Omega);
        const Matrix& eigenvectors = result.first;
        const Vector& eigenvalues = result.second;

        parlay::sequence<eigenpair> eigenpairs = parlay::tabulate(k, [&](size_t i) {
            return eigenpair{eigenvalues[i], eigenvectors[i]};
        });

        auto top_k = sort_eigenpairs(eigenpairs);
        auto proj_matrix = get_proj_matrix(top_k);
        auto projected   = blas_matrix_multiply(U, proj_matrix);

        // end timing
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Projection with " << q << " iterations took " 
                << elapsed.count() << " seconds\n";
        
        size_t n = projected.size();
        std::cout << "Projected matrix size: " << n << " × " << projected[0].size() << std::endl;

        std::string output_file = "pca_cpp_" + std::to_string(q) + "_iters.csv";
        std::ofstream out(output_file);
        out << std::fixed << std::setprecision(6);
        for (size_t j = 0; j < k; ++j) {
            out << "PC" << (j+1) << ',';
        }
        out << "label\n";

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                out << projected[i][j] << ',';
            }
            out << int(labels[i]) << '\n';
        }
        out.close();
    }
}

int main() {
    std::string filename = "train-images-idx3-ubyte";
    Matrix U = load_mnist_images_parlay(filename);
    mean_center(U);

    auto labels = load_mnist_labels("train-labels-idx1-ubyte");
    
    // Define k here and pass to the functions
    int k = 2;
    
    detailed_time_test(U, labels, k, 50);
    // comparison_test(U, labels, k);
    // iterations_test(U, labels, k);
    
    return 0;
}