#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"


struct eigenpair {
    double eigenvalue;
    parlay::sequence<double> eigenvector;
};

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

parlay::sequence<parlay::sequence<double>> get_projections(const parlay::sequence<parlay::sequence<double>>& proj_matrix, const parlay::sequence<parlay::sequence<double>>& data) {
    return parlay_matrix_multiply(proj_matrix, data);
}