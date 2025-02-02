#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::vec IRWLS_Poisson_Rcpp(const arma::mat& X, const arma::vec& y, double tol = 1e-6, int max_iter = 100) {
  int p = X.n_cols;  // Number of predictors
  arma::vec beta = arma::zeros(p);  // Initialize beta (p x 1 vector)
  
  for (int iter = 0; iter < max_iter; ++iter) {
    // Compute linear predictor and mean response
    arma::vec eta = X * beta;  // Linear predictor
    arma::vec lambda = arma::exp(eta);  // Mean response
    
    // Working response
    arma::vec y_star = eta + (y - lambda) / lambda;
    
    // Weight matrix
    arma::vec w = lambda;  // Diagonal weights as a vector
    
    // Weighted least squares
    arma::mat XtW = X.t() * arma::diagmat(w);
    arma::mat XtWX = XtW * X;
    arma::vec XtWy = XtW * y_star;
    
    arma::vec beta_new = arma::solve(XtWX, XtWy);  // Solve for beta
    
    // Check for convergence
    if (arma::norm(beta_new - beta, 2) < tol) {
      Rcpp::Rcout << "Converged in " << iter + 1 << " iterations." << std::endl;
      return beta_new;
    }
    
    beta = beta_new;  // Update beta
  }
  
  // If max_iter is reached without convergence
  Rcpp::warning("Algorithm did not converge within the maximum number of iterations.");
  return beta;
}