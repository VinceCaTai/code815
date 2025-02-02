#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// Ridge regression loss function
// [[Rcpp::export]]
double loss_ridge(const arma::vec& y, const arma::mat& A, const arma::vec& x, double lambda) {
  arma::vec res = y - A * x;
  return 0.5 * arma::dot(res, res) + lambda * arma::dot(x, x);
}

// Gradient descent with fixed step size
// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq(const arma::vec& y, const arma::mat& A, arma::vec x0,
                                double lambda, double gamma, double tol = 0.0001, 
                                int max_iter = 10000, bool printing = false) {
  int n = A.n_rows, p = A.n_cols;
  arma::mat AA = A.t() * A;
  arma::vec Ay = A.t() * y;
  arma::vec grad = AA * x0 - Ay + 2 * lambda * x0;
  double prev_loss = loss_ridge(y, A, x0, lambda);
  
  arma::vec diff_rec(max_iter), loss_rec(max_iter);
  double diff = arma::datum::inf;
  int iter = 0;
  arma::vec x = x0;
  
  while (iter < max_iter && diff > tol) {
    x -= gamma * grad;
    grad = AA * x - Ay + 2 * lambda * x;
    
    double loss = loss_ridge(y, A, x, lambda);
    diff_rec(iter) = std::abs((prev_loss - loss) / prev_loss);
    loss_rec(iter) = loss;
    diff = diff_rec(iter);
    
    prev_loss = loss;
    iter++;
  }
  
  if (printing) Rcpp::Rcout << "Converged after " << iter << " iterations." << std::endl;
  
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("diff") = diff_rec.head(iter),
                            Rcpp::Named("loss") = loss_rec.head(iter));
}

// Adaptive gradient descent
// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq_v2(const arma::vec& y, const arma::mat& A, arma::vec x0,
                                   double lambda, double gamma, double tol = 0.0001,
                                   int max_iter = 10000, bool printing = false) {
  int n = A.n_rows, p = A.n_cols;
  arma::mat AA = A.t() * A;
  arma::vec Ay = A.t() * y;
  arma::vec grad = AA * x0 - Ay + 2 * lambda * x0;
  double prev_loss = loss_ridge(y, A, x0, lambda);
  
  arma::vec diff_rec(max_iter), loss_rec(max_iter);
  double diff = arma::datum::inf;
  int iter = 0;
  arma::vec x = x0;
  
  while (iter < max_iter && diff > tol) {
    x -= gamma * grad;
    grad = AA * x - Ay + 2 * lambda * x;
    
    double loss = loss_ridge(y, A, x, lambda);
    diff_rec(iter) = std::abs((prev_loss - loss) / prev_loss);
    diff = diff_rec(iter);
    
    // Adaptive step size
    if (prev_loss > loss) {
      gamma *= 1.1;
    } else {
      gamma /= 2.0;
    }
    
    loss_rec(iter) = loss;
    prev_loss = loss;
    iter++;
  }
  
  if (printing) Rcpp::Rcout << "Converged after " << iter << " iterations." << std::endl;
  
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("diff") = diff_rec.head(iter),
                            Rcpp::Named("loss") = loss_rec.head(iter));
}

// Barzilai-Borwein method
// [[Rcpp::export]]
Rcpp::List gradient_descent_BB_lsq(const arma::vec& y, const arma::mat& A, arma::vec x0,
                                   double lambda, double tol = 0.0001, int max_iter = 10000,
                                   bool printing = false) {
  int n = A.n_rows, p = A.n_cols;
  arma::mat AA = A.t() * A;
  arma::vec Ay = A.t() * y;
  arma::vec grad = AA * x0 - Ay + 2 * lambda * x0;
  
  arma::vec diff_rec(max_iter);
  double prev_loss = loss_ridge(y, A, x0, lambda);
  arma::vec x = x0 - grad, prev_x = x0, prev_grad = grad;
  
  double diff = arma::datum::inf;
  int iter = 0;
  
  while (iter < max_iter && diff > tol) {
    grad = AA * x - Ay + 2 * lambda * x;
    
    double gamma = arma::dot(x - prev_x, x - prev_x) / arma::dot(x - prev_x, grad - prev_grad);
    prev_x = x;
    prev_grad = grad;
    x -= gamma * grad;
    
    double loss = loss_ridge(y, A, x, lambda);
    diff_rec(iter) = std::abs((prev_loss - loss) / prev_loss);
    diff = diff_rec(iter);
    prev_loss = loss;
    
    iter++;
  }
  
  if (printing) Rcpp::Rcout << "Converged after " << iter << " iterations." << std::endl;
  
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("diff") = diff_rec.head(iter));
}

// Stochastic gradient descent
// [[Rcpp::export]]
Rcpp::List stochastic_gradient_descent_lsq(const arma::vec& y, const arma::mat& A, arma::vec x0,
                                           double lambda, int batch_size, double initial_step_size = 1,
                                           double tol = 1E-6, int max_iter = 10000, bool printing = false) {
  int n = A.n_rows, p = A.n_cols;
  
  arma::vec diff_rec(max_iter), loss_rec(max_iter);
  double prev_loss = loss_ridge(y, A, x0, lambda);
  arma::vec x = x0;
  double diff = arma::datum::inf;
  
  int iter = 0;
  
  while (iter < max_iter && diff > tol) {
    arma::uvec batch = arma::randi<arma::uvec>(batch_size, arma::distr_param(0, n - 1));
    arma::mat Asub = A.rows(batch);
    arma::vec ysub = y.elem(batch);
    
    arma::mat AA = Asub.t() * Asub;
    arma::vec Ay = Asub.t() * ysub;
    
    arma::vec grad = (AA * x - Ay) / batch_size + 2 * lambda * x / n;
    x -= (initial_step_size / (iter + 1)) * grad;
    
    double loss = loss_ridge(y, A, x, lambda);
    diff_rec(iter) = std::abs((prev_loss - loss) / prev_loss);
    diff = diff_rec(iter);
    prev_loss = loss;
    
    iter++;
  }
  
  if (printing) Rcpp::Rcout << "Converged after " << iter << " iterations." << std::endl;
  
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("diff") = diff_rec.head(iter),
                            Rcpp::Named("loss") = loss_rec.head(iter));
}