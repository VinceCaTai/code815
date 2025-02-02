# Iteratively Re-weighted Least Squares (IRLS) for Poisson Regression
IRWLS_Poisson <- function(X, y, tol = 1e-6, max_iter = 100) {

  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p) 
  
  # Iterative process
  for (iter in 1:max_iter) {

    eta <- X %*% beta
    lambda <- exp(eta)  
    
    # Working response
    y_star <- eta + (y - lambda) / lambda
    
    # Weight matrix
    W <- diag(as.vector(lambda))
    

    XtW <- t(X) %*% W
    beta_new <- solve(XtW %*% X, XtW %*% y_star)
    

    if (sum(abs(beta_new - beta)) < tol) {
      cat("Converged in", iter, "iterations.\n")
      return(beta_new)
    }
    
    beta <- beta_new
  }
  
  warning("Algorithm did not converge within the maximum number of iterations.")
  return(beta)
}
