getwd()
#[1] "/Users/taiyang/Desktop/BIOSTAT_Courses/BIOSTAT815/code815"
install.packages("Rcpp")
install.packages("microbenchmark")
install.packages("bigsnpr")
packageVersion("Rcpp")
library(code815)
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
library(ggplot2)
library(bigsnpr)
devtools::check()



#######Jan 22 Lab Session
sourceCpp("src/optimization_methods.cpp")
source("R/optim_1_source.R")

# Example usage
set.seed(124)
n <- 100
p <- 10
A <- matrix(rnorm(n * p), n, p)
x_true <- rnorm(p)
y <- A %*% x_true + rnorm(n)
x0 <- rep(0, p)
lambda <- 0.1
gamma <- 0.01
batch_size <- 10

# Gradient Descent with Fixed Step Size
res1 <- gradient_descent_lsq(y, A, x0, lambda, gamma)
print("Results for gradient_descent_lsq:")
print(res1)
# Adaptive Gradient Descent
res2 <- gradient_descent_lsq_v2(y, A, x0, lambda, gamma)
print("Results for gradient_descent_lsq_v2:")
print(res2)
# Gradient Descent with Barzilai-Borwein Method
res3 <- gradient_descent_BB_lsq(y, A, x0, lambda)
print("Results for gradient_descent_BB_lsq:")
print(res3)
# Stochastic Gradient Descent
batch_size <- 10  # Example batch size
res4 <- stochastic_gradient_descent_lsq(y, A, x0, lambda, batch_size, initial_step_size = 1)
print("Results for stochastic_gradient_descent_lsq:")
print(res4)

# Benchmark both Rcpp and R implementations
benchmark_results <- microbenchmark(
  # Rcpp implementations
  Rcpp_gradient_fixed = gradient_descent_lsq(y, A, x0, lambda, gamma),
  Rcpp_gradient_adaptive = gradient_descent_lsq_v2(y, A, x0, lambda, gamma),
  Rcpp_gradient_BB = gradient_descent_BB_lsq(y, A, x0, lambda),
  Rcpp_stochastic_gradient = stochastic_gradient_descent_lsq(y, A, x0, lambda, batch_size, initial_step_size = 1),
  
  # R implementations
  R_gradient_fixed = gradient.descent.lsq_R(y, A, x0, lambda, gamma),
  R_gradient_adaptive = gradient.descent.lsq.v2_R(y, A, x0, lambda, gamma),
  R_gradient_BB = gradient.descent.BB.lsq_R(y, A, x0, lambda),
  R_stochastic_gradient = stochastic.gradient.descent.lsq_R(y, A, x0, lambda, batch = batch_size, initial.step.size = 1),
  
  times = 10  # Number of repetitions
)
print(benchmark_results)
autoplot(benchmark_results) +
  labs(
    title = "Benchmark: Rcpp vs R Implementations",
    subtitle = "Comparison of Execution Time (10 Repetitions)",
    x = "Method",
    y = "Execution Time (µs)"
  ) +
  theme_minimal(base_size = 14) + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, face = "bold"),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic"),
    legend.position = "none"
  ) +
  scale_y_log10() + 
  geom_jitter(width = 0.2, alpha = 0.7, color = "blue") +  
  geom_boxplot(alpha = 0.5, outlier.shape = NA)  # Overlay boxplot without outliers

# Run one instance of each function and compare results
Rcpp_res <- gradient_descent_lsq(y, A, x0, lambda, gamma)
R_res <- gradient.descent.lsq_R(y, A, x0, lambda, gamma)
# Check coefficient differences
all.equal(Rcpp_res$x, R_res$x)
# Check loss differences
all.equal(Rcpp_res$loss, R_res$loss)



########Jan 27 Lecture Practice
source("Lec_Practice/IRWLS_Poisson.R")
sourceCpp("Lec_Practice/IRWLS_Poisson_Rcpp.cpp")
##Sample Data
set.seed(124) 
n <- 500
p <- 6
X <- matrix(rnorm(n * p), n, p)
beta_true <- rnorm(p)
y <- rpois(n, lambda = exp(X %*% beta_true))  
##Implementation in R
beta_est_r <- IRWLS_Poisson(X, y)
print("R Implementation Results:")
print(beta_est_r)
##Implementation in Rcpp
beta_est_rcpp <- IRWLS_Poisson_Rcpp(X, y)
print("Rcpp Implementation Results:")
print(beta_est_rcpp)
##Benchmark Process
benchmark_results <- microbenchmark(
  R_Implementation = IRWLS_Poisson(X, y),
  Rcpp_Implementation = IRWLS_Poisson_Rcpp(X, y),
  times = 10
)
print(benchmark_results)
##Visualization
autoplot(benchmark_results) +
  labs(
    title = "Benchmark: R vs Rcpp Implementations",
    subtitle = "Comparison of Execution Time for IRWLS Poisson Regression",
    x = "Implementation",
    y = "Execution Time (µs)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, face = "bold"),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic"),
    legend.position = "none"
  ) +
  scale_y_log10() + 
  geom_jitter(width = 0.2, alpha = 0.7, color = "blue") + 
  geom_boxplot(alpha = 0.5, outlier.shape = NA) 

