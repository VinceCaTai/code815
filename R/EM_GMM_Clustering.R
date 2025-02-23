getwd()
#[1] "/Users/taiyang/Desktop/BIOSTAT_Courses/BIOSTAT815/code815"
library(code815)
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
library(ggplot2)
library(bigsnpr)
library(MASS)
library(ggthemes)
library(RColorBrewer)

# ----------------------------
# (1) Function Definition
# ----------------------------

EM_GaussianMixture <- function(data, K, max_iter = 300, tol = 1e-6) {
  n <- nrow(data)
  d <- ncol(data)
  
  # Initialize parameters
  q <- rep(1 / K, K)                          # Mixing proportions
  mu <- as.matrix(data[sample(1:n, K), ])     # Randomly initialize means (ensure matrix)
  Sigma <- array(0, dim = c(d, d, K))         # Covariance matrices
  for (k in 1:K) Sigma[,,k] <- diag(d)        # Initialize with identity matrices
  w <- matrix(0, n, K)                        # Responsibility matrix
  
  # Gaussian PDF function
  gaussian_pdf <- function(x, mu, sigma) {
    x <- as.numeric(x)
    mu <- as.numeric(mu)
    det_sigma <- det(sigma)
    if (det_sigma <= 1e-10) det_sigma <- 1e-10  # Avoid singularity
    inv_sigma <- solve(sigma)
    diff <- x - mu
    exponent <- -0.5 * (t(diff) %*% inv_sigma %*% diff)
    (1 / ((2 * pi)^(length(mu)/2) * sqrt(det_sigma))) * exp(exponent)
  }
  
  log_likelihoods <- numeric(max_iter)
  
  for (iter in 1:max_iter) {
    # ----------------------------
    # E-Step: Update Responsibilities
    # ----------------------------
    for (i in 1:n) {
      density_values <- sapply(1:K, function(k) q[k] * gaussian_pdf(data[i, ], mu[k, ], Sigma[,,k]))
      w[i, ] <- density_values / sum(density_values)
    }
    
    # ----------------------------
    # M-Step: Update Parameters
    # ----------------------------
    N_k <- colSums(w)             # Effective number of points per cluster
    q <- N_k / n                  # Updated mixing proportions
    
    for (k in 1:K) {
      # Update means
      mu[k, ] <- colSums(w[, k] * data) / N_k[k]
      
      # Update covariance
      diff <- as.matrix(sweep(data, 2, mu[k, ], "-"))      # Center data as matrix
      weighted_diff <- diff * matrix(w[, k], nrow = n, ncol = d)  # Weight each row properly
      Sigma[,,k] <- (t(diff) %*% weighted_diff) / N_k[k]   # Covariance update
      
      # Handle potential singularity
      if (det(Sigma[,,k]) <= 1e-10) Sigma[,,k] <- Sigma[,,k] + 1e-6 * diag(d)
    }
    
    # ----------------------------
    # Log-Likelihood Computation
    # ----------------------------
    log_likelihoods[iter] <- sum(log(rowSums(sapply(1:K, function(k) {
      q[k] * apply(data, 1, gaussian_pdf, mu = mu[k, ], sigma = Sigma[,,k])
    }))))
    
    # ----------------------------
    # Convergence Check
    # ----------------------------
    if (iter > 1 && abs(log_likelihoods[iter] - log_likelihoods[iter - 1]) < tol) {
      message(sprintf("Converged at iteration %d.", iter))
      break
    }
  }
  
  list(q = q, mu = mu, Sigma = Sigma, responsibilities = w, log_likelihoods = log_likelihoods[1:iter])
}

# ----------------------------
# (2) Sample Data Generation
# ----------------------------
set.seed(419)
n_samples <- 150

mu1 <- c(2, 2); mu2 <- c(-3, -3); mu3 <- c(5, -5)
Sigma1 <- matrix(c(1, 0.3, 0.3, 1), 2)
Sigma2 <- matrix(c(1, -0.2, -0.2, 1), 2)
Sigma3 <- matrix(c(1, 0, 0, 1), 2)

data1 <- MASS::mvrnorm(n_samples, mu1, Sigma1)
data2 <- MASS::mvrnorm(n_samples, mu2, Sigma2)
data3 <- MASS::mvrnorm(n_samples, mu3, Sigma3)

data <- rbind(data1, data2, data3)
data <- as.data.frame(data)
colnames(data) <- c("x1", "x2")

# ----------------------------
# (3) Run the Function and Show Results
# ----------------------------

result <- EM_GaussianMixture(data, K = 3)

# ----------------------------
# Plot Clustering Results
# ----------------------------

# Assign cluster labels
data$cluster <- factor(apply(result$responsibilities, 1, which.max))
# Convert cluster centers to data frame
centers <- as.data.frame(result$mu)
colnames(centers) <- c("x1", "x2")
# Custom color palette
palette_colors <- brewer.pal(n = 3, name = "Set1")
# Generate professional plot
ggplot(data, aes(x = x1, y = x2, color = cluster)) +
  geom_point(size = 2.5, alpha = 0.8, shape = 16) +  # Data points
  geom_point(data = centers, aes(x = x1, y = x2),
             color = "black", fill = "yellow", size = 5, shape = 23, stroke = 1.2) +  # Cluster centers
  scale_color_manual(values = palette_colors, name = "Cluster") +
  labs(
    title = "EM Gaussian Mixture Clustering",
    x = expression(x[1]), y = expression(x[2])
  ) +
  theme_bw(base_size = 14, base_family = "serif") +  # Use serif fonts for publication
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 11),
    legend.position = "right",
    panel.grid.major = element_line(color = "grey80", size = 0.4),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", size = 0.8),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  guides(color = guide_legend(override.aes = list(size = 4)))  # Larger legend points

# ----------------------------
# Plot Log-Likelihood Convergence
# ----------------------------
# Compute absolute differences in log-likelihood between iterations
log_likelihood_diffs <- abs(diff(result$log_likelihoods))
# Define tolerance (should match the tol used in EM_GaussianMixture)
tol <- 1e-6
# Find the convergence iteration
convergence_iter <- which(log_likelihood_diffs < tol)[1] + 1  # +1 because diff shortens length
# Re-plot with better axis scaling and convergence line
plot(result$log_likelihoods, type = "o", pch = 19, col = "blue",
     xlab = "Iteration", ylab = "Log-Likelihood", 
     main = "Convergence of EM Algorithm",
     xlim = c(1, length(result$log_likelihoods) + 2))  # Extra space for annotation
# Check if convergence iteration was found
if (!is.na(convergence_iter)) {
  # Add vertical line
  abline(v = convergence_iter, col = "red", lty = 2, lwd = 2)
  
  # Adjust y-position for the text label
  y_pos <- result$log_likelihoods[convergence_iter] + 0.02 * diff(range(result$log_likelihoods))
  
  # Add text annotation
  text(convergence_iter, y_pos, labels = paste("Converged at iteration", convergence_iter),
       pos = 4, col = "red", cex = 0.9, font = 2)
} else {
  message("No convergence detected within the specified tolerance.")
}

