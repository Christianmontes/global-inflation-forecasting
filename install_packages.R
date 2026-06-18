# R packages for the replication package.
# Run once before the R script:  Rscript install_packages.R
# Add packages to this vector as the R code develops.
packages <- c(
  # "readxl",
  # "writexl"
)

missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing, repos = "https://cloud.r-project.org")