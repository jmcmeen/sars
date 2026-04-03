# Rscript tests/r_reference/generate_r_reference.R
library(sars)

# 1. Export the exact galap dataset (16 rows)
write.csv(galap, "tests/r_reference/galap.csv", row.names = FALSE)
cat("Exported galap:", nrow(galap), "rows\n")

# 2. Fit all 20 models and export reference values
models <- c("power","powerR","epm1","epm2","p1","p2","loga","koba",
            "mmf","monod","negexpo","chapman","weibull3","asymp",
            "ratio","gompertz","weibull4","betap","heleg","linear")

results <- list()
for (m in models) {
  fn <- tryCatch(get(paste0("sar_", m)), error = function(e) NULL)
  if (is.null(fn)) next
  tryCatch({
    fit <- fn(data = galap)
    results[[m]] <- data.frame(
      model     = m,
      converged = isTRUE(fit$converged),
      r2        = fit$R2,
      aic       = fit$AIC,
      aicc      = fit$AICc,
      bic       = fit$BIC,
      params    = paste(names(fit$par), round(fit$par, 6),
                        sep = "=", collapse = "; ")
    )
  }, error = function(e) {
    results[[m]] <<- data.frame(model=m, converged=FALSE,
                                r2=NA, aic=NA, aicc=NA, bic=NA, params=NA)
  })
}

df <- do.call(rbind, results)
write.csv(df, "tests/r_reference/all_models_galap.csv", row.names = FALSE)
cat("Reference table written.\n\nPower law:\n")
print(sar_power(galap))
