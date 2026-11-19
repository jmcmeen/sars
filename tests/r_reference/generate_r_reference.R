# Rscript tests/r_reference/generate_r_reference.R
#
# Writes the R sars reference values the tests compare against. Run from the
# repository root. The R sars version used is recorded in r_version.txt.
suppressMessages(library(sars))
out <- function(name) file.path("tests/r_reference", name)

# R's default grid_start = "partial" adds random starting values, so each fit
# is seeded on its own: the tables are reproducible, and adding a model
# doesn't change the others.
SEED <- 1

writeLines(
  c(paste("sars", as.character(packageVersion("sars"))), R.version.string),
  out("r_version.txt")
)

# 1. Export the exact galap dataset (16 rows), and a seven-island subset (rows
#    1, 3, ..., 13 of the area-sorted data, 0.52 to 505 km2) for small-sample checks.
write.csv(galap, out("galap.csv"), row.names = FALSE)
galap7 <- galap[seq(1, 13, by = 2), ]
write.csv(galap7, out("galap7.csv"), row.names = FALSE)
cat("Exported galap:", nrow(galap), "rows; galap7:", nrow(galap7), "rows\n")

# 2. Fit all 20 models and export reference values
models <- c("power","powerR","epm1","epm2","p1","p2","loga","koba",
            "mmf","monod","negexpo","chapman","weibull3","asymp",
            "ratio","gompertz","weibull4","betap","heleg","linear")

fit_all <- function(data, stem) {
  results <- list()
  fitted <- data.frame(area = data$a)
  for (m in models) {
    fn <- tryCatch(get(paste0("sar_", m)), error = function(e) NULL)
    if (is.null(fn)) next
    tryCatch({
      set.seed(SEED)
      fit <- suppressWarnings(fn(data = data))
      results[[m]] <- data.frame(
        model     = m,
        converged = isTRUE(fit$converged),
        r2        = fit$R2,
        aic       = fit$AIC,
        aicc      = fit$AICc,
        bic       = fit$BIC,
        params    = paste(names(fit$par), signif(fit$par, 8),
                          sep = "=", collapse = "; ")
      )
      fitted[[m]] <- fit$calculated
    }, error = function(e) {
      results[[m]] <<- data.frame(model=m, converged=FALSE,
                                  r2=NA, aic=NA, aicc=NA, bic=NA, params=NA)
    })
  }
  write.csv(do.call(rbind, results), out(paste0("all_models_", stem, ".csv")),
            row.names = FALSE)
  # Each model's fitted values at the data's areas, so averaging can be
  # checked from R's own fits.
  write.csv(fitted, out(paste0("fitted_", stem, ".csv")), row.names = FALSE)
}

fit_all(galap, "galap")
fit_all(galap7, "galap7")

# 3. Multimodel weights. R's sar_average no longer offers
#    mmf (and Python has no logistic), so both use the 19 models they share.
#    Plain AICc weights: no residual normality, homogeneity or negative-value
#    checks (the R defaults).
shared <- setdiff(models, "mmf")
average_ref <- function(data, stem) {
  set.seed(SEED)
  av <- suppressWarnings(sar_average(obj = shared, data = data,
                                     verb = FALSE, display = FALSE))
  d <- av$details
  write.csv(data.frame(model = names(d$weights_ics), aicc = d$ics[names(d$weights_ics)],
                       weight = d$weights_ics),
            out(paste0("average_", stem, ".csv")), row.names = FALSE)
}
average_ref(galap, "galap")
average_ref(galap7, "galap7")

# 4. Threshold models on ln(area). nisl = 3 keeps three islands in the first
#    and last segments, the same window Python searches; interval = 0.01 is
#    R's default step on the log scale. The two-threshold continuous searches
#    take about a minute.
th <- sar_threshold(galap, mod = c("ContOne", "ZslopeOne", "ContTwo", "ZslopeTwo"),
                    non_th_models = TRUE, interval = 0.01, nisl = 3)
# R's search for the discontinuous models doesn't keep to nisl: DiscOne scores
# each candidate breakpoint as the one nisl - 1 places lower (it indexes x
# where it means the window), and both let the last segment hold nisl - 1
# islands. So their breakpoints are searched here, over the observed areas
# with three islands in the first and last segments, scoring each with R's
# own model function, and the models refitted as sar_threshold fits them.
x <- sort(log(galap$a))
y <- galap$s[order(galap$a)]
cand <- unique(x)
n_left <- sapply(cand, function(t) sum(x <= t))
first_ok <- n_left >= 3
last_ok <- length(x) - n_left >= 3
one <- cand[first_ok & last_ok]
disc_one <- one[which.min(sapply(one, sars:::fct_disc_one, x = x, y = y))]
pairs <- expand.grid(t1 = cand[first_ok], t2 = cand[last_ok])
pairs <- pairs[pairs$t1 < pairs$t2, ]
rss2 <- mapply(sars:::fct_disc_two, pairs$t1, pairs$t2, MoreArgs = list(x = x, y = y))
disc_two <- unlist(pairs[which.min(rss2), ])
fits <- c(th[[1]], list(
  lm(y ~ x * (x <= disc_one) + x * (x > disc_one)),
  lm(y ~ x * (x <= disc_two[1]) + x * (x > disc_two[1] & x <= disc_two[2]) +
       x * (x > disc_two[2]))
))
names_th <- c(th[[2]], "DiscOne", "DiscTwo")
ths <- c(th[[3]], list(disc_one, disc_two))
# Parameters as R's summary.sars counts them (each coefficient, threshold and
# sigma); RSS from each lm, since summary() rounds its table.
k <- c(ContOne = 5, ZslopeOne = 4, DiscOne = 6, ContTwo = 7, ZslopeTwo = 6,
       DiscTwo = 9, Linear = 3, Intercept = 2)
pars <- k[names_th]
rss <- sapply(fits, function(m) sum(residuals(m)^2))
n <- nrow(galap)
ll <- -n / 2 * (log(2 * pi * rss / n) + 1)
aic <- -2 * ll + 2 * pars
write.csv(data.frame(model = names_th,
                     threshold = sapply(ths, function(t) t[1]),
                     threshold2 = sapply(ths, function(t) if (length(t) > 1) t[2] else NA),
                     k = pars, rss = rss, aic = aic,
                     aicc = aic + 2 * pars * (pars + 1) / (n - pars - 1),
                     r2 = 1 - rss / sum((galap$s - mean(galap$s))^2)),
          out("threshold_galap.csv"), row.names = FALSE)

cat("Reference tables written.\n\nPower law:\n")
print(sar_power(galap))
