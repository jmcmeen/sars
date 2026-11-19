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

# 5. The log-log power model for each log base, on galap and on galap with its
#    smallest island emptied (so that con = 1 is added to every richness).
#    power_c and power_z are the nonlinear power fit lin_pow returns with
#    compare = TRUE, which is to the richness as given.
galap_zero <- galap
galap_zero$s[1] <- 0
lin_pow_ref <- function(data, name, base) {
  set.seed(SEED)
  fit <- suppressWarnings(lin_pow(data, con = 1, logT = get(base), compare = TRUE))
  co <- fit$Model$coefficients
  data.frame(data = name, log_base = if (base == "log") "e" else sub("log", "", base),
             logc = co[1, 1], z = co[2, 1], logc_se = co[1, 2], z_se = co[2, 2],
             logc_t = co[1, 3], z_t = co[2, 3], logc_p = co[1, 4], z_p = co[2, 4],
             r2 = fit$Model$r.squared, adj_r2 = fit$Model$adj.r.squared,
             sigma = fit$Model$sigma,
             power_c = fit$power$par[["c"]], power_z = fit$power$par[["z"]])
}
lin_pow_rows <- list()
for (base in c("log", "log2", "log10")) {
  lin_pow_rows[[length(lin_pow_rows) + 1]] <- lin_pow_ref(galap, "galap", base)
  lin_pow_rows[[length(lin_pow_rows) + 1]] <- lin_pow_ref(galap_zero, "galap_zero", base)
}
lin_pow_tab <- do.call(rbind, lin_pow_rows)
num <- sapply(lin_pow_tab, is.numeric)
lin_pow_tab[num] <- lapply(lin_pow_tab[num], signif, digits = 12)
write.csv(lin_pow_tab, out("lin_pow_galap.csv"), row.names = FALSE)

# 6. sar_pred: predicted richness at three areas (one inside the data's range,
#    two beyond it) from single fits, and from the average of the same three
#    models. An averaged prediction depends on the models averaged.
pred_areas <- c(50, 500, 10000)
pred_models <- c("power", "loga", "negexpo")
pred_rows <- lapply(pred_models, function(m) {
  set.seed(SEED)
  p <- sar_pred(suppressWarnings(get(paste0("sar_", m))(data = galap)), area = pred_areas)
  data.frame(model = m, area = p$Area, prediction = p$Prediction)
})
set.seed(SEED)
pred_av <- sar_pred(suppressWarnings(sar_average(obj = pred_models, data = galap,
                                                 verb = FALSE, display = FALSE)),
                    area = pred_areas)
pred_rows[[length(pred_rows) + 1]] <- data.frame(model = "average", area = pred_av$Area,
                                                 prediction = pred_av$Prediction)
write.csv(do.call(rbind, pred_rows), out("pred_galap.csv"), row.names = FALSE)
# 7. Coleman's random placement model on R's simulated abundance matrix
#    (22 species in rows, 11 sites in columns). The matrix and site areas are
#    written out as the test input; R's expected richness, its standard
#    deviation, the relative areas and the observed richness are the reference.
data(cole_sim)
write.csv(cole_sim[[1]], out("coleman_abundance.csv"), row.names = FALSE)
cole <- coleman(cole_sim[[1]], cole_sim[[2]])
write.csv(data.frame(site = names(cole_sim[[2]]),
                     area = unname(cole_sim[[2]]),
                     relative_area = unname(cole$Relative_areas),
                     observed = unname(cole$Species_richness),
                     expected = cole$Predicted_values,
                     sd = cole$Standard_deviation),
          out("coleman_sites.csv"), row.names = FALSE)
# 8. The general dynamic model (gdm), on the example from its help page:
#    galap with an island age column. Each of the four forms is fitted with
#    mod_sel = TRUE, which returns the GDM and its three simpler candidates
#    (area + time, area, intercept) in that order. The fits are nls or lm
#    objects, so coefficients, standard errors, p values, RSS and AIC are read
#    from them; AICc is AIC plus the small-sample term with k = coefficients
#    + 1, which reproduces the table gdm prints. The intercept-only candidate
#    is an lm of richness, so its Int is the mean richness for every form.
galap_t <- galap
galap_t$t <- c(4, 1, 13, 16, 15, 2, 6, 4, 5, 11, 3, 9, 8, 10, 12, 7)
gdm_forms <- c("loga", "linear", "power_area", "power_area_time")
gdm_candidates <- c("GDM", "A + T", "A", "Intercept")
gdm_pars <- c("Int", "A", "Ti", "Ti2")
gdm_rows <- list()
# The ages are kept beside R's fitted values, so the tests read them from here.
gdm_fitted <- data.frame(area = galap_t$a, species = galap_t$s, time = galap_t$t)
tss <- sum((galap_t$s - mean(galap_t$s))^2)
for (m in gdm_forms) {
  g <- gdm(galap_t, model = m, mod_sel = TRUE)
  for (i in seq_along(gdm_candidates)) {
    f <- g[[i]]
    cf <- coef(summary(f))
    rownames(cf)[rownames(cf) == "(Intercept)"] <- "Int"
    k <- nrow(cf) + 1
    rss <- sum(residuals(f)^2)
    row <- data.frame(model = m, candidate = gdm_candidates[i], k = k, rss = rss,
                      r2 = 1 - rss / tss, aic = AIC(f),
                      aicc = AIC(f) + 2 * k * (k + 1) / (n - k - 1), bic = BIC(f))
    for (p in gdm_pars) {
      has <- p %in% rownames(cf)
      row[[p]] <- if (has) cf[p, "Estimate"] else NA
      row[[paste0("se_", p)]] <- if (has) cf[p, "Std. Error"] else NA
      row[[paste0("p_", p)]] <- if (has) cf[p, "Pr(>|t|)"] else NA
    }
    gdm_rows[[length(gdm_rows) + 1]] <- row
  }
  gdm_fitted[[m]] <- as.numeric(fitted(g[[1]]))
}
write.csv(do.call(rbind, gdm_rows), out("gdm_galap.csv"), row.names = FALSE)
write.csv(gdm_fitted, out("gdm_fitted_galap.csv"), row.names = FALSE)
# model = "all" ranks the four forms; its printed table is kept as the record
# that the ranking below comes from the same fits.
print(gdm(galap_t, model = "all", mod_sel = FALSE))

cat("Reference tables written.\n\nPower law:\n")
print(sar_power(galap))
