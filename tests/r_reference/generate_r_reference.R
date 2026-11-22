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
  tables <- list()
  fitted <- data.frame(area = data$a)
  for (m in models) {
    fn <- tryCatch(get(paste0("sar_", m)), error = function(e) NULL)
    if (is.null(fn)) next
    fit <- NULL
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
      fit <<- NULL
    })
    # The parameter table: sigConf's first four columns (its rows aren't always
    # named, so the names come from the parameters). sigConf is NA where R
    # couldn't compute it; those models have no rows. Guarded on its own, so
    # that a table of some other shape can't mark a good fit as failed above.
    tryCatch({
      sc <- fit$sigConf
      if (is.matrix(sc)) {
        tables[[m]] <- data.frame(
          model     = m,
          param     = names(fit$par),
          estimate  = sc[, "Estimate"],
          std_error = sc[, "Std. Error"],
          t_value   = sc[, "t value"],
          p_value   = sc[, "Pr(>|t|)"]
        )
      }
    }, error = function(e) {
      cat("No parameter table for", m, "on", stem, ":", conditionMessage(e), "\n")
    })
  }
  write.csv(do.call(rbind, results), out(paste0("all_models_", stem, ".csv")),
            row.names = FALSE)
  # Each model's fitted values at the data's areas, so averaging can be
  # checked from R's own fits.
  write.csv(fitted, out(paste0("fitted_", stem, ".csv")), row.names = FALSE)
  # Each parameter's standard error, t value and p value.
  write.csv(do.call(rbind, tables), out(paste0("param_table_", stem, ".csv")),
            row.names = FALSE)
}

fit_all(galap, "galap")
fit_all(galap7, "galap7")

# 3. Multimodel weights. R's sar_average no longer offers
#    mmf (and Python has no logistic), so both use the 19 models they share.
#    Plain AICc weights: no residual normality, homogeneity or negative-value
#    checks (the R defaults; section 10 has them).
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
# 9. The other bundled datasets, as R ships them, and the default threshold
#    models on each (same search and parameter counts as section 4): these
#    are the datasets thresholds are usually shown on, and far larger than
#    galap. The intercept-only model R adds isn't one this package fits.
other <- list(aegean = aegean, aegean2 = aegean2, niering = niering)
th_other <- list()
for (nm in names(other)) {
  d <- other[[nm]]
  write.csv(d, out(paste0(nm, ".csv")), row.names = FALSE)
  th_d <- sar_threshold(d, mod = c("ContOne", "ZslopeOne"), non_th_models = TRUE,
                        interval = 0.01, nisl = 3)
  keep <- th_d[[2]] != "Intercept"
  pars_d <- k[th_d[[2]][keep]]
  rss_d <- sapply(th_d[[1]][keep], function(m) sum(residuals(m)^2))
  n_d <- nrow(d)
  aic_d <- n_d * (log(2 * pi * rss_d / n_d) + 1) + 2 * pars_d
  th_other[[nm]] <- data.frame(
    dataset = nm, model = th_d[[2]][keep],
    threshold = sapply(th_d[[3]][keep], function(t) t[1]),
    k = pars_d, rss = rss_d, aic = aic_d,
    aicc = aic_d + 2 * pars_d * (pars_d + 1) / (n_d - pars_d - 1)
  )
}
write.csv(do.call(rbind, th_other), out("threshold_datasets.csv"), row.names = FALSE)
# 10. Residual checks. Each fit stores the tests asked for (normaTest, homoTest,
#    homoCor) and whether any fitted value is negative (neg_check). Three
#    seeded fits per model, the same fit each time, cover Shapiro-Wilk and the
#    three correlations. R's "kolmo" compares the residuals with a standard
#    normal, which Python doesn't offer; the Kolmogorov-Smirnov reference is
#    base R's ks.test on the fit's residuals against a normal with their mean
#    and standard deviation. kept is whether sar_average keeps the model when
#    asked for the Shapiro-Wilk test, the Spearman correlation with the fitted
#    values and the negative-value check (NA for mmf, which it doesn't offer).
#    sar_average fits the models itself, one seed for the whole set, so where
#    R's fit of a model varies with the random starts (betap and chapman on
#    the seven islands) kept can disagree with that model's row.
checks_ref <- function(data, stem) {
  rows <- list()
  for (m in models) {
    fn <- get(paste0("sar_", m))
    fit_with <- function(...) {
      set.seed(SEED)
      suppressWarnings(fn(data = data, normaTest = "shapiro", ...))
    }
    a <- fit_with(homoTest = "cor.fitted", homoCor = "spearman")
    b <- fit_with(homoTest = "cor.area", homoCor = "kendall")
    p <- fit_with(homoTest = "cor.fitted", homoCor = "pearson")
    res <- a$residuals
    ks <- ks.test(res, "pnorm", mean(res), sd(res))
    rows[[m]] <- data.frame(
      model = m,
      shapiro_w = a$normaTest[[2]]$statistic, shapiro_p = a$normaTest[[2]]$p.value,
      ks_d = ks$statistic, ks_p = ks$p.value,
      fitted_spearman = a$homoTest[[2]]$estimate,
      fitted_spearman_p = a$homoTest[[2]]$p.value,
      area_kendall = b$homoTest[[2]]$estimate,
      area_kendall_p = b$homoTest[[2]]$p.value,
      fitted_pearson = p$homoTest[[2]]$estimate,
      fitted_pearson_p = p$homoTest[[2]]$p.value,
      neg_check = a$neg_check
    )
  }
  tab <- do.call(rbind, rows)
  set.seed(SEED)
  av <- suppressWarnings(sar_average(obj = shared, data = data, normaTest = "shapiro",
                                     homoTest = "cor.fitted", homoCor = "spearman",
                                     neg_check = TRUE, verb = FALSE, display = FALSE))
  tab$kept <- ifelse(tab$model %in% shared, tab$model %in% names(av$details$mod_names), NA)
  write.csv(tab, out(paste0("residual_checks_", stem, ".csv")), row.names = FALSE)
}
checks_ref(galap, "galap")
checks_ref(galap7, "galap7")

# 11. Correlation tests away from the SAR fits: base R's cor.test of 1..n with
#    each y. For Spearman's rho the p value is exact up to n = 9 and a series
#    beyond; ties switch both tests to their large-sample approximations.
cor_cases <- list(
  spearman = list(c(2, 1, 4, 3, 5), c(5, 4, 3, 1, 2), c(3, 5, 1, 2, 4),
                  c(1, 3, 2, 5, 4, 7, 6, 9, 8), c(9, 7, 8, 4, 6, 5, 1, 3, 2),
                  c(3, 1, 4, 2, 6, 5, 9, 7, 10, 8),
                  c(12, 10, 11, 7, 9, 8, 4, 6, 5, 1, 3, 2),
                  c(7, 2, 11, 4, 16, 1, 9, 13, 5, 20, 3, 15, 8, 18, 6, 12, 10, 19, 14, 17),
                  c(1, 1, 2, 4, 3, 6, 5)),
  kendall = list(c(2, 1, 4, 3, 5), c(3, 1, 4, 2, 6, 5, 9, 7, 10, 8),
                 c(1, 1, 2, 4, 3, 6, 5, 8))
)
cor_rows <- list()
for (method in names(cor_cases)) for (y in cor_cases[[method]]) {
  ct <- suppressWarnings(cor.test(seq_along(y), y, method = method))
  cor_rows[[length(cor_rows) + 1]] <- data.frame(
    method = method, y = paste(y, collapse = " "),
    estimate = unname(ct$estimate), p = ct$p.value)
}
write.csv(do.call(rbind, cor_rows), out("correlation_tests.csv"), row.names = FALSE)

# 12. Kolmogorov-Smirnov tests against a normal with the sample's mean and
#    standard deviation, for x = sin(i) + (i mod 7) / 3, i = 1..n. ks.test's p
#    value is exact below 100 untied observations and asymptotic otherwise;
#    tied makes the first two values equal.
ks_rows <- list()
for (n in c(20, 99, 100, 150)) for (tied in c(FALSE, TRUE)) {
  i <- seq_len(n)
  x <- sin(i) + (i %% 7) / 3
  if (tied) x[2] <- x[1]
  ks <- suppressWarnings(ks.test(x, "pnorm", mean(x), sd(x)))
  ks_rows[[length(ks_rows) + 1]] <- data.frame(
    n = n, tied = tied, d = unname(ks$statistic), p = ks$p.value,
    exact = isTRUE(ks$exact))
}
write.csv(do.call(rbind, ks_rows), out("ks_tests.csv"), row.names = FALSE)

# 13. Breakpoint intervals from the inverted F test (threshold_ci, method =
#    "F") for the one-threshold continuous models on the bundled datasets, on
#    ln(area) in steps of 0.01. R searches every grid point from the smallest
#    area to the largest, and compares the F statistic with F(cl; 1, p), p the
#    number of regression coefficients; see test_r_reference.py.
f_rows <- list()
for (nm in c("galap", "aegean", "aegean2", "niering")) {
  d <- get(nm)
  f <- sar_threshold(data = d, mod = c("ContOne", "ZslopeOne"), interval = 0.01,
                     nisl = 3, non_th_models = FALSE, logAxes = "area", logT = log)
  for (cl in c(0.95, 0.9)) {
    ci <- threshold_ci(f, cl = cl, method = "F", interval = NULL)
    for (mod in c("ContOne", "ZslopeOne")) {
      f_rows[[length(f_rows) + 1]] <- data.frame(dataset = nm, cl = cl, model = mod,
                                                 lower = ci[[mod]][1], upper = ci[[mod]][2])
    }
  }
}
write.csv(do.call(rbind, f_rows), out("threshold_ci_F.csv"), row.names = FALSE)

# 14. Lilliefors tests (nortest::lillie.test, which R sars's normaTest =
#    "lillie" calls) on fixed samples: x = sin(i) + (i mod 7) / 3, near
#    normal, and y = exp(sin(1.7 i)), skewed, i = 1..n. Its p value is Dallal
#    and Wilkinson's approximation below 0.1, nortest's own fit above. Then
#    the test on each R sars fit's residuals for the Galapagos data.
lillie_rows <- list()
for (n in c(5, 6, 8, 12, 20, 26, 50, 100, 101, 200)) {
  i <- seq_len(n)
  for (shape in c("sin", "exp")) {
    x <- if (shape == "sin") sin(i) + (i %% 7) / 3 else exp(sin(1.7 * i))
    lt <- nortest::lillie.test(x)
    lillie_rows[[length(lillie_rows) + 1]] <- data.frame(
      n = n, shape = shape, d = unname(lt$statistic), p = lt$p.value)
  }
}
write.csv(do.call(rbind, lillie_rows), out("lillie_tests.csv"), row.names = FALSE)

lillie_fits <- list()
for (m in models) {
  fn <- get(paste0("sar_", m))
  set.seed(SEED)
  fit <- tryCatch(suppressWarnings(fn(data = galap, normaTest = "lillie")),
                  error = function(e) NULL)
  if (is.null(fit) || length(fit$normaTest) < 2) next
  lillie_fits[[length(lillie_fits) + 1]] <- data.frame(
    model = m, d = unname(fit$normaTest[[2]]$statistic), p = fit$normaTest[[2]]$p.value)
}
write.csv(do.call(rbind, lillie_fits), out("lillie_galap.csv"), row.names = FALSE)

# 15. Habitat models (sar_habitat) on R sars's habitat data (Skyros land
#    snails: area, number of habitats, species), in each form: every model's
#    parameters, R2, AIC, BIC and AICc, unrounded. Then the countryside model
#    (sar_countryside) on its countryside data, power and logarithmic: each
#    species group's parameters and RSS, its affinities and c, the predicted
#    total richness of every site, and the countryside and power-law RSS.
data(habitat)
data(countryside)
hab_rows <- list()
for (mt in c("power", "logarithmic", "power_log")) {
  set.seed(SEED)
  fits <- suppressMessages(sar_habitat(habitat, modType = mt))
  for (m in names(fits)) {
    f <- fits[[m]]
    co <- if (inherits(f, "lm")) coef(f) else f$m$getPars()
    ic <- sars:::extr_fit(f)
    hab_rows[[length(hab_rows) + 1]] <- data.frame(
      modType = mt, model = m, p1 = unname(co[1]), p2 = unname(co[2]),
      p3 = if (length(co) > 2) unname(co[3]) else NA,
      rss = sum(residuals(f)^2), R2 = ic[1], adjR2 = ic[2], AIC = ic[3], BIC = ic[4],
      AICc = ic[5])
  }
}
write.csv(do.call(rbind, hab_rows), out("habitat_models.csv"), row.names = FALSE)

cs_rows <- list()
cs_tot <- data.frame(site = rownames(countryside))
cs_rss <- list()
for (mt in c("power", "logarithmic")) {
  set.seed(SEED)
  cs <- suppressWarnings(sar_countryside(countryside, modType = mt,
                                         habNam = 1:3, spNam = 4:7))
  for (g in names(cs$fits)) {
    f <- cs$fits[[g]]
    p <- f$m$getPars()
    cs_rows[[length(cs_rows) + 1]] <- data.frame(
      modType = mt, group = g, h1 = p[["h1"]], h2 = p[["h2"]], h3 = p[["h3"]], z = p[["z"]],
      rss = sum(residuals(f)^2), aff1 = cs$affinity[[g]][1], aff2 = cs$affinity[[g]][2],
      aff3 = cs$affinity[[g]][3], c = unname(cs$c[paste0(g, ".c")]))
  }
  cs_tot[[mt]] <- cs$Pred.Tot.Rich
  cs_rss[[length(cs_rss) + 1]] <- data.frame(
    modType = mt, countryside_rss = unname(cs$rss[1]), sar_rss = unname(cs$rss[2]))
}
write.csv(do.call(rbind, cs_rows), out("countryside_fits.csv"), row.names = FALSE)
write.csv(cs_tot, out("countryside_total.csv"), row.names = FALSE)
write.csv(do.call(rbind, cs_rss), out("countryside_rss.csv"), row.names = FALSE)

# 16. Richness standardised for sampling effort, by iNEXT's estimateD (q = 0,
#    no bootstrap) on a simulated species-by-site table (effort_abundance.csv,
#    written by the tests' authors: ten sites of growing size and effort, and
#    two small ones without doubletons or singletons), by coverage and by size,
#    at iNEXT's default level and at fixed ones; and vegan's rarefy at a size
#    every site reaches. Needs iNEXT and vegan; skipped without them.
if (requireNamespace("iNEXT", quietly = TRUE) && requireNamespace("vegan", quietly = TRUE)) {
  ab <- read.csv(out("effort_abundance.csv"), row.names = 1, check.names = FALSE)
  eff_rows <- list()
  for (base in c("coverage", "size")) {
    levels <- if (base == "coverage") list(NULL, 0.6, 0.9, 0.99) else list(NULL, 20, 500, 5000)
    for (lv in levels) {
      est <- suppressWarnings(iNEXT::estimateD(ab, q = 0, datatype = "abundance", base = base,
                                               level = lv, nboot = 0))
      eff_rows[[length(eff_rows) + 1]] <- data.frame(
        base = base, level = if (is.null(lv)) NA else lv, site = est$Assemblage,
        m = est$m, method = est$Method, SC = est$SC, qD = est$qD)
    }
  }
  write.csv(do.call(rbind, eff_rows), out("effort_inext.csv"), row.names = FALSE)
  rare <- vegan::rarefy(t(ab), sample = 10)
  write.csv(data.frame(site = names(rare), rarefied10 = unname(rare)),
            out("effort_vegan.csv"), row.names = FALSE)
  writeLines(c(paste("iNEXT", as.character(packageVersion("iNEXT"))),
               paste("vegan", as.character(packageVersion("vegan"))), R.version.string),
             out("effort_versions.txt"))
}

# 17. The log-log power model fitted to six archipelagos at once, with
#    random intercepts and slopes, by lme4's lmer (REML). mixed_data.csv holds
#    the Python package's bundled gala, aegean, aegean2, niering, schrader and
#    xu data, three of which R sars doesn't have; make_mixed_data.py writes it
#    (run it first). One is added to every richness value, as Schrader's has
#    zeros. Needs lme4; skipped without it.
if (requireNamespace("lme4", quietly = TRUE) && file.exists(out("mixed_data.csv"))) {
  md <- read.csv(out("mixed_data.csv"))
  md$S <- md$species + md$con
  mf <- lme4::lmer(log(S) ~ log(area) + (log(area) | group), data = md, REML = TRUE,
                   control = lme4::lmerControl(optimizer = "bobyqa"))
  vc <- as.data.frame(lme4::VarCorr(mf))
  se <- sqrt(diag(as.matrix(vcov(mf))))
  write.csv(data.frame(
    logc = lme4::fixef(mf)[[1]], z = lme4::fixef(mf)[[2]], logc_se = se[[1]], z_se = se[[2]],
    sd_logc = vc$sdcor[1], sd_z = vc$sdcor[2], corr = vc$sdcor[3], sigma = vc$sdcor[4],
    reml = lme4::REMLcrit(mf)), out("mixed_fit.csv"), row.names = FALSE)
  cf <- coef(mf)$group
  write.csv(data.frame(group = rownames(cf), logc = cf[[1]], z = cf[[2]]),
            out("mixed_groups.csv"), row.names = FALSE)
  writeLines(c(paste("lme4", as.character(packageVersion("lme4"))), R.version.string),
             out("mixed_versions.txt"))
}

cat("Reference tables written.\n\nPower law:\n")
print(sar_power(galap))
