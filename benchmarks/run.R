# R sars timings for benchmarks/run.py, which writes the datasets and calls:
#
#   Rscript benchmarks/run.R <jobs.csv> <out.csv> <repeats> <n_boot>
#
# jobs.csv lists one job per row (task, dataset, path to the dataset's CSV);
# out.csv gets one row per job: its median time in seconds over `repeats` runs
# (one run for bootstraps), and a result to compare with Python's.

suppressPackageStartupMessages(library(sars))

args <- commandArgs(trailingOnly = TRUE)
jobs <- read.csv(args[[1]], stringsAsFactors = FALSE)
out_path <- args[[2]]
repeats <- as.integer(args[[3]])
n_boot <- as.integer(args[[4]])

quietly <- function(expr) {
  suppressWarnings(suppressMessages({
    capture.output(value <- force(expr))
    value
  }))
}

run_job <- function(task, data) {
  switch(task,
    power = {
      fit <- sar_power(data)
      sprintf("z=%.4f", fit$par[["z"]])
    },
    average = {
      fit <- sar_average(data = data, verb = FALSE, display = FALSE)
      w <- fit$details$weights_ics
      sprintf("best=%s", names(w)[which.max(w)])
    },
    bootstrap = {
      fit <- sar_average(data = data, confInt = TRUE, ciN = n_boot, verb = FALSE, display = FALSE)
      sprintf("n_boot=%d", n_boot)
    },
    threshold = {
      fit <- sar_threshold(data, mod = c("ContOne", "ZslopeOne"), non_th_models = TRUE)
      table <- summary(fit)$Model_table
      table <- table[rownames(table) != "Intercept", ]  # Python fits no intercept-only model
      sprintf("best=%s", rownames(table)[which.min(table$AICc)])
    },
    stop("unknown task ", task)
  )
}

rows <- list()
for (i in seq_len(nrow(jobs))) {
  job <- jobs[i, ]
  data <- read.csv(job$path)
  runs <- if (job$task == "bootstrap") 1L else repeats
  if (job$task != "bootstrap") quietly(run_job(job$task, data))  # warm-up (a bootstrap is long enough without)
  times <- numeric(0)
  for (r in seq_len(runs)) {
    started <- proc.time()[["elapsed"]]
    result <- quietly(run_job(job$task, data))
    times <- c(times, proc.time()[["elapsed"]] - started)
  }
  rows[[i]] <- data.frame(
    language = "R", task = job$task, dataset = job$dataset, n = nrow(data),
    seconds = median(times), runs = runs, result = result
  )
  message(sprintf("R %s %s: %.3f s", job$task, job$dataset, median(times)))
}
write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
