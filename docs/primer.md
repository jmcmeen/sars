# A primer on species-area relationships

This page is for readers new to species-area relationships (SARs): what they are, why
their shape matters, and how the tools in `sars` fit together. It gives the ideas and the
cautions; the [Models](models.md) page has the formulas, and
[Getting Started](getting-started.md) the code. The works cited are listed in the
[README](https://github.com/jmcmeen/sars#references).

## What a species-area relationship is

Larger areas hold more species. Few patterns in ecology are as general: it holds for
plants and animals, on oceanic islands, on habitat islands such as lakes, mountaintops and
forest fragments, and in nested plots on a continent. A species-area relationship is the
curve describing how the number of species, S, rises with area, A.

Two kinds of SAR are worth keeping apart, because their curves need not have the same
shape (Matthews et al. 2016; Triantis et al. 2012):

- **Island SARs** (ISARs): one point per island, or per habitat island, each with its own
  area and species count. The bundled datasets (`sars datasets`) are of this kind.
- **Sample-area curves**: nested or accumulating samples within one region, where each
  larger area contains the smaller ones.

The models fitted here apply to both, but how a curve should be read differs: an island
curve mixes area with everything islands of different size differ in (habitat diversity,
isolation, age), while a nested curve mostly reflects how species are spread through one
landscape.

## The power model and its exponent

The best-known SAR is the power model of Arrhenius (1921), S = c A^z. On log axes it is a
straight line, log S = log c + z log A, with slope z: doubling the area multiplies the
species count by 2^z. Values of z for island data typically fall around 0.2 to 0.4, and
much of the classic literature compares z across groups and regions.

There are two ways to fit it, and they give different answers:

- **Log-log least squares** (`lin_pow`, `sars loglog`) fits the straight line on log
  axes. It minimizes error in log S, so it weights proportional errors equally on small
  and large islands, and it gives standard errors and p values for z directly.
- **Nonlinear least squares** (`sar_power`, `sars fit`) fits S = c A^z on the original
  scale, minimizing error in S itself, so the large, species-rich islands carry more
  weight.

On the bundled Galápagos plant data (`galap`), the log-log fit gives z = 0.339 and the
nonlinear fit z = 0.283. Neither is wrong; they answer slightly different questions, and
`sars loglog --compare` shows both.

## Shapes: why more than one model

The power model never levels off, and on log axes it never bends. Real data often do one
or both, and the many SAR models describe the possibilities (Tjørve 2003, 2009):

- **Convex** curves rise ever more slowly but without limit (power, logarithmic).
- **Asymptotic** curves approach a ceiling, the number of species the region can supply
  (negative exponential, Monod, rational).
- **Sigmoid** curves rise slowly at first, then quickly, then slowly again (Gompertz,
  Weibull, Chapman-Richards, beta-P).

The shape matters most when you extrapolate. Two models can fit the observed islands
almost equally well and still disagree wildly about a larger area, or about how many
species a smaller remnant keeps. That is the main reason to fit several models rather than
choose one in advance.

## Choosing among models

`sars` fits 20 models, as the R package does (Matthews et al. 2019), and offers three
complementary ways to compare them.

- **Information criteria.** AICc (and AIC, BIC) balance fit against the number of
  parameters. Akaike weights turn the differences into each model's share of the evidence
  in the set (`sars fit -m all`). On `galap` the negative exponential leads with about 20%
  of the weight, closely followed by Monod and Kobayashi: no single model is clearly best,
  which is typical.
- **Model averaging.** When several models share the weight, an average of their
  predictions, weighted by the Akaike weights (`sar_average`, `sars average`), is usually
  more honest than any one of them (Guilhaumon et al. 2010).
- **Cross-validation.** Leaving out each island in turn and predicting it from the rest
  (`loo_cv`, `sars cv`) asks directly how well a model predicts, rather than how well it
  fits; it penalizes flexible models that chase individual points.

Two checks help keep the comparison fair. **Residual checks** (normality,
homoscedasticity, negative predictions; `--normality`, `--homoscedasticity`,
`--neg-check`) leave out models whose assumptions the data contradict. And a model's
**parameters** deserve a look (`sars fit --params`): a parameter whose standard error
dwarfs its estimate, or that the data can't identify (`NaN`), warns that the fit is
leaning on a corner of the model rather than describing the data.

## Uncertainty

A fitted curve is an estimate, and the data are a sample. `sars` gives uncertainty in
three ways:

- **Parameter tables**: standard errors, t tests and confidence intervals for each
  parameter (`param_table`, `sars fit --params`).
- **Bootstrap intervals** on the averaged curve (`bootstrap_ci`, `sars ci`): the islands
  are resampled, every model refitted and averaged, and the spread of the averaged
  predictions gives the interval. The *fast* method refits each resample from the original
  fits; the *full* method searches every model's start grid again, as R does, and is the
  like-for-like choice when comparing with R.
- **Breakpoint intervals** for threshold models (`threshold_ci`, `sars threshold --ci`).

## Thresholds and the small-island effect

On small islands, richness often varies with area less than the large islands suggest, or
not at all: the **small-island effect**. Chance, disturbance and habitat dominate there,
and the curve flattens below some threshold area (Matthews & Rigal 2021). Piecewise
(threshold) models fit separate segments on either side of one or two breakpoints
(`sar_threshold`, `sars threshold`), and compare them with a single line by AICc
(Toms & Lesperance 2003). On the 30 Galápagos islands of `gala`, a model that is flat
below a breakpoint and rises above it (ZslopeOne) is chosen, with the breakpoint at about 11 km² (a 95% bootstrap interval
of roughly 3 to 43 km²). A wide interval like that is common: thresholds are hard to pin
down with a few dozen islands.

## Habitat loss, with care

Turning the curve around gives the expected number of species when an area shrinks: if a
region loses half its habitat, the curve says how many species a region of half the size
holds (`habitat_loss`, `sars loss`). This is widely used and widely criticized, and the
cautions are worth stating:

- It predicts an **eventual** loss, once the community has relaxed to its new area, not
  how many species disappear at once; the delay can be long (an "extinction debt").
- It tends to **overestimate** losses, because the species-area curve of whole islands is
  not the curve of species unique to the lost area.
- It inherits the curve's **uncertainty and shape**, so `sars loss` averages the models
  and gives bootstrap intervals rather than one model's point estimate.

## Sampling effort

Species counts depend on how hard each site was searched as well as on its area. When
effort varies, larger sites may simply have been sampled more. Standardizing richness to
equal sample coverage, or equal sample size, before fitting (`standardised_richness`, as
iNEXT does) separates the area effect from the effort effect.

## Beyond area

Area is rarely the whole story, and some models add more:

- **The general dynamic model** adds island age, with richness peaking at intermediate age
  as islands grow and then erode (Whittaker et al. 2008; `sar_gdm`).
- **Habitat models** separate area from habitat diversity (the choros, jigsaw and
  Kallimanis models) or fit separate curves for the habitats of a landscape (the
  countryside model; `sar_habitat`, `sar_countryside`).
- **Random placement** asks whether a site's richness is what its area would give if
  individuals were scattered at random (Coleman 1981; Coleman et al. 1982; `coleman`).

## A typical workflow

```console
$ sars fit galap -m all                       # rank all 20 models by AICc
$ sars fit galap -m negexpo,monod --params    # look at the leaders' parameters
$ sars average galap                           # the weighted average
$ sars ci galap --n-boot 200 --seed 1          # intervals on the averaged curve
$ sars cv galap -m all --warm                  # how well each model predicts
$ sars threshold gala --ci                     # a small-island effect? (gala has one)
```

The same steps in Python are in [Getting Started](getting-started.md), and the
[shell scripts](https://github.com/jmcmeen/sars/tree/main/examples/cli) in the repository
run each of them on the bundled data.
