"""sars — Species-area relationship curve fitting in Python.

Conceptual mirror of the R `sars` package (Matthews et al. 2019),
native to the Python scientific stack.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sars")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from sars._checks import ResidualChecks
from sars._coleman import ColemanFit, ColemanTest, coleman
from sars._crossval import LOOCV, loo_cv
from sars._effort import standardised_richness
from sars._gdm import GDMFit, GDMMultiFit, sar_gdm
from sars._habitat import CountrysideFit, HabitatFit, sar_countryside, sar_habitat
from sars._inference import (
    AveragedSAR,
    BootstrappedCI,
    MultiSARFit,
    aicc,
    akaike_weights,
    bootstrap_ci,
    sar_average,
    sar_multi,
)
from sars._io import (
    from_csv,
    from_df,
    from_geodataframe,
    from_pyinaturalist,
    load_aegean,
    load_aegean2,
    load_countryside,
    load_gala,
    load_galap,
    load_habitat,
    load_niering,
    load_schrader,
    load_xu,
    load_xu_abundance,
)
from sars._linpow import LinPowFit, lin_pow
from sars._loss import HabitatLoss, habitat_loss
from sars._many import ManySARFit, sar_many
from sars._models import (
    SARFit,
    sar_asymp,
    sar_betap,
    sar_chapman,
    sar_epm1,
    sar_epm2,
    sar_gompertz,
    sar_heleg,
    sar_koba,
    sar_linear,
    sar_loga,
    sar_mmf,
    sar_monod,
    sar_negexpo,
    sar_p1,
    sar_p2,
    sar_power,
    sar_powerR,
    sar_ratio,
    sar_weibull3,
    sar_weibull4,
)
from sars._plots import (
    plot_average,
    plot_coleman,
    plot_fit,
    plot_lin_pow,
    plot_multi,
    plot_residuals,
    plot_threshold,
)
from sars._predict import sar_pred
from sars._serialize import from_json, load, save, to_json
from sars._threshold import ThresholdCI, ThresholdFit, sar_threshold, threshold_ci

__all__ = [
    "__version__",
    "AveragedSAR",
    "BootstrappedCI",
    "LinPowFit",
    "ColemanFit",
    "ColemanTest",
    "CountrysideFit",
    "GDMFit",
    "GDMMultiFit",
    "HabitatFit",
    "HabitatLoss",
    "LOOCV",
    "ManySARFit",
    "MultiSARFit",
    "ResidualChecks",
    "SARFit",
    "ThresholdCI",
    "ThresholdFit",
    "aicc",
    "akaike_weights",
    "bootstrap_ci",
    "coleman",
    "from_csv",
    "from_df",
    "from_geodataframe",
    "from_json",
    "from_pyinaturalist",
    "habitat_loss",
    "lin_pow",
    "load",
    "load_aegean",
    "load_aegean2",
    "load_countryside",
    "load_gala",
    "load_galap",
    "load_habitat",
    "load_niering",
    "load_schrader",
    "load_xu",
    "load_xu_abundance",
    "loo_cv",
    "sar_average",
    "sar_countryside",
    "sar_gdm",
    "sar_habitat",
    "sar_many",
    "sar_multi",
    "sar_pred",
    "sar_threshold",
    "save",
    "threshold_ci",
    "to_json",
    "plot_average",
    "plot_coleman",
    "plot_fit",
    "plot_lin_pow",
    "plot_multi",
    "plot_residuals",
    "plot_threshold",
    "sar_asymp",
    "sar_betap",
    "sar_chapman",
    "sar_epm1",
    "sar_epm2",
    "sar_gompertz",
    "sar_heleg",
    "sar_koba",
    "sar_linear",
    "sar_loga",
    "sar_mmf",
    "sar_monod",
    "sar_negexpo",
    "sar_p1",
    "sar_p2",
    "sar_power",
    "sar_powerR",
    "sar_ratio",
    "sar_weibull3",
    "sar_weibull4",
    "standardised_richness",
]
