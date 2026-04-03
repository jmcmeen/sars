# sars

Species-area relationship curve fitting in Python.

A conceptual mirror of the R [`sars`](https://cran.r-project.org/package=sars) package (Matthews et al. 2019), native to the Python scientific stack.

## Installation

```bash
pip install sars
```

## Quick start

```python
import sars

# Fit the power-law SAR model
import pandas as pd
data = pd.DataFrame({"area": [1, 2, 5, 10, 50], "species": [10, 15, 25, 40, 80]})
fit = sars.sar_power(data)
print(fit)
```

## License

MIT
