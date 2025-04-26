# CensusAlign 🗃️

CensusAlign is a Python toolkit developed by the UCSB Quantifying Gerrymandering team as part of the ERSP program. It is designed to streamline data processing for MCMC analysis of districting fairness. This tool (and our project as a whole) helps researchers, data analysts, and policymakers better understand the impact districting lines have on election outcomes.

## Features

- Align census data with electoral boundaries.
- Perform data transformations and aggregations.
- Generate graphs for MCMC analysis.

## Installation

### Requirements

- Python 3.12 or higher

### Installation

To get started with CensusAlign, follow these steps:

1. Clone the repository:

   ```bash
   git clone git@github.com:Quantifying-Gerrymandering/censusalign.git
   cd censusalign
   ```

2. Install the required dependencies using `uv`:

   ```bash
   uv pip install -r pyproject.toml
   ```

That's it! You're ready to use CensusAlign.

## Usage

Here's a quick example of how to use CensusAlign:

```python
from censusalign import Cultivate

# Example usage
culivator = Cultivate(election="congressional", year="2022")
G = cultivator.graphify()
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [us](mailto:swayland@ucsc.edu).
