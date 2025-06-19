
# Orbit Combination and Validation Toolkit

This codebase provides tools for the combination, validation, and uncertainty analysis of kinematic orbit (KO) solutions from Swarm satellites. It includes arithmetic mean, inverse-variance, variance component estimation (VCE), residual reweighting, and several optimisation algorithm methods for combining KO data from multiple analysis centres. Validation is included using reduced-dynamic orbits (RDO) and Satellite Laser Ranging (SLR) observations. The performance assessment functionality in this toolkit requires SLR residuals retrieved using the GHOST software kit.

---

## Requirements and Installation

This code has been developed and tested using:

- Python ≥ 3.10
- Core packages:
  - `numpy`, `pandas`, `matplotlib`
  - `scipy`, `pygmo`, `astropy`, `tudat-space`

Ensure all required packages are installed. The `environment.yaml` file can be used to set up the environment via conda. Navigate to the repository in your conda prompts and run:

```bash
conda env create -f environment.yaml
```
Open configuration.py and set 'root' to the absolute path of the 'gravtools' folder. Run this file to automatically populate the file structure if this is not already done. You may already make use of the toolkit to combine orbits. 

If you wish to go further and make use of the SLR validation tools, you need access to the SLRRES function provided by the GHOST software toolkit. Code is provided to make use of this allowing for batch processing of orbit files which may be ran on the Aristarchos server, however the GHOST software itself is not something I may distribute. Otherwise, in this repository an example normal point .csv file is provided for the year of 2023. The same goes for making use of the SLR residual data to rescale input orbit uncertainty.

---

## Structure

gravtools/
├── README.md
├── environment.yaml                # Conda environment
├── configuration.py                # Centralised paths and constants
├── combination_config.yaml         # Input configuration file for pipeline
├── orbit_combination_pipeline.py   # Main orbit combination pipeline
├── data_utilities/
│   ├── parsing_utils.py       # Parser and SP3K writer
│   ├── retrieve_data.py       # Downloads KO/RDO orbits
│   └── tudat_utilities.py     # Utilities using the tudat package
├── kinematic_orbits/
│   ├── produce_results.py                # Pipeline script with user control
│   ├── combine_orbits.py                 # Combination methods
│   ├── retrieve_arcs.py                  # Retrieves KO/RDO arcs
│   ├── classes.py                        # AccessRequest and Arc classes
│   ├── pygmo_optimiser.py                # Pygmo optimiser implementation
│   ├── pygmo_optimiser_independent.py    # Independent optimisation implementation
│   ├── plotting_functions.py             # Plotting helpers for orbits
│   ├── frame_utilties.py                 # Frame conversion utilities for orbits
│   └── kinematic_utilities.py            # General kinematic orbit utilities
├── verification/
│   ├── run_verification_tests.py       # Synthetic orbit testing
│   ├── verify_rdo_fit.py               # Residuals vs RDO reference
│   ├── plot_verification_results.py    # Visual diagnostic plots
│   ├── variance_verification.py        # Tests uncertainty propagation
│   └──  verification_utilities.py      # Utilities for verification
├── SLR/
│   ├── process_normal_points.py          # SLR residual validation + plots
│   ├── slr_variance_validation.py        # LOS projection + uncertainty validation + output plotting
│   ├── variance_validation_plotting.py   # Plot functions for variance validation
│   └── sinex_station_loader.py           # Parses SINEX station info
├── data/                         # Root data directory
│   ├── orbits/
│   │   ├── KO/
│   │   │   ├── AIUB/
│   │   │   │   ├── 47/
│   │   │   │   ├── 48/
│   │   │   │   └── 49/
│   │   │   └── ...
│   │   ├── CO/
│   │   ├── CORR/
│   │   ├── CORS/
│   │   └── RDO/
│   │       ├── ESA/
│   │       │   ├── 47/
│   │       │   ├── 48/
│   │       │   └── 49/
│   │       └── ...
│   │   
│   ├── gravity field models/     # GFM output folder (not currently in use)
│   └── SLR/
│       ├── stations/             # SINEX station info
│       └── [normal point files]  # GHOST residuals and timescales from generated orbits
└── output/ (optional)            # Can contain result SP3K or figures


## Usage

### 1. Orbit Combination (Primary Functionality)

The main pipeline combines KO solutions from multiple analysis centres into combined orbits. You can run this via command-line or Python (ensure you have the correct environment active):

```bash
python gravtools/kinematic_orbits/produce_results.py --config gravtools/combination_config.yaml
```

Or programmatically:
```python
from gravtools.kinematic_orbits.orbit_combination_pipeline import main, load_config
cfg = load_config("gravtools/combination_config.yaml")
main(cfg)
```
Alternatively by simply opening `orbit_combination_pipeline.py` and running it in your IDE

This pipeline:
- Retrieves KO and RDO data (if specified). Downloads from available servers (download functionality requires access to the TU Delft Aristarchos server for KO's)
- Applies multiple combination strategies (e.g., `vce`, `inverse_variance`, `residual_weighted`, etc.)
- Saves daily combined orbits in the .SP3K file format

You can configure:
- Satellite (IDs: 47 = SWA, 48 = SWB, 49 = SWC)
- Timespan (month-based or day-based)
- Combination methods (including optimisers)
- Whether to additionally save original KO and RDO arcs for later SLR residual analysis

---

### 2. Combination Verification

This framework includes dedicated tools to test and visualise the correctness of combination methods under controlled conditions.

#### Run Verification Test Suites:

```bash
python gravtools/verification/run_verification_tests.py
```

This will:
- Generate synthetic orbit data
- Inject known noise and bias
- Compare results across all methods

#### Additional tools:

- `plot_verification_results.py`: Plot the results of the verification tests
- `variance_verification.py`: Validates correct propagation of uncertainty under known synthetic conditions
- `verify_rdo_fit.py`: Testing whether combined orbits fit to the RDO reference
---

### 3. SLR Residual-Based Validation

Use Satellite Laser Ranging (SLR) normal point data to independently assess orbit quality.

#### a. Residual-Based Orbit Assessment

```bash
python gravtools/SLR/process_normal_points.py
```

- Loads and filters SLR normal points
- Computes daily and station-wise RMS of SLR residuals
- Identifies high-performance orbit methods via Pareto front selection

#### b. Line-of-Sight Uncertainty Validation

```bash
python gravtools/SLR/slr_variance_validation.py
```

- Projects 3D orbit uncertainty onto the SLR line-of-sight vector
- Compares predicted uncertainty to residuals
- Outputs rolling RMS/STD ratio and chi-square indicators

All SLR validation scripts require:
- Combined orbit files (SP3K)
- SLR normal point data as .CSV (from GHOST)
- SINEX file for station coordinates (provided in repository for the test period defined in `slr_variance_validation.py`. The user may add additional manually.)

---

## Implemented Methods

### Combination Strategies (code names provided in brackets)
- **Mean** (mean):  Simple arithmetic average across available KO inputs
- **Inverse Variance** (inverse_variance): Weights inputs by inverse reported variance
- **VCE** (vce): Iteratively estimates input variances to improve uncertainty realism
- **Residual Reweighted** (residual_weighted): Dynamically adjusts weights using residuals with respect to a reduced-dynamic reference
- **Optimisation (CMAES, DE, Nelder-Mead, etc)** (nelder_mead, cmaes, de, etc.): Uses a cost function (Residual RMS with respect to a reduced-dynamic reference) to optimise weight allocation across inputs

### Validation Tools
- **SLR Validation**: SLR normal point observations are used to assess performance of the orbits
---

## Documentation
The theory behind combination methods can be found in Berendsen M. (2025 - masters thesis) – Combination of Swarm Kinematic Orbits. This may be accessed through the following url: https://resolver.tudelft.nl/uuid:6ce00620-7be9-441c-8208-786dbe12878a.

---

## Author(s)
This toolkit was developed by Mattijs Berendsen, based on research conducted at the TU Delft.  
For questions or collaboration requests, please contact: wowmattijs@gmail.com

---

## License
Unless otherwise noted, all scripts are released under the Apache 2.0 License.  
Please refer to `LICENSE.txt` for details.

---

## References
- Dirkx et al. (2022) - Tudat space, doi: https://doi.org/10.5194/epsc2022-253
---

## Cite this repository
If you use this toolkit in your work, please cite it as below or check out the CITATION.cff file:

```
@software{Berendsen_2025,
  author = {Mattijs Berendsen},
  title = {Orbit Combination and Validation Toolkit for Swarm Satellites},
  year = 2025,
  url = {https://github.com/Mabersen/kinematic-orbit-combination},
  doi = {https://doi.org/10.4121/03c249d6-674c-47cf-918f-1ef9bdafe749}
}
```

---

## Want to contribute?

If you have ideas regarding improvements go ahead and pull the repository. Feel free to build upon the code!
