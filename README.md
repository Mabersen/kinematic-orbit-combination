# Orbit Combination and Validation Toolkit

This codebase provides tools for the combination, validation, and uncertainty analysis of kinematic orbit (KO) solutions from Swarm satellites. It includes arithmetic mean, inverse-variance, variance component estimation (VCE), residual reweighting, and several optimisation algorithm methods for combining KO data from multiple analysis centres. Validation is included using reduced-dynamic orbits (RDO) and Satellite Laser Ranging (SLR) observations. The performance assessment functionality in this toolkit requires SLR residuals retrieved using the GHOST software kit.

---

## Requirements

This code has been developed and tested using:

- Python ≥ 3.10
- Core packages:
  - `numpy`, `pandas`, `matplotlib`
  - `scipy`, `pygmo`, `astropy`

Ensure all required packages are installed. The `requirements.txt` file can be used to set up the environment.

```bash
pip install -r requirements.txt
```

---

## Structure

```
.
├── data/
│   └── sp3k/                  # Raw SP3K files for Swarm A/B/C satellites
├── scripts/
│   ├── preprocess_orbits.py  # SP3K parsing, alignment, and uncertainty propagation
│   ├── combine_orbits.py     # All orbit combination methods
│   ├── validate_combination.py # Validation using RDO or SLR
│   └── run_pipeline.py       # Main entry point for full pipeline
├── notebooks/
│   └── visualisations.ipynb  # Jupyter Notebooks for analysis and plotting
├── output/
│   └── figures/              # Automatically generated plots and diagnostics
├── README.md
└── requirements.txt
```

---

## Usage

You can run the full pipeline using the `run_pipeline.py` script:

```bash
python scripts/run_pipeline.py --config configs/swarmA_jan2023.yaml
```

Alternatively, run steps manually for more control:

1. **Preprocess KO and RDO data**  
   Parses SP3K files and interpolates them to a common epoch grid.

2. **Run orbit combination**  
   Choose from:
   - Arithmetic mean
   - Inverse-variance weighting
   - Variance component estimation (VCE)
   - Optimisation methods (e.g., CMA-ES, XNES, residual reweighting)

3. **Validate against RDO or SLR**  
   Compute residuals and evaluate consistency with reported uncertainty via:
   - RMS/STD ratio
   - Reduced chi-square
   - Rolling RMS analysis

4. **Visualise results**  
   Run `notebooks/visualisations.ipynb` to generate:
   - Residual plots
   - Daily stability plots
   - Covariance ellipse comparison

---

## Implemented Methods

### Combination Strategies
- **Mean**: Simple arithmetic average across available KO inputs
- **Inverse Variance**: Weights inputs by inverse reported variance
- **VCE**: Iteratively estimates input variances to improve uncertainty realism
- **Residual Reweighted**: Dynamically adjusts weights using residual magnitudes
- **Optimisation (CMA-ES, DE, Nelder-Mead)**: Uses a cost function (e.g. RMS) to optimise weight allocation across inputs

### Validation Tools
- **RDO Validation**: Uses ESA’s reduced-dynamic orbits as reference
- **SLR Validation**: Projects residuals onto LOS from SLR normal point observations
- **Chi-square testing**: Assesses realism of reported uncertainty
- **Rolling RMS**: Visualises localised orbit consistency

---

## Documentation

See the thesis documentation or internal `notebooks/` folder for:

- Methodology summaries with diagrams
- Flowcharts for the `gravtools` pipeline
- LaTeX-ready plots for publication

For theoretical background and references, see:

- Jäggi et al. (2016) – GNSS-based kinematic orbit determination
- Montenbruck et al. – SLR validation of orbit solutions
- Berendsen M. (2025 - masters thesis) – Combination of Swarm Kinematic Orbits

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

- Jäggi, A. et al. (2016). The ESA Swarm kinematic baseline and reduced-dynamic orbit determination.
- Montenbruck, O. et al. (2017). Precision orbit determination using SLR and GNSS.
- [Your thesis / preprint / dataset DOI if available]

---

## Cite this repository

If you use this toolkit in your work, please cite:

```
@software{your_lastname_2025,
  author = {Your Name},
  title = {Orbit Combination and Validation Toolkit for Swarm Satellites},
  year = 2025,
  url = {https://github.com/yourusername/swarm-orbit-combination},
}
```

---

## Want to contribute?

If you have suggestions, improvements, or bugs to report, feel free to open an issue or submit a pull request.

Happy combining!
