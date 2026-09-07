# SPECduration

SPECduration is an open-source Python package for estimating spectral seismic duration from the displacement response spectrum (DRS).

The software computes the DRS of an earthquake acceleration record, automatically identifies the predominant spectral window using the second spectral derivative, and performs harmonic fitting within the detected window to estimate the spectral duration parameter T_s.

SPECduration is designed for reproducible and non-interactive analysis of processed earthquake acceleration records. The fitting window is defined automatically, avoiding manual selection.

---

# Table of Contents

- [Scientific background](#scientific-background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input format](#input-format)
- [Output structure](#output-structure)
- [Interpretation of results](#interpretation-of-results)
- [Methodological workflow](#methodological-workflow)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Citation](#citation)

---

# Scientific background

SPECduration implements the spectral-duration method described in the associated research work. The method uses the displacement response spectrum to characterize the predominant spectral shape of pulse-type and near-fault ground motions.

The harmonic model used for the fitting is:

$$
DRS(T)=P\sin\left(\frac{\pi T_s}{T}+c\right)
$$

where:

- DRS(T) is the displacement response spectrum;
- P is the fixed maximum DRS value;
- T is the natural period;
- T_s is the spectral duration parameter;
- c is the phase constant.

The predominant spectral window is defined from the behavior of the second spectral derivative DRS''(T) around the predominant DRS peak. The harmonic model is then fitted within this automatically detected window to estimate T_s.

The default analysis uses a damping ratio of 5% and a spectral period step of:

$$
\Delta T = 0.01\ \mathrm{s}
$$

For the complete theoretical development and validation of the method, refer to the associated research paper.

---

# Features

- Displacement response spectrum computation using the Newmark-beta method.
- Automatic identification of the predominant period T_peak.
- Second spectral derivative-based detection of the predominant spectral window.
- Automatic definition of T_left and T_right.
- Harmonic fitting within the detected spectral window.
- Estimation of the spectral duration T_s.
- Calculation of the coefficient of determination R^2.
- Automatic validation of the fitting result.
- Generation of Excel and CSV outputs.
- Generation of diagnostic figures.
- Command-line interface for reproducible analysis.
- No manual selection of the fitting window.

---

# Installation

## Requirements

- Python 3.9 or later
- Git, if the repository is cloned directly

The required Python dependencies are installed automatically with the package.

## Install from GitHub

Clone the repository:

```bash
git clone https://github.com/Lizandro-Jara/SPECduration.git
cd SPECduration
```

Install the package:

```bash
python -m pip install .
```

Using a virtual environment is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install .
```

### Linux and macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Install from a ZIP file

The repository can also be downloaded as a ZIP file from GitHub.

After extracting the project, open a terminal in the project root directory, where `pyproject.toml` is located, and run:

```bash
python -m pip install .
```

---

# Usage

After installation, SPECduration can be executed from the command line.

For example:

```bash
specduration --input "examples/LOMA PRIETA.txt" --out "outputs"
```

The program can also be executed directly as a Python module:

```bash
python -m specduration.cli --input "examples/LOMA PRIETA.txt" --out "outputs"
```

## Command-line options

- `--input`: path to the earthquake acceleration record in `.txt` format.
- `--out`: output directory where the results will be saved.
- `--no-plots`: disables the generation of diagnostic figures.
- `--zeta`: damping ratio used for DRS computation. Default: `0.05`.
- `--tmax`: maximum period used for DRS computation, in seconds. Default: `15.0`.
- `--dT`: spectral period step used for DRS computation, in seconds. Default: `0.01`.
- `--min-points-window`: minimum number of points required in the detected spectral window. Default: `10`.
- `--min-points-fit`: minimum number of points required for harmonic fitting. Default: `10`.
- `--min-R2-fit`: minimum R^2 required to accept the fitting result. Default: `0.98`.
- `--no-extend-window`: disables automatic extension of the detected window when the minimum number of points is not reached.

---

# Input format

SPECduration requires a plain-text earthquake acceleration record with at least two columns:

1. Time, in seconds.
2. Ground acceleration, in m/s².

Example:

```text
t(s)    ag(m/s^2)

0.00    -0.00123
0.02     0.00345
0.04    -0.00210
...
```

The time step is obtained from the time column.

The input record should be processed before analysis. SPECduration is intended for earthquake acceleration records prepared for response-spectrum analysis.

---

# Output structure

For each processed record, SPECduration creates a dedicated output directory inside the directory specified with `--out`.

The output includes an Excel report, CSV files, and diagnostic figures.

## Main output

### `results.xlsx`

The main Excel report contains the principal results and information from the DRS, spectral-window detection, harmonic fitting, and validation stages.

## CSV outputs

### `summary.csv`

Compact summary of the main estimated parameters and validation results.

### `results_diagnostics.csv`

Detailed diagnostic information from the spectral-window detection and harmonic fitting procedures.

### `drs_full.csv`

Complete displacement response spectrum data.

### `second_derivative.csv`

Numerical spectral derivative data used in the window-detection procedure.

## Diagnostic figures

### `drs_full.png`

Displacement response spectrum over the analyzed period range.

![Displacement response spectrum](docs/figures/drs_full.png)

### `drs_window.png`

Predominant spectral window, peak period, and harmonic fitting result.

![Detected spectral window and harmonic fit](docs/figures/drs_window.png)

### `second_derivative_window.png`

Second spectral derivative and the detected spectral window.

![Second spectral derivative](docs/figures/second_derivative_window.png)

---

# Interpretation of results

The main parameter estimated by SPECduration is the spectral duration T_s.

Unlike conventional time-domain duration measures, T_s is obtained from the harmonic representation of the predominant region of the displacement response spectrum.

The coefficient of determination R^2 indicates the quality of the harmonic fit within the detected window. By default, the fitting result is considered valid when the specified fitting criteria are satisfied and:

$$
R^2 \geq 0.98
$$

The estimated T_s should be interpreted as a spectral-duration descriptor and not as a direct replacement for conventional time-domain duration measures.

---

# Methodological workflow

The computational workflow implemented in SPECduration is:

1. Load the processed earthquake acceleration record.
2. Compute the displacement response spectrum DRS(T) using Newmark-beta integration.
3. Identify the predominant DRS peak at T_peak.
4. Compute the second spectral derivative DRS''(T).
5. Determine the predominant spectral window from the second-derivative behavior around T_peak, obtaining T_left and T_right.
6. Fit the harmonic model within the detected window.
7. Estimate the spectral duration T_s and calculate R^2.
8. Validate the fitting result.
9. Export numerical results and diagnostic figures.

---

# Reproducibility

SPECduration is designed for deterministic and reproducible analysis.

Given the same input record, software version, and analysis parameters, the program produces the same DRS, detected spectral window, harmonic fitting results, validation status, and output data.

The automatic definition of the fitting window avoids manual selection and allows the same procedure to be applied consistently to different earthquake records.

---

# License

This project is released under the MIT License.

See the `LICENSE` file for the complete license text.

---

# Citation

If you use SPECduration in academic work, please cite the software using the information provided in the `CITATION.cff` file.

The spectral-duration method implemented in SPECduration is described in the associated research paper.