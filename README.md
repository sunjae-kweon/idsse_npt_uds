# Net Playing Time Assessment Task: Coding

This repository contains a coding implementation for the *PhD Project Net Time Assessment Task*.  
The project focuses on accessing, processing, and analysing the IDSSE dataset (Bassek et al., 2025) to develop and compare methods for quantifying **Net Playing Time (NPT)** in professional football.

## 1. Project Objective

The main goals are:

- Build a reproducible pipeline for downloading and parsing IDSSE match data.
- Provide exploratory visualization outputs.
- Compute Net Playing Time with multiple methods.
- Compare computed values with official match information.

## 2. Dataset

- DOI: https://doi.org/10.6084/m9.figshare.28196177  
- Figshare API endpoint used in code: `https://api.figshare.com/v2/articles/28196177`

The dataset includes:

- 7 full matches from the German Bundesliga (1st and 2nd divisions)
  - Match information
  - Event data
  - Position/tracking data by TRACAB

## 3. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the analysis notebook:

- `main_notebook.ipynb`

## 4. Project Structure

```text
.
├── src/
│   ├── download.py       # download + file integrity checks
│   ├── parser.py         # IDSSE XML parsing to pandas DataFrames
│   ├── visualization.py  # pass/shot/heatmap/restart visualizations
│   ├── npt_analysis.py   # NPT methods, validation, episode analysis
│   └── __init__.py       # package exports
├── main_notebook.ipynb   # end-to-end analysis workflow
├── requirements.txt      # Python dependencies
└── README.md             # project overview
```

## 5. Citation

```bibtex
@article{bassek2025integrated,
  title={An integrated dataset of spatiotemporal and event data in elite soccer},
  author={Bassek, Manuel and Rein, Robert and Weber, Hendrik and Memmert, Daniel},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={195},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
