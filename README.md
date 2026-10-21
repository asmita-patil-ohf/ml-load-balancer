# ML-Based Adaptive Load Balancer

A machine learning course project exploring intelligent HTTP request distribution across backend servers.

## Project Overview

The goal is to compare traditional load-balancing algorithms with an ML-based approach that predicts backend response times and routes requests toward healthy servers with lower predicted latency.

## Objectives

- Simulate multiple backend servers with different capacities.
- Collect server metrics and request response times.
- Prepare and analyze a dataset for machine learning.
- Train and evaluate response-time prediction models.
- Compare Round Robin, Least Connections, and ML-based routing.
- Benchmark performance under different traffic conditions.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- scikit-learn (planned)
- FastAPI (planned)
- Docker Compose (planned)
- Locust (planned)

## Current Progress

- [x] Initial synthetic data collection
- [x] Dataset validation and exploratory data analysis
- [x] Dataset cleaning
- [x] Training/test split preparation
- [ ] ML model training
- [ ] Routing algorithm implementation
- [ ] HTTP load balancer integration
- [ ] Performance benchmarking

## Project Structure

```text
ml-load-balancer/
├── data/                 # Generated data and plots (not tracked)
├── src/
│   ├── collect_data.py
│   ├── inspect_data.py
│   ├── validate_data.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   └── split_data.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Generate the Dataset

```bash
python src/collect_data.py
python src/inspect_data.py
python src/validate_data.py
python src/explore_data.py
python src/prepare_data.py
python src/split_data.py
```

The scripts generate synthetic experiment data locally under `data/`.

## Current Limitations

The initial dataset uses simulated CPU and memory estimates and simulated backend processing delays. Results from this stage do not establish real-world performance. Later stages will add actual HTTP backends, ML-based routing, and controlled benchmarking.

## Research Question

Can machine learning-based response-time prediction improve load-balancing performance compared with conventional routing algorithms under heterogeneous backend loads?

## Status

Work in progress — machine learning course project.
