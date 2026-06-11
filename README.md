# Multi-State AADT Enrichment Pipeline

## Overview

This project builds a scalable AADT (Annual Average Daily Traffic) enrichment pipeline for crash datasets across multiple U.S. states(currently for Texas state).

The pipeline:
- matches crashes to nearest traffic stations,
- applies VMT-normalized AADT adjustment,
- assigns ZIP codes using spatial joins,
- generates QA plots,
- exports cleaned yearly datasets,
- and validates outputs through an automated validation framework.

The workflow was refactored into a modular, reusable, high-performance architecture using:
- **Polars**
- **SciPy cKDTree**
- **GeoPandas/Shapely**
- reusable utility modules
- configurable multi-state support

---
# Project Structure

```text
project/
├── config.py
├── aadt_fill_main.py
├── aadt_fill_utils.py
├── validation_util.py
├── run_validation.py
├── discover_mapping.py
│
├── master_data/
│   ├── old_aadt.csv
│   ├── master_cleaned_dataset_2015-2024.csv
│   └── tx_texas_zip_codes_geo.min.json
│
├── outputs/
│   └── TX/
│
└── README.md
```
---

# How to Run

Run the AADT fill pipeline:

```bash
python aadt_fill_main.py --state TX
```

Run validation:

```bash
python run_validation.py --state TX
```

Outputs are stored in:

```text
outputs/<state_code>/
```

Example:

```text
outputs/TX/
```

---

# Column Mapping Discovery

A helper script is included to automatically suggest column mappings for new states using fuzzy matching.

Master crash dataset:

```bash
python discover_mapping.py "path/to/master.csv" --kind master --skip-rows 0
```

Station dataset:

```bash
python discover_mapping.py "path/to/stations.csv" --kind stations
```

Suggested mappings can then be added to:

```python
config.STATE_CONFIGS
```

---

# Validation Framework

The validation checks include:

- Required schema columns
- Crash year consistency
- Latitude/Longitude bounds
- Distance threshold validation
- ZIP code completeness
- Year-gap consistency
- VMT multiplier validation
- AADT recomputation validation
- Match-type consistency
- Fill-rate reporting

Run using:

```bash
python run_validation.py --state TX
```

---

# Performance Improvements

The original implementation used:
- Pandas
- manual haversine loops

The pipeline was redesigned using:
- Polars
- vectorized operations
- cKDTree nearest-neighbor search
- lazy CSV scanning

Result:
- lower memory usage,
- significantly faster processing,
- scalable multi-million-row support,
- reusable multi-state architecture.

# Main Technologies Used

- Python
- Polars
- NumPy
- SciPy cKDTree
- GeoPandas
- Shapely
- Matplotlib