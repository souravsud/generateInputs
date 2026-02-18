# generateInputs - Terrain-Based CFD Input Generator

## Overview

This repository provides a modular pipeline for generating Computational Fluid Dynamics (CFD) input files for terrain-following mesh simulations. It automates the process of:

1. **Downloading terrain data** - Fetches Digital Elevation Model (DEM) data from global datasets
2. **Processing roughness maps** - Downloads and converts land cover data to aerodynamic roughness values
3. **Generating terrain meshes** - Creates terrain-following meshes for multiple wind directions
4. **Generating boundary conditions** - Creates atmospheric boundary layer (ABL) inlet profiles

## What The Code Does

### Main Pipeline (`generateInputs.py`)

The main script orchestrates a complete workflow:

1. **Reads coordinates** from a CSV file (`coords.csv`)
2. **Downloads terrain and roughness data** for each location
3. **Generates meshes** for 16 different wind directions (sectors)
4. **Creates ABL inlet boundary conditions** for each direction

The pipeline is designed for batch processing multiple locations with configurable parameters.

### Module Structure

#### `fetchData/` - Data Acquisition Module
- **`download_raster.py`**: Core functionality for downloading DEM and roughness maps
  - Downloads DEM tiles using `dem_stitcher`
  - Fetches ESA WorldCover land classification data
  - Converts land cover to aerodynamic roughness length (z0)
  - Reprojects data to local UTM coordinates
  
- **`reproject_raster.py`**: Handles coordinate transformations
  - Converts lat/lon data to UTM projections
  - Saves metadata about coordinate systems
  
- **`download_config.py`**: Configuration dataclass for download parameters
- **`csv_utils.py`**: CSV file parsing utilities
- **`parameter_generation.py`**: Generates wind direction angles

#### Git Submodules (External Components)
- **`terrain_following_mesh_generator`**: Creates 3D terrain-following meshes
- **`ABL_BC_generator`**: Generates atmospheric boundary layer profiles

## Configuration Files

- **`terrain_config.yaml`**: Mesh generation parameters (grid size, grading, boundaries)
- **`environment.yml`**: Conda environment specification
- **`coords.csv`**: Input coordinates for batch processing

## Code Quality Issues Identified

### 1. Redundant/Non-Functional Parts

#### In `fetchData/__init__.py` (Line 10)
```python
__all__ = ['DEMDownloader', 'DownloadConfig', 'download_dem']
```
**Issue**: Exports `download_dem` but the actual function is `download_raster_data`
**Impact**: Minor - incorrect export list but doesn't break functionality

#### In `download_raster.py` (Lines 312-324)
**Issue**: Has a `__main__` block for testing but this module is primarily imported
**Impact**: None - it's harmless but could be removed for cleaner code

#### In `parameter_generation.py` (Lines 30-33)
**Issue**: Has unused test code in `__main__` block
**Impact**: None - doesn't affect functionality

### 2. Logic That Can Be Simplified

#### In `parameter_generation.py`
The current implementation generates random directions within sectors. However:
- The `absolute_random` parameter name is misleading
- For CFD simulations, you typically want **evenly distributed** directions, not random
- Current code: `random.randint(lower_bound, upper_bound)` 

**Suggestion**: Replace with deterministic, evenly-spaced directions:
```python
def generate_directions(sectors):
    """Generate evenly-spaced directions around 360 degrees"""
    return [i * (360 // sectors) for i in range(sectors)]
```

This provides consistent, reproducible results: `[0, 22.5, 45, 67.5, ...]` for 16 sectors.

#### In `download_raster.py` (Lines 124-141)
The `_calculate_bounds` function is unnecessarily complex:
- Calculates 4 corners when only bounds are needed
- Returns both `bounds` and `corners` but `corners` is only used for roughness map tile selection

**Suggestion**: Could be simplified using direct bound calculations

#### In `csv_utils.py`
The `get_coordinate_by_index()` function loads ALL coordinates just to return one:
```python
def get_coordinate_by_index(csv_path, index, verbose=False):
    coordinates = load_coordinates_from_csv(csv_path, verbose)  # Loads everything!
    return coordinates[index]
```
**Impact**: Inefficient if you only need one coordinate. However, since the main script loads all coordinates anyway, this function appears to be **unused dead code**.

### 3. Consistency Issues

#### Error Handling
Some functions use try-except blocks while others don't:
- `csv_utils.py` has robust error handling
- `download_raster.py` mostly relies on exceptions propagating up

#### Logging
- Some modules use `if verbose: print(...)` pattern
- Others use `self.log()` method
- Inconsistent formatting of log messages

### 4. Missing Random Seed
In `parameter_generation.py`, using `random.randint()` without setting a seed means:
- Results are not reproducible
- Each run will generate different directions
- This is likely **not desired** for CFD simulations where reproducibility is important

## Recommended Improvements

### High Priority

1. **Fix the `__all__` export list** in `fetchData/__init__.py`
2. **Replace random directions with deterministic spacing** in `parameter_generation.py`
3. **Add random seed or make directions deterministic** for reproducibility
4. **Remove or document unused `get_coordinate_by_index()` function**

### Medium Priority

5. **Add a comprehensive README** (this file!)
6. **Simplify `_calculate_bounds()` function**
7. **Add docstrings** to all public functions
8. **Standardize logging approach** across modules

### Low Priority

9. **Remove `__main__` test blocks** from library modules
10. **Add type hints** consistently
11. **Create unit tests** for core functions

## Repository Merge Question

### Merging into `CFD-dataset` Repository

You asked about pushing this repository into your `CFD-dataset` repository and discarding its contents. Here's how:

#### Option 1: Replace CFD-dataset Contents (Recommended)

```bash
# Navigate to your CFD-dataset repository
cd /path/to/CFD-dataset

# Add this repo as a remote
git remote add generateInputs https://github.com/souravsud/generateInputs.git

# Fetch the content
git fetch generateInputs

# Replace the main branch content
git checkout main
git reset --hard generateInputs/main  # or whatever branch name

# Force push to GitHub (⚠️ This will DELETE all old content!)
git push --force origin main
```

#### Option 2: Merge as Subdirectory (Preserves History)

```bash
cd /path/to/CFD-dataset

# Add as remote
git remote add generateInputs https://github.com/souravsud/generateInputs.git
git fetch generateInputs

# Merge preserving directory structure
git merge --allow-unrelated-histories generateInputs/main

# Or move to subdirectory
git subtree add --prefix=generateInputs https://github.com/souravsud/generateInputs.git main
```

#### Option 3: Archive CFD-dataset and Use This One

Simply:
1. Archive the `CFD-dataset` repository on GitHub (Settings → Archive)
2. Rename this repository to `CFD-dataset` on GitHub
3. Update any references to point to the new location

**Recommendation**: Option 1 or 3 are cleanest if the old `CFD-dataset` content is truly obsolete. Option 3 is simplest if you just want to replace it.

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate sample_generator

# Initialize submodules
git submodule update --init --recursive
```

## Usage

```bash
# Run the complete pipeline
python generateInputs.py
```

The script will:
1. Read coordinates from `coords.csv`
2. Download terrain data to `Data/downloads/`
3. Generate meshes for each location and direction
4. Create ABL inlet boundary conditions

## Dependencies

- Python 3.12
- dem_stitcher - DEM data acquisition
- rasterio - Raster data processing
- windkit - Wind resource toolkit
- pyvista - 3D mesh visualization
- numpy, scipy, matplotlib - Scientific computing

## License

Not specified - please add appropriate license file.

## Contributing

This is a personal research repository. Contact the author for contribution guidelines.
