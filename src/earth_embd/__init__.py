"""
EarthEM — Embedding-based remote sensing analysis utilities.

Modules
-------
ee_utils        Earth Engine geometry, sampling, and embedding helpers
analysis        Classification, similarity, clustering, and statistics
visualization   Publication-oriented plotting functions
"""

from . import ee_utils, analysis, visualization

__version__ = "0.1.0"
__all__ = ["ee_utils", "analysis", "visualization"]
