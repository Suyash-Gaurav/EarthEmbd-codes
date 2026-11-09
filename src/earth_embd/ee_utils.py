"""
Earth Engine utilities for the EarthEM analysis pipeline.

All functions assume that `ee.Initialize()` has already been called
with a valid project.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import ee
import pandas as pd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def krs_region() -> ee.Geometry:
    """Return the primary KRS Reservoir bounding polygon."""
    return ee.Geometry.Polygon(
        [
            [
                [76.3978, 12.5521],
                [76.3978, 12.3550],
                [76.6519, 12.3550],
                [76.6519, 12.5521],
            ]
        ]
    )


def water_body_region() -> ee.Geometry:
    """Return a tighter polygon covering the core water surface of KRS."""
    return ee.Geometry.Polygon(
        [
            [
                [76.4700, 12.4800],
                [76.4700, 12.4500],
                [76.5100, 12.4500],
                [76.5100, 12.4800],
            ]
        ]
    )


def urban_expansion_region() -> ee.Geometry:
    """Approximate urban expansion zone near Mysore."""
    return ee.Geometry.Polygon(
        [
            [
                [76.5800, 12.3800],
                [76.5800, 12.2800],
                [76.6800, 12.2800],
                [76.6800, 12.3800],
            ]
        ]
    )


def agricultural_region() -> ee.Geometry:
    """Agricultural zone used for 2017–2024 change analysis."""
    return ee.Geometry.Polygon(
        [
            [
                [76.5000, 12.4000],
                [76.5000, 12.3200],
                [76.6000, 12.3200],
                [76.6000, 12.4000],
            ]
        ]
    )


# ---------------------------------------------------------------------------
# Embedding access
# ---------------------------------------------------------------------------

def load_annual_embeddings(
    year: int,
    region: ee.Geometry,
    collection_id: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
) -> ee.Image:
    """
    Load and mosaic the annual Satellite Embedding image for a given year
    and region of interest.
    """
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    return (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .filterBounds(region)
        .mosaic()
        .clip(region)
    )


def get_band_names(image: ee.Image) -> List[str]:
    """Return the list of band names for an Earth Engine image."""
    return image.bandNames().getInfo()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate_buffered_samples(
    seed_points: ee.FeatureCollection,
    buffer_radius: float = 500,
    num_points: int = 10,
    label_property: str = "landcover",
) -> ee.FeatureCollection:
    """
    Expand a small set of labeled seed points into a larger sample by
    drawing random points inside a buffer around each seed.
    """

    def _expand(feature: ee.Feature) -> ee.FeatureCollection:
        geom = feature.geometry()
        label = feature.get(label_property)
        random = ee.FeatureCollection.randomPoints(
            region=geom.buffer(buffer_radius),
            points=num_points,
            seed=label,
        )
        return random.map(lambda pt: pt.set(label_property, label))

    return seed_points.map(_expand).flatten()


def sample_embeddings_at_points(
    image: ee.Image,
    points: ee.FeatureCollection,
    properties: Sequence[str],
    scale: int = 100,
) -> ee.FeatureCollection:
    """Sample embedding bands at the given points."""
    return image.sampleRegions(
        collection=points,
        properties=list(properties),
        scale=scale,
        tileScale=2,
    )


def mean_embeddings_over_region(
    image: ee.Image,
    region: ee.Geometry,
    scale: int = 100,
) -> Dict[str, float]:
    """
    Compute the mean embedding vector over a geometry.
    Returns a Python dict of band → mean value.
    """
    result = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    )
    return result.getInfo()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def feature_collection_to_dataframe(
    fc: ee.FeatureCollection,
    band_names: Sequence[str],
    extra_properties: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert an Earth Engine FeatureCollection (already sampled) into a
    clean pandas DataFrame containing the requested bands and properties.
    """
    extra = list(extra_properties or [])
    features = fc.getInfo()["features"]

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {band: props.get(band) for band in band_names}
        for p in extra:
            row[p] = props.get(p)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Drop rows that are missing any embedding dimension
    df = df.dropna(subset=list(band_names)).reset_index(drop=True)
    return df
