# feature_registry.py
# Central definition of every feature and every model version.
# To add a new feature: one entry in FEATURE_REGISTRY + one name in a FEATURE_SETS list.
# Import with:  from feature_registry import FEATURE_REGISTRY, FEATURE_SETS, build_feature_image

import math
import ee

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY
# Each entry:  'feature_name': ('source_key', 'band_name')
# source_key must match a key in the `sources` dict passed to build_feature_image()
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_REGISTRY = {
    # ── Monthly snapshot features ──────────────────────────────────────────────
    # key = monthly composite name ('jan', 'apr', etc.)
    'NDVI_jan': ('jan', 'NDVI'),  'NDVI_apr': ('apr', 'NDVI'),
    'NDVI_may': ('may', 'NDVI'),  'NDVI_jun': ('jun', 'NDVI'),
    'NDVI_jul': ('jul', 'NDVI'),  'NDVI_aug': ('aug', 'NDVI'),
    'NDVI_sep': ('sep', 'NDVI'),  'NDVI_nov': ('nov', 'NDVI'),
    'EVI_jul':  ('jul', 'EVI'),   'EVI_aug':  ('aug', 'EVI'),
    'GCVI_jun': ('jun', 'GCVI'),  'GCVI_jul': ('jul', 'GCVI'),
    'GCVI_aug': ('aug', 'GCVI'),  'GCVI_sep': ('sep', 'GCVI'),
    'LSWI_jul': ('jul', 'LSWI'),  'LSWI_aug': ('aug', 'LSWI'),
    'LSWI_sep': ('sep', 'LSWI'),

    # ── Seasonal curve / shape features ───────────────────────────────────────
    # key = 'shapes'  (one pre-built image with all shape bands)
    'shape_mid_dip':         ('shapes', 'shape_mid_dip'),
    'shape_integrated_ndvi': ('shapes', 'shape_integrated_ndvi'),
    'shape_peak_skewness':   ('shapes', 'shape_peak_skewness'),
    'shape_season_length':   ('shapes', 'shape_season_length'),
    'shape_winter_ratio':    ('shapes', 'shape_winter_ratio'),

    # ── Harmonic / phenological features (V4) ─────────────────────────────────
    # Tier 1 — annual (1x frequency): constant, cos1, sin1, amp1, phase1
    'NDVI_c':      ('harm_NDVI', 'constant'), 'NDVI_cos1':  ('harm_NDVI', 'cos1'),
    'NDVI_sin1':   ('harm_NDVI', 'sin1'),     'NDVI_amp1':  ('harm_NDVI', 'amp1'),
    'NDVI_phase1': ('harm_NDVI', 'phase1'),
    'EVI_c':       ('harm_EVI',  'constant'), 'EVI_cos1':   ('harm_EVI',  'cos1'),
    'EVI_sin1':    ('harm_EVI',  'sin1'),
    'LSWI_c':      ('harm_LSWI', 'constant'), 'LSWI_cos1':  ('harm_LSWI', 'cos1'),
    'LSWI_sin1':   ('harm_LSWI', 'sin1'),
    # Tier 2 — semi-annual (2x frequency): captures double-crop mid-season dip
    'NDVI_cos2':   ('harm_NDVI', 'cos2'),     'NDVI_sin2':  ('harm_NDVI', 'sin2'),
    'NDVI_amp2':   ('harm_NDVI', 'amp2'),
    # Tier 2 — GCVI harmonics
    'GCVI_c':      ('harm_GCVI', 'constant'), 'GCVI_cos1':  ('harm_GCVI', 'cos1'),
    'GCVI_sin1':   ('harm_GCVI', 'sin1'),

    # Robust Annual Stats
    'NDVI_max':    ('robust', 'NDVI_max'),
    'NDVI_amp':    ('robust', 'NDVI_amp'),
    'NDVI_mean':   ('robust', 'NDVI_mean'),
    'NDVI_std':    ('robust', 'NDVI_std'),

    # ── Contextual Features (V6) ─────────────────────────────────────────────
    'text_ndvi':   ('context', 'text_ndvi'),    # Neighborhood NDVI stdDev
    'topo_slope':  ('context', 'topo_slope'),   # SRTM Slope
    'hydro_ndwi':  ('context', 'hydro_ndwi'),   # Max annual NDWI
}

# ══════════════════════════════════════════════════════════════════════════════
# VERSION DEFINITIONS
# Each version inherits and extends the previous one.
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_SETS = {}

FEATURE_SETS['v1'] = [                          # 9 features — baseline
    # Drawn directly from Phase 2 phenology analysis table
    'NDVI_jan',                                 # alfalfa: uniquely elevated in winter
    'NDVI_may',                                 # winter wheat peak vs dormant corn
    'EVI_jul',                                  # peak biomass complement
    'GCVI_jun',                                 # corn/soy separation begins
    'GCVI_jul',                                 # max corn/soy separation (+2.504)
    'GCVI_aug',                                 # direction flip: soy now higher
    'LSWI_jul',                                 # secondary corn/soy water signal
    'LSWI_sep',                                 # corn senescence signal
    'NDVI_nov',                                 # soy holds green vs corn collapsed
]

FEATURE_SETS['v2'] = FEATURE_SETS['v1'] + [    # 16 features — broader temporal coverage
    'NDVI_apr',                                 # early season baseline
    'NDVI_jun',                                 # canopy closure timing
    'NDVI_aug',                                 # late-season peak
    'NDVI_sep',                                 # onset of senescence
    'GCVI_sep',                                 # late GCVI separation
    'EVI_aug',                                  # soy peak EVI
    'LSWI_aug',                                 # water stress at senescence onset
]

FEATURE_SETS['v3'] = FEATURE_SETS['v2'] + [    # 21 features — + seasonal curve shape
    'shape_mid_dip',                            # June dip: double-crop signature
    'shape_integrated_ndvi',                    # season-integrated greenness
    'shape_peak_skewness',                      # weighted center-of-mass of NDVI
    'shape_season_length',                      # months above NDVI threshold
    'shape_winter_ratio',                       # winter/summer greenness ratio
]

FEATURE_SETS['v4'] = FEATURE_SETS['v3'] + [    # 42 features — + harmonics + Robust
    # Annual NDVI harmonic (Tier 1)
    'NDVI_c', 'NDVI_cos1', 'NDVI_sin1', 'NDVI_amp1', 'NDVI_phase1',
    # Semi-annual NDVI harmonic (Tier 2 — double-crop dip)
    'NDVI_cos2', 'NDVI_sin2', 'NDVI_amp2',
    # EVI harmonic (Tier 1)
    'EVI_c', 'EVI_cos1', 'EVI_sin1',
    # LSWI harmonic (Tier 1)
    'LSWI_c', 'LSWI_cos1', 'LSWI_sin1',
    # GCVI harmonic (Tier 2)
    'GCVI_c', 'GCVI_cos1', 'GCVI_sin1',
    # Robust features
    'NDVI_max', 'NDVI_amp', 'NDVI_mean', 'NDVI_std',
]

FEATURE_SETS['v5'] = [                          # 15 features — Weather-Invariant
    # Shape metrics from V3 (timing-independent behavior)
    'shape_mid_dip', 'shape_integrated_ndvi', 'shape_peak_skewness', 
    'shape_season_length', 'shape_winter_ratio',
    # Harmonics (Amplitudes and Constants only — no Phase/Cos/Sin)
    'NDVI_c', 'NDVI_amp1', 'NDVI_amp2',
    'EVI_c', 'LSWI_c', 'GCVI_c',
    # Robust Annual Stats
    'NDVI_max', 'NDVI_amp', 'NDVI_mean', 'NDVI_std',
]

FEATURE_SETS['v6'] = FEATURE_SETS['v5'] + [    # 18 features — Contextual & Structural
    'text_ndvi', 'topo_slope', 'hydro_ndwi'
]

FEATURE_SETS['v7'] = [                          # 14 features — Super-Robust
    # Pure annual stats (No month-specific snapshots)
    'NDVI_max', 'NDVI_amp', 'NDVI_mean', 'NDVI_std',
    'NDVI_c', 'NDVI_amp1', 'NDVI_amp2',
    'EVI_c', 'LSWI_c', 'GCVI_c',
    # Environmental context (Weather independent)
    'topo_slope', 'hydro_ndwi',
    # Robust curve shape
    'shape_peak_skewness', 'shape_winter_ratio'
]

# Quick reference
VERSIONS = list(FEATURE_SETS.keys())
for v in VERSIONS:
    assert len(FEATURE_SETS[v]) == len(set(FEATURE_SETS[v])), f"Duplicate features in {v}"


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT ANALYSIS TOOLS (Phase 1 & 2 Integration)
# ══════════════════════════════════════════════════════════════════════════════

def get_spectral_wisdom(monthly_dict, cdl_image, region, scale=3000):
    """
    Statewide comparison of indices for target crops (Phase 1 Logic).
    Calculates mean NDVI and GCVI for Corn and Soybeans in July.
    """
    import pandas as pd
    july_img = monthly_dict[7].select(['NDVI', 'GCVI'])
    
    results = []
    for code, label in [(1, 'Corn'), (5, 'Soybeans')]:
        mask = cdl_image.eq(code)
        stats = july_img.updateMask(mask).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e8
        ).getInfo()
        results.append({'Crop': label, 'NDVI': stats['NDVI'], 'GCVI': stats['GCVI']})
    
    df = pd.DataFrame(results).set_index('Crop')
    return df


def analyze_phenology(monthly_dict, samples, sample_labels):
    """
    Sampling and aggregation logic for phenology exploration (Phase 2 Logic).
    Returns aggregated monthly stats and shape metrics summary.
    """
    import pandas as pd
    
    # 1. Monthly Sampling
    records = []
    print("  Monthly sampling progress:")
    for month in range(1, 13):
        composite = monthly_dict[month].select(['NDVI', 'EVI', 'LSWI', 'GCVI'])
        sampled = composite.sampleRegions(collection=samples, scale=30, geometries=False)
        features = sampled.getInfo()['features']
        for f in features:
            props = f['properties']
            props['month'] = month
            records.append(props)
        print(f"    - Month {month:02d} ({len(features)} samples) [OK]")
            
    df_stats = pd.DataFrame(records)
    df_stats['crop_name'] = df_stats['cdl_code'].map(sample_labels)
    
    # 2. Aggregation
    monthly_agg = (
        df_stats.groupby(['month', 'crop_name'])
        .agg(
            NDVI_mean=('NDVI', 'mean'), NDVI_std=('NDVI', 'std'),
            EVI_mean=('EVI',   'mean'), EVI_std=('EVI',   'std'),
            LSWI_mean=('LSWI', 'mean'), LSWI_std=('LSWI', 'std'),
            GCVI_mean=('GCVI', 'mean'), GCVI_std=('GCVI', 'std')
        )
        .reset_index()
    )
    
    # 3. Shape Metrics
    shape_img = build_shape_image(monthly_dict)
    shape_samples = (shape_img.sampleRegions(collection=samples, scale=30, geometries=False)
                     .getInfo()['features'])
    df_shapes = pd.DataFrame([f['properties'] for f in shape_samples])
    df_shapes['crop_name'] = df_shapes['cdl_code'].map(sample_labels)
    shape_summary = df_shapes.groupby('crop_name')[['shape_season_length', 'shape_winter_ratio', 'shape_mid_dip']].mean()
    
    return df_stats, monthly_agg, shape_summary


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_image(version, sources):
    """
    Stack registry features into a single multi-band ee.Image.

    Parameters
    ----------
    version : str
        One of 'v1', 'v2', 'v3', 'v4'
    sources : dict[str, ee.Image]
        Maps each source_key used by this version to an ee.Image.
        Required keys depend on which features are in the version:
          monthly composites -> 'jan', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'nov'
          shape features     -> 'shapes'
          harmonic features  -> 'harm_NDVI', 'harm_EVI', 'harm_LSWI', 'harm_GCVI'

    Returns
    -------
    (ee.Image, list[str])
        Stacked image with one band per feature + ordered list of band names.
    """
    feature_names = FEATURE_SETS[version]
    bands = []
    for name in feature_names:
        src_key, band = FEATURE_REGISTRY[name]
        if src_key not in sources:
            raise KeyError(
                f"source '{src_key}' missing (needed for '{name}'). "
                f"Available: {sorted(sources.keys())}"
            )
        bands.append(sources[src_key].select(band).rename(name))
    return ee.Image.cat(bands), feature_names


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_shape_image(monthly):
    """
    Build the five seasonal-curve shape features from monthly composites.
    Uses Reducers to be robust against missing/masked months (clouds).
    """
    m = monthly  # shorthand

    # 1. Mid-season dip — June NDVI relative to May/July average
    #    Double-crop signature. Robust to missing May or July.
    may_july_avg = ee.Image.cat([m[5].select('NDVI'), m[7].select('NDVI')]).reduce(ee.Reducer.mean())
    mid_dip = (
        m[6].select('NDVI')
        .subtract(may_july_avg)
        .rename('shape_mid_dip')
    )

    # 2. Integrated NDVI across the growing season (Apr–Nov)
    #    Reducer.sum() is robust: ignores masked months.
    integrated_ndvi = (
        ee.Image.cat([m[k].select('NDVI') for k in range(4, 12)])
        .reduce(ee.Reducer.sum())
        .rename('shape_integrated_ndvi')
    )

    # 3. Peak skewness — weighted center of mass of NDVI across growing season
    ndvi_growing = ee.Image.cat([m[k].select('NDVI') for k in range(4, 12)])
    month_weights = ee.Image.constant(list(range(4, 12)))
    weighted_sum  = ndvi_growing.multiply(month_weights).reduce(ee.Reducer.sum())
    total_ndvi    = ndvi_growing.reduce(ee.Reducer.sum()).add(0.001)
    peak_skewness = weighted_sum.divide(total_ndvi).rename('shape_peak_skewness')

    # 4. Season length — number of months with NDVI > 0.4
    ndvi_stack = ee.Image.cat([m[k].select('NDVI') for k in range(1, 13)])
    season_length = ndvi_stack.gt(0.4).reduce(ee.Reducer.sum()).rename('shape_season_length')

    # 5. Winter ratio — Jan NDVI / Jul NDVI
    winter_ratio = (
        m[1].select('NDVI')
        .divide(m[7].select('NDVI').add(0.001))
        .rename('shape_winter_ratio')
    )

    return ee.Image.cat([mid_dip, integrated_ndvi, peak_skewness,
                         season_length, winter_ratio])


# ══════════════════════════════════════════════════════════════════════════════
# HARMONIC FEATURE BUILDER  (V4)
# ══════════════════════════════════════════════════════════════════════════════

def build_harmonic_image(collection, index_name):
    """
    Fit  y(t) = c + A1*cos(2pi*t) + B1*sin(2pi*t) + A2*cos(4pi*t) + B2*sin(4pi*t)
    where t = DOY / 365, using GEE's linearRegression reducer.

    Returns ee.Image with bands:
        constant, cos1, sin1, cos2, sin2, amp1, phase1, amp2
    """
    TWO_PI  = 2 * math.pi
    FOUR_PI = 4 * math.pi

    def add_harmonic_bands(image):
        t = ee.Number(image.date().getRelative('day', 'year')).divide(365)
        return image.addBands([
            ee.Image.constant(1).rename('constant').float(),
            ee.Image(t.multiply(TWO_PI).cos()).rename('cos1').float(),
            ee.Image(t.multiply(TWO_PI).sin()).rename('sin1').float(),
            ee.Image(t.multiply(FOUR_PI).cos()).rename('cos2').float(),
            ee.Image(t.multiply(FOUR_PI).sin()).rename('sin2').float(),
        ])

    independents = ['constant', 'cos1', 'sin1', 'cos2', 'sin2']
    col_h        = collection.map(add_harmonic_bands)

    regression = col_h.select(independents + [index_name]).reduce(
        ee.Reducer.linearRegression(numX=len(independents), numY=1)
    )

    coeffs = (regression.select('coefficients')
              .arrayFlatten([independents, ['coef']]))

    c    = coeffs.select('constant_coef').rename('constant')
    cos1 = coeffs.select('cos1_coef').rename('cos1')
    sin1 = coeffs.select('sin1_coef').rename('sin1')
    cos2 = coeffs.select('cos2_coef').rename('cos2')
    sin2 = coeffs.select('sin2_coef').rename('sin2')
    amp1   = cos1.pow(2).add(sin1.pow(2)).sqrt().rename('amp1')
    phase1 = sin1.atan2(cos1).rename('phase1')
    amp2   = cos2.pow(2).add(sin2.pow(2)).sqrt().rename('amp2')

    return ee.Image.cat([c, cos1, sin1, cos2, sin2, amp1, phase1, amp2])


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT FEATURE BUILDER (V6)
# ══════════════════════════════════════════════════════════════════════════════

def build_context_image(monthly_dict, region):
    """Build spatial and environmental context features (V6)."""
    # 1. Texture: NDVI standard deviation in a 3x3 window (July composite)
    july_ndvi = monthly_dict[7].select('NDVI')
    texture = july_ndvi.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(1),
    ).rename('text_ndvi')

    # 2. Terrain: Slope from SRTM
    elevation = ee.Image('USGS/SRTMGL1_003').clip(region)
    slope = ee.Terrain.slope(elevation).rename('topo_slope')

    # 3. Hydrology: Max annual NDWI (Normalized Difference Water Index)
    # NDWI = (Green - NIR) / (Green + NIR)
    # Landsat 8/9: Green=SR_B3, NIR=SR_B5
    def calc_ndwi(img):
        return img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    
    ndwi_stack = ee.Image.cat([calc_ndwi(monthly_dict[m]) for m in range(1, 13)])
    max_ndwi = ndwi_stack.reduce(ee.Reducer.max()).rename('hydro_ndwi')

    return ee.Image.cat([texture.float(), slope.float(), max_ndwi.float()])


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST STATS BUILDER (V4/V5/V6)
# ══════════════════════════════════════════════════════════════════════════════

def build_robust_image(monthly_dict):
    """Build weather-invariant features from monthly NDVI composites."""
    ndvi_stack = ee.Image.cat([monthly_dict[m].select('NDVI') for m in range(1, 13)])
    return ee.Image.cat([
        ndvi_stack.reduce(ee.Reducer.max()).rename('NDVI_max'),
        ndvi_stack.reduce(ee.Reducer.max()).subtract(ndvi_stack.reduce(ee.Reducer.min())).rename('NDVI_amp'),
        ndvi_stack.reduce(ee.Reducer.mean()).rename('NDVI_mean'),
        ndvi_stack.reduce(ee.Reducer.stdDev()).rename('NDVI_std')
    ]).toFloat()


if __name__ == '__main__':
    print('feature_registry.py OK')
    for v, feats in FEATURE_SETS.items():
        print(f'  {v}: {len(feats)} features')
