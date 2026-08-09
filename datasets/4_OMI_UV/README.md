# OMI UV aerosol index

Independent sensor cross-check. Obtained through NASA Giovanni, same route as MODIS.

## Steps

1. Open https://giovanni.gsfc.nasa.gov/giovanni/
2. Keyword `OMAERUVd`, select **UV Aerosol Index (OMAERUVd v003)**
3. Plot type **Time Series, Area-Averaged**
4. Date range `2003-01-01` to `2024-12-31`
5. Region `38,30,50,40`
6. Download as CSV, place in this folder

Record begins 2004-10-01.

## Important limitation

The OMI row anomaly progressively corrupts this retrieval. Correlation with the
MODIS target is 0.46 to 0.71 through 2012, then falls to near zero and turns
negative from 2014 onward, and the series carries implausible values above 5.

The analysis therefore uses this index only over 2004–2012, and excludes it from the
model feature set entirely. See `code/s7_robustness.py`.

## Citation

Torres, O. (2008). OMI/Aura Near UV Aerosol Optical Depth and Single Scattering
Albedo L3 1 day 1.0 degree x 1.0 degree V3. NASA Goddard Earth Sciences Data and
Information Services Center. doi:10.5067/Aura/OMI/DATA3003

Substitute the citation shown on the Giovanni variable page if it differs.
