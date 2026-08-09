# MODIS aerosol optical depth

Obtained through the NASA Giovanni web interface rather than by script. Bulk
download of the Level-3 granules is roughly 500 GB for a subset that reduces to a
few hundred kilobytes, so Giovanni's server-side area averaging is the sensible route.

Free NASA Earthdata account required: https://urs.earthdata.nasa.gov

## Steps

1. Open https://giovanni.gsfc.nasa.gov/giovanni/
2. Keyword `MOD08_D3`, select
   **Combined Dark Target and Deep Blue AOD at 0.55 micron for land and ocean: Mean**
3. Plot type **Time Series, Area-Averaged**
4. Date range `2003-01-01` to `2024-12-31`, times `00:00` to `23:59`
5. Region `38,30,50,40`
6. Plot, then Download as CSV
7. Repeat with keyword `MYD08_D3` for the Aqua sensor

Two CSV files result, one per satellite. Place both in this folder.

The analysed grid Giovanni returns is 38.5–49.5 E, 30.5–39.5 N, slightly inside the
requested box. Report the analysed grid, not the requested one.

## Citation

Platnick, S., et al. (2015). MODIS Atmosphere L3 Daily Product. NASA MODIS Adaptive
Processing System, Goddard Space Flight Center.
doi:10.5067/MODIS/MOD08_D3.061 and doi:10.5067/MODIS/MYD08_D3.061

Acknowledge Giovanni: analyses and visualisations used in this study were produced
with the Giovanni online data system, developed and maintained by the NASA GES DISC.
