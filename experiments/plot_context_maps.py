"""Figure 3: the stations on the E-OBS mean precipitation of 1980-2025, on the ERA5 0.25° grid and the E-OBS 0.1° grid.
The means are cached in data/context_maps_climatology.npz; recomputing them needs the E-OBS and ERA5 files of DATA."""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import netCDF4
import numpy as np
import shapely
import xarray as xr
from shapely.geometry import box

import paths

DATA = Path("/Volumes/Extreme SSD/DuongNGUYEN/DATA")
ECAD_STATIONS_FILE = paths.ECAD_STATIONS
EOBS_FILE = DATA / "EOBS/rr_ens_mean_0.1deg_reg_v33.0e.nc"
ERA5_T2M = DATA / "ERA5/slevel/t2m_{year}.nc"
OUT_DIR = paths.RESULTS / "context_maps"
CACHE_FILE = paths.ROOT / "data" / "context_maps_climatology.npz"

YEARS = range(1980, 2026)
LAT_MIN, LAT_MAX = 41.0, 52.0
LON_MIN, LON_MAX = -6.0, 10.5
EXTENT = [-5.5, 10.0, 41.2, 51.4]

# STAID: short label, label offset in points
STATIONS = {
    434: ("Brest", (-6, 7)),
    737: ("Lille", (7, 4)),
    323: ("Strasbourg", (-7, 7)),
    2190: ("Tours", (7, 4)),
    34: ("Bordeaux", (-7, 4)),
    750: ("Clermont-Ferrand", (7, 4)),
    756: ("Tarbes", (7, -12)),
}
# Shown on the fine map only.
FINE_ONLY_STATIONS = {
    786: ("Montélimar", (7, 4)),
    36: ("Perpignan", (7, -4)),
}

INK, INK_2 = "#0b0b0b", "#52514e"
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def dms_to_decimal(value):
    value = value.strip()
    sign = -1.0 if value[0] == "-" else 1.0
    degree, minute, second = map(float, value[1:].split(":"))
    return sign * (degree + minute / 60.0 + second / 3600.0)


def read_stations():
    lines = ECAD_STATIONS_FILE.read_text(encoding="latin-1").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("STAID"))
    stations = {}
    for raw in csv.DictReader(lines[header:]):
        row = {key.strip(): value.strip() for key, value in raw.items()}
        staid = int(row["STAID"])
        if staid in STATIONS or staid in FINE_ONLY_STATIONS:
            stations[staid] = {"lat": dms_to_decimal(row["LAT"]), "lon": dms_to_decimal(row["LON"])}
    return stations


def compute_climatology():
    """Mean ERA5 t2m [°C] and mean E-OBS rr [mm/day] over France, 1980-2025."""
    t2m_sum, n_hours = None, 0
    for year in YEARS:
        with xr.open_dataset(str(ERA5_T2M).format(year=year)) as ds:
            t2m = ds["t2m"].sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            year_sum = t2m.sum("valid_time").values.astype(np.float64)
            n_hours += t2m.sizes["valid_time"]
            era5_lat, era5_lon = t2m.latitude.values, t2m.longitude.values
        t2m_sum = year_sum if t2m_sum is None else t2m_sum + year_sum
        print(f"ERA5 {year}", flush=True)

    with netCDF4.Dataset(EOBS_FILE) as src:
        lat = np.asarray(src["latitude"][:])
        lon = np.asarray(src["longitude"][:])
        iy = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
        ix = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
        i0 = int(netCDF4.date2index(datetime(1980, 1, 1), src["time"], select="exact"))
        i1 = int(netCDF4.date2index(datetime(2025, 12, 31), src["time"], select="exact"))
        rr_sum = np.zeros((iy.size, ix.size))
        rr_count = np.zeros((iy.size, ix.size))
        for start in range(i0, i1 + 1, 365):
            stop = min(start + 365, i1 + 1)
            rr = np.ma.filled(src["rr"][start:stop, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], np.nan)
            rr_sum += np.nansum(rr, axis=0)
            rr_count += np.isfinite(rr).sum(axis=0)
        rr_mean = np.where(rr_count > 0, rr_sum / np.maximum(rr_count, 1), np.nan)

    np.savez(CACHE_FILE,
             era5_lat=era5_lat, era5_lon=era5_lon, t2m=t2m_sum / n_hours - 273.15,
             eobs_lat=lat[iy], eobs_lon=lon[ix], rr=rr_mean)


def france_mask(lat, lon):
    """True where the grid-cell centre lies in metropolitan France (incl. Corsica)."""
    countries = shpreader.Reader(shpreader.natural_earth("10m", "cultural", "admin_0_countries"))
    france = next(c.geometry for c in countries.records() if c.attributes["ADM0_A3"] == "FRA")
    france = france.intersection(box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
    lon2d, lat2d = np.meshgrid(lon, lat)
    return shapely.contains_xy(france, lon2d, lat2d)


def aggregate_to_grid(field, lat, lon, grid_lat, grid_lon, step):
    """Average the fine cells whose centres fall inside each coarse cell (NaN-aware)."""
    coarse = np.full((grid_lat.size, grid_lon.size), np.nan)
    h = step / 2
    for i, cell_lat in enumerate(grid_lat):
        rows = (lat >= cell_lat - h) & (lat < cell_lat + h)
        for j, cell_lon in enumerate(grid_lon):
            block = field[np.ix_(rows, (lon >= cell_lon - h) & (lon < cell_lon + h))]
            if np.isfinite(block).any():
                coarse[i, j] = np.nanmean(block)
    return coarse


def nearest_cell(lat, lon, grid_lat, grid_lon):
    return float(grid_lat[np.abs(grid_lat - lat).argmin()]), float(grid_lon[np.abs(grid_lon - lon).argmin()])


def draw_map(field, grid_lat, grid_lon, step, cmap, vmin, vmax, cbar_label, title, subtitle,
             stations, coarse, out_stem):
    geo = ccrs.PlateCarree()
    proj = ccrs.LambertConformal(central_longitude=2.5, central_latitude=46.5, standard_parallels=(44, 49))

    fig = plt.figure(figsize=(7.5, 8.2))
    ax = fig.add_subplot(projection=proj)
    ax.set_extent(EXTENT, crs=geo)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#f3f6f9", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ecebe7", zorder=0)

    inside = france_mask(grid_lat, grid_lon)
    station_cells = [nearest_cell(s["lat"], s["lon"], grid_lat, grid_lon) for s in stations]
    for cell_lat, cell_lon in station_cells:
        # keep the station cells whose centre lies over the sea
        inside[np.argmin(np.abs(grid_lat - cell_lat)), np.argmin(np.abs(grid_lon - cell_lon))] = True

    if coarse:
        # One polygon per France cell, so the white cell edges stop at the French border.
        h = step / 2
        iy, ix = np.nonzero(inside)
        verts = [[(grid_lon[j] - h, grid_lat[i] - h), (grid_lon[j] + h, grid_lat[i] - h),
                  (grid_lon[j] + h, grid_lat[i] + h), (grid_lon[j] - h, grid_lat[i] + h)]
                 for i, j in zip(iy, ix)]
        mesh = PolyCollection(verts, array=field[iy, ix], cmap=cmap, transform=geo,
                              edgecolors="white", linewidths=0.3, zorder=1)
        mesh.set_clim(vmin if vmin is not None else field[inside].min(),
                      vmax if vmax is not None else field[inside].max())
        ax.add_collection(mesh)
    else:
        mesh = ax.pcolormesh(grid_lon, grid_lat, np.ma.masked_where(~inside, field), transform=geo,
                             shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#6f6e69", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#6f6e69", linewidth=0.6, zorder=2)

    for s, (cell_lat, cell_lon) in zip(stations, station_cells):
        if coarse:  # a 0.1° box is too small to see at country scale
            h = step / 2
            ax.plot([cell_lon - h, cell_lon + h, cell_lon + h, cell_lon - h, cell_lon - h],
                    [cell_lat - h, cell_lat - h, cell_lat + h, cell_lat + h, cell_lat - h],
                    transform=geo, color=INK, linewidth=1.6, zorder=4)
        ax.scatter(s["lon"], s["lat"], transform=geo, s=14, color=INK, edgecolor="white",
                   linewidth=0.8, zorder=5)
        dx, dy = s["offset"]
        ax.annotate(s["label"], (s["lon"], s["lat"]), xycoords=geo._as_mpl_transform(ax),
                    xytext=(dx, dy), textcoords="offset points", ha="left" if dx >= 0 else "right",
                    fontsize=8.5, fontweight="bold", color=INK, path_effects=HALO, zorder=6)

    ax.set_title(f"{title}\n", fontsize=14, fontweight="bold", color=INK, pad=4)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color=INK_2)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, shrink=0.7)
    cbar.set_label(cbar_label, fontsize=9, color=INK_2)
    cbar.ax.tick_params(labelsize=8, colors=INK_2)
    cbar.outline.set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(OUT_DIR / f"{out_stem}.png", paths.FIGURES / f"{out_stem}.png")
    print("Saved", OUT_DIR / f"{out_stem}.png", "and copied it to", paths.FIGURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context maps of the ECA&D stations.")
    parser.add_argument("--exclude", nargs="*", default=[], help="station labels left out")
    parser.add_argument("--all-on-coarse", action="store_true",
                        help="draw every station on the coarse map too (no fine-only stations)")
    parser.add_argument("--suffix", default="", help="appended to the output file names")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if not CACHE_FILE.exists():
        compute_climatology()
    clim = np.load(CACHE_FILE)

    coords = read_stations()
    stations = [{**coords[staid], "label": label, "offset": offset}
                for staid, (label, offset) in STATIONS.items() if label not in args.exclude]
    fine_only = [{**coords[staid], "label": label, "offset": offset}
                 for staid, (label, offset) in FINE_ONLY_STATIONS.items() if label not in args.exclude]
    if args.all_on_coarse:
        stations, fine_only = stations + fine_only, []

    rr_coarse = aggregate_to_grid(clim["rr"], clim["eobs_lat"], clim["eobs_lon"],
                                  clim["era5_lat"], clim["era5_lon"], 0.25)
    rr_label = "Mean daily precipitation, 1980–2025 [mm/day]"

    draw_map(rr_coarse, clim["era5_lat"], clim["era5_lon"], 0.25, "Blues", 1.0, 5.0, rr_label,
             "Coarse resolution: 0.25° × 0.25°",
             "dots = ECA&D stations · boxes = ERA5 cell",
             stations, coarse=True, out_stem="context_map_era5_coarse" + args.suffix)

    draw_map(clim["rr"], clim["eobs_lat"], clim["eobs_lon"], 0.1, "Blues", 1.0, 5.0, rr_label,
             "Fine resolution: 0.1° × 0.1°",
             "dots = ECA&D stations",
             stations + fine_only, coarse=False, out_stem="context_map_eobs_fine" + args.suffix)
