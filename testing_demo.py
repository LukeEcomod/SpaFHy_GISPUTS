# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:24:03 2022

@author: janousu
"""

from tools import dem_from_mml, write_AsciiGrid_new
from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import xarray as xr
import os
from scipy import ndimage
eps = np.finfo(float).eps
# inputs
subset = [370000,7537500,390000,7557500]
catchment_name = 'Ylisenpaanjankka' # Pallasjoki, Lompolonjankka, Ylisenpaanjankka, Lompolonoja, Pyhajoki
outfolder = 'F:\Pallaslake_Catchment\GIS_inputs'
outpath = os.path.join(outfolder, catchment_name)
if not os.path.exists(outpath):
    os.makedirs(outpath)

outlet_x, outlet_y = 380837.0316, 7552114.662
routing = 'd8'

# 0.125 = 16m, 0.0625 = 32m, 0.03125 = 64m
raster, path = dem_from_mml(outpath=outpath, subset=subset, scalefactor=0.125)

#%%

raster = xr.open_rasterio(path)
grid = Grid.from_raster(path)
dem = grid.read_raster(path)

#%%

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)


plt.imshow(dem, extent=grid.extent, cmap='terrain', vmin=np.quantile(dem, 0.1), vmax= np.quantile(dem, 1), zorder=1)
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

#%%

# Condition DEM
# ----------------------
# Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)

# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)

# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32) # [N, NE, E, SE, S, SW, W, NW]

# Compute flow directions
# -------------------------------------
fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
slope = grid.cell_slopes(inflated_dem, fdir)
aspect = grid.flowdir(inflated_dem, dirmap=dirmap, routing='d8')

# Calculate flow accumulation
# --------------------------
acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)
cell_area = dem.affine[0] * dem.affine[0]
acc_new = acc.copy() + 1

twi = np.log(acc_new / (np.tan(slope) + eps))

# Saving the processed dem
grid.to_ascii(inflated_dem, os.path.join(outpath, 'inflated_dem.asc'))
grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}.asc'))
grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}.asc'))

#%%

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(flooded_dem, extent=grid.extent, cmap='terrain', vmin=np.quantile(flooded_dem, 0.1), vmax= np.quantile(flooded_dem, 1), zorder=1)
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

#%%

fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()

#%%

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

#%%

# Delineate a catchment
# ---------------------

# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (outlet_x, outlet_y))

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                       xytype='coordinate', routing=routing)

# Crop and plot the catchment
# ---------------------------
# Clip the bounding box to the catchment
#grid.clip_to(catch)
#clipped_catch = grid.view(catch)

cmask_temp = np.ones(shape=catch.shape)
cmask_temp[catch == False] = 0
cmask_temp = ndimage.binary_fill_holes(cmask_temp).astype(int)
cmask = np.ones(shape=cmask_temp.shape)
cmask[cmask_temp == 0] = int(-9999)

info = {'ncols':dem.shape[0],
        'nrows':dem.shape[1],
        'xllcorner':dem.bbox[0],
        'yllcorner':dem.bbox[1],
        'cellsize':dem.affine[0],
        'NODATA_value':-9999}

fname=os.path.join(outpath, f'cmask_{routing}.asc')
write_AsciiGrid_new(fname, cmask, info)

#%%

# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(catch, catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r', alpha=0.3)
raster.plot(ax=ax, vmin=200, vmax=600)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear', alpha=0.1)
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)


#%%

# Extract river network
# ---------------------
branches = grid.extract_river_network(fdir, acc > 1000, dirmap=dirmap)

#%%

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

_ = plt.title('D8 channels', size=14)

#%%

# Calculate distance to outlet from each cell
# -------------------------------------------
dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                               xytype='coordinate')

#%%


fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(dist, extent=grid.extent, zorder=2,
               cmap='cubehelix_r')
plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow Distance', size=14)

#%%

# Combine with land cover data
# ---------------------
terrain = grid.read_raster('impervious_area.tiff', window=grid.bbox,
                           window_crs=grid.crs, nodata=0)
# Reproject data to grid's coordinate reference system
projected_terrain = terrain.to_crs(grid.crs)
# View data in catchment's spatial extent
catchment_terrain = grid.view(projected_terrain, nodata=np.nan)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(catchment_terrain, extent=grid.extent, zorder=2,
               cmap='bone')
plt.colorbar(im, ax=ax, label='Percent impervious area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Percent impervious area', size=14)

#%%

# Convert catchment raster to vector and combine with soils shapefile
# ---------------------
# Read soils shapefile
import pandas as pd
import geopandas as gpd
from shapely import geometry, ops
soils = gpd.read_file('soils.shp')
soil_id = 'MUKEY'
# Convert catchment raster to vector geometry and find intersection
shapes = grid.polygonize()
catchment_polygon = ops.unary_union([geometry.shape(shape)
                                     for shape, value in shapes])
soils = soils[soils.intersects(catchment_polygon)]
catchment_soils = gpd.GeoDataFrame(soils[soil_id],
                                   geometry=soils.intersection(catchment_polygon))
# Convert soil types to simple integer values
soil_types = np.unique(catchment_soils[soil_id])
soil_types = pd.Series(np.arange(soil_types.size), index=soil_types)
catchment_soils[soil_id] = catchment_soils[soil_id].map(soil_types)

fig, ax = plt.subplots(figsize=(8, 6))
catchment_soils.plot(ax=ax, column=soil_id, categorical=True, cmap='terrain',
                     linewidth=0.5, edgecolor='k', alpha=1, aspect='equal')
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.set_title('Soil types (vector)', size=14)

#%%

soil_polygons = zip(catchment_soils.geometry.values, catchment_soils[soil_id].values)
soil_raster = grid.rasterize(soil_polygons, fill=np.nan)


fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(soil_raster, cmap='terrain', extent=grid.extent, zorder=1)
boundaries = np.unique(soil_raster[~np.isnan(soil_raster)]).astype(int)
plt.colorbar(boundaries=boundaries,
             values=boundaries)
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.set_title('Soil types (raster)', size=14)















