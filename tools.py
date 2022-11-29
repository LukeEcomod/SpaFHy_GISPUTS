# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:05:11 2022

@author: janousu
"""

import os
import urllib
import rasterio
from numpy import newaxis
from rasterio.plot import show
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
from pysheds.grid import Grid
from scipy import ndimage
import pandas as pd
from matplotlib import colors
import warnings

def dem_from_mml(outpath, subset, layer='korkeusmalli_2m', form='image/tiff', scalefactor=0.125, plot=False, cmap='terrain'):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        '''

    # The base url for maanmittauslaitos
    url = 'https://beta-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v1?'
    scalefactorstr = f'SCALEFACTOR={scalefactor}'
    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  compression='geotiff:compression=LZW',
                  scalefactor=scalefactorstr)

    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)

    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)

    # Open the file with the url:
    raster = rasterio.open(r[0])

    del r
    res = int(2/scalefactor)
    layer = f'korkeusmalli_{res}m'
    out_fp = os.path.join(outpath, layer) + '.tif'

    # Copy the metadata
    out_meta = raster.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs']
                         }
                    )

    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)

    raster_dem = rasterio.open(out_fp)

    return raster_dem, out_fp


def orto_from_mml(outpath, subset, layer='ortokuva_vari', form='image/tiff', scalefactor=0.01, plot=False):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        '''
    scalefactorstr = f'SCALEFACTOR={scalefactor}'

    # The base url for maanmittauslaitos
    url = 'https://beta-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v1?'
    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  scalefactor=scalefactorstr)

    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)

    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)

    # Open the file with the url:
    raster = rasterio.open(r[0])

    del r
    res = int(2/scalefactor)
    layer = f'korkeusmalli_{res}m'
    out_fp = os.path.join(outpath, layer) + '.tif'

    # Copy the metadata
    out_meta = raster.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs']
                         }
                    )

    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)

    raster_dem = rasterio.open(out_fp)

    return raster_dem, out_fp



def read_AsciiGrid(fname, setnans=True):
    """
    reads AsciiGrid format in fixed format as below:
        ncols         750
        nrows         375
        xllcorner     350000
        yllcorner     6696000
        cellsize      16
        NODATA_value  -9999
        -9999 -9999 -9999 -9999 -9999
        -9999 4.694741 5.537514 4.551162
        -9999 4.759177 5.588773 4.767114
    IN:
        fname - filename (incl. path)
    OUT:
        data - 2D numpy array
        info - 6 first lines as list of strings
        (xloc,yloc) - lower left corner coordinates (tuple)
        cellsize - cellsize (in meters?)
        nodata - value of nodata in 'data'
    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    fid = open(fname, 'r')
    info = fid.readlines()[0:6]
    fid.close()

    # print info
    # conversion to float is needed for non-integers read from file...
    xloc = float(info[2].split(' ')[-1])
    yloc = float(info[3].split(' ')[-1])
    cellsize = float(info[4].split(' ')[-1])
    nodata = float(info[5].split(' ')[-1])

    # read rest to 2D numpy array
    data = np.loadtxt(fname, skiprows=6)

    if setnans is True:
        data[data == nodata] = np.NaN
        nodata = np.NaN

    data = np.array(data, ndmin=2)

    return data, info, (xloc, yloc), cellsize, nodata


def write_AsciiGrid(fname, data, info, fmt='%.18e'):
    """ writes AsciiGrid format txt file
    IN:
        fname - filename
        data - data (numpy array)
        info - info-rows (list, 6rows)
        fmt - output formulation coding

    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    # replace nans with nodatavalue according to info
    nodata = int(info[-1].split(' ')[-1])
    data[np.isnan(data)] = nodata
    # write info
    fid = open(fname, 'w')
    fid.writelines(info)
    fid.close()

    # write data
    fid = open(fname, 'a')
    np.savetxt(fid, data, fmt=fmt, delimiter=' ')
    fid.close()

def write_AsciiGrid_new(fname, data, info, fmt='%.18e'):
    """ writes AsciiGrid format txt file
    IN:
        fname - filename
        data - data (numpy array)
        info - info-rows (list, 6rows)
        fmt - output formulation coding

    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    # write info
    fid = open(fname, 'w')
    fid.write("ncols " + str(info['ncols']) + "\n")
    fid.write("nrows " + str(info['nrows']) + "\n")
    fid.write("xllcorner " + str(info['xllcorner']) + "\n")
    fid.write("yllcorner " + str(info['yllcorner']) + "\n")
    fid.write("cellsize " + str(info['cellsize']) + "\n")
    fid.write("NODATA_value " + str(info['NODATA_value']) + "\n")
    #fid.write(data)
    fid.close()

    # write data
    fid = open(fname, 'a')
    np.savetxt(fid, data, fmt=fmt, delimiter=' ')
    fid.close()

def delineate_catchment(outfolder, catchment_name, subset, outlet_file, routing='d8', dem_scale=0.125, plot_catchment=True):

    outlets = pd.read_csv(outlet_file, sep=';', encoding = "ISO-8859-1")
    outlet_x = float(outlets.loc[outlets['stream'] == catchment_name, 'lon'])
    outlet_y = float(outlets.loc[outlets['stream'] == catchment_name, 'lat'])

    outpath = os.path.join(outfolder, catchment_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    raster, path = dem_from_mml(outpath=outpath, subset=subset, scalefactor=0.125)
    warnings.simplefilter("ignore", UserWarning)

    raster = xr.open_rasterio(path)
    grid = Grid.from_raster(path)
    dem = grid.read_raster(path)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)

    grid.to_ascii(inflated_dem, os.path.join(outpath, f'inflated_dem_{catchment_name}.asc'))
    grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}_{catchment_name}.asc'))
    grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}_{catchment_name}.asc'))

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

    fname=os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc')
    write_AsciiGrid_new(fname, cmask, info)

    if plot_catchment == True:
        # Plot the catchment
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        im = ax.imshow(np.where(catch, catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r', alpha=0.3)
        raster.plot(ax=ax, vmin=200, vmax=600)
        ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear', alpha=0.1)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title(f'Delineated {catchment_name} catchment', size=14)
























