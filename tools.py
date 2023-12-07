# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:05:11 2022

@author: janousu
"""

import os
import glob
import urllib
import rasterio
from numpy import newaxis
import rasterio
import rasterio.plot
from rasterio import features
from rasterio.windows import from_bounds
from rasterio.plot import show
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pysheds.grid import Grid
from scipy import ndimage
import pandas as pd
from matplotlib import colors
import warnings
import geopandas as gpd

def dem_from_mml_beta(outpath, subset, layer='korkeusmalli_2m', form='image/tiff', scalefactor=0.125, plot=False, cmap='terrain'):

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

def dem_from_mml(outpath, subset, apikey, layer='korkeusmalli_2m', form='image/tiff', scalefactor=0.125, plot=False, cmap='terrain', ):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        '''

    # The base url for maanmittauslaitos
    url = 'https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2?'
    scalefactorstr = f'SCALEFACTOR={scalefactor}'
    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  compression='geotiff:compression=LZW',
                  scalefactor=scalefactorstr,
                  api=f'api-key={apikey}')

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

def dem_from_puhti_2m(fp, subset, out_fp, plot=True, save_in='geotiff'):
    '''
    fp = file path to be downloaded
    subset = cropping to coordinate box
    out_fp = file path to be saved
    plot = show the raster
    '''
    with rasterio.open(fp) as src:
        data = src.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], src.transform))
        profile = src.profile

    out_meta = profile.copy()
    
    new_affine = rasterio.Affine(out_meta['transform'][0], 
                                 out_meta['transform'][1], 
                                 subset[0], 
                                 out_meta['transform'][3], 
                                 out_meta['transform'][4], 
                                 subset[3])
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                    "height": data.shape[0],
                    "width": data.shape[1],
                    "transform": new_affine,
                    "crs": profile['crs']
                        }
                    )
    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})
    
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        with rasterio.open(out_fp, 'w', **out_meta) as dst:
            src = dst.write(data, 1)
            if plot==True:
                plt.imshow(data)
                #show(src)
            # At the end of the ``with rasterio.Env()`` block, context
            # manager exits and all drivers are de-registered
    
    raster_dem = rasterio.open(out_fp)
    if plot==True:
        show(raster_dem)
        
    return raster_dem, out_fp    

def resample_raster(fp, out_fp, scale_factor=0.125, plot=True, save_in='geotiff'):
    '''
    fp = file path to be downloaded
    out_fp = file path to be saved
    scaling factor (e.g. if 2m to be resampled to 16m scale_factor=0.125   
    '''
    
    with rasterio.open(fp) as dataset:
        
        # resample data to target shape
        data = dataset.read(1, 
                out_shape=(dataset.count,int(dataset.height * scale_factor),int(dataset.width * scale_factor)),
                resampling=Resampling.bilinear
                )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )    
        out_meta = dataset.profile.copy()

        out_meta.update({"driver": "GTiff",
                 "height": data.shape[0],
                  "width": data.shape[1],
                  "transform": transform,
                        }
                    )
        if save_in=='asc':
            out_meta.update({"driver": "AAIGrid"})
        
        with rasterio.open(out_fp, 'w', **out_meta) as dst:
            src = dst.write(data, 1)
            
    raster = rasterio.open(out_fp)
    if plot==True:
        show(raster)
    return raster, out_fp

def orto_from_mml(outpath, subset, layer='ortokuva_vari', form='image/tiff', scalefactor=0.01, plot=False):

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
    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}')


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

def vmi_from_puhti(fd, subset, out_fd, layer='all', plot=True, save_in='geotiff'):
    if layer=='all':
        p = os.path.join(fd, '*.img')
    else:
        p = os.path.join(fd, layer)

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
        
    for file in glob.glob(p):
        print(file)
        out_fn = file.rpartition('/')[-1][:-4]
        if save_in == 'geotiff':
            out_fp = os.path.join(out_fd, out_fn) + '.tif'
        elif save_in == 'asc':
            out_fp = os.path.join(out_fd, out_fn) + '.asc'

        with rasterio.open(file) as src:
            data = src.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], src.transform))
            profile = src.profile
            out_meta = src.profile.copy()

            new_affine = rasterio.Affine(out_meta['transform'][0], 
                                         out_meta['transform'][1], 
                                         subset[0], 
                                         out_meta['transform'][3], 
                                         out_meta['transform'][4], 
                                         subset[3])
            
            if len(data.flatten()[data.flatten() == 32766]) > 0:
                print('*** Data has', len(data.flatten()[data.flatten() == 32766]), 'nan values (=32766) ***')
                print('--> converted to 0 ***')
            if len(data.flatten()[data.flatten() == 32767]) > 0:
                print('*** Data has', len(data.flatten()[data.flatten() == 32767]), 'non land values (=32767) ***')
                print('--> converted to 0 ***')
            data[data > 32765] = 0

            # Update the metadata for geotiff
            out_meta.update({"driver": "GTiff",
                             "height": data.shape[0],
                             "width": data.shape[1],
                              "transform": new_affine,
                              "nodata": -9999})
            
            if save_in == 'asc':
                out_meta.update({"driver": "AAIGrid"})
                
            with rasterio.Env():
                with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=True) as dst:
                    src = dst.write(data, 1)
    
        if plot==True:
            raster = rasterio.open(out_fp)
            show(raster, vmax=100)

def needlemass_to_lai(in_fn, out_fd, species, save_in='asc', plot=True):
    
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}

    bm_raster = rasterio.open(in_fn)
    bm_data = bm_raster.read(1)
    
    LAI = bm_data * 1e-3 * SLA[species]

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    out_meta = bm_raster.meta.copy()

    print(bm_raster.meta)

    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'LAI_{species}') + '.tif'
        out_meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'LAI_{species}') + '.asc'
        out_meta.update({"driver": "AAIGrid"})

    with rasterio.open(out_fn, 'w+', **out_meta) as out:
            src = out.write(LAI, 1)
    if plot==True:
        raster = rasterio.open(out_fn)
        show(raster)

def soilmap_from_puhti(soilmap, subset, out_fd, ref_raster, soildepth='surface', plot=True, save_in='geotiff'):
    '''
    Soilmap 1:20 000
    1: Kalliomaa, 2: Sora, 3: Hiekkamoreeni, 4: Hiekka, 
    5: karkea Hieta , 6: hieno Hieta, 7: Hiesu, 8: Saraturve, 9: Rahkaturve, 10: Vesi
    Soilmap 1:200 000
    1: Kalliomaa, 2: Kalliopaljastuma, 3: Karkearakeinen maalaji, 4: Sekalajitteinen maalaji ,
    5: Hienojakoinen maalaji, 6: Soistuma, 7: Ohut turvekerros, 8: Paksu turvekerros, 9: Vesi
    '''
    if soilmap == 200:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp200k_maalajit.shp'
    elif soilmap == 20:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp20k_maalajit.shp'

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    mpk = {}
    if soilmap == 20:
        mpk[195111.0] = 1 # Kalliomaa
        mpk[195313.0] = 2 # Sora
        mpk[195214.0] = 3 # Hiekkamoreeni
        mpk[195314.0] = 4 # Hiekka
        mpk[195315.0] = 5 # karkea Hieta 
        mpk[195411.0] = 6 # hieno Hieta
        mpk[195412.0] = 7 # Hiesu
        mpk[195512.0] = 8 # Saraturve
        mpk[195513.0] = 9 # Rahkaturve
        mpk[195603.0] = 10 # Vesi
    elif soilmap == 200:
        mpk = {}
        mpk[195111.0] = 1 # Kalliomaa
        mpk[195110.0] = 2 # Kalliopaljastuma
        mpk[195310.0] = 3 # Karkearakeinen maalaji
        mpk[195210.0] = 4 # Sekalajitteinen maalaji 
        mpk[195410.0] = 5 # Hienojakoinen maalaji
        mpk[19551822.0] = 6 # Soistuma
        mpk[19551891.0] = 7 # Ohut turvekerros
        mpk[19551892.0] = 8 # Paksu turvekerros
        mpk[195603.0] = 9 # Vesi

    soil = gpd.read_file(soilfile, include_fields=["PINTAMAALA", "PINTAMAA_1", "POHJAMAALA", "POHJAMAA_1", "geometry"], bbox=subset)
    soil.PINTAMAALA = soil.PINTAMAALA.astype("float64")
    soil.POHJAMAALA = soil.POHJAMAALA.astype("float64")
    
    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'{soildepth}{soilmap}') + '.tif'
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'{soildepth}{soilmap}') + '.asc'

    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
        
    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        if soildepth=='surface':
            shapes = ((geom,value) for geom, value in zip(soil.geometry, soil.PINTAMAALA))
        if soildepth=='bottom':
            shapes = ((geom,value) for geom, value in zip(soil.geometry, soil.POHJAMAALA))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        for key in mpk.keys():
            burned[burned == key] = mpk[key]
        out.write_band(1, burned)
        if plot==True:
            raster = rasterio.open(out_fn)
            show(raster)
            
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
        info - info dictionary
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

def delineate_catchment_from_dem(dem_path, catchment_name, outfolder, outlet_file, clip_catchment=False, snap=True, routing='d8', plot_catchment=True):

    print('')
    print('*** Delineating', catchment_name, 'catchment ***')
    outlets = pd.read_csv(outlet_file, sep=';', encoding = "ISO-8859-1")
    #outlet_x = float(outlets.loc[outlets['stream'] == catchment_name, 'lon'])
    #outlet_y = float(outlets.loc[outlets['stream'] == catchment_name, 'lat'])

    outlet_x = outlets.loc[outlets['stream'] == catchment_name, 'lon'].values[0]
    outlet_y = outlets.loc[outlets['stream'] == catchment_name, 'lat'].values[0]

    outpath = os.path.join(outfolder, catchment_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    warnings.simplefilter("ignore", UserWarning)

    #raster = xr.open_rasterio(dem_path)
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)
    aspect = grid.flowdir(inflated_dem, dirmap=dirmap, routing='d8')
    slope = grid.cell_slopes(inflated_dem, fdir)

    eps = np.finfo(float).eps

    twi = np.log((acc+1) / (np.tan(slope) + eps))

    # Snap pour point to high accumulation cell
    if snap == True:
        x_snap, y_snap = grid.snap_to_mask(acc > 100, (outlet_x, outlet_y))
    else:
        x_snap, y_snap = outlet_x, outlet_y

    # Delineate the catchment
    catch_full = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                       xytype='coordinate', routing=routing)

    if clip_catchment == True:
    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
        grid.clip_to(catch_full)
        clipped_catch = grid.view(catch_full)

    else:
        clipped_catch = catch

    #cmask_temp = np.ones(shape=clipped_catch.shape)
    #cmask_temp[clipped_catch == False] = 0
    #cmask_temp = ndimage.binary_fill_holes(cmask_temp).astype(int)
    #cmask = np.ones(shape=cmask_temp.shape)
    #cmask[cmask_temp == 0] = int(-9999)

    info = {'ncols':clipped_catch.shape[0],
        'nrows':clipped_catch.shape[1],
        'xllcorner':clipped_catch.bbox[0],
        'yllcorner':clipped_catch.bbox[1],
        'cellsize':clipped_catch.affine[0],
        'NODATA_value':-9999}

    subset = [clipped_catch.bbox[0], clipped_catch.bbox[1], clipped_catch.bbox[2], clipped_catch.bbox[3]]
    
    #fname=os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc')
    #write_AsciiGrid_new(fname, cmask, info)

    if plot_catchment == True:
        # Plot the catchment
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        ax.imshow(dem, extent=dem.extent, cmap='terrain', zorder=1)
        ax.imshow(np.where(catch_full, catch_full, np.nan), extent=dem.extent,
               zorder=1, cmap='Greys_r', alpha=0.5)
        #ax.imshow(cmask, alpha=0.3)
        #raster.plot(ax=ax, vmin=200, vmax=600)
        #ax.imshow(acc, extent=clipped_catch.extent, zorder=2,
        #       cmap='cubehelix',
        #       norm=colors.LogNorm(1, acc.max()),
        #       interpolation='bilinear', alpha=0.1)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title(f'Delineated {catchment_name} catchment', size=14)

        grid.to_ascii(clipped_catch, os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(inflated_dem, os.path.join(outpath, f'inflated_dem_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(slope, os.path.join(outpath, f'slope_{routing}_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(aspect, os.path.join(outpath, f'aspect_{routing}_{catchment_name}.asc'), nodata=-9999)
        grid.to_ascii(twi, os.path.join(outpath, f'twi_{routing}_{catchment_name}.asc'), nodata=-9999)
        print('***', catchment_name, 'catchment is delineated and DEM derivatives are saved ***')

        return subset

def fill_cmask_holes(fp, fmt='%i', plot=True):
    '''
    for float fmt='%.18e'
    for int fmt='%i'
    '''
    # new filename
    fn = fp[0:-4]+'_fill.asc'

    # reading the old file for information
    old_cmask = read_AsciiGrid(fp)
    
    # taking the np.array to be edited
    arr = old_cmask[0]
    arr[np.isnan(arr)] = 0
    
    # filling the holes 
    new_arr = ndimage.binary_fill_holes(arr).astype(int)
    
    # assigning zeros to -9999 and plotting
    new_arr = np.where(new_arr==0, int(-9999), int(1))
    #print(new_arr)
    plt.imshow(np.where(new_arr==-9999, np.nan, 1))    
    plt.show()
    write_AsciiGrid(fn, new_arr, old_cmask[1])
    # writing the new cmask file
    #fid = open(fn, 'w')
    #for i in range(len(old_cmask[1])):
    #    fid.write(old_cmask[1][i]) 
    #fid.close()
    #fid = open(fn, 'a')
    #np.savetxt(fid, new_arr, fmt=fmt, delimiter=' ')
    #np.savetxt(fid, new_arr.astype(int), fmt=fmt, delimiter=' ')




















