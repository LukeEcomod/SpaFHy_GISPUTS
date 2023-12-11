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

def dem_from_mml(out_fd, subset, apikey, layer='korkeusmalli_2m', form='image/tiff', scalefactor=0.125, plot=True, cmap='terrain'):

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

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
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
    out_fp = os.path.join(out_fd, layer) + '.tif'

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

    if plot==True:
        show(raster_dem)
    
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
                #print('--> converted to 0 ***')
            if len(data.flatten()[data.flatten() == 32767]) > 0:
                print('*** Data has', len(data.flatten()[data.flatten() == 32767]), 'non land values (=32767) ***')
                #print('--> converted to 0 ***')
            #data[data > 32765] = 0

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

#def data_to_spafhy(in_fd, out_fd, save_in='asc', plot=True)

def needlemass_to_lai(in_fd, in_ff, out_fd, species, save_in='asc', plot=True):

    # specific leaf area (m2/kg) for converting leaf mass to leaf area
    # SLA = {'pine': 5.54, 'spruce': 5.65, 'decid': 18.46}  # m2/kg, Kellomäki et al. 2001 Atm. Env.
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    species_translations = {'spruce': 'kuusi', 
                           'pine': 'manty',
                           'decid': 'lehtip'}
    asked_species = species_translations[species]
    p = os.path.join(in_fd, f'bm*{asked_species}*neulaset*.{in_ff}')
    in_fn = glob.glob(p)[0]
    
    bm_raster = rasterio.open(in_fn)
    bm_data = bm_raster.read(1)
    
    LAI = bm_data * 1e-3 * SLA[species] # 1e-3 converts 10kg/ha to kg/m2

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

#def coords_array(in_raster_fn):
#    with rasterio.open(in_raster_fn) as src:
#        out_arr = np.zeros(src.read(1).shape)
        


def soilmap_from_puhti(soilmap, subset, out_fd, ref_raster, soildepth='surface', plot=True, save_in='geotiff'):
    '''
    Soilmap
    '''
    if soilmap == 200:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp200k_maalajit.shp'
    elif soilmap == 20:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp20k_maalajit.shp'

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)

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
        #for key in mpk.keys():
        #    burned[burned == key] = mpk[key]
        out.write_band(1, burned)
        if plot==True:
            raster = rasterio.open(out_fn)
            show(raster)

def maastolayer_to_raster(in_fn, out_fd, layer, ref_raster, save_in='asc', plot=True):
    '''
    processing maastotietokanta (gpkg) layer to raster
    '''
    if not os.path.exists(out_fd):
        os.makedirs(out_fd)
        
    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'{layer}') + '.tif'
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'{layer}') + '.asc'
    
    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')

    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
    
    subset=list(rst.bounds[:])
    data = gpd.read_file(in_fn, layer=layer, include_fields=["kohdeluokka", "geometry"], bbox=subset)
    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(data.geometry, data.kohdeluokka))

        burned = features.rasterize(shapes=shapes, fill=-9999, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        out.write_band(1, burned)
        
        if plot==True:
            raster = rasterio.open(out_fn)
            show(raster)

def create_catchment(fpath, 
                     plotgrids=True, 
                     plotdistr=False):
    """
    reads gis-data grids from catchment and returns numpy 2d-arrays
    IN:
        fpath - folder (str)
        plotgrids - True plots
    OUT:
        GisData - dictionary with 2d numpy arrays and some vectors/scalars.

        keys [units]:
        'dem'[m],'slope'[deg],'soil'[coding 1-4], 'cf'[-],'flowacc'[m2], 'twi'[log m??],
        'vol'[m3/ha],'ba'[m2/ha], 'age'[yrs], 'hc'[m], 'bmroot'[1000kg/ha],
        'LAI_pine'[m2/m2 one-sided],'LAI_spruce','LAI_decid',
        'info','lat0'[latitude, euref_fin],'lon0'[longitude, euref_fin],
        loc[outlet coords,euref_fin],'cellsize'[cellwidth,m],
        'peatm','stream','cmask','rockm'[masks, 1=True]
    """
    
    # values to be set for 'open peatlands' and 'not forest land'
    nofor = {'vol': 0.1, 'ba': 0.01, 'height': 0.1, 'cf': 0.01, 'age': 0.0,
             'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.01, 'bmroot': 0.01, 'bmleaf': 0.01}
    opeatl = {'vol': 0.01, 'ba': 0.01, 'height': 0.1, 'cf': 0.1, 'age': 0.0,
              'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.1, 'bmroot': 0.01, 'bmleaf': 0.01}

    # SLA = {'pine': 5.54, 'spruce': 5.65, 'decid': 18.46}  # m2/kg, Kellomäki et al. 2001 Atm. Env.
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195
    
    ''' *** READING ALL MAPS *** '''
    # NLF DEM and derivatives
    dem, info, pos, cellsize, nodata = read_AsciiGrid(os.path.join(fpath, 'dem/inflated_dem_kuivajarvi.asc'))
    cmask, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/cmask_d8_kuivajarvi_fill.asc'))
    flowacc, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/acc_d8_kuivajarvi.asc'))
    flowpoint, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/fdir_d8_kuivajarvi.asc'))
    slope, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/slope_d8_kuivajarvi.asc'))
    twi, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/twi_d8_kuivajarvi.asc'))

    # NLF maastotietokanta maps
    stream, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/virtavesikapea.asc'))
    lake, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/jarvi.asc'))
    road, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/tieviiva.asc'))
    peatm, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/suo.asc'))
    peatm2, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/soistuma.asc'))
    rockm, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/kallioalue.asc'))

    #GTK soil maps
    surfsoil, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/surface200.asc'))
    botsoil, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/bottom200.asc'))

    # LUKE VMI maps
    # spruce
    bmleaf_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_neulaset_vmi1x_1721.asc')) # 10kg/ha
    bmroot_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_juuret_vmi1x_1721.asc')) # 10kg/ha
    bmstump_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_kanto_vmi1x_1721.asc')) # 10kg/ha
    bmlivebranch_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_elavatoksat_vmi1x_1721.asc')) # 10kg/ha
    bmdeadbranch_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_kuolleetoksat_vmi1x_1721.asc')) # 10kg/ha
    bmtop_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_latva_vmi1x_1721.asc')) # 10kg/ha
    bmcore_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_kuusi_runkokuori_vmi1x_1721.asc')) # 10kg/ha
    s_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/kuusi_vmi1x_1721.asc'))     #spruce volume [m3 ha-1]
    # pine
    bmleaf_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_neulaset_vmi1x_1721.asc')) # 10kg/ha
    bmroot_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_juuret_vmi1x_1721.asc')) # 10kg/ha
    bmstump_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_kanto_vmi1x_1721.asc')) # 10kg/ha
    bmlivebranch_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_elavatoksat_vmi1x_1721.asc')) # 10kg/ha
    bmdeadbranch_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_kuolleetoksat_vmi1x_1721.asc')) # 10kg/ha
    bmtop_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_latva_vmi1x_1721.asc')) # 10kg/ha
    bmcore_pine, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_manty_runkokuori_vmi1x_1721.asc')) # 10kg/ha
    p_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/manty_vmi1x_1721.asc'))     #pine volume [m3 ha-1]
    # decid
    bmleaf_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_neulaset_vmi1x_1721.asc')) # 10kg/ha
    bmroot_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_juuret_vmi1x_1721.asc')) # 10kg/ha
    bmstump_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_kanto_vmi1x_1721.asc')) # 10kg/ha
    bmlivebranch_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_elavatoksat_vmi1x_1721.asc')) # 10kg/ha
    bmdeadbranch_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_kuolleetoksat_vmi1x_1721.asc')) # 10kg/ha
    bmtop_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_latva_vmi1x_1721.asc')) # 10kg/ha
    bmcore_spruce, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_runkokuori_vmi1x_1721.asc'))
    b_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/koivu_vmi1x_1721.asc'))     #birch volume [m3 ha-1]
    cf_d, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/lehtip_latvuspeitto_vmi1x_1721.asc'))     # canopy closure [%]
    # integrated
    fraclass, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/fra_luokka_vmi1x_1721.asc')) # FRA [1-4]
    cf, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/latvuspeitto_vmi1x_1721.asc'))  # canopy closure [%]
    height, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/keskipituus_vmi1x_1721.asc'))   # tree height [dm]
    diameter, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/keskilapimitta_vmi1x_1721.asc'))  # tree diameter [cm]
    ba, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/ppa_vmi1x_1721.asc') )  # basal area [m2 ha-1]
    age, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/ika_vmi1x_1721.asc'))   # stand age [yrs]
    maintype, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/paatyyppi_vmi1x_1721.asc')) # [1-4]
    sitetype, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/kasvupaikka_vmi1x_1721.asc')) # [1-10]
    forestsoilclass, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/maaluokka_vmi1x_1721.asc')) # forestry soil class [1-3]
    vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/tilavuus_vmi1x_1721.asc'))  # total volume [m3 ha-1]

    # catchment mask cmask ==1, np.NaN outside
    cmask[np.isfinite(cmask)] = 1.0

    # dem, set values outside boundaries to NaN
    # latitude, longitude arrays
    nrows, ncols = np.shape(dem)
    lon0 = np.arange(pos[0], pos[0] + cellsize*ncols, cellsize)
    lat0 = np.arange(pos[1], pos[1] + cellsize*nrows, cellsize)
    lat0 = np.flipud(lat0)  # why this is needed to get coordinates correct when plotting?

    """
    Create soiltype grid and masks for waterbodies, streams, peatlands and rocks
    """
    # Maastotietokanta water bodies: 1=waterbody
    stream[np.isfinite(stream)] = -1.0
    # Maastotietokanta water bodies: 1=waterbody
    lake[np.isfinite(lake)] = -1.0
    # maastotietokanta peatlandmask
    peatm[np.isfinite(peatm)] = 1.0
    # maastotietokanta kalliomaski
    rockm[np.isfinite(rockm)] = 1.0
    # maastotietokanta peatmask2
    peatm2[np.isfinite(peatm2)] = 1.0
    # maastotietokanta roadmask
    road[np.isfinite(road)] = -1.0
    
    """
    gtk soilmap: read and re-classify into 4 texture classes
    #GTK-pintamaalaji grouped to 4 classes (Samuli Launiainen, Jan 7, 2017)
    #Codes based on maalaji 1:20 000 AND ADD HERE ALSO 1:200 000

        ---- 1:20 000 map
        195111.0 = Kalliomaa
        195313.0 = Sora
        195214.0 = Hiekkamoreeni
        195314.0 = Hiekka
        195315.0 = karkea Hieta 
        195411.0 = hieno Hieta
        195412.0 = Hiesu
        195512.0 = Saraturve
        195513.0 = Rahkaturve
        195603.0 = Vesi
        ---- 1:200 000 map
        195111.0 = Kalliomaa
        195110.0 = Kalliopaljastuma
        195310.0 = Karkearakeinen maalaji
        195210.0 = Sekalajitteinen maalaji 
        195410.0 = Hienojakoinen maalaji
        19551822.0 = Soistuma
        19551891.0 = Ohut turvekerros
        19551892.0 = Paksu turvekerros
        195603.0 = Vesi
    """
    
    CoarseTextured = [195213, 195314, 19531421, 195313, 195310]
    MediumTextured = [195315, 19531521, 195215, 195214, 195601, 195411, 195112,
                      195311, 195113, 195111, 195210, 195110, 195312]
    FineTextured = [19531521, 195412, 19541221, 195511, 195413, 195410,
                    19541321, 195618]
    Peats = [195512, 195513, 195514, 19551822, 19551891, 19551892]
    Water = [195603]

    # manipulating surface soil
    rs, cs = np.shape(surfsoil)
    topsoil = np.ravel(surfsoil)
    topsoil[np.in1d(topsoil, CoarseTextured)] = 1.0 
    topsoil[np.in1d(topsoil, MediumTextured)] = 2.0
    topsoil[np.in1d(topsoil, FineTextured)] = 3.0
    topsoil[np.in1d(topsoil, Peats)] = 4.0
    topsoil[np.in1d(topsoil, Water)] = -1.0
    topsoil[topsoil == -1.0] = 2.0
    topsoil[np.where(topsoil == 4.0) and np.where(peatm.flatten() != 1.0) and np.where(topsoil != 1.0)] = 2.0
    # reshape back to original grid
    topsoil = topsoil.reshape(rs, cs)
    del rs, cs
    topsoil[np.isfinite(peatm)] = 4.0

    # manipulating bottom soil
    rb, cb = np.shape(botsoil)
    lowsoil = np.ravel(botsoil)
    lowsoil[np.in1d(lowsoil, CoarseTextured)] = 1.0 
    lowsoil[np.in1d(lowsoil, MediumTextured)] = 2.0
    lowsoil[np.in1d(lowsoil, FineTextured)] = 3.0
    lowsoil[np.in1d(lowsoil, Peats)] = 4.0
    lowsoil[np.in1d(lowsoil, Water)] = -1.0
    lowsoil[lowsoil == -1.0] = 2.0
    lowsoil[np.where(lowsoil == 4.0) and np.where(peatm.flatten() != 1.0) and np.where(lowsoil != 1.0)] = 2.0
    # reshape back to original grid
    lowsoil = lowsoil.reshape(rb, cb)
    del rb, cb
    lowsoil[np.isfinite(peatm)] = 4.0

    # update waterbody mask
    ix = np.where(topsoil == -1.0)
    stream[ix] = -1.0
    stream[~np.isfinite(stream)] = 0.0

    road[~np.isfinite(road)] = 0.0
    lake[~np.isfinite(lake)] = 0.0

    # update catchment mask so that water bodies are left out (SL 20.2.18)
    #cmask[soil == -1.0] = np.NaN

    """ stand data (MNFI)"""
    
    # indexes for cells not recognized in mNFI
    ix_n = np.where((vol >= 32727) | (vol == -9999) )  # no satellite cover or not forest land: assign arbitrary values
    ix_p = np.where((vol >= 32727) & (peatm == 1))  # open peatlands: assign arbitrary values
    ix_w = np.where(((vol >= 32727) & (stream == -1)) | ((vol >= 32727) & (lake == -1)))  # waterbodies: leave out
    #cmask[ix_w] = np.NaN  # NOTE: leaves waterbodies out of catchment mask

    lowsoil[ix_w] = np.NaN
    topsoil[ix_w] = np.NaN

    vol[ix_n] = nofor['vol']
    vol[ix_p] = opeatl['vol']
    vol[ix_w] = np.NaN
    
    p_vol[ix_n] = nofor['vol']
    p_vol[ix_p] = opeatl['vol']
    p_vol[ix_w] = np.NaN

    s_vol[ix_n] = nofor['vol']
    s_vol[ix_p] = opeatl['vol']
    s_vol[ix_w] = np.NaN

    b_vol[ix_n] = nofor['vol']
    b_vol[ix_p] = opeatl['vol']
    b_vol[ix_w] = np.NaN
    
    ba[ix_n] = nofor['ba']
    ba[ix_p] = opeatl['ba']
    ba[ix_w] = np.NaN

    height = 0.1*height  # m
    height[ix_n] = nofor['height']
    height[ix_p] = opeatl['height']
    height[ix_w] = np.NaN

    cf = 1e-2*cf
    cf[ix_n] = nofor['cf']
    cf[ix_p] = opeatl['cf']
    cf[ix_w] = np.NaN

    age[ix_n] = nofor['age']
    age[ix_p] = opeatl['age']
    age[ix_w] = np.NaN

    # leaf biomasses and one-sided LAI
    bmleaf_pine[ix_n]=nofor['bmleaf']; 
    bmleaf_pine[ix_p]=opeatl['bmleaf']; 
    bmleaf_pine[ix_w]=np.NaN; 

    bmleaf_spruce[ix_n]=nofor['bmleaf'];
    bmleaf_spruce[ix_p]=opeatl['bmleaf']; 
    bmleaf_spruce[ix_w]=np.NaN; 
    
    bmleaf_decid[ix_n]=nofor['bmleaf'];
    bmleaf_decid[ix_p]=opeatl['bmleaf']; 
    bmleaf_decid[ix_w]=np.NaN; 

    LAI_pine = 1e-3*bmleaf_pine*SLA['pine']  # 1e-3 converts 10kg/ha to kg/m2
    LAI_pine[ix_n] = nofor['LAIpine']
    LAI_pine[ix_p] = opeatl['LAIpine']
    LAI_pine[ix_w] = np.NaN

    LAI_spruce = 1e-3*bmleaf_spruce*SLA['spruce']
    LAI_spruce[ix_n] = nofor['LAIspruce']
    LAI_spruce[ix_p] = opeatl['LAIspruce']
    LAI_spruce[ix_w] = np.NaN

    LAI_decid = 1e-3*bmleaf_decid*SLA['decid']
    LAI_decid[ix_n] = nofor['LAIdecid']
    LAI_decid[ix_p] = opeatl['LAIdecid']
    LAI_decid[ix_w] = np.NaN

    bmroot = 1e-2*(bmroot_pine + bmroot_spruce + bmroot_decid)  # 1000 kg/ha
    bmroot[ix_n] = nofor['bmroot']
    bmroot[ix_p] = opeatl['bmroot']
    bmroot[ix_w] = np.NaN
    
    bmroot_pine = 1e-2*(bmroot_pine)  # 1000 kg/ha
    bmroot_pine[ix_n] = nofor['bmroot']
    bmroot_pine[ix_p] = opeatl['bmroot']
    bmroot_pine[ix_w] = np.NaN

    bmroot_spruce = 1e-2*(bmroot_spruce)  # 1000 kg/ha
    bmroot_spruce[ix_n] = nofor['bmroot']
    bmroot_spruce[ix_p] = opeatl['bmroot']
    bmroot_spruce[ix_w] = np.NaN

    bmroot_decid = 1e-2*(bmroot_decid)  # 1000 kg/ha
    bmroot_decid[ix_n] = nofor['bmroot']
    bmroot_decid[ix_p] = opeatl['bmroot']
    bmroot_decid[ix_w] = np.NaN
    
    # interpolating maintype to not have nan on roads
    #x = np.arange(0, maintype.shape[1])
    #y = np.arange(0, maintype.shape[0])
    #mask invalid values
    #array = np.ma.masked_invalid(maintype)
    #xx, yy = np.meshgrid(x, y)
    #get only the valid values
    #x1 = xx[~array.mask]
    #y1 = yy[~array.mask]
    #newarr = array[~array.mask]

    #maintype = griddata((x1, y1), newarr.ravel(),
    #                      (xx, yy),
    #                         method='nearest')

    # catchment outlet location and catchment mean elevation
    (iy, ix) = np.where(flowacc == np.nanmax(flowacc))
    loc = {'lat': lat0[iy], 'lon': lon0[ix], 'elev': np.nanmean(dem)}
    # dict of all rasters
    GisData = {'cmask': cmask, 'dem': dem, 'flowacc': flowacc, 'flowpoint': flowpoint, 'slope': slope, 'twi': twi, 
               'topsoil': topsoil, 'lowsoil': lowsoil, 
               'peatm': peatm, 'peatm2': peatm2, 'stream': stream, 'lake': lake, 'road': road, 'rockm': rockm,
               'LAI_pine': LAI_pine, 'LAI_spruce': LAI_spruce, 'LAI_conif': LAI_pine + LAI_spruce, 'LAI_decid': LAI_decid,
               'bmroot': bmroot, 'ba': ba, 'hc': height, 'vol': vol, 'p_vol': p_vol, 's_vol': s_vol, 'b_vol': b_vol, 
               'cf': cf, 'age': age, 'maintype': maintype, 'sitetype': sitetype,
               'cellsize': cellsize, 'info': info, 'lat0': lat0, 'lon0': lon0, 'loc': loc}

    if plotgrids is True:
        # %matplotlib qt
        # xx, yy = np.meshgrid(lon0, lat0)
        plt.close('all')

        plt.figure(figsize=(10,6))
        plt.subplot(111)
        plt.imshow(cmask); plt.colorbar(); plt.title('catchment mask')
        
        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(dem); plt.colorbar(); plt.title('DEM')
        #plt.plot(ix, iy,'rs')
        plt.subplot(222)
        plt.imshow(twi); plt.colorbar(); plt.title('TWI')
        plt.subplot(223)
        plt.imshow(slope); plt.colorbar(); plt.title('slope')
        plt.subplot(224)
        plt.imshow(flowacc); plt.colorbar(); plt.title('flowacc')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(topsoil); plt.colorbar(); plt.title('surface soiltype')
        plt.subplot(222)
        plt.imshow(lowsoil); plt.colorbar(); plt.title('bottom soiltype')
        #mask = cmask.copy()*0.0
        #mask[np.isfinite(peatm)] = 1
        #mask[np.isfinite(rockm)] = 2
        #mask[np.isfinite(stream)] = 3

        #plt.subplot(222)
        #plt.imshow(mask); plt.colorbar(); plt.title('masks')
        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(LAI_pine+LAI_spruce); plt.colorbar(); plt.title('LAI conif (m2/m2)')
        plt.subplot(222)
        plt.imshow(LAI_decid); plt.colorbar(); plt.title('LAI decid (m2/m2)')
        plt.subplot(223)
        plt.imshow(age); plt.colorbar(); plt.title('tree age (yr)')
        plt.subplot(224)
        plt.imshow(height); plt.colorbar(); plt.title('canopy height (m)')
        
        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(cf); plt.colorbar(); plt.title('canopy fraction (-)')
        plt.subplot(222)
        plt.imshow(p_vol+s_vol); plt.colorbar(); plt.title('conif. vol (m3/ha)')
        plt.subplot(223)
        plt.imshow(p_vol); plt.colorbar(); plt.title('decid. vol (m3/ha)')
        plt.subplot(224)
        plt.imshow(ba); plt.colorbar(); plt.title('basal area (m2/ha)')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(1e-3*bmleaf_pine); plt.colorbar(); plt.title('pine needles (kg/m2)')
        plt.subplot(222)
        plt.imshow(1e-3*bmleaf_spruce); plt.colorbar(); plt.title('spruce needles (kg/m2)')
        plt.subplot(223)
        plt.imshow(1e-3*bmleaf_decid); plt.colorbar(); plt.title('decid. leaves (kg/m2)')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(1e-3*bmroot_pine); plt.colorbar(); plt.title('pine roots (kg/m2)')
        plt.subplot(222)
        plt.imshow(1e-3*bmroot_spruce); plt.colorbar(); plt.title('spruce roots (kg/m2)')
        plt.subplot(223)
        plt.imshow(1e-3*bmroot_decid); plt.colorbar(); plt.title('decid. roots (kg/m2)')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(stream); plt.colorbar(); plt.title('stream')
        plt.subplot(222)
        plt.imshow(road); plt.colorbar(); plt.title('roads')
        plt.subplot(223)
        plt.imshow(lake); plt.colorbar(); plt.title('lake')
        
    if plotdistr is True:
        twi0 = twi[np.isfinite(twi)]
        vol = vol[np.isfinite(vol)]
        lai = LAI_pine + LAI_spruce + LAI_decid
        lai = lai[np.isfinite(lai)]
        soil0 = soil[np.isfinite(soil)]

        plt.figure(100)
        plt.subplot(221)
        plt.hist(twi0, bins=100, color='b', alpha=0.5)
        plt.ylabel('f');plt.ylabel('twi')

        s = np.unique(soil0)
        colcode = 'rgcym'
        for k in range(0,len(s)):
            # print k
            a = twi[np.where(soil==s[k])]
            a = a[np.isfinite(a)]
            plt.hist(a, bins=50, alpha=0.5, color=colcode[k], label='soil ' +str(s[k]))
        plt.legend()
        plt.show()

        plt.subplot(222)
        plt.hist(vol, bins=100, color='k'); plt.ylabel('f'); plt.ylabel('vol')
        plt.subplot(223)
        plt.hist(lai, bins=100, color='g'); plt.ylabel('f'); plt.ylabel('lai')
        plt.subplot(224)
        plt.hist(soil0, bins=5, color='r'); plt.ylabel('f');plt.ylabel('soiltype')

    return GisData


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


def delineate_catchment_from_dem(dem_path, catchment_name, out_fd, outlet_file, 
                                 clip_catchment=False, snap=True, routing='d8', 
                                 plot_catchment=True, fill_holes=True):

    print('')
    print('*** Delineating', catchment_name, 'catchment ***')
    outlets = pd.read_csv(outlet_file, sep=';', encoding = "ISO-8859-1")
    #outlet_x = float(outlets.loc[outlets['stream'] == catchment_name, 'lon'])
    #outlet_y = float(outlets.loc[outlets['stream'] == catchment_name, 'lat'])

    outlet_x = outlets.loc[outlets['stream'] == catchment_name, 'lon'].values[0]
    outlet_y = outlets.loc[outlets['stream'] == catchment_name, 'lat'].values[0]

    outpath = os.path.join(out_fd)
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
        clipped_catch = catch_full

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

    grid.to_ascii(clipped_catch, os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(inflated_dem, os.path.join(outpath, f'inflated_dem_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(slope, os.path.join(outpath, f'slope_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(aspect, os.path.join(outpath, f'aspect_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(twi, os.path.join(outpath, f'twi_{routing}_{catchment_name}.asc'), nodata=-9999)
    print('***', catchment_name, 'catchment is delineated and DEM derivatives are saved ***')

    if fill_holes==True:
        in_fn = os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc')
        filled_cmask, cmask_fp = fill_cmask_holes(in_fn, plot=False)

    cmask_fill_grid = Grid.from_ascii(cmask_fp)
    cmask_fill_raster = grid.read_ascii(cmask_fp)
    
    if plot_catchment == True:
        # Plot the catchment
        
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        ax.imshow(inflated_dem, extent=inflated_dem.extent, cmap='terrain', zorder=1)
        #ax.imshow(np.where(catch_full, catch_full, np.nan), extent=dem.extent,
        #       zorder=1, cmap='Greys_r', alpha=0.5)
        ax.imshow(np.where(cmask_fill_raster, cmask_fill_raster, np.nan), extent=inflated_dem.extent,
               zorder=1, cmap='Greys_r', alpha=0.3)       
        
        #ax.imshow(cmask, alpha=0.3)
        #raster.plot(ax=ax, vmin=200, vmax=600)
        #ax.imshow(acc, extent=clipped_catch.extent, zorder=2,
        #       cmap='cubehelix',
        #       norm=colors.LogNorm(1, acc.max()),
        #       interpolation='bilinear', alpha=0.1)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title(f'Delineated {catchment_name} catchment', size=14)

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
    if plot==True:
        plt.imshow(np.where(new_arr==-9999, np.nan, 1))    
        plt.show()
    write_AsciiGrid(fn, new_arr, old_cmask[1])

    return new_arr, fn
    # writing the new cmask file
    #fid = open(fn, 'w')
    #for i in range(len(old_cmask[1])):
    #    fid.write(old_cmask[1][i]) 
    #fid.close()
    #fid = open(fn, 'a')
    #np.savetxt(fid, new_arr, fmt=fmt, delimiter=' ')
    #np.savetxt(fid, new_arr.astype(int), fmt=fmt, delimiter=' ')




















