import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
from tools import read_AsciiGrid
import os
import netCDF4

def create_catchment(fpath, 
                     set_non_forest_as='null',
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
    nofor = {'vol': 0.0, 'ba': 0.01, 'height': 0.1, 'cf': 0.01, 'age': 0.0, 'diameter': 0.0,
                 'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.01, 'bmroot': 0.01, 'bmleaf': 0.01,
                'bmstump': 0.01, 'bmcore': 0.01, 'bmall': 0.01, 'site': 0}
    opeatl = {'vol': 0.0, 'ba': 0.01, 'height': 0.1, 'cf': 0.01, 'age': 0.0, 'diameter': 0.0,
                  'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.1, 'bmroot': 0.01, 'bmleaf': 0.01,
                'bmstump': 0.01, 'bmcore': 0.01, 'bmall': 0.01, 'site': 0}
    water = {'vol': 0.0, 'ba': 0.0, 'height': 0.0, 'cf': 0.0, 'age': 0.0, 'diameter': 0.0,
                  'LAIpine': 0.0, 'LAIspruce': 0.0, 'LAIdecid': 0.0, 'bmroot': 0.0, 'bmleaf': 0.0,
                'bmstump': 0.0, 'bmcore': 0.0, 'bmall': 0.0, 'site': 0}        
    if set_non_forest_as=='nan':
        for key in nofor.keys():
            nofor[key] = np.NaN
            opeatl[key] = np.NaN
            water[key] = np.NaN

    elif set_non_forest_as=='null':
        for key in nofor.keys():
            nofor[key] = 0
            opeatl[key] = 0
            water[key] = np.NaN
        
    # SLA = {'pine': 5.54, 'spruce': 5.65, 'decid': 18.46}  # m2/kg, Kellomäki et al. 2001 Atm. Env.
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195
    
    ''' *** READING ALL MAPS *** '''
    # NLF DEM and derivatives
    dem, info, pos, cellsize, nodata = read_AsciiGrid(os.path.join(fpath, 'dem/korkeusmalli_16m.asc'))
    cmask, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/cmask_d8_kuivajarvi_fill.asc'))
    flowacc, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/acc_d8_kuivajarvi.asc'))
    flowpoint, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/fdir_d8_kuivajarvi.asc'))
    slope, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/slope_d8_kuivajarvi.asc'))
    twi, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'dem/twi_d8_kuivajarvi.asc'))

    # NLF maastotietokanta maps
    stream, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/virtavesikapea.asc'))
    lake, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/jarvi.asc'))
    road, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/tieviiva.asc'))
    peatland, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/suo.asc'))
    paludified, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/soistuma.asc'))
    rockm, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/kallioalue.asc'))

    #GTK soil maps
    topsoil200, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/surface200.asc'))
    lowsoil200, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/bottom200.asc'))
    topsoil20, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/surface20.asc'))
    lowsoil20, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'soil/bottom20.asc'))
    
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
    nonland, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/nonland.asc'))     #vmi nonland mask
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
    bmstump_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_kanto_vmi1x_1721.asc')) # 10kg/ha
    bmlivebranch_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_elavatoksat_vmi1x_1721.asc')) # 10kg/ha
    bmdeadbranch_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_kuolleetoksat_vmi1x_1721.asc')) # 10kg/ha
    bmtop_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_latva_vmi1x_1721.asc')) # 10kg/ha
    bmcore_decid, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/bm_lehtip_runkokuori_vmi1x_1721.asc'))
    b_vol, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/koivu_vmi1x_1721.asc'))     #birch volume [m3 ha-1]
    cf_d, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/lehtip_latvuspeitto_vmi1x_1721.asc'))     # canopy closure decid [%]
    # integrated
    fraclass, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/fra_luokka_vmi1x_1721.asc')) # FRA [1-4]
    cf, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/latvuspeitto_vmi1x_1721.asc'))  # canopy closure [%]
    height, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/keskipituus_vmi1x_1721.asc'))   # tree height [dm]
    cd, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'vmi/keskilapimitta_vmi1x_1721.asc'))  # tree diameter [cm]
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
    stream[np.isfinite(stream)] = 1.0
    # Maastotietokanta water bodies: 1=waterbody
    lake[np.isfinite(lake)] = 1.0
    # maastotietokanta peatlandmask
    peatland[np.isfinite(peatland)] = 1.0
    # maastotietokanta kalliomaski
    rockm[np.isfinite(rockm)] = 1.0
    # maastotietokanta paludified
    paludified[np.isfinite(paludified)] = 1.0
    # maastotietokanta roadmask
    road[np.isfinite(road)] = 1.0
    # nonland mask not to have roads and streams
    nonland[np.isfinite(nonland)] = 0.0
    nonland[~np.isfinite(nonland)] = 1.0
    #nonland[road == 1.0] = 0.0  
    #nonland[stream == 1.0] = 0.0  

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
    rs, cs = np.shape(topsoil200)
    topsoil200 = np.ravel(topsoil200)
    topsoil200[np.in1d(topsoil200, CoarseTextured)] = 1.0 
    topsoil200[np.in1d(topsoil200, MediumTextured)] = 2.0
    topsoil200[np.in1d(topsoil200, FineTextured)] = 3.0
    topsoil200[np.in1d(topsoil200, Peats)] = 4.0
    topsoil200[np.in1d(topsoil200, Water)] = 0.0
    topsoil200[np.where(topsoil200 == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(topsoil200 != 1.0)] = 2.0
    # reshape back to original grid
    topsoil200 = topsoil200.reshape(rs, cs)
    del rs, cs
    topsoil200[np.isfinite(peatland)] = 4.0
    
    rs, cs = np.shape(topsoil20)
    topsoil20 = np.ravel(topsoil20)
    topsoil20[np.in1d(topsoil20, CoarseTextured)] = 1.0 
    topsoil20[np.in1d(topsoil20, MediumTextured)] = 2.0
    topsoil20[np.in1d(topsoil20, FineTextured)] = 3.0
    topsoil20[np.in1d(topsoil20, Peats)] = 4.0
    topsoil20[np.in1d(topsoil20, Water)] = 0.0
    topsoil20[np.where(topsoil20 == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(topsoil20 != 1.0)] = 2.0
    # reshape back to original grid
    topsoil20 = topsoil20.reshape(rs, cs)
    del rs, cs
    topsoil20[np.isfinite(peatland)] = 4.0
    
    # manipulating bottom soil
    rb, cb = np.shape(lowsoil200)
    lowsoil200 = np.ravel(lowsoil200)
    lowsoil200[np.in1d(lowsoil200, CoarseTextured)] = 1.0 
    lowsoil200[np.in1d(lowsoil200, MediumTextured)] = 2.0
    lowsoil200[np.in1d(lowsoil200, FineTextured)] = 3.0
    lowsoil200[np.in1d(lowsoil200, Peats)] = 4.0
    lowsoil200[np.in1d(lowsoil200, Water)] = np.NaN
    lowsoil200[np.where(lowsoil200 == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(lowsoil200 != 1.0)] = 2.0
    # reshape back to original grid
    lowsoil200 = lowsoil200.reshape(rb, cb)
    del rb, cb
    lowsoil200[np.isfinite(peatland)] = 4.0

    rb, cb = np.shape(lowsoil20)
    lowsoil20 = np.ravel(lowsoil20)
    lowsoil20[np.in1d(lowsoil20, CoarseTextured)] = 1.0 
    lowsoil20[np.in1d(lowsoil20, MediumTextured)] = 2.0
    lowsoil20[np.in1d(lowsoil20, FineTextured)] = 3.0
    lowsoil20[np.in1d(lowsoil20, Peats)] = 4.0
    lowsoil20[np.in1d(lowsoil20, Water)] = np.NaN
    lowsoil20[np.where(lowsoil20 == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(lowsoil20 != 1.0)] = 2.0
    # reshape back to original grid
    lowsoil20 = lowsoil20.reshape(rb, cb)
    del rb, cb
    lowsoil20[np.isfinite(peatland)] = 4.0

    topsoil = topsoil20.copy()
    lowsoil = lowsoil20.copy()

    topsoil[(~np.isfinite(topsoil) & (np.isfinite(topsoil200)))] = topsoil200[(~np.isfinite(topsoil) & (np.isfinite(topsoil200)))]
    lowsoil[(~np.isfinite(lowsoil) & (np.isfinite(lowsoil200)))] = lowsoil200[(~np.isfinite(lowsoil) & (np.isfinite(lowsoil200)))]

    # update waterbody mask
    #ix = np.where(topsoil == -1.0)
    #stream[ix] = 1.0
    stream[~np.isfinite(stream)] = 0.0

    road[~np.isfinite(road)] = 0.0
    lake[~np.isfinite(lake)] = 0.0
    rockm[~np.isfinite(rockm)] = 0.0
    peatland[~np.isfinite(peatland)] = 0.0
    paludified[~np.isfinite(paludified)] = 0.0

    # update catchment mask so that water bodies are left out (SL 20.2.18)
    #cmask[soil == -1.0] = np.NaN

    """ stand data (MNFI)"""
    
    # indexes for cells not recognized in mNFI
    ix_n = np.where((vol >= 32727) | (vol == -9999))  # no satellite cover or not forest land: assign arbitrary values
    ix_p = np.where((vol >= 32727) & (peatland == 1))  # open peatlands: assign arbitrary values
    ix_w = np.where((lake == 1))  # waterbodies: leave out
    ix_t = np.where((sitetype >= 32727)) # no sitetypes set
    #cmask[ix_w] = np.NaN  # NOTE: leaves waterbodies out of catchment mask

    # units
    cd = 1e-2*cd # cm to m
    vol = 1e-4*vol # m3/ha to m3/m2
    p_vol = 1e-4*p_vol # m3/ha to m3/m2    
    s_vol = 1e-4*s_vol # m3/ha to m3/m2    
    b_vol = 1e-4*b_vol # m3/ha to m3/m2        
    ba = 1e-4*ba # m2/ha to m2/m2
    height = 0.1*height  # dm -> m
    cf = 1e-2*cf # % -> [-]
    cf_d = 1e-2*cf_d # % -> [-]
    bmleaf_pine = 1e-3*(bmleaf_pine) # 1e-3 converts 10kg/ha to kg/m2
    bmleaf_spruce = 1e-3*(bmleaf_spruce) # 1e-3 converts 10kg/ha to kg/m2
    bmleaf_decid = 1e-3*(bmleaf_decid) # 1e-3 converts 10kg/ha to kg/m2    
    bmroot_pine = 1e-3*(bmroot_pine)  # kg/m2
    bmroot_spruce = 1e-3*(bmroot_spruce)  # kg/m2
    bmroot_decid = 1e-3*(bmroot_decid)  # kg/m2
    bmstump_spruce = 1e-3*(bmstump_spruce)  # kg/m2
    bmstump_decid = 1e-3*(bmstump_decid)  # kg/m2
    bmcore_pine = 1e-3*(bmcore_pine)  # kg/m2
    bmcore_spruce = 1e-3*(bmcore_spruce)  # kg/m2
    bmcore_decid = 1e-3*(bmcore_decid)  # kg/m2
    bmtop_pine = 1e-3*(bmtop_pine)  # kg/m2
    bmtop_spruce = 1e-3*(bmtop_spruce)  # kg/m2
    bmlivebranch_pine = 1e-3*(bmlivebranch_pine)  # kg/m2
    bmlivebranch_spruce = 1e-3*(bmlivebranch_spruce)  # kg/m2
    bmtop_decid = 1e-3*(bmtop_decid)  # kg/m2
    bmlivebranch_decid = 1e-3*(bmlivebranch_decid)  # kg/m2
    bmdeadbranch_pine = 1e-3*(bmdeadbranch_pine)  # kg/m2
    bmdeadbranch_spruce = 1e-3*(bmdeadbranch_spruce)  # kg/m2
    bmdeadbranch_decid = 1e-3*(bmdeadbranch_decid)  # kg/m2

    # computing LAI
    LAI_pine = bmleaf_pine*SLA['pine'] 
    LAI_spruce = bmleaf_spruce*SLA['spruce']
    LAI_decid = bmleaf_decid*SLA['decid']
    
    # to calculate how many trees per area
    stand_density = tree_density(diameter=cd, ba=ba)

    # making 'non-land_mask'
    #nonland = np.zeros(shape=cmask.shape)
    #nonland[(sitetype == 0)] = 1
    #nonland[ix_w] = 0
    
    # manipulate non forest, peatlands and water
    # first water (lakes)
    lowsoil[ix_w] = water['site']
    topsoil[ix_w] = water['site']
    maintype[ix_w] = water['site']
    sitetype[ix_w] = water['site']
    fraclass[ix_w] = water['site']
    cd[ix_w] = water['diameter']
    vol[ix_w] = water['vol'] 
    p_vol[ix_w] = water['vol'] 
    s_vol[ix_w] = water['vol'] 
    b_vol[ix_w] = water['vol']
    ba[ix_w] = water['ba']
    height[ix_w] = water['height']
    cf[ix_w] = water['cf']
    cf_d[ix_w] = water['cf']
    age[ix_w] = water['age']
    bmleaf_pine[ix_w]=water['bmleaf']
    bmleaf_spruce[ix_w]=water['bmleaf']
    bmleaf_decid[ix_w]=water['bmleaf']
    bmroot_pine[ix_w] = water['bmroot']
    bmroot_spruce[ix_w] = water['bmroot']
    bmroot_decid[ix_w] = water['bmroot']
    bmstump_pine[ix_w] = water['bmall']
    bmstump_spruce[ix_w] = water['bmall']
    bmstump_decid[ix_w] = water['bmall']
    bmcore_pine[ix_w] = water['bmall']
    bmcore_spruce[ix_w] = water['bmall']
    bmcore_decid[ix_w] = water['bmall']
    bmtop_pine[ix_w] = water['bmall']
    bmtop_spruce[ix_w] = water['bmall']
    bmtop_decid[ix_w] = water['bmall']
    bmlivebranch_pine[ix_w] = water['bmall']
    bmlivebranch_spruce[ix_w] = water['bmall']
    bmlivebranch_decid[ix_w] = water['bmall']
    bmdeadbranch_pine[ix_w] = water['bmall']
    bmdeadbranch_spruce[ix_w] = water['bmall']
    bmdeadbranch_decid[ix_w] = water['bmall']    
    LAI_pine[ix_w] = water['LAIpine']
    LAI_spruce[ix_w] = water['LAIspruce']
    LAI_decid[ix_w] = water['LAIdecid']
    stand_density[ix_w] = water['bmall']  

    # second non land by VMI
    cd[ix_n] = nofor['diameter']
    vol[ix_n] = nofor['vol']
    p_vol[ix_n] = nofor['vol']
    s_vol[ix_n] = nofor['vol']
    b_vol[ix_n] = nofor['vol'] 
    ba[ix_n] = nofor['ba']
    height[ix_n] = nofor['height']
    cf[ix_n] = nofor['cf']
    cf_d[ix_n] = nofor['cf']
    age[ix_n] = nofor['age'] # years
    bmleaf_pine[ix_n]=nofor['bmleaf']
    bmleaf_spruce[ix_n]=nofor['bmleaf']
    bmleaf_decid[ix_n]=nofor['bmleaf']
    bmroot_pine[ix_n] = nofor['bmroot']
    bmroot_spruce[ix_n] = nofor['bmroot']
    bmroot_decid[ix_n] = nofor['bmroot']
    bmstump_pine[ix_n] = nofor['bmall']
    bmstump_spruce[ix_n] = nofor['bmall']
    bmstump_decid[ix_n] = nofor['bmall']
    bmcore_pine[ix_n] = nofor['bmall']
    bmcore_spruce[ix_n] = nofor['bmall']
    bmcore_decid[ix_n] = nofor['bmall']
    bmtop_pine[ix_n] = nofor['bmall']
    bmtop_spruce[ix_n] = nofor['bmall']
    bmtop_decid[ix_n] = nofor['bmall']
    bmlivebranch_pine[ix_n] = nofor['bmall']
    bmlivebranch_spruce[ix_n] = nofor['bmall']
    bmlivebranch_decid[ix_n] = nofor['bmall']
    bmdeadbranch_pine[ix_n] = nofor['bmall']
    bmdeadbranch_spruce[ix_n] = nofor['bmall']
    bmdeadbranch_decid[ix_n] = nofor['bmall']
    LAI_pine[ix_n] = nofor['LAIpine']
    LAI_spruce[ix_n] = nofor['LAIspruce']
    LAI_decid[ix_n] = nofor['LAIdecid']
    stand_density[ix_n] = nofor['bmall']
    maintype[ix_n] = nofor['site']
    sitetype[ix_n] = nofor['site']
    fraclass[ix_n] = nofor['site']
    
    '''
    maintype[ix_t] = opeatl['site']
    sitetype[ix_t] = opeatl['site']
    fraclass[ix_t] = opeatl['site']
    cd[ix_p] = opeatl['diameter']
    vol[ix_p] = opeatl['vol']
    p_vol[ix_p] = opeatl['vol'] 
    s_vol[ix_p] = opeatl['vol'] 
    b_vol[ix_p] = opeatl['vol'] 
    ba[ix_p] = nofor['ba']
    height[ix_p] = opeatl['height']
    cf[ix_p] = opeatl['cf']
    cf_d[ix_p] = opeatl['cf']
    age[ix_p] = opeatl['age']
    bmleaf_pine[ix_p]=opeatl['bmleaf']
    bmleaf_spruce[ix_p]=opeatl['bmleaf']
    bmleaf_decid[ix_p]=opeatl['bmleaf']
    bmroot_pine[ix_p] = opeatl['bmroot']
    bmroot_spruce[ix_p] = opeatl['bmroot']
    bmroot_decid[ix_p] = opeatl['bmroot']
    # stump
    bmstump_pine[ix_p] = opeatl['bmall']
    bmstump_spruce[ix_p] = opeatl['bmall']
    bmstump_decid[ix_p] = opeatl['bmall']
    # core
    bmcore_pine[ix_p] = opeatl['bmall']
    bmcore_spruce[ix_p] = opeatl['bmall']
    bmcore_decid[ix_p] = opeatl['bmall']
    # crown
    bmtop_pine[ix_p] = opeatl['bmall']
    bmtop_spruce[ix_p] = opeatl['bmall']
    bmtop_decid[ix_p] = opeatl['bmall']
    # livebranch
    bmlivebranch_pine[ix_p] = opeatl['bmall']
    bmlivebranch_spruce[ix_p] = opeatl['bmall']
    bmlivebranch_decid[ix_p] = opeatl['bmall']
    # deadbranch
    bmdeadbranch_pine[ix_p] = opeatl['bmall']
    bmdeadbranch_spruce[ix_p] = opeatl['bmall']
    bmdeadbranch_decid[ix_p] = opeatl['bmall']
    LAI_pine[ix_p] = opeatl['LAIpine']
    LAI_spruce[ix_p] = opeatl['LAIspruce']
    LAI_decid[ix_p] = opeatl['LAIdecid']
    stand_density[ix_p] = opeatl['bmall']
    '''
    
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
    GisData = {'catchment_mask': cmask, 
               'dem': dem, 
               'flow_accumulation': flowacc, 
               'flow_direction': flowpoint, 
               'slope': slope, 
               'twi': twi, 
               'top_soil': topsoil, 
               'low_soil': lowsoil, 
               'peatland_mask': peatland, 
               'paludified_mask': paludified, 
               'stream_mask': stream, 
               'lake_mask': lake, 
               'road_mask': road, 
               'rock_mask': rockm,
               'nonland_mask': nonland,
               'LAI_pine': LAI_pine, 
               'LAI_spruce': LAI_spruce, 
               'LAI_conif': LAI_pine + LAI_spruce, 
               'LAI_decid': LAI_decid,
               'bm_root_decid': bmroot_decid, 
               'bm_root_spruce': bmroot_spruce,  
               'bm_root_pine': bmroot_pine,
               'bm_leaf_decid': bmleaf_decid, 
               'bm_leaf_spruce': bmleaf_spruce,  
               'bm_leaf_pine': bmleaf_pine,     
               'bm_stump_decid': bmstump_decid, 
               'bm_stump_spruce': bmstump_spruce,  
               'bm_stump_pine': bmstump_pine,  
               'bm_livebranch_decid': bmstump_decid,
               'bm_livebranch_spruce': bmlivebranch_spruce,  
               'bm_livebranch_pine': bmlivebranch_pine,
               'bm_deadbranch_decid': bmdeadbranch_decid,
               'bm_deadbranch_spruce': bmdeadbranch_spruce,  
               'bm_deadbranch_pine': bmdeadbranch_pine,
               'bm_crown_decid': bmtop_decid,
               'bm_crown_spruce': bmtop_spruce,  
               'bm_crown_pine': bmtop_pine,  
               'bm_core_decid': bmcore_decid,
               'bm_core_spruce': bmcore_spruce,  
               'bm_core_pine': bmcore_pine,                 
               'basal_area': ba, 
               'canopy_height': height, 
               'canopy_diameter': cd, 
               'volume': vol,
               'volume_pine': p_vol, 
               'volume_spruce': s_vol, 
               'volume_birch': b_vol, 
               'canopy_fraction': cf, 
               'canopy_fraction_decid': cf_d, 
               'stand_age': age,
               'stand_density': stand_density,
               'site_main_class': maintype, 
               'site_fertility_class': sitetype, 
               'fra_land_class': fraclass,
               'cellsize': cellsize, 
               'info': info, 
               'lat0': lat0, 
               'lon0': lon0, 
               'loc': loc}

    GisData_units = {}
    GisData_meta = {}

    for key in GisData.keys():
        if key == 'dem':
           GisData_units[key] = 'm'
           GisData_meta[key] = 'MML digital elevation model 2023'
        elif key == 'stand_age':
           GisData_units[key] = 'years'
           GisData_meta[key] = 'LUKE VMI 2021'
        elif key == 'canopy_height':
           GisData_units[key] = 'm' 
           GisData_meta[key] = 'LUKE VMI 2021'
        elif key == 'nonland_mask':
           GisData_units[key] = '-' 
           GisData_meta[key] = 'LUKE VMI 2021 (fields etc.)'              
        elif key == 'canopy_diameter':
           GisData_units[key] = 'm'   
           GisData_meta[key] = 'LUKE VMI 2021' 
        elif key == 'canopy_fraction':
           GisData_units[key] = '-'   
           GisData_meta[key] = 'LUKE VMI 2021'              
        elif key == 'basal_area':
           GisData_units[key] = 'm2 m-2'
           GisData_meta[key] = 'LUKE VMI 2021'            
        elif key == 'slope':
           GisData_units[key] = 'degrees'
           GisData_meta[key] = 'dem derivative'
        elif key == 'stand_density':
           GisData_units[key] = 'ntrees m-2'
           GisData_meta[key] = 'basal area and canopy diameter derivative'            
        elif key == 'catchment_mask':
           GisData_units[key] = ''
           GisData_meta[key] = 'dem derivative'            
        elif key == 'twi':
           GisData_units[key] = '-'
           GisData_meta[key] = 'dem derivative'            
        elif 'flow' in key.split('_'):
           GisData_units[key] = '-'   
           GisData_meta[key] = 'dem derivative'               
        elif 'bm' in key.split('_'):
           GisData_units[key] = 'kg m-2'   
           GisData_meta[key] = 'LUKE VMI 2021'                        
        elif 'volume' in key.split('_'):
           GisData_units[key] = 'm3 m-2'    
           GisData_meta[key] = 'LUKE VMI 2021'                                    
        elif 'LAI' in key.split('_'):
           GisData_units[key] = 'm2 m-2'  
           GisData_meta[key] = 'LUKE VMI 2021'                     
        elif 'soil' in key.split('_'):
           GisData_units[key] = '-'
           GisData_meta[key] = 'GTK [soil textures: 4=peat, 3=fine, 3=medium, 1=coarse]'    
        elif ('mask' in key.split('_')) & ('catchment' not in key.split('_')) & ('nonland' not in key.split('_')):
           GisData_units[key] = '-'
           GisData_meta[key] = 'MML maastotietokanta 2023'             
        elif 'class' in key.split('_'):
           GisData_units[key] = '-'
           GisData_meta[key] = 'LUKE VMI 2021'   
        else:
           GisData_units[key] = '-'
           GisData_meta[key] = '-'
            
    if plotgrids is True:
        keys = GisData.keys()
        no_plot = ['cellsize', 'info', 'lat0', 'lon0', 'loc']
        figno = 0
        for var in keys:
            if var not in no_plot:
                figno += 1
        figcol = 2
        figrow = int(figno/figcol)
        figh = int(figrow*3)
        figw = 10
        fig, axs = plt.subplots(figsize=(figw, figh), nrows=figrow, ncols=figcol)

        for key, ax in zip(keys, axs.ravel()):
            if key not in ['cellsize', 'info', 'lat0', 'lon0', 'loc']:
                if key == 'flow_accumulation':
                    im = ax.imshow(GisData[key], cmap='viridis', norm=colors.LogNorm())
                    bar = plt.colorbar(im, ax=ax)
                else:
                    im = ax.imshow(GisData[key], cmap='viridis')
                    bar = plt.colorbar(im, ax=ax) 
                ax.set_title(key) 

        plt.show()
            
    '''
    if plotgrids is False:
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
        plt.imshow(cf_d); plt.colorbar(); plt.title('canopy fraction decid. (-)')
        plt.subplot(223)
        plt.imshow(p_vol+s_vol); plt.colorbar(); plt.title('conif. vol (m3/ha)')
        plt.subplot(224)
        plt.imshow(p_vol); plt.colorbar(); plt.title('decid. vol (m3/ha)')


        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(1e-3*bmleaf_pine); plt.colorbar(); plt.title('pine needles (kg/m2)')
        plt.subplot(222)
        plt.imshow(1e-3*bmleaf_spruce); plt.colorbar(); plt.title('spruce needles (kg/m2)')
        plt.subplot(223)
        plt.imshow(1e-3*bmleaf_decid); plt.colorbar(); plt.title('decid. leaves (kg/m2)')
        plt.subplot(224)
        plt.imshow(ba); plt.colorbar(); plt.title('basal area (m2/ha)')
        
        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(1e-3*bmroot_pine); plt.colorbar(); plt.title('pine roots (kg/m2)')
        plt.subplot(222)
        plt.imshow(1e-3*bmroot_spruce); plt.colorbar(); plt.title('spruce roots (kg/m2)')
        plt.subplot(223)
        plt.imshow(1e-3*bmroot_decid); plt.colorbar(); plt.title('decid. roots (kg/m2)')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(sitetype); plt.colorbar(); plt.title('sitetype')
        plt.subplot(222)
        plt.imshow(maintype); plt.colorbar(); plt.title('maintype')
        plt.subplot(223)
        plt.imshow(fraclass); plt.colorbar(); plt.title('fraclass')

        plt.figure(figsize=(10,6))
        plt.subplot(221)
        plt.imshow(stream); plt.colorbar(); plt.title('stream')
        plt.subplot(222)
        plt.imshow(road); plt.colorbar(); plt.title('roads')
        plt.subplot(223)
        plt.imshow(lake); plt.colorbar(); plt.title('lake')
    '''    
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

    return GisData, GisData_units, GisData_meta


def netcdf_from_dict(data, units, meta, out_fp, dict_meta='', description=''):
    """
    netCDF4 format output file initialization

    Args:
        variables (list): list of variables to be saved in netCDF4
        cmask
        filepath: path for saving results
        filename: filename
        description: description
    """
    from netCDF4 import Dataset, date2num
    from datetime import datetime

    # dimensions
    date_dimension = None
    lat_shape, lon_shape = np.shape(data['catchment_mask'])

    xllcorner = int(float(data['info'][2].split()[1]))
    yllcorner = int(float(data['info'][3].split()[1]))
    cellsize = int(float(data['info'][4].split()[1]))

    xcoords = np.arange(xllcorner, (xllcorner + (lon_shape*cellsize)), cellsize)
    ycoords = np.arange(yllcorner, (yllcorner + (lat_shape*cellsize)), cellsize)
    ycoords = np.flip(ycoords)

    # create dataset and dimensions
    ncf = Dataset(out_fp, 'w', format='NETCDF4_CLASSIC')
    ncf.description = 'GIS dataset' + description
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'MML, GTK, LUKE'

    #ncf.createDimension('time', date_dimension)
    ncf.createDimension('lat', lat_shape)
    ncf.createDimension('lon', lon_shape)

    #date = ncf.createVariable('time', 'f8', ('time',))
    #date.units = 'days since 0001-01-01 00:00:00.0'
    #date.calendar = 'standard'
    #tvec = pd.date_range(pgen['spinup_end'], pgen['end_date']).tolist()[1:]
    #date[:] = date2num(tvec, units=date.units, calendar=date.calendar)

    ivar = ncf.createVariable('lat', 'f8', ('lat',))
    ivar.units = 'ETRS-TM35FIN'
    ivar[:] = ycoords

    jvar = ncf.createVariable('lon', 'f8', ('lon',))
    jvar.units = 'ETRS-TM35FIN'
    jvar[:] = xcoords

    no_save = ['cellsize', 'info', 'lat0', 'lon0', 'loc']

    for var in data.keys():
        if var not in no_save:
            var_dim = ('lat', 'lon')
            variable = ncf.createVariable(
                    var, 'f4', var_dim)
            variable.units = units[var]
            variable.meta = meta[var]

    for var in data.keys():
        if var not in no_save:
            print(var)
            ncf[var][:,:] = data[var][:,:]

    return ncf

def tree_density(diameter, ba):
    '''
    in:
    diameter = tree diameter
    ba = basal area
    out:
    no_trees = number of trees
    '''
    # uniform distribution for comparison
    EPS = np.finfo(float).eps
    tree_ba= np.pi * (0.5*diameter)**2 # the area of an average tree m2
    no_trees = ba / (tree_ba + EPS) # how many trees per total basal area
    # clearcuts cause noise; set n to zero
    no_trees[diameter<0.01] = 0.0
    return no_trees



