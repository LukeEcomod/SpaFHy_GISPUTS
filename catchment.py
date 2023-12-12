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
                'bmstump': 0.01, 'bmcore': 0.01, 'bmall': 0.01}
    opeatl = {'vol': 0.0, 'ba': 0.01, 'height': 0.1, 'cf': 0.01, 'age': 0.0, 'diameter': 0.0,
                  'LAIpine': 0.01, 'LAIspruce': 0.01, 'LAIdecid': 0.1, 'bmroot': 0.01, 'bmleaf': 0.01,
                'bmstump': 0.01, 'bmcore': 0.01, 'bmall': 0.01}
        
    if set_non_forest_as=='nan':
        for key in nofor.keys():
            nofor[key] = np.NaN
            opeatl[key] = np.NaN
    elif set_non_forest_as=='null':
        for key in nofor.keys():
            nofor[key] = 0
            opeatl[key] = 0

        
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
    peatland, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/suo.asc'))
    paludified, _, _, _, _ = read_AsciiGrid(os.path.join(fpath, 'maastotietokanta/soistuma.asc'))
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
    topsoil[np.in1d(topsoil, Water)] = np.NaN
    #topsoil[topsoil == -1.0] = 2.0
    topsoil[np.where(topsoil == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(topsoil != 1.0)] = 2.0
    # reshape back to original grid
    topsoil = topsoil.reshape(rs, cs)
    del rs, cs
    topsoil[np.isfinite(peatland)] = 4.0

    # manipulating bottom soil
    rb, cb = np.shape(botsoil)
    lowsoil = np.ravel(botsoil)
    lowsoil[np.in1d(lowsoil, CoarseTextured)] = 1.0 
    lowsoil[np.in1d(lowsoil, MediumTextured)] = 2.0
    lowsoil[np.in1d(lowsoil, FineTextured)] = 3.0
    lowsoil[np.in1d(lowsoil, Peats)] = 4.0
    lowsoil[np.in1d(lowsoil, Water)] = np.NaN
    #lowsoil[lowsoil == -1.0] = 2.0
    lowsoil[np.where(lowsoil == 4.0) and np.where(peatland.flatten() != 1.0) and np.where(lowsoil != 1.0)] = 2.0
    # reshape back to original grid
    lowsoil = lowsoil.reshape(rb, cb)
    del rb, cb
    lowsoil[np.isfinite(peatland)] = 4.0

    # update waterbody mask
    ix = np.where(topsoil == -1.0)
    stream[ix] = 1.0
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
    ix_n = np.where((vol >= 32727) | (vol == -9999) )  # no satellite cover or not forest land: assign arbitrary values
    ix_p = np.where((vol >= 32727) & (peatland == 1))  # open peatlands: assign arbitrary values
    ix_w = np.where(((vol >= 32727) & (stream == 1)) | ((vol >= 32727) & (lake == 1)))  # waterbodies: leave out
    ix_t = np.where((sitetype >= 32727)) # no sitetypes set
    #cmask[ix_w] = np.NaN  # NOTE: leaves waterbodies out of catchment mask

    lowsoil[ix_w] = np.NaN
    topsoil[ix_w] = np.NaN

    maintype[ix_w] = np.NaN
    maintype[ix_t] = np.NaN

    sitetype[ix_w] = np.NaN
    sitetype[ix_t] = np.NaN
    
    fraclass[ix_w] = np.NaN
    fraclass[ix_t] = np.NaN

    cd = 1e-2*cd # cm to m
    cd[ix_n] = nofor['diameter']
    cd[ix_p] = opeatl['diameter']
    cd[ix_w] = np.NaN

    vol = 1e-4*vol # m3/ha to m3/m2
    vol[ix_n] = nofor['vol']
    vol[ix_p] = opeatl['vol']
    vol[ix_w] = np.NaN 

    p_vol = 1e-4*p_vol # m3/ha to m3/m2    
    p_vol[ix_n] = nofor['vol']
    p_vol[ix_p] = opeatl['vol'] 
    p_vol[ix_w] = np.NaN 
    
    s_vol = 1e-4*s_vol # m3/ha to m3/m2    
    s_vol[ix_n] = nofor['vol']
    s_vol[ix_p] = opeatl['vol'] 
    s_vol[ix_w] = np.NaN 

    b_vol = 1e-4*b_vol # m3/ha to m3/m2        
    b_vol[ix_n] = nofor['vol'] # m3/ha
    b_vol[ix_p] = opeatl['vol'] 
    b_vol[ix_w] = np.NaN 

    ba = 1e-4*ba # m2/ha to m2/m2
    ba[ix_n] = nofor['ba']
    ba[ix_p] = nofor['ba']
    ba[ix_w] = np.NaN

    height = 0.1*height  # dm -> m
    height[ix_n] = nofor['height']
    height[ix_p] = opeatl['height']
    height[ix_w] = np.NaN

    cf = 1e-2*cf # % -> [-]
    cf[ix_n] = nofor['cf']
    cf[ix_p] = opeatl['cf']
    cf[ix_w] = np.NaN

    cf_d = 1e-2*cf_d # % -> [-]
    cf_d[ix_n] = nofor['cf']
    cf_d[ix_p] = opeatl['cf']
    cf_d[ix_w] = np.NaN
    
    age[ix_n] = nofor['age'] # years
    age[ix_p] = opeatl['age']
    age[ix_w] = np.NaN

    # leaf biomasses and one-sided LAI
    bmleaf_pine = 1e-3*(bmleaf_pine) # 1e-3 converts 10kg/ha to kg/m2
    bmleaf_pine[ix_n]=nofor['bmleaf']
    bmleaf_pine[ix_p]=opeatl['bmleaf']
    bmleaf_pine[ix_w]=np.NaN

    bmleaf_spruce = 1e-3*(bmleaf_spruce) # 1e-3 converts 10kg/ha to kg/m2
    bmleaf_spruce[ix_n]=nofor['bmleaf']
    bmleaf_spruce[ix_p]=opeatl['bmleaf']
    bmleaf_spruce[ix_w]=np.NaN

    bmleaf_decid = 1e-3*(bmleaf_decid) # 1e-3 converts 10kg/ha to kg/m2    
    bmleaf_decid[ix_n]=nofor['bmleaf']
    bmleaf_decid[ix_p]=opeatl['bmleaf']
    bmleaf_decid[ix_w]=np.NaN

    LAI_pine = bmleaf_pine*SLA['pine']  # 1e-3 converts 10kg/ha to kg/m2
    LAI_pine[ix_n] = nofor['LAIpine']
    LAI_pine[ix_p] = opeatl['LAIpine']
    LAI_pine[ix_w] = np.NaN

    LAI_spruce = bmleaf_spruce*SLA['spruce']
    LAI_spruce[ix_n] = nofor['LAIspruce']
    LAI_spruce[ix_p] = opeatl['LAIspruce']
    LAI_spruce[ix_w] = np.NaN

    LAI_decid = bmleaf_decid*SLA['decid']
    LAI_decid[ix_n] = nofor['LAIdecid']
    LAI_decid[ix_p] = opeatl['LAIdecid']
    LAI_decid[ix_w] = np.NaN
    
    bmroot_pine = 1e-3*(bmroot_pine)  # kg/m2
    bmroot_pine[ix_n] = nofor['bmroot']
    bmroot_pine[ix_p] = opeatl['bmroot']
    bmroot_pine[ix_w] = np.NaN

    bmroot_spruce = 1e-3*(bmroot_spruce)  # kg/m2
    bmroot_spruce[ix_n] = nofor['bmroot']
    bmroot_spruce[ix_p] = opeatl['bmroot']
    bmroot_spruce[ix_w] = np.NaN

    bmroot_decid = 1e-3*(bmroot_decid)  # kg/m2
    bmroot_decid[ix_n] = nofor['bmroot']
    bmroot_decid[ix_p] = opeatl['bmroot']
    bmroot_decid[ix_w] = np.NaN

    # stump
    bmstump_pine = 1e-3*(bmstump_pine)  # kg/m2
    bmstump_pine[ix_n] = nofor['bmall']
    bmstump_pine[ix_p] = opeatl['bmall']
    bmstump_pine[ix_w] = np.NaN

    bmstump_spruce = 1e-3*(bmstump_spruce)  # kg/m2
    bmstump_spruce[ix_n] = nofor['bmall']
    bmstump_spruce[ix_p] = opeatl['bmall']
    bmstump_spruce[ix_w] = np.NaN

    bmstump_decid = 1e-3*(bmstump_decid)  # kg/m2
    bmstump_decid[ix_n] = nofor['bmall']
    bmstump_decid[ix_p] = opeatl['bmall']
    bmstump_decid[ix_w] = np.NaN
    
    # core
    bmcore_pine = 1e-3*(bmcore_pine)  # kg/m2
    bmcore_pine[ix_n] = nofor['bmall']
    bmcore_pine[ix_p] = opeatl['bmall']
    bmcore_pine[ix_w] = np.NaN

    bmcore_spruce = 1e-3*(bmcore_spruce)  # kg/m2
    bmcore_spruce[ix_n] = nofor['bmall']
    bmcore_spruce[ix_p] = opeatl['bmall']
    bmcore_spruce[ix_w] = np.NaN

    bmcore_decid = 1e-3*(bmcore_decid)  # kg/m2
    bmcore_decid[ix_n] = nofor['bmall']
    bmcore_decid[ix_p] = opeatl['bmall']
    bmcore_decid[ix_w] = np.NaN  
    
    # crown
    bmtop_pine = 1e-3*(bmtop_pine)  # kg/m2
    bmtop_pine[ix_n] = nofor['bmall']
    bmtop_pine[ix_p] = opeatl['bmall']
    bmtop_pine[ix_w] = np.NaN

    bmtop_spruce = 1e-3*(bmtop_spruce)  # kg/m2
    bmtop_spruce[ix_n] = nofor['bmall']
    bmtop_spruce[ix_p] = opeatl['bmall']
    bmtop_spruce[ix_w] = np.NaN

    bmtop_decid = 1e-3*(bmtop_decid)  # kg/m2
    bmtop_decid[ix_n] = nofor['bmall']
    bmtop_decid[ix_p] = opeatl['bmall']
    bmtop_decid[ix_w] = np.NaN  

    # livebranch
    bmlivebranch_pine = 1e-3*(bmlivebranch_pine)  # kg/m2
    bmlivebranch_pine[ix_n] = nofor['bmall']
    bmlivebranch_pine[ix_p] = opeatl['bmall']
    bmlivebranch_pine[ix_w] = np.NaN

    bmlivebranch_spruce = 1e-3*(bmlivebranch_spruce)  # kg/m2
    bmlivebranch_spruce[ix_n] = nofor['bmall']
    bmlivebranch_spruce[ix_p] = opeatl['bmall']
    bmlivebranch_spruce[ix_w] = np.NaN

    bmlivebranch_decid = 1e-3*(bmlivebranch_decid)  # kg/m2
    bmlivebranch_decid[ix_n] = nofor['bmall']
    bmlivebranch_decid[ix_p] = opeatl['bmall']
    bmlivebranch_decid[ix_w] = np.NaN 

    # deadbranch
    bmdeadbranch_pine = 1e-3*(bmdeadbranch_pine)  # kg/m2
    bmdeadbranch_pine[ix_n] = nofor['bmall']
    bmdeadbranch_pine[ix_p] = opeatl['bmall']
    bmdeadbranch_pine[ix_w] = np.NaN

    bmdeadbranch_spruce = 1e-3*(bmdeadbranch_spruce)  # kg/m2
    bmdeadbranch_spruce[ix_n] = nofor['bmall']
    bmdeadbranch_spruce[ix_p] = opeatl['bmall']
    bmdeadbranch_spruce[ix_w] = np.NaN

    bmdeadbranch_decid = 1e-3*(bmdeadbranch_decid)  # kg/m2
    bmdeadbranch_decid[ix_n] = nofor['bmall']
    bmdeadbranch_decid[ix_p] = opeatl['bmall']
    bmdeadbranch_decid[ix_w] = np.NaN     
    
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
               'site_main_class': maintype, 
               'site_fertility_class': sitetype, 
               'fra_land_class': fraclass,
               'cellsize': cellsize, 
               'info': info, 
               'lat0': lat0, 
               'lon0': lon0, 
               'loc': loc}

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

    return GisData


def netcdf_from_dict(data, out_fp, dict_meta='', description=''):
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
    ncf.description = 'GIS dataset : ' + description
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'modified ...'

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
            #variable.units = var_unit

    for var in data.keys():
        if var not in no_save:
            #print(var)
            ncf[var][:,:] = data[var][:,:]


    return ncf