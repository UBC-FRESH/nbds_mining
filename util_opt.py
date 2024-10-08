##################################################################################
# This module contains local utility function definitions used in the notebooks.
##################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import ws3.opt
import pickle
import numpy as np



def schedule_harvest_areacontrol(fm, max_harvest, period=None, acode='harvest', util=0.85, 
                                 target_masks=None, target_areas=None,
                                 target_scalefactors=None,
                                 mask_area_thresh=0.,
                                 verbose=0):
    """
    Implement a priority queue heuristic harvest scheduler.

    This function can do a bunch of stuff depending on the parameters, but the basic default
    behaviour is to automatically calculate a per-development-type target periodic harvest area
    based on the concept of normal age class distributions and optimal rotation ages.
    """
    if not target_areas:
        if not target_masks: # default to AU-wise THLB 
            au_vals = []
            au_agg = []
            for au in fm.theme_basecodes(5):
                mask = '? 1 ? ? ? %s' % au
                masked_area = fm.inventory(0, mask=mask)
                if masked_area > mask_area_thresh:
                    au_vals.append(au)
                else:
                    au_agg.append(au)
                    if verbose > 0:
                        print('adding to au_agg', mask, masked_area)
            if au_agg:
                fm._themes[5]['areacontrol_au_agg'] = au_agg 
                if fm.inventory(0, mask='? ? ? ? ? areacontrol_au_agg') > mask_area_thresh:
                    au_vals.append('areacontrol_au_agg')
            target_masks = ['? 1 ? ? ? %s' % au for au in au_vals]
        target_areas = []
        for i, mask in enumerate(target_masks): # compute area-weighted mean CMAI age for each masked DT set
            masked_area = fm.inventory(0, mask=mask, verbose=verbose)
            if not masked_area: continue
            r = sum((fm.dtypes[dtk].ycomp('totvol').mai().ytp().lookup(0) * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))
            r /= masked_area
            asf = 1. if not target_scalefactors else target_scalefactors[i]  
            ta = max_harvest * (1/r) * fm.period_length * masked_area * asf
            target_areas.append(ta)
    periods = fm.periods if not period else [period]
    for period in periods:
        for mask, target_area in zip(target_masks, target_areas):
            if verbose > 0:
                print('calling areaselector', period, acode, target_area, mask)
            fm.areaselector.operate(period, acode, target_area, mask=mask, verbose=verbose)
    sch = fm.compile_schedule()
    return sch



################################################
# HWP effect
################################################

def calculate_co2_value_stock(fm, i, product_coefficient, decay_rate, product_percentage):      
    """
    Calculate carbon stock for harvested wood products for period `i`.
    """
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f'totvol * {product_coefficient} * {product_percentage}') / 10 * (1 - decay_rate)**(i - j)
        for j in range(1, i + 1)
        ) * 460 * 0.5 * 44 / 12
    )
    

def calculate_initial_co2_value_stock(fm, i, product_coefficient, product_percentage):
    """
    Calculate carbon stock for harvested wood products for period 1.
    """
    return fm.compile_product(i, f'totvol * {product_coefficient} * {product_percentage}') * 0.1 * 460 * 0.5 * 44 / 12 / fm.period_length


def hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value):
    """
    Compile periodic harvested wood products carbon stocks data.
    """
    from util_opt import calculate_co2_value_stock, calculate_initial_co2_value_stock
    data_carbon_stock = {'period': [], 'co2_stock': []}    
    for i in range(0, fm.horizon * 10 + 1):
        period_value = i
        co2_values_stock = []
        for product in products:
            product_coefficient = product_coefficients[product]
            product_percentage = product_percentages[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_stock.append(0)
            if i == 1:
                co2_values_stock.append(hwp_pool_effect_value * calculate_initial_co2_value_stock(fm, i, product_coefficient, product_percentage))
            else:
                co2_values_stock.append(hwp_pool_effect_value * calculate_co2_value_stock(fm, i, product_coefficient, decay_rate, product_percentage))
        co2_value_stock = sum(co2_values_stock) / 1000
        data_carbon_stock['period'].append(period_value)
        data_carbon_stock['co2_stock'].append(co2_value_stock)    
    df_carbon_stock = pd.DataFrame(data_carbon_stock)    
    return df_carbon_stock


def calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage):
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f'totvol * {product_coefficient} * {product_percentage}') * 0.1 * (1 - decay_rate)**(i - j)
        for j in range(1, i + 1)
        ) * 460 * 0.5 * 44 / 12 * decay_rate 
 )


def calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage):
    return fm.compile_product(i, f'totvol * {product_coefficient} * {product_percentage}') * 0.1 * 460 * 0.5 * 44 / 12 * decay_rate  / fm.period_length


# Emission (by year)
def hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value):
    from util_opt import calculate_co2_value_emission, calculate_initial_co2_value_emission
    data_carbon_emission = {'period': [], 'co2_emission': []}    
    for i in range(0, fm.horizon * 10  + 1):
        period_value = i
        co2_values_emission = []        
        for product in products:
            product_coefficient = product_coefficients[product]
            product_percentage = product_percentages[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_emission.append(0)
            elif i == 1:
                co2_values_emission.append(hwp_pool_effect_value * calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
            else:
                co2_values_emission.append(hwp_pool_effect_value * calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
        co2_value_emission = sum(co2_values_emission) / 1000
        data_carbon_emission['period'].append(period_value)
        data_carbon_emission['co2_emission'].append(co2_value_emission)    
    df_carbon_emission = pd.DataFrame(data_carbon_emission)
    return df_carbon_emission


def hwp_carbon_emission_immed(fm):
    data_carbon_emission_immed = {'period': [], 'co2_emission_immed': []}    
    for i in range(0, fm.horizon * 10  + 1):
        period_value = i
        co2_values_emission_immed = []                    
        if i == 0:
            co2_values_emission_immed.append(0)
        else:
            period = math.ceil(i / fm.period_length)
            co2_values_emission_immed.append(fm.compile_product(period, 'totvol') * 0.1 * 460 * 0.5 * 44 / 12 / fm.period_length)
        co2_value_emission_immed = sum(co2_values_emission_immed) / 1000
        data_carbon_emission_immed['period'].append(period_value)
        data_carbon_emission_immed['co2_emission_immed'].append(co2_value_emission_immed)    
    df_carbon_emission_immed = pd.DataFrame(data_carbon_emission_immed)
    return df_carbon_emission_immed

################################################
# Displacement effect
################################################
# Displacement of concrete manufacturing
def calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate):            
    period = math.ceil(i / fm.period_length)
    return fm.compile_product(period,'totvol') * product_coefficients['plumber'] * clt_percentage * credibility / clt_conversion_rate 


# Iterate through the rows of the DataFrame
def emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect):
    from util_opt import  calculate_concrete_volume
    df_emission_concrete_manu = {'period': [], 'co2_concrete_manu': []}
    for i in range(0, fm.horizon *10   + 1 ):
        period_value = i
        co2_concrete_manu = []
        if i == 0:
            co2_concrete_manu = 0
        else:
            concrete_volume = calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate)
            co2_concrete_manu = concrete_volume * co2_concrete_manu_factor * 0.1 / 1000
        df_emission_concrete_manu['period'].append(period_value)
        df_emission_concrete_manu['co2_concrete_manu'].append(co2_concrete_manu)
    # Create a DataFrame from the dictionary
    df_emission_concrete_manu = pd.DataFrame(df_emission_concrete_manu)
    return df_emission_concrete_manu


# Displacement of concrete landfill
def emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect):
    from util_opt import  calculate_concrete_volume
    df_emission_concrete_landfill = {'period': [], 'co2_concrete_landfill': []}   
    # Iterate through the rows of the DataFrame
    for i in range(0, fm.horizon *10   + 1 ):
        period_value = i
        co2_concrete_landfill = []
        if i == 0:
            co2_concrete_landfill = 0
        else:
            concrete_volume = calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate)
            co2_concrete_landfill = concrete_volume * co2_concrete_landfill_factor * 0.1                         
        df_emission_concrete_landfill['period'].append(period_value)
        df_emission_concrete_landfill['co2_concrete_landfill'].append(co2_concrete_landfill)    
    # Create a DataFrame from the dictionary
    df_emission_concrete_landfill = pd.DataFrame(df_emission_concrete_landfill)
    return df_emission_concrete_landfill
################################################

def compile_scenario(fm):
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs}
    df = pd.DataFrame(data)
    return df


def plot_scenario(df):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area (ha)')
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume (m3)')
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_title('Growing Stock (m3)')
    return fig, ax


def plot_results(fm):
    pareas = [fm.compile_product(period, '1.') for period in fm.periods]
    pvols = [fm.compile_product(period, 'totvol') for period in fm.periods]
    df = pd.DataFrame({'period':fm.periods, 'ha':pareas, 'hv':pvols})
    fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    ax[0].set_ylabel('harvest area')
    ax[0].bar(df.period, df.ha)
    ax[1].set_ylabel('harvest volume')
    ax[1].bar(df.period, df.hv)
    ax[2].set_ylabel('harvest volume:area ratio')
    ax[2].bar(df.period, (df.hv/df.ha).fillna(0))
    ax[2].set_ylim(0, None)
    return fig, ax, df


def compile_scenario_maxstock(fm):
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    ocp = [fm.inventory(period, 'ecosystem') for period in fm.periods]
    ocf = [fm.inventory(period, 'total_emissions') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs,
            'ocp':ocp,
            'ocf':ocf}
    df = pd.DataFrame(data)
    return df


def plot_scenario_maxstock(df):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot and label the first subplot for harvested area
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area')
    ax[0].set_xlabel('Period')
    ax[0].set_ylabel('Area (ha)')
    
    # Plot and label the second subplot for harvested volume
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume')
    ax[1].set_xlabel('Period')
    ax[1].set_ylabel('Volume (m3)')

    # Plot and label the third subplot for growing stock
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_xlabel('Period')
    ax[2].set_title('Growing Stock')
    ax[2].set_ylabel('Volume (m3)')

    # Plot and label the fourth subplot for ecosystem carbon stock
    ax[3].bar(df.period, df.ocp)
    ax[3].set_ylim(0, None)
    ax[3].set_title('Ecosystem C stock')
    ax[3].set_xlabel('Period')
    ax[3].set_ylabel('Stock (ton)')

    # # Plot and label the fifth subplot for total carbon emission
    # ax[4].bar(df.period, df.ocf)
    # ax[4].set_ylim(0, None)
    # ax[4].set_title('Total Carbon Emission')
    # ax[4].set_xlabel('Period')
    # ax[4].set_ylabel('tons of C')
    return fig, ax
################################################
# Optimization
################################################

def cmp_c_z(fm, path, expr):
    """
    Compile objective function coefficient (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):        
        d = n.data()    
        if fm.is_harvest(d['acode']):
            result += fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result

def cmp_c_cflw(fm, path, expr, mask=None): # product, all harvest actions
    """
    Compile flow constraint coefficient for product indicator (given ForestModel 
    instance, leaf-to-root-node path, expression to evaluate, and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if fm.is_harvest(d['acode']):
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_caa(fm, path, expr, acodes, mask=None): # product, named actions
    """
    Compile constraint coefficient for product indicator (given ForestModel 
    instance, leaf-to-root-node path, expression to evaluate, list of action codes, 
    and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if d['acode'] in acodes:
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_ci(fm, path, yname, mask=None): # product, named actions
    """
    Compile constraint coefficient for inventory indicator (given ForestModel instance, 
    leaf-to-root-node path, expression to evaluate, and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result[t] = fm.inventory(t, yname=yname, age=d['_age'], dtype_keys=[d['_dtk']]) 
        #result[t] = fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']]) 
    return result

def gen_scenario(fm, name='base', util=0.85, harvest_acode='harvest',
                 cflw_ha={}, cflw_hv={}, 
                 cgen_ha={}, cgen_hv={}, cgen_gs={}, 
                 tvy_name='totvol', obj_mode='max_hv', mask=None):
    from functools import partial
    import numpy as np
    coeff_funcs = {}
    cflw_e = {}
    cgen_data = {}
    acodes = ['null', harvest_acode] # define list of action codes
    vexpr = '%s * %0.2f' % (tvy_name, util) # define volume expression
    if obj_mode == 'max_hv': # maximize harvest volume
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = vexpr
    elif obj_mode == 'min_ha': # minimize harvest area
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = '1.'
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode)        
    coeff_funcs['z'] = partial(cmp_c_z, expr=zexpr) # define objective function coefficient function  
    T = fm.periods
    if cflw_ha: # define even flow constraint (on harvest area)
        cname = 'cflw_ha'
        coeff_funcs[cname] = partial(cmp_c_caa, expr='1.', acodes=[harvest_acode], mask=None) 
        cflw_e[cname] = cflw_ha
    if cflw_hv: # define even flow constraint (on harvest volume)
        cname = 'cflw_hv'
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cflw_e[cname] = cflw_hv         
    if cgen_ha: # define general constraint (harvest area)
        cname = 'cgen_ha'
        coeff_funcs[cname] = partial(cmp_c_caa, expr='1.', acodes=[harvest_acode], mask=None) 
        cgen_data[cname] = cgen_ha
    if cgen_hv: # define general constraint (harvest volume)
        cname = 'cgen_hv'
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cgen_data[cname] = cgen_hv
    if cgen_gs: # define general constraint (growing stock)
        cname = 'cgen_gs'
        coeff_funcs[cname] = partial(cmp_c_ci, yname=tvy_name, mask=None)
        cgen_data[cname] = cgen_gs
    return fm.add_problem(name, coeff_funcs, cflw_e, cgen_data=cgen_data, acodes=acodes, sense=sense, mask=mask)


def run_scenario(fm, obj_mode, scenario_name='base'):
    import gurobipy as grb
    initial_inv_equit = 869737. #ha
    initial_gs_equit = 106582957.  #m3   
    initial_inv_red = 390738.
    initial_gs_red =18018809.
    initial_inv_gold = 191273.
    initial_gs_gold = 7017249.
    aac_equity =  18255528. # AAC per year * 10
    aac_red =  1072860. # AAC per year * 10
    aac_gold =  766066. # AAC per year * 10
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}

    if scenario_name == 'no_cons': 
        # no_cons scenario : 
        print('running no constraints scenario')
        
    # Golden Bear scenarios
    elif scenario_name == 'bau_gldbr': 
        # Business as usual scenario for Golden Bear mining site: 
        print('running business as usual scenario for the Golden Bear mine site')
        cgen_hv = {'lb':{1:aac_gold}, 'ub':{1:aac_gold}}
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)

    # Red Chris Scenarios
    elif scenario_name == 'bau_redchrs': 
        # Business as usual scenario for the Red Chris mining site: 
        print('running business as usual scenario for the Red Chris mining site')
        cgen_hv = {'lb':{1:aac_red}, 'ub':{1:aac_red}} 
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)

    elif scenario_name == 'redchrs_gs_hv_ha_100': 
        # BAU scenario, plus harvest area general constraints 100%
        print('running alternative scenario with harvest area constraints (100%)')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_ha = {'lb':{1:0}, 'ub':{1:initial_inv_red*1}}
        cgen_hv = {'lb':{1:0.9*aac_red}, 'ub':{1:aac_red}} # at least 90% of aac
        cgen_gs = {'lb':{10:initial_gs_red}, 'ub':{10:initial_gs_red*10}} #Not less than 90% of initial growing stock at the end

    elif scenario_name == 'redchrs_gs_hv_ha_90': 
        # BAU scenario, plus harvest area general constraints 100%
        print('running alternative scenario with harvest area constraints (90%)')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_ha = {'lb':{1:0}, 'ub':{1:initial_inv_red*0.9}}
        cgen_hv = {'lb':{1:0.9*aac_red}, 'ub':{1:aac_red}} # at least 90% of aac
        cgen_gs = {'lb':{10:initial_gs_red}, 'ub':{10:initial_gs_red*10}} #Not less than 90% of initial growing stock at the end

    
    # Equity Silver scenarios
    elif scenario_name == 'bau_eqtslvr': 
        # Business as usual scenario for the Equity Silver mining site: 
        print('running business as usual scenario for the Equity Silver mining site')
        cgen_hv = {'lb':{1:0.7*aac_equity}, 'ub':{1:0.7*aac_equity}} 
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        
    

    
    elif scenario_name == 'base-cgen_ha_90%': 
        # Base scenario, plus harvest area general constraints 90%
        print('running base scenario plus harvest area constraints')
        cgen_ha = {'lb':{1:initial_inv*0.1}, 'ub':{1:initial_inv*0.9}}   
    elif scenario_name == 'base-cgen_ha_80%': 
        # Base scenario, plus harvest area general constraints 80%
        print('running base scenario plus harvest area constraints')
        cgen_ha = {'lb':{1:initial_inv*0.1}, 'ub':{1:initial_inv*0.8}}
    elif scenario_name == 'base-cgen_ha_0%': 
        # Base scenario, plus harvest area general constraints 70%
        print('running base scenario plus harvest area constraints 0%')
        cgen_ha = {'lb':{1:initial_inv*1}, 'ub':{1:initial_inv*1}} 
    elif scenario_name == 'base-cgen_hv': 
        # Base scenario, plus harvest volume general constraints
        print('running base scenario plus harvest volume constraints')
        cgen_hv = {'lb':{1:100000.}, 'ub':{1:100100.}}    
    elif scenario_name == 'base-cgen_gs': 
        # Base scenario, plus growing stock general constraints
        print('running base scenario plus growing stock constraints')
        cgen_gs = {'lb':{10:10000000.}, 'ub':{10:10000100.}}
    elif scenario_name == 'base-cgen_gs_ha_100': 
        # Base scenario, plus growing stock general constraints
        print('running maxmizie harvest scenario scenario plus growing stock constraints plus harvest area constraints 100%')
        cgen_gs = {'lb':{x:initial_gs*0.9 for x in range(1,11)}, 'ub':{x:initial_gs*100 for x in range(1,11)}} #Not less than 90% of initial growing stock
        # cgen_hv = {'lb':{20:AAC-1}, 'ub':{20:AAC}} #Achieve the Annual Allowable Cut
        cgen_ha = {'lb':{1:initial_inv*0.1}, 'ub':{1:initial_inv*1}} 
    
    else:
        assert False # bad scenario name
    p = gen_scenario(fm=fm, 
                     name=scenario_name, 
                     cflw_ha=cflw_ha, 
                     cflw_hv=cflw_hv,
                     cgen_ha=cgen_ha,
                     cgen_hv=cgen_hv,
                     cgen_gs=cgen_gs,
                    obj_mode=obj_mode)
    fm.reset()
    m = p.solve()
    if m.status != grb.GRB.OPTIMAL:
        print('Model not optimal.')
        sys.exit()
    sch = fm.compile_schedule(p)
    fm.apply_schedule(sch, 
                      force_integral_area=False, 
                      override_operability=False,
                      fuzzy_age=False,
                      recourse_enabled=False,
                      verbose=False,
                      compile_c_ycomps=True)
    # df = compile_scenario(fm)
    # fig, ax = plot_scenario(df)
    return sch


##############################################################
# Implement a simple function to run CBM from ws3 export data
##############################################################
def run_cbm_maxstock(sit_config, sit_tables, n_steps):
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables['sit_classifiers'],
                                sit_disturbance_types=sit_tables['sit_disturbance_types'],
                                sit_age_classes=sit_tables['sit_age_classes'],
                                sit_inventory=sit_tables['sit_inventory'],
                                sit_yield=sit_tables['sit_yield'],
                                sit_events=sit_tables['sit_events'],
                                sit_transitions=sit_tables['sit_transitions'],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)      
    return cbm_output

def run_cbm(df_carbon_stock, df_carbon_emission, df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot=True):
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables['sit_classifiers'],
                                sit_disturbance_types=sit_tables['sit_disturbance_types'],
                                sit_age_classes=sit_tables['sit_age_classes'],
                                sit_inventory=sit_tables['sit_inventory'],
                                sit_yield=sit_tables['sit_yield'],
                                sit_events=sit_tables['sit_events'],
                                sit_transitions=sit_tables['sit_transitions'],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',                        
                     'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']
    dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
                 'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']
    biomass_result = pi[['timestep']+biomass_pools]
    dom_result = pi[['timestep']+dom_pools]
    total_eco_result = pi[['timestep']+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({'Year':pi['timestep'],
                                         'Biomass':pi[biomass_pools].sum(axis=1),
                                         'DOM':pi[dom_pools].sum(axis=1),
                                         'Total Ecosystem': pi[biomass_pools+dom_pools].sum(axis=1)})
    annual_carbon_stocks = annual_carbon_stocks.groupby('Year').sum()
    df_carbon_stock = df_carbon_stock.groupby('period').sum()    
    annual_carbon_stocks['HWP'] = df_carbon_stock['co2_stock']        
    annual_carbon_stocks['Total Ecosystem'] += df_carbon_stock['co2_stock']
    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True,  figsize=(8, 8))
        annual_carbon_stocks.groupby('Year').sum().plot(ax=axes[0], xlim=(0, n_steps), ylim=(0, None))
        axes[0].set_title('Carbon stocks over years')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Carbon stocks')
        # plt.show()
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])    
    ecosystem_decay_emissions_pools = [
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir']
    GrossGrowth_pools = [
        'DeltaBiomass_AG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'DeltaBiomass_BG',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput']
    ecosystem_decay_emissions_result = fi[['timestep']+ecosystem_decay_emissions_pools]
    GrossGrowth_result = fi[['timestep']+GrossGrowth_pools]
    net_emission_result = fi[['timestep']+ecosystem_decay_emissions_pools+GrossGrowth_pools]
    annual_net_emission = pd.DataFrame({ "Year": fi["timestep"],
                                        "Ecosystem decay emission": 44/12 * fi[ecosystem_decay_emissions_pools].sum(axis=1),
                                        "Gross growth": 44/12 * -1*fi[GrossGrowth_pools].sum(axis=1),
                                        "Net emission": 44/12 * (fi[ecosystem_decay_emissions_pools].sum(axis=1)-fi[GrossGrowth_pools].sum(axis=1))})
    annual_net_emission = annual_net_emission.groupby('Year').sum()
    
    df_carbon_emission =  df_carbon_emission.groupby('period').sum()
    df_carbon_emission_immed =  df_carbon_emission_immed.groupby('period').sum()
    
    df_emission_concrete_manu = -1 * df_emission_concrete_manu.groupby('period').sum()
    df_emission_concrete_landfill = -1 * df_emission_concrete_landfill.groupby('period').sum()
    annual_net_emission['HWP'] =  (1 - release_immediately_value) * df_carbon_emission['co2_emission'] 
    annual_net_emission['Carbon release immediately'] = release_immediately_value * df_carbon_emission_immed['co2_emission_immed']
    annual_net_emission['Concrete_manufacturing'] = df_emission_concrete_manu['co2_concrete_manu']
    annual_net_emission['Concrete_landfill'] = df_emission_concrete_landfill['co2_concrete_landfill']
    annual_net_emission['Net emission'] += annual_net_emission['HWP']
    annual_net_emission['Net emission'] += annual_net_emission['Carbon release immediately'] 
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_manufacturing']
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_landfill']
    if plot:
        annual_net_emission.groupby('Year').sum().plot(ax=axes[1], xlim = (0, n_steps)).axhline(y=0, color='red', linestyle='--') 
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_title('Carbon emission over years')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Carbon emission')
    return annual_carbon_stocks, annual_net_emission


def stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode):   
    decay_rates = {'plumber':math.log(2.)/35., 'ppaper':math.log(2.)/2.}
    product_coefficients = {'plumber':0.9, 'ppaper':0.1}
    product_percentages = {'plumber':0.5, 'ppaper':1.}
    products = ['plumber', 'ppaper']
    clt_conversion_rate = 1.
    co2_concrete_manu_factor = 298.
    concrete_density = 2.40 #ton/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_alt_scenario = run_scenario(fm, obj_mode, scenario_name)
    df = compile_scenario(fm)
    plot_scenario(df)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value)
    df_carbon_emission_immed = hwp_carbon_emission_immed(fm)

    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect)
    disturbance_type_mapping = [{'user_dist_type': 'harvest', 'default_dist_type': 'Clearcut harvesting without salvage'},
                            {'user_dist_type': 'fire', 'default_dist_type': 'Wildfire'}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[5] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                       hardwood_volume_yname='hwdvol', 
                                       admin_boundary='British Columbia', 
                                       eco_boundary='Montane Cordillera',
                                       disturbance_type_mapping=disturbance_type_mapping)
    cbm_output_1, cbm_output_2 = run_cbm(df_carbon_stock, df_carbon_emission,  df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot = False)
    return cbm_output_1, cbm_output_2     


def stock_emission_scenario_equivalent(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode):   
    decay_rates = {'plumber':math.log(2.)/35., 'ppaper':math.log(2.)/2.}
    product_coefficients = {'plumber':0.9, 'ppaper':0.1}
    product_percentages = {'plumber':0.5, 'ppaper':1.}
    products = ['plumber', 'ppaper']
    clt_conversion_rate = 1.
    co2_concrete_manu_factor = 298.
    concrete_density = 2.40 #ton/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_base_scenari = schedule_harvest_areacontrol(fm, max_harvest) #equivalent harvesting with heuristics
    df = compile_scenario(fm)
    plot_scenario(df)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates, hwp_pool_effect_value)
    df_carbon_emission_immed = hwp_carbon_emission_immed(fm)
    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect)
    disturbance_type_mapping = [{'user_dist_type': 'harvest', 'default_dist_type': 'Clearcut harvesting without salvage'},
                            {'user_dist_type': 'fire', 'default_dist_type': 'Wildfire'}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[5] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                       hardwood_volume_yname='hwdvol', 
                                       admin_boundary='British Columbia', 
                                       eco_boundary='Montane Cordillera',
                                       disturbance_type_mapping=disturbance_type_mapping)
    cbm_output_3, cbm_output_4 = run_cbm(df_carbon_stock, df_carbon_emission,  df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot = False)
    return cbm_output_3, cbm_output_4     


def plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps, case_study, obj_mode, scenario_name, output_pdf_path):
    if not os.path.exists(output_pdf_path):
        os.makedirs(output_pdf_path)
    output_filename = f"{case_study}_{obj_mode}_{scenario_name}_Carbon_emissions_stocks.pdf"
    output_file_path = os.path.join(output_pdf_path, output_filename)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 10))   
    cbm_output_1.groupby('Year').sum().plot(ax=axes[0, 0], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 0].set_title('Carbon stocks over years (alternative scenario)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Carbon stocks')   
    cbm_output_2.groupby('Year').sum().plot(ax=axes[1, 0], xlim=(0, n_steps))
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_title('Carbon emission over years (alternative scenario)')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon emission')
    cbm_output_3.groupby('Year').sum().plot(ax=axes[0, 1], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 1].set_title('Carbon stocks over years (base scenario)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Carbon stocks')    
    cbm_output_4.groupby('Year').sum().plot(ax=axes[1, 1], xlim=(0, n_steps))
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Carbon emission over years (base scenario)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Carbon emission')    
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.show()
    


def scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps, case_study, obj_mode, scenario_name, output_pdf_path):
    if not os.path.exists(output_pdf_path):
        os.makedirs(output_pdf_path)
    output_filename = f"{case_study}_{obj_mode}_{scenario_name}_net_emission_difference.pdf"
    output_file_path = os.path.join(output_pdf_path, output_filename)
    cbm_output_2.reset_index(drop=False, inplace=True)
    dif_scenario = pd.DataFrame({"Year": cbm_output_2["Year"],
                       "Net emission": cbm_output_2['Net emission'] - cbm_output_4['Net emission']})
    ax = dif_scenario.groupby('Year').sum().plot(xlim = (0, n_steps))
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Net emission difference between base and alternative scenarios')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net Carbon emission diffrence')   
    dollar_per_ton = abs(budget_input / dif_scenario.iloc[:25]['Net emission'].sum())
    print( "Net emission difference", dif_scenario.iloc[:25]['Net emission'].sum())
    print( "Net emission base scenario", cbm_output_2.iloc[:25]['Net emission'].sum())
    print( "Net emission alternative scenario", cbm_output_4.iloc[:25]['Net emission'].sum())    
    print('dollar_per_ton is: ', dollar_per_ton)
    plt.savefig(output_file_path)
    return ax


def results_scenarios(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode, output_csv_path, output_pdf_path, pickle_output_base,  
                  pickle_output_alter):
    from util_opt import stock_emission_scenario, plot_scenarios, scenario_dif, stock_emission_scenario_equivalent

    # Ensure output path exists
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    # Define pickle file paths_alter
    pickle_file_1 = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_1.pkl')
    pickle_file_2 = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_2.pkl')

   # Check if pickled cbm_output_1 and cbm_output_2 already exist
    if pickle_output_alter and os.path.exists(pickle_file_1) and os.path.exists(pickle_file_2):
        # Load pickle files if they exist
        with open(pickle_file_1, 'rb') as f:
            cbm_output_1 = pickle.load(f)
        with open(pickle_file_2, 'rb') as f:
            cbm_output_2 = pickle.load(f)
        print("Loaded cbm_output_1 and cbm_output_2 from pickle files.")
    else:
        cbm_output_1, cbm_output_2 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode)
        with open(pickle_file_1, 'wb') as f:
            pickle.dump(cbm_output_1, f)
        with open(pickle_file_2, 'wb') as f:
            pickle.dump(cbm_output_2, f)
        print("Saved cbm_output_1 and cbm_output_2 as pickle files.")

    # Save cbm_output_2 as CSV
    cbm_output_2_df = pd.DataFrame(cbm_output_2)
    cbm_output_2_file = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_2.csv')
    cbm_output_2_df.to_csv(cbm_output_2_file, index=False)
    # print(cbm_output_2)

    fm.reset()

    # Define pickle file paths_base
    pickle_file_3 = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_3.pkl')
    pickle_file_4 = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.pkl')

    # Check if pickled cbm_output_3 and cbm_output_4 already exist
    if pickle_output_base and os.path.exists(pickle_file_3) and os.path.exists(pickle_file_4):
        # Load pickle files if they exist
        with open(pickle_file_3, 'rb') as f:
            cbm_output_3 = pickle.load(f)
        with open(pickle_file_4, 'rb') as f:
            cbm_output_4 = pickle.load(f)
        print("Loaded cbm_output_3 and cbm_output_4 from pickle files.")
    else:
        # Run base scenario if pickle files don't exist
        if case_study == 'redchrs':
            scenario_name = 'bau_redchrs'
        elif case_study == 'eqtslvr':
            scenario_name = 'bau_eqtslvr'
        elif case_study == 'gldbr':
            scenario_name = 'bau_gldbr'

        cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode)

        # Save cbm_output_3 and cbm_output_4 as pickle
        with open(pickle_file_3, 'wb') as f:
            pickle.dump(cbm_output_3, f)
        with open(pickle_file_4, 'wb') as f:
            pickle.dump(cbm_output_4, f)
        print("Saved cbm_output_3 and cbm_output_4 as pickle files.")

    # Save cbm_output_4 as CSV
    cbm_output_4_df = pd.DataFrame(cbm_output_4)
    cbm_output_4_file = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.csv')
    cbm_output_4_df.to_csv(cbm_output_4_file, index=False)
    # print(cbm_output_4)

    # Plot scenarios
    plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps, case_study, obj_mode, scenario_name, output_pdf_path)
    
    # Scenario difference plot
    dif_plot = scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps, case_study, obj_mode, scenario_name, output_pdf_path)


def cbm_report(fm, cbm_output, biomass_pools, dom_pools, fluxes, gross_growth):
    # Add carbon pools indicators 
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    annual_carbon_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Biomass': pi[biomass_pools].sum(axis=1),
                                         'DOM': pi[dom_pools].sum(axis=1),
                                         'Ecosystem': pi[biomass_pools + dom_pools].sum(axis=1)})
    
    annual_product_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Product': pi['Products']})
    
    annual_stock_change = annual_carbon_stock[['Year', 'Ecosystem']].copy()
    annual_stock_change['Stock_Change'] = annual_stock_change['Ecosystem'].diff()
    annual_stock_change = annual_stock_change[['Year', 'Stock_Change']]
    annual_stock_change.loc[annual_stock_change['Year'] == 0, 'Stock_Change'] = 0
     
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    annual_all_emission = pd.DataFrame({'Year': fi['timestep'],
                                         'All_Emissions': fi[fluxes].sum(axis=1)})
    
    annual_gross_growth = pd.DataFrame({'Year': fi['timestep'],
                                        'Gross_Growth': fi[gross_growth].sum(axis=1)})
     
    n_steps = fm.horizon * fm.period_length
    annual_carbon_stock.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None), xlabel="Year", ylabel="Stock (ton C)",
        title="Annual Carbon Stock"
    )

    annual_all_emission.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Emission", xlabel="Year", ylabel="Stock (ton C)"
    )

    annual_stock_change.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Stock Change", xlabel="Year", ylabel="tons of C"
    )

    annual_gross_growth.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Forest Gross Growth", xlabel="Year", ylabel="tons of C"
    )

    df_cs = annual_carbon_stock.groupby('Year').sum()
    df_ae = annual_all_emission.groupby('Year').sum()
    df_gg = annual_gross_growth.groupby('Year').sum()
    df_sc = annual_stock_change.groupby('Year').sum()
    
    # Show the results (tables)
    print("\n--- Annual Carbon Stock ---")
    print(df_cs)
    
    print("\n--- Annual Ecosystem Carbon Emissions ---")
    print(df_ae)
    
    print("\n--- Annual Forest Gross Growth ---")
    print(df_gg)
    
    print("\n--- Annual Stock Change ---")
    print(df_sc)

    # Optionally, return the DataFrames for external use
    return df_cs, df_ae, df_gg, df_sc

# def cbm_report(fm, cbm_output, biomass_pools, dom_pools, fluxes, gross_growth):
#     # Add carbon pools indicators 
#     pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
#                                                   left_on=["identifier", "timestep"], 
#                                                   right_on=["identifier", "timestep"])

#     # Create annual carbon stock DataFrame
#     annual_carbon_stock = pd.DataFrame({'Year': pi['timestep'],
#                                          'Biomass': pi[biomass_pools].sum(axis=1),
#                                          'DOM': pi[dom_pools].sum(axis=1),
#                                          'Ecosystem': pi[biomass_pools + dom_pools].sum(axis=1)})
    
#     # Create annual product stock DataFrame (optional, not needed in the final output)
#     annual_product_stock = pd.DataFrame({'Year': pi['timestep'],
#                                          'Product': pi['Products']})
    
#     # Create annual stock change DataFrame
#     annual_stock_change = annual_carbon_stock[['Year', 'Ecosystem']].copy()
#     annual_stock_change['Stock_Change'] = annual_stock_change['Ecosystem'].diff()
#     annual_stock_change = annual_stock_change[['Year', 'Stock_Change']]
#     annual_stock_change.loc[annual_stock_change['Year'] == 0, 'Stock_Change'] = 0
     
#     # Create emissions DataFrame
#     fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
#                                                   left_on=["identifier", "timestep"], 
#                                                   right_on=["identifier", "timestep"])
    
#     annual_all_emission = pd.DataFrame({'Year': fi['timestep'],
#                                          'All_Emissions': fi[fluxes].sum(axis=1)})
    
#     # Create gross growth DataFrame
#     annual_gross_growth = pd.DataFrame({'Year': fi['timestep'],
#                                         'Gross_Growth': fi[gross_growth].sum(axis=1)})
     
#     # Merge all the DataFrames into one final DataFrame based on 'Year'
#     final_df = pd.merge(annual_carbon_stock, annual_all_emission, on='Year', how='left')
#     final_df = pd.merge(final_df, annual_gross_growth, on='Year', how='left')
#     final_df = pd.merge(final_df, annual_stock_change, on='Year', how='left')

#     # Optional: set 'Year' as the index (if you prefer the format with 'Year' as an index)
#     final_df.set_index('Year', inplace=True)

#     # Show the final table
#     print("\n--- Final Carbon Report ---")
#     print(final_df)

#     # Plot the graphs as before (optional)
#     n_steps = fm.horizon * fm.period_length
#     final_df[['Biomass', 'DOM', 'Ecosystem']].plot(
#         figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None), xlabel="Year", ylabel="Stock (ton C)",
#         title="Annual Carbon Stock"
#     )
#     final_df[['All_Emissions']].plot(
#         figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
#         title="Annual Ecosystem Carbon Emission", xlabel="Year", ylabel="Stock (ton C)"
#     )
#     final_df[['Stock_Change']].plot(
#         figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
#         title="Annual Ecosystem Carbon Stock Change", xlabel="Year", ylabel="tons of C"
#     )
#     final_df[['Gross_Growth']].plot(
#         figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
#         title="Annual Forest Gross Growth", xlabel="Year", ylabel="tons of C"
#     )

#     # Return the final DataFrame
#     return final_df
    

def compare_ws3_cbm(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, plots):
    import numpy as np
    eco_pools = biomass_pools + dom_pools
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    df_cbm = pd.DataFrame({'period': pi["timestep"] * 0.1, 
                       'biomass_stock': pi[biomass_pools].sum(axis=1),
                       'dom_stock': pi[dom_pools].sum(axis=1),
                       'eco_stock': pi[eco_pools].sum(axis=1)}).groupby('period').sum().iloc[1::10, :].reset_index()
    df_cbm['period'] = (df_cbm['period'] + 0.9).astype(int)

    df_cbm['eco_stock_change'] = df_cbm['eco_stock'].diff()
    df_cbm.at[0, 'eco_stock_change'] = 0.

    df_ws3 = pd.DataFrame({'period': fm.periods,
                           'biomass_stock': [sum(fm.inventory(period, pool) for pool in ['biomass']) for period in fm.periods],
                           'dom_stock': [sum(fm.inventory(period, pool) for pool in ['DOM']) for period in fm.periods],
                           'eco_stock': [sum(fm.inventory(period, pool) for pool in ['ecosystem']) for period in fm.periods]})

    df_ws3['eco_stock_change'] = df_ws3['eco_stock'].diff()
    df_ws3.at[0, 'eco_stock_change'] = 0.

    if plots == "whole":
        # Create a figure for all comparisons in one plot
        plt.figure(figsize=(10, 6))
    
        # Plotting the ecosystem stock comparison
        plt.plot(df_cbm['period'], df_cbm['eco_stock'], label='CBM Ecosystem Stock')
        plt.plot(df_ws3['period'], df_ws3['eco_stock'], label='WS3 Ecosystem Stock')
    
        # Plotting the biomass stock comparison
        plt.plot(df_cbm['period'], df_cbm['biomass_stock'], label='CBM Biomass Stock')
        plt.plot(df_ws3['period'], df_ws3['biomass_stock'], label='WS3 Biomass Stock')
    
        # Plotting the DOM stock comparison
        plt.plot(df_cbm['period'], df_cbm['dom_stock'], label='CBM DOM Stock')
        plt.plot(df_ws3['period'], df_ws3['dom_stock'], label='WS3 DOM Stock')
    
        # Set labels and title
        plt.xlabel('Period')
        plt.ylabel('Stock (ton C)')
        plt.ylim(0, None)  # Ensure y-axis starts from 0
    
        # Customize x-axis ticks to show every 2 periods
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        plt.xticks(ticks)
        
        # Add a legend to differentiate the lines
        plt.legend()

    if plots == "individual":
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        
        # Define x-axis ticks (0 to 20 with a step of 2)
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        
        # Plotting the ecosystem stock comparison
        axs[0].plot(df_cbm['period'], df_cbm['eco_stock'], label='cbm ecosystem stock')
        axs[0].plot(df_ws3['period'], df_ws3['eco_stock'], label='ws3 ecosystem stock')
        axs[0].set_xlabel('Period')
        axs[0].set_ylabel('Stock (ton C)')
        # axs[0].set_ylim(0, None)  # Set y-axis to start from 0
        axs[0].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[0].legend()
        
        # Plotting the biomass stock comparison
        axs[1].plot(df_cbm['period'], df_cbm['biomass_stock'], label='cbm biomass stock')
        axs[1].plot(df_ws3['period'], df_ws3['biomass_stock'], label='ws3 biomass stock')
        axs[1].set_xlabel('Period')
        axs[1].set_ylabel('Stock (ton C)')
        # axs[1].set_ylim(0, None)  # Set y-axis to start from 0
        axs[1].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[1].legend()
        
        # Plotting the DOM stock comparison
        axs[2].plot(df_cbm['period'], df_cbm['dom_stock'], label='cbm dom stock')
        axs[2].plot(df_ws3['period'], df_ws3['dom_stock'], label='ws3 dom stock')
        axs[2].set_xlabel('Period')
        axs[2].set_ylabel('Stock (ton C)')
        # axs[2].set_ylim(0, None)  # Set y-axis to start from 0
        axs[2].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return df_cbm, df_ws3

                
# def plugin_c_curves(fm, c_curves_p, c_curves_f, pools, fluxes):
#     # for dtype_key in dt_tuples:
#     for dtype_key in fm.dtypes:
#         dt = fm.dt(dtype_key)
#          mask = ('?', '?', '?', '?', dtype_key[4], dtype_key[5])
#         for _mask, ytype, curves in fm.yields:
#             if _mask != mask: continue # we know there will be a match so this works
#             print('found match for mask', mask)
#             # print('found match for development key', dtype_key)
#             pool_data = c_curves_p.loc[' '.join(dtype_key)]
#             for yname in pools:
#                 points = list(zip(pool_data.index.values, pool_data[yname]))
#                 curve = fm.register_curve(ws3.core.Curve(yname, 
#                                                          points=points, 
#                                                          type='a', 
#                                                          is_volume=False,
#                                                          xmax=fm.max_age,
#                                                          period_length=fm.period_length))
#                 curves.append((yname, curve))
#                 dt.add_ycomp('a', yname, curve)
#             flux_data = c_curves_f.loc[' '.join(dtype_key)]
#             for yname in fluxes:
#                 points = list(zip(flux_data.index.values, flux_data[yname]))
#                 curve = fm.register_curve(ws3.core.Curve(yname, 
#                                                          points=points, 
#                                                          type='a', 
#                                                          is_volume=False,
#                                                          xmax=fm.max_age,
#                                                          period_length=fm.period_length))
#                 curves.append((yname, curve))
#                 dt.add_ycomp('a', yname, curve)
#         #mask = '? ? %s ? %' % (dtype_key[2], dtype_key[4])
#         #points = c_curves_p





def plugin_c_curves(fm, c_curves_p, pools):
    # for dtype_key in dt_tuples:
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        mask = ('?', '?', '?', '?', dtype_key[4], dtype_key[5])
        for _mask, ytype, curves in fm.yields:
            if _mask != mask: continue # we know there will be a match so this works
            print('found match for mask', mask)
            # print('found match for development key', dtype_key)
            pool_data = c_curves_p.loc[' '.join(dtype_key)]
            for yname in pools:
                points = list(zip(pool_data.index.values, pool_data[yname]))
                curve = fm.register_curve(ws3.core.Curve(yname, 
                                                         points=points, 
                                                         type='a', 
                                                         is_volume=False,
                                                         xmax=fm.max_age,
                                                         period_length=fm.period_length))
                curves.append((yname, curve))
                dt.add_ycomp('a', yname, curve)

#Without repeatition
# def plugin_c_curves(fm, c_curves_p, pools):
#     processed_masks = set()  # To track processed masks
#     for dtype_key in fm.dtypes:
#         dt = fm.dt(dtype_key)
#         mask = ('?', '?', '?', '?', dtype_key[4], dtype_key[5])
        
#         if mask in processed_masks:
#             continue  # Skip if this mask has already been processed
        
#         for _mask, ytype, curves in fm.yields:
#             if _mask != mask:
#                 continue  # Continue if no match
            
#             print('found match for mask', mask)
#             pool_data = c_curves_p.loc[' '.join(dtype_key)]
#             for yname in pools:
#                 points = list(zip(pool_data.index.values, pool_data[yname]))
#                 curve = fm.register_curve(ws3.core.Curve(
#                     yname, 
#                     points=points, 
#                     type='a', 
#                     is_volume=False,
#                     xmax=fm.max_age,
#                     period_length=fm.period_length
#                 ))
#                 curves.append((yname, curve))
#                 dt.add_ycomp('a', yname, curve)
        
        # Add the processed mask to the set
        # processed_masks.add(mask)

def compile_events(self, softwood_volume_yname, hardwood_volume_yname, n_yield_vals):
    
    def leading_species(dt):
        """
        Determine if softwood or hardwood leading species by comparing softwood and hardwood
        volume at peak MAI age.
        """
        svol_curve, hvol_curve = dt.ycomp(softwood_volume_yname), dt.ycomp(hardwood_volume_yname)
        tvol_curve = svol_curve + hvol_curve
        x_cmai = tvol_curve.mai().ytp().lookup(0)
        return 'softwood' if svol_curve[x_cmai] > hvol_curve[x_cmai] else 'hardwood'

    for dtype_key in self.dtypes:
        dt = self.dt(dtype_key)
        dt.leading_species = leading_species(dt)
    
    theme_cols = [theme['__name__'] for theme in self._themes]
    columns = theme_cols.copy()
    columns += ['species',
                'using_age_class',
                'min_softwood_age',
                'max_softwood_age',
                'min_hardwood_age',
                'max_hardwood_age',
                'MinYearsSinceDist',
                'MaxYearsSinceDist',
                'LastDistTypeID',
                'MinTotBiomassC',
                'MaxTotBiomassC',
                'MinSWMerchBiomassC',
                'MaxSWMerchBiomassC',
                'MinHWMerchBiomassC',
                'MaxHWMerchBiomassC',
                'MinTotalStemSnagC',
                'MaxTotalStemSnagC',	
                'MinSWStemSnagC',
                'MaxSWStemSnagC',
                'MinHWStemSnagC',
                'MaxHWStemSnagC',
                'MinTotalStemSnagMerchC',
                'MaxTotalStemSnagMerchC',
                'MinSWMerchStemSnagC',
                'MaxSWMerchStemSnagC',
                'MinHWMerchStemSnagC',
                'MaxHWMerchStemSnagC',
                'efficiency',
                'sort_type',
                'target_type',
                'target',
                'disturbance_type',
                'disturbance_year']
    data = {c:[] for c in columns}
    for dtype_key, age, area, acode, period, _ in self.compile_schedule():
        #set_trace()
        for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
        data['species'].append(self.dt(dtype_key).leading_species)
        data['using_age_class'].append('FALSE')
        #############################################################################
        # might need to be more flexible with age range, to avoid OBO bugs and such?
        data['min_softwood_age'].append(-1)
        data['max_softwood_age'].append(-1)
        data['min_hardwood_age'].append(-1)
        data['max_hardwood_age'].append(-1)
        # data['min_softwood_age'].append(age)
        # data['max_softwood_age'].append(age)
        # data['min_hardwood_age'].append(age)
        # data['max_hardwood_age'].append(age)
        #############################################################################
        for c in columns[11:-6]: data[c].append(-1)
        data['efficiency'].append(1)
        data['sort_type'].append(3) # oldest first (see Table 3-3 in the CBM-CFS3 user guide)
        data['target_type'].append('A') # area target
        data['target'].append(area)
        data['disturbance_type'].append(acode)
        data['disturbance_year'].append((period-1)*self.period_length+1)
        # if period == 1:
        #     data['disturbance_year'].append(1)
        # else:
        #     data['disturbance_year'].append((period-1)*self.period_length)
    sit_events = pd.DataFrame(data)         
    return sit_events