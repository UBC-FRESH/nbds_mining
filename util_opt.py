##################################################################################
# This module contains local utility function definitions used in the notebooks.
##################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import ws3.opt
import pickle



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


# def results_scenarios(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode, output_csv_path, output_pdf_path):
#     from util_opt import stock_emission_scenario, plot_scenarios, scenario_dif, stock_emission_scenario_equivalent
#     if not os.path.exists(output_csv_path):
#         os.makedirs(output_csv_path)
#     cbm_output_1, cbm_output_2 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode) #alternative optimization
#     # cbm_output_1, cbm_output_2 = stock_emission_scenario_equivalent(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode) # alternativr equivalent

#     print(cbm_output_2)
#     cbm_output_2_df = pd.DataFrame(cbm_output_2)  
#     cbm_output_2_file = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_2.csv')
#     cbm_output_2_df.to_csv(cbm_output_2_file, index=False)
#     fm.reset()
#     if case_study == 'redchrs':
#         scenario_name = 'bau_redchrs'
#         cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode) #base scenario
#     elif case_study == 'eqtslvr':
#         scenario_name = 'bau_eqtslvr'
#         cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode) #base scenario
#         cbm_output_4_df = pd.DataFrame(cbm_output_4)  
#         cbm_output_4_df.to_csv(f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.csv', index=False)
#     elif case_study == 'gldbr':
#         scenario_name = 'bau_gldbr'
#         cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, obj_mode) #base scenario
#     print(cbm_output_4)
#     cbm_output_4_df = pd.DataFrame(cbm_output_4)  
#     cbm_output_4_file = os.path.join(output_csv_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.csv')
#     cbm_output_4_df.to_csv(cbm_output_4_file, index=False)

#     plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps, case_study, obj_mode, scenario_name, output_pdf_path)
#     dif_plot = scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps, case_study, obj_mode, scenario_name, output_pdf_path)



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





