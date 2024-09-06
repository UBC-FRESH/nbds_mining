##################################################################################
# This module contain local utility function defintions that we can reuse 
# in example notebooks to help reduce clutter.
# #################################################################################

import ws3
import pandas as pd
import matplotlib.pyplot as plt

##########################################################
# Implement a priority queue heuristic harvest scheduler
# #########################################################

def schedule_harvest_areacontrol(fm, period=None, acode='harvest', util=0.85, 
                                 target_masks=None, target_areas=None,
                                 target_scalefactors=None,
                                 mask_area_thresh=0.,
                                 verbose=0):
    if not target_areas:
        if not target_masks: # default to AU-wise THLB 
            au_vals = []
            au_agg = []
            for au in fm.theme_basecodes(2):
                mask = '? 1 %s ? ?' % au
                masked_area = fm.inventory(0, mask=mask)
                if masked_area > mask_area_thresh:
                    au_vals.append(au)
                else:
                    au_agg.append(au)
                    if verbose > 0:
                        print('adding to au_agg', mask, masked_area)
            if au_agg:
                fm._themes[2]['areacontrol_au_agg'] = au_agg 
                if fm.inventory(0, mask='? ? areacontrol_au_agg ? ?') > mask_area_thresh:
                    au_vals.append('areacontrol_au_agg')
            target_masks = ['? 1 %s ? ?' % au for au in au_vals]
        target_areas = []
        for i, mask in enumerate(target_masks): # compute area-weighted mean CMAI age for each masked DT set
            masked_area = fm.inventory(0, mask=mask, verbose=verbose)
            if not masked_area: continue
            r = sum((fm.dtypes[dtk].ycomp('totvol').mai().ytp().lookup(0) * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))
            r /= masked_area
            asf = 1. if not target_scalefactors else target_scalefactors[i]  
            ta = (1/r) * fm.period_length * masked_area * asf
            target_areas.append(ta)
    periods = fm.periods if not period else [period]
    for period in periods:
        for mask, target_area in zip(target_masks, target_areas):
            if verbose > 0:
                print('calling areaselector', period, acode, target_area, mask)
            fm.areaselector.operate(period, acode, target_area, mask=mask, verbose=verbose)
    sch = fm.compile_schedule()
    return sch



##############################################################
# Implement an LP optimization harvest scheduler
# #############################################################

def cmp_c_z(fm, path, expr):
    """
    Compile objective function coefficient for product indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if fm.is_harvest(d['acode']):
            result += fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result

def cmp_c_i(fm, path, yname, mask=None): # product, named actions
    """
    Compile objective function coefficient for inventory indicator (given ForestModel instance, 
    leaf-to-root-node path, expression to evaluate, and optional mask).
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        result += fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']])
        #result[t] = fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']])
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
                 cgen_ha={}, cgen_hv={}, 
                 cgen_gs={}, cgen_cp = {}, cgen_cf = {},
                 tvy_name='totvol', cp_name='ecosystem', cf_name='all_fluxes', obj_mode='max_iv', mask=None):
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
    elif obj_mode == 'min_hv': # maximize harvest volume
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = vexpr
    elif obj_mode == 'max_iv': # minimize forest inventory values
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = '1.'
    elif obj_mode == 'min_iv': # minimize forest inventory values
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = '1.'
    elif obj_mode == 'min_ha': # minimize harvest area
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = '1.'
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode)        
    coeff_funcs['z'] = partial(cmp_c_i, yname=cp_name) # define objective function coefficient function for inventory data    
    # coeff_funcs['z'] = partial(cmp_c_z, expr=vexpr) # define objective function coefficient function for havrest volume 
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
    if cgen_cp: # define general constraint (carbon pools)
        cname = 'cgen_cp'
        coeff_funcs[cname] = partial(cmp_c_ci, yname=cp_name, mask=None)
        cgen_data[cname] = cgen_cp
    if cgen_cf: # define general constraint (carbon fluxes)
        cname = 'cgen_cf'
        coeff_funcs[cname] = partial(cmp_c_ci, yname=cf_name, mask=None)
        cgen_data[cname] = cgen_cf
    return fm.add_problem(name, coeff_funcs, cflw_e, cgen_data=cgen_data, acodes=acodes, sense=sense, mask=mask)


def compile_scenario(fm):
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


def plot_scenario(df):
    fig, ax = plt.subplots(1, 5, figsize=(12, 4))
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area (ha)')
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume (m3)')
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_title('Growing Stock (m3)')
    ax[3].bar(df.period, df.ocp)
    ax[3].set_ylim(0, None)
    ax[3].set_title('Ecosystem C stock (tons)')
    ax[4].bar(df.period, df.ocf)
    ax[4].set_ylim(0, None)
    ax[4].set_title('Total Carbon Emission (tons)')
    return fig, ax

def run_scenario(fm, scenario_name='base'):
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}
    cgen_cp = {}
    cgen_cf = {}
    
    # define harvest area and harvest volume flow constraints
    cflw_ha = ({p:0.05 for p in fm.periods}, 1)
    cflw_hv = ({p:0.05 for p in fm.periods}, 1)
    
    in_gs = 750290200. #initial growing stock volume
    in_cp = 1319073591.63 #initial total ecosystem carbon stock
    in_cf = 20034534.75 #intial total ecosystem carbon emission
    AAC = 7031963. # AAC of TSA24

    if scenario_name == 'base': 
        # Base scenario
        print('running base scenario')
        cgen_gs = {'lb':{1:0}, 'ub':{1:9999999999}} 
        cgen_hv = {'lb':{1:0}, 'ub':{1:9999999999}}    
    elif scenario_name == 'base_h': 
        # Base scenario
        print('running base scenario')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC-1}, 'ub':{1:AAC}}
    elif scenario_name == 'base_c': 
        # Cabron indicators constraints
        print('running base scenario with carbon constraints')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC-1}, 'ub':{1:AAC}}   
        cgen_cp = {'lb':{10:in_cp*0.9}, 'ub':{10:in_cp*0.9+1}} #Not less than 90% of initial total ecosystem carbon stock
        cgen_cf = {'lb':{10:in_cf}, 'ub':{10:in_cf*1.1}} #Not more than 110% of initial total ecosystem carbon stock
    elif scenario_name == 'reduce_10%_AAC': 
        # Reduce 10% of harvest volume from base scenario
        print('running base scenario reduced 10% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}}#Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*0.9-1}, 'ub':{1:AAC*0.9}}    
    elif scenario_name == 'reduce_20%_AAC': 
        # Reduce 20% of harvest volume from base scenario
        print('running base scenario reduced 20% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*0.8-1}, 'ub':{1:AAC*0.8}}   
    elif scenario_name == 'increase_10%_AAC': 
        # Increase 10% of harvest volume from base scenario
        print('running base scenario increased 10% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*1.1-1}, 'ub':{1:AAC*1.1}}
    elif scenario_name == 'increase_20%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 20% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*1.2-1}, 'ub':{1:AAC*1.2}}
    elif scenario_name == 'increase_50%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 50% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*1.5-1}, 'ub':{1:AAC*1.5}}
    elif scenario_name == 'increase_100%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 100% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*1.2-1}, 'ub':{1:AAC*1.2}}
    elif scenario_name == 'increase_500%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 500% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*2-1}, 'ub':{1:AAC*2}}
    elif scenario_name == 'increase_1000%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 1000% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*10-1}, 'ub':{1:AAC*10}}
    elif scenario_name == 'increase_5000%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 5000% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*50-1}, 'ub':{1:AAC*50}}
    elif scenario_name == 'increase_10000%_AAC': 
        # Increase 20% of harvest volume from base scenario
        print('running base scenario increased 10000% of AAC')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*0.9+1}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*100-1}, 'ub':{1:AAC*100}}    
    else:
        assert False # bad scenario name
      
    # p = gen_scenario(fm=fm, 
    #                  name=scenario_name, 
    #                  cflw_ha=cflw_ha, 
    #                  cflw_hv=cflw_hv,
    #                  cgen_ha=cgen_ha,
    #                  cgen_hv=cgen_hv,
    #                  cgen_gs=cgen_gs)
    
    p = gen_scenario(fm=fm, 
                     name=scenario_name, 
                     cflw_ha=cflw_ha, 
                     cflw_hv=cflw_hv,
                     cgen_ha=cgen_ha,
                     cgen_hv=cgen_hv,
                     cgen_gs=cgen_gs,
                     cgen_cp=cgen_cp,
                     cgen_cf=cgen_cf)

    fm.reset()
    m = p.solve()

    import gurobipy as grb
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
    
    from util import compile_scenario, plot_scenario
    df = compile_scenario(fm)
    fig, ax = plot_scenario(df)
    # cbm_results = cbm_hardlink(fm,disturbance_type_mapping)
    
    return fig, df, p


# def run_scenario(fm, scenario_name='base'):
#     cflw_ha = {}
#     cflw_hv = {}
#     cgen_ha = {}
#     cgen_hv = {}
#     cgen_gs = {}
    
#     # define harvest area and harvest volume flow constraints
#     cflw_ha = ({p:0.05 for p in fm.periods}, 1)
#     cflw_hv = ({p:0.05 for p in fm.periods}, 1)

#     if scenario_name == 'base': 
#         # Base scenario
#         print('running bsae scenario')
#     elif scenario_name == 'base-cgen_ha': 
#         # Base scenario, plus harvest area general constraints
#         print('running base scenario plus harvest area constraints')
#         cgen_ha = {'lb':{1:100.}, 'ub':{1:101.}}    
#     elif scenario_name == 'base-cgen_hv': 
#         # Base scenario, plus harvest volume general constraints
#         print('running base scenario plus harvest volume constraints')
#         cgen_hv = {'lb':{1:1000.}, 'ub':{1:1001.}}    
#     elif scenario_name == 'base-cgen_gs': 
#         # Base scenario, plus growing stock general constraints
#         print('running base scenario plus growing stock constraints')
#         cgen_gs = {'lb':{10:100000.}, 'ub':{10:100001.}}
#     else:
#         assert False # bad scenario name

#     p = gen_scenario(fm=fm, 
#                      name=scenario_name, 
#                      cflw_ha=cflw_ha, 
#                      cflw_hv=cflw_hv,
#                      cgen_ha=cgen_ha,
#                      cgen_hv=cgen_hv,
#                      cgen_gs=cgen_gs)

#     fm.reset()
#     m = p.solve()

#     if m.status != grb.GRB.OPTIMAL:
#         print('Model not optimal.')
#         sys.exit()
#     sch = fm.compile_schedule(p)
#     fm.apply_schedule(sch, 
#                       force_integral_area=False, 
#                       override_operability=False,
#                       fuzzy_age=False,
#                       recourse_enabled=False,
#                       verbose=False,
#                       compile_c_ycomps=True)
#     df = compile_scenario(fm)
#     fig, ax = plot_scenario(df)
#     return fig, df, p


##############################################################
# Implement simple functions to run CBM from ws3 export data and output resutls
# #############################################################
def run_cbm(sit_config, sit_tables, n_steps):
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

def cbm_hardlink(fm, disturbance_type_mapping):
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[2] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                           hardwood_volume_yname='hwdvol', 
                                           admin_boundary='British Columbia', 
                                           eco_boundary='Montane Cordillera',
                                           disturbance_type_mapping=disturbance_type_mapping)
    n_steps = fm.horizon * fm.period_length
    cbm_output = run_cbm(sit_config, sit_tables, n_steps)
    return cbm_output

def cbm_report(fm, cbm_output, biomass_pools, dom_pools, fluxes):
    # Add carbon pools indicators 
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    # biomass_result = pi[['timestep']+biomass_pools]
    # dom_result = pi[['timestep']+dom_pools]
    # total_eco_result = pi[['timestep']+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({'Year':pi['timestep'],
                                         'Biomass':pi[biomass_pools].sum(axis=1),
                                         'DOM':pi[dom_pools].sum(axis=1),
                                         'Total_Ecosystem': pi[biomass_pools+dom_pools].sum(axis=1),
                                         'Product':pi['product'].sum(axis=1)})
    # print(pi[biomass_pools+dom_pools].sum(axis=1).diff())
    # print(pi[['Products']].sum(axis=1).diff())
    # print(pi[biomass_pools+dom_pools].sum(axis=1).diff()-pi[['Products']].sum(axis=1).diff())

    annual_net_fluxes = annual_carbon_stocks[['Year','Total_Ecosystem']].copy()
    annual_net_fluxes['Net_Fluxes'] = annual_net_fluxes['Total_Ecosystem'].diff()
    annual_net_fluxes = annual_net_fluxes[['Year','Net_Fluxes']]
    annual_net_fluxes.loc[annual_net_fluxes['Year'] == 0, 'Net_Fluxes'] = 0
 
    # annual_CO2_emissions = pd.DataFrame({'Year':fi['timestep'],
    #                                      'Emissions':pi['CO2'].sum(axis=1)})
    
    #annual_carbon_stockchanges = annual_carbon_stocks.diff()
    
    # Add carbon fluxes indicators      
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    annual_all_emissions = pd.DataFrame({'Year':fi['timestep'],
                                         'All_Emissions':fi[fluxes].sum(axis=1)})
     
    
    n_steps = fm.horizon * fm.period_length
    annual_carbon_stocks.groupby('Year').sum().plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual Ecosystem Carbon Stocks")
    annual_all_emissions.groupby('Year').sum().plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual Ecosystem Carbon Emissions")
    annual_net_fluxes.groupby('Year').sum().plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual Ecosystem Net Fluxes")
    annual_carbon_stockchanges.plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual EcosystemCarbon Stock Changes")

##############################################################
# Implement simple functions to generate, plug-in, and fix carbon yield curves into ws3 models
# #############################################################

def generate_c_curves(fm, disturbance_type_mapping, pools, fluxes):
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[2] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                           hardwood_volume_yname='hwdvol', 
                                           admin_boundary='British Columbia', 
                                           eco_boundary='Montane Cordillera',
                                           disturbance_type_mapping=disturbance_type_mapping)
    
    df = sit_tables['sit_inventory']
    df = df.iloc[0:0]
    data = []
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        values = list(dtype_key) 
        values += [dt.leading_species, 'FALSE', 0, 1., 0, 0, 'fire', 'fire' if dtype_key[2] == dtype_key[4] else 'harvest']
        data.append(dict(zip(df.columns, values)))
    sit_tables['sit_inventory'] = pd.DataFrame(data)
    
    n_steps = fm.horizon * fm.period_length
    cbm_output = run_cbm(sit_config, sit_tables, n_steps)
    
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    pi['dtype_key'] = pi.apply(lambda r: '%s %s %s %s %s' % (r['theme0'], r['theme1'], r['theme2'], r['theme3'], r['theme4']), axis=1)
    fi['dtype_key'] = fi.apply(lambda r: '%s %s %s %s %s' % (r['theme0'], r['theme1'], r['theme2'], r['theme3'], r['theme4']), axis=1)

    c_curves_p = pi.groupby(['dtype_key', 'timestep'], as_index=True)[pools].sum()
    c_curves_f = fi.groupby(['dtype_key', 'timestep'], as_index=True)[fluxes].sum()
    
    #c_curves_p[pools] = c_curves_p[pools].apply(lambda x: x*(1+pool_corr))
    #c_curves_f[fluxes] = c_curves_f[fluxes].apply(lambda x: x*flux_corr)
    
    return c_curves_p, c_curves_f
    
def plugin_c_curves(fm, c_curves_p, c_curves_f, pools, fluxes):
    # for dtype_key in dt_tuples:
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        mask = ('?', '?', dtype_key[2], '?', dtype_key[4])
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
            flux_data = c_curves_f.loc[' '.join(dtype_key)]
            for yname in fluxes:
                points = list(zip(flux_data.index.values, flux_data[yname]))
                curve = fm.register_curve(ws3.core.Curve(yname, 
                                                         points=points, 
                                                         type='a', 
                                                         is_volume=False,
                                                         xmax=fm.max_age,
                                                         period_length=fm.period_length))
                curves.append((yname, curve))
                dt.add_ycomp('a', yname, curve)
        #mask = '? ? %s ? %' % (dtype_key[2], dtype_key[4])
        #points = c_curves_p

def correct_c_curves(fm, c_curves_p, c_curves_f, cbm_output, pools, fluxes, cbm_x_shift=False):
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    if cbm_x_shift:
        df_cbm = pd.DataFrame({'period':pi['timestep'] * 0.1, 
                               'pool':pi[pools].sum(axis=1),
                               'flux':fi[fluxes].sum(axis=1)}).groupby('period').sum().iloc[1::10, :].reset_index()
        df_cbm['period'] = (df_cbm['period'] - 0.1 + 1.0).astype(int)
    else:
        df_cbm = pd.DataFrame({'period':pi['timestep'] * 0.1, 
                               'pool':pi[pools].sum(axis=1),
                               'flux':fi[fluxes].sum(axis=1)}).groupby('period').sum().iloc[10::10, :].reset_index()
        df_cbm['period'] = (df_cbm['period']).astype(int)

    df_ws3 = pd.DataFrame({'period':fm.periods,
                           'pool':[sum(fm.inventory(period, pool) for pool in pools) for period in fm.periods],
                           'flux':[sum(fm.inventory(period, flux) for flux in fluxes) for period in fm.periods],
                           'ha':[fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]})
    
    pool_corr = (df_cbm['pool'] / df_ws3['pool']).mean()
    flux_corr = ((df_cbm['flux'] - df_ws3['flux']) / df_ws3['ha']).mean()

    c_curves_p[pools] = c_curves_p[pools].apply(lambda x: x*pool_corr)
    c_curves_f[fluxes] = c_curves_f[fluxes].apply(lambda x: x+flux_corr*df_ws3['ha'].mean())
    
    return c_curves_p, c_curves_f

def draw_c_curves(fm, c_curves_p, c_curves_f, pools, fluxes, show):
    if show == "pool":
        for pool in ecosystem_pools: 
            x, y = fm.periods, [fm.inventory(p, pool) for p in fm.periods]
            plt.plot(x, y, label=pool)
            plt.legend(bbox_to_anchor=(1, 1))
    elif show  == "flux":
        for flux in fluxes:
            x, y = fm.periods, [fm.inventory(p, flux) for p in fm.periods]
            plt.plot(x, y, label=flux)
            plt.legend(bbox_to_anchor=(1, 1))
    elif show == "both":
        for pool in pools: 
            x, y = fm.periods, [fm.inventory(p, pool) for p in fm.periods]
            plt.plot(x, y, label=pool)
            plt.legend(bbox_to_anchor=(1, 1))
        for flux in fluxes:
            x, y = fm.periods, [fm.inventory(p, flux) for p in fm.periods]
            plt.plot(x, y, label=flux)
            plt.legend(bbox_to_anchor=(1, 1))
    else:
        print("Please indicate show 'pool', 'flux', or 'both'")
        
def compare_ws3_cbm(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, fluxes, cbm_x_shift=False):
    eco_pools = biomass_pools+dom_pools
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    # df_cbm = pd.DataFrame({'period':pi["timestep"] * 0.1, 
    #                        'eco_pool':pi[eco_pools].sum(axis=1)}).groupby('period').sum().iloc[10::10, :].reset_index()
    # df_cbm['period'] = (df_cbm['period'] - 0.1 + 1.0).astype(int)
    #df_cbm['flux'] = pd.DataFrame(df_cbm['pool'].diff())
    
    #df_cbm['flux'] = df_cbm['pool'].diff()
    
    if cbm_x_shift:
        df_cbm = pd.DataFrame({'period':pi["timestep"] * 0.1, 
                               'biomass_pool':pi[biomass_pools].sum(axis=1),
                               'dom_pool':pi[dom_pools].sum(axis=1),
                               'eco_pool':pi[eco_pools].sum(axis=1),
                               'net_fluxes':pi[biomass_pools+dom_pools].sum(axis=1).diff(),
                               'total_emissions':fi[fluxes].sum(axis=1)}).groupby('period').sum().iloc[1::10, :].reset_index()
        df_cbm['period'] = (df_cbm['period'] - 0.1 + 1.0).astype(int)
    else:
        df_cbm = pd.DataFrame({'period':pi["timestep"] * 0.1, 
                               'biomass_pool':pi[biomass_pools].sum(axis=1),
                               'dom_pool':pi[dom_pools].sum(axis=1),
                               'eco_pool':pi[eco_pools].sum(axis=1),
                               'total_emissions':fi[fluxes].sum(axis=1)}).groupby('period').sum().iloc[10::10, :].reset_index()
        df_cbm['period'] = (df_cbm['period']).astype(int)
        
    df_cbm['net_fluxes']=df_cbm['eco_pool'].diff()
    df_cbm.at[0,'net_fluxes'] = 0.0
        # df_product= pd.DataFrame({'period':pi["timestep"] * 0.1,
        #                           'Products':pi[['Products']].sum(axis=1),
        #                           'Products_fluxes':pi[['Products']].diff().sum(axis=1)}).groupby('period').sum().iloc[10::10, :].reset_index()
        # df_product.at[0, 'product_fluxes'] =0.0
    
        # return df_product
    
        
    # if summary == False: # When there are individual carbon indicators
    df_ws3 = pd.DataFrame({'period':fm.periods,
                           'biomass_pool':[sum(fm.inventory(period, pool) for pool in ['biomass']) for period in fm.periods],
                           'dom_pool':[sum(fm.inventory(period, pool) for pool in ['DOM']) for period in fm.periods],
                           'eco_pool':[sum(fm.inventory(period, pool) for pool in ['ecosystem']) for period in fm.periods],
                           # 'CO2':[sum(fm.inventory(period, pool) for pool in ['CO2']) for period in fm.periods],
                           'net_fluxes':[sum(fm.inventory(period, flux) for flux in ['net_fluxes']) for period in fm.periods],
                           'total_emissions':[sum(fm.inventory(period, flux) for flux in ['total_emissions']) for period in fm.periods]})

    df_ws3['net_fluxes']=df_ws3['eco_pool'].diff()
    df_ws3.at[0,'net_fluxes'] = 0.
    
    # df_ws3['flux'] = pd.DataFrame(df_ws3['pool'].diff())
        
    # else: # When there are not individual carbon indicators
    #     df_ws3 = pd.DataFrame({'period':fm.periods,
    #                            'biomass_pool':[sum(fm.inventory(period, 'biomass')) for period in fm.periods],
    #                            'dom_pool': [sum(fm.inventory(period, 'DOM')) for period in fm.periods],
    #                            'eco_pool': [sum(fm.inventory(period, 'ecosystem')) for period in fm.periods],
    #                            'flux': [sum(fm.inventory(period, 'all_fluxes')) for period in fm.periods]})
    #     #df_ws3['flux'] = pd.DataFrame(df_ws3['pool'].diff())
        
    fix, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(df_cbm['period'], df_cbm['eco_pool'], label='cbm ecosystem pool')
    ax[0].plot(df_ws3['period'], df_ws3['eco_pool'], label='ws3 ecosystem pool')
    ax[0].plot(df_cbm['period'], df_cbm['biomass_pool'], label='cbm biomass pool')
    ax[0].plot(df_ws3['period'], df_ws3['biomass_pool'], label='ws3 biomass pool')
    ax[0].plot(df_cbm['period'], df_cbm['dom_pool'], label='cbm DOM pool')
    ax[0].plot(df_ws3['period'], df_ws3['dom_pool'], label='ws3 DOM pool')
    ax[0].plot(df_cbm['period'], df_cbm['eco_pool'], label='cbm pool')
    ax[0].plot(df_ws3['period'], df_ws3['eco_pool'], label='ws3 pool')
    # ax[0].plot(df_cbm['period'], df_cbm['CO2'], label='cbm CO2 pool')
    # ax[0].plot(df_ws3['period'], df_ws3['CO2'], label='ws3 CO2 pool')
    ax[1].plot(df_cbm['period'], df_cbm['total_emissions'], label='cbm total fluxes')
    ax[1].plot(df_ws3['period'], df_ws3['total_emissions'], label='ws3 total fluxes')
    ax[2].plot(df_cbm['period'], df_cbm['net_fluxes'], label='cbm net flux')
    ax[2].plot(df_ws3['period'], df_ws3['net_fluxes'], label='ws3 net flux')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylim(None, None)
    ax[1].set_ylim(None, None)
    ax[2].set_ylim(None, None)
    return df_ws3, df_cbm

def complie_events(self, softwood_volume_yname, hardwood_volume_yname, n_yield_vals):
    
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
        data['disturbance_year'].append(period*self.period_length)
    sit_events = pd.DataFrame(data)         
    return sit_events
