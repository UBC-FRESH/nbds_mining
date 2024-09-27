##################################################################################
# This module contain local utility function defintions that we can reuse 
# in example notebooks to help reduce clutter.
# #################################################################################

import ws3
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

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
    Compile objective function coefficient (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if fm.is_harvest(d['acode']):
            result += fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
            # print('t')
            # print(result)
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

def cmp_c_i(fm, path, yname, mask=None): # product, named actions
    """
    Compile objective function coefficient for inventory indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result = fm.inventory(t, yname=yname, age=d['_age'], dtype_keys=[d['_dtk']])
        #result[t] = fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']])
    return result

def cmp_c_id(fm, path, yname, mask=None): # product, named actions
    """
    Compile objective function coefficient for inventory indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        if t==1:
            result = 0.
        else:
            result += (fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']])-fm.inventory(t-1, yname, age=d['_age'], dtype_keys=[d['_dtk']]))
        #result[t] = fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']])
    return result


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

    # Correctly merging all dataframes
    merged_df = pd.merge(pd.merge(pd.merge(df_cs, df_ae, left_index=True, right_index=True, how='outer'),
                                  df_gg, left_index=True, right_index=True, how='outer'),
                         df_sc, left_index=True, right_index=True, how='outer')

    merged_df['Stock_Change'] = merged_df['Ecosystem'].diff() * (-1)
    merged_df.at[0, 'Stock_Change'] = 0

    return merged_df

def compare_ws3_cbm(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, plots):
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

#     df_period['Net_Ecosystem_Emission_co2'] = (df_period['All_Emissions'] - df_period['Gross_Growth'])*44/12

    # return annual_carbon_stock, annual_all_emission, annual_stock_change, annual_gross_growth, annual_product_stock
    
    # annual_product_stock.groupby('Year').sum().plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual Product Carbon Stock")
    #annual_carbon_stockchanges.plot(figsize=(5,5),xlim=(0,n_steps),ylim=(None,None),title="Annual EcosystemCarbon Stock Changes")
    # return df_period

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

def plugin_c_curves(fm, c_curves_p, pools):
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
                
# def plugin_c_curves(fm, c_curves_p, c_curves_f, pools, fluxes):
#     # for dtype_key in dt_tuples:
#     for dtype_key in fm.dtypes:
#         dt = fm.dt(dtype_key)
#         mask = ('?', '?', dtype_key[2], '?', dtype_key[4])
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

def cmp_c_ss(fm, path, expr, yname, half_life_solid_wood=100, half_life_paper=2, proportion_solid_wood=1, mask=None):
    """
    Compile objective function coefficient for total system carbon stock indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    
    result = 0.
    
    # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood  # Decay rate for solid wood products (30-year half-life)
    k_paper = math.log(2) / half_life_paper  # Decay rate for paper (2-year half-life)
    
    # Define the allocation distribution
    proportion_paper = 1 - proportion_solid_wood
    
    # wood density (Kennedy, 1965)
    wood_density = 460

    # carbon content
    carbon_content = 0.5
    
    # k_solid_wood = 0 # Decay rate for solid wood products (0-year half-life)
    # k_paper = 0 # Decay rate for paper (0-year half-life)
    
    product_stock_dict = {}  # Dictionary to track product stock for each node across iterations
    
    for t, n in enumerate(path, start=1):

        d = n.data()
        node_id = id(n)  # or another unique identifier specific to your application
        
        # Track the ecosystem carbon stock
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result = fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']])
        
        # Retrieve the last tuple of stocks from the dictionary
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks
        
        if fm.is_harvest(d['acode']):
            # Calculate new product stock
            new_product_carbon = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False) * wood_density * carbon_content / 1000 # Convert kg to ton
            new_product_stock_solid_wood = new_product_carbon * proportion_solid_wood
            new_product_stock_paper = new_product_carbon * proportion_paper 

            # Apply decay to old stocks and add new stocks
            # Apply decay to all stocks within the same period they're produced
            sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
            sum_product_stock_paper = (old_product_stock_paper) * (1 - k_paper)**10 + new_product_stock_paper
        
        else:
            # If not harvesting, simply apply decay to the old product stocks
            sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)
            sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)
            
        # Update product_stock_dict with the new sum product stocks for this node
        product_stock_dict[node_id] = (sum_product_stock_solid_wood, sum_product_stock_paper)

        ecosystem_stock = fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']])
        result = ecosystem_stock + sum_product_stock_solid_wood + sum_product_stock_paper
        
    return result

def cmp_c_se(fm, path, expr, yname, half_life_solid_wood=1000000, half_life_paper=2, proportion_solid_wood=1, displacement_factor=2.2, mask=None):
    """
    Compile objective function coefficient for net system carbon emission indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    
    result = 0.
    
   # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood  # Decay rate for solid wood products (30-year half-life)
    k_paper = math.log(2) / half_life_paper  # Decay rate for paper (2-year half-life)
    
    # Define the allocation distribution
    proportion_paper = 1 - proportion_solid_wood
    
    # wood density (Kennedy, 1965)
    wood_density = 460 #kg/m^3

    # carbon content
    carbon_content = 0.5
    
    product_stock_dict = {}  # Dictionary to track product stock for each node across iterations

    for t, n in enumerate(path, start=1):

        d = n.data()
        node_id = id(n)  # or another unique identifier specific to your application
        
        # Track the ecosystem carbon stock change
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result += (fm.inventory(t-1, yname, age=d['_age'], dtype_keys=[d['_dtk']])-fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']]))*44/12 #Convert C to CO2
        
        # Retrieve the last tuple of stocks from the dictionary
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks
        
        if fm.is_harvest(d['acode']):
            # Calculate new product stock
            new_product_stock = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False) * wood_density * carbon_content / 1000 # Convert kg to ton
            new_product_stock_solid_wood = new_product_stock * proportion_solid_wood
            new_product_stock_paper = new_product_stock * proportion_paper 

            # Apply decay to old stocks and add new stocks
            sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
            sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10 + new_product_stock_paper
            
            sum_product_emission_solid_wood = old_product_stock_solid_wood * (1-(1 - k_solid_wood)**10) * 44 / 12 # Convert C to CO2
            sum_product_emission_paper = old_product_stock_paper * (1-(1 - k_paper)**10) * 44 / 12 # Convert C to CO2

            # Update product_stock_dict with the new sum product stocks for this node
            product_stock_dict[node_id] = (sum_product_stock_solid_wood, sum_product_stock_paper)
            
            sum_product_emission = sum_product_emission_solid_wood + sum_product_emission_paper # Convert C to CO2
            
            substitution_effect = new_product_stock_solid_wood*displacement_factor*44/12*-1 # negative emission aviod by displacing high GHG-intensive materials and products with HWPs 
           
            result -= new_product_stock*44/12 # Aviod double-accounting the HWPs carbon emissions
            result += sum_product_emission
            result += substitution_effect
        
        else:
            # If not harvesting, simply apply decay to the old product stocks
            sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10
            sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10
            
            sum_product_emission_solid_wood = old_product_stock_solid_wood * (1-(1 - k_solid_wood)**10) * 44 / 12 # Convert C to CO2
            sum_product_emission_paper = old_product_stock_paper * (1-(1 - k_paper)**10) * 44 / 12 # Convert C to CO2

            # Update product_stock_dict with the new sum product stocks for this node
            product_stock_dict[node_id] = (sum_product_stock_solid_wood, sum_product_stock_paper)

            sum_product_emission = sum_product_emission_solid_wood + sum_product_emission_paper
            result += sum_product_emission

        ecosystem_stock_change = (fm.inventory(t-1, 'ecosystem') - fm.inventory(t, 'ecosystem')) * 44 / 12 if t > 0 else 0
        result += ecosystem_stock_change

    return result

def track_system_stock(fm, half_life_solid_wood, half_life_paper, proportion_solid_wood):
    
    product_stock_dict = {}
    solid_wood_stock_list = []
    paper_stock_list = []
    product_stock_list = []
    ecosystem_stock_list = []
    system_stock_list = []

    # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood
    k_paper = math.log(2) / half_life_paper

    # Define the allocation distribution
    proportion_paper = 1-proportion_solid_wood

    # Constants
    wood_density = 460 #(Kennedy, 1965)
    carbon_content = 0.5

    for period in fm.periods:
        # Get old product stocks
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks

        # Calculate new product stocks
        new_product_stock = fm.compile_product(period, 'totvol * 0.85', acode='harvest')* wood_density * carbon_content / 1000 # Convert kg to ton
        new_product_stock_solid_wood = new_product_stock * proportion_solid_wood
        new_product_stock_paper = new_product_stock * proportion_paper 

        # Apply decay to all emissions within the same period they're produced
        sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
        sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10 + new_product_stock_paper

        # Update product_stock_dict for this period
        product_stock_dict[period] = (sum_product_stock_solid_wood, sum_product_stock_paper)

        # Calculate total system carbon stock
        ecosystem_stock = fm.inventory(period, 'ecosystem')
        sum_product_stock = sum_product_stock_solid_wood + sum_product_stock_paper
        total_system_stock = ecosystem_stock + sum_product_stock

        # Update stock lists for this period
        solid_wood_stock_list.append(sum_product_stock_solid_wood)
        paper_stock_list.append(sum_product_stock_paper)
        product_stock_list.append(sum_product_stock)
        ecosystem_stock_list.append(ecosystem_stock)
        system_stock_list.append(total_system_stock)

    # Prepare data for plotting
    data = {
        'period': fm.periods,
        'solid_wood': solid_wood_stock_list,
        'paper': paper_stock_list,
        'sum_product': product_stock_list,
        'ecosystem': ecosystem_stock_list,
        'system': system_stock_list
    }

    df = pd.DataFrame(data)

    # df.to_excel('results/no_harvest_stock.xlsx', index=False)
    df.to_excel('results/stock.xlsx', index=False)

    # Plotting
    fig, ax = plt.subplots(1, 5, figsize=(16, 4))  # Adjusted for 5 subplots
    ax[0].bar(df.period, df.solid_wood)
    ax[0].set_title('Solid Wood Product C Stock')
    ax[1].bar(df.period, df.paper)
    ax[1].set_title('Paper Product C Stock')
    ax[2].bar(df.period, df.sum_product)
    ax[2].set_title('Total Product C Stock')
    ax[3].bar(df.period, df.ecosystem)
    ax[3].set_title('Ecosystem C Stock')
    ax[4].bar(df.period, df.system)
    ax[4].set_title('Total System C Stock')

    for a in ax:
        a.set_ylim(None, None)
        a.set_xlabel('Period')
        a.set_ylabel('Stock (tons)')

    plt.tight_layout()
    return fig, ax, df, product_stock_dict

def track_system_emission(fm, half_life_solid_wood, half_life_paper, proportion_solid_wood, displacement_factor):
    
    product_stock_dict = {}
    solid_wood_emission_list = []
    paper_emission_list = []
    product_emission_list = []
    ecosystem_emission_list = []
    system_emission_list = []

    # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood
    k_paper = math.log(2) / half_life_paper

    # Define the allocation distribution
    proportion_paper = 1-proportion_solid_wood

    # Constants
    wood_density = 460 #(Kennedy, 1965)
    carbon_content = 0.5
    
    for period in fm.periods:
        # Get old product emissions
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks

        # Calculate new product emissions
        new_product_stock = fm.compile_product(period, 'totvol * 0.85', acode='harvest') * wood_density * carbon_content / 1000 # convert the unit from kg to ton
        new_product_stock_solid_wood = new_product_stock * proportion_solid_wood
        new_product_stock_paper = new_product_stock * proportion_paper

        # Apply decay to all emissions within the same period they're produced
        sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
        sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10 + new_product_stock_paper

        sum_product_emission_solid_wood = old_product_stock_solid_wood * (1-(1 - k_solid_wood)**10) * 44 / 12 # Convert C to CO2
        sum_product_emission_paper = old_product_stock_paper * (1-(1 - k_paper)**10) * 44 / 12 # Convert C to CO2
        
        # Update product_emission_dict for this period
        product_stock_dict[period] = (sum_product_stock_solid_wood, sum_product_stock_paper)

        # Calculate total system carbon emission
        sum_product_emission = sum_product_emission_solid_wood + sum_product_emission_paper
        ecosystem_emission = (fm.inventory(period-1, 'ecosystem') - fm.inventory(period, 'ecosystem') - new_product_stock) * 44 / 12 if period > 0 else 0
        substitution_effect = new_product_stock_solid_wood*displacement_factor*44/12*-1 # negative emission aviod by displacing high GHG-intensive materials and products with HWPs 
        total_system_emission = ecosystem_emission + sum_product_emission + substitution_effect
        
        # Update stock lists for this period
        solid_wood_emission_list.append(sum_product_emission_solid_wood)
        paper_emission_list.append(sum_product_emission_paper)
        product_emission_list.append(sum_product_emission)
        ecosystem_emission_list.append(ecosystem_emission)
        system_emission_list.append(total_system_emission)

    # Prepare data for plotting
    data = {
        'period': fm.periods,
        'solid_wood': solid_wood_emission_list,
        'paper': paper_emission_list,
        'sum_product': product_emission_list,
        'ecosystem': ecosystem_emission_list,
        'system': system_emission_list
    }

    df = pd.DataFrame(data)
    # df.to_excel('results/no_harvest_emission.xlsx', index=False)
    df.to_excel('results/emission.xlsx', index=False)

    # Plotting
    fig, ax = plt.subplots(1, 5, figsize=(16, 4))  # Adjusted for 5 subplots
    ax[0].bar(df.period, df.solid_wood)
    ax[0].set_title('Solid Wood Product CO2 Emission')
    ax[1].bar(df.period, df.paper)
    ax[1].set_title('Paper Product CO2 Emission')
    ax[2].bar(df.period, df.sum_product)
    ax[2].set_title('Total Product CO2 Emission')
    ax[3].bar(df.period, df.ecosystem)
    ax[3].set_title('Ecosystem CO2 Emission')
    ax[4].bar(df.period, df.system)
    ax[4].set_title('Total System CO2 Emission')

    for a in ax:
        a.set_ylim(None, None)
        a.set_xlabel('Period')
        a.set_ylabel('Emission (tons)')

    plt.tight_layout()
    return fig, ax, df

def gen_scenario(fm, name='base', util=0.85, harvest_acode='harvest',
                 cflw_ha={}, cflw_hv={}, 
                 cgen_ha={}, cgen_hv={}, 
                 cgen_gs={}, tvy_name='totvol', cp_name='ecosystem', cf_name='total_emissions', obj_mode='max_hv', mask=None):
    
    from functools import partial
    import numpy as np
    coeff_funcs = {}
    cflw_e = {}
    cgen_data = {}
    acodes = ['null', harvest_acode] # define list of action codes
    vexpr = '%s * %0.2f' % (tvy_name, util) # define volume expression

    #Define constants from product carbon estimation

    if obj_mode == 'max_hv': # maximize harvest volume
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = vexpr
    elif obj_mode == 'min_hv': # maximize harvest volume
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = vexpr
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode)
    
    # coeff_funcs['z'] = partial(cmp_c_i, yname=cf_name) # define objective function coefficient function for inventory data
    # coeff_funcs['z'] = partial(cmp_c_id, yname=cf_name) # define objective function coefficient function for inventory change data
    coeff_funcs['z'] = partial(cmp_c_z, expr=vexpr) # define objective function coefficient function for harvest volume
    # coeff_funcs['z'] = partial(cmp_c_ss, expr=vexpr, yname=cp_name) # define objective function coefficient function for total system carbon stock
    # coeff_funcs['z'] = partial(cmp_c_se, expr=vexpr, yname=cp_name) # define objective function coefficient function for net system carbon emission
    
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
    # if cgen_cp: # define general constraint (carbon pools)
    #     cname = 'cgen_cp'
    #     coeff_funcs[cname] = partial(cmp_c_ci, yname=cp_name, mask=None)
    #     cgen_data[cname] = cgen_cp
    # if cgen_cf: # define general constraint (carbon fluxes)
    #     cname = 'cgen_cf'
    #     coeff_funcs[cname] = partial(cmp_c_ci, yname=cf_name, mask=None)
    #     cgen_data[cname] = cgen_cf
    return fm.add_problem(name, coeff_funcs, cflw_e, cgen_data=cgen_data, acodes=acodes, sense=sense, mask=mask)

def run_scenario(fm, scenario_name='base'):
    #Import Module
    import gurobipy as grb
    
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}
    # cgen_cp = {}
    # cgen_cf = {}
    
    # define harvest area and harvest volume even-flow constraints
    # cflw_ha = ({p:0.05 for p in fm.periods}, 1)
    # cflw_hv = ({p:0.05 for p in fm.periods}, 1)
    
    in_gs = 750290200. #initial growing stock volume
    end_gs_lb = 1374892622.6 #Ending growing stock inventory of the TSA 24 without harvesting
    end_gs_ub = 1580212978.5 #Ending growing stock inventory of the TSA 24 after harvesting
    AAC = 6935023. # AAC of TSA24

    if scenario_name == 'single_cut': 
        # Base scenario
        print('running scenario')
        cgen_hv = {'lb':{x:0.0 for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}, 'ub':{x:0.0 for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}} #Achieve the Annual Allowable Cut
        # cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*2}} #Not less than 90% of initial growing stock
        # cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*2}} #Not less than 90% of initial growing stock
        # cgen_hv = {'lb':{x:AAC*0.5 for x in fm.periods}, 'ub':{x:AAC*1000 for x in fm.periods}} #Achieve the Annual Allowable Cu
        # cgen_hv = {'lb':{10:in_gs*10}, 'ub':{10:in_gs*10+1}}
    elif scenario_name == 'base': 
        # Base scenario
        print('running base scenario')
        # cgen_gs = {'lb':{20:end_gs_ub*0.9}, 'ub':{20:end_gs_ub*1.1}} # Non-declining Yield Constraint
        # cgen_gs = {'lb':{x:fm.inventory(x-1, 'totvol') for x in range(16,21)}, 'ub':{x:fm.inventory(x-1, 'totvol')+1 for x in range(0,21)}} # Non-declining Yield Constraint
        # cgen_gs = {'lb':{0:in_gs}, 'ub':{0:in_gs*2}} # Initial Growing Stock Constraint
        # cgen_gs = {'lb':{20:end_gs}, 'ub':{20:end_gs*2}} # Last Period Growing Stock Constraint
        # cgen_gs = {'lb':{x:in_gs for x in range(16,21)}，'ub':{x:in_gs*2 for x in range(0,21)} #Not less than 90% of initial growing stock
        # cgen_gs = {'lb':{x:in_gs*0.9 for x in range(0,21)}, 'ub':{x:in_gs*2 for x in range(0,21)}} #Not less than 90% of initial growing stock
        # cgen_hv = {'lb':{x:AAC*0.9 for x in range(0,21)}, 'ub':{x:AAC for x in range(0,21)}} #Maintain the Annual Allowable Cut
    elif scenario_name == 'base_m': 
        # Base scenario
        print('running maxmizie harvest scenario')
        # cgen_gs = {'lb':{x:in_gs*0.9 for x in range(0,21)}, 'ub':{x:in_gs*100 for x in range(0,21)}} #Not less than 90% of initial growing stock
        # cgen_hv = {'lb':{20:AAC-1}, 'ub':{20:AAC}} #Achieve the Annual Allowable Cut
    elif scenario_name == 'base_c': 
        # Cabron indicators constraints
        print('running base scenario with even-flow constraints')
        cgen_gs = {'lb':{10:in_gs*0.9}, 'ub':{10:in_gs*2}} #Not less than 90% of initial growing stock
        cgen_hv = {'lb':{1:AAC*0.5}, 'ub':{1:AAC*2}}  #Not less than 10% of annual allowable cut
        # cgen_cf = {'lb':{10:in_cf}, 'ub':{10:in_cf*1.1}} #Not more than 110% of initial total ecosystem carbon stock
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
                     cgen_gs=cgen_gs,)

    fm.reset()
    m = p.solve()

    if m.status != grb.GRB.OPTIMAL:
        print('Model not optimal.')
        # sys.exit()
        
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