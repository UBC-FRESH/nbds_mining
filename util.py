##################################################################################
# This module contains local utility function definitions used in the notebooks.
##################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math


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


def hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates):
    """
    Compile periodic harvested wood products carbon stocks data.
    """
    from util import calculate_co2_value_stock, calculate_initial_co2_value_stock
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
                co2_values_stock.append(calculate_initial_co2_value_stock(fm, i, product_coefficient, product_percentage))
            else:
                co2_values_stock.append(calculate_co2_value_stock(fm, i, product_coefficient, decay_rate, product_percentage))
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
    return fm.compile_product(i, f'totvol * {product_coefficient} * {product_percentage}') * 0.1 * 460 * 0.5 * 44 / 12 * decay_rate / fm.period_length


# Emission (by year)
def hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates):
    from util import calculate_co2_value_emission, calculate_initial_co2_value_emission
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
                co2_values_emission.append(calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
            else:
                co2_values_emission.append(calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
        co2_value_emission = sum(co2_values_emission) / 1000
        data_carbon_emission['period'].append(period_value)
        data_carbon_emission['co2_emission'].append(co2_value_emission)    
    df_carbon_emission = pd.DataFrame(data_carbon_emission)
    return df_carbon_emission


################################################
# Displacement effect
################################################
# Displacement of concrete manufacturing
def calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate):            
    period = math.ceil(i / fm.period_length)
    return fm.compile_product(period,'totvol') * product_coefficients['plumber'] * clt_percentage * credibility / clt_conversion_rate 


# Iterate through the rows of the DataFrame
def emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor):
    from util import  calculate_concrete_volume
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
def emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor):
    from util import  calculate_concrete_volume
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


def run_scenario(fm, scenario_name='base'):
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}
    # define harvest area and harvest volume flow constraints
    cflw_ha = ({p:0.05 for p in fm.periods}, 1)
    cflw_hv = ({p:0.05 for p in fm.periods}, 1)
    if scenario_name == 'base': 
        # Base scenario
        print('running bsae scenario')
    elif scenario_name == 'base-cgen_ha': 
        # Base scenario, plus harvest area general constraints
        print('running base scenario plus harvest area constraints')
        cgen_ha = {'lb':{1:100.}, 'ub':{1:101.}}    
    elif scenario_name == 'base-cgen_hv': 
        # Base scenario, plus harvest volume general constraints
        print('running base scenario plus harvest volume constraints')
        cgen_hv = {'lb':{1:1000.}, 'ub':{1:1001.}}    
    elif scenario_name == 'base-cgen_gs': 
        # Base scenario, plus growing stock general constraints
        print('running base scenario plus growing stock constraints')
        cgen_gs = {'lb':{10:100000.}, 'ub':{10:100001.}}
    else:
        assert False # bad scenario name
    p = gen_scenario(fm=fm, 
                     name=scenario_name, 
                     cflw_ha=cflw_ha, 
                     cflw_hv=cflw_hv,
                     cgen_ha=cgen_ha,
                     cgen_hv=cgen_hv,
                     cgen_gs=cgen_gs)
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
    df = compile_scenario(fm)
    fig, ax = plot_scenario(df)
    return fig, df, p


##############################################################
# Implement a simple function to run CBM from ws3 export data
##############################################################

def run_cbm(df_carbon_stock, df_carbon_emission, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, plot=True):
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
    annual_carbon_stocks['HPW'] = df_carbon_stock['co2_stock']        
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
    
    df_emission_concrete_manu = -1 * df_emission_concrete_manu.groupby('period').sum()
    df_emission_concrete_landfill = -1 * df_emission_concrete_landfill.groupby('period').sum()
    annual_net_emission['HWP'] = df_carbon_emission['co2_emission']
    annual_net_emission['Concrete_manufacturing'] = df_emission_concrete_manu['co2_concrete_manu']
    annual_net_emission['Concrete_landfill'] = df_emission_concrete_landfill['co2_concrete_landfill']
    annual_net_emission['Net emission'] += annual_net_emission['HWP']
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_manufacturing']
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_landfill']
    if plot:
        annual_net_emission.groupby('Year').sum().plot(ax=axes[1], xlim = (0, n_steps)).axhline(y=0, color='red', linestyle='--') 
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_title('Carbon emission over years')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Carbon emission')
    return annual_carbon_stocks, annual_net_emission


def stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest):   
    decay_rates = {'plumber':math.log(2.)/35., 'ppaper':math.log(2.)/2.}
    product_coefficients = {'plumber':0.9, 'ppaper':0.1}
    product_percentages = {'plumber':0.5, 'ppaper':1.}
    products = ['plumber', 'ppaper']
    clt_conversion_rate = 1.
    co2_concrete_manu_factor = 298.
    concrete_density = 2.40 #ton/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_base_scenari = schedule_harvest_areacontrol(fm, max_harvest)
    # df = compile_scenario(fm)
    # plot_scenario(df)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates)
    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor)
    disturbance_type_mapping = [{'user_dist_type': 'harvest', 'default_dist_type': 'Clearcut harvesting without salvage'},
                            {'user_dist_type': 'fire', 'default_dist_type': 'Wildfire'}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[2] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                       hardwood_volume_yname='hwdvol', 
                                       admin_boundary='British Columbia', 
                                       eco_boundary='Montane Cordillera',
                                       disturbance_type_mapping=disturbance_type_mapping)
    cbm_output_1, cbm_output_2 = run_cbm(df_carbon_stock, df_carbon_emission, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, plot = False)
    return cbm_output_1, cbm_output_2     


def plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 10))   
    cbm_output_1.groupby('Year').sum().plot(ax=axes[0, 0], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 0].set_title('Carbon stocks over years (base scenario)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Carbon stocks')   
    cbm_output_2.groupby('Year').sum().plot(ax=axes[1, 0], xlim=(0, n_steps))
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_title('Carbon emission over years (base scenario)')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon emission')
    cbm_output_3.groupby('Year').sum().plot(ax=axes[0, 1], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 1].set_title('Carbon stocks over years (alternative scenario: no harvesting)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Carbon stocks')    
    cbm_output_4.groupby('Year').sum().plot(ax=axes[1, 1], xlim=(0, n_steps))
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Carbon emission over years (alternative scenario: no harvesting)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Carbon emission')    
    plt.tight_layout()
    plt.show()


def scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps):
    cbm_output_2.reset_index(drop=False, inplace=True)
    dif_scenario = pd.DataFrame({"Year": cbm_output_2["Year"],
                       "Net emission": cbm_output_4['Net emission'] - cbm_output_2['Net emission']})
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
    return ax


def results_scenarios(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest):
    from util import stock_emission_scenario, plot_scenarios, scenario_dif
    cbm_output_1, cbm_output_2 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest)
    fm.reset()
    cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, 0)
    plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps)
    dif_plot =scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps)


