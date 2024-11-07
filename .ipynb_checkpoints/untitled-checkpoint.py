# def kpi_age(fm, case_study, obj_mode, scenario_name, base_path='.'): 
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os
#     import pandas as pd
    
#     canfi_map_inverse = {'1211': 'AC', 
#                          '1201': 'AT', 
#                          '304': 'BL', 
#                          '1303': 'EP', 
#                          '500': 'FDI', 
#                          '402': 'HW',
#                          '403': 'HM',
#                          '204': 'PL', 
#                          '204': 'PLI', 
#                          '101': 'SB', 
#                          '104': 'SE', 
#                          '105': 'SW', 
#                          '100': 'SX',
#                          '100': 'S',
#                          '1201': 'AT+SX',
#                          '100': 'SX+AT'}
    
#     Aspen = ['AC', 'ACT', 'AT', 'EP', 'VB', 'MB', 'AT+SX']
#     Bal = ['B', 'BA', 'BG', 'BL']
#     Cedar = ['CW', 'YC']
#     Alder = ['D', 'DR']
#     DougFir = ['F', 'FD', 'FDC', 'FDI']
#     Hem = ['H', 'HM', 'HW']
#     Pine = ['PA', 'PL', 'PLC', 'PW', 'PLI', 'PY']
#     Spruce = ['S', 'SS', 'SW', 'SX', 'SE', 'SXW', 'SB', 'SX+AT']
    
#     def find_corresponding_species(number):
#         values = canfi_map_inverse.get(str(number))
#         if not values:
#             return "No corresponding value found."
        
#         values = values.split('+')
#         for value in values:
#             if value in Aspen:
#                 return 'Aspen'
#             elif value in Bal:
#                 return 'Bal'
#             elif value in Cedar:
#                 return 'Cedar'
#             elif value in Alder:
#                 return 'Alder'
#             elif value in DougFir:
#                 return 'DougFir'
#             elif value in Hem:
#                 return 'Hem'
#             elif value in Pine:
#                 return 'Pine'
#             elif value in Spruce:
#                 return 'Spruce'
        
#         return "No matching set found."
    
#     # Define old growth threshold in years
#     # old_growth_threshold = 100
    
#     # Store old growth data
#     old_growth_data = {0: {}, 10: {}}  # For time periods 0 and 10
    
#     bin_edges = np.arange(0, 480, 20)
#     colors = {
#         'Aspen': '#FF0000',
#         'Bal': '#FF8C00',
#         'Cedar': '#FFD700',
#         'Alder': '#00FF00',
#         'DougFir': '#00FFFF',
#         'Hem': '#1E90FF',
#         'Pine': '#9400D3',
#         'Spruce': '#FF00FF'
#     }  
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     for idx, time_period in enumerate([0, 10]):
#         cumulative_hist = np.zeros(len(bin_edges) - 1)
        
#         for i, theme3 in enumerate(fm.theme_basecodes(3)):
#             data = fm.age_class_distribution(time_period, mask=f'? ? ? {theme3} ? ?')
#             x_values = list(data.keys())
#             y_values = list(data.values())
            
#             hist, _ = np.histogram(x_values, bins=bin_edges, weights=y_values)
            
#             species = find_corresponding_species(theme3)
#             axes[idx].bar(bin_edges[:-1], hist, width=20, bottom=cumulative_hist, color=colors[species], edgecolor='black', alpha=0.7, label=f'Species {species}')
            
#             # Calculate old growth area for this species
#             # old_growth_area = sum(y for x, y in data.items() if x >= old_growth_threshold)
#             # old_growth_data[time_period][species] = old_growth_data[time_period].get(species, 0) + old_growth_area
#             old_growth_area = fm.inventory(time_period, 'ogi', mask=f'? ? ? {theme3} ? ?')
#             old_growth_data[time_period][species] = old_growth_data[time_period].get(species, 0) + old_growth_area
            
#             cumulative_hist += hist
        
#         axes[idx].set_xlabel('Age')
#         axes[idx].set_ylabel('Area (ha)')
#         axes[idx].set_title(f'Age Distribution at time period {time_period}')
#         axes[idx].legend()  
    
#     plt.tight_layout()
    
#     # Save the plot
#     folder_path = os.path.join('./outputs/fig', case_study)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)   
#     file_name = f"{case_study}_{obj_mode}_{scenario_name}_age_distribution.pdf"
#     file_path = os.path.join(folder_path, file_name)  
#     plt.savefig(file_path)
#     plt.show()
#     plt.close()   
#     print(f"Plot saved to {file_path}")
    
#  # Convert old growth data to a DataFrame for better display
#     old_growth_df = pd.DataFrame(old_growth_data).fillna(0)
#     old_growth_df['Difference'] = old_growth_df[10] - old_growth_df[0]
    
#     # Print old growth data as a table
#     print(f"\nOld Growth Data (in hectares). \nNegative value indicates loss of old growth and positive value indicates gain of old growth.")
#     print(old_growth_df)
    
#     # Print conclusion about diversity change based on difference
#     if old_growth_df['Difference'].sum() < 0:
#         print(f"\nOverall diversity has **decreased** by {old_growth_df['Difference'].sum():.2f} hectares from time period 0 to time period 10.")
#     else:
#         print(f"\nOverall diversity has **increased** by {old_growth_df['Difference'].sum():.2f} hectares from time period 0 to time period 10.")
    
#     # Plot clustered column chart for old growth areas by species for each time period
#     fig, ax = plt.subplots(figsize=(7, 6))
#     old_growth_df[[0, 10]].plot(kind='bar', color=["#FF8C00", "#9400D3"], ax=ax)
    
#     ax.set_title("Old Growth Area by Species (Period 0 vs Period 10)")
#     ax.set_xlabel("Species")
#     ax.set_ylabel("Old Growth Area (ha)")
#     ax.legend(["Period 0", "Period 10"])
    
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Save the clustered column chart
#     clustered_chart_file = f"{case_study}_{obj_mode}_{scenario_name}_old_growth_comparison.pdf"
#     clustered_chart_path = os.path.join(folder_path, clustered_chart_file)
#     plt.savefig(clustered_chart_path)
#     plt.show()
#     plt.close()
    
#     print(f"Clustered column plot saved to {clustered_chart_path}")
    
#     return old_growth_df







# def kpi_species(fm, case_study, obj_mode, scenario_name, base_path='.'):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os
#     import math

#     # Mapping numbers to species codes
#     canfi_map_inverse = {'1211': 'AC', 
#                          '1201': 'AT', 
#                          '304': 'BL', 
#                          '1303': 'EP', 
#                          '500': 'FDI', 
#                          '402': 'HW',
#                          '403': 'HM',
#                          '204': 'PL', 
#                          '204': 'PLI', 
#                          '101': 'SB', 
#                          '104': 'SE', 
#                          '105': 'SW', 
#                          '100': 'SX',
#                          '100': 'S',
#                          '1201': 'AT+SX',
#                          '100': 'SX+AT'}
       
#     Aspen = ['AC', 'ACT', 'AT', 'EP', 'VB', 'MB', 'AT+SX']
#     Bal = ['B', 'BA', 'BG', 'BL']
#     Cedar = ['CW', 'YC']
#     Alder = ['D', 'DR']
#     DougFir = ['F', 'FD', 'FDC', 'FDI']
#     Hem = ['H', 'HM', 'HW']
#     Pine = ['PA', 'PL', 'PLC', 'PW', 'PLI', 'PY']
#     Spruce = ['S', 'SS', 'SW', 'SX', 'SE', 'SXW', 'SB', 'SX+AT']

#     colors = {
#         'Aspen': '#FF0000',
#         'Bal': '#FF8C00',
#         'Cedar': '#FFD700',
#         'Alder': '#00FF00',
#         'DougFir': '#00FFFF',
#         'Hem': '#1E90FF',
#         'Pine': '#9400D3',
#         'Spruce': '#FF00FF'
#     }

#     def find_corresponding_species(number):
#         values = canfi_map_inverse.get(str(number))
#         if not values:
#             return "No corresponding value found."
        
#         values = values.split('+')
#         for value in values:
#             if value in Aspen:
#                 return 'Aspen'
#             elif value in Bal:
#                 return 'Bal'
#             elif value in Cedar:
#                 return 'Cedar'
#             elif value in Alder:
#                 return 'Alder'
#             elif value in DougFir:
#                 return 'DougFir'
#             elif value in Hem:
#                 return 'Hem'
#             elif value in Pine:
#                 return 'Pine'
#             elif value in Spruce:
#                 return 'Spruce'
        
#         return "No matching set found."
    
#     def calculate_shannon_index(fm, time_period):
#         portion = {}
#         total_volume = fm.inventory(time_period, 'totvol')

#         for theme3 in fm.theme_basecodes(3):
#             volume = fm.inventory(time_period, 'totvol', mask=f'? ? ? {theme3} ? ?')
#             portion[theme3] = volume / total_volume if total_volume > 0 else 0

#         print(f"\nPortion for time period {time_period}:")
#         for theme3, value in portion.items():
#             species = find_corresponding_species(theme3)
#             print(f"{species}: {value:.4f}")

#         shannon_index = -sum(
#             portion[theme3] * math.log(portion[theme3]) / math.log(len(fm.theme_basecodes(3)))
#             for theme3 in portion if portion[theme3] > 0
#         )
#         return shannon_index, portion

#     shannon_0, portion_0 = calculate_shannon_index(fm, time_period=0)
#     shannon_10, portion_10 = calculate_shannon_index(fm, time_period=10)

#     print(f"\nShannon Evennes Index for time period 0: {shannon_0:.4f}")
#     print(f"Shannon Evennes Index for time period 10: {shannon_10:.4f}")

    
#     portion_0_named = {find_corresponding_species(theme3): value for theme3, value in portion_0.items()}
#     portion_10_named = {find_corresponding_species(theme3): value for theme3, value in portion_10.items()}
    
#     shannon_difference = shannon_10 - shannon_0
#     if shannon_difference < 0:
#         print(f"\nDiversity has **decreased** by {abs(shannon_difference) * 100:.2f}% from time 0 to time 10.")
#     else:
#         print(f"\nDiversity has **increased** by {abs(shannon_difference) * 100:.2f}% from time 0 to time 10.")

#     # Prepare data for pie charts (portions of each species for both time periods)
#     labels_0 = [find_corresponding_species(theme3) for theme3 in portion_0.keys()]
#     sizes_0 = [value for value in portion_0.values()]
#     labels_10 = [find_corresponding_species(theme3) for theme3 in portion_10.keys()]
#     sizes_10 = [value for value in portion_10.values()]

#     # Create subplots for pie charts (one row, two columns)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     # Pie chart for time period 0
#     axes[0].pie(sizes_0, labels=labels_0, colors=[colors[find_corresponding_species(theme3)] for theme3 in portion_0.keys()], autopct='%1.1f%%', startangle=140)
#     axes[0].set_title("Species Distribution at Time Period 0")

#     # Pie chart for time period 10
#     axes[1].pie(sizes_10, labels=labels_10, colors=[colors[find_corresponding_species(theme3)] for theme3 in portion_10.keys()], autopct='%1.1f%%', startangle=140)
#     axes[1].set_title("Species Distribution at Time Period 10")

#     # Create a dynamic legend with species present in both time periods
#     unique_species = set(labels_0 + labels_10)  # Unique species from both time periods
#     handles = [plt.Rectangle((0, 0), 1, 1, color=colors[species]) for species in unique_species]
#     fig.legend(handles, unique_species, loc="upper right", title="Species Present")

#     # Save figure
#     folder_path = os.path.join('./outputs/fig', case_study)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
    
#     file_name = f"{case_study}_{obj_mode}_{scenario_name}_species_pie.pdf"
#     file_path = os.path.join(folder_path, file_name)
#     plt.savefig(file_path)
#     plt.show()
#     plt.close()
    
#     print(f"Pie Charts for Time Periods 0 and 10 saved to {file_path}")
#     return portion_10_named, shannon_10






