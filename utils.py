# Data processing
def check_for_dual_pricing(df):
    """
    Spot time steps with dual pricing and attach Boolean column to dataframe

    Input: price dataframe with long/short as column names
    Output: same dataframe with 'dual_pricing' column attached
    """
    dual_pricing_bool = []

    for index, row in df.iterrows():
        if row.long != row.short:
            dual_pricing_bool.append(True)
        else:
            dual_pricing_bool.append(False)

    df['dual_pricing'] = dual_pricing_bool
    return df

# Some small helper functions
def soc_to_idx(soc):
    return int(soc / 0.25)

def idx_to_soc(idx):
    return idx * 0.25

# Getting steering and SOC sequence
def extract_optimal_path(optimal_actions, initial_soc=0):
    
    current_soc = initial_soc
    action_sequence = []
    soc_development = [current_soc]
    
    for t in range(len(optimal_actions)):
        # Get optimal action for current timestep and SOC
        action = optimal_actions[t][soc_to_idx(current_soc)]
        action_sequence.append(action)
        
        # Update SOC based on action
        current_soc = current_soc + (action * -0.25)
        soc_development.append(current_soc)
    
    return action_sequence, soc_development

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import pandas as pd
import numpy as np

# Data visualisation
# Price plot
def plot_prices(dff):
        dff['t'] = pd.to_datetime(dff['t'])

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Create arrays for plotting
        times = dff['t'].values
        long_vals = dff['long'].values
        short_vals = dff['short'].values
        dual_pricing = dff['dual_pricing'].values

        # Plot step function with proper connections
        for i in range(len(dff)-1):
                x_segment = [times[i], times[i+1]]
                
                if not dual_pricing[i]:  # Single pricing
                        plt.step(x_segment, [long_vals[i], long_vals[i]], 
                                where='post', color='purple', linewidth=2)
                        # Connect to next segment
                        if i < len(dff)-2:
                                plt.plot([times[i+1], times[i+1]], [long_vals[i], long_vals[i+1]], 
                                        color='purple', linewidth=2)
                else:  # Dual pricing
                        plt.step(x_segment, [long_vals[i], long_vals[i]], 
                                where='post', color='green', linewidth=2)
                        plt.step(x_segment, [short_vals[i], short_vals[i]], 
                                where='post', color='red', linewidth=2)
                        # Connect to next segments
                        if i < len(dff)-2:
                                plt.plot([times[i+1], times[i+1]], [long_vals[i], long_vals[i+1]], 
                                        color='green', linewidth=2)
                                plt.plot([times[i+1], times[i+1]], [short_vals[i], short_vals[i+1]], 
                                        color='red', linewidth=2)

        # Add legend
        legend_elements = [Line2D([0], [0], color='purple', lw=2, label='Single Price'),
                        Line2D([0], [0], color='green', lw=2, label='Long Price'),
                        Line2D([0], [0], color='red', lw=2, label='Short Price')]
        plt.legend(handles=legend_elements)

        # Set tick positions
        tick_positions = dff['t'].iloc[::4]
        plt.xticks(tick_positions, rotation=45)

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        plt.title('15-Minute Block Prices')
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/MWh)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Steering strategy and revenue overlay onto price plot
def plot_battery_strategy(df, algorithm_name, action_sequence, revenue_per_action):
    """
    Plot battery trading strategy with price signals and actions
    
    Parameters:
    - df: DataFrame with price data
    - algorithm_name: String name for the algorithm (for title)
    - action_sequence: List of actions (-1, 0, 1)
    - revenue_per_action: List of revenues per action
    """
    
    # Set seaborn style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Prepare data
    df = df.copy()
    df['t'] = pd.to_datetime(df['t'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create arrays for plotting
    times = df['t'].values
    long_vals = df['long'].values
    short_vals = df['short'].values
    dual_pricing = df['dual_pricing'].values
    
    # Plot colors
    colors = sns.color_palette("husl", 3)
    purple_color = sns.color_palette("dark")[4]
    green_color = colors[1] 
    red_color = colors[0]
    
    # Plot step function
    for i in range(len(df)-1):
        x_segment = [times[i], times[i+1]]
        
        if not dual_pricing[i]:
            ax.step(x_segment, [long_vals[i], long_vals[i]], 
                    where='post', color=purple_color, linewidth=2.5, alpha=0.8)
            if i < len(df)-2:
                ax.plot([times[i+1], times[i+1]], [long_vals[i], long_vals[i+1]], 
                        color=purple_color, linewidth=2.5, alpha=0.8)
        else:
            ax.step(x_segment, [long_vals[i], long_vals[i]], 
                    where='post', color=green_color, linewidth=2.5, alpha=0.8)
            ax.step(x_segment, [short_vals[i], short_vals[i]], 
                    where='post', color=red_color, linewidth=2.5, alpha=0.8)
            if i < len(df)-2:
                ax.plot([times[i+1], times[i+1]], [long_vals[i], long_vals[i+1]], 
                        color=green_color, linewidth=2.5, alpha=0.8)
                ax.plot([times[i+1], times[i+1]], [short_vals[i], short_vals[i+1]], 
                        color=red_color, linewidth=2.5, alpha=0.8)
    
    # Plot actions
    y_range = max(long_vals) - min(long_vals)
    icon_offset = y_range * 0.03
    text_offset = y_range * 0.08
    
    for i, (action, row) in enumerate(zip(action_sequence, df.itertuples())):
        if i < len(times):
            time_center = times[i] + pd.Timedelta(minutes=7.5)
            
            # Determine price level
            if not row.dual_pricing:
                price_level = row.long
            else:
                if action == -1:
                    price_level = row.short
                elif action == 1:
                    price_level = row.long
                else:
                    price_level = (row.long + row.short) / 2
            
            # Plot icons
            if action == -1:
                ax.scatter(time_center, price_level + icon_offset, 
                          marker='$⬆$', s=200, color='black', alpha=0.9, 
                          edgecolors='white', linewidth=1, 
                          label='Charge' if i == 0 and action == -1 else "")
            elif action == 1:
                ax.scatter(time_center, price_level + icon_offset, 
                          marker='$⬇$', s=200, color='black', alpha=0.9,
                          edgecolors='white', linewidth=1, 
                          label='Discharge' if i == 0 and action == 1 else "")
            elif action == 0:
                ax.scatter(time_center, price_level + icon_offset, 
                          marker='$―$', s=200, color='grey', alpha=0.7,
                          label='Idle' if i == 0 and action == 0 else "")
            
            # Add revenue text
            revenue = revenue_per_action[i]
            if revenue != 0:
                color = '#F24236' if revenue < 0 else '#28A745'
                ax.text(time_center, price_level + text_offset, f'€{revenue:.1f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=color, alpha=0.8)
    
    # Styling
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=purple_color, lw=3, label='Single Price'),
        plt.Line2D([0], [0], color=green_color, lw=3, label='Long Price'),
        plt.Line2D([0], [0], color=red_color, lw=3, label='Short Price'),
        plt.Line2D([0], [0], marker='$⬆$', color='black', lw=0, markersize=10, label='Charge'),
        plt.Line2D([0], [0], marker='$⬇$', color='black', lw=0, markersize=10, label='Discharge'),
        plt.Line2D([0], [0], color='grey', lw=2, alpha=0.6, label='Idle')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=True, shadow=True, ncol=2, fontsize=10)
    
    # Labels and title
    tick_positions = df['t'].iloc[::4]
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.set_title(f'Battery Trading Strategy: {algorithm_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (EUR/MWh)', fontsize=12, fontweight='bold')
    
    # Total revenue
    total_revenue = sum(revenue_per_action)
    ax.text(0.8, 0.2, f'Total Revenue: €{total_revenue:.2f}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            verticalalignment='bottom')
    
    sns.despine()
    plt.tight_layout()
    plt.show()



