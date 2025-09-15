import numpy as np
import pandas as pd
from utils import soc_to_idx, idx_to_soc, extract_optimal_path

def dynamic_programming_backwards_walkthrough(df, initial_soc=2, bess_capacity=2, bess_power_rating=1, time_step_in_hours=0.25, verbose=False):
    """
    This function calculates the globally optimal revenue and action for each time step using a backwards crawl.
    The optimal action and maximum value for each action and SOC combination are calculated by summing the immediate 
    value of each action with the maximum future value of the SOC in the next step, given the action taken. 
    Due to the recursive nature of this algorithm, it runs in linear time and guarantees to find the global optimum.

    Input:
    - Imbalance price and timestamp dataframe
    - The capacity of our BESS asset
    - The power rating of our BESS asset
    - The duration of each time step in hours

    Output:
    - List of actions that result in the optimal revenue

    Steps:
    0. Start at the last time step
    1. Loop through each State Of Charge
        2. Check which charge/discharge actions are available
        3. Calculate and store the immediate value for each available action (action * price)
        4. Sum with the maximum future value of the SOC in the future time step given the previous action (SOC_10 = 1.5 and action = charge -> SOC_11 = 1.75)
        5. Find and store the action that maximizes total value
    6. Go one time step backward in time and repeat

    Note:
    Actions are from the perspective of the grid, meaning -1 is charging the BESS and 1 is discharging the BESS.
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Calculate possible options for SOC given capacity & power rating of the battery
    soc_options = list(np.arange(0, bess_capacity + bess_power_rating*time_step_in_hours, bess_power_rating*time_step_in_hours))

    # Create DP table: dp[timestep][soc_index]
    n_periods = len(df) # Number of time steps
    n_soc_levels = len(soc_options)  # 9 SOC levels
    dp = np.zeros((n_periods + 1, n_soc_levels)) # Set up placeholder array for DP values

    # Create list to track optimal steering decisions
    optimal_actions = np.full((n_periods, n_soc_levels), 999)  # [timestep][soc_index] = best_action

    # Walk backwards through each time step
    for step, row in df[::-1].iterrows():

        vprint(f"=========== Current step: {step} =============")

        # Calculate optimal action for each SOC possibility in each time step
        for soc in soc_options:

            # Check available actions based on SOC
            charge_options = []
            charge_options.append(0)  # Always possible to idle
            
            if soc > 0:
                charge_options.append(1)  # Can discharge
            if soc < 2:
                charge_options.append(-1)  # Can charge

            vprint(f"Checking SOC {soc} with subsequent options {charge_options}")

            max_action_value = -np.inf # Find which action represents the highest total value
            best_action = 3 # Not -1, 0, 1
            
            for action in charge_options:
                vprint("\n")
                vprint(f"Action taken: {action}")

                next_soc =(soc+(action*-0.25))
                vprint(f"Next SOC given action: {next_soc}")

                if not row.dual_pricing:
                    vprint(f"Single price: {row.long}")
                    assert row.long == row.short
                    immediate_value = time_step_in_hours * action * row.long # Long and short price are identical
                    max_future_value = dp[step+1][soc_to_idx(next_soc)] # Take the maximum value for the SOC in the t+1 time step given the current action taken
                    total_value = round(immediate_value + max_future_value, 2)

                    vprint(f"Immediate value: {immediate_value}")
                    vprint(f"Maximum future value: {max_future_value}")
                    vprint(f"Total value: {total_value}")

                else:
                    vprint("Dual price")
                    if action == 0:
                        immediate_value = 0

                    # In theory, this does not necessarily take worst price, however I checked that short price > long price, so practically it does
                    elif action == -1:
                        vprint(f"Short price: {row.short}")
                        immediate_value = time_step_in_hours * action * row.short 
                    
                    elif action == 1:
                        vprint(f"Long price: {row.long}")
                        immediate_value = time_step_in_hours * action * row.long 

                    max_future_value = dp[step+1][soc_to_idx(next_soc)] 
                    total_value = round(immediate_value + max_future_value, 2)

                    vprint(f"Immediate value: {immediate_value}")
                    vprint(f"Maximum future value: {max_future_value}")
                    vprint(f"Total value: {total_value}")
                    
                if total_value > max_action_value:
                    max_action_value = total_value # Find which action represents the highest total value
                    best_action = action
                    vprint(f"New maximum action value: {max_action_value}, by action: {best_action}")
                
            optimal_actions[step][soc_to_idx(soc)] = best_action

            vprint(f"Final max. value for time step {step} and SOC {soc}: {max_action_value}")
            vprint("\n")
            dp[step][soc_to_idx(soc)] = max_action_value

        vprint("\n")

    action_sequence, soc_development = extract_optimal_path(optimal_actions, initial_soc)

    # Calculate revenue per action
    revenue_per_action = []
    for i, (action, row) in enumerate(zip(action_sequence, df.itertuples())):
        if not row.dual_pricing:
            revenue = 0.25 * action * row.long
        else:
            if action == 0:
                revenue = 0
            elif action == -1:  # charging
                revenue = 0.25 * action * row.short
            elif action == 1:   # discharging
                revenue = 0.25 * action * row.long
        revenue_per_action.append(revenue)
    
    return action_sequence, revenue_per_action, sum(revenue_per_action), soc_development


def naive_greedy_algorithm(df, initial_soc=2, bess_capacity=2, bess_power_rating=1, time_step_in_hours=0.25):
    """
    Naive greedy: immediately charge/discharge if any profit can be made
    - Discharge if price > 0 and available energy
    - Charge if price < 0 and available storage capacity
    - Idle otherwise

    This algorithm functions as the lower bound for trading revenue from the BESS. 
    Any operational steering algorithm must attain revenue inbetween that of the backwards crawl and naive-greedy algo's. 
    """
    
    # Track battery state and actions
    current_soc = initial_soc  # Start empty
    actions = []
    revenues = []
    soc_development = [initial_soc]
    
    # Walk forward through each time step
    for step, row in df.iterrows():
        
        # Determine available actions based on current SOC
        if current_soc > 0 and current_soc < bess_capacity:
            available_actions = [-1, 0, 1]  # charge, idle, discharge
        elif current_soc == 0:
            available_actions = [-1, 0]     # charge, idle (can't discharge)
        elif current_soc == bess_capacity:
            available_actions = [0, 1]      # idle, discharge (can't charge)
        
        # Naive greedy decision making
        chosen_action = 0  # Default to idle
        
        if not row.dual_pricing:
            # Single pricing - simple profit check
            if row.long > 0 and 1 in available_actions:
                chosen_action = 1   # discharge (make money)
            elif row.long < 0 and -1 in available_actions:
                chosen_action = -1  # charge (get paid to take energy)
                
        else:
            # Dual pricing - use worst-case prices
            if row.long > 0 and 1 in available_actions:
                chosen_action = 1   # discharge at long price (make money)
            elif row.short < 0 and -1 in available_actions:
                chosen_action = -1  # charge at short price (get paid)
        
        # Execute action
        actions.append(chosen_action)
        
        # Calculate revenue
        if not row.dual_pricing:
            revenue = time_step_in_hours * chosen_action * row.long
        else:
            if chosen_action == 0:
                revenue = 0
            elif chosen_action == -1:  # charging
                revenue = time_step_in_hours * chosen_action * row.short
            elif chosen_action == 1:   # discharging
                revenue = time_step_in_hours * chosen_action * row.long
        
        revenues.append(revenue)
        
        # Update SOC
        current_soc = current_soc + (chosen_action * -(time_step_in_hours*bess_power_rating))
        soc_development.append(current_soc)
    
    return actions, revenues, sum(revenues), soc_development

# Intelligent steering algorithm 
def holdout_phase_of_day_algorithm(df, initial_soc=2, bess_capacity=2, bess_power_rating=1, time_step_in_hours=0.25):

    """
    The goal here is te design some type of operational steering algorithm that outperforms naive-greedy and improves with more data.

    The idea for my steering algorithm will combine two concepts:  
        1. Simple hold-out for prices in top and bottom 25% instead of naive-greedily taking any and all trades that result in positive revenue
        2. Modelling the price distribution of the day in three phases (1-8, 9-16, 17-0) and learning from data in those blocks
    
    Operates in three modes:
        - Naive-greedy: No historical data available. Charges when prices are negative, discharges when prices are positive.
        - Hold-out: Some historical data exists but not for current time phase. 
        Only acts on top/bottom 25% of historical prices instead of any positive revenue.
        - Phase-aware: Full mode when historical data exists for current phase. 
        Only acts on top/bottom 25% of historical prices for that specific phase of the day.
    
    Divides each day into three phases:
    - Phase 1: Night (00:00-07:59)  
    - Phase 2: Day (08:00-15:59)
    - Phase 3: Evening (16:00-23:59)    
    """

    # Track battery state and actions
    current_soc = initial_soc  # Start empty
    actions = []
    revenues = []
    soc_development = [initial_soc]

    # Keep track of which prices we've encountered in order to calculate what 'high' and 'low' prices look like approximately
    # Dual pricing is an edge-case as it causes two prices per step, but we will simply add both to our price list
    prices_seen = []

    prices_seen_phase_1 = []
    prices_seen_phase_2 = []
    prices_seen_phase_3 = []

    # Walk forward through each time step
    for step, row in df.iterrows():

        timestamp = pd.to_datetime(row.t)
        hour = timestamp.hour
        
        # Determine available actions based on current SOC
        if current_soc > 0 and current_soc < bess_capacity:
            available_actions = [-1, 0, 1]  # charge, idle, discharge
        elif current_soc == 0:
            available_actions = [-1, 0]     # charge, idle (can't discharge)
        elif current_soc == bess_capacity:
            available_actions = [0, 1]      # idle, discharge (can't charge)
        
        # Naive greedy decision making
        chosen_action = 0  # Default to idle
        
        # Here are the three stages in our algorithm
        # Stage 1: No data at all - revert to naive-greedy until we have at least 4 price points

        if step < 4:
            if not row.dual_pricing:
                # Single pricing - simple profit check
                if row.long > 0 and 1 in available_actions:
                    chosen_action = 1   # discharge 
                elif row.long < 0 and -1 in available_actions:
                    chosen_action = -1  # charge
                    
            else:
                # Dual pricing - use worst-case prices
                if row.long > 0 and 1 in available_actions:
                    chosen_action = 1   # discharge at long price 
                elif row.short < 0 and -1 in available_actions:
                    chosen_action = -1  # charge at short price 
        
        # Stage 2: Enough data for simple hold-out, but not enough to model daily price distribution
        if (step >= 4) and (step < 96): # One full day of data (0-95) in 15-minute blocks

            if not row.dual_pricing:
                if row.long > np.quantile(prices_seen, 0.75) and 1 in available_actions: # 'High' price found, so discharge
                    chosen_action = 1   # discharge 
                elif row.long < np.quantile(prices_seen, 0.25) and -1 in available_actions: # 'Low' price found, so charge
                    chosen_action = -1  # charge
                    
            else:
                if row.long > np.quantile(prices_seen, 0.75) and 1 in available_actions:
                    chosen_action = 1   # discharge at long price 
                elif row.short < np.quantile(prices_seen, 0.25) and -1 in available_actions:
                    chosen_action = -1  # charge at short price 

        # Stage 3: Full version of algorithm - hold-out which takes into account phase of the day
        if step >= 96: 

            # Collect the prices for the specific phase of the day
            if 0 <= hour < 8:
                prices_used = prices_seen_phase_1
            elif 8 <= hour < 16:
                prices_used = prices_seen_phase_2
            else:  # 16 <= hour < 24
                prices_used = prices_seen_phase_3

            if not row.dual_pricing:
                if row.long > np.quantile(prices_used, 0.75) and 1 in available_actions: # 'High' price found, so discharge
                    chosen_action = 1   # discharge 
                elif row.long < np.quantile(prices_used, 0.25) and -1 in available_actions: # 'Low' price found, so charge
                    chosen_action = -1  # charge
                    
            else:
                if row.long > np.quantile(prices_used, 0.75) and 1 in available_actions:
                    chosen_action = 1   # discharge at long price 
                elif row.short < np.quantile(prices_used, 0.25) and -1 in available_actions:
                    chosen_action = -1  # charge at short price 

        if not row.dual_pricing:
            prices_seen.append(row.long) 

            if 0 <= hour < 8:
                prices_seen_phase_1.append(row.long)  
            elif 8 <= hour < 16:
                prices_seen_phase_2.append(row.long)
            else:  # 16 <= hour < 24
                prices_seen_phase_3.append(row.long)

        else:
            prices_seen.append(row.long) 
            prices_seen.append(row.short) 

            if 0 <= hour < 8:
                prices_seen_phase_1.append(row.long)
                prices_seen_phase_1.append(row.short)  
            elif 8 <= hour < 16:
                prices_seen_phase_2.append(row.long)
                prices_seen_phase_2.append(row.short)
            else:  # 16 <= hour < 24
                prices_seen_phase_3.append(row.long)
                prices_seen_phase_3.append(row.short)

        # Execute action
        actions.append(chosen_action)
        
        # Calculate revenue
        if not row.dual_pricing:
            revenue = time_step_in_hours * chosen_action * row.long
        else:
            if chosen_action == 0:
                revenue = 0
            elif chosen_action == -1:  # charging
                revenue = time_step_in_hours * chosen_action * row.short
            elif chosen_action == 1:   # discharging
                revenue = time_step_in_hours * chosen_action * row.long
        
        revenues.append(revenue)
        
        # Update SOC
        current_soc = current_soc + (chosen_action * -(time_step_in_hours*bess_power_rating))
        soc_development.append(current_soc)

    return actions, revenues, sum(revenues), soc_development