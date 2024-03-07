import pandas as pd
import datetime as dt
import pyomo.environ as pyo


def results(model):
    """
     This function takes a model as input and returns a DataFrame containing the results of the model. 
     The DataFrame includes columns for the hour, price (in GBP/MWh), charge (in MWh), discharge (in MWh), 
     and state of charge (in MWh) for each hour in the model's time range.
    
    Parameters:
    - model: Pyomo model object representing the battery optimization model.
    
    Returns:
    - results: DataFrame containing charge schedule, state of charge and price for each hour in the model's time range.
    """
    results = pd.DataFrame()
    results["Hour"] = list(range(model.T.first(), model.T.last()+1))
    results["Price (GBP/MWh)"] = model.P()
    results['Charge (MWh)'] = [model.E_charge[t]() for t in model.T]
    results['Discharge (MWh)'] = [model.E_discharge[t]() for t in model.T]
    results['State of Charge (MWh)'] = [model.S[t]() for t in model.T]

    return results


def optimise_battery(df):
    """
    Takes a price curve as input and returns the optimal dispatch schedule for a default
    1MW/2MWh battery. The battery is constrained to a daily discharge volume of two full cycles
    with a round trip efficiency of 90%. 

    Input dataframe should have the following columns:
    - Hour: index for the hour in the time series (for a full year this would be 1 to 8760)
    - Price (GBP/MWh): price of electricity at each hour in GBP/MWh

    Returns a pyomo model object with the following attributes:
    - T: set of hours in the time series
    - Rmax: power capacity. Max rate of charge/discharge. (MW)
    - Smax: energy capacity. Max state of charge. (MWh)
    - Dmax: max daily discharge volume (MWh)
    - F_charge: charging efficiency (%)
    - F_discharge: discharging efficiency (%)
    - P: price of electricity at hour t (GBP/MWh)
    - E_charge: energy charged at hour 't' (MWh)
    - E_discharge: energy discharged at hour 't' (MWh)
    - S: state of charge at hour 't' (MWh)
    """


    model = pyo.ConcreteModel()

    # battery parameters
    model.T = pyo.Set(initialize = df["Hour"], ordered = True, doc = 'Hour')

    model.Rmax = pyo.Param(initialize = 1, doc = 'Power capacity. Max rate of charge/discharge. (MW)')
    model.Smax = pyo.Param(initialize = 2, doc = 'Energy capacity. Max state of charge. (MWh)')
    model.Dmax = pyo.Param(initialize = 4, doc = 'Max daily discharge volume (MWh)')
    model.F_charge = pyo.Param(initialize = 0.9**0.5, doc = 'Charging efficiency (%)')
    model.F_discharge = pyo.Param(initialize = 0.9**0.5, doc = 'Discharging efficiency (%)')
    model.P = pyo.Param(initialize = df["Price (GBP/MWh)"].to_list(), domain = pyo.Any, doc = 'Price of electricity at hour t (GBP/MWh)')

    # optimisation variables
    model.E_charge = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Energy charged at hour 't' (MWh)")
    model.E_discharge = pyo.Var(model.T, domain = pyo.NonNegativeReals, doc = "Energy discharged at hour 't' (MWh)")
    model.S = pyo.Var(model.T, bounds=(0, model.Smax), doc = "State of charge at hour 't' (MWh)")


    # battery constraints
    def state_of_charge(model, t):
        """
        At hour 't', the state of charge (SOC) is the SOC from hour 't-1'
        plus the net flow of energy into the battery, accounting for efficiency loss.
        """
        if t == 1:
            return model.S[t] == 0
        else:
            return model.S[t] == (model.S[t-1]) \
                                + (model.E_charge[t-1] * model.F_charge) \
                                - (model.E_discharge[t-1] * model.F_charge)
        
    model.state_of_charge = pyo.Constraint(model.T, rule = state_of_charge) 

    def charge_constraint(model, t):
        """
        The battery charges up to the power capacity for any hour t.
        """
        return model.E_charge[t] <= model.Rmax

    model.charge = pyo.Constraint(model.T, rule = charge_constraint)

    def discharge_constraint(model, t):
        """
        The battery discharges up to the power capacity for any hour t.
        """
        return model.E_discharge[t] <= model.Rmax

    model.discharge = pyo.Constraint(model.T, rule = discharge_constraint)

    def daily_cycle_constraint(model,t):
        """
        The battery discharges up to the volume prescribed in a single 24 hour cycle, except for the final day.
        """
        max_t = model.T.last()

        if t < max_t - 24:
                return sum(model.E_charge[i] for i in range(t, t+24)) <= model.Dmax
        else:
            return pyo.Constraint.Skip

    model.daily_limit = pyo.Constraint(model.T, rule=daily_cycle_constraint)


    #  battery constrained to not discharge when SOC is 0  
    def postive_charge(model, t):
        """
        Limit discharge to amount of charge in battery, including losses
        """
        return model.E_discharge[t] <= model.S[t] * model.F_discharge


    # objective function
    income = sum(model.P()[t-1] * model.E_discharge[t] for t in model.T)
    cost = sum(model.P()[t-1] * model.E_charge[t] for t in model.T)
    profit = income - cost
    model.objective = pyo.Objective(expr = profit, sense = pyo.maximize)

    # Solve the model
    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    return model