# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:02:01 2020

@author: ashle
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
from torch.distributions import Normal, LogNormal, Uniform, NegativeBinomial, Gamma
import time
from typing import List
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
import os
import scipy
from scipy import stats
import sys

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

import itertools
from math import pi


### PYTORCH VARIABLES
device = "cpu" 
run_ode = True
torch.manual_seed(0)
version = 1 # For saving state data

### SETPOINTS AND TIMES
set_points = range(48,64)
initialization_runs = 1
initialization_end_time = 86400
initialization_time_steps = 5400
initialization_shower_start = initialization_end_time + 10000
initialization_times = torch.linspace(0.0, initialization_end_time, steps=initialization_time_steps).to(device)

#INFO FOR INITIALIZATION
samples = initialization_runs
end_time = initialization_end_time
time_steps = initialization_time_steps
shower_start = initialization_shower_start
times = initialization_times

### FOR MONTE CARLO
montecarlo = 10000
mc_end_time = 800.0
mc_time_steps = 600
mc_shower_start = 1
mc_times = torch.linspace(0.0, mc_end_time, steps=mc_time_steps).to(device)

##################################################################
### USER VARIABLES 

case = 0 if len(sys.argv)<2 else int(sys.argv[1]) #choose case outlined below
num_pipe_nodes_options = 	[26,   52,   26,   26,   52,   52,   26,  26,   52,   26,   26,   52,   52,   26   ]
tank_velocity_options = 	[295,  295,  295,  295,  295,  295,  295, 295,  295,  295,  295,  295,  295,  295  ]
pipe_velocity_options = 	[3.04, 3.04, 1.52, 1.90, 0.3,  1.52, 0.3, 3.04, 3.04, 1.52, 1.90, 0.3,  1.52, 0.3  ]
pipe_vel_round = 			[3,    3,    1,    2,    0,    2,    0,   3,    3,    1,    2,    0,    2,    0    ] #for saving file 
branch_insulation =         [True, True, True, True, True, True, True,False,False,False,False,False,False,False]

pipe_vel = pipe_vel_round[case]
shower_diameter = .012 # m	
pipe_diameter = .019  # m
tank_volume = tank_velocity_options[case]#295 #L or 78 gallons, # 363 #L or 96 gallons
num_heater_nodes = 12
num_pipe_nodes = num_pipe_nodes_options[case]#26 
num_branch_nodes = 3
total_nodes = num_heater_nodes + num_pipe_nodes + num_branch_nodes
pipe_velocity =  pipe_velocity_options[case] #1.52 # m/s
rins = 0.0127 # m, insulation radius
insulation_pipe = True # Must be true
insulation_branch = branch_insulation[case] #Can be true or false
#insulation_branch = True #Can be true or false
##################################################################

### PRESET VARIABLES 
	# PLUMBING PARAMETERS
density_water = 997    
showers_per_day = .69
liters_per_day = 59
pipe_radius = pipe_diameter / 2 # m
shower_radius = shower_diameter / 2 # m	
pipe_area = 2 * math.pi * pipe_radius * 1 # m^2 surface area of pipe per 1 m section
branch_area = 2 * math.pi * shower_radius * 1 # m^2 surface area of branch pipe
	# VARIABLES FOR OVERALL HEAT TRANSFER COEFFICIENT
r1_b, r2_b, r3_b = shower_radius, shower_radius + 0.0045, shower_radius + rins #m, inner radius branch, outer radius branch, outer radius and insulation
r1_p, r2_p, r3_p = pipe_radius, pipe_radius + 0.005, pipe_radius + 0.005 + rins #m, inner radius pipe, outer radius pipe, outer radius and insulation
T1, T2, T3, T4, T5 = 55.50, 55.10, 55.01, 33.50, 23.50 # C, these are estimated values and will not be available unless found in laboratory testing.
k_cu = 401 #W/mK
k_ins = 0.16 #W/mK
h_rad_ins = (k_ins*(T3-T4)) / (rins*(T4-T5))
	# HEAT TRANSFER FOR PIPE/REC LINE
h_conv_pipe =    (k_cu*(T3-T2)) / (r1_p*(T2-T1)) # [W/m2 K]
h_rad_cu_pipe =  (k_cu*(T2-T3)) / (r1_p*(T3-T5))
Uuninsulated_pipe = ((1/h_conv_pipe) + (r1_p/k_cu)*math.log(r2_p/r1_p) + (r1_p/r2_p)*(1/h_rad_cu_pipe))**-1 #math.log() in natural log, math.log10() is the regular log
Uinsulated_pipe = ((1/h_conv_pipe) + (r1_p/k_cu)*math.log(r2_p/r1_p) + (r1_p/k_ins)*math.log(r3_p/r2_p) + (r1_p/r3_p)*(1/h_rad_ins))**-1
if insulation_pipe: 
    htc_pipe = Uinsulated_pipe
    insp = "I"
else:
    htc_pipe = Uuninsulated_pipe
    insp = "U"
	# HEAT TRANSFER FOR BRANCH PIPE 
h_conv_branch =    (k_cu*(T3-T2)) / (r1_b*(T2-T1)) # [W/m2 K]
h_rad_cu_branch =  (k_cu*(T2-T3)) / (r1_b*(T3-T5))
Uuninsulated_branch = ((1/h_conv_branch) + (r1_b/k_cu)*math.log(r2_b/r1_b) + (r1_b/r2_b)*(1/h_rad_cu_branch))**-1 #math.log() in natural log, math.log10() is the regular log
Uinsulated_branch = ((1/h_conv_branch) + (r1_b/k_cu)*math.log(r2_b/r1_b) + (r1_b/k_ins)*math.log(r3_b/r2_b) + (r1_b/r3_b)*(1/h_rad_ins))**-1
if insulation_branch: 
    htc_branch = Uinsulated_branch
    insb = "I"
else:
    htc_branch = Uuninsulated_branch
    insb = "U"       
	# INFECTION COST PARAMETERS
elderly_morbidity_ratio = .75
disability_adj_life_years = .97
value_of_statistical_life = 9300000
avg_life_exp = 78
specific_heat_water = 4186.8
joules_to_watts = 2.77e-7					 
	# BIOFILM PARAMETERS
biofilm_volume_per_area = 1e-4 # m^3/m^2 (greek letter nu in huang et al 2020)
	# CHLORINE DECAY PARAMETERS
arrhenius_a = 1.8e6/60 #/3600 #L/mg h -> L/mgs ## Is this not L/mg min?? I think it is!! Unit analysis!!
arrhenius_b = 6050 #K
c_to_k = 273 #K
chl_threshold = 0.009 #decay to zero (0 = 0.009 mg/L)
	# REYNOLDS NUMBERS
reynolds_pipe = (pipe_velocity * pipe_diameter) / (1e-6)
reynolds_branch = (pipe_velocity * shower_diameter) / (1e-6)

###############################################################################
#### FUNCTIONS
###############################################################################
def percentile(n): #For fancy plots
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def plot_shaded(x, mid, lower, upper, color, label): #For fancy plots
    plt.figure(1)
    plt.plot(x, mid, color + '-', label=label)
    plt.fill_between(x, lower, upper, color=color, alpha=.3)
    
@torch.jit.script
def piecewise(xs, upper_bounds, ys):
    results = torch.empty_like(xs)
    for i in range(ys.shape[0]):
        results[xs < upper_bounds[i]] = ys[i]
    return results

@torch.jit.script
def mixing(nodes: torch.Tensor, node_volume: float, below_coeff: float, above_coeff: float):
    mixing_from_below = torch.zeros_like(nodes)
    mixing_from_below[1:] = below_coeff * (nodes[0:-1] - nodes[1:]) / node_volume

    mixing_from_above = torch.zeros_like(nodes)
    mixing_from_above[0:-1] = above_coeff * (nodes[1:] - nodes[0:-1]) / node_volume
    return mixing_from_below + mixing_from_above


class LegionellaModel(nn.Module):

    def __init__(self,
                 state: dict,
                 device: str,
                 num_heater_nodes: int,
                 num_pipe_nodes: int,
                 num_branch_nodes: int,
                 env_temp: torch.Tensor,
                 main_line_temp: torch.Tensor,
                 main_line_chl: torch.Tensor, ###ADDED
                 initial_leg: torch.Tensor,
                 sloughing_rate: torch.Tensor,
                 toc: torch.Tensor,
                 set_point: float,
                 shower_start: float,
                 shower_duration: torch.Tensor):
												
        self.device = device
        self.env_temp = env_temp
        self.main_line_temp = main_line_temp
        self.main_line_chl = main_line_chl
        self.main_line_leg = initial_leg
        self.sloughing_rate = sloughing_rate
        self.toc = toc

        ### CHLORINE DECAY RATE WITH TEMPERATURE
        self.arrhenius_a = arrhenius_a #1.8e6/3600 #L/mg h -> L/mg s
        self.arrhenius_b = arrhenius_b #6050 #C
        self.c_to_k = c_to_k 
        
		### LEGIONELLA DECAY DUE TO TEMPERATURE AND CHLORINE
        self.leg_temp_decay_upper_bounds = torch.tensor([float('inf'), 70, 65, 60, 55, 50, 45, 42, 37, 30]) # Degrees C
        self.leg_temp_decay_rates = torch.tensor([-7.41E-02, -8.33E-02, -3.33E-02, -6.66E-03, -5.69E-05,
                                                  0.00E+00, 4.82E-05, 4.73E-05, 3.08E-05, 2.30E-05]) #1/s
        self.leg_chl_decay_upper_bounds = torch.tensor([float('inf'), 0.35, 0.15, 0.01]) #mg/L
        self.leg_chl_decay_rates = torch.tensor([-2.31E-2, -1.92E-3, -1.82E-3, 0]) #1/s

		### ENERGY PARAMETERS FOR HEATING
        watts = 5500
        self.heater_power = (watts * 1.89) / (3600 * 57.86)
        self.heater_decay = ((watts * .1) * 1.89) / (3600 * 57.86)
        self.heating_elements = torch.tensor([3, 9])

		### SYSTEM PARAMETERS
        self.set_point = set_point
        self.density_water = density_water 
        self.heater_mixing_coeff = 0.3  # Water exchange between nodes (L/s).
        # self.recirculating_flow_without_shower = .3  # L/s

        self.pipe_velocity = pipe_velocity  # m/s
        self.branch_velocity = pipe_velocity  # m/s
        self.pipe_length = 1 # m
        
        ### PIPE SIZES
        self.pipe_diameter = pipe_diameter  # m
        self.pipe_radius = self.pipe_diameter / 2 # m
        self.shower_diameter = shower_diameter # m								 
        self.shower_radius = self.shower_diameter / 2 # m

		### OVERALL HEAT TRANSFER COEFFICIENTS 
        self.pipe_to_env_heat_flux = htc_pipe#20.0#315.0		# INSULATED
        self.branch_to_env_heat_flux = htc_branch#315.0#20.0		# UNINSULATED COPPER

		### VOLUME OF SECTIONS
        self.tank_volume = tank_volume      
        self.pipe_node_volume = math.pi * (self.pipe_radius) ** 2 * self.pipe_length * 1000  # L
        self.pipe_outflow = math.pi * (self.pipe_radius) ** 2 * self.pipe_velocity * 1000  # L/s

        ### SHOWER PARAMETERS
        self.shower_start = shower_start
        self.shower_duration = shower_duration
        self.shower_outflow = math.pi * (self.shower_radius) ** 2 * self.pipe_velocity * 1000

        ### DEFINING SECTIONS OF THE SYSTEM
        self.num_heater_nodes = num_heater_nodes
        self.num_pipe_nodes = num_pipe_nodes
        self.num_branch_nodes = num_branch_nodes
        self.total_nodes = self.num_pipe_nodes + self.num_heater_nodes + self.num_branch_nodes

        super(LegionellaModel, self).__init__()

    def forward(self, t, densities):
		### DENSITY SECTIONS
        volume_densities = densities[:, 0:4]
        heater_densities = volume_densities[:self.num_heater_nodes, :]
        pipe_densities = volume_densities[self.num_heater_nodes:self.num_heater_nodes + self.num_pipe_nodes, :]
        branch_densities = volume_densities[self.num_heater_nodes + self.num_pipe_nodes:, :]

		### INITIAL DENSITY VALUES							
        temps = volume_densities[:, 0]
        chls = volume_densities[:, 1]
        legs = volume_densities[:, 2]
        sloughs = volume_densities[:, 3]
        bfs = densities[:, 4]

		### TEMP SECTIONS				   
        heater_temps = heater_densities[:, 0]
        pipe_temps = pipe_densities[:, 0]
        branch_temps = branch_densities[:, 0]

		### DERIVATIVE SECTIONS						 
        heater_derivatives = torch.zeros_like(heater_densities)
        pipe_derivatives = torch.zeros_like(pipe_densities)
        branch_derivatives = torch.zeros_like(branch_densities)

        ### MIXING IN THE HEATER
        heater_node_volume = self.tank_volume / heater_densities.shape[0]
        heater_mixing = mixing(heater_densities,
                               heater_node_volume,
                               self.heater_mixing_coeff,
                               self.heater_mixing_coeff)
        heater_derivatives += heater_mixing

        ### PLUG FLOW IN PIPES
        prev_pipe_densities = torch.cat([heater_densities[-1:], pipe_densities[:-1]], dim=0)
        pipe_flow = (prev_pipe_densities - pipe_densities) * self.pipe_velocity / self.pipe_length
        pipe_derivatives += pipe_flow

        ### RECIRCULATION
        shower_on = ((t < (self.shower_start + self.shower_duration)) * (self.shower_start < t)).to(float)
        branch_flow_rate = self.shower_outflow * shower_on
        recirculation_flow_rate = -self.shower_outflow * shower_on + self.pipe_outflow

        recirculation_derivative = recirculation_flow_rate * \
                                   (pipe_densities[-1] - heater_densities[0]) / heater_node_volume
        heater_derivatives[0] += recirculation_derivative

		# Stacking tensors
        intake_densities = torch.stack([
            self.main_line_temp,
            torch.ones_like(self.main_line_temp) * self.main_line_chl,
#            torch.ones_like(self.main_line_temp) * main_line_chl,
            torch.ones_like(self.main_line_temp) * self.main_line_leg,
            torch.zeros_like(self.main_line_temp)
        ], dim=0)

        intake_rate = self.shower_outflow * shower_on
        intake_derivatives = intake_rate * (intake_densities - heater_densities[0]) / heater_node_volume
        heater_derivatives[0] += intake_derivatives

        ### PLUG FLOW IN BRANCH PIPE
        prev_branch_densities = torch.cat([pipe_densities[12:13], branch_densities[:-1]], dim=0)
        branch_flow = (prev_branch_densities - branch_densities) * branch_flow_rate
        branch_derivatives += branch_flow

        ### HEATERS
        top_temp = torch.mean(heater_temps[-7:], dim=0)
        heater_on = top_temp < (self.set_point - 1)
        heat_flux = -torch.ones_like(heater_temps) * self.heater_decay

        heaters = torch.zeros((self.num_heater_nodes,), device=self.device)
        heaters[self.heating_elements] = 1.0
        heat_flux[:, heater_on] = self.heater_power * heaters.unsqueeze(-1)
        heater_derivatives[:, 0] += heat_flux

        #### PIPE HEAT DECAY
        pipe_heat_decay_rate = -4 * self.pipe_to_env_heat_flux / (
                specific_heat_water * self.pipe_diameter * self.density_water * self.pipe_velocity)
        pipe_heat_decay = (pipe_temps - self.env_temp) * pipe_heat_decay_rate
        pipe_derivatives[:, 0] += pipe_heat_decay

        ### BRANCH HEAT DECAY
        branch_heat_decay_rate = -4 * self.branch_to_env_heat_flux / (
                specific_heat_water * self.shower_radius * self.density_water*self.pipe_velocity)
        branch_heat_decay = (branch_temps - self.env_temp) * branch_heat_decay_rate
        branch_derivatives[:, 0] += branch_heat_decay

        ### CONCATENATE DERIVATIVE TENSORS
        volume_dervivatives = torch.cat([heater_derivatives, pipe_derivatives, branch_derivatives], dim=0)
        bf_derivatives = torch.zeros_like(bfs)

        # Chlorine decay
        chl_decay_rate = - self.arrhenius_a * self.toc * torch.exp(-self.arrhenius_b/(temps + self.c_to_k))
        volume_dervivatives[:, 1] += chl_decay_rate * chls

        # Legionella decay from temp
        leg_temp_decay = piecewise(temps, self.leg_temp_decay_upper_bounds, self.leg_temp_decay_rates)
        volume_dervivatives[:, 2] += leg_temp_decay * legs

        # Legionella decay from chl
        leg_chl_decay = piecewise(chls, self.leg_chl_decay_upper_bounds, self.leg_chl_decay_rates)
        volume_dervivatives[:, 2] += leg_chl_decay * legs

		### BIOFILM SLOUGHING RATES
        time_since_shower_start = t - self.shower_start
        if time_since_shower_start < 5 * 60:
            sluff_chl_decay = -.46 / 60 * shower_on
            sluff_rate = 1.3 / 60 * shower_on
        else:
            sluff_chl_decay = -.1 / 60 * shower_on
            sluff_rate = .06 / 60 * shower_on

        is_pipe = torch.zeros(temps.shape[0])
        is_pipe[self.num_heater_nodes:] = 1
        sluff_rates = sluff_rate.unsqueeze(0) * is_pipe.unsqueeze(1)

        bf_derivatives += -sluff_rates * bfs
        volume_dervivatives[:, 3] += sluff_rates * bfs 	
        volume_dervivatives[:, 3] += sluff_chl_decay * chls * leg_temp_decay * sloughs * self.sloughing_rate # CFU/s

        return torch.cat([volume_dervivatives, bf_derivatives.unsqueeze(1)], dim=1)

# From table 3, sampling fixed
def scalding_outcome(temp, time):
    l_time = np.log10(time)
    l_temp = torch.log10(temp)
    #NEW!
    injury_limit = -0.0342 * l_time + 1.783
    necrosis_limit = -0.0359 * l_time + 1.793
    costs = torch.zeros_like(temp)
    # Only samples one point from distribution
#     injury_cost = Uniform(141.76, 221.89).sample([1]).to(device)
#     necrosis_cost = Uniform(628.69, 862.90).sample([1]).to(device)
#     costs[l_temp > injury_limit] = int(injury_cost)
#     costs[l_temp > necrosis_limit] = int(necrosis_cost)
    # Corrected sampling
    injury_cost = Uniform(141.76, 221.89).sample([len(costs)]).to(device).double()
    necrosis_cost = Uniform(628.69, 862.90).sample([len(costs)]).to(device).double()
    costs[l_temp > injury_limit] = injury_cost[l_temp > injury_limit]
    costs[l_temp > necrosis_limit] = necrosis_cost[l_temp > necrosis_limit]
    return costs


###############################################################################
### Sensitivity Analysis
###############################################################################
# INFO FOR MONTECARLO move in a minute
samples = montecarlo
end_time = mc_end_time
time_steps = mc_time_steps
shower_start = mc_shower_start
times = mc_times




#biofilm density changed to Uniform instead of Weibull 

dists = {
    'num_vars':28,
    'names': ["Initial Legionella", 
		"Initial CFU in Biofilm", 
		"Biofilm Density", 
		"Environmental Temperature",
		"Main Line Temperature", 
		"Main Line Chlorine", 
		"Shower Duration", 
		"Jump time", 
		"Sloughing Rate", 
		"Total Organic Carbon", 
		
		"Price per Watt", 
		"Energy Factor", 
		
        #"Clinical Dose Response", 
		"Subclinical Dose Response",
		"VSL", 
		"Remaining LE", 
		"Breathing Rate", 
		
		"Caer1","Caer2","Caer3","Caer4","Vaer1","Vaer2","Vaer3","Vaer4",'De1_2','De2_3','De3_6','De6_10'      
			  ],
    'bounds':[[6.6034, 0.80388], # initial_leg
              [3.9e5, 7.8e9], # initial_c0
              [15580, 55880], # biofilm_density
              [20, 27],  # env_temp
              [16.5,24],#main_line_temp
              [0.01, 4], # main_line_chl
              [468, 72], # shower_duration
			  [1,5], # jump time
              [-18.96,0.709], #sloughing_rate
              [1, 3], # total_org_carbon
              
              [-2.005792, 0.2493262], # price_per_watt
              [0.904, 0.95], #energy_factor
              
              #[-9.69, 0.30], #dose_response_clinical
              [-2.934, 0.488], #dose_response_subclinical
			  [5324706, 17368683], # VSL
			  [46.92, 18.32], # Remaining Life Expectancy
              [0.013 / 60, 0.017 / 60], #breath
              
              [17.533,0.296], # Caer1
              [17.515,0.170], # Caer2
              [19.364,0.348], # Caer3
              [20.000,0.309], # Caer4
              [1,2], # Vaer1
              [2,3], # Vaer2
              [3,6], # Vaer3
              [6,10], # Vaer4             
              [0.23, 0.53], #De1_2
              [0.36, 0.62], #De2_3
              [0.1, 0.62], #De3_6
              [0.01, 0.29] #De6_10             
              ], 
     #'unif','triang','norm','lognorm',          
    'dists':['lognorm', # initial_leg
             'unif', # initial_c0
             'unif', # biofilm_density
             'unif', # env_temp
             'unif', # main_line_temp
             'unif', # main_line_chl 
			 'norm', # shower_duration
			 'unif', # jump time
             'lognorm', #sloughing_rate
             'unif', # total_org_carbon
             
             'lognorm', #price_per_watt
             'unif', #energy_factor
             
             #'lognorm', #dose_response_clinical
             'lognorm', #dose_response_subclinical
			 'unif', # VSL
			 'norm', # Remaining life expectancy
             'unif', #breath
             
             'lognorm', #Caer1
             'lognorm', #Caer2
             'lognorm', #Caer3
             'lognorm', #Caer4            
             'unif', #Vaer1
             'unif', #Vaer2
             'unif', #Vaer3
             'unif', #Vaer4             
             'unif', #De1_2
             'unif', #De2_3
             'unif', #De3_6
             'unif' #De6_10
            ]
}

#samples = N ∗ ( 2D + 2 )
param_values = saltelli.sample(dists, 50, calc_second_order= True)
main_line_chl = param_values[:,4]
set_points = [48]
states = []

leg_growth_rates = [0.0, 0.0, 0.0, -2.775637221930083e-05, -5.689991667168215e-05, -5.689991667168215e-05, -5.689991667168215e-05, -5.689991667168215e-05, -0.0032841498032212257, -0.006659992504864931, -0.006659992504864931, -0.006659992504864931, -0.006659992504864931, -0.019680220633745193, -0.03329996392130852, -0.03329996392130852]

initial_temperature = torch.Tensor([[46.8310, 47.8296, 48.8271, 49.8250, 50.8229, 51.8201, 52.8185, 53.8151, 54.8127, 55.8115, 56.8085, 57.8066, 58.8036, 59.8015, 60.8001, 61.7965],
        [46.8559, 47.8549, 48.8530, 49.8514, 50.8496, 51.8470, 52.8451, 53.8426, 54.8423, 55.8407, 56.8385, 57.8366, 58.8336, 59.8330, 60.8301, 61.8285],
        [47.1438, 48.1430, 49.1418, 50.1407, 51.1392, 52.1367, 53.1356, 54.1334, 55.1334, 56.1324, 57.1306, 58.1292, 59.1266, 60.1264, 61.1239, 62.1229],
        [47.6845, 48.6945, 49.6932, 50.6946, 51.6954, 52.6846, 53.6860, 54.6786, 55.6945, 56.6952, 57.6913, 58.6911, 59.6778, 60.6896, 61.6790, 62.6802],
        [47.0494, 48.0495, 49.0489, 50.0484, 51.0476, 52.0458, 53.0453, 54.0438, 55.0446, 56.0441, 57.0430, 58.0424, 59.0404, 60.0409, 61.0391, 62.0387],
        [46.6673, 47.6677, 48.6671, 49.6669, 50.6664, 51.6650, 52.6646, 53.6635, 54.6644, 55.6642, 56.6633, 57.6629, 58.6612, 59.6618, 60.6604, 61.6601],
        [46.5482, 47.5489, 48.5485, 49.5484, 50.5482, 51.5471, 52.5469, 53.5460, 54.5473, 55.5471, 56.5465, 57.5463, 58.5449, 59.5457, 60.5445, 61.5444],
        [46.6921, 47.6930, 48.6928, 49.6931, 50.6930, 51.6920, 52.6921, 53.6914, 54.6928, 55.6930, 56.6926, 57.6926, 58.6913, 59.6924, 60.6914, 61.6916],
        [47.0991, 48.1000, 49.1003, 50.1007, 51.1008, 52.0999, 53.1003, 54.0997, 55.1013, 56.1017, 57.1016, 58.1018, 59.1007, 60.1020, 61.1012, 62.1017],
        [47.7589, 48.7703, 49.7703, 50.7730, 51.7751, 52.7658, 53.7685, 54.7624, 55.7797, 56.7816, 57.7790, 58.7803, 59.7683, 60.7813, 61.7721, 62.7746],
        [47.2430, 48.2442, 49.2447, 50.2453, 51.2456, 52.2448, 53.2455, 54.2451, 55.2470, 56.2476, 57.2476, 58.2481, 59.2472, 60.2487, 61.2481, 62.2488],
        [46.9800, 47.9813, 48.9816, 49.9822, 50.9826, 51.9820, 52.9825, 53.9822, 54.9842, 55.9847, 56.9847, 57.9852, 58.9844, 59.9858, 60.9853, 61.9859],
        [46.9790, 47.9781, 48.9792, 49.9795, 50.9791, 51.9792, 52.9796, 53.9797, 54.9802, 55.9803, 56.9807, 57.9813, 58.9812, 59.9815, 60.9816, 61.9817],
        [46.9738, 47.9760, 48.9766, 49.9765, 50.9761, 51.9760, 52.9761, 53.9767, 54.9770, 55.9775, 56.9771, 57.9772, 58.9765, 59.9775, 60.9771, 61.9772],
        [46.9748, 47.9731, 48.9735, 49.9734, 50.9733, 51.9733, 52.9732, 53.9730, 54.9730, 55.9734, 56.9733, 57.9738, 58.9732, 59.9733, 60.9738, 61.9740],
        [46.9704, 47.9708, 48.9710, 49.9703, 50.9700, 51.9699, 52.9695, 53.9702, 54.9700, 55.9695, 56.9699, 57.9688, 58.9696, 59.9694, 60.9688, 61.9668],
        [46.9689, 47.9675, 48.9675, 49.9678, 50.9673, 51.9671, 52.9676, 53.9661, 54.9676, 55.9667, 56.9653, 57.9675, 58.9649, 59.9659, 60.9655, 61.9696],
        [46.9648, 47.9661, 48.9655, 49.9644, 50.9643, 51.9638, 52.9615, 53.9634, 54.9590, 55.9620, 56.9645, 57.9599, 58.9621, 59.9609, 60.9616, 61.9546],
        [46.9646, 47.9620, 48.9618, 49.9626, 50.9615, 51.9611, 52.9640, 53.9594, 54.9679, 55.9603, 56.9542, 57.9602, 58.9572, 59.9588, 60.9561, 61.9625],
        [46.9584, 47.9610, 48.9594, 49.9573, 50.9584, 51.9574, 52.9527, 53.9569, 54.9426, 55.9539, 56.9618, 57.9555, 58.9546, 59.9526, 60.9547, 61.9476],
        [46.9617, 47.9558, 48.9573, 49.9595, 50.9556, 51.9555, 52.9588, 53.9531, 54.9715, 55.9551, 56.9435, 57.9449, 58.9496, 59.9504, 60.9471, 61.9523],
        [46.9498, 47.9570, 48.9521, 49.9483, 50.9525, 51.9506, 52.9468, 53.9496, 54.9262, 55.9439, 56.9552, 57.9589, 58.9464, 59.9463, 60.9471, 61.9400],
        [46.9623, 47.9480, 48.9545, 49.9568, 50.9496, 51.9498, 52.9508, 53.9477, 54.9719, 55.9530, 56.9377, 57.9276, 58.9434, 59.9406, 60.9385, 61.9454],
        [46.9367, 47.9550, 48.9430, 49.9411, 50.9467, 51.9440, 52.9421, 53.9419, 54.9157, 55.9302, 56.9458, 57.9551, 58.9363, 59.9393, 60.9396, 61.9275],
        [46.9672, 47.9384, 48.9529, 49.9506, 50.9434, 51.9438, 52.9435, 53.9420, 54.9646, 55.9539, 56.9312, 57.9248, 58.9388, 59.9337, 60.9293, 61.9460],
        [46.9215, 47.9547, 48.9341, 49.9371, 50.9408, 51.9381, 52.9355, 53.9352, 54.9144, 55.9157, 56.9398, 57.9360, 58.9251, 59.9272, 60.9324, 61.9038],
        [46.9702, 47.9277, 48.9490, 49.9423, 50.9373, 51.9372, 52.9384, 53.9349, 54.9487, 55.9523, 56.9202, 57.9324, 58.9342, 59.9337, 60.9211, 61.9603],
        [46.9122, 47.9551, 48.9301, 49.9339, 50.9351, 51.9324, 52.9268, 53.9290, 54.9175, 55.9064, 56.9383, 57.9139, 58.9164, 59.9077, 60.9226, 61.8678],
        [46.9651, 47.9172, 48.9387, 49.9339, 50.9309, 51.9311, 52.9355, 53.9286, 54.9337, 55.9445, 56.9057, 57.9376, 58.9243, 59.9396, 60.9175, 61.9791],
        [46.9107, 47.9543, 48.9327, 49.9319, 50.9299, 51.9250, 52.9158, 53.9207, 54.9158, 55.9020, 56.9402, 57.8953, 58.9157, 59.8870, 60.9059, 61.8407],
        [46.9526, 47.9092, 48.9227, 49.9222, 50.9236, 51.9280, 52.9352, 53.9259, 54.9242, 55.9338, 56.8881, 57.9416, 58.9052, 59.9380, 60.9231, 61.9739],
        [46.9179, 47.9497, 48.9396, 49.9356, 50.9258, 51.9133, 52.9015, 53.9073, 54.9106, 55.8993, 56.9441, 57.8746, 58.9221, 59.8862, 60.8779, 61.8472],
        [46.9283, 47.9054, 48.9037, 49.9029, 50.9153, 51.9307, 52.9391, 53.9300, 54.9141, 55.9207, 56.8698, 57.9491, 58.8841, 59.9039, 60.9414, 61.9362],
        [46.9409, 47.9415, 48.9485, 49.9475, 50.9230, 51.8945, 52.8812, 53.8857, 54.9126, 55.9010, 56.9481, 57.8521, 58.9239, 59.9246, 60.8371, 61.8747],
        [46.8876, 47.9030, 48.8839, 49.8770, 50.9052, 51.9405, 52.9524, 53.9425, 54.8860, 55.9012, 56.8503, 57.9525, 58.8736, 59.8346, 60.9720, 61.8933],
        [46.9714, 47.9353, 48.9554, 49.9617, 50.9225, 51.8704, 52.8465, 53.8569, 54.9483, 55.9106, 56.9568, 57.8455, 58.9117, 59.9826, 60.7844, 61.8972],
        [46.8623, 47.8945, 48.8718, 49.8562, 50.8918, 51.9524, 52.9867, 53.9590, 54.8034, 55.8728, 56.8195, 57.9211, 58.8770, 59.7685, 61.0139, 61.8530],
        [46.9495, 47.9392, 48.9453, 49.9600, 50.9265, 51.8491, 52.7834, 53.8303, 55.0652, 55.9324, 56.9873, 57.9014, 58.8883, 60.0161, 60.7215, 61.9346],
        [23.5334, 23.5072, 23.5228, 23.5015, 23.5062, 23.5387, 23.5094, 23.4944, 23.4780, 23.5164, 23.5069, 23.4729, 23.4952, 23.5098, 23.4747, 23.4966],
        [23.5334, 23.5072, 23.5228, 23.5015, 23.5062, 23.5387, 23.5094, 23.4944, 23.4780, 23.5164, 23.5069, 23.4729, 23.4952, 23.5098, 23.4747, 23.4966],
        [23.5334, 23.5072, 23.5228, 23.5015, 23.5062, 23.5387, 23.5094, 23.4944, 23.4780, 23.5164, 23.5069, 23.4729, 23.4952, 23.5098, 23.4747, 23.4966]])

def SA(param_values):
    dfsa = defaultdict(list)
    for set_point in set_points:
        samples = len(param_values[:,0])
       # Monte Carlo Distributions        
        initial_leg = param_values[:,0] #LogNormal(6.6034,0.80388).sample([samples]).to(device) # CFU/L to initialize the system
        initial_c0 = param_values[:,1] #Uniform(3.9e5, 7.8e9).sample([samples]).to(device)  # CFU/m^2, initial CFU in biofilm, (Schoen and Ashbolt, 2011; Thomas, 2012)
        biofilm_density = param_values[:,2] #torch.Tensor([scipy.stats.weibull_min.rvs(c=3.14,scale=38.66,size=samples)*1000]).to(device) #kg/m3 -> g/m3 from multiplication by *1000
        env_temp = param_values[:,3] #Uniform(20, 27).sample([samples]).to(device) # Degrees C
        main_line_temp = param_values[:,4]#torch.cat([Uniform(17, 24).sample([samples // 2]), Uniform(16.5, 21.5).sample([samples - samples // 2])], dim=0)[torch.randperm(samples)].to(device)
        main_line_chl = param_values[:,5] #Uniform(0.01,4).sample([samples]).to(device) # mg/L	(AWWA, 2018)													  
        shower_duration = np.clip(param_values[:,6],a_min=0,a_max=None) #torch.clamp_min(Normal(468, 72).sample([samples]).to(device), 0) # Seconds (DeOreo et al., 2016)
        jumptime = param_values[:,7]
        sloughing_rate = param_values[:,8].div(0.0001)#LogNormal(-18.96,0.709).sample([samples]).to(device).div(0.0001) #g/cm^2 s -> g/m^2s  ### BIOFILM SLOUGHING DISTRIBUTION
        toc = param_values[:,9] #Uniform(1,3).sample([samples]).to(device) # mg/L
        
        initial_bf_pipe = (pipe_area * initial_c0) / (biofilm_density * biofilm_volume_per_area) # CFU per g per m^2, sloughing variable takes it to CFU/s
        initial_bf_branch = (branch_area * initial_c0) / (biofilm_density * biofilm_volume_per_area) # CFU per g per m^2
   
        state = {} # Add to state for sensitivity analysis
        state["env_temp"] = env_temp.detach().cpu().numpy()
        state["main_line_temp"] = main_line_temp.detach().cpu().numpy()
        state["initial_leg"] = initial_leg.detach().cpu().numpy()																	 
        state["sloughing_rate"] = sloughing_rate.detach().cpu().numpy()
        state["total_organic_carbon"] = toc.detach().cpu().numpy()
        state["initial_c0"] = initial_c0.detach().cpu().numpy()																	 
        state["biofilm_density"] = biofilm_density.detach().cpu().numpy()																	 
        state["main_line_chl"] = main_line_chl.detach().cpu().numpy()			
        state["shower_duration"] = shower_duration.detach().cpu().numpy()

        model = LegionellaModel(
            state,
            device=device,
            num_heater_nodes=num_heater_nodes,
            num_pipe_nodes=num_pipe_nodes,
            num_branch_nodes=num_branch_nodes,
            env_temp = env_temp,
            main_line_temp = main_line_temp,
            main_line_chl = main_line_chl,
            initial_leg = initial_leg,
            sloughing_rate = sloughing_rate,
            toc = toc,
            set_point=set_point,
            shower_start=shower_start,
            shower_duration=shower_duration
    
        ).to(device)
        initial_temps = torch.ones(total_nodes, 1, samples).to(device) * (set_point-10)  # C
        initial_chls = torch.ones(total_nodes, 1, samples).to(device) * main_line_chl  # mg/L
        initial_legs = torch.ones(total_nodes, 1, samples).to(device) * initial_leg  # CFU/L
        initial_sluffs = torch.zeros(total_nodes, 1, samples).to(device)  # CFU/L
        initial_bfs = torch.ones(total_nodes, 1, samples).to(device) #* initial_bf  # CFU/m^2
        initial_bfs[:num_heater_nodes + num_pipe_nodes] = initial_bf_pipe
        initial_bfs[num_heater_nodes + num_pipe_nodes:] = initial_bf_branch
           
        #initial_temp_setpt = initial_temperature[:,set_point-min(set_points)]        #MOVED TWO LINES DOWN
        initial_temps = torch.ones(total_nodes, 1, samples).to(device) #* env_temp #initial_temp_setpt  # Celsius
        initial_temps = initial_temperature[:,set_point-min(set_points)].unsqueeze(dim = 1)  # Celsius
        initial_temps = initial_temps.repeat(1, samples).unsqueeze(dim=1)
      
        # Chlorine decay calculation (ALREADY DEFINED ABOVE)
        #chl_threshold = 0.009  # decay to zero (0 = 0.009 mg/L)
        k_chl_from_temp = -arrhenius_a * toc * math.exp(-arrhenius_b / (set_point + c_to_k))
    ############### !!!!!!! K WAS ALREADY NEGATIVE !!!!!!!!    
        time_chl_decay = (1 / k_chl_from_temp) * np.log(chl_threshold / main_line_chl)  # math.log() in natural log, math.log10() is the regular log
        initial_chls = torch.ones(total_nodes, 1, samples).to(device) * 0.0  # * main_line_chl  # mg/L
    
        # Int Leg. Growth
        time_for_leg_growth = 86400 - time_chl_decay
        k_leg_env_temp = 2.30E-05  # its growth in the env temp range
        int_leg_branch = initial_leg * np.exp(k_leg_env_temp * time_for_leg_growth)
        linear_grow_rate = leg_growth_rates[set_point-min(set_points)]
        #int_leg_pipe_wh = initial_leg + linear_grow_rate * time_for_leg_growth  # y=mx+b
        int_leg_pipe_wh = initial_leg * np.exp(linear_grow_rate*time_for_leg_growth)  # y=mx+b
    
        # Cap the Leg growth
        leg_grow_cap = 10 ** 4.17  # CFU/L from Yee and Wadosky
        int_leg_branch[int_leg_branch > leg_grow_cap] = torch.tensor(leg_grow_cap)
        int_leg_pipe_wh[int_leg_pipe_wh > leg_grow_cap] = torch.tensor(leg_grow_cap)
    
        initial_legs = torch.ones(total_nodes, 1, samples).to(device)  # CFU/L
        initial_legs[:num_heater_nodes + num_pipe_nodes] = int_leg_pipe_wh
        initial_legs[num_heater_nodes + num_pipe_nodes:] = int_leg_branch
    
        initial_sluffs = torch.zeros(total_nodes, 1, samples).to(device)  # CFU/L
    
        initial_bfs = torch.ones(total_nodes, 1, samples).to(device)  # * initial_bf  # CFU/m^2
    #     initial_bfs[:num_heater_nodes + num_pipe_nodes] = initial_bf_pipe
    #     initial_bfs[num_heater_nodes + num_pipe_nodes:] = initial_bf_branch
    ### INITIALIZE THESE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        day = 86400
        initial_bfs[:num_heater_nodes + num_pipe_nodes] = initial_bf_pipe * np.exp(linear_grow_rate * day)
        initial_bfs[num_heater_nodes + num_pipe_nodes:] = initial_bf_branch #* np.exp(k_leg_env_temp * day) #Cap that growth!!
       
        initial_conds = torch.cat([initial_temps,
                                   initial_chls,
                                   initial_legs,
                                   initial_sluffs,
                                   initial_bfs], dim=1).to(device)
    
        start = time.time()
        print("Solving sensitivity analysis: set_point={}".format(set_point))
        all_results = odeint(model, initial_conds, times, rtol=1e-4, atol=1e-2)
        state["all_results"] = all_results
        state["model"] = model
        states.append(state)
        print("Solved eq in {}s".format(time.time() - start))
    
        price_per_watt = param_values[:,10] #LogNormal(-2.005792, 0.2493262).sample([samples]).to(device)
        energy_factor = param_values[:,11]#Uniform(0.904, 0.95).sample([samples]).to(device)
    
        dose_response_subclinical = param_values[:,12] # LogNormal(-2.934, 0.488).sample([samples]).to(device) #subclinical
        dose_response = dose_response_subclinical
        value_of_statistical_life = param_values[:,13]
        remaining_life_exp = np.clip(param_values[:,14],a_min=0,a_max=None)
        breath = param_values[:,15]# Uniform(0.013 / 60, 0.017 / 60).sample([samples]).to(device)  # m3/s
    
        # Cost calculations
        times_b = times.unsqueeze(1).expand((time_steps, samples))
        shower_on = (
                (times_b > shower_start) &
                (times_b < (shower_duration + shower_start))
        ).to(float)
    	
        Caer1 =		param_values[:,16]#LogNormal(17.533,0.296).sample([samples]).to(device) 	#Shower data from O'Toole et al 2009 per m3 #conventional shower
        Caer2 =		param_values[:,17]#LogNormal(17.515,0.170).sample([samples]).to(device)
        Caer3 =		param_values[:,18]#LogNormal(19.364,0.348).sample([samples]).to(device)
        Caer4 =		param_values[:,19]#LogNormal(20.000,0.309).sample([samples]).to(device)
        Vaer1 =		((param_values[:,20]* 1e-6 / 2) **3) * (4/3) * math.pi * 1000#((Uniform(1,2).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
        Vaer2 =		((param_values[:,21]* 1e-6 / 2) **3) * (4/3) * math.pi * 1000#((Uniform(2,3).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
        Vaer3 =		((param_values[:,22]* 1e-6 / 2) **3) * (4/3) * math.pi * 1000#((Uniform(3,6).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
        Vaer4 =		((param_values[:,23]* 1e-6 / 2) **3) * (4/3) * math.pi * 1000#((Uniform(6,10).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
        De1_2 =		param_values[:,24]#Uniform(0.23, 0.53).sample([samples]).to(device) 	#Di= aerosol deposition effeciency 
        De2_3 =		param_values[:,25]#Uniform(0.36, 0.62).sample([samples]).to(device) 		
        De3_6 =		param_values[:,26]#Uniform(0.1, 0.62) .sample([samples]).to(device)		
        De6_10=		param_values[:,27]#Uniform(0.01, 0.29).sample([samples]).to(device) 
        Fi = [0.175,0.1639,0.1556,0.0667,0.0389,0.0250,0.0278,0.05,0.0528,0.0389]	#Fi= partitioning to aerosol of a given size
        Fi1_2=Fi[0]+Fi[1]
        Fi2_3=Fi[2]		
        Fi3_6=Fi[3]+Fi[4]+Fi[5] 		
        Fi6_10=Fi[6]+Fi[7]+Fi[8]+Fi[9]     
        aerosols = (Caer1*Vaer1+Caer2*Vaer2+Caer3*Vaer3+Caer4*Vaer4)*(De1_2*Fi1_2+De2_3*Fi2_3+De3_6*Fi3_6+De6_10*Fi6_10)   
        aeresolized_leg = all_results[:, -1, 2:4, :].sum(dim=1) * aerosols * shower_on # for full data (12 GB, too big for my laptop)
    
        time_since_aeresolized = torch.flip(torch.cumsum(torch.flip(shower_on, [0]), 0), [0]) * (end_time / time_steps)
        decayed_aresolized_leg = torch.exp((-0.989 / 60) * time_since_aeresolized) * aeresolized_leg
        total_aresolized_leg = torch.sum(decayed_aresolized_leg * shower_on * (end_time / time_steps), dim=0)
    
        dose = (total_aresolized_leg * breath)
        risk_ill_per_exposure = 1 - np.exp(- dose_response * dose)
        ### New: clinical risk infection is risk illness, no infection risk for clinical severity.
        infection_cost = risk_ill_per_exposure * disability_adj_life_years * (value_of_statistical_life / remaining_life_exp)
        infection_cost[infection_cost < 0] = 0.0
    
        pipe_start_temp = all_results[-1, num_heater_nodes, 0]
        pipe_end_temp = all_results[-1, num_heater_nodes + num_pipe_nodes - 1, 0]
    
        # Flow or recirculating line per day
        shower_frac = .005
        liters_rec_per_day = 60 * 60 * 24 * (
                model.pipe_outflow * (1 - shower_frac) +
                (model.pipe_outflow - model.shower_outflow) * shower_frac
        )
        cost_per_deg_liter = (price_per_watt / energy_factor) * (specific_heat_water * joules_to_watts)

        Tf = env_temp + (set_point - env_temp) * np.exp((-4 * htc_pipe * num_pipe_nodes) / (specific_heat_water * density_water * pipe_velocity * pipe_diameter))
        energy_cost = cost_per_deg_liter * (liters_per_day * (set_point - main_line_temp) +
                        liters_rec_per_day * (set_point - Tf)) 
     
        
        max_shower_temps = torch.max(all_results[:, -1, 0] * shower_on, dim=0)[0]
        scalding_cost = scalding_outcome(max_shower_temps, jumptime)  # Jump time if burned

        #infection_cost, energy_cost, scalding_cost = get_costs(state)
        dfsa["Infection"].extend(infection_cost.tolist())
        dfsa["Energy"].extend(energy_cost.tolist())
        dfsa["Scalding"].extend(scalding_cost.tolist())
        dfsa["Set Point"].extend([set_point] * samples)
    dfsa = pd.DataFrame(dfsa)
    dfsa["Total"] = dfsa["Infection"] + dfsa["Energy"] + dfsa["Scalding"]
    return dfsa["Total"]


np.savetxt("param_values.txt", param_values)
#Y = np.loadtxt("param_values.txt")
Y = SA(torch.Tensor(param_values))
    
Yrav = Y.ravel()
Si = sobol.analyze(dists, Yrav, calc_second_order= True,print_to_console=True)#False)



###############################################################################
### Sensitivity Analysis PLOTS 
### https://pynetlogo.readthedocs.io/en/latest/_docs/SALib_ipyparallel.html
###############################################################################
problem = dists
problem = {
	'num_vars':28,
		'names': ["Initial Legionella", 
			"Initial CFU in Biofilm", 
			"Biofilm Density", 
			"Environmental Temperature",
			"Main Line Temperature", 
			"Main Line Chlorine", 
			"Shower Duration", 
			"Jump time", 
			"Sloughing Rate", 
			"Total Organic Carbon", 
			
			"Price per Watt", 
			"Energy Factor", 
			
			#"Clinical Dose Response", 
			"Subclinical Dose Response",
			"VSL", 
			"Remaining LE", 
			"Breathing Rate", 
			
			"Caer1","Caer2","Caer3","Caer4","Vaer1","Vaer2","Vaer3","Vaer4",'De1_2','De2_3','De3_6','De6_10'              
			  ],
   'bounds':[[6.6034, 0.80388], # initial_leg
              [3.9e5, 7.8e9], # initial_c0
              [15580, 55880], # biofilm_density
              [20, 27],  # env_temp
              [16.5,24],#main_line_temp
              [0.01, 4], # main_line_chl
              [468, 72], # shower_duration
			  [1,5], # jump time
              [-18.96,0.709], #sloughing_rate
              [1, 3], # total_org_carbon
              
              [-2.005792, 0.2493262], # price_per_watt
              [0.904, 0.95], #energy_factor
              
              #[-9.69, 0.30], #dose_response_clinical
              [-2.934, 0.488], #dose_response_subclinical
			  [5324706, 17368683], # VSL
			  [46.92, 18.32], # Remaining Life Expectancy
              [0.013 / 60, 0.017 / 60], #breath
              
              [17.533,0.296], # Caer1
              [17.515,0.170], # Caer2
              [19.364,0.348], # Caer3
              [20.000,0.309], # Caer4
              [1,2], # Vaer1
              [2,3], # Vaer2
              [3,6], # Vaer3
              [6,10], # Vaer4             
              [0.23, 0.53], #De1_2
              [0.36, 0.62], #De2_3
              [0.1, 0.62], #De3_6
              [0.01, 0.29] #De6_10             
              ], 
     #'unif','triang','norm','lognorm',          
    'dists':['lognorm', # initial_leg
             'unif', # initial_c0
             'unif', # biofilm_density
             'unif', # env_temp
             'unif', # main_line_temp
             'unif', # main_line_chl 
			 'norm', # shower_duration
			 'unif', # jump time
             'lognorm', #sloughing_rate
             'unif', # total_org_carbon
             
             'lognorm', #price_per_watt
             'unif', #energy_factor
             
             #'lognorm', #dose_response_clinical
             'lognorm', #dose_response_subclinical
			 'unif', # VSL
			 'norm', # Remaining life expectancy
             'unif', #breath
             
             'lognorm', #Caer1
             'lognorm', #Caer2
             'lognorm', #Caer3
             'lognorm', #Caer4            
             'unif', #Vaer1
             'unif', #Vaer2
             'unif', #Vaer3
             'unif', #Vaer4             
             'unif', #De1_2
             'unif', #De2_3
             'unif', #De3_6
             'unif' #De6_10
            ]
}




def normalize(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def plot_circles(ax, locs, names, max_s, stats, smax, smin, fc, ec, lw, zorder):
    s = np.asarray([stats[name] for name in names])
    s = 0.01 + max_s * np.sqrt(normalize(s, smin, smax))
    fill = True
    for loc, name, si in zip(locs, names, s):
        if fc=='w':
            fill=False
        else:
            ec='none'
        x = np.cos(loc)
        y = np.sin(loc)
        circle = plt.Circle((x,y), radius=si/2, ec=ec, fc=fc, transform=ax.transData._b,
                            zorder=zorder, lw=lw, fill=True)
        ax.add_artist(circle)

def filter(sobol_indices, names, locs, criterion, threshold):
    if criterion in ['ST', 'S1', 'S2']:
        data = sobol_indices[criterion]
        data = np.abs(data)
        data = data.flatten() # flatten in case of S2
        # TODO:: remove nans
        filtered = ([(name, locs[i]) for i, name in enumerate(names) if
                     data[i]>threshold])
        filtered_names, filtered_locs = zip(*filtered)
    elif criterion in ['ST_conf', 'S1_conf', 'S2_conf']:
        raise NotImplementedError
    else:
        raise ValueError('unknown value for criterion')
    return filtered_names, filtered_locs


def plot_sobol_indices(sobol_indices, criterion='ST', threshold=0.01):
    '''plot sobol indices on a radial plot
    sobol_indices : dict the return from SAlib
    criterion : {'ST', 'S1', 'S2', 'ST_conf', 'S1_conf', 'S2_conf'}, optional
    threshold : float only visualize variables with criterion larger than cutoff '''
    max_linewidth_s2 = 15#25*1.8
    max_s_radius = 0.3
    # prepare data
    # use the absolute values of all the indices
    #sobol_indices = {key:np.abs(stats) for key, stats in sobol_indices.items()}
    # dataframe with ST and S1
    sobol_stats = {key:sobol_indices[key] for key in ['ST', 'S1']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    smax = sobol_stats.max().max()
    smin = sobol_stats.min().min()
    # dataframe with s2
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
                      columns=problem['names'])
    s2[s2<0.0]=0. #Set negative values to 0 (artifact from small sample sizes)
    s2max = s2.max().max()
    s2min = s2.min().min()
    names = problem['names']
    n = len(names)
    ticklocs = np.linspace(0, 2*pi, n+1)
    locs = ticklocs[0:-1]
    filtered_names, filtered_locs = filter(sobol_indices, names, locs,criterion, threshold)
    # setup figure
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(names)
    ax.set_yticklabels([])
    ax.set_ylim(top=1.4)
    legend(ax)
    # plot ST
    plot_circles(ax, filtered_locs, filtered_names, max_s_radius, sobol_stats['ST'], smax, smin, 'w', 'k', 1, 9)
    # plot S1
    plot_circles(ax, filtered_locs, filtered_names, max_s_radius, sobol_stats['S1'], smax, smin, 'k', 'k', 1, 10)
    # plot S2
    for name1, name2 in itertools.combinations(zip(filtered_names, filtered_locs), 2):
        name1, loc1 = name1
        name2, loc2 = name2
        weight = s2.loc[name1, name2]
        lw = 0.5+max_linewidth_s2*normalize(weight, s2min, s2max)
        ax.plot([loc1, loc2], [1,1], c='darkgray', lw=lw, zorder=1)
    return fig

from matplotlib.legend_handler import HandlerPatch
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = plt.Circle(xy=center, radius=orig_handle.radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def legend(ax):
    some_identifiers = [plt.Circle((0,0), radius=5, color='k', fill=False, lw=1),
                        plt.Circle((0,0), radius=5, color='k', fill=True),
                        plt.Line2D([0,0.5], [0,0.5], lw=8, color='darkgray')]
    ax.legend(some_identifiers, ['ST', 'S1', 'S2'],
              loc=(1.25,0.75), borderaxespad=0.1, mode='expand',
              #loc=(1,0.75), borderaxespad=0.1, mode='expand',
              handler_map={plt.Circle: HandlerCircle()})


sns.set_style('whitegrid')
fig = plot_sobol_indices(Si, criterion='ST', threshold=0.005)
#fig = plot_sobol_indices(Si, criterion='ST', threshold=0.005)
fig.set_size_inches(48,48)
#fig.set_size_inches(14,14)
plt.show()


