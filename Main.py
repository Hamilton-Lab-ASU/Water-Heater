### Quantatitive model for evaluating risk trade-offs in Legionnaires' 
### Disease risk, energy cost, and scalding risk for hot water systems.
### Code by Ashley Heida, Mark Hamilton, and Kerry A. Hamilton.
###
### MAIN FILE 
###
### Updated September 22, 2021
### New scalding curve (multiple curves)
### Corrected MR in cost of inf eq
### New cost of inf eq with VSLY instead of VSL
### New Life expectancy (remaining) truncated at zero
### Adding median cost text file

### Import Statements
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
# from SALib.sample import saltelli
# from SALib.analyze import sobol
# from SALib.test_functions import Ishigami
import itertools
from math import pi
from matplotlib.legend_handler import HandlerPatch
import csv

# ### Define PATH if needed
# path = "/home/aheida/water_heater/"
# os.chdir(path)
# os.getcwd()

### PYTORCH VARIABLES
device = "cpu" 
run_ode = True
torch.manual_seed(0)

### SETPOINTS AND TIMES
set_points = range(48,64)
initialization_runs = 1
initialization_end_time = 86400
initialization_time_steps = 5400
initialization_shower_start = initialization_end_time + 10000
initialization_times = torch.linspace(0.0, initialization_end_time, steps=initialization_time_steps).to(device)

### INFO FOR INITIALIZATION
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
num_pipe_nodes_options = 	[26,   52,   26,   26,   52,   52,   26,  26   ]
pipe_velocity_options = 	[3.04, 3.04, 1.52, 1.90, 0.3,  1.52, 0.3, 3.04 ]
pipe_vel_round = 			[3,    3,    1,    2,    0,    2,    0,   3    ] #for saving file 
branch_insulation =         [True, True, True, True, True, True, True,False]

pipe_vel = pipe_vel_round[case]
shower_diameter = .012 # m	
pipe_diameter = .019  # m
tank_volume = 295 #tank_velocity_options[case]#295 #L or 78 gallons, # 363 #L or 96 gallons
num_heater_nodes = 12
num_pipe_nodes = num_pipe_nodes_options[case]#26 
num_branch_nodes = 3
total_nodes = num_heater_nodes + num_pipe_nodes + num_branch_nodes
pipe_velocity =  pipe_velocity_options[case] #1.52 # m/s
rins = 0.0127 # m, insulation radius
insulation_pipe = True # Must be true
insulation_branch = branch_insulation[case] #Can be true or false
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
#value_of_statistical_life = 9300000
avg_life_exp = 78.8
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
### Set up median values csv file
###############################################################################
 ### If running in parallel, create a csv file before running and keep this section commented
# sp=list(set_points)
# sp.insert(0,"Case #")
# sp.insert(1,"Optimal Temp")
# sp.insert(2,"Value (USD)")
# header = [sp]
# midCost = open('results/median_results.csv', 'w')
# with midCost:
#    writer = csv.writer(midCost)
#    writer.writerows(header)

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
        heater_node_volume = self.tank_volume / heater_densities.shape[0] #volume/number of nodes
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

		# Main line parameters, Stacking tensors
        intake_densities = torch.stack([
            self.main_line_temp,
            torch.ones_like(self.main_line_temp) * self.main_line_chl,
            torch.ones_like(self.main_line_temp) * self.main_line_leg,
            torch.zeros_like(self.main_line_temp)
        ], dim=0)

        intake_rate = self.shower_outflow * shower_on #intake from the main line is equal to the water lost at the shower when the shower is on.
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
            sluff_rate = 1.3 / 60 * shower_on #this is negative, the negative is below
        else:
            sluff_chl_decay = -.1 / 60 * shower_on
            sluff_rate = .06 / 60 * shower_on #this is negative, the negative is below

        is_pipe = torch.zeros(temps.shape[0])
        is_pipe[self.num_heater_nodes:] = 1 #This only sloughs from the pipes.
        sluff_rates = sluff_rate.unsqueeze(0) * is_pipe.unsqueeze(1)

        bf_derivatives += -sluff_rates * bfs
        volume_dervivatives[:, 3] += sluff_rates * bfs 	
        volume_dervivatives[:, 3] += sluff_chl_decay * chls * leg_temp_decay * sloughs * self.sloughing_rate # CFU/s

        return torch.cat([volume_dervivatives, bf_derivatives.unsqueeze(1)], dim=1)

    
###############################################################################
### Initialiation
###############################################################################
sam = montecarlo
initial_leg = torch.mean(LogNormal(6.6034,0.80388).sample([sam]).to(device)) # CFU/L to initialize the system
initial_c0 = torch.mean(Uniform(3.9e5, 7.8e9).sample([sam]).to(device))  # CFU/m^2, initial CFU in biofilm, (Schoen and Ashbolt, 2011; Thomas, 2012)
biofilm_density = torch.mean(torch.Tensor([scipy.stats.weibull_min.rvs(c=3.14,scale=38.66,size=samples)*1000]).to(device)) #kg/m3 -> g/m3 from multiplication by *1000
env_temp = torch.mean(Uniform(20, 27).sample([sam]).to(device)) # Degrees C
main_line_temp = torch.cat([Uniform(17, 24).sample([samples // 2]), Uniform(16.5, 21.5).sample([samples - samples // 2])], dim=0)[torch.randperm(samples)].to(device)
main_line_chl = torch.mean(Uniform(0.01,4).sample([samples]).to(device)) # mg/L	(AWWA, 2018)													  
shower_duration = torch.mean(torch.clamp_min(Normal(468, 72).sample([samples]).to(device), 0)) # Seconds (DeOreo et al., 2016)
sloughing_rate = LogNormal(-18.96,0.709).sample([samples]).to(device).div(0.0001) #g/cm^2 s -> g/m^2s  ### BIOFILM SLOUGHING DISTRIBUTION
toc = Uniform(1,3).sample([samples]).to(device) # mg/L

initial_bf_pipe = (pipe_area * initial_c0) / (biofilm_density * biofilm_volume_per_area) # CFU per g per m^2, sloughing variable takes it to CFU/s
initial_bf_branch = (branch_area * initial_c0) / (biofilm_density * biofilm_volume_per_area) # CFU per g per m^2

case_file = "case{}_IntTemp.pkl".format(case)  
initial_temperature = torch.load("results/{}".format(case_file))

decay_file = "case{}_IntDecay.pkl".format(case)
leg_growth_rates = torch.load("results/{}".format(decay_file))


###############################################################################
### RUN MONTE CARLO SIMULATION FOR 10,000 ITERATIONS
###############################################################################
samples = montecarlo
end_time = mc_end_time
time_steps = mc_time_steps
shower_start = mc_shower_start
times = mc_times

###############################################################################
### DECISION ANALYSIS FUNCTIONS
###############################################################################
# # Single scalding curve
# def scalding_outcome(temp, time):
#     l_time = np.log10(time)
#     l_temp = torch.log10(temp)
#     injury_limit = -0.0342 * l_time + 1.783
#     costs = torch.zeros_like(temp)
#     injury_cost = Uniform(141.76, 862.90).sample([len(costs)]).to(device).double()
#     costs[l_temp > injury_limit] = injury_cost[l_temp > injury_limit]
#     return costs

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
### Costs for SUBCLINICAL
###############################################################################
def get_costs(state):
    model = state["model"]
    
    env_temp = torch.from_numpy(state['env_temp'])
    all_results_purge = torch.from_numpy(state["all_results_purge"])
    shower_duration = torch.from_numpy(state["shower_duration"])

    price_per_watt = LogNormal(-2.005792, 0.2493262).sample([samples]).to(device)
    energy_factor = Uniform(0.904, 0.95).sample([samples]).to(device)

    dose_response_clinical = LogNormal(-9.69, 0.30).sample([samples]).to(device) #clinical
    dose_response_subclinical = LogNormal(-2.934, 0.488).sample([samples]).to(device) #subclinical
    dose_response = dose_response_subclinical ### SUBCLINICAL DOSE RESPONSE
    breath = Uniform(0.013 / 60, 0.017 / 60).sample([samples]).to(device)  # m3/s
    value_of_statistical_life = Uniform(5324706, 17368683).sample([samples]).to(device)
    remaining_life_exp = torch.clamp_min(avg_life_exp - Normal(46.92, 18.32).sample([samples]).to(device),0) # truncated
 
    # Cost calculations
    times_b = times.unsqueeze(1).expand((time_steps, samples))
    shower_on = (
            (times_b > shower_start) &
            (times_b < (shower_duration + shower_start))
    ).to(float)
	
    Caer1 =		LogNormal(17.533,0.296).sample([samples]).to(device) 	#Shower data from O'Toole et al 2009 per m3 #conventional shower
    Caer2 =		LogNormal(17.515,0.170).sample([samples]).to(device)
    Caer3 =		LogNormal(19.364,0.348).sample([samples]).to(device)
    Caer4 =		LogNormal(20.000,0.309).sample([samples]).to(device)
    Vaer1 =		((Uniform(1,2).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer2 =		((Uniform(2,3).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer3 =		((Uniform(3,6).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer4 =		((Uniform(6,10).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
    De1_2 =		Uniform(0.23, 0.53).sample([samples]).to(device) 	#Di= aerosol deposition effeciency 
    De2_3 =		Uniform(0.36, 0.62).sample([samples]).to(device) 		
    De3_6 =		Uniform(0.1, 0.62) .sample([samples]).to(device)		
    De6_10=		Uniform(0.01, 0.29).sample([samples]).to(device) 
    Fi = [0.175,0.1639,0.1556,0.0667,0.0389,0.0250,0.0278,0.05,0.0528,0.0389]	#Fi= partitioning to aerosol of a given size
    Fi1_2=Fi[0]+Fi[1]
    Fi2_3=Fi[2]		
    Fi3_6=Fi[3]+Fi[4]+Fi[5] 		
    Fi6_10=Fi[6]+Fi[7]+Fi[8]+Fi[9]     
    aerosols = (Caer1*Vaer1+Caer2*Vaer2+Caer3*Vaer3+Caer4*Vaer4)*(De1_2*Fi1_2+De2_3*Fi2_3+De3_6*Fi3_6+De6_10*Fi6_10)   
    aeresolized_leg = all_results_purge[:, -1, 1:3, :].sum(dim=1) * aerosols * shower_on #for cropped data

    time_since_aeresolized = torch.flip(torch.cumsum(torch.flip(shower_on, [0]), 0), [0]) * (end_time / time_steps)
    decayed_aresolized_leg = torch.exp((-0.989 / 60) * time_since_aeresolized) * aeresolized_leg
    total_aresolized_leg = torch.sum(decayed_aresolized_leg * shower_on * (end_time / time_steps), dim=0)

    dose = (total_aresolized_leg * breath)
    risk_inf_per_exposure = 1 - np.exp(- dose_response * dose)

    ### Subclinical cost calculation cannot include morbidity ratio, instead put risk_inf_per_exposure into the infection_cost equation below 
    # risk_ill_per_exposure = risk_inf_per_exposure * elderly_morbidity_ratio
    # infection_cost = risk_ill_per_exposure * disability_adj_life_years * (value_of_statistical_life / avg_life_exp)
#     infection_cost = risk_inf_per_exposure * disability_adj_life_years * (value_of_statistical_life / avg_life_exp)
#     infection_cost[infection_cost < 0] = 0.0


 #### New: subclinical, calculate risk of illness using morbidity ratio
    risk_ill_per_exposure = risk_inf_per_exposure * elderly_morbidity_ratio
    infection_cost = risk_ill_per_exposure * disability_adj_life_years * (value_of_statistical_life / remaining_life_exp)

    
    pipe_start_temp = all_results_purge[0, 0, 0] # changed from [-1,0,0] -> [0,0,0], because the water at the end will be cooler because the hot water is (partially) depleted.
    pipe_end_temp = all_results_purge[0, 1, 0]

    # Flow or recirculating line per day
    shower_frac = .005
    liters_rec_per_day = 60 * 60 * 24 * (
            model.pipe_outflow * (1 - shower_frac) +
            (model.pipe_outflow - model.shower_outflow) * shower_frac
    )
    cost_per_deg_liter = (price_per_watt / energy_factor) * (specific_heat_water * joules_to_watts)
    
    #########################################
    ### New Cost Equation
    #########################################
    Tf = env_temp + (set_point - env_temp) * np.exp((-4 * htc_pipe * num_pipe_nodes) / (specific_heat_water * density_water * pipe_velocity * pipe_diameter))

    energy_cost = cost_per_deg_liter * (liters_per_day * (set_point - main_line_temp) +
                    liters_rec_per_day * (set_point - Tf)) 

    max_shower_temps = torch.max(all_results_purge[:, -1, 0] * shower_on, dim=0)[0]
    jumptime = Uniform(1.0, 5.0).sample([samples]).to(device) # 5.0 #
    scalding_cost = scalding_outcome(max_shower_temps, jumptime)  # Jump time if burned

    state["price_per_watt"] = price_per_watt.detach().cpu().numpy()
    state["energy_factor"] = energy_factor.detach().cpu().numpy()
    
    state["dose_response_clinical"] = dose_response_clinical.detach().cpu().numpy()
    state["dose_response_subclinical"] = dose_response_subclinical.detach().cpu().numpy()
    state["dose_response"] = dose_response.detach().cpu().numpy()
    state["breath"] = breath.detach().cpu().numpy()
    state["value_of_statistical_life"] = value_of_statistical_life.detach().cpu().numpy()
    state["remaining_life_exp"] = remaining_life_exp.detach().cpu().numpy()

    state["Caer1"] = Caer1.detach().cpu().numpy()
    state["Caer2"] = Caer2.detach().cpu().numpy()
    state["Caer3"] = Caer3.detach().cpu().numpy()
    state["Caer4"] = Caer4.detach().cpu().numpy()
    
    state["Vaer1"] = Vaer1.detach().cpu().numpy()
    state["Vaer2"] = Vaer2.detach().cpu().numpy()
    state["Vaer3"] = Vaer3.detach().cpu().numpy()
    state["Vaer4"] = Vaer4.detach().cpu().numpy() 
    
    state["De1_2"] = De1_2.detach().cpu().numpy()
    state["De2_3"] = De2_3.detach().cpu().numpy()
    state["De3_6"] = De3_6.detach().cpu().numpy()
    state["De6_10"] = De6_10.detach().cpu().numpy()

    state["jumptime"] = jumptime.detach().cpu().numpy()
    
    ### We want return of risk of INFECTION for SUBCLINICAL 
    return infection_cost, energy_cost, scalding_cost, risk_inf_per_exposure

###############################################################################
### Costs for CLINICAL
###############################################################################
def get_costs_clin(state):
    model = state["model"]
    
    env_temp = torch.from_numpy(state['env_temp'])
    all_results_purge = torch.from_numpy(state["all_results_purge"])
    shower_duration = torch.from_numpy(state["shower_duration"])

    price_per_watt = LogNormal(-2.005792, 0.2493262).sample([samples]).to(device)
    energy_factor = Uniform(0.904, 0.95).sample([samples]).to(device)

    dose_response_clinical = LogNormal(-9.69, 0.30).sample([samples]).to(device) #clinical
    dose_response_subclinical = LogNormal(-2.934, 0.488).sample([samples]).to(device) #subclinical
    dose_response = dose_response_clinical ### CLINICAL DOSE RESPONSE
    breath = Uniform(0.013 / 60, 0.017 / 60).sample([samples]).to(device)  # m3/s
    value_of_statistical_life = Uniform(5324706, 17368683).sample([samples]).to(device)
    #value_of_statistical_life_yearly = Uniform(5324706,17368683).sample([samples]).to(device)
    remaining_life_exp = torch.clamp_min(avg_life_exp - Normal(46.92, 18.32).sample([samples]).to(device),0) # truncated
    
    # Cost calculations
    times_b = times.unsqueeze(1).expand((time_steps, samples))
    shower_on = (
            (times_b > shower_start) &
            (times_b < (shower_duration + shower_start))
    ).to(float)
	
    Caer1 =		LogNormal(17.533,0.296).sample([samples]).to(device) 	#Shower data from O'Toole et al 2009 per m3 #conventional shower
    Caer2 =		LogNormal(17.515,0.170).sample([samples]).to(device)
    Caer3 =		LogNormal(19.364,0.348).sample([samples]).to(device)
    Caer4 =		LogNormal(20.000,0.309).sample([samples]).to(device)
    Vaer1 =		((Uniform(1,2).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer2 =		((Uniform(2,3).sample([samples]).to(device) * 1e-6 / 2) **3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer3 =		((Uniform(3,6).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
    Vaer4 =		((Uniform(6,10).sample([samples]).to(device) * 1e-6 / 2)**3) * (4/3) * math.pi * 1000 #L/aerosol 
    De1_2 =		Uniform(0.23, 0.53).sample([samples]).to(device) 	#Di= aerosol deposition effeciency 
    De2_3 =		Uniform(0.36, 0.62).sample([samples]).to(device) 		
    De3_6 =		Uniform(0.1, 0.62) .sample([samples]).to(device)		
    De6_10=		Uniform(0.01, 0.29).sample([samples]).to(device) 
    Fi = [0.175,0.1639,0.1556,0.0667,0.0389,0.0250,0.0278,0.05,0.0528,0.0389]	#Fi= partitioning to aerosol of a given size
    Fi1_2=Fi[0]+Fi[1]
    Fi2_3=Fi[2]		
    Fi3_6=Fi[3]+Fi[4]+Fi[5] 		
    Fi6_10=Fi[6]+Fi[7]+Fi[8]+Fi[9]     
    aerosols = (Caer1*Vaer1+Caer2*Vaer2+Caer3*Vaer3+Caer4*Vaer4)*(De1_2*Fi1_2+De2_3*Fi2_3+De3_6*Fi3_6+De6_10*Fi6_10)   
    aeresolized_leg = all_results_purge[:, -1, 1:3, :].sum(dim=1) * aerosols * shower_on #for cropped data

    time_since_aeresolized = torch.flip(torch.cumsum(torch.flip(shower_on, [0]), 0), [0]) * (end_time / time_steps)
    decayed_aresolized_leg = torch.exp((-0.989 / 60) * time_since_aeresolized) * aeresolized_leg
    total_aresolized_leg = torch.sum(decayed_aresolized_leg * shower_on * (end_time / time_steps), dim=0)

    dose = (total_aresolized_leg * breath)
    risk_ill_per_exposure = 1 - np.exp(- dose_response * dose)

  ### Previous  
#     ### We include the morbidity ratio for CLINICAL dose response cases
#     risk_ill_per_exposure = risk_inf_per_exposure * elderly_morbidity_ratio
#     infection_cost = risk_ill_per_exposure * disability_adj_life_years * (value_of_statistical_life / avg_life_exp)
#     infection_cost[infection_cost < 0] = 0.0

### New: clinical risk infection is risk illness, no infection risk for clinical severity.
    infection_cost = risk_ill_per_exposure * disability_adj_life_years * (value_of_statistical_life / remaining_life_exp)

    
    pipe_start_temp = all_results_purge[0, 0, 0] # changed from [-1,0,0] -> [0,0,0], because the water at the end will be cooler because the hot water is (partially) depleted.
    pipe_end_temp = all_results_purge[0, 1, 0]

    # Flow or recirculating line per day
    shower_frac = .005
    liters_rec_per_day = 60 * 60 * 24 * (
            model.pipe_outflow * (1 - shower_frac) +
            (model.pipe_outflow - model.shower_outflow) * shower_frac
    )
    cost_per_deg_liter = (price_per_watt / energy_factor) * (specific_heat_water * joules_to_watts)
    
    #########################################
    ### New Cost Equation
    #########################################
    Tf = env_temp + (set_point - env_temp) * np.exp((-4 * htc_pipe * num_pipe_nodes) / (specific_heat_water * density_water * pipe_velocity * pipe_diameter))

    energy_cost = cost_per_deg_liter * (liters_per_day * (set_point - main_line_temp) +
                    liters_rec_per_day * (set_point - Tf)) 

    max_shower_temps = torch.max(all_results_purge[:, -1, 0] * shower_on, dim=0)[0]
    jumptime = Uniform(1.0, 5.0).sample([samples]).to(device) # 5.0 #
    scalding_cost = scalding_outcome(max_shower_temps, jumptime)  # Jump time if burned

    state["price_per_watt"] = price_per_watt.detach().cpu().numpy()
    state["energy_factor"] = energy_factor.detach().cpu().numpy()
    
    state["dose_response_clinical"] = dose_response_clinical.detach().cpu().numpy()
    state["dose_response_subclinical"] = dose_response_subclinical.detach().cpu().numpy()
    state["dose_response"] = dose_response.detach().cpu().numpy()
    state["breath"] = breath.detach().cpu().numpy()
    state["value_of_statistical_life"] = value_of_statistical_life.detach().cpu().numpy()
    state["remaining_life_exp"] = remaining_life_exp.detach().cpu().numpy()
    
    state["Caer1"] = Caer1.detach().cpu().numpy()
    state["Caer2"] = Caer2.detach().cpu().numpy()
    state["Caer3"] = Caer3.detach().cpu().numpy()
    state["Caer4"] = Caer4.detach().cpu().numpy()
    
    state["Vaer1"] = Vaer1.detach().cpu().numpy()
    state["Vaer2"] = Vaer2.detach().cpu().numpy()
    state["Vaer3"] = Vaer3.detach().cpu().numpy()
    state["Vaer4"] = Vaer4.detach().cpu().numpy() 
    
    state["De1_2"] = De1_2.detach().cpu().numpy()
    state["De2_3"] = De2_3.detach().cpu().numpy()
    state["De3_6"] = De3_6.detach().cpu().numpy()
    state["De6_10"] = De6_10.detach().cpu().numpy()

    state["jumptime"] = jumptime.detach().cpu().numpy()

    return infection_cost, energy_cost, scalding_cost, risk_ill_per_exposure

###############################################################################
###############################################################################
###############################################################################
#### SUBCLINICAL: Run Decision Analysis Functions and Plot   
###############################################################################
###############################################################################
###############################################################################

samples = montecarlo
df = defaultdict(list)
dfrisk = defaultdict(list)
scaldrisk = []

for i, set_point in enumerate(set_points): 
    state_file = "results/case{}_statebio_{}.pkl".format(case, set_point)
    
    with open(state_file, "rb") as f:
        state = torch.load(f)
    infection_cost, energy_cost, scalding_cost, risk_inf = get_costs(state)
    df["Infection"].extend(infection_cost.tolist())
    df["Energy"].extend(energy_cost.tolist())
    df["Scalding"].extend(scalding_cost.tolist())
    df["Set Point"].extend([set_point] * samples)
    
    dfrisk["Risk Infection"].extend(risk_inf.tolist())
    dfrisk["Set Point"].extend([set_point] * samples)

    noinjury = len(scalding_cost[scalding_cost<141])
    injury_pt1 = scalding_cost[scalding_cost>141]
    injury_pt2 = len(injury_pt1[injury_pt1<628])
    necrosis = len(scalding_cost[scalding_cost>628])
    scaldrisk.append([noinjury, injury_pt2, necrosis])

sr = pd.DataFrame(scaldrisk)
sr.to_csv('results/sept22_2021/scald_case{}.csv'.format(case))
    
df = pd.DataFrame(df)
df["Total"] = df["Infection"] + df["Energy"] + df["Scalding"]
df = pd.melt(df, id_vars=['Set Point'], value_vars=["Infection", 'Energy', "Scalding", "Total"])
df = df.rename(columns={"value": "Cost ($)", "variable": "Cost Type"})


###############################################################################
### PLOTS: TOTAL COSTS SUBCLINICAL
###############################################################################
grouped = df.groupby(["Set Point", "Cost Type"])["Cost ($)"].agg(
    [percentile(5), percentile(50), percentile(95)]).unstack()
grouped.to_csv('results/sept22_2021/grouped_case{}.csv'.format(case))

cost_types = sorted(set(df["Cost Type"]))
colors = ["g", "b", "y", "r"]
for cost_type, color in zip(cost_types, colors):
    mid = []
    lower = []
    upper = []
    for set_point in set_points:
        lower.append(grouped["percentile_5"][cost_type][set_point])
        mid.append(grouped["percentile_50"][cost_type][set_point])
        upper.append(grouped["percentile_95"][cost_type][set_point])
    plt.figure(16,figsize=(3,3))
    plot_shaded(set_points, mid, lower, upper, color, cost_type)
plt.legend()
plt.title("Case {} Total Costs".format(case))#, Subclinical")
plt.xlabel("Set Point [Degrees C]")
plt.ylabel("Log10(Cost) [$]")
plt.semilogy() #log scales the y axis
#plt.axes().set_ylim([10e-4, 10e4])
plt.axes().set_ylim([10e-4, 10e5])

# Circle optimal Temp
optimal_temp = 48 + mid.index(min(mid))
plt.plot(optimal_temp, min(mid), marker="o",color='k', markersize=15,fillstyle='none')
plt.savefig("results/sept22_2021/case{}scaleSUB.png".format(case))
plt.close()

########################################################################
### Add median values for total costs (subclinical) to csv file
midcsv = mid
midcsv.insert(0,case)
midcsv.insert(1,optimal_temp)
#midcsv.insert(2,mid[optimal_temp-48])
midcsv.insert(2,min(mid[3:]))

midData = [midcsv]
midCost = open('results/median_results.csv', 'a')
with midCost:
   writer = csv.writer(midCost)
   writer.writerows(midData)
    

###############################################################################
### PLOTS RISK SUBCLINICAL
###############################################################################

dfrisk = pd.DataFrame(dfrisk)
grouped = dfrisk.groupby(["Set Point"])["Risk Infection"].agg([percentile(5), percentile(50), percentile(95)]).unstack()
grouped.to_csv('results/sept22_2021/riskill_case{}.csv'.format(case))

cost_types = sorted(set(dfrisk["Set Point"]))
colors = ["b"]
for cost_type, color in zip(cost_types, colors):
    mid = []
    lower = []
    upper = []
    for set_point in set_points:
        lower.append(grouped["percentile_5"][set_point])
        mid.append(grouped["percentile_50"][set_point])
        upper.append(grouped["percentile_95"][set_point])
    plt.figure(16,figsize=(3,3))
    plot_shaded(set_points, mid, lower, upper, color, cost_type)

plt.legend()
plt.title("Case {} Risk of Infection".format(case))#, Subclinical")
plt.xlabel("Set Point [Degrees C]")
plt.ylabel("Log10(Risk)")
plt.semilogy() #log scales the y axis
plt.axes().set_ylim([10e-6, 10e-1]) # 10^-5 to 10^0

# Circle optimal Temp
optimal_temp = 48 + mid.index(min(mid))
plt.plot(optimal_temp, min(mid), marker="o",color='k', markersize=15,fillstyle='none')
plt.savefig("results/sept22_2021/RISKSUB_case{}scale.png".format(case))
plt.close()




###############################################################################
###############################################################################
###############################################################################
#### CLINICAL: Run Decision Analysis Functions and Plot   
###############################################################################
###############################################################################
###############################################################################

samples = montecarlo
df = defaultdict(list)
dfrisk = defaultdict(list)
scaldrisk = []
for i, set_point in enumerate(set_points): 
    state_file = "results/case{}_statebio_{}.pkl".format(case, set_point)
    with open(state_file, "rb") as f:
        state = torch.load(f)
    infection_cost, energy_cost, scalding_cost, risk_ill = get_costs_clin(state)
    df["Infection"].extend(infection_cost.tolist())
    df["Energy"].extend(energy_cost.tolist())
    df["Scalding"].extend(scalding_cost.tolist())
    df["Set Point"].extend([set_point] * samples)

    dfrisk["Risk Illness"].extend(risk_ill.tolist())
    dfrisk["Set Point"].extend([set_point] * samples)
    
    noinjury = len(scalding_cost[scalding_cost<141])
    injury_pt1 = scalding_cost[scalding_cost>141]
    injury_pt2 = len(injury_pt1[injury_pt1<628])
    necrosis = len(scalding_cost[scalding_cost>628])
    scaldrisk.append([noinjury, injury_pt2, necrosis])

sr = pd.DataFrame(scaldrisk)
sr.to_csv('results/sept22_2021/scald_case{}.csv'.format(case+8))
    
df = pd.DataFrame(df)
df["Total"] = df["Infection"] + df["Energy"] + df["Scalding"]
df = pd.melt(df, id_vars=['Set Point'], value_vars=["Infection", 'Energy', "Scalding", "Total"])
df = df.rename(columns={"value": "Cost ($)", "variable": "Cost Type"})

###############################################################################
### PLOTS: TOTAL COSTS CLINICAL
###############################################################################
grouped = df.groupby(["Set Point", "Cost Type"])["Cost ($)"].agg([percentile(5), percentile(50), percentile(95)]).unstack()
grouped.to_csv('results/sept22_2021/grouped_case{}.csv'.format(case+8))

cost_types = sorted(set(df["Cost Type"]))
colors = ["g", "b", "y", "r"]
for cost_type, color in zip(cost_types, colors):
    mid = []
    lower = []
    upper = []
    for set_point in set_points:
        lower.append(grouped["percentile_5"][cost_type][set_point])
        mid.append(grouped["percentile_50"][cost_type][set_point])
        upper.append(grouped["percentile_95"][cost_type][set_point])
    plt.figure(16,figsize=(3,3))
    plot_shaded(set_points, mid, lower, upper, color, cost_type)
plt.legend()
plt.title("Case {} Total Costs".format(case+8))#, Subclinical")
plt.xlabel("Set Point [Degrees C]")
plt.ylabel("Log10(Cost) [$]")
plt.semilogy() #log scales the y axis
#plt.axes().set_ylim([10e-4, 10e4])
plt.axes().set_ylim([10e-4, 10e5])

# Circle optimal Temp
optimal_temp = 48 + mid.index(min(mid))
plt.plot(optimal_temp, min(mid), marker="o",color='k', markersize=15,fillstyle='none')
plt.savefig("results/sept22_2021/case{}scaleCLIN.png".format(case+8))
plt.close()

########################################################################
### Add median values for total costs (clinical) to csv file
midcsv = mid
midcsv.insert(0,case+8)
midcsv.insert(1,optimal_temp)
#midcsv.insert(2,mid[optimal_temp-48])
midcsv.insert(2,min(mid[3:]))

midData = [midcsv]
midCost = open('results/median_results.csv', 'a')
with midCost:
   writer = csv.writer(midCost)
   writer.writerows(midData)
    

###############################################################################
### PLOTS RISK CLINICAL
###############################################################################

dfrisk = pd.DataFrame(dfrisk)
grouped = dfrisk.groupby(["Set Point"])["Risk Illness"].agg([percentile(5), percentile(50), percentile(95)]).unstack()
grouped.to_csv('results/sept22_2021/riskill_case{}.csv'.format(case+8))

cost_types = sorted(set(dfrisk["Set Point"]))
colors = ["b"]
for cost_type, color in zip(cost_types, colors):
    mid = []
    lower = []
    upper = []
    for set_point in set_points:
        lower.append(grouped["percentile_5"][set_point])
        mid.append(grouped["percentile_50"][set_point])
        upper.append(grouped["percentile_95"][set_point])
    plt.figure(16,figsize=(3,3))
    plot_shaded(set_points, mid, lower, upper, color, cost_type)

plt.legend()
plt.title("Case {} Risk of Illness".format(case+8))#, Subclinical")
plt.xlabel("Set Point [Degrees C]")
plt.ylabel("Log10(Risk)")
plt.semilogy() #log scales the y axis
plt.axes().set_ylim([10e-9, 10e-1]) # 10^-5 to 10^0

# Circle optimal Temp
optimal_temp = 48 + mid.index(min(mid))
plt.plot(optimal_temp, min(mid), marker="o",color='k', markersize=15,fillstyle='none')
plt.savefig("results/sept22_2021/RISKCLIN_case{}scale.png".format(case+8))
plt.close()




# ##############################################################################
# ##############################################################################
# ##############################################################################
# ## SENSITIVITY ANALYSIS
# ##############################################################################
# ##############################################################################
# ##############################################################################

# ## Spearman info if you need a refresher: https://geographyfieldwork.com/SpearmansRankCalculator.html#:~:text=P%2Dvalues%20are%20determined%20by,that%20stated%20in%20H0.
# names = ["Environmental Temperature", "Main Line Temperature", "Initial Legionella", "Sloughing Rate",
#  "Total Organic Carbon", "Initial CFU in Biofilm", "Biofilm Density", "Main Line Chlorine",
#  "Shower Duration", "Jump time", "Price per Watt", "Energy Factor", "Clinical Dose Response",  "Subclinical Dose Response",
#          "VSL", "Remaining LE",
#  #"Dose Response",
#  "Breathing Rate", "Caer1","Caer2","Caer3","Caer4","Vaer1","Vaer2","Vaer3","Vaer4",'De1_2','De2_3','De3_6','De6_10']
# name_ib = ["env_temp", "main_line_temp", "initial_leg", "sloughing_rate", "total_organic_carbon", "initial_c0",
#            "biofilm_density", "main_line_chl", "shower_duration", 'jumptime','price_per_watt', 'energy_factor', 
#            'dose_response_clinical', 'dose_response_subclinical', 
#            'value_of_statistical_life','remaining_life_exp',
#             'breath', "Caer1","Caer2","Caer3","Caer4",
#            "Vaer1","Vaer2","Vaer3","Vaer4",'De1_2','De2_3','De3_6','De6_10']



# spearman_stats = pd.DataFrame([], columns = list(name_ib))
# pvalue_stats = pd.DataFrame([], columns = list(name_ib))

# case_list = [0,1,2,3,4,5,6,7]
# #set_points = [48]
# set_points = range(48,64)

# ###############################################################################
# ### SUBCLINICAL SENSITIVITY ANALYSIS
# ###############################################################################

# for case in case_list:
# #    name_ib = name_ib_subclinical
#     samples = montecarlo
#     for i, set_point in enumerate(set_points):  
#         df = defaultdict(list)
#         state_file = "results/case{}_statebio_{}.pkl".format(case, set_point)
#         #state_file = "results/case{}_state_{}.pkl".format(case, set_point)
#         #state_file = "results/state_file_{}_{}p_{}b_{}pipe_{}tank_{}vel.pkl".format(set_point,insp, insb, num_pipe_nodes,tank_volume,pipe_vel)    
#         with open(state_file, "rb") as f:
#             state = torch.load(f)

#         infection_cost, energy_cost, scalding_cost, risk_inf = get_costs(state)

#         df["Infection"].extend(infection_cost.tolist())
#         df["Energy"].extend(energy_cost.tolist())
#         df["Scalding"].extend(scalding_cost.tolist())
#         df["Set Point"].extend([set_point] * samples)

#         df = pd.DataFrame(df)
#         df["Total"] = df["Infection"] + df["Energy"] + df["Scalding"]

#     spearman = []
#     pvalue = []
#     significant = []
#     colors = []
#     for i in name_ib:
#         if i == "biofilm_density":
#             r = stats.spearmanr(df["Total"], state[i].squeeze())[0]
#             p = stats.spearmanr(df["Total"], state[i].squeeze())[1]
#         else:
#             r = stats.spearmanr(df["Total"], state[i])[0]
#             p = stats.spearmanr(df["Total"], state[i])[1]
#         spearman.append(r)
#         pvalue.append(p) 
#         if p <= 0.025:
#             significant.append(i)
#             colors.append("red")
#         else:
#             colors.append("blue")

#     df_spearman = pd.DataFrame([spearman], index= ["spearman_{}".format(case)], columns = list(name_ib))
#     df_pvalue = pd.DataFrame([pvalue], index= ["pvalue_{}".format(case)], columns = list(name_ib))
#     spearman_stats = spearman_stats.append(df_spearman)
#     pvalue_stats = pvalue_stats.append(df_pvalue)

#     #spearman_stats = pd.DataFrame([name_ib, spearman, pvalue])
#     #spearman_stats = np.array([name_ib, spearman, pvalue])


# ###############################################################################
# ### CLINICAL SENSITIVITY ANALYSIS
# ###############################################################################

# for case in case_list:
# #    name_ib = name_ib_clinical
#     samples = montecarlo
#     for i, set_point in enumerate(set_points):  
#         df = defaultdict(list)
#         state_file = "results/case{}_statebio_{}.pkl".format(case, set_point)
#         #state_file = "results/case{}_state_{}.pkl".format(case, set_point)
#         #state_file = "results/state_file_{}_{}p_{}b_{}pipe_{}tank_{}vel.pkl".format(set_point,insp, insb, num_pipe_nodes,tank_volume,pipe_vel)    
#         with open(state_file, "rb") as f:
#             state = torch.load(f)

#         infection_cost, energy_cost, scalding_cost, risk_ill = get_costs_clin(state)

#         df["Infection"].extend(infection_cost.tolist())
#         df["Energy"].extend(energy_cost.tolist())
#         df["Scalding"].extend(scalding_cost.tolist())
#         df["Set Point"].extend([set_point] * samples)

#         df = pd.DataFrame(df)
#         df["Total"] = df["Infection"] + df["Energy"] + df["Scalding"]

#     spearman = []
#     pvalue = []
#     significant = []
#     colors = []
#     for i in name_ib:
#         if i == "biofilm_density":
#             r = stats.spearmanr(df["Total"], state[i].squeeze())[0]
#             p = stats.spearmanr(df["Total"], state[i].squeeze())[1]
#         else:
#             r = stats.spearmanr(df["Total"], state[i])[0]
#             p = stats.spearmanr(df["Total"], state[i])[1]
#         spearman.append(r)
#         pvalue.append(p) 
#         if p <= 0.025:
#             significant.append(i)
#             colors.append("red")
#         else:
#             colors.append("blue")

#     df_spearman = pd.DataFrame([spearman], index= ["spearmanclin_{}".format(case)], columns = list(name_ib))
#     df_pvalue = pd.DataFrame([pvalue], index= ["pvalueclin_{}".format(case)], columns = list(name_ib))
#     spearman_stats = spearman_stats.append(df_spearman)
#     pvalue_stats = pvalue_stats.append(df_pvalue)

# # ###############################################################################
# # ### SAVE SPEARMAN DATA
# # ###############################################################################  
# # spear_stats = "case{}_SpearmanStats.pkl".format(case)  
# # torch.save(spear_stats, "results/{}".format(spear_stats))

# ### Plotting Spearman 
# plt.figure(17)
# sns.set(style='darkgrid')
# df_spear = pd.DataFrame()
# df_spear['input'] = names
# df_spear['+'] = spearman
# # df_spear['input'] = names[:14]
# # df_spear['+'] = spearman[:14]
# df2 = pd.melt(df_spear, id_vars ='input', var_name='type of change', value_name='change in the output' )
# fig, ax = plt.subplots(figsize=(5,6))
# for typ, df_spear in zip(df2['type of change'].unique(),df2.groupby('type of change')):
#     ax.barh(df_spear[1]['input'], df_spear[1]['change in the output'], height=0.75, label=typ,color=colors[:14])
# leg_name=[Patch(facecolor='red',label='Significant, p:0.025'),Patch(facecolor='blue',label='Not Significant')]
# ax.legend(handles=leg_name, loc='lower right')
# plt.title("Spearman Rank Corrrelation Coefficients")

# ##############################################################################
# ## SAVE TORNADO PLOT (MAYBE??)
# ##############################################################################   
# fig.savefig("results/case{}_spearman.png".format(case))
    
    
# ##############################################################################
# ## Spearman heatmap
# ##############################################################################    
# data = spearman_stats
# case_count = ["Case 0","Case 1","Case 2","Case 3","Case 4","Case 5",
#              "Case 6","Case 7","Case 8","Case 9","Case 10","Case 11",
#              "Case 12","Case 13","Case 14","Case 15"]

# yticks = case_count#data.index
# xticks = names# data.columns
# fig, ax = plt.subplots(figsize=(7.5,6))



# #sns.heatmap(data, cmap="YlGnBu", mask=data.isnull())
# #cmaps = ["bwr", "viridis", 'plasma', 'plasma', 'cividis',"YlGnBu"]
# mask = data.isnull()
# mask["dose_response_subclinical"][:8] = True
# mask["dose_response_clinical"][8:] = True

# ax.set_facecolor("black")

# sns.heatmap(data, linewidth=0, yticklabels=yticks, xticklabels=xticks, center=0, cmap="bwr", mask=mask,cbar_kws={'label': 'Correlation Coefficient'})#data.iloc[0][0])#,cmap="colorblin")

# # This sets the yticks "upright" with 0, as opposed to sideways with 90.
# plt.yticks(rotation=0, fontsize = 12) 
# plt.xticks(fontsize = 12) 

# plt.title("Spearman Correlation Coefficient by Case", fontsize = 14)
# #plt.xlabel("Variable")
# #plt.ylabel("Case Number")
# plt.tight_layout()
# fig.savefig("results/sept22_2021/spearman_by_case2.png")
# plt.show()


# ###############################################################################
# ###############################################################################
# ###############################################################################
# ###  Sobol Sensitivity Analysis
# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### https://salib.readthedocs.io/en/latest/

# set_points = [48]

# dists_intvalues = {
#     'num_vars':9,
#     'names': ['initial_leg',
#               'initial_c0',
#               'biofilm_density',
#               'env_temp',
#               'main_line_temp',
#               'main_line_chl',
#               'shower_duration',
#               'sloughing_rate',
#               'total_org_carbon'            
# 			  ],
#     'bounds':[[6.6034, 0.80388], # initial_leg
#               [3.9e5, 7.8e9], # initial_c0
#               [15580, 55880], # biofilm_density
#               [20, 27],  # env_temp
#               [16.5,24],#main_line_temp
#               [0.01, 4], # main_line_chl
#               [468, 72], # shower_duration
#               [-18.96,0.709], #sloughing_rate
#               [1, 3] # total_org_carbon        
#               ], 
#      #'unif','triang','norm','lognorm',          
#     'dists':['lognorm', # initial_leg
#              'unif', # initial_c0
#              'unif', # biofilm_density
#              'unif', # env_temp
#              'unif', # main_line_temp
#              'unif', # main_line_chl 
# 			 'norm', # shower_duration
#              'lognorm', #sloughing_rate
#              'unif' # total_org_carbon
#             ]
# }

# dists_energy = {
#     'num_vars':2,
#     'names': ['price_per_watt',
#               'energy_factor' ],
#     'bounds':[[-2.005792, 0.2493262], # price_per_watt
#               [0.904, 0.95]], #energy_factor
#      #'unif','triang','norm','lognorm',          
#     'dists':['lognorm', #price_per_watt
#              'unif'] #energy_factor           
# }

# dists_infection = {
#     'num_vars':15,
#     'names': ['dose_response_clinical',
#               'dose_response_subclinical',
#               'breath',
              
#               'Caer1',
#               'Caer2',
#               'Caer3',
#               'Caer4',           
#               'Vaer1',
#               'Vaer2',
#               'Vaer3',
#               'Vaer4',              
#               'De1_2',
#               'De2_3',
#               'De3_6',
#               'De6_10'             
# 			  ],
#     'bounds':[[-9.69, 0.30], #dose_response_clinical
#               [-2.934, 0.488], #dose_response_subclinical
#               [0.013 / 60, 0.017 / 60], #breath
              
#               [17.533,0.296], # Caer1
#               [17.515,0.170], # Caer2
#               [19.364,0.348], # Caer3
#               [20.000,0.309], # Caer4
#               [1,2], # Vaer1
#               [2,3], # Vaer2
#               [3,6], # Vaer3
#               [6,10], # Vaer4             
#               [0.23, 0.53], #De1_2
#               [0.36, 0.62], #De2_3
#               [0.1, 0.62], #De3_6
#               [0.01, 0.29], #De6_10             
#               ], 
#      #'unif','triang','norm','lognorm',          
#     'dists':['lognorm', #dose_response_clinical
#              'lognorm', #dose_response_subclinical
#              'unif', #breath
             
#              'lognorm', #Caer1
#              'lognorm', #Caer2
#              'lognorm', #Caer3
#              'lognorm', #Caer4            
#              'unif', #Vaer1
#              'unif', #Vaer2
#              'unif', #Vaer3
#              'unif', #Vaer4             
#              'unif', #De1_2
#              'unif', #De2_3
#              'unif', #De3_6
#              'unif', #De6_10
#             ]
# }

# n = 100
# param_values_intvalues = saltelli.sample(dists_intvalues, n, calc_second_order= True)
# param_values_energy = saltelli.sample(dists_energy, n, calc_second_order= True)
# param_values_infection = saltelli.sample(dists_infection, n, calc_second_order= True)

# ###############################################################################
# ### Sensitivity Analysis and PLOTS 
# ### https://pynetlogo.readthedocs.io/en/latest/_docs/SALib_ipyparallel.html
# ###############################################################################


# Y_intvalues = SA_intvalues(torch.Tensor(param_values_intvalues))
# Si_intvalues = sobol.analyze(dists_intvalues, Y_intvalues.ravel(), calc_second_order= True,print_to_console=True)#False)
# problem = dists_intvalues
# sns.set_style('whitegrid')
# fig = plot_sobol_indices(Si_intvalues, criterion='ST', threshold=0.005)
# fig.set_size_inches(7,7)
# plt.show()


# Y_energy = SA_energy(torch.Tensor(param_values_energy))
# Si_energy = sobol.analyze(dists_energy, Y_energy.ravel(), calc_second_order= True,print_to_console=True)#False)
# problem = dists_energy
# sns.set_style('whitegrid')
# fig = plot_sobol_indices(Si_energy, criterion='ST', threshold=0.005)
# fig.set_size_inches(7,7)
# plt.show()


# Y_infection = SA_infection(torch.Tensor(param_values_infection))
# Si_infection = sobol.analyze(dists_infection, Y_infection.ravel(), calc_second_order= True,print_to_console=True)#False)
# problem = dists_infection
# sns.set_style('whitegrid')
# fig = plot_sobol_indices(Si_infection, criterion='ST', threshold=0.005)
# fig.set_size_inches(7,7)
# plt.show()

