### Quantatitive model for evaluating risk trade-offs in Legionnaires' 
### Disease risk, energy cost, and scalding risk for hot water systems.
### Code by Ashley Heida, Mark Hamilton, and Kerry A. Hamilton.
###
### INITIALIZATION DOCUMENT
###
### Note: be sure to have a "results" folder created in your working directory
###
### Updated September 2, 2021

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

### SET WORKING DIRECTORY IF NEEDED   
#os.getcwd()
#path = "C:/Users/ashle/Dropbox/Ashley Heida/Water heater project/Scripts/"
#os.chdir(path)

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

states = []
heatmaps = []
for set_point in set_points:
   # Monte Carlo Distributions        
    initial_leg = LogNormal(6.6034,0.80388).sample([samples]).to(device) # CFU/L to initialize the system
    initial_c0 = Uniform(3.9e5, 7.8e9).sample([samples]).to(device)  # CFU/m^2, initial CFU in biofilm, (Schoen and Ashbolt, 2011; Thomas, 2012)
    biofilm_density = torch.Tensor([scipy.stats.weibull_min.rvs(c=3.14,scale=38.66,size=samples)*1000]).to(device) #kg/m3 -> g/m3 from multiplication by *1000
    env_temp = Uniform(20, 27).sample([samples]).to(device) # Degrees C
    main_line_temp = torch.cat([Uniform(17, 24).sample([samples // 2]), Uniform(16.5, 21.5).sample([samples - samples // 2])], dim=0)[torch.randperm(samples)].to(device)
    main_line_chl = Uniform(0.01,4).sample([samples]).to(device) # mg/L	(AWWA, 2018)													  
    shower_duration = torch.clamp_min(Normal(468, 72).sample([samples]).to(device), 0) # Seconds (DeOreo et al., 2016)
    sloughing_rate = LogNormal(-18.96,0.709).sample([samples]).to(device).div(0.0001) #g/cm^2 s -> g/m^2s  ### BIOFILM SLOUGHING DISTRIBUTION
    toc = Uniform(1,3).sample([samples]).to(device) # mg/L
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

    #initial_temp_setpt = initial_temperature[:,set_point-min(set_points)]        #MOVED TWO LINES DOWN
    initial_temps = torch.ones(total_nodes, 1, samples).to(device) #* env_temp #initial_temp_setpt  # Celsius
    initial_temps = initial_temperature[:,set_point-min(set_points)].unsqueeze(dim = 1)  # Celsius
    initial_temps = initial_temps.repeat(1, samples).unsqueeze(dim=1)
  
    # Chlorine decay calculation (ALREADY DEFINED ABOVE)
    k_chl_from_temp = -arrhenius_a * toc * math.exp(-arrhenius_b / (set_point + c_to_k))
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

    day = 86400
    initial_bfs[:num_heater_nodes + num_pipe_nodes] = initial_bf_pipe * np.exp(linear_grow_rate * day)
    initial_bfs[num_heater_nodes + num_pipe_nodes:] = initial_bf_branch #* np.exp(k_leg_env_temp * day) #Cap that growth!!
   
    initial_conds = torch.cat([initial_temps,
                               initial_chls,
                               initial_legs,
                               initial_sluffs,
                               initial_bfs], dim=1).to(device)

    start = time.time()
    print("solving eqn: set_point={}".format(set_point))
    all_results = odeint(model, initial_conds, times, rtol=1e-4, atol=1e-2)

    # Save heatmap medians
    heatmaps.append(torch.median(all_results, dim=-1)[0])
    state["heatmap"] = torch.median(all_results, dim=-1)[0]
    
    #Keep the params at the top node, end of rec pipe, and end of branch pipe
    all_results_purge = all_results[:,[num_heater_nodes-1, num_heater_nodes+num_pipe_nodes-1,-1],:,:]#[0,2,3],:]
    #Keep Temp, Leg (planktonic), Sloughed biofilm
    all_results_purge = all_results_purge[:,:,[0,2,3],:]
    state["all_results_purge"] = all_results_purge.detach().cpu().numpy()
    state["model"] = model
    states.append(state)

    ###############################################################################    		
    ### SAVE STATE FILE	
    ###############################################################################
    
    state_file = "results/case{}_statebio_{}.pkl".format(case, set_point)
    with open(state_file, "wb+") as f:
        torch.save(state, f)#, protocol=4)
        
    print("solved eq in {}s".format(time.time() - start))

##  SAVES MEDIAN HEATMAP INFO FOR ALL SET POINTS 
save_heatmap = "case{}_heatmaps.png".format(case)
torch.save(heatmaps, "results/{}".format(save_heatmap))


