# Water-Heater
Code for "Computational framework for evaluating risk trade-offs in costs associated with Legionnaires’ Disease risk, energy, and scalding risk for hospital hot water systems"

***Please note: This code was written to be run on the Agave computing cluster at Arizona State University and is currently being modified to be run on from standard Python.

The codes in this folder are written for the Quantatitive model for evaluating risk trade-offs in Legionnaires' Disease risk, energy cost, and scalding risk for hot water systems.

This code was written by Ashley Heida, Mark Hamilton, Alexis Mraz, Mark H. Weir and Kerry A. Hamilton.

Last updated September 24, 2021


# OVERVIEW OF FILES
In your working directory, create a folder called "results".

# Initialization.py
This is the first file to be run and will produce pickle files in the format "caseX_statebio_XX.pkl", "caseX_IntTemp.pkl", and "caseX_IntDecay.pkl". These will be saved in the "results" folder. This takes a few hours to run on my computing cluster in parallel (see "job.sh" and "caserecord" files below) so running this on a cluster is strongly recommend.


# Main.py
This file will use the files created by "Initialization_09_02_2021.py" to run the Monte Carlo risk assessment. This will produce all the line plots for total cost and risk from L. pneumophila and will circle the optimal temperatures. A Spearman sensitivity analysis will be conducted for all variables for all cases and plotted as a heatmmap. This file will also produce 3 tupes of files: 
1) "grouped_caseX.csv": returns 5th, 50th, and 95th percentile for infection cost, scalding cost, and energy cost. It also returns the 5th, 50th, and 95th percentile for total cost.
2) "riskill_caseX.csv": returns 5th, 50th, and 95th percentile for risk of illness.
3) "scald_caseX.csv": returns the number of the monte carlo iterations that are in the no injury category (col 1), the injusy categoy (col 2), and the necrosis category (col 3).
You can change the user variable here but they should also be changed in the initialization file. 

# sobol.py
A Sobol sensitivity analysis will be done using SALib (https://salib.readthedocs.io/en/latest/) If you change variables in the initialization file or the main file it should also be changed here.
 
# job.sh
This is the file I used to call "Initialization_09_02_2021.py" and "Main_09_02_2021.py" in  parallel in the computing cluster. This file calls "caserecord", a file that holds the variable "case" and runs each case in parallel. This file should be edited with the name of the file you want to run.


# caserecord
This file holds the variable "case". This variable determines which Case conditions are used for each iteration. Variations included insulation on the branching pipe, pipe length, water velocity and dose response parameter.
