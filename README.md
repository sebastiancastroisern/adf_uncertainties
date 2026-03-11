# ADF Uncertainties Calculations

This work aims at the calculation of the Cramer-Rao Bound over the reconstructed parameters made by ADF fitting, and then also run a MCMC algorithm on such reconstructions to better understand the precision of the method.


## Context

This work was motivated by the idea that the quantification of the uncertainties of the method were not done, and we explored different methods of doing so. The CRB bound should be a lower value on uncertainties, but happens also to be less precise than the MCMC calculations. 

You need **FIRST** to verify the parameters of the site (altitude, B field, etc) in the `params_config.py` file to make sure that you are working in the right conditions.

You then need to run `CRB.py` **before** running `MCMC.py`, in order to have the file needed for the MCMC part. You need these files to have a starting point close to the fitted value for the walkers. It give you also information about the fitted values and their CRB bounds. The `CRB.py` file is pretty quick to run, much quicker than the MCMC.


## Requirements 

You need 2 `.txt` file to be able to run the scripts. You first need `Rec_coinctable.txt` a file that has the informations about each coincidence of the set of EAS you decide to study, with the following columns: EventID | peak time [s] | peak amplitude [uV/m] | DU_ID. You then also need a `coord_antennas.txt` file to give you the coordinates of the antennas, with the following columns : EventID | x antenna [m] | y antenna [m] | z antenna [m] | DU_ID. [My pipeline](https://github.com/sebastiancastroisern/DC2_root2txt) can reconstruct those files frome DC2 datasets.

Modules required can be installed as such:
pip install -r requirement.txt


## Usage

python3 CRB.py --filepath path/to/txt/files
python3 MCMC.py --multi --filepath path/to/txt/files

(other options are available to you as arguments, although should not be needed in a first run of the algorithms)


## Parameters 

The main parameters to be modified for each file are as such. For the physical constants:
- `groundAltitude`
For the atmospheric model : 
- `ns` (air refraction index)
- `kr` (altitude induced `ns` variation rate)
For the magnetic field:
- `B_vec` (Declination of B field, azimuthal angle)
- `B_inc` (Inclination of B field, zenithal angle)
For the detector parameters:
- `jitter_time` (the precision on time of the antennas, in [s])
- `galactic_noise_floor` (the minimum uncertainty on the value of the E field at the antennas, same unit as the peak amplitude)
- `asym_coeff` (see ADF formula, Marion Guelfand and Valentin Decoene thesis)
For the algorithm in itself:
- `bounds` for all the 8 parameters


## Expected results

The code should generate you a dozen of `.npy` files (saved in `data_npy`) to be later used by the CRB calculations. Then produce `results_df.parquet` that stores the fitting results. The results are stored in the dataframe as, 
- for SWF '`recons_alpha`', '`recons_beta`', '`recons_rxmax`', '`recons_t0`' and `SWF_loss`,
- for ADF '`recons_theta`', '`recons_phi`', '`recons_delta_omega`', '`recons_amplitude`' and '`ADF_loss`' 
- for CRB : '`stds_alpha`', '`stds_beta`', '`stds_rxmax`', '`stds_t0`', '`stds_theta`', '`stds_phi`', '`stds_delta_omega`', '`stds_amplitude`'
- for energy calculations: '`recons_energy`', '`recons_energy_uncertainty`'
- for grammage calculations: '`recons_grammage`'
- for X_core reconstruction: '`x_core`', '`y_core`', '`z_core`' and '`dist_xcore`'
- `std_psi` the angular error based on CRB calculations
  In addition to the dataframe of the simulation data.

The `MCMC.py` can either give you back a file with all the samples and logprob values (`emcee_samples_res.h5`) if you chose to run it sequentially (to save up memory), or simply give back the mode of the distribution (`emcee_modes_res.h5`) if you decide to run it with multiprocessing. 
For the **modes** : An HDF5 file containing a mcmc_emcee_results group with one sub-group per coincidence (coinc_XXXXX), storing only the MCMC mode vector and lightweight metadata.
For the **samples** : An HDF5 file containing a mcmc_emcee_results group with one sub-group per coincidence (coinc_XXXXX), storing the full set of MCMC samples, the associated log-probabilities, and their dimensions.

## Run tests

To run the test on the files, please use the argument `--test` when running the files.

## CRB_old.py

This file is used to conduct an analysis on the old root extractor pipeline from Marion Guelfand [see her github](https://github.com/sebastiancastroisern/reconstruction_marion/tree/main/Data_processing). 

## Author 

Sebastian Castro-Isern