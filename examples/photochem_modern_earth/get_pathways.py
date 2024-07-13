from chempath import Chempath
import pandas as pd
import pathlib
from multiprocessing import Pool
from functools import partial
import numpy as np


OUTPUT_FOLDER = 'pathways'
INPUT_PATH = 'chempath_input'


def get_fmin(chempath):
    '''Gets minimum rates of pathways'''
    species_idxs = np.array([chempath.species_list.index(sp) for sp in 
            ['O2', 'CO', 'H2', 'CH4', 'O3']])
    possij = np.multiply(chempath.sij, chempath.sij>0)
    production_rates = np.dot(possij, chempath.fk)[species_idxs]
    fmin = min(production_rates)/1e4
    return fmin


def get_pathways_contributions(alt_idx, time_idx):
    '''Gets pathways contributions at an specific altitude and time index'''
    # get a chempath object
    chempath = Chempath(
        reactions_path=f'{INPUT_PATH}/{alt_idx}/reactions.txt',
        rates_path=f'{INPUT_PATH}/{alt_idx}/rates_{time_idx}.dat',
        species_path=f'{INPUT_PATH}/{alt_idx}/species.txt',
        conc_path=f'{INPUT_PATH}/{alt_idx}/num_densities_{time_idx}.dat',
        time_path=f'{INPUT_PATH}/{alt_idx}/time_{time_idx}.dat',
        transport_species = True
    )

    # ignore these species as branching-points
    chempath.ignored_sb +=\
        ['O2', 'H2O', 'O3', 'CH4', 'HV', 'N2', 'CO2', 'CO', 'H2', 'M']
    
    # set fmin
    f_min = get_fmin(chempath)
    chempath.f_min = f_min

    # find all pathways
    chempath.find_all_pathways()

    # get contributions
    alt = np.arange(0.5,100, 1)
    species_list = ['O2', 'O3', 'CH4', 'CO', 'H2']
    prod_dfs = []
    loss_dfs = []
    for species in species_list:
        prod = chempath.get_pathways_contributions(species, on='production')
        loss = chempath.get_pathways_contributions(species, on='loss')
        prod['species'] = species
        loss['species'] = species
        prod['time'] = chempath.mean_time
        loss['time'] = chempath.mean_time
        prod['alt'] = alt[alt_idx]
        loss['alt'] = alt[alt_idx]
        prod_dfs.append(prod)
        loss_dfs.append(loss)
    prod_dfs = pd.concat(prod_dfs)
    loss_dfs = pd.concat(loss_dfs)

    prod_dfs['time'] = prod_dfs['time']
    loss_dfs['time'] = loss_dfs['time']

    # save contributions
    outfolder = f'{INPUT_PATH}/{OUTPUT_FOLDER}/{time_idx}/{alt_idx}'
    pathlib.Path(outfolder).mkdir(exist_ok=True, parents=True)
    prod_dfs.to_csv(f'{outfolder}/prod_pathways.csv', index=0,
        compression='gzip')
    loss_dfs.to_csv(f'{outfolder}/loss_pathways.csv', index=0,
        compression='gzip')
    
    # save pathway chempath info
    # chempath.save_pathway_info(outfolder)

    return prod_dfs, loss_dfs


def get_all_contrib_dfs(time_idx):
    '''Runs get_pathways_contributions function in different processes'''
    alt_idxs = list(np.arange(0, 100))
    with Pool(processes=30) as pool:
        results = pool.map(partial(get_pathways_contributions,
            time_idx=time_idx), alt_idxs)
    
    prod, loss = zip(*results)
    prod = pd.concat(prod)
    loss = pd.concat(loss)

    prod.to_csv(f'{INPUT_PATH}/{OUTPUT_FOLDER}/{time_idx}/prod_pathways_profiles.csv',
        compression='gzip')
    loss.to_csv(f'{INPUT_PATH}/{OUTPUT_FOLDER}/{time_idx}/loss_pathways_profiles.csv',
        compression='gzip')


time_idx = 169
get_all_contrib_dfs(time_idx)