import numpy as np
from photochem import EvoAtmosphere, zahnle_earth
import pathlib
from multiprocessing import Pool
    
INPUT_PATH = 'photochem_output'
OUTPUT_PATH = 'chempath_input'

# read photochem model output
num_den_shape = np.loadtxt(f'{INPUT_PATH}/num_densities.shape').astype(int)
num_densities = np.fromfile(f'{INPUT_PATH}/num_densities.dat',
    dtype=np.float128).reshape(num_den_shape)
react_shape = np.loadtxt(f'{INPUT_PATH}/reaction_rates.shape').astype(int)
reaction_rates = np.fromfile(f'{INPUT_PATH}/reaction_rates.dat',
    dtype=np.float128).reshape(react_shape)
rain_shape = np.loadtxt(f'{INPUT_PATH}/rainout_rates.shape').astype(int)
rainout_rates = np.fromfile(f'{INPUT_PATH}/rainout_rates.dat',
    dtype=np.float128).reshape(rain_shape)
times = np.fromfile(f'{INPUT_PATH}/time.dat', dtype=np.float128)
ispec = np.loadtxt(f'{INPUT_PATH}/species.txt', dtype=str, delimiter=',')
reactions = np.loadtxt(f'{INPUT_PATH}/reactions.txt', dtype=str, delimiter=',')

def get_sij(species_names, reaction_equations):
    '''Gets number of molecules/cm^3 or ppb of species i produced by reaction j in
    the reaction system
    Arguments:
        species_names (list): list of species names in reaction system
        reaction_equations (list): list of reaction equations in reaction system
    '''
    ni = len(species_names)
    nj = len(reaction_equations)
    # get sij matrix
    sij = np.zeros((ni, nj))
    for i in range(ni):
        for j in range(nj):
            reaction = reaction_equations[j].replace(' ', '')
            # if species in reactants
            reactants = reaction.split('=')[0].split('+')
            products = reaction.split('=')[1].split('+')
            if species_names[i] in reactants:
                # count of species in reactants
                n = reactants.count(species_names[i])
                # substract n
                sij[i][j] -=  n 
            if species_names[i] in products:
                # count of species in products
                n = products.count(species_names[i])
                # sum n
                sij[i][j] += n
    return sij

def photochem_to_cehmpath(output_path, layer=None):
    ''' Converts photochem model output to files readable by cehmpath
    Arguments:
        output_path (str): path where files used by chempath will be saved
        layer (int, optional): altitude index to convert
    '''
    # get sij matrix
    sij = get_sij(ispec, reactions)
    # altitudes
    alts = np.arange(0.5, 100, 1)
    if layer or layer != 0:
        # use only one layer
        alts = [alts[layer]]

    # for each altitude
    for j, alt in enumerate(alts):
        if layer or layer != 0:
            j = layer

        # get output at this altitude
        num_densities_alt = num_densities[:, :, j].astype(np.float128)
        reaction_rates_alt = reaction_rates[:,:,j].astype(np.float128)
        rainout_rates_alt = rainout_rates[:,j,:].astype(np.float128)

        # for each time
        for i in range(len(times)-1):
            print(i)
            pathlib.Path(f'{output_path}/{j}/').mkdir(exist_ok=True, parents=True)
            
            # get dt
            dt = times[i+1] - times[i]
    
            # get conc change
            num_den_change = num_densities_alt[i+1] - num_densities_alt[i]

            # get mean reaction rates
            mean_reaction_rate = (reaction_rates_alt[i+1] + reaction_rates_alt[i])/2

            # get rainout rates
            rainout_reactions = [f'{x} = {x}_rainout' for x in ispec]
            mean_rainout_rate = (rainout_rates_alt[i+1] + rainout_rates_alt[i])/2
            # append zeros so rainout has the same shape as rates
            mean_rainout_rate = np.concatenate([mean_rainout_rate, np.zeros(2)])

            # calculate transport
            dn_dt = num_den_change/dt
            chemprod = np.dot(sij, mean_reaction_rate)
            # chemprod[73:] = 0 # photochemical steady state for SL species
            # assuming that dn = (P-L)dt + Rdt + Phi*dt
            chemprod_and_rainout = chemprod - mean_rainout_rate
            transport_rates = (dn_dt -chemprod_and_rainout)
            # get transport reactions
            transport_reactions = [f'{sp}_transport = {sp}' for sp in ispec]

            # concatenate all reaction and rates
            all_reactions = np.concatenate([reactions,
                rainout_reactions, transport_reactions])
            all_rates = np.concatenate([mean_reaction_rate, 
                mean_rainout_rate, transport_rates])
            
            # save rates
            all_rates.tofile(f'{output_path}/{j}/rates_{i}.dat')
            # save model time
            np.array([times[i], times[i+1]]).\
                tofile(f'{output_path}/{j}/time_{i}.dat')
            # save number densities
            np.array([num_densities_alt[i], num_densities_alt[i+1]]).\
                tofile(f'{output_path}/{j}/num_densities_{i}.dat')
            
            # save reaction system and species names
            if i == 0:
                np.savetxt(f'{output_path}/{j}/reactions.txt', all_reactions,
                    fmt="%s", delimiter=',')
                np.savetxt(f'{output_path}/{j}/species.txt', ispec,
                    fmt="%s", delimiter=',')
                

# run each altutude in a different process
layers = np.arange(0,100, dtype=int)
def photochem2PAP_wrapper(layer):
    photochem_to_cehmpath(OUTPUT_PATH, layer=layer)
with Pool(processes=30) as pool:
    pool.map(photochem2PAP_wrapper, layers)
