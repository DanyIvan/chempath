from photochem import EvoAtmosphere, zahnle_earth
import pathlib
import numpy as np
import pandas as pd

OUTPUT_FOLDER = 'photochem_output'

def get_reactants_idxs(species_names, reaction_equations):
    '''Gets species indexes of the reactants in each reaction
    Arguments:
        species_names (list): list of species names in reaction system
        reaction_equations (list): list of reaction equations in reaction system
    '''
    nj = len(reaction_equations)
    r1, r2 = [], []
    for j in range(nj):
        reaction = reaction_equations[j].replace(' ', '')
        # if species in reactants
        reactants = reaction.split('=')[0].split('+')
        r1.append(species_names.index(reactants[0]))
        r2.append(species_names.index(reactants[1]))
    r1 = np.array(r1)
    r2 = np.array(r2)
    return r1, r2

def run_model(output_path, o2_flux=2.5e11, tf=1e15, t_save=1e11):
    '''
    Run photochem model until model time reaches tf and save output after model
    time reaches t_save
    Arguments:
    o2_flux (float): O2 surface flux to use in the model
    tf (float): final time
    t_save (float): the model output will be saved after the model time reaches
        this time
    '''
    # reactions
    reaction_file = zahnle_earth
    # Various settings and boundary conditions
    settings_file = "ModernEarthFlux/settings.yaml"
    # Star
    star_file = "ModernEarthFlux/Sun_now.txt"
    # Initial conditions and eddy diffusion and temperature profile
    atmosphere_file = "ModernEarthFlux/atmosphere.txt"

    # initialize photochem object
    pc = EvoAtmosphere(reaction_file,\
                    settings_file,\
                    star_file,\
                    atmosphere_file)

    # decrease O2 flux
    pc.set_lower_bc('O2',bc_type='flux',flux=o2_flux)

    # get reactants indexes in each reaction
    reactions = [x.replace('=>', '=') for x in pc.dat.reaction_equations]
    r1,r2 = get_reactants_idxs(pc.dat.species_names, reactions) 

    num_densities = []
    reaction_rates = []
    rainout_rates = []
    time = []
    
    # run model until time tf
    pc.var.atol = 1e-21
    pc.initialize_stepper(pc.wrk.usol)
    tn = 0
    while tn < tf:
        tn = pc.step()
        if tn > t_save:
            # save number densities
            num_densities.append(pc.wrk.densities.astype(np.float128))
            densities = np.vstack([pc.wrk.densities, pc.wrk.densities[-1]])
            # calculate and save reaction rates
            rates = np.multiply(pc.wrk.rx_rates.T, densities[r1,:])
            rates = np.multiply(rates, densities[r2,:])
            reaction_rates.append(rates.astype(np.float128))
            # save rainout rates
            rainout_rates.append(np.transpose(pc.wrk.rainout_rates.astype(np.float128)))
            # save model time
            time.append(np.float128(tn))
    pc.destroy_stepper()

    num_densities = np.array(num_densities)
    # get rif of HV and M
    num_densities = num_densities[:, :-1, :]
    reaction_rates = np.array(reaction_rates)
    rainout_rates = np.array(rainout_rates)
    time = np.array(time).astype(np.float128)

    # save model output to files
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    num_densities.tofile(f'{output_path}/num_densities.dat')
    np.savetxt(f'{output_path}/num_densities.shape', num_densities.shape)
    reaction_rates.tofile(f'{output_path}/reaction_rates.dat')
    np.savetxt(f'{output_path}/reaction_rates.shape', reaction_rates.shape)
    rainout_rates.tofile(f'{output_path}/rainout_rates.dat')
    time.tofile(f'{output_path}/time.dat')
    np.savetxt(f'{output_path}/rainout_rates.shape', rainout_rates.shape)
    reactions = [x.replace('=>', '=') for x in pc.dat.reaction_equations]
    np.savetxt(f'{output_path}/reactions.txt', reactions,
        delimiter=" ", fmt="%s")
    np.savetxt(f'{output_path}/species.txt', pc.dat.species_names[:-2],
        delimiter=" ", fmt="%s")
    
def save_number_densities(path):
    # species to save data for
    species = ['O2', 'O3', 'CH4', 'CO', 'H2', 'OH', 'HO2', 'O', 'NO','NO2']
    # read species file
    ispec = np.loadtxt(f'{path}/species.txt', dtype=str, delimiter=',')

    # read number densities
    num_den_shape = np.loadtxt(f'{path}/num_densities.shape').astype(int)
    num_densities = np.fromfile(f'{path}/num_densities.dat',
        dtype=np.float128).reshape(num_den_shape)

    # read time
    times = np.fromfile(f'{path}/time.dat', dtype=np.float128)
    alts = np.arange(0.5, 100, 1)

    # get species indexes
    sp_idx = np.array([np.where(ispec == sp)[0][0] for sp in species])

    # create a dataframe with number densities
    data = []
    for i in range(0, len(times)):
        data_sp = pd.DataFrame(num_densities[i, sp_idx, :].T.astype(float),
            columns=species)
        data_sp['time'] = times[i]
        data_sp['alt'] = alts
        data.append(data_sp)
    data = pd.concat(data)

    # save data to csv file
    data.to_csv(f'{OUTPUT_FOLDER}/number_densities.csv', compression='gzip')

# run model for 5 my
yrs=60*60*24*365
run_model(OUTPUT_FOLDER, tf=5*yrs*1e6, t_save=0.1*yrs*1e6)

# save number densities as a csv file
save_number_densities(OUTPUT_FOLDER)