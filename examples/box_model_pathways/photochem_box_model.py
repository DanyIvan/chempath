import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from rates import three_body_rate, weird_rates


class PhotochemBoxModel():
    ''' Simple photochemical box model that solves the equation dn_i = Pi - L_i
    '''
    def __init__(self, template_path) -> None:
        self.template_path = template_path
        self.reactions_dict = self.read_reactions()
        self.init_conc_list = self.read_inital_concentrations()
        self.species_list = [x['species'] for x in self.init_conc_list]
        self.set_reaction_lists()
        self.s_ij = get_s_ij(self.species_list, self.reactants, self.products)
        self.solution = {}

    def read_reactions(self):
        '''Reads reactions information input file'''
        with open(f'{self.template_path}/reactions.json') as f:
            reactions = json.load(f)
        return reactions

    def read_inital_concentrations(self):
        '''Reads species initial concentrations input file'''
        with open(f'{self.template_path}/initial_conentrations.json') as f:
            init_conc = json.load(f)
        return init_conc
    
    def set_reaction_lists(self):
        reactants = [x['reactants'] for x in self.reactions_dict]
        self.reactants = reactants
        products = [x['products'] for x in self.reactions_dict]
        self.products = products
        rate_params = [x['rate_params'] for x in self.reactions_dict]
        self.rate_params = rate_params
        reaction_types = [x['type'] for x in self.reactions_dict]
        self.reaction_types = reaction_types

        reactants_by_idx = []
        products_by_idx = []
        for reaction in self.reactions_dict:
            reactants = reaction['reactants']
            products = reaction['products']
            reactants_idxs = [self.species_list.index(sp) for sp in reactants]
            products_idxs = [self.species_list.index(sp) for sp in products]
            reactants_by_idx.append(reactants_idxs)
            products_by_idx.append(products_idxs)
        
        reactants_by_idx = reactants_by_idx
        products_by_idx = products_by_idx
        
        self.reactants_by_idx = reactants_by_idx
        self.products_by_idx = products_by_idx  
        self.reaction_equations = ['+'.join(x['reactants']) + '->' +\
                '+'.join(x['products']) for x in self.reactions_dict]

    def calculate_rates(self, current_conc, temp=217, n_air=1.82e18):
        ''' Calculates reaction rates at an assumed temperature of 217K and
        an air number density of 1.82e18. These numbers roughly correspond to
        20km in altitude
        '''
        rates = []
        for i in range(len(self.reactants_by_idx)):
            # get reactants concentrations
            r1, r2 = self.reactants_by_idx[i]
            n1, n2 = current_conc[r1], current_conc[r2]
            if self.reaction_types[i] == '2body':
                rate_constant = self.rate_params[i][0] * np.exp(self.rate_params[i][1]/temp) 
                rate = rate_constant * n1 * n2
            if self.reaction_types[i] == '3body':
                A0, AI, CN, CM = self.rate_params[i]
                rate_constant = three_body_rate(A0, AI, CN, CM, temp, n_air)
                rate = rate_constant * n1 * n2 
            # constant photochemical reactions for now
            if self.reaction_types[i] == 'photo':
                rate = self.rate_params[i][0] * n1
            if self.reaction_types[i] == 'weird':
                rate_constant = weird_rates(self.reactants[i], self.products[i], temp, n_air)
                rate = rate_constant * n1 * n2 
            # rates[i] = rate
            rates.append(rate)
        rates = np.array(rates)
        return rates

    def rhs(self, t, conc):
        '''Right hand side of equation'''
        rates = self.calculate_rates(conc)
        chempl = np.dot(self.s_ij, rates)

        return chempl  
    
    def solve_bdf(self, total_time, max_step=np.inf,
            rtol=1e-5, atol=1e-30, dense_output=True, t_eval=None):
        '''Solves ODE system'''

        # initial conditions
        time = [0, total_time]
        init_conc = [x['n'] for x in self.init_conc_list]
        
        sol = solve_ivp(self.rhs, time, init_conc, 'BDF', 
            t_eval = t_eval,
            dense_output=dense_output, max_step=max_step, rtol=rtol,
            atol=atol)
        
        if sol.success:
            # save solution
            time = sol.t
            conc = sol.y
            rates = np.transpose(np.array([self.calculate_rates(conc[:,i]) for i in 
                range(conc.shape[1])]))
            chempl = np.transpose(np.array([np.dot(self.s_ij, rates[:,i]) for i in 
                range(rates.shape[1])]))
            self.solution = Solution(time=time, conc=conc, rates=rates,
                chempl=chempl)
            
        else:
            print(sol.message)
            raise Exception('There was a problem integrating the ODEs')


    def plot_solution(self, savepdf=True):
        if not self.solution:
            raise Exception('Model must be integrated before plotting')
        fig, axs = plt.subplots(3,1, figsize=[8,12])

        def plot_conc(conc, time):
            for i in range(len(self.species_list)):
                plt.plot(time, conc[i, :], label= self.species_list[i])

        def plot_rates(rates, time):
            for i in range(rates.shape[0]):
                plt.plot(time, rates[i, :])
        
        plt.sca(axs[0])
        plot_conc(self.solution.conc, self.solution.time)
        plt.xlabel('Time (s)')
        plt.ylabel('Number density (molec/cm^3)')
        plt.yscale('log')


        plt.sca(axs[1])
        plot_rates(self.solution.rates, self.solution.time)
        plt.xlabel('Time (s)')
        plt.ylabel('Rate (molec/cm^2/s)')
        plt.yscale('log')
        fig.savefig('solution.pdf')

        plt.sca(axs[2])
        plot_conc(self.solution.chempl, self.solution.time)
        plt.xlabel('Time (s)')
        plt.ylabel('Production (molec/cm^2/s)')
        # plt.yscale('log')
        if savepdf:
            fig.savefig(f'{self.template_path}/solution.pdf')
            plt.close()
        else:
            return fig
        
    def solution_to_chempath_input(self, outpath, step=1, start=0):
        '''Saves the solution to files readable by chempath
        Arguments:
            outpath (str): path were files will be saved
            step (int): save files with this step between model times indexes
            start (int): time index to start saving files
        '''
        if not self.solution:
            raise Exception('Model must be integrated first')
    
        conc = self.solution.conc
        rates = self.solution.rates
        time = self.solution.time

        ntimes = len(time)
        # time idx in wich to save files
        idxs = np.arange(start,ntimes,step)
        
        # for each time idx
        for i, idx in enumerate(idxs):
            
            # save model time
            np.array([time[idx], time[idx+1]]).\
                tofile(f'{outpath}/time_{i}.dat') 
              
            # save species concentrations
            np.array([conc[:, idx], conc[:, idx+1]]).\
                tofile(f'{outpath}/num_densities_{i}.dat')

            # get mean reaction rate and save it
            mean_reaction_rate = (rates[:, idx+1] + rates[:, idx])/2
            mean_reaction_rate.tofile(f'{outpath}/rates_{i}.dat')

            # get reactions
            reactions = ['+'.join(x['reactants']) + '=' +\
                '+'.join(x['products']) for x in self.reactions_dict]

            # save reaction equations and species names
            if i == 0:
                np.savetxt(f'{outpath}/reactions.txt', reactions,
                    fmt="%s", delimiter=',')
                np.savetxt(f'{outpath}/species.txt', self.species_list,
                    fmt="%s", delimiter=',')      

    def print_reactions(self):
        for reaction in self.reactions_dict:
            print(' + '.join(reaction['reactants']) + ' --> ' +\
                ' + '.join(reaction['products']))
            

class Solution():
    '''Calss to store solution'''
    def __init__(self, time, conc, rates, chempl) -> None:
        self.time = time
        self.conc = conc
        self.rates = rates
        self.chempl = chempl


def get_s_ij(species_list, reactants, products):
    '''Gets number of molecules/cm^3 or ppb of species i produced by reaction j in
    the time dt'''
    sp = species_list
    ni = len(sp)
    nj = len(reactants)
    s_ij = np.zeros((ni, nj))
    for i in range(ni):
        for j in range(nj):
            # if species in reactants
            if sp[i] in reactants[j]:
                # count of species in reactants
                if sp[i].lower() == 'hv':
                    n = 0
                else:
                    n = reactants[j].count(sp[i])
                # substract rate
                s_ij[i][j] -=  n 
            if sp[i] in products[j]:
                # count of species in products
                n = products[j].count(sp[i])
                # substract rate
                s_ij[i][j] += n
    return s_ij





