import math
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import lsq_linear, linprog
from scipy import sparse
from string import Template
import errno
import os
import h5py
import signal
import warnings 
from joblib import Parallel, delayed
from functools import partial

class Chempath():
    '''
    Pathway analysis program class.
    Arguments:
        h5py_path (str): path of h5py data file containing input data
        f_min (float): minimum rate of pathways. Defaults to 0.0
        ignored_sb (list): List of species ignored as branching-points. These
            species will not be considered as branching-point species.
        n_processes (int): Number of processes to use to construct pathways. If
            n_processes > 1, pathways will be computed in parallel using
            multiprocessing
    '''
    def __init__(self, 
        h5py_path,
        f_min=0, 
        warnings=True, 
        transport_species = False,
        ignored_sb = [],
        n_processes = 1,
        delete_error_reactions = False
        ):

        with h5py.File(h5py_path, 'r') as datafile:
            self.f_min = f_min
            # ignore_warnings
            self.warnings = warnings
            # time 
            self.time = datafile['model_time'][:]
            self.dt = self.time[1] - self.time[0]
            # model time
            self.mean_time = np.mean(self.time)
            # species list
            species_list = datafile['species_names'][:]
            self.species_list = [x.decode("utf-8").strip() for x in species_list]
            # concentrations
            self.conc = datafile['num_densities'][:]
            # concentratio change
            self.dconc = self.conc[1] - self.conc[0]
            # mean concentration
            self.mean_conc = np.trapezoid(self.conc,self.time, axis=0) / self.dt
            # mean rate of concentration change
            self.mean_dconc = np.array(self.dconc) / self.dt 
            # reactions and rates
            reaction_equations = datafile['reaction_equations'][:]
            self.reaction_equations = [x.decode("utf-8").replace(' ', '') 
                for x in reaction_equations]
            self.rj = datafile['rates'][:]
            self.delete_zero_reactions()
            self.invert_negative_rates()
            # part of rate of reaction j associated with deleted pathways
            self.rj_del = np.zeros(len(self.reaction_equations), dtype=np.longdouble)
            # part of rate of reaction j associated with error pathways
            self.rj_err = np.zeros(len(self.reaction_equations), dtype=np.longdouble)
            # molecules of species i produces or destroyed by reaction j
            self.sij = get_sij(self.species_list, self.reaction_equations)
            # multiplicity of reaction j in pathway k
            self.xjk = np.diag(np.ones(len(self.reaction_equations), dtype=int))
            self.xjk = sparse.csc_matrix(self.xjk)
            # list of pathways pathways by unique id
            self.pathway_ids = xjk_to_id_list(self.xjk)
            # molecules of species i produces or destroyed by pathway k
            self.mik = sparse_dot(self.sij, self.xjk)
            # rates of pathways
            self.fk = self.rj
            # rate of production of species i by deleted pathways
            self.pi_del = np.zeros(len(self.species_list), dtype=np.longdouble)
            # rate of production of species i by error pathways
            self.pi_err = np.zeros(len(self.species_list), dtype=np.longdouble)
            # rate of destruction of species i by deleted pathways
            self.di_del = np.zeros(len(self.species_list), dtype=np.longdouble)
            # rate of destruction of species i by error pathways
            self.di_err = np.zeros(len(self.species_list), dtype=np.longdouble)
            # total rate of production of species i by all pathways
            self.pi = self.pi_del + np.dot(np.multiply(self.mik, self.mik>0), self.fk)
            # total rate of destruction of species i by all pathways
            self.di = self.di_del + np.dot(np.abs(np.multiply(self.mik, self.mik<0)), self.fk)
            # list of used banching species
            self.sb_list = []
            self.sb_order = {}
            self.ignored_sb = ignored_sb
            # number of processes to use to construct pathways
            self.n_processes = n_processes
            # full list of species including transport species
            self.transport_species = transport_species
            # temporary variables to store deleted pathway rates
            self.rj_del_temp = np.zeros(len(self.reaction_equations), dtype=np.longdouble)
            self.pi_del_temp = np.zeros(len(self.species_list), dtype=np.longdouble)
            self.di_del_temp = np.zeros(len(self.species_list), dtype=np.longdouble)
            if transport_species:
                transport_species = [f'{x}_transport' for x in self.species_list]
                self.full_species_list = self.species_list + transport_species
                self.full_sij = get_sij(self.full_species_list, self.reaction_equations)

            self.delete_error_reactions = delete_error_reactions
            if self.delete_error_reactions:
                self.del_error_reactions()


    def reinit(self):
        '''Re-initializes the chempath object with the input information'''
        self.__init__(
        reactions_path = self.reactions_path,
        rates_path = self.rates_path,
        species_path = self.species_path,
        conc_path = self.conc_path,
        time_path = self.time_path,
        f_min = self.f_min,
        warnings = self.warnings,
        dtype = self.dtype,
        transport_species = self.transport_species,
        ignored_sb = self.ignored_sb,
        n_processes = self.n_processes
        )

    def load_pathways_from_files(self, filespath, dtype=np.longdouble):
        '''Loads pathway info  saved using the save_pathway_info method
        Arguments:
            filespath(str): path of files to read
            dtype(type): data type of numbers if the files
        '''
        # multiplicity of reaction j in pathway k
        self.xjk = sparse.load_npz(f'{filespath}/sparse_xjk.npz')
        # list of pathways pathways by unique id
        self.pathway_ids = xjk_to_id_list(self.xjk)
        # molecules of species i produces or destroyed by pathway k
        self.mik = sparse_dot(self.sij, self.xjk)
        with h5py.File(f'{filespath}/chempath_info.hdf5', 'r') as datafile:
            # part of rate of reaction j deleted pathways
            self.rj_del = datafile['rj_del'][:]
            # rates of pathways
            self.fk = datafile['fk'][:]
            # rate of production of species i by deleted pathways
            self.pi_del = datafile['pi_del'][:]
            # rate of destruction of species i by deleted pathways
            self.di_del = datafile['di_del'][:]
            # rate of production of species i by error pathways
            self.pi_err = datafile['pi_err'][:]
            # rate of destruction of species i by error pathways
            self.di_err = datafile['di_err'][:]
            # total rate of production of species i by all pathways
            self.pi = datafile['pi'][:]
            # total rate of destruction of species i by all pathways
            self.di = datafile['di'][:]
            # list of used banching species
            self.sb_list = [x.decode("utf-8").strip() 
                for x in datafile['sb_list'][:]]
            # list of species not considered as brancing species 
            self.ignored_sb = [x.decode("utf-8").strip() 
                for x in datafile['ignored_sb'][:]]

        
    def get_sb(self, tau_max=None, min_conc=None):
        ''' Gets the next branching-point species
        Arguments:
            tau_max (float, Optional): maximum lifetime of branching point species. 
            Species with a lifetime higher than this will not be considered as 
            branching-point species.
            min_conc (float, Optional): minimum concentration of branching point 
            species.  Species with a concentration lower than this will not be 
            considered as branching-point species.
        '''
        # calculate lifetime. If di=0 warning is raised and the species is not
        # choose as branching pont.
        if not self.warnings:
            np.seterr(divide='ignore', invalid='ignore')
        tau = np.divide(self.mean_conc, self.di,
            out=np.zeros(len(self.dconc)), where= self.di!=0)
        df = pd.DataFrame({'mean_conc': self.mean_conc, 'di': self.di, 'tau': tau, 
            'species': self.species_list})
        
        # sort by lifetime
        df = df.sort_values(['tau'], ascending=[True])

        # filter out ignored species and species that have been branching species
        df = df[~df.species.isin(self.ignored_sb)]
        df = df[~df.species.isin(self.sb_list)]

        # filter out ignored species with lifetime>tau_max
        if tau_max:
            df = df[df.tau < tau_max]

        # filter out species with concentration < min_conc
        if min_conc:
            df = df[df.mean_conc > min_conc]

        sb_list = df.species.to_list()
        
        if len(sb_list) > 0:
            next_sb = sb_list[0]
            next_sb_idx = self.species_list.index(next_sb)
            self.sb_list.append(next_sb)
            self.sb_order[next_sb_idx] = len(self.sb_list) - 1
            return next_sb
        return None

    def delete_zero_reactions(self):
        '''Deletes reactions with a zero rate'''
        delete_idxs = np.where(self.rj == 0)[0]
        self.rj = np.delete(self.rj, delete_idxs)
        self.reaction_equations = list(np.delete(self.reaction_equations, delete_idxs))

    def invert_negative_rates(self):
        '''Inverts reactions with negative rates. We assumed that all rates are
        positive. Reactions with negative rates are inverted'''
        idxs = np.where(self.rj < 0)[0]
        for i in idxs:
            reaction = self.reaction_equations[i]
            # invert reaction
            inverted_reaction = '='.join(reaction.split('=')[::-1])
            self.reaction_equations[i] = inverted_reaction
        self.rj = np.abs(self.rj)

    def get_prod_destr_idxs(self, sb):
        ''' Gets the indexes of the pathways producing and destroying Sb
        '''
        self.sb_idx = self.species_list.index(sb)
        # find row of mik corresponding to sb
        mbk = self.mik[self.sb_idx, :]
        # find reactions (k) producing and destroying sb
        self.prod_idxs = np.where(mbk>0)[0]
        self.destr_idxs = np.where(mbk<0)[0]

    def recompute_pathway_dependent_variables(self):
        ''' Recomputes mik, pi and di '''
        self.mik = sparse_dot(self.sij, self.xjk)
        posmik = get_positive_values(self.mik)
        negmik = get_negative_values(self.mik)
        self.pi = self.pi_del + np.dot(posmik, self.fk)
        self.di = self.di_del + np.dot(np.abs(negmik), self.fk)
        
    def connect_pathways(self, idxs):
        '''Connects pathways producing and destroying Sb
        Arguments:
            idxs (tuple of arrays): indexes of the pathways producing and
            destroying Sb. idxs = (prod_idxs, destr_idxs)
        Returns tuple with:
            xjk_new: list of new pathways multiplicities
            fk_new: list of new pathways rates
            pid_new: list of new pathways ids
            fk_temp: array of already existing cumulative repeated pathways rates
            rj_del_temp: array of reaction rates associated with deleted pathways
            pi_del_temp: array of production of species by deleted pathways
            di_del_temp: array of destruction of species by deleted pathways
        '''
        sb_idx = self.sb_idx
        Db = np.max([self.di[sb_idx], self.pi[sb_idx]])
        
        # variables to store new pathways
        xjk_new = []
        fk_new = []
        pid_new = []

        # variables to store deleted pathway rates
        rj_del_temp = np.zeros(len(self.reaction_equations), dtype=np.longdouble)
        pi_del_temp = np.zeros(len(self.species_list), dtype=np.longdouble)
        di_del_temp = np.zeros(len(self.species_list), dtype=np.longdouble)
        fk_temp = np.zeros(len(self.fk), dtype=np.longdouble)

        # calculate new multiplicities so that sb is recycled
        # and calculate rates of new pathways
        prod_idxs, destr_idxs = idxs
        combinations = product(prod_idxs, destr_idxs)
        for p,d in combinations:
            xjn = np.abs(self.mik[sb_idx, d]) * self.xjk[:,p] +\
                self.mik[sb_idx, p] * self.xjk[:,d]
            # divide all multiplicities by greater common divisor
            gcd = np.gcd.reduce(xjn.data.astype(int))
            xjn = xjn / gcd
            xjn = xjn.astype(int)

            # calculate rates
            # rate has to be multiplied by gcd because pathways produce gcd
            # molecules of the species
            fn = np.multiply(gcd, np.divide(np.multiply(self.fk[p], self.fk[d]), Db))
            
            # of rate of pathway es lower than f_min, do not consider it and
            # update deleted pathway variables
            if fn < self.f_min:
                mi_n = np.squeeze(sparse_dot(self.sij, xjn).T)
                posmi_n = np.multiply(mi_n, mi_n>0)
                negmi_n = np.multiply(mi_n, mi_n<0)
                rj_del_temp = rj_del_temp + np.squeeze(xjn.toarray()) * fn
                pi_del_temp = pi_del_temp + posmi_n * fn
                di_del_temp = di_del_temp + np.abs(negmi_n) * fn
                continue

            # if pathway already exists, do not repeat it, just add its rate:
            # the same pathway can be formed through different combination order
            # of same reactions 
            pid_n = get_pathway_id(xjn)
            if pid_n in self.pathway_ids:
                idx = np.where(self.pathway_ids == pid_n)[0]
                fk_temp[idx] += fn
            elif pid_n in pid_new:
                idx = pid_new.index(pid_n)
                fk_new[idx] += fn
            else:
                xjk_new.append(xjn)
                fk_new.append(fn)
                pid_new.append(pid_n)

        return xjk_new, fk_new, pid_new, fk_temp, rj_del_temp, pi_del_temp, di_del_temp
    
    def form_new_pathways(self):
        '''Finds new pathways and appends their multiplicities and rates  to
        xjk and fk'''
        # for book keeping
        self.old_pathways_total_rates = 0
        self.new_pathways_total_rates = 0

        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0
        if new_pathways_flag: 
            
            # if multiprocessing and number of new pathways > 1000, find new
            # new pathways in parallel
            nprod = len(self.prod_idxs)
            ndestr = len(self.destr_idxs)
            n_combinations = nprod * ndestr
            if self.n_processes > 1 and n_combinations > 1000:
                if nprod > ndestr:
                    prod_idxs = np.array_split(self.prod_idxs, self.n_processes)
                    destr_idxs = [self.destr_idxs] * self.n_processes
                else:
                    prod_idxs = [self.prod_idxs] * self.n_processes
                    destr_idxs = np.array_split(self.destr_idxs, self.n_processes)
                idxs = list(zip(prod_idxs, destr_idxs))
                # find pathways in parallel
                results = Parallel(n_jobs=self.n_processes)(
                        delayed(self.connect_pathways)(i) for i in idxs)
                # collect results from multiple jobs
                xjk_new, fk_new, pid_new, fk_temp, rj_del_temp, pi_del_temp,\
                    di_del_temp = zip(*results)
                xjk_new = flatten_list(xjk_new)
                fk_new = flatten_list(fk_new)
                pid_new = flatten_list(pid_new)
                fk_temp = np.sum(fk_temp, axis=0)
                rj_del_temp = np.sum(rj_del_temp, axis=0)
                pi_del_temp = np.sum(pi_del_temp, axis=0)
                di_del_temp = np.sum(di_del_temp, axis=0)
            else:
                # find new pathways in a single process
                xjk_new, fk_new, pid_new, fk_temp, rj_del_temp, pi_del_temp,\
                    di_del_temp = self.connect_pathways((self.prod_idxs, self.destr_idxs))
            
            # update deleted pathway rates
            self.rj_del_temp = rj_del_temp
            self.pi_del_temp = pi_del_temp
            self.di_del_temp = di_del_temp

            # include new pathways
            if len(xjk_new) > 0:
                xjk_new = sparse.hstack(xjk_new)
            
                # append new pathways
                self.xjk = sparse.hstack([self.xjk, xjk_new])
                self.fk += fk_temp
                self.fk = np.concatenate([self.fk, fk_new])
                self.pathway_ids = np.append(self.pathway_ids, pid_new)   

                # for book keeping
                idxs = np.concatenate([self.prod_idxs,self.destr_idxs])
                self.old_pathways_total_rates =\
                    math.fsum(self.xjk[:,idxs].dot(self.fk[idxs]))
                self.new_pathways_total_rates = math.fsum(xjk_new.dot(fk_new))
            self.delete_duplicated_pathways()
   
    def delete_duplicated_pathways(self):
        '''Deletes duplicated pathways'''
        # find duplicated pathways
        u, c = np.unique(self.pathway_ids, return_counts=True)
        duplicates = u[c>1]

        # if there are duplicates...
        if len(duplicates) > 0:
            # sum rates of duplicates pathways and keep just one instance
            df = pd.DataFrame({'pid': self.pathway_ids, 'fk': self.fk,
                'idx': np.arange(0, len(self.fk))})
            groups = df.groupby('pid').groups
            group_idxs = list(groups.values())
            new_idxs = [x[0] for x in group_idxs]

            self.xjk = self.xjk[:, new_idxs]
            self.pathway_ids = self.pathway_ids[new_idxs]
            self.fk = np.array([np.sum(self.fk[idx]) for idx in group_idxs])

            self.recompute_pathway_dependent_variables()
            self.get_prod_destr_idxs(self.sb_list[-1])

    def calculate_rates_explaining_conc_change(self):
        '''Calculate part of the pathway rates that contribute to the 
        concentration change of Sb
        '''
        # for book keeping
        self.rates_not_explaining_dc_sb = 0

        # Calculate part of the rates that contribute to the concentration 
        # change of Sb
        sb_idx = self.sb_idx
        Db = np.max([self.di[sb_idx], self.pi[sb_idx]])
        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0

        if new_pathways_flag:
            if self.dconc[sb_idx] > 0:
                self.fk[self.prod_idxs] = np.divide(np.multiply(self.fk[self.prod_idxs],
                    self.mean_dconc[sb_idx]), Db)
            if self.dconc[sb_idx] < 0:
                self.fk[self.destr_idxs] = np.divide(np.multiply(self.fk[self.destr_idxs],
                    np.abs(self.mean_dconc[sb_idx])), Db)

            # for book keeping
            idxs = np.concatenate([self.prod_idxs,self.destr_idxs])
            rates_explaining_sb = math.fsum(self.xjk[:,idxs].dot(self.fk[idxs]))
            self.rates_not_explaining_dc_sb =\
                self.old_pathways_total_rates - rates_explaining_sb 
                

    def delete_old_pathways(self):
        ''' Delete old pathways after connection with all partners, except if 
        they contribute to explaining change in concentration of species Sb
        '''
        # for book keeping
        self.deleted_rates = 0    

        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0
        if new_pathways_flag:
            if self.dconc[self.sb_idx] > 0:
                delete_idxs = self.destr_idxs
            elif self.dconc[self.sb_idx] < 0:
                delete_idxs = self.prod_idxs
            else:
                delete_idxs = np.concatenate([self.prod_idxs, self.destr_idxs])
            
            #for book keeping
            self.deleted_rates = math.fsum(self.xjk[:,delete_idxs].dot(
                 self.fk[delete_idxs]))

            self.xjk = delete_columns_sparse(self.xjk, delete_idxs)
            self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)
            self.fk = np.delete(self.fk, delete_idxs)
            
            # redefine stuff depending on pathways
            self.recompute_pathway_dependent_variables()
    
    def calculate_deleted_pathways_effect(self):
        '''Calculates the  fraction of reaction rates associated with deleted 
            pathways'''
        sb_idx = self.sb_idx
        Db = np.max([self.di[sb_idx], self.pi[sb_idx]])
        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0
 
         # update deleted pathway variables
        self.connection_del_pathways_rates = 0
        self.connection_del_pathways_rates1 = 0
        if new_pathways_flag:
            # calculate fraction of rates associated with deleted pathways
            fdel_prod = np.divide(np.multiply(self.fk[self.prod_idxs],
                self.di_del[sb_idx]), Db)
            fdel_destr = np.divide(np.multiply(self.fk[self.destr_idxs],
            self.pi_del[sb_idx]), Db)
        
            connection_btwn_del_pathways = self.pi_del[sb_idx] * self.di_del[sb_idx] / Db
            self.connection_btwn_del_pathways = connection_btwn_del_pathways
            # connection with deleted pathways
            for i, p in enumerate(self.prod_idxs):
                posmik = np.multiply(self.mik[:,p], self.mik[:,p]>0)
                negmik =np.multiply(self.mik[:,p], self.mik[:,p]<0)
                self.rj_del = self.rj_del +\
                    np.squeeze((self.xjk[:,p] * fdel_prod[i]).toarray())
                self.connection_del_pathways_rates = math.fsum([
                    self.connection_del_pathways_rates,
                    np.sum(np.multiply(self.xjk[:,p], fdel_prod[i]))
                ])
                self.pi_del = self.pi_del + posmik * fdel_prod[i]
                self.di_del = self.di_del + np.abs(negmik) * fdel_prod[i]
      
            for i, d in enumerate(self.destr_idxs):
                posmik = np.multiply(self.mik[:,d], self.mik[:,d]>0)
                negmik =np.multiply(self.mik[:,d], self.mik[:,d]<0)
                self.rj_del = self.rj_del +\
                    np.squeeze((self.xjk[:,d] * fdel_destr[i]).toarray())
                self.connection_del_pathways_rates = math.fsum([
                    self.connection_del_pathways_rates,
                    np.sum(np.multiply(self.xjk[:,d], fdel_destr[i]))
                ])
                self.pi_del = self.pi_del + posmik * fdel_destr[i]
                self.di_del = self.di_del + np.abs(negmik) * fdel_destr[i]
                
            # # connection of deleted pathways between themselves
            # if self.dconc[sb_idx] > 0:
            #     self.pi_del[sb_idx] = self.pi_del[sb_idx] * self.mean_dconc[sb_idx] / Db
            #     self.di_del[sb_idx] = 0
            # elif self.dconc[sb_idx] < 0:
            #     self.pi_del[sb_idx] = 0
            #     self.di_del[sb_idx] = self.di_del[sb_idx] * np.abs(self.mean_dconc[sb_idx]) / Db
            
    def delete_insignificant_pathways(self):
        '''Deletes pathways with rates lower than self.fmin'''
        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0
        if new_pathways_flag:  
            # delete pathways with rate lower than fmin
            delete_idxs = np.where(self.fk < self.f_min)[0]
            for i in delete_idxs :
                posmik = np.multiply(self.mik[:, i], self.mik[:,i]>0)
                negmik =np.multiply(self.mik[:, i], self.mik[:, i]<0)
                self.rj_del = self.rj_del + np.squeeze(self.xjk[:,i].toarray()) * self.fk[i]
                self.pi_del = self.pi_del + posmik * self.fk[i]
                self.di_del = self.di_del + np.abs(negmik) * self.fk[i]
            
            # add deleted rates deleted during new pathway formation
            self.rj_del += self.rj_del_temp
            self.pi_del += self.pi_del_temp
            self.di_del += self.di_del_temp

            self.fk = np.delete(self.fk, delete_idxs)
            self.xjk = delete_columns_sparse(self.xjk, delete_idxs)
            self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)

            # redefine stuff depending on pathways
            self.recompute_pathway_dependent_variables()

    def del_error_reactions(self):
        '''Deletes error reactions'''
        # get indexes of error pathways
        error_payhway_idxs = []
        for i in range(self.xjk.shape[1]):
            pathway_str = self.get_pathway_str(self.xjk[:, i])
            if 'err' in pathway_str:
                error_payhway_idxs.append(i)

        # delete error reactions
        delete_idxs = error_payhway_idxs
        for i in delete_idxs :
            posmik = np.multiply(self.mik[:, i], self.mik[:,i]>0)
            negmik =np.multiply(self.mik[:, i], self.mik[:, i]<0)
            self.rj_del = self.rj_del + np.squeeze(self.xjk[:,i].toarray()) * self.fk[i]
            self.pi_del = self.pi_del + posmik * self.fk[i]
            self.di_del = self.di_del + np.abs(negmik) * self.fk[i]

            self.rj_err = self.rj_err + np.squeeze(self.xjk[:,i].toarray()) * self.fk[i]
            self.pi_err = self.pi_err + posmik * self.fk[i]
            self.di_err = self.di_err + np.abs(negmik) * self.fk[i]
    

        self.fk = np.delete(self.fk, delete_idxs)
        self.xjk = delete_columns_sparse(self.xjk, delete_idxs)
        self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)

        # redefine stuff depending on pathways
        self.recompute_pathway_dependent_variables()
        
        
    def print_book_keeping_variables(self):
        '''Prints variables useful to keep track of rates during the formation
        of new pathways'''
        print('-----------------------')
        print(f'old pathways rates: {self.old_pathways_total_rates}')
        # new_pathways_rates = math.fsum([self.new_pathways_total_rates,
        #     self.connection_del_pathways_rates])
        print(f'connection_del_pathways_rates: {self.connection_del_pathways_rates}')
        print(f'connection_del_pathways_rates1: {self.connection_del_pathways_rates1}')
        print(f'new pathways rates: {-self.new_pathways_total_rates}')
        total_deleted = math.fsum([self.rates_not_explaining_dc_sb,
            self.deleted_rates])
        print(f'deleted rates: {total_deleted}')
        old_minus_new = math.fsum([self.old_pathways_total_rates, 
            -self.new_pathways_total_rates])
        new_minus_deleted = math.fsum([self.old_pathways_total_rates,
            -total_deleted])
        print(f'old rates - new rates: {old_minus_new}')
        print(f'old rates - delted rates: {new_minus_deleted}')

        rates_pathways = self.xjk.dot(self.fk) + self.rj_del
        print('------------------------------------')
        print(f'total reaction rates: {math.fsum(self.rj)}')
        print(f'total pathways rates: {np.sum(rates_pathways)}')
        print(f'difference: {math.fsum(self.rj) - np.sum(rates_pathways)}')
        print(f'division: {math.fsum(self.rj)/np.sum(rates_pathways)}')


    def split_into_subpathways(self, pathway_ids, method='lsq_linear'):
        '''Finds the subpathways of the pathways within pathway_ids
        Arguments:
            pathway_ids (list): ids of pathways to be splitted
            method (str): one of 'lsq_linear' or 'lehmann'. This option
            determines the method to use to split pathways into subpathways
        Returns tuple with:
            xjk_elem_list: list of new subpathways multiplicities
            fk_elem_list: list of new subpathways rates 
            delete_idxs: list of indexes of splitted pathways
            fk_temp: array of cumulative rates of repeated pathways
        '''
        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0

        # variables to store subpathways
        xjk_elem_list = []
        fk_elem_list = []
        pid_elem_list = []
        delete_idxs = []
        fk_temp = np.zeros(len(self.fk), dtype=np.longdouble)

        if new_pathways_flag:
            # for each pathway...
            for p_id in pathway_ids:
                p_index = np.where(self.pathway_ids == p_id)[0]
                p = self.xjk[:, p_index].T
                f = self.fk[p_index]

                # find elementary subpathways
                xjk_elem = self.find_elementary_pathways(p, p_index)
                
                # if a pathway does not have subpathways do nothing
                if xjk_elem.shape[1] == 1:
                    continue
                
                p = np.squeeze(p.toarray())
                a = xjk_elem
                b = p

                if method == 'lsq_linear':
                    # solve system of equations ax = b 
                    x = solve_system_eq(a, b)
                elif method == 'lehmann':
                    # get elementary pathways ids
                    pid_elem = np.array([get_pathway_id_dense(xjk_elem[:, i])
                        for i in range(xjk_elem.shape[1])])
                    # get number of elementary pathways
                    num_elem = len(pid_elem)
                    # check if elementary  pathways already exists
                    is_old_pathway = [False] * num_elem

                    # get rates of already existing pathways, and set the rate
                    # equals zero if the pathway does not exists
                    rates = np.zeros(num_elem)
                    for i, pid in enumerate(pid_elem):
                        if pid in  self.pathway_ids:
                            idx = np.where(self.pathway_ids == pid)[0][0]
                            rates[i] = self.fk[idx]
                            is_old_pathway[i] = True
                        else:
                            is_old_pathway[i] = False

                    # rank subpathways by their rate. Higher rate = lower rank
                    order1 = rates.argsort()[::-1]
                    rank1 = order1.argsort()
                    num_old_pathways = len(np.where(is_old_pathway)[0])

                    # if rate = 0, rank subpathways by the sum of their
                    # multiplicities
                    order2 = np.argsort(xjk_elem.sum(axis=0))
                    rank2 = order2.argsort()

                    # if two pathways have the same rate, rank them by their
                    # simplicity
                    u,c = np.unique(rates, return_counts=1)
                    repeated_rates = u[c>1]   
                    for rate in repeated_rates:
                        idxs = np.where(rates == rate)[0]
                        rank1_ = rank1[idxs]
                        rank2_ = rank2[idxs]
                        rank2_ = rank2_.argsort().argsort()
                        new_rank = min(rank1_) + rank2_
                        rank1[idxs] = new_rank    

                    # combine the two ranks
                    rank2 = rank2 + num_old_pathways
                    rank =  np.array([rank1[i] if is_old_pathway[i] else 
                        rank2[i] for i in range(num_elem)])
                    
                    # minimize the function  rank**2 dot x, subject to the
                    # constraint ax = b
                    x = solve_system_eq_rank(a, b, rank)   
                else:
                    raise Exception('method must be one of lsq_linear, lehmann')
                # if solution is not exact, do nothing
                if not np.all(np.isclose(np.dot(a,x), b)):
                    # print(np.dot(a,x) - b)
                    continue
                
                # distribute rate to subpathways
                fk_elem = f * x

                # append new subpathways
                delete_idxs.append(p_index)
                for i in range(len(fk_elem)):
                    pid_elem = get_pathway_id_dense(xjk_elem[:,i])
                    # if subpathway already exists, just add its rate
                    if pid_elem in self.pathway_ids:
                        idx = np.where(self.pathway_ids == pid_elem)[0]
                        fk_temp[idx] += fk_elem[i]
                    elif pid_elem in pid_elem_list:
                        idx = pid_elem_list.index(pid_elem)
                        fk_elem_list[idx] += fk_elem[i]
                    else:
                        pid_elem_list.append(pid_elem)
                        fk_elem_list.append(fk_elem[i])
                        xjk_elem_list.append(np.c_[xjk_elem[:,i]])

        return xjk_elem_list, fk_elem_list, delete_idxs, fk_temp
        
    def split_pathways(self, method='lsq_linear'):
        '''Splits pathways into elementary subpathways
        '''
        # If multiprocessing and number of pathways > 1000, split pathways in 
        # parallel
        num_pathways = len(self.pathway_ids)
        if self.n_processes > 1 and num_pathways > 1000:
            # split pathways in parallel
            results = Parallel(n_jobs=self.n_processes)(delayed(
                    partial(self.split_into_subpathways, method=method))(i) 
                    for i in np.array_split(self.pathway_ids, self.n_processes))
            # collect results form multiple jobs
            if results:
                xjk_elem, fk_elem, delete_idxs, fk_temp = zip(*results)

                if len(xjk_elem) > 0:
                    xjk_elem = [x for x in xjk_elem if len(x)>0]
                    fk_elem = [x for x in fk_elem if len(x)>0]
                    delete_idxs = [x for x in delete_idxs if len(x)>0]
                    fk_temp = np.sum(fk_temp, axis=0)

                    if len(xjk_elem) > 0:  
                        xjk_elem = np.concatenate(xjk_elem)
                        fk_elem = np.concatenate(fk_elem)
                    if len(delete_idxs) > 0:
                        delete_idxs = np.concatenate(delete_idxs)
            else:
                xjk_elem, fk_elem, delete_idxs = [], [], []
        else:
            # split pathways in a single process
            xjk_elem, fk_elem, delete_idxs, fk_temp =\
                self.split_into_subpathways(self.pathway_ids, method=method)
        
        # stack new subpathways
        if len(xjk_elem) > 0: 
            xjk_elem = np.hstack(xjk_elem)
        # add rates from repeated subpathways
        self.fk += fk_temp

        # Delete splitted pathways
        if len(delete_idxs) > 0:
            self.xjk = delete_columns_sparse(self.xjk, delete_idxs)
            self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)
            self.fk = np.delete(self.fk, delete_idxs)
        
        # append new subpathways
        if len(xjk_elem) > 0:
            xjk_elem = sparse.csc_matrix(xjk_elem)
            pid_elem = xjk_to_id_list(xjk_elem)
            
            self.xjk = sparse.hstack([self.xjk, xjk_elem])
            self.fk = np.concatenate([self.fk, fk_elem])
            self.pathway_ids = np.concatenate([self.pathway_ids, pid_elem])  
           
        # delete repeated pathways
        self.delete_duplicated_pathways()
        self.recompute_pathway_dependent_variables()
            
    def find_elementary_pathways(self, xjc, pathway_index):
        ''' Finds the elementary pathways of a pathway
        Arguments:
            xjc (numpy array): multiplicities of pathway to be split
            pathway_index (int): index of pathway to be split
        '''
        is_steady_state_pathway = self.is_steady_sate_pathway(xjc)
        # if pathway is not in steady state, enforce steady state by adding
        # pseudo reactions to the reaction system
        if not is_steady_state_pathway:
            new_reactions, new_mik = [], []
            for sb in self.sb_list:
                sb_idx = self.species_list.index(sb)
                if self.mik[sb_idx, pathway_index] != 0:
                    new_mik.append(np.abs(self.mik[sb_idx, pathway_index][0]))
                # id dc_sb>0 add pseudo-reaction destroying sb
                if self.mik[sb_idx, pathway_index] > 0:
                    new_reactions.append(f'{sb}=...')
                # id dc_sb<0 add pseudo-reaction producing sb
                elif self.mik[sb_idx, pathway_index] < 0:
                    new_reactions.append(f'...={sb}')
            reactions = np.append(self.reaction_equations, new_reactions)
            sij = get_sij(self.species_list, reactions)
            new_mik = sparse.csc_matrix(new_mik)
            xjc = sparse.hstack([xjc, new_mik])
        else:
            sij = self.sij
            reactions = self.reaction_equations

        # find reactions in pathway
        rxns = xjc.nonzero()[1]

        # initialize subpathways
        xjk_sub = []
        for  i in rxns:
            pi = np.zeros(len(reactions))
            # pi[i] = p[i]
            pi[i] = 1   
            xjk_sub.append(np.c_[pi])
        xjk_sub = np.concatenate(xjk_sub, axis=1)
        # init mik
        mik_sub = np.dot(sij, xjk_sub)

        # found subpathways
        for sb in self.sb_list:
            xjk_sub_new = []
            sb_idx = self.species_list.index(sb)
            mbk = mik_sub[sb_idx, :]

            # copy pathways with zero production to  xjk_sub_new
            zero_prod = np.where(mbk ==0)[0]
            for i in zero_prod:
                xjk_sub_new.append(np.c_[xjk_sub[:,i]])

            # find reactions producing and destroying sb
            prod = np.where(mbk>0)[0]
            destr = np.where(mbk<0)[0]
            # find all combinations of pathways producing and destroying sb
            combinations = product(prod, destr)

            # combine producing and consuming pathways
            for p,d in combinations:
                xjn = np.abs(mik_sub[sb_idx, d]) * xjk_sub[:,p] +\
                    mik_sub[sb_idx, p] * xjk_sub[:,d]
                # divide all multiplicities by greater common divisor
                gcd = np.gcd.reduce(xjn.astype(int))
                xjn = xjn / gcd
                xjn = xjn.astype(int)
                
                if self.is_elementary_pathway(xjk_sub, p, d):
                    xjk_sub_new.append(np.c_[xjn])
                
            if len(xjk_sub_new) > 0:
                # update xjk_sub and mik_sub
                xjk_sub = np.concatenate(xjk_sub_new, axis=1)
                mik_sub = np.dot(sij, xjk_sub)

        if not is_steady_state_pathway:
            xjk_sub = xjk_sub[:len(self.reaction_equations), :]
        return xjk_sub

    def is_elementary_pathway(self, xjk_sub, p, d):
        '''Checks if new pathway formed combining patwhays with indexes p and d
        is elemnetary in the sense that there is no pathway in xjk_sub which 
        reactions are a subset of the reactions of the new formed
        pathway
        Arguments:
            xjk_sub (numpy 2d array): multiplicities of subpathways
            p (int): index of pathway producing Sb
            d (int): index of pathway destroying Sb
        '''
        # reactions in p and d
        rxns_p = np.where(xjk_sub[:,p] != 0)[0]
        rxns_d = np.where(xjk_sub[:,d] != 0)[0]
        rxns_pd = np.union1d(rxns_p, rxns_d)
        
        is_subset = lambda x, y: math.fsum(np.isin(x, y)) == len(x)
        for m in range(xjk_sub.shape[1]):
            if m not in [p,d]:
                rxns_m = np.where(xjk_sub[:,m] != 0)[0]
                # check if rxns_m is subset of rxns_pd
                if is_subset(rxns_m, rxns_pd):
                    return False
        return True
    
    def is_steady_sate_pathway(self, xjn):
        '''Checks if a pathway is in steady state, meaning that it does not 
            produce or destroy the branching species
        Arguments:
            xjn (numpy 1d array): multiplicities of pathway
            '''
        sb_list = self.sb_list
        sb_idxs = [self.species_list.index(x) for x in sb_list]

        for idx in sb_idxs:
            if xjn.dot(self.sij[idx,:]) != 0:
                return False
        return True
    
    def find_new_pathways(self, sb, verbose=False, split_pathways=True, method='lsq_linear'):
        '''Finds new pathways trough the branching-point species sb
        Arguments:
            sb (str): branching-points species
        '''
        self.get_prod_destr_idxs(sb)
        if verbose:
            print('---------------------')
            print(sb)
            print(self.prod_idxs)
            print(self.destr_idxs)
        self.form_new_pathways()
        self.calculate_deleted_pathways_effect()
        self.calculate_rates_explaining_conc_change()
        self.delete_old_pathways()
        self.delete_insignificant_pathways()
        if split_pathways:
            self.split_pathways(method=method)
        self.check_rate_distribution()
        if verbose:
            self.print_book_keeping_variables()

    def find_all_pathways(self, tau_max=None, min_conc=None, timeout=0, 
            verbose=False, method='lsq_linear'):
        ''' Finds all pathways in the system
        Arguments:
            tau_max (float, optional): max lifetime of branching-point species
            min_conc (float, optional): min concentration of branching-point species
            timeout (int): time in seconds to wait for the algorithm to finish. If
                this is equal 0 there is no timeout.
            verbose (bool): if True the book keeping variables are printed in
                each iteration
        '''
        def _handle_timeout(signum, frame):
            raise TimeoutError(os.strerror(errno.ETIME))
        
        self.check_mass_conservation()
        sb = self.get_sb(tau_max=tau_max, min_conc=min_conc, )

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout)
        try:
            while sb:              
                self.find_new_pathways(sb, verbose=verbose, method=method)
                sb = self.get_sb(tau_max=tau_max, min_conc=min_conc)
        except Exception as e:
            if type(e)==TimeoutError:
                return 'timed out'
            else:
                raise e
        finally:
            signal.alarm(0)
     
    def get_pathways_contributions(self, sp, on='loss', format='txt'):
        '''
        Calculate the contribution of all pathways to the loss or production 
        of a species
        Arguments:
            sp(str): species to calculate contributions for
            on(str): can be 'loss' or 'production'
            format(str): format of pathway strings. Can be 'txt', 'html' and 
            'latex'
        Returns:
            contrib_df: pandas dataframe with contributions    
        '''
        # calculate number of molecules of sp produced by each pathway
        sp_idx = self.species_list.index(sp)
        sp_mik = self.mik[sp_idx, :] 

        prod_idxs = np.where(sp_mik>0)[0]
        destr_idxs = np.where(sp_mik<0)[0]   
        # calculate rates of production or destruction of sp by specific pathways
        production = np.abs(sp_mik) * self.fk 
        # calculate contributions
        if on == 'loss':
            idxs = destr_idxs
            deleted_pathways_prod = self.di_del[sp_idx]
            total_production = self.di[sp_idx]
            err_pathways_prod = self.di_err[sp_idx]
        elif on == 'production':
            idxs = prod_idxs
            deleted_pathways_prod = self.pi_del[sp_idx]
            total_production = self.pi[sp_idx]
            err_pathways_prod = self.pi_err[sp_idx]

        contrib = production[idxs] / total_production
        pathways = [self.xjk[:, i] for i in idxs]
        rates = self.fk[idxs]
        total_prod = np.multiply(self.mik[sp_idx, idxs], self.fk[idxs])*self.dt
        p_ids = self.pathway_ids[idxs]
        p_strs = [self.get_pathway_str(p, format=format) for p in pathways]
        
        # make a pandas dataframe with the info
        contrib_dict = {'pathway_id': p_ids, 'pathway': p_strs,
            'contribution': contrib, 'rate':rates,
            'total_prod': total_prod}
        contrib_df = pd.DataFrame(contrib_dict)

        deleted_contrib = pd. DataFrame({'pathway_id': ['del'], 
                'pathway': ['deleted_pathways'],
                'contribution': [deleted_pathways_prod/total_production], 
                'rate':[deleted_pathways_prod],
                'total_prod': deleted_pathways_prod * self.dt})
        
        err_contrib = pd. DataFrame({'pathway_id': ['err'], 
                'pathway': ['err_pathways'],
                'contribution': [err_pathways_prod/total_production], 
                'rate':[err_pathways_prod],
                'total_prod': err_pathways_prod * self.dt})

        contrib_df = pd.concat([contrib_df, deleted_contrib, err_contrib])
        contrib_df['dconc'] = self.dconc[sp_idx]
        contrib_df.sort_values('contribution', ascending=False, inplace=True)
        contrib_df = contrib_df.reset_index(drop=True)
        contrib_df=contrib_df.astype({'contribution':np.float64, 'rate':np.float64,
            'total_prod':np.float64})
        return contrib_df        
       
    def check_mass_conservation(self, min_concentration=1.0, atol=1e-3,
            rtol=1e-3):
        '''
        Checks that concentration changes are balanced by the reactions
        Arguments:
            min_concentration (flaot): minimum concentration consiudered as 
                important. Defaults to 1 molec/cm^3. Species with concentration
                changes lower that this will use rtol to check the balance, and
                other species will use atol.
            atol (flat): absoloute tolerance
            rtol (float): relative tolerance
        '''
        # find species with concentration changes greater that min
        dconc_gt = np.where(np.abs(self.dconc) > min_concentration)[0]
        # find species with concentration changes lower that min
        dconc_lt = np.where(np.abs(self.dconc) < min_concentration)[0]

        # calculate production - destruction
        chemprod = np.dot(self.sij, self.rj)

        # check balance and diplay warning if unbalanced
        for i in dconc_gt:
            if not np.isclose(self.dconc[i]/self.dt, chemprod[i], rtol=rtol):
                msg = f'{self.species_list[i]} concentration change not balanced' +\
                f' by reactions. Concentartion change: {self.dconc[i]/self.dt}, production' +\
                f' by reactions: {chemprod[i]}'
                warnings.warn(msg)

        for i in dconc_lt:
            if not np.isclose(self.dconc[i]/self.dt, chemprod[i], atol=atol):
                msg = f'{self.species_list[i]} concentration change not balanced' +\
                f'by reactions. Concentartion change: {self.dconc[i]/self.dt}, production' +\
                f' by reactions: {chemprod[i]}'
                warnings.warn(msg)

    def check_rate_distribution(self):
        ''' Checks in reaction rates are completely distributed to the pathway's
        rates. If not, raises a warning
        '''
        total_rates = math.fsum(self.rj)
        total_pathway_rates = np.sum(self.rj_del + self.xjk.dot(self.fk))
        rate_conservation = np.isclose(total_rates, total_pathway_rates)
        if not rate_conservation:
            warnings.warn('Rates are not correctly distributed!')
            print(f'reaction rates:{total_rates}')
            print(f'pathway rates:{total_pathway_rates}')

    def get_pathways_explained_change(self):
        '''Gets dataframe with the fraction of concentration changes explained by
        pathways'''
        rates_pathways = self.xjk.dot(self.fk)
        total_prod = np.dot(self.sij, rates_pathways)
        explained_change = total_prod / self.mean_dconc 
        explained_change = {k:[v] for k,v in zip(self.species_list, explained_change)}
        explained_change = pd.DataFrame(explained_change)
        return explained_change 

    def get_deleted_pathways_explained_change(self):
        '''Gets dataframe with the fraction of concentration changes explained by
        deleted pathways'''
        total_prod = np.dot(self.sij, self.rj_del) * self.dt
        explained_change = total_prod / self.dconc 
        explained_change = {k:[v] for k,v in zip(self.species_list, explained_change)}
        explained_change = pd.DataFrame(explained_change)
        return explained_change 
    
    def get_total_pathway_rates(self):
        '''Gets the total pathways rates'''
        total_pathway_rates = math.fsum(self.rj_del + self.xjk.dot(self.fk))
        return total_pathway_rates
    
    def order_reactions(self, reaction_idxs, type='loss'):
        if len(reaction_idxs) < 3:
            return reaction_idxs
        order = []
        prods = {idx: self.reaction_equations[idx].split('=')[1].split('+')
            for idx in reaction_idxs}
        reacts = {idx: self.reaction_equations[idx].split('=')[0].split('+')
            for idx in reaction_idxs}
        
        if type == 'loss':
            in_ = reacts
            out_ = prods
        else:
            in_ = prods
            out_ = reacts
        
        # find first reaction(s)
        for idx in reacts.keys():
            for sp in self.ignored_sb:
                if sp.upper() not in ['HV', 'M'] and sp in in_[idx]:
                    if idx not in order:
                        order.append(idx)


        def get_next_reaction(curr_react_idx):
            curr_react_out = out_[curr_react_idx]
            
            next_reactions = []
            for idx in in_.keys():
                if idx not in order:
                    if in_[idx] == curr_react_out:
                        next_reactions.append(idx)
                        return np.unique(next_reactions)
                    
                    for prod in curr_react_out:
                        if prod in in_[idx]:
                            next_reactions.append(idx)
            return np.unique(next_reactions)
        
        while True:
            next_reactions = get_next_reaction(order[-1])
            if len(next_reactions) == 0:
                break
            else:
                order = np.concatenate([order, next_reactions])
        
        for idx in reaction_idxs:
            if idx not in order:
                order = np.append(order, idx)
        return order

    def get_pathway_str(self, xjc, format='txt', include_net_reaction=True):
        ''' Gets the string of a pathway given its multiplicities.
        Arguments:
            xjc (numpy 1d array): multiplicities of pathway
            format(str): format of the pathway string. Can be 'txt', 'latex' or
                'html
            include_net_reaction(bool): If true includes the net reaction in 
                the pathway string
        Returns:
            pathway string (str)
        '''
        react_string = ''
        idxs = xjc.nonzero()[0]
        coeffs = xjc.data
        coeffs_dict = {k:v for k,v in zip(idxs, coeffs)}

        if format == 'txt':
            format_react = format_react_txt
            lnbrk = '\n'
        elif format == 'latex':
            format_react = format_react_latex
            lnbrk = '\\\\'
        elif format == 'html':
            lnbrk = '<br>'
            format_react = format_react_txt

        for i, idx in enumerate(idxs):
            coeff = coeffs_dict[idx]
            # reacts, prods = self.reaction_equations[idx]['reacts'], self.reaction_equations[idx]['prods']
            reacts = self.reaction_equations[idx].split('=')[0].split('+')
            prods = self.reaction_equations[idx].split('=')[1].split('+')
            if coeff != 1:
                coeff = int(coeff) if coeff==int(coeff) else coeff
                react_string += f'{coeff}({format_react(reacts, prods)})'  +lnbrk
            else:
                react_string += f'{format_react(reacts, prods)}' + lnbrk

        # add net reaction
        if include_net_reaction:
            net_reaction = self.get_net_reaction(xjc)
            if format == 'txt':
                net_str = 'Net:'
            elif format == 'latex':
                net_str = '\\text{Net:} '
            react_string += net_str + net_reaction

        if format == 'latex':
            react_string = "\ce{ %s }" % react_string
        return react_string

    def get_net_reaction(self, xjc):
        '''Gets the net reaction of a pathway with multiplicities xjc.
        Arguments:
            xjc (numpy 1d array): multiplicities of pathway
        Returns:
            net_reaction (str)
        '''
        if not self.transport_species:
            species_list = self.species_list
            net = sparse_dot(self.sij, xjc).astype(int)
        else:
            species_list = self.full_species_list
            net = sparse_dot(self.full_sij, xjc).astype(int)
        reactants_idxs = np.where(net<0)[0]
        products_idxs = np.where(net>0)[0]

        # delete hv and M from reactants
        reactants_idxs = [x for x in reactants_idxs
            if species_list[x].upper() not in ['HV', 'M']]

        to_int = lambda x: int(x) if x == int(x) else x
        to_str = lambda x: str(x) if x != 1 else ''

        reactants = [to_str(to_int(-net[i])) + species_list[i]
            for i in reactants_idxs]

        products = [to_str(to_int(net[i])) + species_list[i]
            for i in products_idxs]
        
        if products and reactants:
            net_reaction = f'{" + ".join(reactants)} -> {" + ".join(products)}'
        else:
            net_reaction = 'Null'
        return net_reaction
    
    def save_pathway_info(self, path):
        '''Saves variables to numpy binary files
        Arguments:
            path(str): path where the files will be saved
        '''
        sparse.save_npz(f'{path}/sparse_xjk', self.xjk)
        h5py_filename = f"{path}/chempath_info.hdf5"
        with h5py.File(h5py_filename, "w") as datafile:
            datafile.create_dataset("fk", self.fk.shape, dtype='f16', 
                data=self.fk)
            datafile.create_dataset("pi", self.pi.shape, dtype='f16', 
                data=self.pi)
            datafile.create_dataset("di", self.di.shape, dtype='f16', 
                data=self.di)
            datafile.create_dataset("pi_del", self.pi_del.shape, dtype='f16', 
                data=self.pi_del)
            datafile.create_dataset("di_del", self.di_del.shape, dtype='f16', 
                data=self.di_del)
            datafile.create_dataset("pi_err", self.pi_err.shape, dtype='f16', 
                data=self.pi_err)
            datafile.create_dataset("di_err", self.di_err.shape, dtype='f16', 
                data=self.di_err)
            datafile.create_dataset("rj_del", self.rj_del.shape, dtype='f16', 
                data=self.rj_del)
            datafile.create_dataset("sb_list", [len(self.sb_list)], 
                    dtype=h5py.string_dtype(), data=self.sb_list)
            datafile.create_dataset("ignored_sb", [len(self.ignored_sb)], 
                    dtype=h5py.string_dtype(), data=self.ignored_sb)

def solve_system_eq(a,b):
    ''' Solves system of equations ax=b
    Arguments
        a (numpy array)
        b (numpy array)
    Returns
        x (numpy array)
    '''
    sol = lsq_linear(a, b,
        bounds=np.array([(0,np.inf) for i in range(a.shape[1])]).T,
        tol=1e-10 )
    x = sol.x
    return x

def solve_system_eq_rank(a,b, rank):
    sol = linprog(rank**2, A_eq = a, b_eq = b)
    return sol.x

def get_sij(species_list, reactions):
    '''Gets number of molecules/cm^3 or ppb of species i produced by reaction j
    Arguments:
        species_list (list): list of species names
        reactions (list): list of reactions equations
    Returns:
        sij (numpy array)
    '''
    ni = len(species_list)
    nj = len(reactions)
    sij = np.zeros((ni, nj), dtype=int)
    for i in range(ni):
        for j in range(nj):
            reaction = reactions[j]
            # if species in reactants
            reactants = reaction.split('=')[0].split('+')
            products = reaction.split('=')[1].split('+')
            if species_list[i] in reactants:
                # count of species in reactants
                n = reactants.count(species_list[i])
                # substract rate
                sij[i][j] -=  n 
            if species_list[i] in products:
                # count of species in products
                n = products.count(species_list[i])
                # substract rate
                sij[i][j] += n
    return sij

def get_pathway_id_dense(xjc):
    ''' Gets the unique identifier of a pathway with multiplicities xjc in 
    a dense format
    Arguments:
        xjc (numpy array): multiplicities of pathway
    Returns:
        pathway_id (str)
    '''
    xjc = np.squeeze(xjc.astype(int))
    rxns = np.where(xjc !=0)[0]
    coeffs = xjc[rxns]
    pathway_id = ','.join([f'{x}*{y}' for x,y in zip(coeffs, rxns)])
    return pathway_id

def get_pathway_id(xjc):
    ''' Gets the unique identifier of a pathway with multiplicities xjc in
    a sparse format
    Arguments:
        xjc (numpy array): multiplicities of pathway
    Returns:
        pathway_id (str)
    '''
    xjc = xjc.astype(int)
    rxns = xjc.nonzero()[0]
    coeffs = xjc.data
    pathway_id = ','.join([f'{x}*{y}' for x,y in zip(coeffs, rxns)])
    return pathway_id

def xjk_to_id_list(xjk):
    '''Converts xjk matrix to a list of pathway ids'''
    id_list = []
    for i in range(xjk.shape[1]):
        id_list.append(get_pathway_id(xjk[:, i]))
    return np.array(id_list)

def format_react_txt(reacts, prods):
    '''Gets a reaction string in a txt format
    Arguments:
        reacts (list): reactants
        prods (list): products
    Returns:
        reaction (str)
    '''
    if type(reacts) == str:
        reaction = f'{reacts} -> {"+".join(prods)}'
    elif type(prods) == str:
        reaction = f'{"+".join(reacts)} -> {prods}'
    else:
        reaction = f'{"+".join(reacts)} -> {"+".join(prods)}'
    return reaction

def format_react_latex(reacts, prods):
    '''Gets a reaction string in a latex format
    Arguments:
        reacts (list): reactants
        prods (list): products
    Returns:
        reaction (str)
    '''
    if type(reacts) == str:
        reaction = "%s -> %s" % (reacts, " + ".join(prods))
    elif type(prods) == str:
        reaction = "%s -> %s" % (" + ".join(reacts), prods)
    else:
        reaction = "%s -> %s" % (" + ".join(reacts), " + ".join(prods))
    return reaction

def get_positive_values(mik):
    '''Gets mik where mik>0'''
    posmik = np.multiply(mik, mik>0)
    return posmik

def get_negative_values(mik):
    '''Gets mik where mik<0'''
    negmik = np.multiply(mik, mik<0)
    return negmik

def sparse_dot(A, xjk):
    '''Perform dot product of A and xjk'''
    return xjk.T.dot(A.T).T

def delete_columns_sparse(xjk, delete_idxs):
    '''Delete columns in delete_idxs from sparse matrix xjk'''
    idxs = np.arange(0, xjk.shape[1])
    keep_idxs = np.setdiff1d(idxs, delete_idxs)
    return xjk[:, keep_idxs]

def flatten_list(x):
    flat = []
    for elem in x:
        flat += elem
    return flat

def get_latex_contribution_table(contribution_df, nrows=5, id_suffix=''):
    '''Cobverts a contribution dataframe to a latex table string
    Arguments:
        contribution_df (pandas dataframe)
        nrows (int): number of rows to consider in the latex table
        id_suffix (str) : suffix added to the id of a pathway
    Returns:
        latex_table (str)
    '''
    contribution_df = contribution_df.iloc[:nrows]

    latex_table = Template('''
        \\begin{longtable}{ |c|c|c|c| }
        \hline
        ID & Pathway & Contribution (\\%) & Rate \\\\
        \hline
        ${rows}
        \end{longtable}
        '''
    )
    to_str = lambda x: str(round(x, 3))
    rows = ''
    for i in range(nrows):
        id_str = id_suffix + str(i+1)
        dfrow = contribution_df.loc[i]
        pathway = '\\begin{tabular}{@{}c@{}}' + dfrow.pathway + '\end{tabular}'
        row =  f'{id_str} & {pathway} & {to_str(100 * dfrow.contribution)} & {to_str(dfrow.rate)}'
        rows += row + '\\\\' + '\n \\hline \n'
    return latex_table.substitute(rows=rows)

class TimeoutError(Exception):
    pass


