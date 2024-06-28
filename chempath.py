import math
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import lsq_linear, linprog
from copy import deepcopy
from scipy import sparse
from string import Template
import errno
import os
import signal
import warnings 

class Chempath():
    '''
    Pathway analysis program class.
    Arguments:
        reactions_path(str): path of input reactions equations
        rates_path (str): path of input reactions rates
        species_path (str): path of input species names
        conc_path (str): path of input species concentrations
        time_path (str): path of input model time
        f_min (float): minimum rate of pathways. Defaults to 0.0
        dtype (type): number type of numerical fields
        species_of_interest (list): List of species of interest. These species
            will not be considered as branching-point species.
    '''
    def __init__(self, 
        reactions_path,
        rates_path,
        species_path,
        conc_path,
        time_path,
        f_min=0, 
        warnings=True, 
        dtype=np.float128,
        species_of_interest = []):

        # ignore warnings
        self.warnings = warnings
        # minimum reate of pathways
        self.f_min = f_min
        # time 
        self.time = read_time_file(time_path, dtype=dtype)
        self.dt = self.time[1] - self.time[0]
        # model time
        self.model_time = np.mean(self.time)
        # species list
        self.species_list = read_species_file(species_path)
        # concentrations
        self.conc = read_conc_file(conc_path, len(self.species_list), dtype=dtype)
        # concentratio change
        self.dconc = self.conc[1] - self.conc[0]
        # mean concentration
        self.mean_conc = np.trapz(self.conc,self.time, axis=0) / self.dt
        # mean rate of concentration change
        self.mean_dconc = np.array(self.dconc) / self.dt 
        # reactions and rates
        self.reaction_equations = read_reactions_file(reactions_path)
        self.rj = read_rates(rates_path, dtype=dtype)
        self.delete_zero_reactions()
        self.invert_negative_rates()
        # part of rate of reaction j deleted pathways
        self.rj_del = np.zeros(len(self.reaction_equations), dtype=np.float128)
        # molecules of species i produces or destroyed by reaction j
        self.sij = get_sij(self.species_list, self.reaction_equations)
        # multiplicity of reaction j in pathway k
        self.xjk = np.diag(np.ones(len(self.reaction_equations), dtype=int))
        # list of pathways pathways by unique id
        self.pathway_ids = xjk_to_id_list(self.xjk)
        # molecules of species i produces or destroyed by pathway k
        self.mik = np.dot(self.sij, self.xjk)
        # rates of pathways
        self.fk = self.rj
        # rate of production of species i by deleted pathways
        self.pi_del = np.zeros(len(self.species_list), dtype=dtype)
        # rate of destruction of species i by deleted pathways
        self.di_del = np.zeros(len(self.species_list), dtype=dtype)
        # total rate of production of species i by all pathways
        self.pi = self.pi_del + np.dot(np.multiply(self.mik, self.mik>0), self.fk)
        # total rate of destruction of species i by all pathways
        self.di = self.di_del + np.dot(np.abs(np.multiply(self.mik, self.mik<0)), self.fk)
        # list of used banching species
        self.sb_list = []
        self. species_of_interest = species_of_interest
        # full list of species including transport species and full sij
        transport_species = [f'{x}_transport' for x in self.species_list]
        self.full_species_list = self.species_list + transport_species
        self.full_sij = get_sij(self.full_species_list, self.reaction_equations)
        # list of species not considered as branching species 
        self.ignored_sb = deepcopy(species_of_interest)


    def load_pathways_from_files(self, filespath, dtype=np.float128):
        '''Loads pathway info  saved using the save_pathway_info method
        Arguments:
            filespath(str): path of files to read
            dtype(type): data type of numbers if the files
        '''
        # part of rate of reaction j deleted pathways
        self.rj_del = np.fromfile(f'{filespath}/rj_del.dat', dtype=dtype)
        # multiplicity of reaction j in pathway k
        self.xjk = sparse.load_npz(f'{filespath}/sparse_xjk.npz').toarray()
        # list of pathways pathways by unique id
        self.pathway_ids = xjk_to_id_list(self.xjk)
        # molecules of species i produces or destroyed by pathway k
        self.mik = np.dot(self.sij, self.xjk)
        # rates of pathways
        self.fk = np.fromfile(f'{filespath}/fk.dat', dtype=dtype)
        # rate of production of species i by deleted pathways
        self.pi_del = np.fromfile(f'{filespath}/pi_del.dat', dtype=dtype)
        # rate of destruction of species i by deleted pathways
        self.di_del = np.fromfile(f'{filespath}/di_del.dat', dtype=dtype)
        # total rate of production of species i by all pathways
        self.pi = np.fromfile(f'{filespath}/pi.dat', dtype=dtype)
        # total rate of destruction of species i by all pathways
        self.di = np.fromfile(f'{filespath}/di.dat', dtype=dtype)
        # list of used banching species
        self.sb_list = list(np.loadtxt(f'{filespath}/sb_list.txt', dtype=str,
            delimiter=','))
        # list of species not considered as brancing species 
        self.ignored_sb = list(np.loadtxt(f'{filespath}/ignored_sb.txt',
            dtype=str, delimiter=','))

        
    def get_sb(self, tau_max=None):
        ''' Gets the next branching-point species
        Arguments:
            tau_max(float, Optional): maximum lifetime of branching point species. 
            Species with a lifetime higher than this will not be considered as 
            branching-point species.
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

        sb_list = df.species.to_list()
        
        if len(sb_list) > 0:
            self.sb_list.append(sb_list[0])
            return sb_list[0]
        return None

    def delete_zero_reactions(self):
        '''Deletes reactions with a zero rate'''
        delete_idxs = np.where(self.rj == 0)[0]
        self.rj = np.delete(self.rj, delete_idxs)
        self.reaction_equations = np.delete(self.reaction_equations, delete_idxs)

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
        self.mik = np.dot(self.sij, self.xjk)
        posmik = get_positive_values(self.mik)
        negmik = get_negative_values(self.mik)
        self.pi = self.pi_del + np.dot(posmik, self.fk)
        self.di = self.di_del + np.dot(np.abs(negmik), self.fk)
        
    def connect_pathways(self, mik, xjk):
        '''Connects pathways producing and destroying Sb
        Arguments:
            mik (numpy 2d array): matrix with the number molecules of species i 
                produced or  destroyed by pathway k
            xjk: (numpy 2d array): matrix with the multiplicity of reaction j 
                in pathway k
        '''
        sb_idx = self.sb_idx
        Db = np.max([self.di[sb_idx], self.pi[sb_idx]])
        
        # find all combinations of pathways producing and destroying sb
        combinations = product(self.prod_idxs, self.destr_idxs)

        # calculate new multiplicities so that sb is recycled
        # and calculate rates of new pathways
        xjk_new = []
        fk_new = []
        new_pathway_ids = []
        all_new_pathways = []

        for p,d in combinations:
            xjn = np.abs(mik[sb_idx, d]) * xjk[:,p] +\
                mik[sb_idx, p] * xjk[:,d]
            # divide all multiplicities by greater common divisor
            gcd = np.gcd.reduce(xjn.astype(int))
            xjn = xjn / gcd

            # calculate rates
            # rate has to be multiplied by gcd because pathways produce gcd
            # molecules of the species
            fn = np.multiply(gcd, np.divide(np.multiply(self.fk[p], self.fk[d]), Db))

            # if pathway already exists, do not repeat it, just add its rate:
            # the same pathway can be formed through different combination order
            # of same reactions 
            pid_n = get_pathway_id(xjn)
            if (pid_n not in self.pathway_ids) and (pid_n not in all_new_pathways):
                fk_new.append(fn)
                xjk_new.append(np.c_[xjn])
                new_pathway_ids.append(pid_n)
            elif pid_n in self.pathway_ids:
                pid_idx = np.where(self.pathway_ids == pid_n)[0]
                self.fk[pid_idx] = self.fk[pid_idx] + fn
            elif pid_n in new_pathway_ids:
                pid_idx = new_pathway_ids.index(pid_n)
                fk_new[pid_idx] += fn            
            all_new_pathways.append(pid_n)
        return xjk_new, fk_new, new_pathway_ids
    
    def find_new_pathways(self):
        '''Finds new pathways and appends their multiplicities and rates  to
        xjk and fk'''
        # for book keeping
        self.old_pathways_total_rates = 0
        self.new_pathways_total_rates = 0

        # connect pathways
        xjk_new, fk_new, new_pathway_ids = self.connect_pathways(self.mik,
                                                self.xjk)
       
        # include new pathways
        if len(xjk_new) > 0:
            xjk_new = np.hstack(xjk_new)

            # append new pathways
            self.xjk = np.concatenate([self.xjk, xjk_new], axis=1)
            self.fk = np.concatenate([self.fk, fk_new])
            self.pathway_ids = np.append(self.pathway_ids, new_pathway_ids)        

            # for book keeping
            idxs = np.concatenate([self.prod_idxs,self.destr_idxs])
            self.old_pathways_total_rates =\
                 math.fsum(np.dot(self.xjk[:,idxs], self.fk[idxs]))
            self.new_pathways_total_rates = math.fsum(np.dot(xjk_new, fk_new))
            
    
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
            rates_explaining_sb = math.fsum(np.dot(self.xjk[:,idxs], self.fk[idxs]))
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
            self.deleted_rates = math.fsum(np.dot(self.xjk[:,delete_idxs],
                 self.fk[delete_idxs]))

            self.xjk = np.delete(self.xjk, delete_idxs, axis=1)
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

        # calculate fraction of rates associated with deleted pathways
        fdel_prod = np.divide(np.multiply(self.fk[self.prod_idxs],
            self.di_del[sb_idx]), Db)
        fdel_destr = np.divide(np.multiply(self.fk[self.destr_idxs],
            self.pi_del[sb_idx]), Db)
 

         # update deleted pathway variables
        self.connection_del_pathways_rates = 0
        self.connection_del_pathways_rates1 = 0
        if new_pathways_flag:
            connection_btwn_del_pathways = self.pi_del[sb_idx] * self.di_del[sb_idx] / Db
            self.connection_btwn_del_pathways = connection_btwn_del_pathways
            # connection with deleted pathways
            for i, p in enumerate(self.prod_idxs):
                posmik = np.multiply(self.mik[:,p], self.mik[:,p]>0)
                negmik =np.multiply(self.mik[:,p], self.mik[:,p]<0)
                self.rj_del = self.rj_del +\
                    np.multiply(self.xjk[:,p], fdel_prod[i])
                self.connection_del_pathways_rates = math.fsum([
                    self.connection_del_pathways_rates,
                    math.fsum(np.multiply(self.xjk[:,p], fdel_prod[i]))
                ])
                self.pi_del = self.pi_del + posmik * fdel_prod[i]
                self.di_del = self.di_del + np.abs(negmik) * fdel_prod[i]
      
            for i, d in enumerate(self.destr_idxs):
                posmik = np.multiply(self.mik[:,d], self.mik[:,d]>0)
                negmik =np.multiply(self.mik[:,d], self.mik[:,d]<0)
                self.rj_del = self.rj_del +\
                    np.multiply(self.xjk[:,d], fdel_destr[i])
                self.connection_del_pathways_rates = math.fsum([
                    self.connection_del_pathways_rates,
                    math.fsum(np.multiply(self.xjk[:,d], fdel_destr[i]))
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
                self.rj_del = self.rj_del + self.xjk[:,i] * self.fk[i]
                self.pi_del = self.pi_del + posmik * self.fk[i]
                self.di_del = self.di_del + np.abs(negmik) * self.fk[i]

            self.fk = np.delete(self.fk, delete_idxs)
            self.xjk = np.delete(self.xjk, delete_idxs, axis=1)
            self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)

            # redefine stuff depending on pathways
            self.recompute_pathway_dependent_variables()
        
    def print_book_keeping_variables(self):
        print(self.prod_idxs)
        print(self.destr_idxs)
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

        rates_pathways = np.dot(self.xjk, self.fk) + self.rj_del
        print('------------------------------------')
        print(f'total reaction rates: {math.fsum(self.rj)}')
        print(f'total pathways rates: {math.fsum(rates_pathways)}')
        print(f'difference: {math.fsum(self.rj) - math.fsum(rates_pathways)}')
        print(f'division: {math.fsum(self.rj)/math.fsum(rates_pathways)}')


    def split_into_subpathways(self, exact_solutions=True):
        '''Splits pathways into elementary subpathways
        Arguments:
            exact_solutions(bool, optional): if True, only exact solutions to
            system of equations to split pathways are accepted
        '''
        new_pathways_flag = len(self.prod_idxs) > 0 and len(self.destr_idxs) > 0
        if new_pathways_flag:
            delete_idxs = []
            # for each pathway...
            for p_id in self.pathway_ids:
                p_index = np.where(self.pathway_ids == p_id)[0]
                p = self.xjk[:, p_index]
                f = self.fk[p_index]

                # find elementary subpathways
                xjk_elem = self.find_elementary_pathways(p, p_index)
                
                # if a pathway does not have subpathways do nothing
                if xjk_elem.shape[1] == 1:
                    continue
                
                # solve system of equations ax = b 
                a = xjk_elem
                b = p.reshape(1, p.shape[0])[0]
                x = solve_system_eq(a, b)
                
                # if solution is not exact, do nothing
                if exact_solutions:
                    if not np.all(np.isclose(np.dot(a,x), b)):
                        # print(np.dot(a,x) - b)
                        continue
                
                # distribute rate to subpathways
                fk_sub = f * x

                delete_idxs.append(p_index)

                # look for subpathways in current pathways. Add rates if they already
                # exists or add them to list otherwise
                for i in range(xjk_elem.shape[1]):
                    elem_pid = get_pathway_id(xjk_elem[:, i])
                    if np.isin(elem_pid, self.pathway_ids):
                        # print(f'Adding rate to existing pathway:{elem_pid}')
                        idx = np.where(self.pathway_ids == elem_pid)[0][0]
                        self.fk[idx] = self.fk[idx] + fk_sub[i]
                    else:
                        # print(f'Adding new pathway:{elem_pid}')
                        if elem_pid not in self.pathway_ids:
                            self.xjk = np.concatenate(
                                [self.xjk, np.c_[xjk_elem[:, i]]], axis=1)
                            self.fk = np.concatenate([self.fk, [fk_sub[i]]])
                            self.pathway_ids = np.append(
                                self.pathway_ids, elem_pid)
    
            # Delete splitted pathways
            self.xjk = np.delete(self.xjk, delete_idxs, axis=1)
            self.pathway_ids = np.delete(self.pathway_ids, delete_idxs)
            self.fk = np.delete(self.fk, delete_idxs)

            # redefine stuff depending on pathways
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
                    new_mik.append(np.abs(self.mik[sb_idx, pathway_index]))
                # id dc_sb>0 add pseudo-reaction destroying sb
                if self.mik[sb_idx, pathway_index] > 0:
                    new_reactions.append(f'{sb}=...')
                # id dc_sb<0 add pseudo-reaction producing sb
                elif self.mik[sb_idx, pathway_index] < 0:
                    new_reactions.append(f'...={sb}')
            reactions = np.append(self.reaction_equations, new_reactions)
            sij = get_sij(self.species_list, reactions)
            xjc = np.append(xjc, new_mik)
        else:
            sij = self.sij
            reactions = self.reaction_equations

        # find reactions in pathway
        rxns = np.where(xjc != 0)[0]

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

        # fund subpathways
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
            if np.dot(xjn.squeeze(), self.sij[idx,:]) != 0:
                return False
        return True
    
    def find_all_pathways(self, tau_max=None, timeout=60*40,
            split_into_subpathways=True, verbose=False, exact_solutions=True):
        
        def _handle_timeout(signum, frame):
            raise TimeoutError(os.strerror(errno.ETIME))
        
        sb = self.get_sb(tau_max=tau_max)

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout)
        try:
            while sb:
                # print('get_prod_destr_idxs')                
                self.get_prod_destr_idxs(sb)
                # print('find_new_pathways')
                self.find_new_pathways()
                # print('calculate_deleted_pathways_effect')
                self.calculate_deleted_pathways_effect()
                # print('calculate_rates_explaining_conc_change')
                self.calculate_rates_explaining_conc_change()
                # print('delete_old_pathways')
                self.delete_old_pathways()
                # print('delete_insignificant_pathways')
                self.delete_insignificant_pathways()
                if split_into_subpathways:
                    # print('split_into_subpathways')
                    self.split_into_subpathways(exact_solutions=exact_solutions)
                if verbose:
                    print('########################')
                    print(sb) 
                    print('########################')
                    self.print_book_keeping_variables()
                    print(self.get_pathways_explained_change())
                sb = self.get_sb(tau_max=tau_max)
            self.check_rate_distribution()
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
        Db = np.max([self.di[sp_idx], self.pi[sp_idx]])
        prod_idxs = np.where(sp_mik>0)[0]
        destr_idxs = np.where(sp_mik<0)[0]   
        # calculate rates of production or destruction of sp by specific pathways
        production = np.abs(sp_mik) * self.fk 
        # calculate contributions
        if on == 'loss':
            idxs = destr_idxs
            deleted_pathways_prod = self.di_del[sp_idx]
            total_production = self.di[sp_idx]
        elif on == 'production':
            idxs = prod_idxs
            deleted_pathways_prod = self.pi_del[sp_idx]
            total_production = self.pi[sp_idx]
        # elif on == 'dc':
        #     if self.dconc[sp_idx] > 0:
        #         idxs = prod_idxs
        #         deleted_pathways_prod = self.pi_del[sp_idx]
        #     elif self.dconc[sp_idx] < 0:
        #         idxs = destr_idxs
        #         deleted_pathways_prod = self.di_del[sp_idx]
        #     total_production = Db

        contrib = production[idxs] / total_production
        pathways = [self.xjk[:, i] for i in idxs]
        rates = self.fk[idxs]
        total_prod = np.multiply(self.mik[sp_idx, idxs], self.fk[idxs])*self.dt
        p_ids = self.pathway_ids[idxs]
        p_strs = [self.get_pathway_str(p, format=format) for p in pathways]
        p_net_reacts = [self.get_net_reaction(p) for p in pathways]
        
        # make a pandas dataframe with the info
        contrib_dict = {'pathway_id': p_ids, 'pathway': p_strs,
            'contribution': contrib, 'rate':rates,
            'net_reaction': p_net_reacts, 'total_prod': total_prod}
        contrib_df = pd.DataFrame(contrib_dict)

        deleted_contrib = pd. DataFrame({'pathway_id': ['del'], 
                'pathway': ['deleted_pathways'],
                'contribution': [deleted_pathways_prod/total_production], 
                'rate':[self.di_del[sp_idx]],
                'total_prod': deleted_pathways_prod * self.dt})

        contrib_df = pd.concat([contrib_df, deleted_contrib])
        contrib_df['dconc'] = self.dconc[sp_idx]
        contrib_df.sort_values('contribution', ascending=False, inplace=True)
        contrib_df = contrib_df.reset_index(drop=True)
        return contrib_df        
   
    def check_mass_conservation(self):
        prod = np.dot(self.sij, self.rj) * self.dt
        return self.dconc/(prod+1e-100)

    def check_rate_distribution(self):
        ''' Checks in reaction rates are completely distributed to the pathway's
        rates. If not, raises a warning
        '''
        total_rates = math.fsum(self.rj)
        total_pathway_rates = math.fsum(self.rj_del + np.dot(self.xjk, self.fk))
        rate_conservation = np.isclose(total_rates, total_pathway_rates)
        if not rate_conservation:
            warnings.warn('Rates are not correctly distributed!')

    def get_pathways_explained_change(self):
        '''Gets dataframe with the fraction of concentration changes explained by
        pathways'''
        rates_pathways = np.dot(self.xjk, self.fk)+ self.rj_del
        total_prod = np.dot(self.sij, rates_pathways)
        explained_change = self.mean_dconc / total_prod
        explained_change = {k:[v] for k,v in zip(self.species_list, explained_change)}
        explained_change = pd.DataFrame(explained_change)
        return explained_change 

    def get_deleted_pathways_explained_change(self):
        '''Gets dataframe with the fraction of concentration changes explained by
        deleted pathways'''
        total_prod = np.dot(self.sij, self.rj_del) * self.dt
        explained_change = self.dconc / total_prod
        explained_change = {k:[v] for k,v in zip(self.species_list, explained_change)}
        explained_change = pd.DataFrame(explained_change)
        return explained_change 
    
    def get_total_pathway_rates(self):
        '''Gets the total pathways rates'''
        total_pathway_rates = math.fsum(self.rj_del + np.dot(self.xjk, self.fk))
        return total_pathway_rates
    
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
        idxs = np.where(xjc !=0)[0]
        coeffs = xjc[idxs]

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
            coeff = coeffs[i]
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
        net = np.dot(self.full_sij, xjc).astype(int)
        reactants_idxs = np.where(net<0)[0]
        products_idxs = np.where(net>0)[0]

        # delete hv and M from reactants
        reactants_idxs = [x for x in reactants_idxs
            if self.full_species_list[x].upper() not in ['HV', 'M']]

        to_int = lambda x: int(x) if x == int(x) else x
        to_str = lambda x: str(x) if x != 1 else ''

        reactants = [to_str(to_int(-net[i])) + self.full_species_list[i]
            for i in reactants_idxs]

        products = [to_str(to_int(net[i])) + self.full_species_list[i]
            for i in products_idxs]
        
        net_reaction = f'{" + ".join(reactants)} -> {" + ".join(products)}'
        return net_reaction
    
    def save_pathway_info(self, path):
        '''Saves variables to numpy binary files
        Arguments:
            path(str): path where the files will be saved
        '''
        sparse_xjk = sparse.csr_matrix(self.xjk)
        sparse.save_npz(f'{path}/sparse_xjk', sparse_xjk)
        self.fk.tofile(f'{path}/fk.dat')
        np.float128(self.pi).tofile(f'{path}/pi.dat')
        np.float128(self.di).tofile(f'{path}/di.dat')
        np.float128(self.pi_del).tofile(f'{path}/pi_del.dat')
        np.float128(self.di_del).tofile(f'{path}/di_del.dat')
        np.float128(self.rj_del).tofile(f'{path}/rj_del.dat')
        np.savetxt(f'{path}/sb_list.txt', self.sb_list, delimiter=",", fmt="%s")
        np.savetxt(f'{path}/ignored_sb.txt', self.ignored_sb, delimiter=",", fmt="%s")


def read_reactions_file(filepath):
    '''Reads input reactions file
    Arguments:
        filepath(str): path of reaction equations file
    Returns:
        reactions (list): list of reactions equation strings      
    '''
    reactions =  np.loadtxt(filepath, dtype=str, delimiter=',')
    reactions = [reaction.replace(' ', '') for reaction in reactions]
    return reactions

def read_rates(filepath, dtype=np.float128):
    '''Reads input reactions rates file
    Arguments:
        filepath(str): path of reactions rates file
        dtype (type): number type of reaction rates
    Returns:
        rates (numpy array): array of reactions rates      
    '''
    rates = np.fromfile(filepath, dtype=dtype)
    return rates
     
def read_conc_file(filepath, n_species, dtype=np.float128):
    '''Reads input species concentrations file
    Arguments:
        filepath(str): path of species concentrations file
        n_species(int): number of species in the reaction system
        dtype (type): number type of concentrations
    Returns:
        conc (numpy array): array of species concentrations   
    '''
    conc = np.fromfile(filepath, dtype=dtype).reshape(2, n_species)
    return  conc

def read_species_file(filepath):
    '''Reads input species names file
    Arguments:
        filepath(str): path of species names file
    Returns:
        species (numpy array): array of species names      
    '''
    species =  list(np.loadtxt(filepath, dtype=str, delimiter=','))
    return species

def read_time_file(filepath, dtype=np.float128):
    '''Reads input model time file
    Arguments:
        filepath(str): path of model time file
        dtype (type): number type of model time
    Returns:
        species (numpy array): array of species names      
    '''
    time = np.fromfile(filepath, dtype=dtype)
    return time

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

# def solve_system_eq1(a,b):
#     rank = np.argsort(a.sum(axis=0))**2
#     sol = linprog(rank, A_eq = a, b_eq = b)
#     return sol.x

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

def get_pathway_id(xjc):
    ''' Gets the unique identifier of a pathway with multiplicities xjc
    Arguments:
        xjc (numpy array): multiplicities of pathway
    Returns:
        pathway_id (str)
    '''
    rxns = np.where(xjc != 0)[0]
    coeffs = [int(x) for x in xjc[rxns]]
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

    toestr  = lambda x: '{:.3e}'.format(x)
    to_str = lambda x: str(round(x, 3))
    rows = ''
    for i in range(nrows):
        id_str = id_suffix + str(i+1)
        dfrow = contribution_df.loc[i]
        pathway = '\\begin{tabular}{@{}c@{}}' + dfrow.pathway + '\end{tabular}'
        row =  f'{id_str} & {pathway} & {to_str(100 * dfrow.contribution)} & {toestr(dfrow.rate)}'
        rows += row + '\\\\' + '\n \\hline \n'
    return latex_table.substitute(rows=rows)

class TimeoutError(Exception):
    pass


