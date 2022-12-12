# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:55:56 2022

@author: Xiaox
"""
from dwave.system.samplers import LeapHybridBQMSampler
from dwave.system import DWaveSampler, EmbeddingComposite,DWaveCliqueSampler
import datetime
import dimod
import numpy as np
from hybrid import traits
from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from hybrid.core import Runnable, SampleSet
import hybrid


from dimod import quicksum, BinaryQuadraticModel, Real, Binary, SampleSet

import hybrid
from hybrid.decomposers import EnergyImpactDecomposer


import argparse
import time
from itertools import combinations, permutations
import numpy as np
from typing import Tuple
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch


from unilts import read_instance

class QPUSubproblemAutoEmbeddingSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    """A quantum sampler for a subproblem with automated heuristic minor-embedding.
    """
    def __init__(self, num_reads=100, num_retries=0, qpu_sampler=None, sampling_params=None,
                 auto_embedding_params=None, **runopts):
        super(QPUSubproblemAutoEmbeddingSampler, self).__init__(**runopts)
        self.num_reads = num_reads
        self.num_retries = num_retries
        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params
        # embed on the fly and only if needed
        if auto_embedding_params is None:
            auto_embedding_params = {}
        self.sampler = AutoEmbeddingComposite(qpu_sampler, **auto_embedding_params)
        self.qpu_access_time=0
    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        sampling_params = runopts.get('sampling_params', self.sampling_params)
        params = sampling_params.copy()
        params.update(num_reads=num_reads)
        num_retries = runopts.get('num_retries', self.num_retries)
        embedding_success = False
        num_tries = 0

        while not embedding_success:
            try:
                num_tries += 1
                response = self.sampler.sample(state.subproblem, **params)
                self.qpu_access_time+=response.info['timing']['qpu_access_time']*(0.001)
            except ValueError as exc:
                if num_tries <= num_retries:
                    pass
                else:
                    raise exc
            else:
                embedding_success = True
        return state.updated(subsamples=response)

class KerberosSampler(dimod.Sampler):
    
    """An opinionated dimod-compatible hybrid asynchronous decomposition sampler
    for problems of arbitrary structure and size.

    Examples:
        This example solves a two-variable Ising model.

        >>> import dimod
        >>> import hybrid
        >>> response = hybrid.KerberosSampler().sample_ising(
        ...                     {'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})    # doctest: +SKIP
        >>> response.data_vectors['energy']      # doctest: +SKIP
        array([-1.5])

    """
    properties = None
    parameters = None
    runnable = None

    def __init__(self):
        self.parameters = {
            'num_reads': [],
            'init_sample': [],
            'max_iter': [],
            'max_time': [],
            'convergence': [],
            'energy_threshold': [],
            'sa_reads': [],
            'sa_sweeps': [],
            'tabu_timeout': [],
            'qpu_reads': [],
            'qpu_sampler': [],
            'qpu_params': [],
            'max_subproblem_size': []
        }
        self.properties = {}
        self.QPUSubproblemAutoEmbeddingSampler=QPUSubproblemAutoEmbeddingSampler
    def Kerberos(self,sampler,max_iter=100, max_time=None, convergence=3, energy_threshold=None,
                 sa_reads=1, sa_sweeps=10000, tabu_timeout=500,
                 qpu_reads=100, qpu_sampler=None, qpu_params=None,
                 max_subproblem_size=50):
        """An opinionated hybrid asynchronous decomposition sampler for problems of
        arbitrary structure and size. Runs Tabu search, Simulated annealing and QPU
        subproblem sampling (for high energy impact problem variables) in parallel
        and returns the best samples.

        Kerberos workflow is used by :class:`KerberosSampler`.

        
        Returns:
            Workflow (:class:`~hybrid.core.Runnable` instance).

        """
        #=self.QPUSubproblemAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
        energy_reached = None
        if energy_threshold is not None:
            energy_reached = lambda en: en <= energy_threshold
        iteration = hybrid.Race(
            hybrid.BlockingIdentity(),
            hybrid.InterruptableTabuSampler(
                timeout=tabu_timeout),
            hybrid.InterruptableSimulatedAnnealingProblemSampler(
                num_reads=sa_reads, num_sweeps=sa_sweeps),
            hybrid.EnergyImpactDecomposer(
                size=max_subproblem_size, rolling=True, rolling_history=0.3, traversal='bfs')
                | sampler
                | hybrid.SplatComposer()
        ) | hybrid.ArgMin()

        workflow = hybrid.Loop(iteration, max_iter=max_iter, max_time=max_time,
                               convergence=convergence, terminate=energy_reached)
        return workflow

    def sample(self, bqm,init_sample=None, num_reads=1, max_iter=100, max_time=None, convergence=3, energy_threshold=None,
                 sa_reads=1, sa_sweeps=10000, tabu_timeout=500,
                 qpu_reads=100, qpu_sampler=None, qpu_params=None,
                 max_subproblem_size=50):
            """Run Tabu search, Simulated annealing and QPU subproblem sampling (for
            high energy impact problem variables) in parallel and return the best
            samples.
    
            Sampling Args:
    
                bqm (:obj:`~dimod.BinaryQuadraticModel`):
                    Binary quadratic model to be sampled from.
    
                init_sample (:class:`~dimod.SampleSet`, callable, ``None``):
                    Initial sample set (or sample generator) used for each "read".
                    Use a random sample for each read by default.
    
                num_reads (int):
                    Number of reads. Each sample is the result of a single run of the
                    hybrid algorithm.
    
            Termination Criteria Args:
    
                max_iter (int):
                    Number of iterations in the hybrid algorithm.
    
                max_time (float/None, optional, default=None):
                    Wall clock runtime termination criterion. Unlimited by default.
    
                convergence (int):
                    Number of iterations with no improvement that terminates sampling.
    
                energy_threshold (float, optional):
                    Terminate when this energy threshold is surpassed. Check is
                    performed at the end of each iteration.
    
            Simulated Annealing Parameters:
    
                sa_reads (int):
                    Number of reads in the simulated annealing branch.
    
                sa_sweeps (int):
                    Number of sweeps in the simulated annealing branch.
    
            Tabu Search Parameters:
    
                tabu_timeout (int):
                    Timeout for non-interruptable operation of tabu search (time in
                    milliseconds).
    
            QPU Sampling Parameters:
    
                qpu_reads (int):
                    Number of reads in the QPU branch.
    
                qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
                    Quantum sampler such as a D-Wave system.
    
                qpu_params (dict):
                    Dictionary of keyword arguments with values that will be used
                    on every call of the QPU sampler.
    
                max_subproblem_size (int):
                    Maximum size of the subproblem selected in the QPU branch.
    
            Returns:
                :obj:`~dimod.SampleSet`: A `dimod` :obj:`.~dimod.SampleSet` object.
    
            """
            if callable(init_sample):
                init_state_gen = lambda: hybrid.State.from_sample(init_sample(), bqm)
            elif init_sample is None:
                init_state_gen = lambda: hybrid.State.from_sample(hybrid.random_sample(bqm), bqm)
            elif isinstance(init_sample, dimod.SampleSet):
                init_state_gen = lambda: hybrid.State.from_sample(init_sample, bqm)
            else:
                raise TypeError("'init_sample' should be a SampleSet or a SampleSet generator")
            sampler=self.QPUSubproblemAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            self.runnable = self.Kerberos(sampler,max_iter, max_time, convergence, energy_threshold,sa_reads, sa_sweeps, tabu_timeout,qpu_reads, qpu_sampler, qpu_params,
                         max_subproblem_size)
    
            samples = []
            energies = []
            for _ in range(num_reads):
                init_state = init_state_gen()
                final_state = self.runnable.run(init_state)
                # the best sample from each run is one "read"
                ss = final_state.result().samples
                ss.change_vartype(bqm.vartype, inplace=True)
                samples.append(ss.first.sample)
                energies.append(ss.first.energy)
            
            return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype, energy=energies),sampler.qpu_access_time


class zweiD_Problem():
    def __init__(self,KerberosSampler,data = read_instance()):
        self.stueck_ids = np.repeat(data["stueck_ids"], data["quantity"])
        self.num_stueck = np.sum(data["quantity"], dtype=np.int32)
        self.stueck_lange = np.repeat(data["stueck_lange"], data["quantity"])
        self.stueck_breite = np.repeat(data["stueck_breite"], data["quantity"])
        print(f'Anzahl der Stücke: {self.num_stueck}')
        self.platte_lange = data["platte_dim"][0]
        self.platte_breite = data["platte_dim"][1]
        self.num_platte = data["num_platte"]
        self.gesamte_platte_lange=self.platte_lange*self.num_platte
        self.lowest_num_platte = np.ceil(
            np.sum(self.stueck_lange*self.stueck_breite) / (
                    self.platte_lange*self.platte_breite))
        self.lowest_flaeche=self.lowest_num_platte*self.platte_lange*self.platte_breite
        self.gesamte_flaeche=self.gesamte_platte_lange*self.platte_breite
        if self.lowest_num_platte > self.num_platte:
            raise RuntimeError(
                f'anzahl der Platten ist im mindesten {self.lowest_num_platte}'+
                    'try increasing the number of platte'
            )
        print(f'anzahl der Platten ist im mindesten:{self.lowest_num_platte}')
        
        
        self.X_i={}
        
        self.KerberosSampler=KerberosSampler
        
    
    def define_variables(self):
        self.P_j={(j):'P_{}'.format(j)for j in range(self.num_platte)}
        self.X_i={(i,j,r,a,b):'x_{}_{}_{}_{}_{}'.format(i,j,r,a,b)for i in range(self.num_stueck)
                   for j in range(self.num_platte)
                   for r in range(2) #0:nicht umgedrehnt 1:umgedrehnt
                   for a in range(j*self.platte_lange,(j+1)*self.platte_lange-self.eff_dim[i][r][0]+1)
                   for b in range(0,self.platte_breite - self.eff_dim[i][r][1]+1)}
        self.variables=[self.X_i,self.P_j]
        return self.variables
    
    def define_bqm(self):
        """define bqm model
        For the bqm model, the variables should be added to the bqm model by the command "bqm.add_variable" """
        self.bqm=BinaryQuadraticModel('BINARY')
        for i in self.variables:      
            for j in i.values():
                self.bqm.add_variable(j)
        return self.bqm
    
    def variables_constraints(self,weight):
        for i in range(self.num_stueck):
            self.bqm.add_linear_equality_constraint([(self.X_i[(i,j,r,a,b)],1)for j in range(self.num_platte)
                                                     for r in range(2) #0:nicht umgedrehnt 1:umgedrehnt
                                                     for a in range(j*self.platte_lange,(j+1)*self.platte_lange-self.eff_dim[i][r][0]+1)
                                                     for b in range(0,self.platte_breite - self.eff_dim[i][r][1]+1)],
                                                            lagrange_multiplier=weight,
                                                            constant=-1)
            '''
            self.bqm.add_linear_equality_constraint([(self.Y_ia[(i,b)],1)for b in range(self.platte_breite)],
                                                            lagrange_multiplier=weight, 
                                                            constant=-1)
            self.bqm.add_linear_equality_constraint([(self.R_ir[(i,r)],1)for r in range(2)],
                                                            lagrange_multiplier=weight, 
                                                            constant=-1)
            '''
        return
    
   
    def stuecke_position_constraint(self,weight):
     #stück_i只能在已经启用的stang_j上切割
         for i in range(self.num_stueck):
            for j in range(self.num_platte):
                for r in range(2): #0:nicht umgedrehnt 1:umgedrehnt
                    for a in range(j*self.platte_lange,(j+1)*self.platte_lange-self.eff_dim[i][r][0]+1):
                        for b in range(0,self.platte_breite - self.eff_dim[i][r][1]+1):
                                self.bqm.add_quadratic(self.P_j[(j)],self.X_i[(i,j,r,a,b)],weight)
                    
                    
    
    def effective_dim(self):
        self.eff_dim={}
        for i in range(self.num_stueck):
            self.eff_dim[i]={}
            p1=list(permutations([self.stueck_lange[(i)],self.stueck_breite[(i)]]))
            self.eff_dim[i][0]=p1[(0)]
            self.eff_dim[i][1]=p1[(1)]
        return [self.eff_dim]
        '''
        self.el={}
        self.eb={}
        for i in range(self.num_stueck):
            self.el[(i)]=[(self.R_ir[(i,0)], -(a+self.stueck_lange[i]))for a in range(self.gesamte_stange_lange+1)]
            self.eb[(i)]=(self.stueck_breite[i]* self.R_i[(i)])+(self.stueck_lange[i]*(1-self.R_i[(i)]))
        return [self.el,self.eb]
        '''
    def geomerie_constraint(self,weight):
        for i, k in combinations(range(self.num_stueck),r=2):
            for j in range(self.num_platte):
                for r in range(2):
                    for s in range(2):
                        for a in range(j*self.platte_lange,(j+1)*self.platte_lange-self.eff_dim[i][r][0]+1):
                            for c in range(j*self.platte_lange,(j+1)*self.platte_lange-self.eff_dim[k][s][0]+1):
                                if ((a-self.eff_dim[k][s][0]) < c < (a+self.eff_dim[i][r][0])):
                                    for b in range(0,self.platte_breite - self.eff_dim[i][r][1]+1):
                                        for d in range(0,self.platte_breite - self.eff_dim[k][s][1]+1):
                                            if ((b-self.eff_dim[k][s][1])< d < (b+self.eff_dim[i][r][1])):
                            #for c in range(a-self.eff_dim[k][s][0]+1,a+self.eff_dim[i][r][0]):
                                #for d in range(b-self.eff_dim[k][s][1]+1,b+self.eff_dim[i][r][1]):
                                    #if ((a-self.eff_dim[k][s][0]) < c < (a+self.eff_dim[i][r][0])) and ((b-self.eff_dim[k][s][1])< d < (b+self.eff_dim[i][r][1])):
                                                self.bqm.add_quadratic(self.X_i[(i,j,r,a,b)],self.X_i[(k,j,s,c,d)],weight)
        return
    

    
    
    def anzahl_objektive(self,weight):
        for j in range(self.num_platte):
            self.bqm.add_linear(self.P_j[(j)],-weight)
        return
    
       
    def call_bqm_solver(self,max_iter,convergence):
        
        sampler,qpu_access_time= self.KerberosSampler.sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=DWaveSampler(),qpu_params={'label': '2d_bqm_ker'},qpu_reads=1500)
        samplerSET=sampler.samples()
        self.solution={}
        for key in samplerSET:
            sampler =key
        for key in sampler:
            if sampler[key] !=0 and "slack" not in key:
                self.solution[key]=sampler[key]
        qpu_access_times=0
        qpu_access_times+=qpu_access_time
        
                
        '''
        sampler = LeapHybridBQMSampler(token='EbWg-1dc4f5bb6bc7384895496a5dff589cc676d6e1a9')
        raw_sampleset = sampler.sample(self.bqm,label="2d_Problem_hy")
        best_sample = raw_sampleset.first.sample
        self.solution={}
        for key in best_sample:
            if best_sample[key] !=0 and "slack" not in key:
                self.solution[key]=best_sample[key]
        
        
        '''
        return self.solution,qpu_access_times
    
    def klassische_solve(self,max_iter,convergence,num_reads):
        iteration = hybrid.RacingBranches(hybrid.Const(subsamples=None)
        |hybrid.InterruptableTabuSampler(num_reads=None, tenure=None, timeout=100, initial_states_generator='random'),
        hybrid.EnergyImpactDecomposer(size=2)
        | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=None, num_sweeps=1000,
                 beta_range=None, beta_schedule_type='geometric',
                 initial_states_generator='random')
        | hybrid.SplatComposer()
        ) | hybrid.ArgMin()
        workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=convergence,max_iter=max_iter)
    
        # Solve the problem
        init_state = hybrid.State.from_problem(self.bqm)
        final_state = workflow.run(init_state).result()
        self.solution={}
        for key in final_state.samples.first.sample:            
            if final_state.samples.first.sample[key] !=0:
               self.solution[key]=final_state.samples.first.sample[key]
       
        return self.solution
    
    def zeichnung(self):
        self.eff_daten=list(key for key in self.solution.keys() if key in self.X_i.values())
        print(self.eff_daten)
        self.ids=[]
        self.i_loc=[]
        self.um=[]
        self.x_achse=[]
        self.y_achse=[]
        for n in self.eff_daten:
                zahlen=re.findall(r"\d+",n)
                ids_pos= list(map(int,zahlen))
                self.ids.append(ids_pos[(0)])
                self.i_loc.append(ids_pos[(1)])
                self.um.append(ids_pos[(2)])
                self.x_achse.append(ids_pos[(3)])
                self.y_achse.append(ids_pos[(4)])
        print(self.ids)
        print(self.i_loc)
        print(self.um)
        print(self.x_achse)
        print(self.y_achse)
        
        
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.title("2D-CSP")
        
        ids_size = []
        for (i,l,w)in (list(zip(self.stueck_ids,self.stueck_lange,self.stueck_breite))):
            if (i,l,w) not in ids_size:
                ids_size.append((i,l,w))
        #ids=list(set(self.stueck_ids))
        
        n=len(ids_size)
        colors=plt.cm.jet(np.linspace(0, 1, n))
        legend_i=[]
        for i in range(n):
            legend_i.append(i)
            legend_i[i] = mpatch.Patch(color=colors[i], label=f'id{ids_size[i][0]}:{ids_size[i][1]} cm X {ids_size[i][2]} cm',alpha=0.5)
        platte_legend=mpatch.Patch(facecolor="white", edgecolor = 'blue',label=f'Platte:{self.platte_lange} cm X {self.platte_breite}',alpha=0.3)
        legend_i.append(platte_legend)
        ax.legend(handles=legend_i,bbox_to_anchor=(1,0.8),loc='center left')


        for x,y,i,r in zip(self.x_achse,self.y_achse,self.ids,self.um):
                txt=f"id{self.stueck_ids[(i)]}"+" "+f"Teil {i}"
                plt.text(x+0.1,y+0.1,txt,fontsize=7)
                rect=mpatch.Rectangle((x,y),self.eff_dim[i][r][0],self.eff_dim[i][r][1],edgecolor = 'green',facecolor =colors[self.stueck_ids[(i)]],fill=True,alpha=0.3)
                ax.add_patch(rect)
        
        ticks=[]
        labels=[]
        for j in range(self.num_platte):
            plt.axvline(x=((j+1)*self.platte_lange), ymax = 0.5 ,ymin=0)
            ticks.append((j+1)*self.platte_lange)
            labels.append('Platte_' + str(j+1))
        plt.xticks(ticks,labels)
        plt.axhline(y=self.platte_breite,xmax=0.5)
        ax.axis([0,2*self.gesamte_platte_lange,0,2*self.platte_breite])
        ax.set_aspect(1)
        plt.show()
        

    def prufung(self):
        
        
        #for i in range(1):
            #for a in range(self.gesamte_platte_lange):
                #print(self.bqm.get_linear(self.X_i[(3,0,0,0,5)]))
                print(self.bqm.get_linear(self.P_j[(0)]))
        #print(self.bqm.get_quadratic(self.X_i[(1,0,12,18)],self.X_i[(3,0,18,18)]))
        
        
if __name__== "__main__":
    KerberosSampler=KerberosSampler()
    a=zweiD_Problem(KerberosSampler)
    a.effective_dim()
    v=a.define_variables()
    bqm=a.define_bqm()
    
    a.variables_constraints(2000)
    
    a.geomerie_constraint(1500)
    a.stuecke_position_constraint(1500)
    #a.x_grenze_constraint(300)
    #a.y_grenze_constraint(300)
    
    a.anzahl_objektive(1500)
    
    #a.reste_objektive(200)
    
    #a.prufung()
    
    starttime=datetime.datetime.now()
    solution,qpu_access_time= a.call_bqm_solver(max_iter=3,convergence=2)
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime))
    print(solution,qpu_access_time)
    '''
    starttime=datetime.datetime.now()
    solution= a.klassische_solve(max_iter=3,num_reads=1500,convergence=2)
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime))
    print(solution)
    '''
    a.zeichnung()




      
            