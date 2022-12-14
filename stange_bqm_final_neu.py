# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:15:40 2022

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
from dwave.preprocessing.composites import SpinReversalTransformComposite


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
from matplotlib.pyplot import MultipleLocator


from unilts1 import read_instance
#from output import zeichnung

class QPUSubproblemExternalEmbeddingSampler(traits.SubproblemSampler,
                                            traits.EmbeddingIntaking,
                                            traits.SISO, Runnable):
    r"""A quantum sampler for a subproblem with a defined minor-embedding.
    Note:
        Externally supplied embedding must be present in the input state.
    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler()` ):
            Quantum sampler such as a D-Wave system.
        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (external-embedding-wrapped QPU) sampler.
        logical_srt (int, optional, default=False):
            Perform a spin-reversal transform over the logical space.
    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, sampling_params=None,
                 logical_srt=False, **runopts):
        super(QPUSubproblemExternalEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        self.logical_srt = logical_srt
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

        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        if self.logical_srt:
            params.update(num_spin_reversal_transforms=1)
            sampler = SpinReversalTransformComposite(sampler)
        response = sampler.sample(state.subproblem, **params)
        self.qpu_access_time+=response.info['timing']['qpu_access_time']*(0.001)
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
        self.QPUSubproblemExternalEmbeddingSampler=QPUSubproblemExternalEmbeddingSampler
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
            external_sampler=self.QPUSubproblemExternalEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            sampler=hybrid.SubproblemCliqueEmbedder(sampler=qpu_sampler,) | external_sampler
            #self.QPUSubproblemAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
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
            
            return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype, energy=energies),external_sampler.qpu_access_time


class eindim_Problem():
    def __init__(self,KerberosSampler,data = read_instance()):
        self.stueck_ids = np.repeat(data["stueck_ids"], data["quantity"])
        self.num_stueck = np.sum(data["quantity"], dtype=np.int32)
        self.stueck_lange = np.repeat(data["stueck_lange"], data["quantity"])
        print(f'Anzahl der Stücke: {self.num_stueck}')
        self.stange_lange = data["stange_lange"]
        self.num_stange = data["num_Stange"]
        print(f'Anzahl der Stange: {self.num_stange}')
        print(f'Länge der Stange: {self.stange_lange}')
        self.gesamte_stange_lange=self.stange_lange*self.num_stange
        self.lowest_num_stange = np.ceil(
            np.sum(self.stueck_lange) / (
                    self.stange_lange))
        if self.lowest_num_stange > self.num_stange:
            raise RuntimeError(
                f'anzahl der stangen ist im mindesten {self.lowest_num_stange}'+
                    'try increasing the number of stange'
            )
        print(f'anzahl der stangen ist im mindesten:{self.lowest_num_stange}')
        
        
        self.KerberosSampler=KerberosSampler
        
        self.S_j={}
        #self.U_ij={}
        self.X_ija={}
        #self.Q_ik={}
        
    def define_variables(self):
        #self.S_j={(j):'s_{}'.format(j)for j in range(self.num_stange)}
        #S_j没有被用
        self.X_ija={(i,j,a):'x_{}_{}_{}'.format(i,j,a)for i in range(self.num_stueck)
                   for j in range(self.num_stange)
                   for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1)}
        self.S_j={(j):'s_{}'.format(j)for j in range(self.num_stange)}
        #stück_i左端坐标是a
        self.variables=[self.X_ija,self.S_j]
        '''[self.S_j,self.U_ij,self.X_ia,self.Q_ik]'''
        return #self.variables
    
    
    
    def define_bqm(self):
        self.bqm=BinaryQuadraticModel('BINARY')
        
        for i in self.variables:      
            for j in i.values():
                self.bqm.add_variable(j)
        return self.bqm
        
                
    def geomerie_constraint(self,weight):
        #stück_i和stück_k不能重叠
        for i, k in combinations(range(self.num_stueck),r=2):
            for j in range(self.num_stange):
                for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1):
                    for b in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(k)]+1):
                        if (a-self.stueck_lange[(k)]) < b < (a+self.stueck_lange[(i)]):
                            self.bqm.add_quadratic(self.X_ija[(i,j,a)],self.X_ija[(k,j,b)],weight*self.gesamte_stange_lange)
                        #elif a<b and (b-a)<self.stueck_lange[(i)]:
                            #self.bqm.add_quadratic(self.X_ija[(i,j,a)],self.X_ija[(k,j,b)],weight)
            '''
            slear=[(self.X_ia[(i,a)], -(a+self.stueck_lange[i]))for a in range(self.gesamte_stange_lange+1)]
            slebr=[(self.X_ia[(k,b)], b)for b in range(self.gesamte_stange_lange+1)]
            sleal=[(self.X_ia[(i,a)], a)for a in range(self.gesamte_stange_lange+1)]
            slebl=[(self.X_ia[(k,b)], -(b+self.stueck_lange[k]))for b in range(self.gesamte_stange_lange+1)]
            ikbr=[(self.Q_ik[(i,k)], -self.gesamte_stange_lange)]
            ikbl=[(self.Q_ik[(i,k)],self.gesamte_stange_lange)]
            self.bqm.add_linear_inequality_constraint(slear+slebr+ikbr,
                                                              lagrange_multiplier=weight,
                                                              lb=0,
                                                              constant=self.gesamte_stange_lange,
                                                              label='geometrie_rechts')
            self.bqm.add_linear_inequality_constraint(sleal+slebl+ikbl,
                                                              lagrange_multiplier=weight,
                                                              lb=0,
                                                              label='geometrie_links')
            '''
        return
                    
    def variables_constraints(self,weight):
        for i in range(self.num_stueck):
                '''
                self.bqm.add_linear_equality_constraint([(self.U_ij[(i,j)],1)for j in range(self.num_stange)],
                                                                lagrange_multiplier=weight, 
                                                                constant=-1)
                '''
                self.bqm.add_linear_equality_constraint([(self.X_ija[(i,j,a)],1)for j in range(self.num_stange)
                                                         for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1)],
                                                                lagrange_multiplier=weight*self.gesamte_stange_lange, 
                                                                constant=-1)
                
                
        return
                        
    def stuecke_position_constraint(self,weight):
        #stück_i只能在已经启用的stang_j上切割
       
        for i in range(self.num_stueck):
            for j in range(self.num_stange):
                for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1):
                            self.bqm.add_quadratic(self.S_j[(j)],self.X_ija[(i,j,a)],weight*self.gesamte_stange_lange)
        '''
        for j in range(self.num_stange):
            for i in range(self.num_stueck):
                
                self.bqm.add_quadratic(self.S_j[(j)],self.U_ij[(i,j)],-weight)
                self.bqm.add_linear(self.U_ij[(i,j)],weight)
        
        
                
                self.bqm.add_linear_inequality_constraint([(self.S_j[(j)],1),(self.U_ij[(i,j)],-1)],
                                                          lagrange_multiplier=weight,
                                                          lb=0,
                                                          label='geometrie_rechts')
                '''
                
        return
        
                    
    def position_objektive(self,weight):
        #所用最少的stange
        bias={}
        
            
        for i in range(self.num_stueck):
            for j in range(self.num_stange):
                for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1):
                    bias[self.X_ija[(i,j,a)]]=(a+self.stueck_lange[(i)])*weight
            #self.bqm.add_linear(self.X_ija[i,j,a],a)
            
        self.bqm.add_linear_from(bias)
        return
                
    def anzhal_objektive(self,weight):
        #所用面积最小
        for j in range(self.num_stange):
            self.bqm.add_linear(self.S_j[(j)],-weight*self.gesamte_stange_lange)
        return
                    
    def call_bqm_solver(self,max_iter,convergence):
        '''
        sampler = LeapHybridBQMSampler(token='EbWg-1dc4f5bb6bc7384895496a5dff589cc676d6e1a9')
        raw_sampleset = sampler.sample(self.bqm,label="Stange_Problem")
        best_sample = raw_sampleset.first.sample
        self.solution={}
        for key in best_sample:
            if best_sample[key] !=0 and "slack" not in key:
                self.solution[key]=best_sample[key]
        return self.solution
        '''
        sampler,qpu_access_time= KerberosSampler.sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=DWaveSampler(),qpu_params={'label': 'stange_bqm'}, qpu_reads=1500)
        samplerSET=sampler.samples()
        self.solution={}
        for key in samplerSET:
            sampler =key
        for key in sampler:
            if sampler[key] !=0 and "slack" not in key:
                self.solution[key]=sampler[key]
        
        qpu_access_times=0
        qpu_access_times+=qpu_access_time
        return self.solution,qpu_access_times
    
    def klassische_solve(self,max_iter,convergence):
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
        self.eff_daten=list(key for key in self.solution.keys() if key in self.X_ija.values())
        print(self.eff_daten)
        self.ids=[]
        self.i_loc=[]
        self.x_achse=[]
        
        for n in self.eff_daten:
            zahlen=re.findall(r"\d+",n)
            ids_x_achse= list(map(int,zahlen))
            self.ids.append(ids_x_achse[0])
            self.i_loc.append(ids_x_achse[1])
            self.x_achse.append(ids_x_achse[2])
        print(self.ids)
        print(self.i_loc)
        print(self.x_achse)
        '''
        self.x_rest=[]
        for i in range(self.num_stueck):
            self.x_rest.append(self.x_achse[(i)]%self.stange_lange)
        '''
        

        fig = plt.figure()
        ax = fig.add_subplot()
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.title("1D-CSP")
        ticks=[]
        labels=[]
        for j in range(self.num_stange):
            ticks.append(j)
            labels.append('stang_' + str(j+1))
        plt.yticks(ticks,labels)
        ids = []
        for i in self.stueck_ids:
            if i not in ids:
                ids.append(i)
        #ids=list(set(self.stueck_ids))
        langes=[]
        for i in self.stueck_lange:
            if i not in langes:
                langes.append(i)
        #langes.sort(key=self.stueck_lange.index)
        n=len(ids)
        colors=plt.cm.jet(np.linspace(0, 1, n))
        legend_i=[]
        for i in ids:
            legend_i.append(i)
            legend_i[i] = mpatch.Patch(color=colors[i], label=f'id{i}:{langes[i]} cm',alpha=0.5)
        stange_legend=mpatch.Patch(color="tab:gray", label=f'Stange:{self.stange_lange} cm',alpha=0.3)
        legend_i.append(stange_legend)
        ax.legend(handles=legend_i,bbox_to_anchor=(1,0.8),loc='center left')
        for j in range (self.num_stange):
            rect=mpatch.Rectangle((0,j),self.stange_lange,0.25,color="tab:gray",alpha=0.3)
            ax.add_patch(rect)
        for x,i,j in zip(self.x_achse,self.ids,self.i_loc):
                txt=f"{self.stueck_ids[(i)]}"+":"+f"{i}"
                plt.text(x % self.stange_lange+0.5,j+0.1,txt,fontsize=8)
                rect=mpatch.Rectangle((x % self.stange_lange,j),self.stueck_lange[(i)],0.25,edgecolor = 'green',facecolor =colors[self.stueck_ids[(i)]],fill=True,alpha=0.5)
                ax.add_patch(rect)
        '''
        ticks=[]
        labels=[]
        for j in range(self.num_stange):
            ticks.append(j*self.stange_lange)
            labels.append('stang_' + str(j+1))
        plt.yticks(ticks,labels)
        '''
        ax.axis([0,self.stange_lange,0,self.num_stange])
        plt.show()
        plt.savefig('1d.png',bbox_inches = 'tight')
        
    def prufung(self):
    
        
            
                    #print(self.bqm.get_linear(self.X_ia[(i,a)]))
                    
                    #for j in range(self.num_stange):
                        #print(self.bqm.get_quadratic(self.X_ia[(1,12)],self.X_ia[(6,28)]))
                        for i in range(1):
                            for j in range(self.num_stange):
                                for a in range(j*self.stange_lange,(j+1)*self.stange_lange-self.stueck_lange[(i)]+1):
                                    print(self.bqm.get_linear(self.X_ija[(i,j,a)]))
                        #print(self.bqm.get_linear(self.S_j[(j)]))
                                    #print(self.bqm.get_quadratic(self.S_j[(j)],self.X_ia[(i,a)]))
                                    
    def runtime(self):
        starttime=datetime.datetime.now()
        endtime=datetime.datetime.now()
        print("Starttime:",starttime)
        print("endtime:",endtime)
        print (endtime-starttime)
        

        
if __name__=="__main__":
    KerberosSampler=KerberosSampler()
    #starttime=datetime.datetime.now()
    #print("Starttime:",starttime)
    a=eindim_Problem(KerberosSampler)
    v=a.define_variables()
    bqm=a.define_bqm()
    
    a.variables_constraints(2)

    a.geomerie_constraint(1)
    a.stuecke_position_constraint(1)
    
    a.position_objektive(0.2)
    a.anzhal_objektive(0.9)
    
    #a.prufung() 
    #a.runtime()
    
    
    
    
    
    
    '''
    a.variables_constraints(1200)
    
    a.stange_reihenfolge_constraint(827)
    a.stuecke_position_constraint(331)
    
    a.geomerie_constraint(500)
    a.grenze_constraint(10)
    
    a.anzhal_objektive(200)
    a.reste_objektive(200)
    '''
    
    '''
    starttime=datetime.datetime.now()
    solution,qpu_access_time= a.call_bqm_solver(max_iter=6,convergence=6)
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime))
    print(solution,qpu_access_time)
    
    
    '''
    starttime=datetime.datetime.now()
    solution= a.klassische_solve(max_iter=6,convergence=6)
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime).seconds)
    print(solution)
    
    
    a.zeichnung()
    
