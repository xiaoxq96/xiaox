# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:15:40 2022

@author: Xiaox
"""

from dwave.system.samplers import LeapHybridBQMSampler
from dwave.system import DWaveSampler, EmbeddingComposite,DWaveCliqueSampler
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import SteepestDescentSampler
from dwave.samplers import TabuSampler
import datetime
import dimod
import numpy as np
from hybrid import traits
from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from hybrid.core import Runnable, SampleSet
import hybrid
import minorminer

from collections import defaultdict
import numpy as np



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

import time
import logging
import threading
from collections import namedtuple


from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from dwave.preprocessing.composites import SpinReversalTransformComposite

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler
from greedy import SteepestDescentSolver
from minorminer.busclique import busgraph_cache

from hybrid.core import Runnable, SampleSet
from hybrid.flow import Loop
from hybrid.utils import random_sample
from hybrid import traits

import minorminer
from hybrid.core import State

from dwave.system import FixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding



from unilts1 import read_instance

class SubproblemCliqueEmbedder(traits.SubproblemIntaking,
                               traits.EmbeddingProducing,
                               traits.SISO, Runnable):
    """Subproblem-induced-clique embedder on sampler's target graph.

    Args:
        sampler (:class:`dimod.Structured`):
            Structured :class:`dimod.Sampler` such as a
            :class:`~dwave.system.samplers.DWaveSampler`.

    Example:
        To replace :class:`.QPUSubproblemAutoEmbeddingSampler` with a sampler
        that uses fixed clique embedding (adapted to subproblem on each run),
        use ``SubproblemCliqueEmbedder | QPUSubproblemExternalEmbeddingSampler``
        construct::

            from dwave.system import DWaveSampler

            qpu = DWaveSampler()
            qpu_branch = (
                hybrid.EnergyImpactDecomposer(size=50)
                | hybrid.SubproblemCliqueEmbedder(sampler=qpu)
                | hybrid.QPUSubproblemExternalEmbeddingSampler(qpu_sampler=qpu))
    """

    def __init__(self, sampler, **runopts):
        super(SubproblemCliqueEmbedder, self).__init__(**runopts)
        self.sampler = sampler

    def __repr__(self):
        return "{self}(sampler={self.sampler!r})".format(self=self)

    @staticmethod
    def find_clique_embedding(variables, sampler):
        """Given a :class:`dimod.Structured` ``sampler``, and a list of
        variable labels, return a clique embedding.

        Returns:
            dict:
                Clique embedding map with source variables from ``variables``
                and target graph taken from ``sampler``.

        """
        g = sampler.to_networkx_graph()
        return busgraph_cache(g).find_clique_embedding(variables)

    def next(self, state, **runopts):
        embedding = self.find_clique_embedding(
            state.subproblem.variables, self.sampler)
        return state.updated(embedding=embedding)




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
class ReverseAnnealingSampler(dimod.Sampler):
    
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
    def ReverseAnnealing(self,sampler,max_iter=100, max_time=None, convergence=3, energy_threshold=None,
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
            
            sampler=self.ReverseAnnealingAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            self.runnable = self.ReverseAnnealing(sampler,max_iter, max_time, convergence, energy_threshold,sa_reads, sa_sweeps, tabu_timeout,qpu_reads, qpu_sampler, qpu_params,
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
        


class eindim_Problem():
    def __init__(self,ReverseAnnealingSampler,KerberosSampler,QPUSubproblemExternalEmbeddingSampler,SubproblemCliqueEmbedder,data = read_instance()):
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
        
        
        self.ReverseAnnealingSampler=ReverseAnnealingSampler
        self.KerberosSampler=KerberosSampler
        self.QPUSubproblemExternalEmbeddingSampler=QPUSubproblemExternalEmbeddingSampler
        self.SubproblemCliqueEmbedder=SubproblemCliqueEmbedder
        
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
    
    def make_reverse_anneal_schedule(self,s_target=0.0, hold_time=10.0, ramp_back_slope=0.2, ramp_up_time=0.0201,
                                     ramp_up_slope=None):
        """This should be rewritten"""
        """Build annealing waveform pattern for reverse anneal feature.
        Waveform starts and ends at s=1.0, descending to a constant value
        s_target in between, following a linear ramp.
          s_target:   s-parameter to descend to (between 0 and 1)
          hold_time:  amount of time (in us) to spend at s_target (must be >= 2.0us)
          ramp_slope: slope of transition region, in units 1/us
        """
        # validate parameters
        if s_target < 0.0 or s_target > 1.0:
            raise ValueError("s_target must be between 0 and 1")
        if hold_time < 0.0:
            raise ValueError("hold_time must be >= 0")
        if ramp_back_slope > 0.2:
            raise ValueError("ramp_back_slope must be <= 0.2")
        if ramp_back_slope <= 0.0:
            raise ValueError("ramp_back_slope must be > 0")
        ramp_time = (1.0 - s_target) / ramp_back_slope
        initial_s = 1.0
        pattern = [[0.0, initial_s]]
        # don't add new points if s_target == 1.0
        if s_target < 1.0:
            pattern.append([round(ramp_time, 4), round(s_target, 4)])
            if hold_time != 0:
                pattern.append([round(ramp_time+hold_time, 4), round(s_target, 4)])
        # add last point
        if ramp_up_slope is not None:
            ramp_up_time = (1.0-s_target)/ramp_up_slope
            pattern.append([round(ramp_time + hold_time + ramp_up_time, 4), round(1.0, 4)])
        else:
            pattern.append([round(ramp_time + hold_time + ramp_up_time, 4), round(1.0, 4)])
        return pattern  
                    
    def call_bqm_solver(self,max_iter,convergence,num_reads,sa_reads=1,sa_sweeps=10000,max_subproblem_size=50):
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
        sampler=DWaveSampler()
        max_slope = 1.0/sampler.properties["annealing_time_range"][0]
        print(sampler.properties["annealing_time_range"][0])
        reverse_schedule=self.make_reverse_anneal_schedule(s_target=0.4, hold_time=180, ramp_up_slope=max_slope)
        
        time_total = reverse_schedule[3][0]
        forward_answer= self.KerberosSampler().sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=sampler,qpu_params={'label': 'JSSP_bqm_iter_reverse_forward'},qpu_reads=num_reads,sa_reads=sa_reads,sa_sweeps=sa_sweeps,max_subproblem_size=max_subproblem_size)
        forward_solutions, forward_energies = forward_answer.record.sample, forward_answer.record.energy
        i5 = int(5.0/95*len(forward_answer))  # Index i5 is about a 5% indexial move from the sample of lowest energy
        initial = dict(zip(forward_answer.variables, forward_answer.record[i5].sample))
        reverse_anneal_params = dict(anneal_schedule=reverse_schedule,initial_state=initial, reinitialize_state=False,label='JSSP_bqm_iter_reverse')        
        sampler,qpu_access_time2 = self.ReverseAnnealingSampler.sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=DWaveSampler(),anneal_schedule= reverse_schedule,qpu_params=reverse_anneal_params,qpu_reads=num_reads,sa_reads=sa_reads,sa_sweeps=sa_sweeps,max_subproblem_size=max_subproblem_size)
        samplerSET=sampler.samples()
        self.solution={}
        for key in samplerSET:
            sampler =key
        for key in sampler:
            if sampler[key] !=0 and "slack" not in key:
                self.solution[key]=sampler[key]
        
        qpu_access_times=0
        qpu_access_times+=qpu_access_time1+qpu_access_time2
        
        '''
        self.ReverseAnnealingAutoEmbeddingSampler=ReverseAnnealingAutoEmbeddingSampler
        workflow = hybrid.RacingBranches(
            hybrid.BlockingIdentity(),
            hybrid.SimulatedAnnealingProblemSampler(num_reads=50),
            hybrid.TabuProblemSampler(num_reads=50),hybrid.EnergyImpactDecomposer(size=2)
            |self.ReverseAnnealingAutoEmbeddingSampler(num_reads=100, anneal_schedule= reverse_schedule, qpu_sampler=sampler,sampling_params=reverse_anneal_params)
            | hybrid.SplatComposer() )| hybrid.MergeSamples()
        init_state = hybrid.State.from_sample(initial,self.bqm)
        final_state,qpu_access_time2= workflow.run(init_state).result()
        self.solution={}
        for key in final_state.samples.first.sample:            
            if final_state.samples.first.sample[key] !=0:
               self.solution[key]=final_state.samples.first.sample[key]
        
        qpu_access_times=0
        qpu_access_times+=qpu_access_time1+qpu_access_time2
        '''
        return self.solution,qpu_access_times


# 定义CSP模型。
    
# 求解
    
    def pureRA(self,s_target,hold_time):#,max_iter,convergence,num_reads,sa_reads=1,sa_sweeps=10000,max_subproblem_size=50):
        sampler=DWaveSampler()
        max_slope = 1.0/sampler.properties["annealing_time_range"][0]
        reverse_schedule=self.make_reverse_anneal_schedule(s_target=s_target, hold_time=hold_time, ramp_up_slope=max_slope)
        
        time_total = reverse_schedule[3][0]

        embedding=minorminer.find_embedding(list(self.bqm.quadratic.keys()), sampler.edgelist)
        sampler_emb=FixedEmbeddingComposite(DWaveSampler(),embedding=embedding)

        forward_answer = TabuSampler().sample(self.bqm,num_reads=1)
        forward_solutions, forward_energies = forward_answer.record.sample, forward_answer.record.energy
        i5 = int(5.0/95*len(forward_answer))  # Index i5 is about a 5% indexial move from the sample of lowest energy
        initial = dict(zip(forward_answer.variables, forward_answer.record[i5].sample))
        reverse_anneal_params = dict(anneal_schedule=reverse_schedule,initial_state=initial, reinitialize_state=False,label='JSSP_bqm_iter_reverse')
        samplerSET_forward=forward_answer
        samplerSET=sampler_emb.sample(bqm=self.bqm,num_reads=1000,anneal_schedule=reverse_schedule,initial_state=initial, reinitialize_state=False,label='JSSP_bqm_iter_reverse')
        
        """
        iteration =( hybrid.IdentityDecomposer() | 
                   hybrid.SubproblemCliqueEmbedder(sampler=DWaveSampler())
                    | hybrid.QPUSubproblemExternalEmbeddingSampler(qpu_sampler=DWaveSampler()) |
                     hybrid.SplatComposer() )
        
        workflow = hybrid.LoopUntilNoImprovement(iteration, max_iter=1)
        init_state = hybrid.State.from_sample(hybrid.random_sample(self.bqm),self.bqm)
        final_state = workflow.run(init_state).result()
        """
        #iteration = (
            #hybrid.IdentityDecomposer()
            #| self.SubproblemCliqueEmbedder(sampler=sampler)
            #| hybrid.QPUSubproblemExternalEmbeddingSampler(qpu_sampler=sampler,sampling_params=reverse_anneal_params)
            #| hybrid.SplatComposer())
        
        self.solution_forward={}
        for key in samplerSET_forward.first.sample:            
            if samplerSET_forward.first.sample[key] !=0:
               self.solution_forward[key]=samplerSET_forward.first.sample[key]
        self.energy_forward = samplerSET_forward.first.energy
        self.solution={}
        for key in samplerSET.first.sample:            
            if samplerSET.first.sample[key] !=0:
               self.solution[key]=samplerSET.first.sample[key]
        self.energy = samplerSET.first.energy
        
        #qpu_access_times=0
        #qpu_access_times+=qpu_access_time1+qpu_access_time2
       
        return self.solution,self.solution_forward,self.energy_forward,self.energy#,qpu_access_times
    
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
                txt=f"id {self.stueck_ids[(i)]}"+":"+f"Nr {i}"
                plt.text(x % self.stange_lange+0.5,j+0.4,txt,fontsize=8)
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
        plt.savefig('1dsa.png',format='png',bbox_inches = 'tight')
        plt.show()
        
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
    ReverseAnnealingSampler=ReverseAnnealingSampler()
    
    #starttime=datetime.datetime.now()
    #print("Starttime:",starttime)
    a=eindim_Problem(ReverseAnnealingSampler,KerberosSampler,QPUSubproblemExternalEmbeddingSampler,SubproblemCliqueEmbedder)
    v=a.define_variables()
    bqm=a.define_bqm()
    
    a.variables_constraints(2)

    a.geomerie_constraint(1)
    a.stuecke_position_constraint(1)
    
    a.position_objektive(0.001)
    a.anzhal_objektive(0.9)
    
    #a.prufung() 
    #a.runtime()
    #solution=a.define_csp()
    #print(solution)
    
    
    
    
    
    
    '''
    a.variables_constraints(1200)
    
    a.stange_reihenfolge_constraint(827)
    a.stuecke_position_constraint(331)
    
    a.geomerie_constraint(500)
    a.grenze_constraint(10)
    
    a.anzhal_objektive(200)
    a.reste_objektive(200)
    '''
    
    
    starttime=datetime.datetime.now()
    ecxel_hold_time={}
    for hold_time in [1]:
        excel_s_target={}
        excel_s_target['hold_time']=hold_time
        excel_s_target['s_target']=[]
        excel_s_target['energy_forward']=[]
        excel_s_target['energy']=[]
        for s_target in [0.45]:
            solution,solution_forward,energy_forward,energy= a.pureRA(s_target,hold_time)
            endtime=datetime.datetime.now()
            print ("starttime:",starttime)
            print ("endtime:",endtime)
            print ("runtime:",(endtime-starttime))
            print('solution:',solution)
            print('solution_forward:',solution_forward)
            print('energy_forward',energy_forward)
            excel_s_target['energy_forward'].append(energy_forward)
            print('energy',energy)
            excel_s_target['energy'].append(energy)
            print(s_target)
            excel_s_target['s_target'].append(s_target)
        print(excel_s_target)
        ecxel_hold_time[hold_time] = excel_s_target
    print(ecxel_hold_time)
    
    
    '''
    starttime=datetime.datetime.now()
    solution= a.klassische_solve(max_iter=1,convergence=1)
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime).seconds)
    print(solution)
    '''
    
    a.zeichnung()
    
