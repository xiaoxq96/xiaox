# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:15:40 2022

@author: Xiaox
"""

import argparse
from dimod import quicksum, BinaryQuadraticModel, Real, Binary, SampleSet,ConstrainedQuadraticModel,Integer
from dwave.system import LeapHybridBQMSampler,LeapHybridCQMSampler
from hybrid.reference.kerberos import KerberosSampler
from itertools import combinations, permutations
import numpy as np
from typing import Tuple
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.pyplot import MultipleLocator
from dwave.system.samplers import DWaveSampler


from unilts1 import read_instance
import warnings
#from output import zeichnung


class eindim_Problem():
    def __init__(self,data = read_instance()):
        self.stueck_ids = np.repeat(data["stueck_ids"], data["quantity"])
        self.num_stueck = np.sum(data["quantity"], dtype=np.int32)
        self.stueck_lange = np.repeat(data["stueck_lange"], data["quantity"])
        print(f'Anzahl der Stücke: {self.num_stueck}')
        self.stange_lange = data["stange_lange"]
        self.num_stange = data["num_Stange"]
        print(f'Anzahl der Stange: {self.num_stange}')
        print(f'Länge der Stange: {self.stange_lange}')
        self.gesamte_stange_lange=self.stange_lange*self.num_stange
        self.gesamte_stueck_lange=sum(self.stueck_lange)
        self.lowest_num_stange = np.ceil(
            np.sum(self.stueck_lange) / (
                    self.stange_lange))
        if self.lowest_num_stange > self.num_stange:
            raise RuntimeError(
                f'anzahl der stangen ist im mindesten {self.lowest_num_stange}'+
                    'try increasing the number of stange'
            )
        print(f'anzahl der stangen ist im mindesten:{self.lowest_num_stange}')
        
        self.solution_x={i:'x_{}'.format(i)for i in range(self.num_stueck)}
        self.solution_p={(i,j):'p_{}_{}'.format(i,j)for i in range(self.num_stueck) for j in range(self.num_stange)}
        self.solution_b={(i,k):'b_{}_{}_{}'.format(i,k,b)
                         for i, k in combinations(range(self.num_stueck), r=2)
                         for b in range(2)}
        self.solution_s={j:'s_{}'.format(j)for j in range(self.num_stange)}
        
    def define_variables(self):
        for i in range(self.num_stueck):
            self.x ={i:Integer(f'x_{i}',
                         lower_bound=0,
                         upper_bound=self.stange_lange-self.stueck_lange[(i)])
                     for i in range(self.num_stueck)}

        self.p = {
           (i, j): Binary(f'p_{i}_{j}') 
           for i in range(self.num_stueck) for j in range(self.num_stange)}


        self.b = {(i, k,b): Binary(f'b_{i}_{k}_{b}')
                        for i, k in combinations(range(self.num_stueck), r=2)
                        for b in range(2)}
        self.s={j: Binary(f's_{j}') for j in range(self.num_stange)}
        return 
    
    
    
    def define_cqm(self):
        self.cqm = ConstrainedQuadraticModel()
        
        return self.cqm
        
    def grenze_constraint(self):
        #stück_i的左端和右端不能超过所在stange的长度
        for i in range(self.num_stueck):
            self.cqm.add_discrete(quicksum([self.p[(i, j)] for j in range(self.num_stange)]),
                             label=f'discrete_{i}')
            '''
            for j in range(self.num_stange):
                
                self.cqm.add_constraint(self.p[(i,j)]*(self.x[(i)] + self.stueck_lange[(i)] - (j+1)*self.stange_lange)<= 0,
                                        label=f'maxx_{i}_{j}_less')
                self.cqm.add_constraint(self.p[(i,j)]*(self.x[(i)]-j*self.stange_lange) >=0,
                    label=f'maxx_{i}_{j}_greater')
                
            '''
                
                
                        
                    
       
        return
                
    def stange_on_constraint(self):
            for j in range(self.num_stange):
                self.cqm.add_constraint((1 - self.s[(j)]) * quicksum(
                [self.p[(i, j)] for i in range(self.num_stueck)]) <= 0,
                               label=f'stueck_on_{j}')
                
                
    def geomerie_constraint(self):
        
        #stück_i和stück_k不能重叠
        for i, k in combinations(range(self.num_stueck),r=2):
            '''
            self.cqm.add_constraint((self.x[(i)]-self.stueck_lange[(k)]-self.x[(k)])*
                                    (self.x[(k)]-self.stueck_lange[(i)]-self.x[(i)]) <=0,
                                    label=f'overlap_{i}_{k}')
            '''
            self.cqm.add_discrete(quicksum([self.b[(i,k,b)] for b in range(2)]),
                             label=f'discrete_{i}_{k}')
            for j in range(self.num_stange):
                self.ik=self.p[(i,j)]*self.p[(k,j)]
                self.cqm.add_constraint(((j-1)*self.stange_lange+self.x[(i)]+self.stueck_lange[(i)]-(j-1)*self.stange_lang*self.x[(k)])-(
                                            (2-self.ik-
                                             self.b[i,k,0])*
                                            self.num_stange*self.stange_lange) <= 0,
                                            label=f'overlap_{i}_{k}_{j}_0')
                self.cqm.add_constraint(((j-1)*self.stange_lange+self.x[(k)]+self.stueck_lange[(k)]-(j-1)*self.stange_lange+self.x[(i)])-(
                                            (2-self.ik-
                                             self.b[i,k,1])*
                                            self.num_stange*self.stange_lange) <= 0,
                    label=f'overlap_{i}_{k}_{j}_1')
             
            
        return
                    
    
                
    def stell_obejektive(self,weight_1,weight_2):
        obj_1= quicksum(self.x[(i)]for i in range(self.num_stueck))
        obj_2= quicksum(self.s[(j)]for j in range(self.num_stange))
        self.cqm.set_objective(weight_1*obj_1+weight_2*obj_2)
        return
                    
    def call_bqm_solver(self):
        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(self.cqm,label="1d_cqm")    #Optimal solution with time_limit 
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)  
        '''
        num_feasible = len(feasible_sampleset)
        if num_feasible > 0:
            self.best_samples = \
                feasible_sampleset.truncate(min(10, num_feasible))
        else:
            warnings.warn("Warning: Did not find feasible solution")
            self.best_samples = raw_sampleset.truncate(10) 
        '''
        self.best_sample = feasible_sampleset.first.sample
        print(self.best_sample)
        self.solution={}
        for key in self.best_sample.keys():
            if key in self.solution_x.values():
                self.solution[key]=self.best_sample[key]
            elif key in self.solution_p.values():
                if self.best_sample[key] !=0:
                    self.solution[key]=self.best_sample[key]
            
            elif key in self.solution_s.values():
                if self.best_sample[key] !=0:
                    self.solution[key]=self.best_sample[key]
            
        return self.solution
    
    
    def zeichnung(self):
        
        self.x_solution={}
        for key in self.solution.keys():
            if key in self.solution_x.values():
                self.x_solution[key]=self.solution[key]
        print(sum(self.x_solution.values()))
        self.p_solution=list(key for key in self.solution.keys() if key in self.solution_p.values())
        self.ids=[]
        self.i_loc=[]
        
        for n in self.p_solution:
            zahlen=re.findall(r"\d+",n)
            ids_x_achse= list(map(int,zahlen))
            self.ids.append(ids_x_achse[0])
            self.i_loc.append(ids_x_achse[1])
        print(self.ids)
        print(self.i_loc)
        
        '''
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
        for i in self.ids:
            txt=f"id{self.stueck_ids[(i)]}"+" "+f"Teil {i}"
            plt.text(self.x_solution[f'x_{i}']+0.5,self.i_loc[i]+0.5,txt,fontsize=8)
            rect=mpatch.Rectangle((self.x_solution[f'x_{i}'],self.i_loc[i]),
                                  self.stueck_lange[(i)],0.25,edgecolor = 'green',facecolor =colors[self.stueck_ids[(i)]],fill=True,alpha=0.5)
            ax.add_patch(rect)
        
        ax.axis([0,self.stange_lange,0,self.num_stange])
        plt.show()
        
    def prufung(self):
    
       
        print(self.cqm.objective)
        #print(self.cqm.constraints[f'maxx_{d}_{e}_greater'].to_polystring())
        #print(self.cqm.constraints[f'maxx_{d}_{e}_greate'].to_polystring())
        #print(self.cqm.constraints[f'overlap_{d}_{f}_{e}_0'].to_polystring())
        #print(self.cqm.constraints[f'overlap_{d}_{f}_{e}_1'].to_polystring())

        
if __name__=="__main__":
    a=eindim_Problem()
    
    v=a.define_variables()
    cqm=a.define_cqm()
    
    

    a.geomerie_constraint()
    a.grenze_constraint()
    a.stange_on_constraint()
    
    #a.stuecke_position_constraint(850)  
    #a.anzhal_objektive(200)
    #a.stange_reihenfolge_constraint(700)
    a.stell_obejektive(5,100)

    
    #a.prufung() 
    
    
    
    
    
    
    '''
    a.variables_constraints(1200)
    
    a.stange_reihenfolge_constraint(827)
    a.stuecke_position_constraint(331)
    
    a.geomerie_constraint(500)
    a.grenze_constraint(10)
    
    a.anzhal_objektive(200)
    a.reste_objektive(200)
    '''
    
    solution= a.call_bqm_solver()
    print(solution)
    
    a.zeichnung()
    