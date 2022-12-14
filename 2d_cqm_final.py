# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:15:40 2022

@author: Xiaox
"""

import argparse
import datetime
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


from unilts import read_instance
import warnings
#from output import zeichnung


class zweidim_Problem():
    def __init__(self,data = read_instance()):
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
        
        self.solution_x={i:'x_{}'.format(i)for i in range(self.num_stueck)}
        self.solution_y={i:'y_{}'.format(i)for i in range(self.num_stueck)}
        self.solution_p={(i,j):'p_{}_{}'.format(i,j)for i in range(self.num_stueck) for j in range(self.num_platte)}
        self.solution_d={(i,r):'d_{}_{}'.format(i,r)for i in range(self.num_stueck) for r in range(2)}
        self.solution_b={(i,k):'b_{}_{}_{}'.format(i,k,b)
                         for i, k in combinations(range(self.num_stueck), r=2)
                         for b in range(4)}
        
    def define_variables(self):
        self.x ={i:Integer(f'x_{i}',
                         lower_bound=0,
                         upper_bound=self.num_platte*self.platte_lange)
                     for i in range(self.num_stueck)}
        self.y ={i:Integer(f'y_{i}',
                         lower_bound=0,
                         upper_bound=self.platte_breite)
                     for i in range(self.num_stueck)}

        self.p = {
           (i, j): Binary(f'p_{i}_{j}') 
           for i in range(self.num_stueck) for j in range(self.num_platte)}
        
        self.d={(i,r):Binary(f'd_{i}_{r}')for i in range(self.num_stueck) for r in range(2)}


        self.b = {(i, k,b): Binary(f'b_{i}_{k}_{b}')
                        for i, k in combinations(range(self.num_stueck), r=2)
                        for b in range(4)}
        self.pl={
           j: Binary(f'pl{j}') 
           for j in range(self.num_platte)}
        return 
    
    def effective_dim(self):
        for i in range(self.num_stueck):
            self.cqm.add_discrete(quicksum([self.d[(i, r)] for r in range(2)]),
                         label=f'discrete_Drehrung_{i}')
        self.eff_dim_x={}
        self.eff_dim_y={}
        for i in range(self.num_stueck):
            p1=list(permutations([self.stueck_lange[(i)],self.stueck_breite[(i)]]))
            self.eff_dim_x[i]=0
            self.eff_dim_y[i]=0
            for r, (a, b) in enumerate(p1):
                self.eff_dim_x[i] += a * self.d[(i,r)]
                self.eff_dim_y[i] += b * self.d[(i,r)]
        return [self.eff_dim_x,self.eff_dim_y]
    
    
    
    def define_cqm(self):
        self.cqm = ConstrainedQuadraticModel()
        
        return self.cqm
        
    def grenze_constraint(self):
        #stück_i的左端和右端不能超过所在stange的长度
        for i in range(self.num_stueck):
            self.cqm.add_discrete(quicksum([self.p[(i, j)] for j in range(self.num_platte)]),
                             label=f'discrete_{i}')
            
            for j in range(self.num_platte):
                    '''
                self.cqm.add_constraint(self.p[(i,j)]*(self.x[(i)] + self.stueck_lange[(i)] - (j+1)*self.stange_lange)<= 0,
                                        label=f'maxx_{i}_{j}_less')
                self.cqm.add_constraint(self.p[(i,j)]*(self.x[(i)]-j*self.stange_lange) >=0,
                    label=f'maxx_{i}_{j}_greater')
                    '''
                    self.cqm.add_constraint((self.x[(i)] + self.eff_dim_x[i]) - (j+1)*self.platte_lange-
                                        ((1-self.p[(i,j)])*self.gesamte_platte_lange)<= 0,
                                        label=f'maxx_{i}_{j}_less')
                    self.cqm.add_constraint((
                    self.x[(i)] - self.platte_lange * j * self.p[(i, j)]) >= 0,
                    label=f'maxx_{i}_{j}_greater')
                    self.cqm.add_constraint(
                    self.y[(i)]+ self.eff_dim_y[i] - self.platte_breite  <= 0,
                    label=f'maxx_{i}_{j}_y')
                    
                    
                
                
                        
                    
       
        return
                
    def platte_on_constraint(self):
            for j in range(self.num_platte):
                self.cqm.add_constraint((1 - self.pl[(j)]) * quicksum(
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
            self.cqm.add_discrete(quicksum([self.b[(i,k,b)] for b in range(4)]),
                             label=f'discrete_{i}_{k}')
            for j in range(self.num_platte):
                        self.ik=self.p[(i,j)]*self.p[(k,j)]
                        self.cqm.add_constraint((self.x[(i)]+self.eff_dim_x[i]-self.x[(k)])-(
                                            (2-self.ik-
                                             self.b[i,k,0])*
                                            self.num_platte*self.platte_lange) <= 0,
                                            label=f'overlap_{i}_{k}_{j}_0')
                        self.cqm.add_constraint((self.x[(k)]+self.eff_dim_x[k]-self.x[(i)])-(
                                            (2-self.ik-
                                             self.b[i,k,1])*
                                            self.num_platte*self.platte_lange) <= 0,
                                            label=f'overlap_{i}_{k}_{j}_1')
                        self.cqm.add_constraint((self.y[(i)]+self.eff_dim_y[i]-self.y[(k)])-(
                                            (2-self.ik-
                                             self.b[i,k,2])*
                                            self.platte_breite) <= 0,
                                            label=f'overlap_{i}_{k}_{j}_2')
                        self.cqm.add_constraint((self.y[(k)]+self.eff_dim_y[k]-self.y[(i)])-(
                                            (2-self.ik-
                                             self.b[i,k,3])*
                                            self.platte_breite) <= 0,
                                            label=f'overlap_{i}_{k}_{j}_3')
             
            
        return
                    
    
                        
    
        
    
                    
    def anzhal_objektive(self,weight_1):
        #所用最少的stange
        
        obj_1= quicksum(self.pl[(j)] for j in range(self.num_platte))
        self.cqm.set_objective(weight_1*obj_1)
        return
                
    
                    
    def call_bqm_solver(self):
        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(self.cqm,label="2d_cqm_20")    #Optimal solution with time_limit 
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)      
        num_feasible = len(feasible_sampleset)
        if num_feasible > 0:
            self.best_samples = \
                feasible_sampleset.truncate(min(10, num_feasible))
        else:
            warnings.warn("Warning: Did not find feasible solution")
            self.best_samples = raw_sampleset.truncate(10)    
        self.best_sample = self.best_samples.first.sample
        print(self.best_sample)
        self.solution={}
        for key in self.best_sample.keys():
            if key in self.solution_x.values():
                self.solution[key]=self.best_sample[key]
            elif key in self.solution_y.values():
                self.solution[key]=self.best_sample[key]
            elif key in self.solution_p.values():
                if self.best_sample[key] !=0:
                    self.solution[key]=self.best_sample[key]
            elif key in self.solution_d.values():
                if self.best_sample[key] !=0:
                    self.solution[key]=self.best_sample[key]
        return self.solution
    
    
    def zeichnung(self):
        self.x_solution={}
        self.y_solution={}
        self.p_solution={}
        self.d_solution={}
        
        for key in self.solution.keys():
            if key in self.solution_x.values():
                self.x_solution[key]=self.solution[key]
        for key in self.solution.keys():
            if key in self.solution_y.values():
                self.y_solution[key]=self.solution[key]
        for key in self.solution.keys():
            if key in self.solution_p.values():
                self.p_solution[key]=self.solution[key]
        for key in self.solution.keys():
            if key in self.solution_d.values():
                self.d_solution[key]=self.solution[key]
        
        
        self.ids=[]
        self.i_loc=[]
        self.i_um=[]
        for n in self.p_solution:
            zahlen_n=re.findall(r"\d+",n)
            ids_loc= list(map(int,zahlen_n))
            for m in self.d_solution:
                zahlen_m=re.findall(r"\d+",m)
                ids_um=list(map(int,zahlen_m))
                if ids_loc[0] == ids_um[0]:
                    self.ids.append(ids_loc[0])
                    self.i_loc.append(ids_loc[1])
                    self.i_um.append(ids_um[1])
        print(self.ids)
        print(self.i_loc)
        print(self.i_um)
        
        self.eff_dim={}
        for i in range(self.num_stueck):
            self.eff_dim[i]={}
            p1=list(permutations([self.stueck_lange[(i)],self.stueck_breite[(i)]]))
            self.eff_dim[i][0]=p1[(0)]
            self.eff_dim[i][1]=p1[(1)]
        print(self.eff_dim)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        n=self.num_stueck
        colors=plt.cm.jet(np.linspace(0, 1, n))
        for i in self.ids:
            txt=f"Teil {i}"
            plt.text(self.x_solution[f'x_{i}']+0.5,self.y_solution[f'y_{i}']+0.5,txt,fontsize=8)
            rect=mpatch.Rectangle((self.x_solution[f'x_{i}'],self.y_solution[f'y_{i}']),
                                  self.eff_dim[i][self.i_um[i]][0],self.eff_dim[i][self.i_um[i]][1],edgecolor = 'green',facecolor =colors[self.stueck_ids[(i)]],fill=True,alpha=0.5)
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
        for i in range(self.num_stueck):
            txt=f"id{self.stueck_ids[(i)]}"+" "+f"Teil {i}"
            plt.text(self.x_solution[f'x_{i}'] % self.stange_lange+0.5,int(self.x_solution[f'x_{i}']/self.stange_lange)+0.5,txt,fontsize=8)
            rect=mpatch.Rectangle((self.x_solution[f'x_{i}'] % self.stange_lange,int(self.x_solution[f'x_{i}']/self.stange_lange)),
                                  self.stueck_lange[(i)],0.25,edgecolor = 'green',facecolor =colors[self.stueck_ids[(i)]],fill=True,alpha=0.5)
            ax.add_patch(rect)
        
        ax.axis([0,self.stange_lange,0,self.num_stange])
        plt.show()
        '''
    def prufung(self):
    
        d=1
        e=15
        f=7  
        print(self.cqm.constraints[f'overlap_{d}_{e}'].to_polystring())
        #print(self.cqm.constraints[f'maxx_{d}_{e}_greater'].to_polystring())
        #print(self.cqm.constraints[f'maxx_{d}_{e}_greate'].to_polystring())
        #print(self.cqm.constraints[f'overlap_{d}_{f}_{e}_0'].to_polystring())
        #print(self.cqm.constraints[f'overlap_{d}_{f}_{e}_1'].to_polystring())

        
if __name__=="__main__":
    a=zweidim_Problem()
   
    v=a.define_variables()
    
    cqm=a.define_cqm()
    
    
    a.effective_dim()
    a.geomerie_constraint()
    a.grenze_constraint()
    a.platte_on_constraint()
    
    #a.stuecke_position_constraint(850)  
    a.anzhal_objektive(800)
    #a.stange_reihenfolge_constraint(700)

    
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
    starttime=datetime.datetime.now()
    solution= a.call_bqm_solver()
    endtime=datetime.datetime.now()
    print ("starttime:",starttime)
    print ("endtime:",endtime)
    print ("runtime:",(endtime-starttime))
    print(solution)
    
    a.zeichnung()
    