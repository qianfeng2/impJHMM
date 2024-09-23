#!/usr/bin/env python
# coding: utf-8


# Note:
# # In this file, I aim to estimate \rho using forward algorithm when other parameters are known.
# # This code is to implement improved Zilversmit model.

# input of this script consists of six parameters: inputfasta(protein sequences,target and db sequences), output_txtfile, reffasta, radius, delta and epsilon



import numpy as np
import pandas as pd
from Bio import SeqIO
import glob
import os
import sys
import math
from datetime import datetime
import csv
pd.set_option("display.precision", 8)





# step 1 define initial parameters, inputdata and transition/emission probabilities
inputfasta=sys.argv[1]
outputfile=sys.argv[2]
reffasta=sys.argv[3]#used for subsampling. All target sequences in this file is used for determining emission probs of insert.
width=int(sys.argv[4])
delta_input=float(sys.argv[5])
eps_input=float(sys.argv[6])











# emission probs from insert state are from all target sequences from reference fasta file
seq_dict_ref={};seq_ID_ref=[]
for seq_record in SeqIO.parse(reffasta, "fasta"):
    if "target" in str(seq_record.id):
        seq_dict_ref[str(seq_record.id)]=str(seq_record.seq)
        seq_ID_ref.append(str(seq_record.id))
# emission probs from insert state are from target sequence composition, DNA or aa
nstate_keys = ['?','A', 'R', 'N', 'D','C', 'Q', 'E', 'G','H','I', 'L', 'K', 'M','F', 'P', 'S', 'T', 'W','Y', 'V', 'B', 'Z','X','*']
nstate_values =[0.01]*25
#nstate_keys = ['?','T', 'C', 'A', 'G'];nstate_values = [0.01, 0.01, 0.01, 0.01, 0.01]#0.01 is pseudo-count for preventing zero count
state_frequency_target=dict(zip(nstate_keys, nstate_values))#state_frequency["T"]=0.01
for target in seq_ID_ref:
    for nucle in nstate_keys:
        temp=seq_dict_ref[target].count(nucle)
        state_frequency_target[nucle]=state_frequency_target[nucle]+temp
state_frequency={k: v / sum(state_frequency_target.values()) for k, v in state_frequency_target.items()}#normalization 
print("Insert state emission probabilities:\n",state_frequency,"\n")


seq_dict={}
seq_ID=[]
for seq_record in SeqIO.parse(inputfasta, "fasta"):
    if "target" in str(seq_record.id):
        seq_dict[str(seq_record.id)]=str(seq_record.seq)
        seq_ID.append(str(seq_record.id))
#add source database
sourceID=[]
for seq_record in SeqIO.parse(inputfasta, "fasta"):
    if "db" in str(seq_record.id):
        seq_dict[str(seq_record.id)]=str(seq_record.seq)
        sourceID.append(str(seq_record.id))

    
# set initial emission probs from match state
#pairstate_frequency= np.array([[0.2, 0.2, 0.2, 0.2, 0.2],[0.2,0.9,0.05,0.025,0.025],[0.2,0.05,0.9,0.025,0.025],[0.2,0.025,0.025,0.9,0.05],[0.2,0.025,0.025,0.05,0.9]])
X=np.ones((24,24))*(0.15/23);X[np.diag_indices_from(X)] = 0.81
X0 = np.ones((24,1))*0.04;Xnew = np.hstack((X0,X))#add first column
X0 = np.ones(((24+1),1))*0.04;pairstate_frequency = np.vstack((np.transpose(X0),Xnew))#add first row
pairstate_frequency= pd.DataFrame(pairstate_frequency, columns=nstate_keys, index=nstate_keys)#print(pairstate_frequency)#pairstate_frequency["T"]["T"]=0.9
print("Match state initial emission probabilities:\n",pairstate_frequency,"\n")
lstate_frequency=dict(zip(nstate_keys, [np.log(i) for i in state_frequency.values()]))
lpairstate_frequency=np.log(pairstate_frequency)





def calculate_llk_over_rho_grid(my_data, my_pars_trans, my_pars_emiss,target_ID,RHO,Term):
    delta=my_pars_trans[0];eps=my_pars_trans[1];nsource=len(my_data)
    ldelta=np.log(delta);leps=np.log(eps)
    mm=1-2*delta-RHO-Term
    mi=md=delta
    im=1-eps-RHO-Term
    ii=dd=eps
    dm=1-eps
    lTerm=np.log(Term);lrho=np.log(RHO)
    lmm=np.log(mm)
    lmi=lmd=ldelta
    lim=np.log(im)
    lii=ldd=leps
    ldm=np.log(dm)    
    
    lpairstate_frequency=np.log(my_pars_emiss)
    
    target=seq_dict[target_ID]
    m=len(target)
    
    size=0;w_k1_dict={}#compute the number of existing relative positions in all source sequences \iota
    for k in range(nsource):
        l_k=len(seq_dict[my_data[k]]);r_k1=round(l_k/m+0.01);w_k1=[x for x in range(r_k1-width,r_k1+width+1) if x > 0 and x <= l_k];w_k1_dict[k]=w_k1
        size=size+len(w_k1)   
    lsize=np.log(size) if size >0 else SMALL
    ############ forward############ 
    # (1)define and initialize the first two rows of matrices for each source
    vt_m={};vt_i={};vt_d={};max_r=SMALL
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)

        temp1=np.ones((m+1,l_k+1))*SMALL# vt_m
        temp2=np.ones((m+1,l_k+1))*SMALL# vt_i
        temp3=np.ones((m+1,l_k+1))*SMALL# vt_d
    
        for inipos_source in w_k1_dict[k]:
            temp1[1][inipos_source]=lfmatch-lsize+lpairstate_frequency.loc[target[0],source[inipos_source-1]]
            temp2[1][inipos_source]=lfinsert-lsize+lstate_frequency[target[0]]
            if temp1[1][inipos_source]>max_r:max_r=temp1[1][inipos_source]
            if temp2[1][inipos_source]>max_r:max_r=temp2[1][inipos_source]
        for pos_source in w_k1_dict[k]:
            if pos_source>1:
                #temp3[1][pos_source]=temp1[1][pos_source-1]+np.log(delta+eps*math.exp(temp3[1][pos_source-1]-temp1[1][pos_source-1]))
                temp3[1][pos_source]=np.log(math.exp(ldelta+temp1[1][pos_source-1])+math.exp(leps+temp3[1][pos_source-1])) 
                if temp3[1][pos_source]>max_r:max_r=temp3[1][pos_source]
             
        vt_m[k]=temp1
        vt_i[k]=temp2
        vt_d[k]=temp3
            
    # (2) loop to finish the matrices
    for i in range(2, m+1):
        #print("i=",i)
        max_rn=SMALL
        count_jump={};count=0;count_removejump={}
        llk_jump={};llk_r=0;llk_removejump={}
        w_ki_dict={}
        for k in range(nsource):
            l_k=len(seq_dict[my_data[k]])
            r_ki=round(l_k*i/m+0.01);rr_ki=range(r_ki-width,r_ki+width+1);w_ki=[x for x in rr_ki if x >0 and x <= l_k]
            count=count+len(w_ki)
            w_ki_dict[k]=w_ki
            count_removejump[k]=len(w_ki)

            temp4=vt_m[k]
            temp5=vt_i[k]
            r_ti=round(l_k*(i-1)/m+0.01);rr_ti=range(r_ti-width,r_ti+width+1);w_ti=[x for x in rr_ti if x > 0 and x <= l_k]
            rec_contri=0
            for pos_source_t in w_ti:
                temp_temp=math.exp(temp4[i-1][pos_source_t]-max_r)+math.exp(temp5[i-1][pos_source_t]-max_r)
                llk_r=llk_r+temp_temp
                rec_contri=rec_contri+temp_temp
            llk_removejump[k]=rec_contri
            


    
        for k in range(nsource):
            temp1=vt_m[k]
            temp2=vt_i[k]
            temp3=vt_d[k]
            source=seq_dict[my_data[k]];l_k=len(source)
            count_jump[k]=count-count_removejump[k] if count-count_removejump[k]>0 else SMALL
            llk_jump[k]=llk_r-llk_removejump[k] if llk_r-llk_removejump[k]>0 else SMALL
                    
            for pos_source in w_ki_dict[k]:#range(1,l_k+1)
                compair_M=[math.exp(temp1[i-1][pos_source-1]-max_r)*mm,math.exp(temp2[i-1][pos_source-1]-max_r)*im,math.exp(temp3[i-1][pos_source-1]-max_r)*dm,RHO*fmatch*llk_jump[k]/count_jump[k]]
                compair_I=[math.exp(temp1[i-1][pos_source]-max_r)*delta,math.exp(temp2[i-1][pos_source]-max_r)*eps,RHO*finsert*llk_jump[k]/count_jump[k]]
                SUM_M=sum(compair_M);SUM_I=sum(compair_I);
                temp1[i][pos_source]=np.log(SUM_M)+lpairstate_frequency.loc[target[i-1],source[pos_source-1]]+max_r
                temp2[i][pos_source]=np.log(SUM_I)+lstate_frequency[target[i-1]]+max_r
                if pos_source>1 and i<m:
                    compair_D=[math.exp(temp1[i][pos_source-1]-max_r)*delta,math.exp(temp3[i][pos_source-1]-max_r)*eps]
                    SUM_D=sum(compair_D)
                    if SUM_D==0:temp3[i][pos_source]=SMALL+max_r
                    else:temp3[i][pos_source]=np.log(SUM_D)+max_r
                
                if temp1[i][pos_source]>max_rn: max_rn=temp1[i][pos_source]
                if temp2[i][pos_source]>max_rn: max_rn=temp2[i][pos_source]
            
            vt_m[k]=temp1
            vt_i[k]=temp2
            vt_d[k]=temp3
        max_r=max_rn

    # (3)define termination v^E=t*Sum v(m,z_{k}^{j})
    forward=0
    for k in range(nsource):
        l_k=len(seq_dict[my_data[k]])
        temp1=vt_m[k]
        temp2=vt_i[k]
        rr_km=range(l_k-width,l_k+width+1);w_km=[x for x in rr_km if x > 0 and x <= l_k]
        for pos_source in w_km:
            forward=forward+math.exp(temp1[m][pos_source]-max_r)
            forward=forward+math.exp(temp2[m][pos_source]-max_r) 
    return np.log(forward)+lTerm+max_r-np.log(nsource)#discussed with Dr.Yao-ban, we should normalize it by the number of source





#return the rho with maximum composite likelihood


#with open("results/nonjump_we_"+str(width)+".txt", 'r') as infile:
#    for i, line in enumerate(infile.readlines()):
#        line=line.strip()
#        if i==0:delta_input=float(line)
#        if i==1:eps_input=float(line)
rho_input_min=0.0001;
rho_input_max=min(1-eps_input-0.001,1-2*delta_input-0.001)-0.0001



start=datetime.now()
fmatch=0.75;finsert=1-fmatch;lfmatch=np.log(fmatch);lfinsert=np.log(finsert);SMALL=-1e32
#golden section search starts here
##def function named sum_llk for a particular rho
def sum_llk(var_rho):
    llk_all_targets=[];
    for target_index in range(len(seq_ID)):
        targetID=seq_ID[target_index];target=seq_dict[targetID]#define target
        nsource=len(sourceID)#define source set
        llk=calculate_llk_over_rho_grid(sourceID, [delta_input,eps_input], pairstate_frequency,targetID,var_rho,0.001)#term is 0.001, same with Zilversmit et al (2013)
        llk_all_targets.append(llk)
    return(sum(llk_all_targets))


a = rho_input_min; b = rho_input_max; gr = (math.sqrt(5) + 1) / 2
c = b - (b - a) / gr; d = a + (b - a) / gr
fc=sum_llk(c);fd=sum_llk(d);
while abs(b - a) > 1e-4:
    if fc > fd:
        b = d
        d = c
        fd = fc
        c = b - (b - a) / gr
        fc=sum_llk(c)
    else:
        a = c
        c = d
        fc = fd
        d = a + (b - a) / gr
        fd = sum_llk(d)
rho_best = (b+a)/2
end=datetime.now()
timediff=end-start

with open(outputfile, 'w') as outfile:
    outfile.write(str(delta_input)+ "\n"+str(eps_input)+"\n"+str(rho_best)+"\n"+str(timediff.total_seconds()))



   







