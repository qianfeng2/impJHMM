#!/usr/bin/env python
# coding: utf-8


# Note:
# # In this file, I aim to implement the viterbi algorithm for getting viterbi path.
# # This code is to implement improved Zilversmit model.

# input of this script is protein sequences consisting of target and source (ID starts with db) sequence, output txt, radius and pars.  



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
output_txt=sys.argv[2]
width=int(sys.argv[3])
delta_input=float(sys.argv[4])
eps_input=float(sys.argv[5])
rho_input=float(sys.argv[6])












# emission probs from insert state are from all sequences
seq_dict={}
seq_ID=[]
for seq_record in SeqIO.parse(inputfasta, "fasta"):
    seq_dict[str(seq_record.id)]=str(seq_record.seq)
    seq_ID.append(str(seq_record.id))

nstate_keys = ['?','A', 'R', 'N', 'D','C', 'Q', 'E', 'G','H','I', 'L', 'K', 'M','F', 'P', 'S', 'T', 'W','Y', 'V', 'B', 'Z','X','*']
nstate_values =[0.01]*25
#nstate_keys = ['?','T', 'C', 'A', 'G'];nstate_values = [0.01, 0.01, 0.01, 0.01, 0.01]#0.01 is pseudo-count for preventing zero count
state_frequency_target=dict(zip(nstate_keys, nstate_values))#state_frequency["T"]=0.01
for target in seq_ID:
    for nucle in nstate_keys:
        temp=seq_dict[target].count(nucle)
        state_frequency_target[nucle]=state_frequency_target[nucle]+temp
state_frequency={k: v / sum(state_frequency_target.values()) for k, v in state_frequency_target.items()}#normalization 
print("Insert state emission probabilities:\n",state_frequency,"\n")





# set initial emission probs from match state
#pairstate_frequency= np.array([[0.2, 0.2, 0.2, 0.2, 0.2],[0.2,0.9,0.05,0.025,0.025],[0.2,0.05,0.9,0.025,0.025],[0.2,0.025,0.025,0.9,0.05],[0.2,0.025,0.025,0.05,0.9]])
X=np.ones((24,24))*(0.15/23);X[np.diag_indices_from(X)] = 0.81
X0 = np.ones((24,1))*0.04;Xnew = np.hstack((X0,X))#add first column
X0 = np.ones(((24+1),1))*0.04;pairstate_frequency = np.vstack((np.transpose(X0),Xnew))#add first row
pairstate_frequency= pd.DataFrame(pairstate_frequency, columns=nstate_keys, index=nstate_keys)#print(pairstate_frequency)#pairstate_frequency["T"]["T"]=0.9
print("Match state initial emission probabilities:\n",pairstate_frequency,"\n")
lstate_frequency=dict(zip(nstate_keys, [np.log(i) for i in state_frequency.values()]))
lpairstate_frequency=np.log(pairstate_frequency)  






def kalign_vt(my_data, my_pars_trans, my_pars_emiss,target_ID,RHO,Term):
    delta=my_pars_trans[0];eps=my_pars_trans[1];nsource=len(my_data)
    ldelta=np.log(delta);leps=np.log(eps);
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

    size=0;w_k1_dict={};source_len=[]#compute the number of existing relative positions in all source sequences \iota
    for k in range(nsource):
        l_k=len(seq_dict[my_data[k]]);source_len.append(l_k);r_k1=round(l_k/m+0.01);w_k1=[x for x in range(r_k1-width,r_k1+width+1) if x > 0 and x <=l_k];w_k1_dict[k]=w_k1
        size=size+len(w_k1)
    lsize=np.log(size) if size >0 else SMALL
    ############viterbi############ 
    # (1)define and initialize the first two rows of matrices for each source
    vt_m={};vt_i={};vt_d={}
    tb_m={};tb_i={};tb_d={}
    maxr=SMALL
    max_each_k_dict={};list_store_max=[]
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        tb_temp1=np.zeros((m+1,l_k+1), dtype=object)# tb_m
        tb_temp2=np.zeros((m+1,l_k+1), dtype=object)# tb_i
        tb_temp3=np.zeros((m+1,l_k+1), dtype=object)# tb_d

        temp1=np.ones((m+1,l_k+1))*SMALL# vt_m
        temp2=np.ones((m+1,l_k+1))*SMALL# vt_i
        temp3=np.ones((m+1,l_k+1))*SMALL# vt_d

        maxr_temp2=SMALL
        for inipos_source in w_k1_dict[k]:
            temp1[1][inipos_source]=lfmatch-lsize+lpairstate_frequency.loc[target[0],source[inipos_source-1]]
            temp2[1][inipos_source]=lfinsert-lsize+lstate_frequency[target[0]]
            if temp1[1][inipos_source]>maxr:
                maxr=temp1[1][inipos_source]
                who_max=my_data[k]
                pos_max=inipos_source
                state_max=1            
            if temp2[1][inipos_source]>maxr:
                maxr=temp2[1][inipos_source]
                who_max=my_data[k]
                pos_max=inipos_source
                state_max=2

            if temp1[1][inipos_source]>maxr_temp2:
                maxr_temp2=temp1[1][inipos_source]
                who_max_temp2=my_data[k]
                pos_max_temp2=inipos_source
                state_max_temp2=1
            if temp2[1][inipos_source]>maxr_temp2:
                maxr_temp2=temp2[1][inipos_source]
                who_max_temp2=my_data[k]
                pos_max_temp2=inipos_source
                state_max_temp2=2


            tb_temp1[1][inipos_source]=[1,my_data[k],inipos_source]# in the order of state(1 is M,2 is I,3 is D), who and pos
            tb_temp2[1][inipos_source]=[2,my_data[k],inipos_source]
        max_each_k_dict[k]=[maxr_temp2,who_max_temp2,pos_max_temp2,state_max_temp2]
        list_store_max.append(maxr_temp2)

        for pos_source in range(1,l_k+1):
            if pos_source>1:
                if ldelta+temp1[1][pos_source-1]>=leps+temp3[1][pos_source-1]:
                    temp3[1][pos_source]=ldelta+temp1[1][pos_source-1]     
                    tb_temp3[1][pos_source]=[1,my_data[k],pos_source-1]
                else:
                    temp3[1][pos_source]=leps+temp3[1][pos_source-1]
                    tb_temp3[1][pos_source]=[3,my_data[k],pos_source-1]
        vt_m[k]=temp1
        vt_i[k]=temp2
        vt_d[k]=temp3
        tb_m[k]=tb_temp1
        tb_i[k]=tb_temp2
        tb_d[k]=tb_temp3


    rep=[]
    if sum(map(lambda x : x== maxr, list_store_max))>1:    
        for (key,value) in max_each_k_dict.items():
            if value[0]==maxr:
                rep.append([key,value])
    else:
        list_store_max.sort(reverse = True)
        second_max=list_store_max[1]
        for (key,value) in max_each_k_dict.items():
            if value[0]==second_max:
                rep.append([key,value])
                break



    # (2) loop to finish the matrices
    for i in range(2, m+1):
        #print("i=",i)
        
        count_jump={};count=0;count_removejump={};w_ki_dict={};w_ki_previous_dict={}
        for k in range(nsource):
            l_k=len(seq_dict[my_data[k]])
            r_ki=round(l_k*i/m+0.01);rr_ki=range(r_ki-width,r_ki+width+1)
            w_ki=[x for x in rr_ki if x > 0 and x <= l_k]
            count=count+len(w_ki)
            count_removejump[k]=len(w_ki)
            w_ki_dict[k]=w_ki

            r_ki_previous=round(l_k*(i-1)/m+0.01);rr_ki_previous=range(r_ki_previous-width,r_ki_previous+width+1)
            w_ki_previous=[x for x in rr_ki_previous if x > 0 and x <= l_k]
            w_ki_previous_dict[k]=w_ki_previous

        
        maxr_temp=SMALL;list_store_max=[]
        for k in range(nsource):
            #print("k=",k)
            #if k>0:continue

            temp1=vt_m[k]
            temp2=vt_i[k]
            temp3=vt_d[k]
            source=seq_dict[my_data[k]];l_k=len(source)
            count_jump[k]=np.log(count-count_removejump[k]) if count-count_removejump[k]>0 else SMALL

            tb_temp1=tb_m[k]# tb_m
            tb_temp2=tb_i[k]# tb_i
            tb_temp3=tb_d[k]# tb_d

            
            
            
            if max_each_k_dict[k][0]==maxr:#find biggest value
                if len(rep)==1:
                    maxr=rep[0][1][0]
                    who_max=rep[0][1][1]
                    pos_max=rep[0][1][2]
                    state_max=rep[0][1][3]
                else:
                    repnew=[i for i in rep if i[0] != k]
                    maxr=repnew[0][1][0]
                    who_max=repnew[0][1][1]
                    pos_max=repnew[0][1][2]
                    state_max=repnew[0][1][3]
            

            maxr_temp2=SMALL
            for pos_source in w_ki_dict[k]:#range(1,l_k+1)
                compair_M=[lmm+temp1[i-1][pos_source-1],lim+temp2[i-1][pos_source-1],ldm+temp3[i-1][pos_source-1],lrho+lfmatch+maxr-count_jump[k]]
                compair_I=[ldelta+temp1[i-1][pos_source],leps+temp2[i-1][pos_source],lrho+lfinsert+maxr-count_jump[k]]
                MAX_M=max(compair_M);MAX_I=max(compair_I);
                temp1[i][pos_source]=MAX_M+lpairstate_frequency.loc[target[i-1],source[pos_source-1]]
                temp2[i][pos_source]=MAX_I+lstate_frequency[target[i-1]]
                compair_D=[ldelta+temp1[i][pos_source-1],leps+temp3[i][pos_source-1]]
                MAX_D=max(compair_D)
                temp3[i][pos_source]=MAX_D

                
                if temp1[i][pos_source]>maxr_temp:
                    maxr_temp=temp1[i][pos_source]
                    who_max_temp=my_data[k]
                    pos_max_temp=pos_source
                    state_max_temp=1
                if temp2[i][pos_source]>maxr_temp:
                    maxr_temp=temp2[i][pos_source]
                    who_max_temp=my_data[k]
                    pos_max_temp=pos_source
                    state_max_temp=2

                if temp1[i][pos_source]>maxr_temp2:
                    maxr_temp2=temp1[i][pos_source]
                    who_max_temp2=my_data[k]
                    pos_max_temp2=pos_source
                    state_max_temp2=1
                if temp2[i][pos_source]>maxr_temp2:
                    maxr_temp2=temp2[i][pos_source]
                    who_max_temp2=my_data[k]
                    pos_max_temp2=pos_source
                    state_max_temp2=2
            
                #print("pos_source",pos_source)
                #print(compair_M)
                tb_compair_M=[[1,my_data[k],pos_source-1],[2,my_data[k],pos_source-1],[3,my_data[k],pos_source-1],[state_max,who_max,pos_max]]
                tb_compair_I=[[1,my_data[k],pos_source],[2,my_data[k],pos_source],[state_max,who_max,pos_max]]
                tb_compair_D=[[1,my_data[k],pos_source-1],[3,my_data[k],pos_source-1]]
                tb_temp1[i][pos_source]=tb_compair_M[compair_M.index(MAX_M)]
                tb_temp2[i][pos_source]=tb_compair_I[compair_I.index(MAX_I)]
                tb_temp3[i][pos_source]=tb_compair_D[compair_D.index(MAX_D)]
                #print(compair_M.index(max(compair_M)))
            
            vt_m[k]=temp1
            vt_i[k]=temp2
            vt_d[k]=temp3
            tb_m[k]=tb_temp1
            tb_i[k]=tb_temp2
            tb_d[k]=tb_temp3

            max_each_k_dict[k]=[maxr_temp2,who_max_temp2,pos_max_temp2,state_max_temp2]
            list_store_max.append(maxr_temp2)
        maxr=maxr_temp
        who_max=who_max_temp
        pos_max=pos_max_temp
        state_max=state_max_temp

        rep=[]
        if sum(map(lambda x : x== maxr, list_store_max))>1:    
            for (key,value) in max_each_k_dict.items():
                if value[0]==maxr:
                    rep.append([key,value])
        else:
            list_store_max.sort(reverse = True)
            second_max=list_store_max[1]
            for (key,value) in max_each_k_dict.items():
                if value[0]==second_max:
                    rep.append([key,value])
                    break



    # (3)define Termination v^E=t*max v(m,z_{k}^{j})
    maxr=SMALL
    for k in range(nsource):
        temp1=vt_m[k]
        temp2=vt_i[k]
        source=seq_dict[my_data[k]];l_k=len(source)
        rr_km=range(l_k-width,l_k+width+1);w_km=[x for x in rr_km if x > 0 and x <= l_k]
        for pos_source in w_km:#range(1,l_k+1)
            if temp1[m][pos_source] > maxr:
                maxr=temp1[m][pos_source]
                who_max=my_data[k]
                pos_max=pos_source
                state_max=1
            if temp2[m][pos_source] > maxr:
                maxr=temp2[m][pos_source]
                who_max=my_data[k]
                pos_max=pos_source
                state_max=2








    # step3, trace back to get the ML path    
    maxl=max(source_len)
    cp=2*maxl

    maxpath_state=[SMALL] * (cp+1);maxpath_who=[""] * (cp+1);maxpath_pos=[SMALL] * (cp+1);
    maxpath_state[cp] = state_max
    maxpath_who[cp] = who_max
    maxpath_pos[cp] = pos_max
    pos_target=m
    while pos_target>=1:
        #print("cp=",cp)
        #print("Hidden state is:",state_max, who_max,pos_max)
        #print("pos_target",pos_target,"\n")
        if state_max==1:
            state_next=tb_m[my_data.index(who_max)][pos_target][pos_max][0]
            who_next=tb_m[my_data.index(who_max)][pos_target][pos_max][1]
            pos_next=tb_m[my_data.index(who_max)][pos_target][pos_max][2]
        elif state_max==2:
            state_next=tb_i[my_data.index(who_max)][pos_target][pos_max][0]
            who_next=tb_i[my_data.index(who_max)][pos_target][pos_max][1]
            pos_next=tb_i[my_data.index(who_max)][pos_target][pos_max][2]
        else :
            state_next=tb_d[my_data.index(who_max)][pos_target][pos_max][0]
            who_next=tb_d[my_data.index(who_max)][pos_target][pos_max][1]
            pos_next=tb_d[my_data.index(who_max)][pos_target][pos_max][2]
        cp=cp-1
        if cp<=0:
            print("\n\n ***Error: reconstructed path longer than maximum possible*** \n\n")
            break
        maxpath_state[cp] = state_next
        maxpath_who[cp] = who_next
        maxpath_pos[cp] = pos_next

        state_max=state_next
        who_max=who_next
        pos_max=pos_next
        if maxpath_state[cp+1]!=3: pos_target=pos_target-1




    # step4 print out the mosaic alignment
    ## first print target sequence
    align_target=[];align_target_pos=0
    for i in range(cp+1,2*maxl+1):    
        if maxpath_state[i]!=3 and align_target_pos<=m-1:
            align_target.append(target[align_target_pos])
            align_target_pos=align_target_pos+1
        elif maxpath_state[i]==3:
            align_target.append("-") 
    with open(output_txt, 'a+') as outmidfile:
        outmidfile.write("\n"+"\n"+"\n"+"Target:"+ target_ID+"\t"+ "Length:"+ str(m)+"\t"+"Maximum Log likelihood:"+str(maxr+lTerm)+"\n")
        outmidfile.write('{0: <20}'.format(target_ID)+"\t"+''.join(align_target)+"\n")
    #print("Target:", target_ID,"\t", "Length:", m,"\t","Maximum Log likelihood:",maxr+lTerm,"\n" )
    #print('{0: <20}'.format(target_ID),"\t",''.join(align_target),sep="")

    ## now print match
    align_symbol=[];align_target_pos=0
    for i in range(cp+1,2*maxl+1):   
        if maxpath_state[i]==1 and align_target_pos<=m-1:
            if target[align_target_pos]==seq_dict[maxpath_who[i]][maxpath_pos[i]-1]:align_symbol.append("|")
            else: align_symbol.append(" ")
            align_target_pos=align_target_pos+1
        elif maxpath_state[i]==2:
            align_symbol.append("^")
            align_target_pos=align_target_pos+1
        elif maxpath_state[i]==3:
            align_symbol.append("~")
    #print('{0: <20}'.format(""),"\t",''.join(align_symbol),sep="")
    with open(output_txt, 'a+') as outmidfile:
        outmidfile.write('{0: <20}'.format("")+"\t"+''.join(align_symbol)+"\n")

    with open(output_txt, 'a+') as outmidfile:
        outmidfile.write('{0: <20}'.format(maxpath_who[cp+1])+"\t")    
    for t in range(cp+1,2*maxl+1):
        if t>(cp+1) and maxpath_who[t]!=maxpath_who[t-1]:
            blank=[" "]*(t-(cp+1))
            with open(output_txt, 'a+') as outmidfile:
                outmidfile.write("\n"+'{0: <20}'.format(maxpath_who[t])+"\t"+''.join(blank))
        if maxpath_state[t]==2:
            with open(output_txt, 'a+') as outmidfile:
                outmidfile.write("-")
        else:
            with open(output_txt, 'a+') as outmidfile:
                outmidfile.write(seq_dict[maxpath_who[t]][maxpath_pos[t]-1])
        





#with open("results/we_"+str(width)+"_rho_gss.txt", 'r') as infile:
#    for i, line in enumerate(infile.readlines()):
#        line=line.strip()
#       if i==0:delta_input=float(line)
#        if i==1:eps_input=float(line)
#        if i==2:rho_input=float(line)

#return the viterbi path for all seqs, as each seq is treated as a target.
start=datetime.now()
fmatch=0.75;finsert=1-fmatch;lfmatch=np.log(fmatch);lfinsert=np.log(finsert);SMALL=-1e32
for target_index in range(len(seq_ID)):
    targetID=seq_ID[target_index];target=seq_dict[targetID]#define target
    sourceID=[item for item in seq_ID if item not in [targetID]]
    #nsource=len(sourceID)#define source set
    kalign_vt(sourceID, [delta_input,eps_input], pairstate_frequency,targetID,rho_input,0.001)
end=datetime.now()
timediff=end-start   

with open(output_txt, 'a+') as outmidfile:
    outmidfile.write("\n"+"\n"+"Above all viterbi paths with estimated parameters cost "+str(timediff.total_seconds())+"secs")







