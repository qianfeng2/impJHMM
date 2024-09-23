#!/usr/bin/env python
# coding: utf-8


# Note:
# # In this file, I aim to implement the Baum-Welch algorithm for estimating \delta and \eps when setting \rho=0.
# # This code is to implement improved Zilversmit model.

# input of this script is protein sequences, consisting of target and db sequences. Each target will be searched across all source (db) sequences. 




import numpy as np
import pandas as pd
from Bio import SeqIO
import glob
import os
import sys
import math
from datetime import datetime
pd.set_option("display.precision", 8)





# step 1 define initial parameters, inputdata and transition/emission probabilities
inputfasta=sys.argv[1]
outputfile=sys.argv[2]
reffasta=sys.argv[3]#used for subsampling. All target sequences in this file is used for determining emission probs of insert.
width=int(sys.argv[4])
delta=0.025
eps=0.75
term=0.001
fmatch=0.75
SMALL=-1e32

lterm=np.log(term)
lfmatch=np.log(fmatch)
finsert=1-fmatch
lfinsert=np.log(finsert)



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





def kalign_fb_no_rec(my_data, my_pars_trans, my_pars_emiss,target_ID):
    delta=my_pars_trans[0];eps=my_pars_trans[1]
    ldelta=np.log(delta);leps=np.log(eps)
    mm=1-2*delta
    mi=md=delta
    im=1-eps
    ii=dd=eps
    dm=1-eps

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
        l_k=len(seq_dict[my_data[k]]);r_k1=round(l_k/m+0.01);w_k1=[x for x in range(r_k1-width,r_k1+width+1) if x > 0 and x<= l_k];w_k1_dict[k]=w_k1
        size=size+len(w_k1)        
    lsize=np.log(size) if size >0 else SMALL
    ############ forward############  
    # (1)define and initialize the first two rows of matrices for each source
    forward_m={};forward_i={};forward_d={};max_r=SMALL
    for k in range(nsource):
        #print(my_data[k])
        source=seq_dict[my_data[k]]
        l_k=len(source)
    
        temp1=np.ones((m+1,l_k+1))*SMALL# vt_m
        temp2=np.ones((m+1,l_k+1))*SMALL# vt_i
        temp3=np.ones((m+1,l_k+1))*SMALL# vt_d
    
        for inipos_source in w_k1_dict[k]:
            temp1[1][inipos_source]=lfmatch-lsize+lpairstate_frequency.loc[target[0],source[inipos_source-1]]
            temp2[1][inipos_source]=lfinsert-lsize+lstate_frequency[target[0]]
        
        for pos_source in w_k1_dict[k]:#for pos_source in range(1,l_k+1):
            if pos_source>1:
                #temp3[1][pos_source]=temp1[1][pos_source-1]+np.log(delta+eps*math.exp(temp3[1][pos_source-1]-temp1[1][pos_source-1]))
                temp3[1][pos_source]=np.log(math.exp(ldelta+temp1[1][pos_source-1])+math.exp(leps+temp3[1][pos_source-1])) 
            
            
        forward_m[k]=temp1
        forward_i[k]=temp2
        forward_d[k]=temp3
        
    # (2)loop to finish the matrices
    for i in range(2, m+1):
        for k in range(nsource):            
            temp1=forward_m[k]
            temp2=forward_i[k]
            temp3=forward_d[k]
            source=seq_dict[my_data[k]];l_k=len(source);r_ki=round(l_k*i/m+0.01);rr_ki=range(r_ki-width,r_ki+width+1)
            w_ki=[x for x in rr_ki if x > 0 and x <= l_k]        
                        
            for pos_source in w_ki:#for pos_source in range(1,l_k+1):
                #match
                max_r = temp1[i-1][pos_source-1];
                if (temp2[i-1][pos_source-1]>max_r): max_r = temp2[i-1][pos_source-1];
                if (temp3[i-1][pos_source-1]>max_r): max_r = temp3[i-1][pos_source-1];
                temp1[i][pos_source] =  math.exp(temp1[i-1][pos_source-1]-max_r)*mm;
                temp1[i][pos_source] += math.exp(temp2[i-1][pos_source-1]-max_r)*im;
                temp1[i][pos_source] += math.exp(temp3[i-1][pos_source-1]-max_r)*dm;
                temp1[i][pos_source] = np.log(temp1[i][pos_source])+max_r;
                temp1[i][pos_source] += lpairstate_frequency.loc[target[i-1],source[pos_source-1]];
                
                #insert
                max_r = temp1[i-1][pos_source];
                if (temp2[i-1][pos_source]>max_r): max_r = temp2[i-1][pos_source];
                temp2[i][pos_source] =  math.exp(temp1[i-1][pos_source]-max_r)*delta;
                temp2[i][pos_source] += math.exp(temp2[i-1][pos_source]-max_r)*eps;
                temp2[i][pos_source] = np.log(temp2[i][pos_source])+max_r;
                temp2[i][pos_source] += lstate_frequency[target[i-1]];

                
                #compair_M=[math.exp(lmm+temp1[i-1][pos_source-1]),math.exp(lim+temp2[i-1][pos_source-1]),math.exp(ldm+temp3[i-1][pos_source-1])]
                #compair_I=[math.exp(lmi+temp1[i-1][pos_source]),math.exp(lii+temp2[i-1][pos_source])]
                #q1=sum(compair_M)*my_pars_emiss.loc[target[i-1],source[pos_source-1]]
                #temp1[i][pos_source]=np.log(q1) if q1>0 else SMALL
                #q2=sum(compair_I)*state_frequency[target[i-1]]
                #temp2[i][pos_source]=np.log(q2) if q2>0 else SMALL
                
                #delete
                if pos_source>1 and i<m:
                    max_r = temp3[i][pos_source-1];
                    if (temp1[i][pos_source-1]>max_r): max_r = temp1[i][pos_source-1];
                    temp3[i][pos_source] =  math.exp(temp3[i][pos_source-1]-max_r)*eps;
                    temp3[i][pos_source] += math.exp(temp1[i][pos_source-1]-max_r)*delta;
                    temp3[i][pos_source] = np.log(temp3[i][pos_source]) + max_r;
                    
                    #compair_D=[math.exp(lmd+temp1[i][pos_source-1]),math.exp(ldd+temp3[i][pos_source-1])]
                    #temp3[i][pos_source]=np.log(sum(compair_D)) if sum(compair_D) >0 else SMALL
                    
            forward_m[k]=temp1
            forward_i[k]=temp2
            forward_d[k]=temp3
            
    # (3)define termination f^E=t*Sum v(m,z_{k}^{j})
    forward=0.0;max_r=SMALL
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        temp1=forward_m[k];temp2=forward_i[k]
        rr_km=range(l_k-width,l_k+width+1);w_km=[x for x in rr_km if x > 0 and x <= l_k]        
        for pos_source in w_km:#for pos_source in range(1,l_k+1):
            if temp1[m][pos_source]>max_r:max_r=temp1[m][pos_source]
            if temp2[m][pos_source]>max_r:max_r=temp2[m][pos_source]
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        temp1=forward_m[k];temp2=forward_i[k]
        rr_km=range(l_k-width,l_k+width+1);w_km=[x for x in rr_km if x > 0 and x <= l_k]        
        for pos_source in w_km:#for pos_source in range(1,l_k+1):
            forward+= math.exp(temp1[m][pos_source]-max_r)
            forward+= math.exp(temp2[m][pos_source]-max_r) 

                
    llk[target_ID]=np.log(forward)+lterm+max_r
    print("Sequence",target_ID,"\t", "Log likelihood from forward algorithm  = ", "{:.5f}".format(llk[target_ID]))
    #print("Forward matrices  = ",  forward_m,"\n",forward_i,"\n",forward_d)
    
    
    
    ############ backward############
    # (1)define and initialize the three matrices for each source
    backward_m={};backward_i={};backward_d={}
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        rr_km=range(l_k-width,l_k+width+1);w_km=[x for x in rr_km if x > 0 and x <= l_k]
    
        temp1=np.ones((m+2,l_k+2))*SMALL# m
        temp2=np.ones((m+2,l_k+2))*SMALL# i
        temp3=np.ones((m+2,l_k+2))*SMALL# d
        for pos_source in w_km:#for pos_source in range(1,l_k+1):
            temp1[m][pos_source]=np.log(term)
            temp2[m][pos_source]=np.log(term)
        
        backward_m[k]=temp1
        backward_i[k]=temp2
        backward_d[k]=temp3
    # (2)loop to finish the matrices
    for i in range(m-1,0,-1):
        for k in range(nsource):            
            temp1=backward_m[k]
            temp2=backward_i[k]
            temp3=backward_d[k]
            source=seq_dict[my_data[k]];l_k=len(source)
            r_ki=round(l_k*i/m+0.01);rr_ki=range(r_ki-width,r_ki+width+1);w_ki=[x for x in rr_ki if x > 0 and x <= l_k] 
            w_ki.reverse()            
            for pos_source in w_ki:#for pos_source in range(l_k,0,-1):
                if pos_source<l_k:
                    #Delete
                    max_r = temp3[i][pos_source+1];
                    if (temp1[i+1][pos_source+1]>max_r): max_r = temp1[i+1][pos_source+1];
                    temp3[i][pos_source] =  math.exp(temp3[i][pos_source+1]-max_r)*eps;
                    temp3[i][pos_source] += math.exp(temp1[i+1][pos_source+1]-max_r)*my_pars_emiss.loc[target[i],source[pos_source]]*dm;
                    temp3[i][pos_source] = np.log(temp3[i][pos_source]) + max_r;
                    #Insert
                    max_r = temp2[i+1][pos_source];
                    if (temp1[i+1][pos_source+1]>max_r): max_r = temp1[i+1][pos_source+1];
                    temp2[i][pos_source] =  math.exp(temp2[i+1][pos_source]-max_r)*ii*state_frequency[target[i]];
                    temp2[i][pos_source] += math.exp(temp1[i+1][pos_source+1]-max_r)*im*my_pars_emiss.loc[target[i],source[pos_source]];
                    temp2[i][pos_source] = np.log(temp2[i][pos_source])+max_r;

                    #Match
                    max_r = temp1[i+1][pos_source+1];
                    if (temp2[i+1][pos_source]>max_r): max_r = temp2[i+1][pos_source];
                    if (temp3[i][pos_source+1]>max_r): max_r = temp3[i][pos_source+1];
                    temp1[i][pos_source] =  math.exp(temp1[i+1][pos_source+1]-max_r)*mm*my_pars_emiss.loc[target[i],source[pos_source]];
                    temp1[i][pos_source] += math.exp(temp2[i+1][pos_source]-max_r)*delta*state_frequency[target[i]];
                    temp1[i][pos_source] += math.exp(temp3[i][pos_source+1]-max_r)*delta;
                    temp1[i][pos_source] = np.log(temp1[i][pos_source])+max_r;
                           
                #else:
                elif pos_source==l_k:
                    temp1[i][l_k]=np.log(delta)+lstate_frequency[target[i]]+temp2[i+1][pos_source]
                    temp2[i][l_k]=np.log(eps)+lstate_frequency[target[i]]+temp2[i+1][pos_source];  
                
                    
            backward_m[k]=temp1
            backward_i[k]=temp2
            backward_d[k]=temp3
    # (3)define termination 
    backward=0.0;max_r=SMALL
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        temp1=backward_m[k]
        temp2=backward_i[k]
        for pos_source in w_k1_dict[k]:
            if temp1[1][pos_source]>max_r:max_r=temp1[1][pos_source]
            if temp2[1][pos_source]>max_r:max_r=temp2[1][pos_source]
    
    for k in range(nsource):
        source=seq_dict[my_data[k]]
        l_k=len(source)
        temp1=backward_m[k]
        temp2=backward_i[k]
        for pos_source in w_k1_dict[k]:
            backward+=math.exp(temp1[1][pos_source]-max_r)*fmatch*my_pars_emiss.loc[target[0],source[pos_source-1]]/size
            backward+=math.exp(temp2[1][pos_source]-max_r)*finsert*state_frequency[target[0]]/size
    llk_b = max_r + np.log(backward)
    print("Sequence",target_ID,"\t", "Log likelihood from backward algorithm  = ", "{:.5f}".format(llk_b))#check the llk should be the same from forward and backward algorithms
    #print("Backward matrices  = ",  backward_m,"\n",backward_i,"\n",backward_d)
    

    for k in range(nsource):
        source=seq_dict[my_data[k]];l_k=len(source)
        for pos_target in range(1,m+1):
            r_ki=round(l_k*pos_target/m+0.01);rr_ki=range(r_ki-width,r_ki+width+1);w_ki=[x for x in rr_ki if x > 0 and x <= l_k]            
            for pos_source in w_ki:#for pos_source in range(1,l_k+1):
                row=nstate_keys.index(source[pos_source-1])
                col=nstate_keys.index(target[pos_target-1])
                #expected_emissions[row][col]+=math.exp(forward_m[k][pos_target][pos_source]+backward_m[k][pos_target][pos_source]-llk[target_ID])
                expected_emissions[col][row]+=math.exp(forward_m[k][pos_target][pos_source]+backward_m[k][pos_target][pos_source]-llk[target_ID])
                
                #Match-match
                x = forward_m[k][pos_target-1][pos_source-1]+backward_m[k][pos_target][pos_source]+lmm+lpairstate_frequency.loc[target[pos_target-1],source[pos_source-1]]-llk[target_ID];
                expected_transitions[0][0] += math.exp(x);
                
                #Match-insert
                x = forward_m[k][pos_target-1][pos_source]+backward_i[k][pos_target][pos_source]+ldelta+lstate_frequency[target[pos_target-1]]-llk[target_ID];
                expected_transitions[0][1] += math.exp(x);

                #Match-delete
                x = forward_m[k][pos_target][pos_source-1]+backward_d[k][pos_target][pos_source]+ldelta-llk[target_ID];
                expected_transitions[0][2] += math.exp(x);

                #Insert-match
                x = forward_i[k][pos_target-1][pos_source-1]+backward_m[k][pos_target][pos_source]+lim+lpairstate_frequency.loc[target[pos_target-1],source[pos_source-1]]-llk[target_ID];
                expected_transitions[1][0] += math.exp(x);

                #Insert-insert
                x = forward_i[k][pos_target-1][pos_source]+backward_i[k][pos_target][pos_source]+leps+lstate_frequency[target[pos_target-1]]-llk[target_ID];
                expected_transitions[1][1] += math.exp(x);

                #Delete-match
                x = forward_d[k][pos_target][pos_source-1]+backward_m[k][pos_target][pos_source]+ldm+lpairstate_frequency.loc[target[pos_target-1],source[pos_source-1]]-llk[target_ID];
                expected_transitions[2][0] += math.exp(x);

                #Delete-delete
                x = forward_d[k][pos_target][pos_source-1]+backward_d[k][pos_target][pos_source]+leps-llk[target_ID];
                expected_transitions[2][2] += math.exp(x);
                
    #print("Sequence",target_ID,"\t","expected transitions:",expected_transitions)
    #print("Sequence",target_ID,"\t","expected emissions:",expected_emissions)





# step 3 do iteration
start=datetime.now()

MAX_IT_EM=10#stop rule is number of iterations exceeds MAX_IT_EM or change in llk is less than LLK_TOL.
LLK_TOL=0.01
iteration=1
tol=10000
current_llk=SMALL
nstate=len(nstate_keys)
llk={}
while iteration<=MAX_IT_EM and tol>LLK_TOL:
    start_iter=datetime.now()
    combined_llk=0.0
    expected_transitions=np.ones((3,3))*0.01# 0.01 again, pseudo count
    expected_emissions=np.ones((nstate,nstate))*0.01# in the order of ?TCAG
    for target_index in range(len(seq_ID)):
        targetID=seq_ID[target_index];target=seq_dict[targetID]#define target
        nsource=len(sourceID)#define source set
        kalign_fb_no_rec(sourceID, [delta,eps], pairstate_frequency,targetID)
      
    #for target_index in range(len(seq_ID)):
        #targetID=seq_ID[target_index];target=seq_dict[targetID]#define target
        #sourceID=[item for item in seq_ID if item not in [targetID]]
        #nsource=len(sourceID)#define source set
        #kalign_fb_no_rec(sourceID, [delta,eps], pairstate_frequency,targetID)
            
    
    #Update emission from match parameters
    sm=expected_emissions/expected_emissions.sum(axis=0)#divide by colsum
    pairstate_frequency= pd.DataFrame(sm, columns=nstate_keys, index=nstate_keys)
    
    #Update transition parameters
    delta=(expected_transitions[0][1]+expected_transitions[0][2])/(2*(expected_transitions[0][1]+expected_transitions[0][2]+expected_transitions[0][0]))
    eps=(expected_transitions[1][1]+expected_transitions[2][2])/(expected_transitions[1][1]+expected_transitions[2][2]+expected_transitions[1][0]+expected_transitions[2][0]+0.02)
    combined_llk=sum(llk.values())
    end_iter=datetime.now()
    timediff_iter=end_iter-start_iter
    print("Iteration = ",iteration,"\t","Combined log likelihood = ",combined_llk)
    print("This iteration cost ",timediff_iter.total_seconds(),"secs")
    print("Parameters: gap insertion = ",delta,"\t","gap extension = ",  eps,"\n")
    #print("This iteration expected transitions = ",expected_transitions,"\n")
    #print("Emission from match",pairstate_frequency)
        
    diff=combined_llk-current_llk;tol=diff
    current_llk=combined_llk
    iteration+=1

end=datetime.now()
timediff=end-start   
print("\n\nEM estimation completed")
print("Parameters:")
print("Transition: gap insertion = ",delta,"\t","gap extension = ", eps)
print("Emission parameters from match:")
print(pairstate_frequency)#row represents target,col represents source
print("Program completed in ",timediff.total_seconds(),"secs")

with open(outputfile, 'w') as outfile:
    outfile.write(str(delta)+ "\n"+str(eps)+"\n"+str(iteration)+"\n"+str(timediff.total_seconds())+"\n"+str(timediff.total_seconds()/(iteration-1)))













