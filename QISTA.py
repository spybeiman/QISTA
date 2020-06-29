# In this code, I compare the performance between ISTA and QISTA.
# However, ISTA can be accelerated via Nesterov's acceleration,
# the result is FISTA, and, QISTA also can be accelerated,
# the result is called FQISTA.
# Note that FQISTA is only considered in this code, not appeared
# in the paper. FQISTA is for checking the performance of QISTA
# with the lesser iteration number.
# 
# If this is your first time, this code will show you that both
# QISTA and FQISTA have the same reconstruction result (not the same
# parameter setting within the paper), the code will take a while.
# After that, you can check QISTA by using FQISTA.

first_time = True

print(chr(27) + "[2J")
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################
############ parameter setting ############
###########################################
n,m,k = 1024,256,64
q = 0.05
epsilon = 1.0 * torch.ones(n,1)
soft_lambda = 1e-5
q_lambda = 1e-4
###########################################
###########################################
###########################################


signal_var = 1.0
signal_std = math.sqrt(signal_var)
if first_time == True:
    run_time = 1
    max_iter_ISTA = 100000
    max_iter_FISTA = 100000
    max_iter_QISTA = 500000
    max_iter_FQISTA = 30000
    max_iter = max(max_iter_ISTA,max_iter_FISTA,max_iter_QISTA,max_iter_FQISTA)
    x_axis_show = max_iter / 20
    show_per = 5000
    epsilon = 1.0 * torch.ones(n,1)
    soft_lambda = 1e-3
    q_lambda = 1e-1
else:
    run_time = 20
    max_iter_ISTA = 10000
    max_iter_FISTA = 50000
    max_iter_QISTA = 10000
    max_iter_FQISTA = 300000
    max_iter = max(max_iter_ISTA,max_iter_FISTA,max_iter_QISTA,max_iter_FQISTA)
    x_axis_show = max_iter / 20
    show_per = 5000

def generate_x(k):
    x_temp = torch.zeros(n,1)
    support = torch.randperm(n)[0:k]
    x_temp[support] = torch.normal(0.0,signal_std*torch.ones(k,1))
    return x_temp

#def generate_x():
#    support = torch.bernoulli(p * torch.ones(n,1))
#    nonzero = torch.normal(0.0, signal_std * torch.ones(n,1))
#    return torch.mul(nonzero, support)

def soft_threshold(x,s):
    temp_p = (x-s).clamp(min=0)
    temp_n = (x+s).clamp(max=0)
    temp = temp_p + temp_n
    return temp


rec_ISTA = torch.zeros(max_iter_ISTA,run_time)
rec_FISTA = torch.zeros(max_iter_FISTA,run_time)
rec_QISTA = torch.zeros(max_iter_QISTA,run_time)
rec_FQISTA = torch.zeros(max_iter_FQISTA,run_time)

rec_ISTA_RE = torch.zeros(run_time)
rec_FISTA_RE = torch.zeros(run_time)
rec_QISTA_RE = torch.zeros(run_time)
rec_FQISTA_RE = torch.zeros(run_time)

total_using_time = 0.0

for run_i in range(run_time):
    time_begin = time.time()
    x0 = generate_x(k)
    A = torch.normal(0.0,math.sqrt(1.0)*torch.ones(m,n))
    y = A.mm(x0)
    At = A.t()
    reference = x0.norm(2).pow(2.0)
    
    w,v = torch.eig(At.mm(A),eigenvectors=False)
    max_evalue = w[:,0].max()
    beta = 1.0/max_evalue
        
    print('\n')
####################################
##########   begin ISTA   ##########
    x_ISTA = torch.zeros(n,1)
    for i in range(max_iter_ISTA):
        if (i+1)%show_per==0:
            print('\rISTA   : round',run_i+1,' i =',i+1,'/',max_iter_ISTA,end='')
        temp_ISTA = y - A.mm(x_ISTA)
        r_ISTA = x_ISTA + beta*At.mm(temp_ISTA)
        x_ISTA = soft_threshold(r_ISTA,soft_lambda)
        residue = (x0 - x_ISTA).norm(2).pow(2.0)
        SNR = 10.0*torch.log(reference/residue)/math.log(10.0)
        rec_ISTA[i,run_i] = SNR.clone()
    rec_ISTA_RE[run_i] = torch.sqrt(residue/reference).clone()
    print('')
##########   end ISTA   ############
####################################
        
    
####################################
##########   begin FISTA   #########
    x_FISTA = torch.zeros(n,1)
    x_FISTA_prev = torch.zeros(n,1)
    for i in range(max_iter_FISTA):
        if (i+1)%show_per==0:
            print('\rFISTA  : round',run_i+1,' i =',i+1,'/',max_iter_FISTA,end='')
        temp_FISTA = y - A.mm(x_FISTA)
        r_FISTA = x_FISTA + beta*At.mm(temp_FISTA)
        r_FISTA += ((i-2)/(i+1))*(x_FISTA-x_FISTA_prev)
        x_FISTA_prev = x_FISTA.clone()
        x_FISTA = soft_threshold(r_FISTA,soft_lambda)
        residue = (x0 - x_FISTA).norm(2).pow(2.0)
        SNR = 10.0*torch.log(reference/residue)/math.log(10.0)
        rec_FISTA[i,run_i] = SNR.clone()
    rec_FISTA_RE[run_i] = torch.sqrt(residue/reference).clone()
    print('')
##########   end FISTA   ###########
####################################
        
    
####################################
##########   begin QISTA   #########
    x_QISTA = torch.zeros(n,1)
    for i in range(max_iter_QISTA):
        if (i+1)%show_per==0:
            print('\rQISTA  : round',run_i+1,' i =',i+1,'/',max_iter_QISTA,end='')
        temp_QISTA = y - A.mm(x_QISTA)
        r_QISTA = x_QISTA + beta*At.mm(temp_QISTA)
        q_truncated = q_lambda*beta/(1.0*((x_QISTA.abs()+epsilon)**(1-q)))
        x_QISTA = soft_threshold(r_QISTA,q_truncated)
        residue = (x0 - x_QISTA).norm(2).pow(2.0)
        SNR = 10.0*torch.log(reference/residue)/math.log(10.0)
        rec_QISTA[i,run_i] = SNR.clone()
    rec_QISTA_RE[run_i] = torch.sqrt(residue/reference).clone()
    print('')
##########   end QISTA   ###########
####################################
        
        
####################################
##########   begin FQISTA   ########
    x_FQISTA = torch.zeros(n,1)
    x_FQISTA_prev = torch.zeros(n,1)
    for i in range(max_iter_FQISTA):
        if (i+1)%show_per==0:
            print('\rFQISTA : round',run_i+1,' i =',i+1,'/',max_iter_FQISTA,end='')
        temp_FQISTA = y - A.mm(x_FQISTA)
        r_FQISTA = x_FQISTA + beta*At.mm(temp_FQISTA)
        r_FQISTA += ((i-2)/(i+1))*(x_FQISTA-x_FQISTA_prev)
        x_FQISTA_prev = x_FQISTA.clone()
        q_truncated = q_lambda*beta/(1.0*((x_FQISTA.abs()+epsilon)**(1-q)))
        x_FQISTA = soft_threshold(r_FQISTA,q_truncated)
        residue = (x0 - x_FQISTA).norm(2).pow(2.0)
        SNR = 10.0*torch.log(reference/residue)/math.log(10.0)
        rec_FQISTA[i,run_i] = SNR.clone()
    rec_FQISTA_RE[run_i] = torch.sqrt(residue/reference).clone()
    print('')
##########   end FQISTA   ##########
####################################
        
        
    results_ISTA = rec_ISTA.cpu().numpy()
    results_FISTA = rec_FISTA.cpu().numpy()
    results_QISTA = rec_QISTA.cpu().numpy()
    results_FQISTA = rec_FQISTA.cpu().numpy()
    
    print('\n  k=',k,' q=',q,'  mean SNR of last 1000 iter.:')
    print('    ISTA   : round',run_i+1,' SNR=[',results_ISTA[-1000:,run_i].mean(),']  ')
    print('    FISTA  : round',run_i+1,' SNR=[',results_FISTA[-1000,run_i].mean(),']  ')
    print('    QISTA  : round',run_i+1,' SNR=[',results_QISTA[-1000,run_i].mean(),']  ')
    print('    FQISTA : round',run_i+1,' SNR=[',results_FQISTA[-1000,run_i].mean(),']  ')
    
    print('\nresults:')
    print('ISTA   : round {0}  SNR=[{1:>.5f}]  RE=[{2:>.5e}]'.format(run_i+1,results_ISTA[max_iter_ISTA-1,run_i],rec_ISTA_RE[run_i]))
    print('FISTA  : round {0}  SNR=[{1:>.5f}]  RE=[{2:>.5e}]'.format(run_i+1,results_FISTA[max_iter_FISTA-1,run_i],rec_FISTA_RE[run_i]))
    print('QISTA  : round {0}  SNR=[{1:>.5f}]  RE=[{2:>.5e}]'.format(run_i+1,results_QISTA[max_iter_QISTA-1,run_i],rec_QISTA_RE[run_i]))
    print('FQISTA : round {0}  SNR=[{1:>.5f}]  RE=[{2:>.5e}]'.format(run_i+1,results_FQISTA[max_iter_FQISTA-1,run_i],rec_FQISTA_RE[run_i]))
    
    
    using_time = time.time() - time_begin
    print('\nusing time of round {0} : {1:>.3f} sec'.format(run_i+1,using_time))
    total_using_time += using_time
    
    x_axis = np.arange(1,max_iter+1,1)
    plt.figure(figsize=(12,6))
    plt.plot(x_axis[0:max_iter_ISTA],results_ISTA[:,run_i],color="b", linewidth=2,linestyle="-",label="ISTA")
    plt.xlabel('iter',fontsize=16,fontweight='bold')
    plt.ylabel('SNR',fontsize=16,fontweight='bold')
    plt.xticks(np.arange(0,max_iter+1,x_axis_show))
    plt.xscale('log')
    y_max_F = np.ceil((results_FISTA.max()/10))*10
    y_max_FQ = np.ceil((results_FQISTA.max()/10))*10
    y_max = max(y_max_F,y_max_FQ)
    plt.ylim((0, y_max))
    if y_max >=40:
        ytick_per = 5
    else:
        ytick_per = 2
    plt.yticks(np.arange(0,y_max+1,ytick_per))
    plt.grid()
    plt.title("ISTA vs. FISTA vs. QISTA vs. FQISTA",fontsize=16,fontweight='bold')
    
    plt.plot(x_axis[0:max_iter_FISTA], results_FISTA[:,run_i],color="r", linewidth=2,linestyle="-",label="FISTA")
    plt.plot(x_axis[0:max_iter_QISTA], results_QISTA[:,run_i],color="g", linewidth=2,linestyle="-",label="QISTA")
    plt.plot(x_axis[0:max_iter_FQISTA],results_FQISTA[:,run_i],color="c", linewidth=2,linestyle="-",label="FQISTA")
    plt.legend(loc='upper left', prop={'size': 20})
    plt.savefig('n-m-k='+str(n)+'-'+str(m)+'-'+str(k)+'-round_'+str(run_i+1)+'.png')
    plt.show()
    
    
    

print('\n\n\nfinish...')
print('  k = {0}, q = {1}, total using time: {2:>.3f}'.format(k,q,total_using_time))
results_ISTA_mean   = results_ISTA.mean(1)
results_FISTA_mean  = results_FISTA.mean(1)
results_QISTA_mean  = results_QISTA.mean(1)
results_FQISTA_mean = results_FQISTA.mean(1)

print('mean SNR of last 1000 iter.:')
print('  mean SNR of ISTA     :[',results_ISTA_mean[-1000].mean(),']  ',end='\n')
print('  mean SNR of FISTA    :[',results_FISTA_mean[-1000].mean(),']  ',end='\n')
print('  mean SNR of QISTA-I  :[',results_QISTA_mean[-1000].mean(),']  ',end='\n')
print('  mean SNR of FQISTA-I :[',results_FQISTA_mean[-1000].mean(),']  ',end='\n')



x_axis = np.arange(1,max_iter+1,1)
plt.figure(figsize=(12,6))
plt.plot(x_axis[0:max_iter_ISTA],results_ISTA_mean,color="b", linewidth=2,linestyle="-",label="ISTA")
plt.xlabel('iter',fontsize=16,fontweight='bold')
plt.ylabel('SNR',fontsize=16,fontweight='bold')
plt.xticks(np.arange(0,max_iter+1,x_axis_show))
plt.xscale('log')
y_max_F = np.ceil((results_FISTA_mean.max()/10))*10
y_max_FQ = np.ceil((results_FQISTA_mean.max()/10))*10
y_max = max(y_max_F,y_max_FQ)
plt.ylim((0, y_max))
if y_max >=40:
    ytick_per = 5
else:
    ytick_per = 2
plt.yticks(np.arange(0,y_max+1,ytick_per))
plt.grid()
plt.title("ISTA vs. FISTA vs. QISTA vs. FQISTA",fontsize=16,fontweight='bold')

plt.plot(x_axis[0:max_iter_FISTA], results_FISTA_mean,color="r", linewidth=2,linestyle="-",label="FISTA")
plt.plot(x_axis[0:max_iter_QISTA], results_QISTA_mean,color="g", linewidth=2,linestyle="-",label="QISTA")
plt.plot(x_axis[0:max_iter_FQISTA],results_FQISTA_mean,color="c", linewidth=2,linestyle="-",label="FQISTA")
plt.legend(loc='upper left', prop={'size': 20})
plt.savefig('n-m-k='+str(n)+'-'+str(m)+'-'+str(k)+'-mean.png')
plt.show()

count_ISTA = 0
count_FISTA = 0
count_QISTA = 0
count_FQISTA = 0
print('')
for ii in range(run_time):
    print('ISTA   : round {0} RE = [{1:>.5e}]'.format(ii+1,rec_ISTA_RE[ii].item()))
    if rec_ISTA_RE[ii] < 1e-4:
        count_ISTA += 1
for ii in range(run_time):
    print('FISTA  : round {0} RE = [{1:>.5e}]'.format(ii+1,rec_FISTA_RE[ii].item()))
    if rec_FISTA_RE[ii] < 1e-4:
        count_FISTA += 1
for ii in range(run_time):
    print('QISTA  : round {0} RE = [{1:>.5e}]'.format(ii+1,rec_QISTA_RE[ii].item()))
    if rec_QISTA_RE[ii] < 1e-4:
        count_QISTA += 1
for ii in range(run_time):
    print('FQISTA : round {0} RE = [{1:>.5e}]'.format(ii+1,rec_FQISTA_RE[ii].item()))
    if rec_FQISTA_RE[ii] < 1e-4:
        count_FQISTA += 1

print('success rate of   ISTA = ',count_ISTA,'/',run_time)
print('success rate of  FISTA = ',count_FISTA,'/',run_time)
print('success rate of  QISTA = ',count_QISTA,'/',run_time)
print('success rate of FQISTA = ',count_FQISTA,'/',run_time)




