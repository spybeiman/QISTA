print(chr(27) + "[2J")
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0') # choose 'cpu' or 'cuda'\

import logging
logging.basicConfig(filename='./QISTA-Net-results.log',level=logging.INFO,format='%(message)s',filemode='a')


# =========================================
# ==========  parameter setting  ==========
# =========================================
n,m = 500,150
p_nonzero = 1/10 # percentage of nonzero k=n*p
q = 0.05
Qista_Net_Plus = False # true: QISTA-Net$^+$, false: QISTA-Net
max_layers = 16

num_in_testing = 100 # for testing in 100 tests
testing_batch_size = 1 # in each test, run only single vector
# =========================================
# =========================================


k = int(n*p_nonzero)
lambda_init = 1e-4
eps_init = 1e-1
eps_min = 1e-1
batch_size = 1000
num_train_for_each_lr = 1000
training_show_in_every = 50
signal_var = 1.0
signal_std = math.sqrt(signal_var)

final_plot = True

def generate_x(batch_size):
    support = torch.bernoulli(p_nonzero * torch.ones(n,batch_size))
    nonzero = torch.normal(0.0, signal_std * torch.ones(n,batch_size))
    return torch.mul(nonzero, support)

def soft_threshold(x,s):
    temp_p = (x-s).clamp(min=0)
    temp_n = (x+s).clamp(max=0)
    temp = temp_p + temp_n
    return temp
  
def error_plot(results,label):
    x_axis = np.arange(1,max_layers+1,1)
    plt.figure(figsize=(12,8))
    print('\nresults (',label,'):\n',results.cpu().data.numpy())
    results_ = results.cpu().data.numpy()
    if label == 'SNR':
        plt.plot(x_axis,results_)
        max_print = (int(results_.max()/10)+1)*10+1
        if max_print > 40:
            print_per = 5
        else:
            print_per = 2
        plt.yticks(np.arange(0,max_print,print_per))
    else:
        plt.semilogy(x_axis,results.cpu().data.numpy())
    plt.xlabel('iter',fontsize=20,fontweight='bold')
    plt.ylabel(label,fontsize=20,fontweight='bold')
    if Qista_Net_Plus == True:
        plt.title('QISTA-Net$^+$ ('+label+')-n-m-k-q='+str(n)+'-'+str(m)+'-'+str(k)+'-'+str(q),fontsize=20,fontweight='bold')
    else:
        plt.title('QISTA-Net ('+label+')-n-m-k-q='+str(n)+'-'+str(m)+'-'+str(k)+'-'+str(q),fontsize=20,fontweight='bold')
    plt.xticks(np.arange(1,max_layers+1,1))
    plt.grid(True)
    if Qista_Net_Plus == True:
        plt.savefig('QISTA-Net-plus ('+label+')-n-m-k-q='+str(n)+'-'+str(m)+'-'+str(k)+'-'+str(q)+'.png')
    else:
        plt.savefig('QISTA-Net ('+label+')-n-m-k-q='+str(n)+'-'+str(m)+'-'+str(k)+'-'+str(q)+'.png')
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    
def isnan(x):
    return x != x

temp = generate_x(batch_size)
break_crit = (temp.to(device).norm(2)**2)*10  # if loss>crit then stop
del temp

A = torch.normal(0.0,math.sqrt(1.0/m)*torch.ones(m,n)).to(device)

import os
if os.path.isfile('./A_data.h5') == True:
    os.remove('./A_data.h5')

import h5py
f = h5py.File('./A_data.h5', 'w')
f['A_data'] = np.zeros(shape=(m, n))
f['A_data'][...] = A.data.cpu().numpy()
f.close()

At = A.t().to('cpu')
w,v = torch.eig(At.mm(A.to('cpu')),eigenvectors=False)
max_evalue = w[:,0].max()
beta = 1.0/max_evalue
MSE_results = torch.zeros(max_layers).to(device)
RE_results  = torch.zeros(max_layers).to(device)
SNR_results = torch.zeros(max_layers).to(device)

Ati = torch.Tensor(n,m,max_layers)
for i in range(max_layers):
    Ati[...,i] = At.clone()

class QISTA_Net(nn.Module):
    def __init__(self):
        super(QISTA_Net,self).__init__()
        self.At_train = nn.Parameter(beta*Ati.clone())
        self.E = nn.Parameter(eps_init*torch.ones(n,batch_size,max_layers))
        self.lambda_t = nn.Parameter(lambda_init*torch.ones(max_layers,1))

    if Qista_Net_Plus == True:
        def forward(self,x,y,num_layers):
            if x.size()[1] == batch_size:
                DD_train = torch.zeros(n,batch_size,max_layers).to(device)
                for i in range(num_layers):
                    DD_train[...,i] = self.At_train[...,i].mm(y - A.mm(x))
                    temp = x + DD_train.sum(2)
                    DD_train = DD_train * (k/m)
                    trun_param = self.lambda_t[i] * beta.to(device) / ((self.E[...,i] + x.abs())**(1-q))
                    x = soft_threshold(temp,trun_param)
            elif x.size()[1] == testing_batch_size:
                DD_test = torch.zeros(n,testing_batch_size,max_layers).to(device)
                for i in range(num_layers):
                    DD_test[...,i] = self.At_train[...,i].mm(y - A.mm(x))
                    temp = x + DD_test.sum(2)
                    DD_test = DD_test * (k/m)
                    trun_param = self.lambda_t[i] * beta.to(device) / ((self.E.mean(1)[:,i].reshape(n,1) + x.abs())**(1-q))
                    x = soft_threshold(temp,trun_param)
            return x
    else:
        def forward(self,x,y,num_layers):
            if x.size()[1] == batch_size:
                for i in range(num_layers):
                    temp = x + self.At_train[...,i].mm(y - A.mm(x))
                    trun_param = self.lambda_t[i] * beta.to(device) / ((self.E[...,i] + x.abs())**(1-q))
                    x = soft_threshold(temp,trun_param)
            elif x.size()[1] == testing_batch_size:
                for i in range(num_layers):
                    temp = x + self.At_train[...,i].mm(y - A.mm(x))
                    trun_param = self.lambda_t[i] * beta.to(device) / ((self.E.mean(1)[:,i].reshape(n,1) + x.abs())**(1-q))
                    x = soft_threshold(temp,trun_param)
            return x

network = QISTA_Net().to(device)
total_time_start = time.time()

# begin training
for num_layers in range(1,max_layers+1):
    training_start = time.time()
    for num_lr in range(1):
        if num_lr == 0:
            learning_rate = 0.001
        elif num_lr == 1:
            learning_rate = 0.001*0.5
        elif num_lr == 2:
            learning_rate = 0.001*0.1
        elif num_lr == 3:
            learning_rate = 0.001*0.01
        if num_layers > 10:
            learning_rate = learning_rate / 4
        print('\n\n layer:{layer}'.format(layer=num_layers),end='')
        print(' lr: {num_lr}/4'.format(num_lr=num_lr+1))
        
        train_no = 0
        while train_no < num_train_for_each_lr:
            x0 = generate_x(batch_size).to(device)
            y = A.mm(x0)
            x = torch.zeros(n,batch_size).to(device)
            opt = torch.optim.Adam(network.parameters(),lr=learning_rate)
            x_output = network(x,y,num_layers)
            loss = F.mse_loss(x_output,x0,reduction='sum')
            if train_no%training_show_in_every == training_show_in_every-1:
                print('  L={layer},lr:{lr}, no.{no:>3d}, loss: {loss:<7.5f}'.format(lr=num_lr+1,no=train_no+1,loss=loss,layer=num_layers),end='', flush=True)
            if loss < break_crit:
                NaN_exists = False
                opt.zero_grad()
                loss.backward(retain_graph=True)
                if isnan(network.At_train).any():
                    NaN_exists = True
                if isnan(network.At_train.grad).any():
                    NaN_exists = True
                if isnan(network.E).any():
                    NaN_exists = True
                if isnan(network.E.grad).any():
                    NaN_exists = True
                if isnan(network.lambda_t).any():
                    NaN_exists = True
                if isnan(network.lambda_t.grad).any():
                    NaN_exists = True
                if NaN_exists == False:
                    opt.step()
                    E_net_temp = network.state_dict()
                    E_net_temp['E'] = E_net_temp['E'].clamp(min=eps_min)
                    network.load_state_dict(E_net_temp)
                    train_no += 1
            del x0, y, x, x_output, loss, opt
            
            if isnan(network.At_train.grad).any():
                break
            if isnan(network.E.grad).any():
                break
            if isnan(network.lambda_t.grad).any():
                break
        
# for my check
            if train_no%training_show_in_every == 0:
                MSE = torch.Tensor([0]).to(device)
                RE  = torch.Tensor([0]).to(device)
                SNR = torch.Tensor([0]).to(device)
                for i in range(num_in_testing):
                    x0 = generate_x(testing_batch_size).to(device)
                    y = A.mm(x0)
                    x = torch.zeros(n,testing_batch_size).to(device)
                    x_output = network(x,y,num_layers)
                    reference = x0.norm(2,0).norm(2).pow(2.0)
                    residue = (x0 - x_output).norm(2,0).norm(2).pow(2.0)
                    MSE += (residue/n).clone()
                    RE += torch.sqrt(residue/reference).clone()
                    SNR += 10.0*torch.log(reference/residue)/math.log(10.0)
                    del x0, y, x, x_output
                MSE = MSE/num_in_testing
                RE  = RE /num_in_testing
                SNR = SNR/num_in_testing
                print(' MSE={0:>8.2e}'.format(MSE.item()),end='')
                print(' RE={0:>8.2e}'.format(RE.item()),end='')
                print(' SNR={0:>4.2f}'.format(SNR.item()))
# end my check
    
    training_time = time.time() - training_start
    torch.save(network.state_dict(), './model-' + str(num_layers) + '.pkl')
# end training


# begin testing
    testing_start = time.time()
    MSE = torch.Tensor([0]).to(device)
    RE  = torch.Tensor([0]).to(device)
    SNR = torch.Tensor([0]).to(device)
    for i in range(num_in_testing):
        x0 = generate_x(testing_batch_size).to(device)
        y = A.mm(x0)
        x = torch.zeros(n,testing_batch_size).to(device)
        x_output = network(x,y,num_layers)
        reference = x0.norm(2,0).norm(2).pow(2.0)
        residue = (x0 - x_output).norm(2,0).norm(2).pow(2.0)
        MSE += (residue/n).clone()
        RE += torch.sqrt(residue/reference).clone()
        SNR += 10.0*torch.log(reference/residue)/math.log(10.0)
        del x0, y, x, x_output
    MSE = MSE/num_in_testing
    RE  = RE /num_in_testing
    SNR = SNR/num_in_testing
    print('\nresults of layer:',num_layers)
    print('        MSE = {0:>10.4e}'.format(MSE.item()))
    print('        RE  = {0:>10.4e}'.format(RE.item()))
    print('        SNR = {0:>10.4f}'.format(SNR.item()))
    logging.info('\nresults of layer:{0}'.format(num_layers))
    logging.info('        MSE = {0:>10.4e}'.format(MSE.item()))
    logging.info('        RE  = {0:>10.4e}'.format(RE.item()))
    logging.info('        SNR = {0:>10.4e}'.format(SNR.item()))
    MSE_results[num_layers-1] = MSE.clone()
    RE_results[num_layers-1] = RE.clone()
    SNR_results[num_layers-1] = SNR.clone()
    testing_time = time.time() - testing_start
    print("training time:{0:>6.2f}".format(training_time) + " (sec)")
    print("testing time :{0:>6.2f}".format(testing_time) + " (sec)")
    logging.info('        training time:{0:>6.2f} (sec)'.format(training_time))
    logging.info('        testing time :{0:>6.2f} (sec)'.format(testing_time))
    print("\n")
    if num_layers == max_layers and final_plot == True:
        error_plot(MSE_results,'MSE')
        error_plot(RE_results,'RE')
        error_plot(SNR_results,'SNR')
# end testing
        
total_using_time = time.time() - total_time_start
print('\ntotal using time: {0:.2f} sec'.format(total_using_time))
print('with setting:')
print('  n =',n,'m =',m,'k ~',k,'q =',q)
print('  max layer =',max_layers)
print('final results:')
print('  MSE =',MSE_results.cpu().data.numpy()[-1])
print('  RE  =',RE_results.cpu().data.numpy()[-1])
print('  SNR =',SNR_results.cpu().data.numpy()[-1])
logging.info('\ntotal using time: {0:.2f} sec'.format(total_using_time))
logging.info('with setting:')
logging.info('  n = {0}, m = {1}, k ~ {2}, q = {3}'.format(n,m,k,q))
logging.info('  max layer = {0}'.format(max_layers))
logging.info('final results:')
logging.info('  MSE = {0:.6f}'.format(MSE_results.cpu().data.numpy()[-1]))
logging.info('  RE  = {0:.6f}'.format(RE_results.cpu().data.numpy()[-1]))
logging.info('  SNR = {0:.6f}\n\n\n\n'.format(SNR_results.cpu().data.numpy()[-1]))
