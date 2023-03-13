import os
from tqdm import tqdm
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

split_list = [2, 3, 4]
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4]
id_list_4 = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
distance_list = [1, 2, 3, 4, 5, 6]
Exp_list = ['fig2', 'fig3']
Para_list = ['MSE', 'SSIM', 'PSNR']

dir_seed = '../MSE/data_seed.json'
dir_CPA = '../CPA/data_CPA.json'
dir_transformer = '../CVAE_Transformer_Batch/data_transformer.json'

with open(dir_seed, 'r') as fp:
    data_seed = json.load(fp)
with open(dir_CPA, 'r') as fp:
    data_CPA = json.load(fp)
with open(dir_transformer, 'r') as fp:
    data_transformer = json.load(fp)

# fig2, fig3数据可视化
for exp in Exp_list:
    # fig = plt.figure()
    fig = plt.figure(figsize=(80, 30))
    for p in range(len(Para_list)):
        para = Para_list[p]
        for i in range(7):
            ax = fig.add_subplot(3, 7, i+1+(p*7))
            name = name_list[i]
            ax.set_title(name,fontsize=30)
            ax.set_xlabel('distance',fontsize=30)
            ax.set_ylabel(para,fontsize=30)
            plt.xticks(size = 25)
            plt.yticks(size = 25)
            ax.plot(np.array(distance_list), np.array(data_seed[exp][para][name][0:6]), 'bo-', label='seed', linewidth=2)
            ax.plot(np.array(distance_list), np.array(data_CPA[exp][para][name][0:6]), 'y^-', label='CPA', linewidth=2)
            if(exp == 'fig3' and name == 'E15.5'):
                pass
            else:
                ax.plot(np.array(distance_list), np.array(data_transformer[exp][para][name][0:6]), 'r*-', label='Transformer', linewidth=2)
            ax.legend(fontsize=25)
    fig.tight_layout()
    plt.savefig(exp+'.png', dpi=100)

# fig4数据可视化
for pro in range(len(proportion_list)):
# for pro in range(1):
    proportion = proportion_list[pro]
    # fig = plt.figure()
    fig = plt.figure(figsize=(80, 30))
    for p in range(len(Para_list)):
        para = Para_list[p]
        for i in range(7):
            ax = fig.add_subplot(3, 7, i+1+(p*7))
            name = name_list[i]
            ax.set_title(name,fontsize=30)
            ax.set_xlabel('distance',fontsize=30)
            ax.set_ylabel(para,fontsize=30)
            plt.xticks(size = 25)
            plt.yticks(size = 25)
            ax.plot(np.array(distance_list), np.array(data_seed['fig4'][para][name][str(proportion)][0:6]), 'bo-', label='seed', linewidth=2)
            ax.plot(np.array(distance_list), np.array(data_CPA['fig4'][para][name][str(proportion)][0:6]), 'y^-', label='CPA', linewidth=2)
            ax.plot(np.array(distance_list), np.array(data_transformer['fig4'][para][name][str(proportion)][0:6]), 'r*-', label='Transformer', linewidth=2)
            ax.legend(loc=2, fontsize=25)
    fig.tight_layout()
    plt.savefig('fig4_'+str(proportion*200)+'%'+'.png', dpi=100)
        




