#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:44:53 2023

@author: bryanhaddock
"""
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import nibabel as nib
from scipy.optimize import curve_fit
from tqdm import tqdm

def T2star_function(X, a1, a2,a3):
    return a1*math.e**(-X/a2)+ a3


def T2star_map(T2data4D,mask_data,data_img,TE_time,dataname):
    T2map=np.zeros(mask_data.shape)
    S0map=np.zeros(mask_data.shape)
    Cmap=np.zeros(mask_data.shape)
    Errmap=np.zeros(mask_data.shape)
    coordinates = np.argwhere(mask_data)
    
    totalcounter=0
    fitcounter=0
    
    for x in tqdm(range(0,len(coordinates))):
        totalcounter=totalcounter+1
        coord=coordinates[x]
        T2array=T2data4D[coord[0],coord[1],coord[2],:]
        Sigmax=np.max(T2array)
        initial_guess = [Sigmax, 30., 1.0]  # Initial guess for parameters
        try:
            Params, _ = curve_fit(T2star_function, TE_time,T2array, p0=initial_guess,  bounds=[(20,1,-100),(1.3*Sigmax,1000,0.4*Sigmax)])
            fitdat=T2star_function(TE_time,Params[0],Params[1],Params[2])
            
            T2map[coord[0],coord[1],coord[2]]=Params[1]  
            S0map[coord[0],coord[1],coord[2]]=Params[0]
            Cmap[coord[0],coord[1],coord[2]]=Params[2]
            Errmap[coord[0],coord[1],coord[2]]=100*np.mean(np.abs((T2array-fitdat)/np.mean(T2array)))
            
            fitcounter=fitcounter+1
            if np.random.rand() < 0.001:
                plt.show()
                plt.plot(TE_time,T2array, 'o', label='data')
                plt.plot(TE_time,T2star_function(TE_time,Params[0],Params[1],Params[2]),  '-', label='T2='+str(int(Params[1])))
                plt.legend()
            
        except:
            print('failed '+ str(T2array))
            time.sleep(2)
            
    print( 'totalcounter=' +str(totalcounter)+ '   fitcounter=' +str(fitcounter))
    modified_img = nib.Nifti1Image(T2map, data_img.affine, data_img.header)
    nib.save(modified_img, dataname+'_T2map.nii.gz')
 
    #modified_img = nib.Nifti1Image(S0map, data_img.affine, data_img.header)
    #nib.save(modified_img, dataname+'_S0map.nii.gz')
 
    #modified_img = nib.Nifti1Image(Cmap, data_img.affine, data_img.header)
    #nib.save(modified_img, dataname+'_Cmap.nii.gz')
    
    #modified_img = nib.Nifti1Image(Errmap, data_img.affine, data_img.header)
    #nib.save(modified_img, dataname+'_Errmap.nii.gz')
 
    print('saving maps to '+dataname)
    return 'heello'


# get files, send data to fitting and save maps
source_base = '/run/user/1000/gvfs/smb-share:server=10.141.40.29,share=loggededata/GA17/GLP2_Ant/01_Forsøgs_data/FP02/DagA/Nifti'
savedir='/home/klinfys/Desktop/T2_star/Outputs_GLP2_Ant/02A'

#Echo times defined
TE_time=np.array([2.04, 4.95, 7.85, 10.76, 13.66, 16.57, 19.47, 22.38, 25.28, 28.18])
print("starting")
for i in range(1,9):
    print("\t",i)
    dataname = f'MR{i}_stacked.nii'
    file_path = f"{source_base}/{dataname}"
    
    # Load the NIfTI file
    data_img = nib.load(file_path)
    T2data4D = data_img.get_fdata()
    T2data4D = T2data4D[:, :, :, 0:10]
    dim = T2data4D.shape
    
    # Create a mask based on the first echo
    First_echo = T2data4D[:, :, :, 0]
    mask_data = np.zeros(dim[0:3])
    mask_data[First_echo > 200] = 1

    # Call the processing function, save results
    output_filepath = f"{savedir}/MR{i}_T2star_output"  # Define where to save the output for each file
    T2star_map(T2data4D, mask_data, data_img, TE_time, output_filepath)  # Assuming T2star_map handles saving