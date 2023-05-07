import numpy as np
import time
import datetime
import os

from math import *
import scipy
from scipy.linalg import expm, logm

## General

class IrrelevantForDim(Exception):
    """Raised when the asked task is irrelevant for the dimension at play"""
    pass

def val_to_ind(val, vect):
    if val <= vect[0]:
        return 0
    for k in range(len(vect)-1):
        if val <= vect[k+1]:
            if np.abs(vect[k]-val)< np.abs(vect[k+1]-val):
                return k
            else:
                return k+1
    return k+1

def _next(tpl, shape):
    if not np.sum(np.array(tpl)+1 != np.array(shape)):
        return None
    if tpl[-1] == shape[-1]-1 :
        return _next(tpl[:-1], shape[:-1] )+(0,)
    else:
        return tpl[:-1]+(tpl[-1]+1 ,)

def _analyze_axis(car):
    if car == 'X' or car == 'x' :
        return 0
    if car == 'Y' or car == 'y' :
        return 1
    if car == 'Z' or car == 'z' :
        return 2
    if car in [str(i) for i in range(20)] :
        return int(car)
    else:
        print("axis code not recognized, supported values are 'x','y','z' ; 'X','Y','Z' ; and ints below 20")
        raise ValueError("axis code")

def resize_nD(arr, shape, tol = 1e-12):#for odd N and spins or int
    if arr.shape == shape :
        return arr
    red = False
    test_bad_size = [arr.shape[k] > shape[k] for k in range(len(shape))]
    if np.sum(test_bad_size) >tol :
        print("Resized to smaller size...")
        print("Desired size : ", shape, " ; Actual size : ", arr.shape)
#         raise ValueError
        red = True

    if arr[(0,)*len(shape)].size == 1:
        res = np.zeros(shape)
    elif arr[(0,)*len(shape)].size > 1:
        res = np.zeros(shape + (arr.shape[len(shape):]) )*1.j



    if red :
        tpl1 =  [int((arr.shape[k]-1)/2) - int((shape[k]-1)/2)  for k in range(len(shape))]
        tpl2 =  [int((arr.shape[k]-1)/2) + int((shape[k]+1)/2)  for k in range(len(shape))]
        res = arr[tuple([slice(tpl1[i], tpl2[i]) for i in range(len(tpl2))])]

    else:
        tpl1 =  [int((shape[k]-1)/2) - int((arr.shape[k]-1)/2)  for k in range(len(shape))]
        tpl2 =  [int((shape[k]-1)/2) + int((arr.shape[k]+1)/2)  for k in range(len(shape))]
        res[tuple([slice(tpl1[i], tpl2[i]) for i in range(len(tpl2))])] = arr

    return res

def mk_stamp(dim=None):
    if dim is None :
        dim = 'N'
    else:
        dim = str(dim)
    now = datetime.datetime.now()
    res = dim+"D_viaND_"+str(now.year)+'.'+str(now.month)+"."+str(now.day)+'_'+str(now.hour).rjust(2,'0')+"h"+str(now.minute).rjust(2,'0')
    nb=1
    if len([filename for filename in os.listdir('.') if filename.startswith(res)]) == 0 :
        return res
    while len([filename for filename in os.listdir('.') if filename.startswith(res+"_"+str(nb))]) != 0 :
        nb+=1
    return res+"_"+str(nb)

def timer(it, max_it, freq_mess, deb, extra_mess = ''):
    if it == 0:
        print("\n\nStarting at ", datetime.datetime.now())
    if it/max_it >0 and it/max_it // freq_mess != (it-1)/max_it // freq_mess  :
        avan = it/max_it
        print("\nProgress :", np.round(avan*100, max(0, int(-np.log10(freq_mess)-2))), "%")
        if extra_mess != '' :
            print(extra_mess)
        print("At :", datetime.datetime.now())
        print("Refined ending :", datetime.datetime.now()+datetime.timedelta(seconds=(time.time() - deb)*(1-avan)/avan))


def _mk_param_edit_fct(keys, update = lambda x:x):
    if type(keys)==str:
        def fct(param, val):
            param[keys] = val
            param = update(param)
            return param
        return fct
    else:
        def fct(param, vals):
            for k in range(len(keys)):
                param[keys[k]] = vals[k]
                param = update(param)
            return param
        return fct