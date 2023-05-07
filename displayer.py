import cv2
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import cmath

import datetime
from math import *
from scipy.linalg import expm, logm
import scipy.linalg
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (10, 7)

from quantum_and_walks import *

## spectrum
def _plot_spec_1D(eig, frame, z_min, z_max):
    plt.figure()
    plt.plot(frame[0], eig)
    plt.ylim([z_min, z_max])
    plt.xlabel(r'$k_x$')
    plt.ylabel('eig_val')
    plt.title("Eigenvalues of the hamiltonian as function of the momentum")

def _plot_spec_2D(eig, frame, z_min, z_max):
    vp0 = eig[:,:,0]
    vp1 = eig[:,:,1]
    X,Y = frame

    fig, axs = plt.subplots(2, 1 , figsize = (10, 10))

    axs[0].imshow(vp0, vmin = z_min, vmax = z_max, origin = 'lower', extent = (X[0], X[-1], Y[0], Y[-1]))
    axs[0].set_xlabel(r'$k_x$')
    axs[0].set_ylabel(r'$k_y$')
    axs[0].set_title("Lower eigenvalue")

    pcm = axs[1].imshow(vp1, vmin = z_min, vmax = z_max, origin = 'lower', extent = (X[0], X[-1], Y[0], Y[-1]))
    axs[1].set_xlabel(r'$k_x$')
    axs[1].set_ylabel(r'$k_y$')
    axs[1].set_title("Upper eigenvalue")

    fig.colorbar(pcm, ax = axs[:])

def _plot_spec(eig, frame, z_min = None, z_max = None, typ = 'real', talk = True):
    if typ == 'real':
        if talk :
            print("\nBeware, we only take the real part of the eigenvalues")
            print("Max of imag is ", np.max(np.abs(eig.imag)))
        eig = eig.real
    elif typ == 'imag':
        if talk :
            print("\nBeware, we only take the imaginary part of the eigenvalues")
            print("Max of real is ", np.max(np.abs(eig.real)))
        eig = eig.imag
    elif typ == 'abs':
        if talk :
            print("\nBeware, we only take the modulus of the eigenvalues")
            print("Max of imag is ", np.max(np.abs(eig.imag)))
            print("Max of real is ", np.max(np.abs(eig.real)))
        eig = np.abs(eig)

    if z_min is None :
        z_min = 1.1*np.min(eig)
    if z_max is None:
        z_max = 1.1*np.max(eig)

    if len(frame) == 1:
        return _plot_spec_1D(eig, frame, z_min, z_max)
    elif len(frame) == 2:
        return _plot_spec_2D(eig, frame, z_min, z_max)
    else:
        print('The representation of the spectrum only makes sense for dim 1 and 2, here dim is '+str(len(frame)))
        raise IrrelevantForDim("_plot_spec")


## states
def _plot_state_2D(state, dx, dy, X=None, Y=None, z_max = None, suff_name = ''):
    frame, mat = display(state, [dx,dy], [X,Y])
    if z_max is None :
        z_max = 1.1 *np.max(pb)

    plt.figure()
    plt.imshow(mat, origin = 'lower', vmax = z_max, extent = (frame[0][0], frame[0][-1], frame[1][0], frame[1][-1]))
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Probability of presence of the state" + suff_name)

def _plot_state_1D(state, dx, X=None, z_max = None, suff_name = ''):
    frame, pb = display(state, [dx], [X])
    if z_max is None :
        z_max = 1.1*np.max(pb)

    plt.figure()
    plt.plot(frame[0], pb)
    plt.ylim([0, z_max])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Probability of presence of the state" + suff_name)

def plot_state(state, dxs, frame = None, z_max = None, suff_name = ''):
    dim = len(dxs)
    if frame is None :
        frame = [None]*dim

    if dim==1:
        _plot_state_1D(state, dxs[0], frame[0], z_max, suff_name)
    elif dim == 2:
        _plot_state_2D(state, dxs[0], dxs[1], frame[0], frame[1], z_max, suff_name)
    else:
        print('The state representation only makes sense for dim 1 and 2, here dim is '+str(dim))
        raise IrrelevantForDim("plot_state")


## Spreadmap
def _plot_spread_map(mat, X, Y, z_max = None):
    if z_max is None :
        z_max = np.max(mat)

    plt.figure()
    plt.imshow(mat, aspect = X[-1]/Y[-1], vmax = z_max, origin = 'lower', extent = (X[0], X[-1], Y[0], Y[-1]), interpolation = None)
    plt.xlabel("X")
    plt.ylabel("T")
    plt.title("Variation of the density probability in time and space")
    plt.colorbar()




## Mean

def _plot_mean_from_mean(absc, ordo, z_min=None, z_max = None, scatter = False, marker = None, s=None, c = None):
    if scatter and s is None:
        s = 4
    if scatter and marker is None:
        marker = 'o'

    plt.figure()
    for k in range(len(ordo[0])):
        if scatter :
            plt.scatter(absc, ordo[:,k], label='$\mu_{'+str(k+1)+'}$', s=s,marker=marker, c=c)
        else:
            plt.plot(absc, ordo[:,k], label='$\mu_{'+str(k+1)+'}$')

    plt.xlabel("t")
    plt.ylabel(r'$\mu$')
    plt.title("Mean")
    if not z_max is None and not z_min is None:
        plt.ylim([z_min- np.abs(z_min-z_max)/10, z_max+ np.abs(z_min-z_max)/10])
    plt.legend()


## std
def _plot_std_from_std_1D(absc, ordo, z_max = None, scatter = False, marker = None, s=None, c = None):
    plt.figure()

    if scatter :
        plt.scatter(absc, np.sqrt(ordo[:,0,0]), label='$\sigma_{X-X}$', s=s,marker=marker, c=c)
    else:
        plt.plot(absc, np.sqrt(ordo[:,0,0]), label='$\sigma_{X-X}$')

    plt.xlabel("t")
    plt.ylabel(r'$\sigma$')
    plt.title("Standard deviation")
    if not z_max is None:
        plt.ylim([0, z_max *1.1])
    plt.legend()


def _plot_std_from_std_nD(absc, ordo, z_max = None, cov_min = None, cov_max = None):
    fig, axs = plt.subplots(2, 1 , figsize = (10, 10))
    for k in range(dim):
        for kk in range(k, dim):
            if k == kk :
                sup = np.max(np.sqrt(ordo[:,k,kk]))
                if z_max < sup :
                    z_max = sup
                if scatter :
                    axs[0].scatter(absc, np.sqrt(ordo[:,k,kk]), label='$\sigma_{'+str(k+1)+'-'+str(kk+1)+'}$', s=s,marker=marker, c=c)
                else:
                    axs[0].plot(absc, np.sqrt(ordo[:,k,kk]), label='$\sigma_{'+str(k+1)+'-'+str(kk+1)+'}$')
            elif k != kk :
                inf = np.min(ordo[:,k,kk])
                sup = np.max(ordo[:,k,kk])
                if cov_min > inf :
                    cov_min = inf
                if cov_max < sup :
                    cov_max = sup
                if scatter :
                    axs[1].scatter(absc, ordo[:,k,kk], label=r'$Cov_{'+str(k+1)+'-'+str(kk+1)+'}$', s=s,marker=marker, c=c)
                else:
                    axs[1].plot(absc, ordo[:,k,kk], label=r'$Cov_{'+str(k+1)+'-'+str(kk+1)+'}$')

    axs[0].set_xlabel("t")
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].set_title("Diagonal elements of the covariance matrix (square root)")
    if not z_max is None:
        axs[0].set_ylim([0, z_max*1.1])
    axs[0].legend()

    axs[1].set_xlabel("t")
    axs[1].set_ylabel(r'$\sigma$')
    axs[1].set_title("Off-diagonal elements of the covariance matrix")
    if not cov_min is None and not cov_max is None:
        axs[1].set_ylim([cov_min- np.abs(cov_min-cov_max)/10, cov_max + np.abs(cov_min-cov_max)/10])
    axs[1].legend()

def _plot_std_from_std(absc, ordo, z_max = None, cov_min = None, cov_max = None, scatter = False, marker = None, s=None, c = None):
    dim = len(ordo[0])
    if dim == 1:
        _plot_std_from_std_1D(absc, ordo, z_max, scatter, marker, s, c)
    else :
        _plot_std_from_std_nD(absc, ordo, z_max, cov_min, cov_max, scatter, marker, s, c)



## entrop
def _plot_entrop_from_entrop(absc, ordo, z_min = None, z_max = None, bool_figure=True, lab = ''):

    if bool_figure:
        plt.figure()

    plt.plot(absc,ordo, label = lab)

    plt.xlabel("t")
    plt.ylabel(r'$\mathcal{S}(t)$')
    plt.title("Entropy of Entanglement")
    if not z_max is None and not z_min is None:
        plt.ylim([z_min , z_max])
    if not lab == '':
        plt.legend()





































##
def mk_gif_gen(plot_fct, param, change_var, rg_var, key_var, name, fps = 25, stamp = None, bool_save = False, talk = True, freq_mess = .2, **kw_plot_param):
    img_array = []

    plt.close('all')
    plt.ioff()

    if stamp is None :
        if 'stamp' in param.keys():
            stamp = param['stamp']
        else :
            stamp = mk_stamp(param['dim'])

    deb = time.time()
    for k in range(len(rg_var)):
        if talk :
            timer(k, len(rg_var), freq_mess, deb, extra_mess = '')

        param = change_var(param, rg_var[k])

        plot_fct(param, bool_show = False, **kw_plot_param)
        plt.title(key_var + " = "+str(rg_var[k]))
        plt.savefig(stamp+"__"+str(k)+".png")
        plt.close()


        img = cv2.imread(stamp+"__"+str(k)+".png")
        h,w,l = img.shape
        size = (w,h)
        img_array.append(img)
        os.remove(stamp+"__"+str(k)+".png")

    plt.ion()

    if bool_save :
        if not [filename for filename in os.listdir('.') if filename == stamp] :
            os.mkdir(stamp)
        np.save(stamp+"\\"+stamp+"_param", param, allow_pickle=True)

    vid_name = stamp+"_"+name+'_VarOf_'+key_var
    suff = 0
    if vid_name+'.mp4' in os.listdir('.') or vid_name+'.avi' in os.listdir('.') :
        vid_name = vid_name+"__"+str(suff)
    while vid_name+'.mp4' in os.listdir('.') or vid_name+'.avi' in os.listdir('.') :
        suff+=1
        vid_name = vid_name[:-1 -len(str(suff-1))]+"_"+str(suff)

    out = cv2.VideoWriter(vid_name+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
#     out = cv2.VideoWriter(vid_name+'.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    if bool_save :
        shutil.copy(vid_name+'.avi', stamp+'\\')
#         shutil.copy(vid_name+'.mp4', stamp+'\\')

def _2D_plot_from_vals(X, Y, Z, z_min = None, z_max = None, nameX = '', nameY = '', name_plot = '', bool_3D = True, bool_show = True, bool_save = False):
    plt.close('all')

    if z_min is None :
        z_min = np.min(Z)
    if z_max is None:
        z_max = np.max(Z)

    if bool_3D :
        name_plot += " (3D)"
        X_c, Y_c = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X=X_c, Y=Y_c, Z=Z.T, vmin = z_min, vmax = z_max, linewidth=0, antialiased=False, cmap = 'viridis')
        fig.colorbar(surf)
        ax.set_xlim([X[0], X[-1]])
        ax.set_ylim([Y[0], Y[-1]])
        ax.set_zlim([z_min, z_max])
        ax.set_title(name_plot)
        ax.set_xlabel(nameX)
        ax.set_ylabel(nameY)

    else:
        name_plot += " (color)"
        plt.figure()
        plt.imshow(Z.T, vmin = z_min, vmax = z_max, aspect = 'auto', origin = 'lower', extent = (X[0], X[-1], Y[0], Y[-1]))
        plt.xlabel(nameX)
        plt.ylabel(nameY)
        plt.colorbar()
        plt.title(name_plot)

    if bool_save:
#         print(stamp+'_'+name_plot+'_var_'+nameX+'&'+nameY+'.pdf')
#         plt.savefig(stamp+'_'+name_plot+'_var_'+nameX+'&'+nameY+'.pdf')
        plt.savefig(stamp+'_'+name_plot+'.pdf')
    if bool_show :
        plt.show()

def _dyn_2D_plot_from_vals(X, Y, T, Zs, z_min = None, z_max = None, nameX = '', nameY = '', name_plot = '', name_vid = "", bool_3D = False, fps = 5,
                        bool_show = True, bool_save = False, stamp = None, talk = False, freq_mess = .2):
    plt.ioff()
    plt.close('all')

    if z_min is None :
        z_min = np.min(Zs)
    if z_max is None:
        z_max = np.max(Zs)
    if stamp is None:
        stamp = mk_stamp()

    Nt_gif = len(T)
    name_plot += " (dyn)"

    img_array = []
    deb = time.time()
    for ii in range(Nt_gif):
        if talk :
            timer(ii, Nt_gif, freq_mess, deb, extra_mess = 'Part 2')

        _2D_plot_from_vals(X, Y, Zs[ii], z_min, z_max, nameX = nameX, nameY = nameY,
                           name_plot = name_plot+' & t ='+str(T[ii]), bool_3D = bool_3D, bool_show = False, bool_save = False)

        plt.savefig(stamp+"__"+str(ii)+".png")
        plt.close()


        img = cv2.imread(stamp+"__"+str(ii)+".png")
        h,w,l = img.shape
        size = (w,h)
        img_array.append(img)
        os.remove(stamp+"__"+str(ii)+".png")

    plt.ion()

    if bool_save :
        if not [filename for filename in os.listdir('.') if filename == stamp] :
            os.mkdir(stamp)
        np.save(stamp+"\\"+stamp+"_param", param, allow_pickle=True)

    if bool_3D:
        name_vid = name_vid + "_3D"
    else :
        name_vid = name_vid + "_color"
    name_vid = stamp+"_"+name_vid+"_dyn"
    suff = 0
    if name_vid+'.mp4' in os.listdir('.') or name_vid+'.avi' in os.listdir('.') :
        name_vid = vid_name+"__"+str(suff)
    while name_vid+'.mp4' in os.listdir('.') or name_vid+'.avi' in os.listdir('.') :
        suff+=1
        name_vid = name_vid[:-1 -len(str(suff-1))]+"_"+str(suff)

    out = cv2.VideoWriter(name_vid+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
#     out = cv2.VideoWriter(vid_name+'.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    if bool_save :
        shutil.copy(name_vid+'.avi', stamp+'\\')
#         shutil.copy(vid_name+'.mp4', stamp+'\\')


# def _2D_rg_plot(fct, param, rg1, rg2, change_var1, change_var2, key_var1, key_var2, name_plot,
#              z_min = None, z_max = None, bool_3D = False, stamp = None, bool_save = False, bool_show = True, talk = True, freq_mess = .2, **kw_plot_param):
#
#     res = np.zeros((len(rg1), len(rg2)))
#
#     if stamp is None :
#         if 'stamp' in param.keys():
#             stamp = param['stamp']
#         else :
#             stamp = mk_stamp(param['dim'])
#
#     deb = time.time()
#     for k in range(len(rg1)):
#         for kk in range(len(rg2)):
#             if talk :
#                 timer(len(rg2)*k + kk, len(rg1)*len(rg2), freq_mess, deb, extra_mess = '')
#
#             param = change_var1(param, rg1[k])
#             param = change_var2(param, rg2[kk])
#
#             res[k,kk] = fct(param,  bool_show = False, **kw_plot_param)
#
#     if z_min is None :
#         z_min = np.min(res)
#     if z_max is None:
#         z_max = np.max(res)
#
#     _2D_plot_from_vals(rg1, rg2, res, z_min = z_min,z_max = z_max, nameX = key_var1, nameY = key_var2, name_plot = name_plot+'_3D_',
#                        bool_3D = bool_3D, bool_show = bool_show, bool_save = bool_save)
#
#     return res


# def _2D_evolving_rg_plot(fct, param, Nt_gif, rg1, rg2, change_var1, change_var2, key_var1, key_var2, name_plot, name_vid, bool_3D = False, fps = 5,
#              z_min = None, z_max = None, stamp = None, bool_save = False, talk = True, freq_mess = .2, **kw_plot_param):
#
#     res = np.zeros((Nt_gif, len(rg1), len(rg2)))
#
#     if stamp is None :
#         if 'stamp' in param.keys():
#             stamp = param['stamp']
#         else :
#             stamp = mk_stamp(param['dim'])
#
#     deb = time.time()
#     for k in range(len(rg1)):
#         for kk in range(len(rg2)):
#             if talk :
#                 timer(len(rg2)*k + kk, len(rg1)*len(rg2), freq_mess, deb, extra_mess = 'Part 1')
#
#             param = change_var1(param, rg1[k])
#             param = change_var2(param, rg2[kk])
#
#             absc, ordo = fct(param, **kw_plot_param)
#             for kkk in range(Nt_gif):
#                 stp = kkk*int(len(absc)/Nt_gif)
#                 res[kkk,k,kk] = ordo[stp]
#
#     _dyn_2D_plot_from_vals(rg1, rg2, np.arange(Nt_gif)*param['dt'], res, z_min = z_min, z_max = z_max, nameX = key_var1, nameY = key_var2, name_plot = name_plot,
#                         name_vid=name_vid, bool_3D = bool_3D,
#                         bool_show = True, bool_save = bool_save, stamp = param['stamp'], talk = talk, freq_mess = freq_mess)
#
#     return res

