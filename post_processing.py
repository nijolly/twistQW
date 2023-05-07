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

curr_dir = "C:\\Users\\nicol\Documents\Work\ENS related stuff\M1\Stage M1\Code\Python"
sys.path.append(curr_dir)

from formalism import *

from utils import *
from utils import _mk_param_edit_fct, _next, _analyze_axis

from quantum_and_walks import *
from quantum_and_walks import _spin_mat_to_op_nD, _operator_composition, _i_ax

from displayer import *
from displayer import _plot_spec, _plot_spread_map, _plot_mean_from_mean, _plot_std_from_std, _plot_entrop_from_entrop

path_data =curr_dir+"\data"
os.chdir(path_data)


default_force=True
## Look at Final state

def _plot_final_state(param, talk = True, frame = None, z_max = None, freq_mess = .1, **kw_param): #iterable in mk_gif_gen...
    if 'dic_freq_mess' in kw_param.keys():
        freq_mess = kw_param['dic_freq_mess']
    if 'dic_talk' in kw_param.keys():
        talk = kw_param['dic_talk']
    if 'dic_frame' in kw_param.keys():
        frame = kw_param['dic_frame']
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']


    final, _ = simu(param, bool_save = False, force = True, talk=talk, freq_mess=freq_mess)
    plot_state(final, param['dxs'], frame, z_max)

def plot_final_state(stamp, talk = True, frame = None, z_max = None, freq_mess = .1, force = default_force, bool_save_final = False, bool_save_data = False, bool_plot = True):
    final, _ = simu(param, bool_save = bool_save_data, force = force, talk=talk, freq_mess=freq_mess)

    if bool_save_final :
        np.save(stamp+"_final", final)

    if bool_plot :
        plot_state(final, param['dxs'], frame, z_max)



## Look at spreadmap / density plot
def _mk_spread_map_1D_from_data(data, dt, dx, X = None, bool_plot = False, z_max = None):
    if X is None:
        X = _i_ax(0, data[-1], [dx])
    Y = np.arange(len(data))*dt
    mat = np.zeros((len(data), len(X)))
    for k in range(len(data)):
        mat[k,:] = proba(data[k], [dx], [X])

    if bool_plot :
        _plot_spread_map(mat, X, Y, z_max)

    return mat, X, Y

def _mk_spread_map_1D_from_param(param, z_max = None, X = None, bool_plot = True, **kw_param):
    if param['dim'] > 1 :
        print('The spread map representation only makes sense for dim 1, here dim is '+str(dim))
        raise IrrelevantForDim("spread_map")

    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_X' in kw_param.keys():
        X = kw_param['dic_X']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    os.remove(temp_stamp+"_data.npy")
    param["stamp"] = real_stamp

    if X is None:
        X = _i_ax(0, data[-1], [param['dxs'][0]])

    return _mk_spread_map_1D_from_data(data, param['dt'], param['dxs'][0], X = X, bool_plot = bool_plot, z_max = z_max)

def mk_spread_map(stamp, bool_plot = True, z_max = None, force = default_force):
    param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
    if param['dim'] > 1 :
        print('The spread map representation only makes sense for dim 1, here dim is '+str(dim))
        raise IrrelevantForDim("spread_map")

    if len([filename for filename in os.listdir('.') if filename == stamp+'_spread_mat.npy']) >0 and not force :
        print("re-use spread")
        mat = np.load(stamp+"_spread_mat.npy")
        X = np.load(stamp+"_spread_mat_X.npy")
        Y = np.load(stamp+"_spread_mat_Y.npy")
    elif len([filename for filename in os.listdir('.') if filename == stamp+'_data.npy']) >0 and not force:
        print("re-create spread from existing data")
        data = np.load(stamp+"_data.npy",  allow_pickle=True)
        dt = param['dt']
        dx = param['dxs'][0]
        mat, X, Y = _mk_spread_map_1D_from_data(data, dt, dx, bool_plot=False)
    else :
        print("re-create spread from scratch")
        mat, X, Y = _mk_spread_map_1D_from_param(param, z_max = None, bool_plot=False)

    np.save(stamp+"_spread_mat", mat)
    np.save(stamp+"_spread_mat_X", X)
    np.save(stamp+"_spread_mat_Y", Y)

    if bool_plot :
        _plot_spread_map(mat, X,Y, z_max)
        plt.show()

    return mat, X, Y



## Gif of state evolution
def _mk_gif_2D(stamp, data, dx, dy, fps, keep, z_max=None, X=None, Y=None, path=None):
    img_array = []
    nb=0

    dxs = [dx, dy]

    if X is None:
        X = _i_ax(0, data[-1], dxs)
    if Y is None:
        Y = _i_ax(1, data[-1], dxs)
    frame = [X,Y]

    if z_max is None:
        z_max = np.max(proba(data[0], dxs))

    plt.close('all')
    plt.ioff()
    for st in data:
        if not int(keep*nb) != int(keep*(nb-1)) :
            nb+=1
            continue

        mat = proba(st, dxs, frame)
        plt.figure()
        plt.imshow(mat, vmin=0, vmax=z_max, origin = 'lower', extent = (X[0], X[-1], Y[0], Y[-1]))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()
        plt.title("Evolution of the probability density")
        plt.savefig(stamp+"__"+str(nb)+".png")
        plt.close()


        img = cv2.imread(stamp+"__"+str(nb)+".png")
        h,w,l = img.shape
        size = (w,h)
        img_array.append(img)
        os.remove(stamp+"__"+str(nb)+".png")
        nb+=1

    plt.ion()
    if path is None:
        path = ''
    out = cv2.VideoWriter(path+stamp+'.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def _mk_gif_1D(stamp, data, dx, fps, keep, z_max = None, X=None, path=None):
    img_array = []
    nb=0

    dxs = [dx]

    if X is None:
        X = _i_ax(0, data[-1], dxs)
    frame = [X]

    if z_max is None:
        z_max = np.max(proba(data[0], dxs))

    plt.close('all')
    plt.ioff()
    for st in data:
        if not int(keep*nb) != int(keep*(nb-1)) :
            nb+=1
            continue

        fr, ordo= display(st, dxs, frame)
        absc = fr[0]
        plt.plot(absc, ordo)
        plt.xlabel("X")
        plt.ylabel("p")
        plt.ylim(0, z_max)
        plt.title("Evolution of the probability density")
        plt.savefig(stamp+"__"+str(nb)+".png")
        plt.close()


        img = cv2.imread(stamp+"__"+str(nb)+".png")
        h,w,l = img.shape
        size = (w,h)
        img_array.append(img)
        os.remove(stamp+"__"+str(nb)+".png")
        nb+=1

    plt.ion()
    if path is None:
        path = ''
    vid_name = path+stamp
    out = cv2.VideoWriter(vid_name+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

#     print("giff", nb, vid_name)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def mk_gif(stamp, fps, keep, z_max=None, frame = None, path='', force = default_force):
    #keep is the ratio of image we keep in the video

    param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
    if param['dim'] > 2 :
        print('The gif representation only makes sense for dim 1 and 2, here dim is '+str(param['dim']))
        raise IrrelevantForDim("gif")

    if not force and len([filename for filename in os.listdir('.') if filename == stamp+'.mp4']) >0 :
        print("re-use gif")
        if (not path is None) and path != '.' and path != '':
            shutil.copy(stamp+".mp4", path)
        return None

    print("re create gif")
    data = np.load(stamp+"_data.npy",  allow_pickle=True)
    if param['dim'] == 1:
        dx = param['dxs'][0]
        X = None
        if not frame is None:
            X = frame[0]
        return _mk_gif_1D(stamp, data, dx, fps, keep, z_max, X, path)

    if param['dim'] == 2:
        dx = param['dxs'][0]
        dx = param['dxs'][1]
        X = None
        Y = None
        if not frame is None:
            X = frame[0]
            Y = frame[1]
        return _mk_gif_2D(stamp, data, dx, dy, fps, keep, z_max, X, Y, path)


def mk_gif_from_param(param, fps=25, keep = 1, z_max=None, frame = None, path = '', force = default_force):
    _ = simu(param, talk = False, bool_save=True)
    np.save(param['stamp']+"_param.npy",param)
    mk_gif(param['stamp'], fps, keep, z_max, frame, path, force)




## Look at Mean
def _mean_state(state, dxs):
    frame,pb = display(state, dxs)
    dim = len(dxs)

    avgs = np.zeros(dim)

    tpl = (0,)*dim
    while not tpl is None :
        for k in range(dim):
            avgs[k] += pb[tpl]*frame[k][tpl[k]]

        tpl = _next(tpl, state.shape[:dim])

    return avgs


def _mean_from_data(data, dt, dxs, bool_plot=True, z_min = None, z_max = None):
    dim = len(dxs)
    ordo = np.zeros((len(data),dim))
    absc = np.arange(len(data))*dt

    for k in range(len(data)):
        ordo[k] = _mean_state(data[k], dxs)

    if bool_plot :
        _plot_mean_from_mean(absc, ordo, z_min, z_max)
        plt.show()

    return absc, ordo

def repr_mean(stamp, bool_plot = True, force = default_force, z_min = None, z_max = None):
    if not force and len([filename for filename in os.listdir('.') if filename == stamp+'_mean_absc.npy']) >0 :
        print("re-use mu")
        ordo = np.load(stamp+"_mean_ord.npy")
        absc = np.load(stamp+"_mean_absc.npy")

    else:
        print("re create mu")
        data = np.load(stamp+"_data.npy",  allow_pickle=True)
        param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
        dt = param['dt']
        dxs = param['dxs']
        absc, ordo = _mean_from_data(data, dt, dxs, bool_plot = False)
        np.save(stamp+"_mean_ord", ordo)
        np.save(stamp+"_mean_absc", absc)

    if bool_plot :
        _plot_mean_from_mean(absc, ordo, z_min, z_max)
        plt.show()

    return absc, ordo


def _plot_mean_data(data, dt, dxs, z_min = None, z_max = None, subset = None, scatter = False, marker = None, s=None, c = None, bool_plot = True):
    absc, ordo = _mean_from_data(data, dt, dxs, bool_plot=False)
    if not subset is None:
        sub_ordo = []
        sub_absc = []
        for k in subset:
            sub_ordo.append(ordo[val_to_ind(k, absc)])
            sub_absc.append(absc[val_to_ind(k, absc)])

        ordo = np.array(sub_ordo)
        absc = np.array(sub_absc)
    if bool_plot :
        _plot_mean_from_mean(absc, ordo, z_min, z_max, scatter, marker, s, c)
    return absc, ordo

def _plot_mean_from_param(param, z_min = None, z_max = None, subset = None, scatter = False, marker = None, s=None, c = None, bool_plot = True, **kw_param):
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_z_min' in kw_param.keys():
        z_min = kw_param['dic_z_min']
    if 'dic_subset' in kw_param.keys():
        subset = kw_param['dic_subset']
    if 'dic_scatter' in kw_param.keys():
        scatter = kw_param['dic_scatter']
    if 'dic_marker' in kw_param.keys():
        marker = kw_param['dic_marker']
    if 'dic_s' in kw_param.keys():
        s = kw_param['dic_s']
    if 'dic_c' in kw_param.keys():
        c = kw_param['dic_c']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    os.remove(temp_stamp+"_data.npy")
    param["stamp"] = real_stamp

    return _plot_mean_data(data, param['dt'], param['dxs'], z_min, z_max, subset, scatter, marker, s, c, bool_plot)

def _plot_mean(stamp, z_min = None, z_max = None):
    absc, ordo = repr_mean(stamp, bool_plot = False)
    _plot_mean_from_mean(absc, ordo, z_min, z_max)



## Look at Variance
def _cov_mat_state(state, dxs):
    frame,pb = display(state, dxs)
    dim = len(dxs)

    res = np.zeros((dim, dim))
    avgs = np.zeros(dim)

    tpl = (0,)*dim
    while not tpl is None :
        for k in range(dim):
            avgs[k] += pb[tpl]*frame[k][tpl[k]]
            for kk in range(dim):
                res[k,kk] += pb[tpl] * frame[k][tpl[k]] * frame[kk][tpl[kk]]

        tpl = _next(tpl, state.shape[:dim])

    for k in range(dim):
        for kk in range(dim):
            if k == kk and avgs[k]**2 > res[k,kk]:
                print(np.sum(pb))
                print(avgs[k], avgs[kk], res[k,kk])
            res[k,kk] -= avgs[k]*avgs[kk]

    return res



def _std_from_data(data, dt, dxs, bool_plot=True, z_max = None, cov_min=None, cov_max=None):
    dim = len(dxs)
    ordo = np.zeros((len(data),dim, dim))
    absc = np.arange(len(data))*dt

    for k in range(len(data)):
        ordo[k] = _cov_mat_state(data[k], dxs)
        if np.min(np.diag(ordo[k])) < 0 :
            print(ordo[k])

    if bool_plot :
        _plot_std_from_std(absc, ordo, z_max, cov_min, cov_max)
        plt.show()

    return absc, ordo

def repr_std(stamp, bool_plot = True, force = default_force, z_max = None, cov_min=None, cov_max=None):
    if not force and len([filename for filename in os.listdir('.') if filename == stamp+'_sigma_absc.npy']) >0 :
        print("re-use sig")
        ordo = np.load(stamp+"_sigma_ord.npy")
        absc = np.load(stamp+"_sigma_absc.npy")

    else:
        print("re create sig")
        data = np.load(stamp+"_data.npy",  allow_pickle=True)
        param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
        dt = param['dt']
        dxs = param['dxs']
        absc, ordo = _std_from_data(data, dt, dxs, bool_plot = False)
        np.save(stamp+"_sigma_ord", ordo)
        np.save(stamp+"_sigma_absc", absc)

    if bool_plot :
        _plot_std_from_std(absc, ordo, z_max, cov_min, cov_max)
        plt.show()

    return absc, ordo


def _plot_std_data(data, dt, dxs, z_max = None, cov_min=None, cov_max=None, subset = None, scatter = False, marker = None, s=None, c = None, bool_plot = True):
    absc, ordo = _std_from_data(data, dt, dxs, bool_plot=False)

    if not subset is None:
        sub_ordo = []
        sub_absc = []
        for k in subset:
            sub_ordo.append(ordo[val_to_ind(k, absc)])
            sub_absc.append(absc[val_to_ind(k, absc)])

        ordo = np.array(sub_ordo)
        absc = np.array(sub_absc)

    if bool_plot:
        _plot_std_from_std(absc, ordo, z_max, cov_min, cov_max, scatter, marker, s, c)
    return absc, ordo

# param, z_max = None, bool_plot = False, X = None, **kw_param):
def _plot_std_from_param(param, z_max = None, cov_min = None, cov_max = None, subset = None, scatter = False, marker = None, s=None, c = None, bool_plot = True, **kw_param):
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_cov_min' in kw_param.keys():
        cov_min = kw_param['dic_cov_min']
    if 'dic_cov_max' in kw_param.keys():
        cov_max = kw_param['dic_cov_max']
    if 'dic_subset' in kw_param.keys():
        subset = kw_param['dic_subset']
    if 'dic_scatter' in kw_param.keys():
        scatter = kw_param['dic_scatter']
    if 'dic_marker' in kw_param.keys():
        marker = kw_param['dic_marker']
    if 'dic_s' in kw_param.keys():
        s = kw_param['dic_s']
    if 'dic_c' in kw_param.keys():
        c = kw_param['dic_c']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    os.remove(temp_stamp+"_data.npy")
    param["stamp"] = real_stamp

    return _plot_std_data(data, param['dt'], param['dxs'], z_max, cov_min, cov_max, subset, scatter, marker, s, c, bool_plot)

def _plot_std(stamp, z_max = None, cov_min=None, cov_max=None):
    absc, ordo = repr_std(stamp, bool_plot = False)
    _plot_std_from_std(absc, ordo, z_max, cov_min, cov_max)




## Look at Entropy of Entanglement
def _entrop(state, dxs):
    rho_c = np.zeros((state.shape[-1], state.shape[-1]))*0j
    tpl = (0,)*len(dxs)
    while not tpl is None :
        rho_c = rho_c + np.outer(state[tpl], state[tpl].T.conj())
        tpl = _next(tpl, state.shape[:-1])


    rho_c = np.array(rho_c, dtype = np.complex128)
    vp, _ = np.linalg.eig(rho_c)
    res = 0
    for val in vp:
        if np.abs(val.imag) < 1e-15 and val.real > 0 :
            val = np.abs(val)
            res -= val * np.log2(val)
        elif val.real < -1e-15 or np.abs(val.imag) > 1e-15 :
            print("eigval not strict positive", val)
#             raise ValueError
    return res





def _entrop_data(data, dt, dxs, z_min = None, z_max = None, bool_plot = True):
    dim = len(dxs)
    absc = np.arange(len(data))*dt
    ordo = []
    for k in range(len(data)):
        ordo.append(_entrop(data[k], dxs))

    if bool_plot :
        _plot_entrop_from_entrop(absc, ordo, z_min, z_max)
        plt.show()

    return absc, ordo

def repr_entrop(stamp, bool_plot = True, force = default_force, z_min = None, z_max = None):
    if not force and len([filename for filename in os.listdir('.') if filename == stamp+'_entrop.npy']) >0 :
        print("re-use S")
        ordo = np.load(stamp+"_entrop_ord.npy")
        absc = np.load(stamp+"_entrop_absc.npy")

    else:
        print("re create sig")
        data = np.load(stamp+"_data.npy",  allow_pickle=True)
        param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
        dt = param['dt']
        dxs = param['dxs']
        absc, ordo = _entrop_data(data, dt, dxs, z_min, z_max, bool_plot = False)
        np.save(stamp+"_entrop_ord", ordo)
        np.save(stamp+"_entrop_absc", absc)

    if bool_plot :
        _plot_entrop_from_entrop(absc, ordo, z_min, z_max)


    return absc, ordo


def _plot_entrop_from_param(param, z_min =  None, z_max = None, bool_figure = True, **kw_param):
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_z_min' in kw_param.keys():
        z_max = kw_param['dic_z_min']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_data.npy")
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    param["stamp"] = real_stamp
    return _plot_entrop_data(data, param['dt'], param['dxs'], z_min, z_max, bool_figure)

def _get_entrop_from_param(param, **kw_param):
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_z_min' in kw_param.keys():
        z_max = kw_param['dic_z_min']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_data.npy")
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    param["stamp"] = real_stamp
    return  _entrop_data(data, param['dt'], param['dxs'], bool_plot=False)

def _plot_entrop_data(data, dt, dxs, z_min = None, z_max = None, bool_figure = True):
    absc, ordo = _entrop_data(data, dt, dxs, bool_plot=False)
    _plot_entrop_from_entrop(absc, ordo, z_min, z_max, bool_figure)
    return absc, ordo

def _get_entrop_final_from_param(param, force = default_force, **kw_param):
    if 'dic_force' in kw_param.keys():
        force = kw_param['dic_force']

    final,_ = simu(param, bool_save = False, force = force, talk = False)
    return _entrop(final, param['dxs'])

def _get_entrop_from_param(param, z_min =  None, z_max = None, **kw_param):
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']
    if 'dic_z_min' in kw_param.keys():
        z_max = kw_param['dic_z_min']

    real_stamp = param['stamp']
    temp_stamp = 'temp_'+mk_stamp(dim=1)
    param['stamp'] = temp_stamp
    _, _ = simu(param, bool_save = True, force = True, talk = False)

    data = np.load(temp_stamp+"_data.npy", allow_pickle = True)
    os.remove(temp_stamp+"_data.npy")
    os.remove(temp_stamp+"_param.npy")
    os.remove(temp_stamp+"_final.npy")
    param["stamp"] = real_stamp
    return _entrop_data(data, param['dt'], param['dxs'], bool_plot=False)




# ## 1.10 entrop analysis
#
# def _ind_first_max_from_val(absc, ordo):
#     for k in range(len(absc)-1):
#         if ordo[k+1] < ordo[k]:
#             return k
#     return k+1
#
#
# def ind_first_max_from_param(param):
#     absc, ordo = _get_entrop_from_param(param)
#     return _ind_first_max_from_val(absc, ordo)
# def _absc_first_max_from_val(absc, ordo):
#     return absc[_ind_first_max_from_val(absc, ordo)]
# def absc_first_max_from_param(param):
#     return absc[ind_first_max_from_param(param)]
#
# def _val_cvg_from_val(absc, ordo):
#     k = _ind_first_max_from_val(absc,ordo)
#     if k == len(absc)-1 : #case where strict increasing
#         return ordo[k]
#
#     else:
#         kk = k+_ind_first_max_from_val(absc[k:],-ordo[k:])
#         if kk == len(absc)-1: #case where one peak then cvg
#             return ordo[kk]
#         else:  #case with oscillations
#             return np.mean(ordo[k:])
# def val_cvg_from_param(param):
#     absc, ordo =  _get_entrop_from_param(param)
#     return _val_cvg_from_val(absc, ordo)
#
# def _number_extrema_from_val(absc, ordo):
#     for k in range(len(absc)-2):
#         if (ordo[k+2]-ordo[k+1])*(ordo[k+1]-ordo[k]) < 0:
#             return 1+_number_extrema_from_val(absc[k+1:], ordo[k+1:])
#     return 0
# def number_extrema_from_param(param):
#     absc, ordo =  _get_entrop_from_param(param)
#     return _number_extrema_from_val(absc, ordo)
#
# def ordo_from_res_dyn(res, k, kk):
#     return res[:,k,kk]
#
#
# def _then_within_from_vals(absc, ordo, val = None, acc = .1):
#     if val is None:
#         val = _val_cvg_from_val(absc, ordo)
#
#     for k in range(len(ordo)-1, -1, -1):
#         if np.abs(ordo[k]-val) > acc*val :
#             return absc[k]
#     return absc[0]
#
# def then_within_from_param(param, val=None, acc=.1):
#     absc, ordo =  _get_entrop_from_param(param)
#     return _then_within_from_vals(absc, ordo, val, acc)
#
#
# def _fit_from_vals(absc, ordo, fct = lambda t,tau,om,lim : (1-np.exp(-t/tau) * np.cos(om*t))*lim ):
#     pamms,qual = scipy.optimize.curve_fit(fct, absc, ordo)
#     return pamms
#
# def fit_from_param(param,  fct):
#     absc, ordo =  _get_entrop_from_param(param)
#     return _fit_from_vals(absc, ordo, fct)