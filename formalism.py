import sys
import os
import inspect
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time

import datetime
from math import *
from scipy.linalg import expm, logm
import scipy.linalg
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (10, 7)

curr_dir = "C:\\Users\\nicol\Documents\Work\ENS related stuff\M1\Stage M1\Code\Python"
sys.path.append(curr_dir)


from utils import *
from utils import _mk_param_edit_fct, _next, _analyze_axis

from quantum_and_walks import *
from quantum_and_walks import _spin_mat_to_op_nD, _operator_composition, _i_ax

from displayer import *
from displayer import _plot_spec, _plot_spread_map, _plot_mean_from_mean, _plot_std_from_std, _plot_entrop_from_entrop



path_data =curr_dir+"\data"
os.chdir(path_data)

## Brief introduction
# All the data about the system is contained in a dictionnary called param.
# It contains :
#             The type of walk
#             The matrices used at each step encoded in a string called 'order' (cf later)
#             All the angles and parameters detailled in the paper (cf the article or the notebook)
#             The parameters regarding the implementation of the walk (number of points, number of iterations...)
#             Finally it contains the initial state.
#
#             All this data is identified by a stamp that can be used to save and load data
#
#             Because of the redundancy of the data, one can specify a small umber of parameters
#             and then call one of the 'update-functions'. (These are also useful when varying one
#             parameters, the parameters that are defined from this one will then be updated.)
#
#
#             From this one can look at the spectrumin momuntum space.
#             One can also run a simulation.
#             One can look at the post processing quantities (mean, variance, entropy...) and the various
#             representations of the run (plot of the final state, gif, spreadmap...). This can be done
#             directly of from the data obtained from the simulation.












## Construct the system

def _analyze_gate(st, param):
    """ Treats individual gates of the 'order' parameter :

    The gates must be separated by a '_' ;
    The '_Sm_' will be recognized as shifts in direction m ;
    The '_Cn_' will be recognized as coin gates with angles prefixed by '_n' ;
    Any other string, will be considered as an operator acting uniformly on all spins
    with the matrix accessible in param by the key corresping to this string"""

    if st[0] == 'S' :
        return shift_op_mk_nD(axis = st[1], overflow = param['overflow'])
    elif st[0] == 'C' :
        return coin_op_mk(param['delta_'+st[1]], param['theta_'+st[1]], param['phi_'+st[1]], param['zeta_'+st[1]])
    elif st[0] == 'H':
        return coin_op_mk(*_hadamard_pm)
    else :
        return _spin_mat_to_op_nD(param[st])

def _talk_gate(st):
    if st[0] == 'S' :
        print("  ---> Shift in direction ", st[1])
    elif st[0] == 'C' :
        print('  ---> Coin with angles delta_'+st[1]+'; theta_'+st[1]+'; phi_'+st[1]+'; zeta_'+st[1])
    elif st[0] == 'H' :
        print('  ---> Hadamard Coin ')
    else :
        print("  ---> Operator defined by matrix stored in parameters as ", st)

def _analyze_order(param, talk = True):
    ''' treats the whole "order" parameter '''

    string = param['order']
    gate_list = string.split("_")

    if 'M' in gate_list :
        gate_list = gate_list[:gate_list.index('M')] +['M']+ gate_list[gate_list.index('M')+1 :] * param['tau']


    res = []
    if talk :
        print("The Unitary consists in the following process :")
        for st in gate_list[::-1] :
            _talk_gate(st)

    for st in gate_list:
        res.append(_analyze_gate(st, param))
    return res

def unitary_from_order(param, talk = False):
    ''' turns the order parameter (string) into a unitary operator'''

    op_list = _analyze_order(param, talk)
    return unitary_mk(op_list)


## Complete param
def update_param_1D(param):
    res = {}

    res['dim'] = param['dim']
    res['eps'] = param['eps']
    res['a'] = param['a']
    res['b'] = param['b']
    res['Nt'] = param['Nt']
    res['max_width'] = param['max_width']

    res['theta_0_x'] = param['theta_0_x']
    res['theta_1_x'] = param['theta_1_x']

    eps = res['eps']
    a = res['a']
    b = res['b']
    Nt = res['Nt']
    max_width = res['max_width']

    dxs = [eps**a]
    dt = eps
    Lt = Nt*dt
    overflow = max_width/dxs[0] #unit is number of array indices
    theta_X = res['theta_0_x'] + eps**b * res['theta_1_x']

    res['dxs'] = dxs
    res['dt'] = dt
    res['Lt'] = Lt
    res['overflow'] = overflow
    res["theta_x"] = theta_X

    for ke in param.keys():
        if not ke in res.keys():
            res[ke] = param[ke]

    if len(res['dxs']) != res['dim'] :
        raise IrrelevantForDim

    return res


def update_param_1D_twist(param):
    res = {}

    res['dim'] = param['dim']
    res['eps'] = param['eps']
    res['a'] = param['a']
    res['b'] = param['b']
    res['Nt'] = param['Nt']
    res['max_width'] = param['max_width']

    res['alpha_1'] = param['alpha_1']
    res['theta_0_x'] = param['theta_0_x']
    res['alpha_1_x'] = param['alpha_1']
    res['theta_0_y'] = param['theta_0_y']
    res['alpha_1_y'] = param['alpha_1'] #beware, theta_1_y = theta_1_x !!

    res['m'] = param['m']
    res['theta'] = param['theta']
    res['phi'] = param['phi']


    eps = res['eps']
    a = res['a']
    b = res['b']
    Nt = res['Nt']
    max_width = res['max_width']
    alpha_1 = res['alpha_1']

    m = res['m']
    theta = res['theta']
    phi = res['phi']

    dxs = [eps**a]
    dt = eps
    Lt = Nt*dt
    overflow = max_width/dxs[0]
    theta_X = res['theta_0_x'] - eps**b * 2*alpha_1
    theta_Y = res['theta_0_y'] - eps**b * 2*alpha_1

    R = rot(2, theta).dot(rot(3, phi))
    T = np.linalg.inv(R)
    M = rot(3, -2*m*dt)

    res['dxs'] = dxs
    res['dt'] = dt
    res['Lt'] = Lt
    res['overflow'] = overflow
    res["theta_x"] = theta_X
    res["theta_y"] = theta_Y

    param['R'] = R
    param['T'] = T
    param['M'] = M

    for ke in param.keys():
        if not ke in res.keys():
            res[ke] = param[ke]

    if len(res['dxs']) != res['dim'] :
        raise IrrelevantForDim

    return res

def update_param_1D_order2(param):
    res = {}

    res['dim'] = param['dim']
    res['eps'] = param['eps']
    res['a'] = param['a']
    res['b'] = param['b']
    res['c'] = param['c']
    res['Nt'] = param['Nt']
    res['max_width'] = param['max_width']

    res['ax1'] = param['ax1']
    res['ax2'] = param['ax2']

    res['alpha_1'] = param['alpha_1']
    res['theta_0_x'] = param['theta_0_x']
    res['alpha_1_x'] = param['alpha_1']
    res['theta_0_y'] = param['theta_0_y']
    res['alpha_1_y'] = param['alpha_1'] #beware, theta_1_y = theta_1_x !!

    res['m'] = param['m']
    res['theta_1'] = param['theta_1']


    eps = res['eps']
    a = res['a']
    b = res['b']
    c = res['c']
    ax1 = res['ax1']
    ax2 = res['ax2']
    Nt = res['Nt']
    max_width = res['max_width']
    alpha_1 = res['alpha_1']

    m = res['m']
    theta = res['theta_1'] * eps**c

    dxs = [eps**a]
    dt = eps
    Lt = Nt*dt
    overflow = max_width/dxs[0]
    theta_X = res['theta_0_x'] - eps**b * 2*alpha_1
    theta_Y = res['theta_0_y'] - eps**b * 2*alpha_1

    R = rot(ax1, theta)
    T = np.linalg.inv(R)

    A = rot(ax2, theta)
    B = np.linalg.inv(A)

    M = rot(3, -2*m*dt)

    res['dxs'] = dxs
    res['dt'] = dt
    res['Lt'] = Lt
    res['overflow'] = overflow
    res['theta'] = theta
    res["theta_x"] = theta_X
    res["theta_y"] = theta_Y

    param['R'] = R
    param['T'] = T
    param['M'] = M
    param['A'] = A
    param['B'] = B

    for ke in param.keys():
        if not ke in res.keys():
            res[ke] = param[ke]

    if len(res['dxs']) != res['dim'] :
        raise IrrelevantForDim

    return res



## Init simu
def init_from_spin(spin, sz = 10):
    ''' Returns a state with 0 everywhere but at the center where lies the normalized spin'''
    res = []
    for _ in range(sz):
        res.append([0,0])
    res.append(normalize(spin))
    for _ in range(sz):
        res.append([0,0])
    return np.array(res)

def init_simple(shape = (2,)):
    spin = normalize(np.ones(shape))
    return init_from_spin(spin)

def gaussian(mu_vect, var_mat, dim):
    def fct(x):
        return 1/np.sqrt((2*np.pi)**dim * np.linalg.det(var_mat)) *  \
                  np.exp(-1/2 * (x-mu_vect).dot(np.linalg.inv(var_mat)).dot(x-mu_vect))
    return fct

# def get_eig_sp_1D(param, vect_choice = 'up'):
#
#     vect_func = spectrum(param, only_vect= True)
#     if vect_choice == 'up':
#         right_vect_func = lambda k : vect_func([k]).T[1]
#     elif vect_choice == 'down':
#         right_vect_func = lambda k : vect_func([k]).T[0]
#     else:
#         print("vect_choice should be 'up' or 'down'")
#         raise ValueError
#
#     def vect_phi(x):
#         comp0_real = scipy.integrate.quad(lambda kk:(np.exp(1j*kk*x)*right_vect_func(kk)[0]).real, -inf, +inf)
#         comp0_imag = scipy.integrate.quad(lambda kk:(np.exp(1j*kk*x)*right_vect_func(kk)[0]).imag, -inf, +inf)
#         comp1_real = scipy.integrate.quad(lambda kk:(np.exp(1j*kk*x)*right_vect_func(kk)[1]).real, -inf, +inf)
#         comp1_imag = scipy.integrate.quad(lambda kk:(np.exp(1j*kk*x)*right_vect_func(kk)[1]).imag, -inf, +inf)
#         return normalize(np.array([comp0_real[0]+1j*comp0_imag[0], comp1_real[0]+1j*comp1_imag[0]]))
#     return vect_phi


def init_gaussian_nD(shape, dxs, var_mat = None, mu_vect=None, spin=np.array([1,1]), rel_size_fact = 20):
    '''returns a gaussian distribution multiplied by the given spin'''
    if var_mat is None :
        var_mat = np.diag([(dxs[k]*shape[k]/rel_size_fact)**2 for k in range(len(dxs))])
    else:
        var_mat = var_mat/np.prod(dxs)
    if mu_vect is None :
        mu_vect = np.array([0]*len(shape))
    else:
        mu_vect = mu_vect/np.sqrt(np.prod(dxs))

    spin = normalize(spin)

    res = np.zeros(tuple(2*np.array(shape) + 1) + spin.shape)*1j
    frame = [np.arange(-shape[k], shape[k]+1)*dxs[k] for k in range(len(shape))]
    shape = tuple(2*np.array(shape) + 1)

    tpl = (0,)*len(shape)
    while not tpl is None :
        res[tpl] = spin*np.sqrt(gaussian(mu_vect, var_mat, len(tpl))(np.array([frame[i][tpl[i]] for i in range(len(tpl))])))
        tpl = _next(tpl, shape)
    return res*np.sqrt(np.prod(dxs))

# def init_gaussian_eigvect(shape, dxs, var_mat = None, mu_vect=None, vect_choice = 'up', rel_size_fact = 20):
#     if var_mat is None :
#         var_mat = np.diag([(dxs[k]*shape[k]/rel_size_fact)**2 for k in range(len(dxs))])
#     else:
#         var_mat = var_mat/np.prod(dxs)
#     if mu_vect is None :
#         mu_vect = np.array([0]*len(shape))
#     else:
#         mu_vect = mu_vect/np.sqrt(np.prod(dxs))
#
#     sp_fun = get_eig_sp_1D(param, vect_choice = vect_choice)
#     res = np.zeros(tuple(2*np.array(shape) + 1) + (2,))*1j
#     frame = [np.arange(-shape[k], shape[k]+1)*dxs[k] for k in range(len(shape))]
#     shape = tuple(2*np.array(shape) + 1)
#
#     tpl = (0,)*len(shape)
#     while not tpl is None :
#         spin = sp_fun(frame[0][tpl[0]])
#         res[tpl] = spin*np.sqrt(gaussian(mu_vect, var_mat, len(tpl))(np.array([frame[i][tpl[i]] for i in range(len(tpl))])))
#         tpl = _next(tpl, shape)
#     return res*np.sqrt(np.prod(dxs))

## Spectrum
def get_hamil_fourrier(param, form = 'mat'):
    '''takes in the param with all about the system
    and returns a function that gives the hamil(k) '''
    def hamil(k):
        string = param['order']
        tau = param['tau']
        gate_list = string.split("_")
        if 'M' in gate_list :
            gate_list = gate_list[:gate_list.index('M')] +['M']+ gate_list[gate_list.index('M')+1 :] * param['tau']

        mat_list = []
        for st in gate_list:
            if st[0] == 'C':
                mat_list.append(coin(param['delta_'+st[1]], param['theta_'+st[1]], param['phi_'+st[1]], param['zeta_'+st[1]]))
            elif st[0] == 'S':
                ax = _analyze_axis(st[1])
                mat_list.append(rot(3, -2*k[ax]*param['dxs'][ax]))
            elif st[0] == 'H':
                mat_list.append(hadamard_coin())
            else:
                mat_list.append(param[st])

        res = sig0
        for op_m in mat_list:
            res = res.dot(op_m)

        Wmat = res

        ham = 1j* logm(Wmat)/(tau*param["dt"])

        if form == 'mat':
            return ham
        elif form == 'op':
            return _spin_mat_to_op_nD(ham)
        else:
            print("form parameter not recognized")
            raise ValueError

    return hamil

def _diagonalize(mat):
    val, vect = np.linalg.eig(mat)
    idx = val.argsort()
    val = val[idx]
    vect = vect[idx]
    return val, vect

def spectrum(param, with_vect = False, only_vect=False):
    '''takes in the param with all about the system
    and returns a function that gives the spectrum(k) '''
    def fct_spec(k):
        ha = get_hamil_fourrier(param, form = 'mat')(k)
        if with_vect:
            return _diagonalize(ha)
        if only_vect :
            return _diagonalize(ha)[1]
        return _diagonalize(ha)[0]
    return fct_spec

#plotters
def _mk_spec_from_param(param, frame = None, z_min = None, z_max = None, bool_plot = True, bool_show = True, talk = True, typ = 'real', **kw_param): #iterable in mk_gif_gen...
    if 'dic_frame' in kw_param.keys():
        frame = kw_param['dic_frame']
    if 'dic_typ' in kw_param.keys():
        typ = kw_param['dic_typ']
    if 'dic_talk' in kw_param.keys():
        talk = kw_param['dic_talk']
    if 'dic_z_min' in kw_param.keys():
        z_min = kw_param['dic_z_min']
    if 'dic_z_max' in kw_param.keys():
        z_max = kw_param['dic_z_max']


    if frame is None :
        print("In function _mk_spec_from_param : \
                frame should either be defined OR put as dic_frame in **kwargs")
        raise ValueError

    if len(frame) != param['dim']:
        print("Size of frame_spec not consistant with dimension ")
        raise IrrelevantForDim
    vp_fct = spectrum(param)
    nb = len(vp_fct([0]*len(frame)))
    shape = tuple([len(f) for f in frame])
    eig = np.zeros( shape + (nb,))*1j

    tpl = (0,)*len(frame)
    while not tpl is None :
        eig[tpl] = vp_fct([frame[k][tpl[k]] for k in range(len(frame))])
        tpl = _next(tpl, shape)

    if bool_plot:
        _plot_spec(eig, frame, z_min, z_max, typ, talk = talk)
        if bool_show :
            plt.show()

    return eig


def mk_spec(stamp, frame = None, z_min = None, z_max = None, bool_plot = True, bool_show = True, talk = True, force = True, typ = 'real'):
    '''constructs spectrum from previously saved data (prefixed by stamp).
    There needs to be a param saved.

    if force : will re-compute the whole thing and save
    if data is missing, will re-compute and save'''

    #signature is as is so that the function can be used in general loops

    param = np.load(stamp+"_param.npy",  allow_pickle=True).item()
    if frame is None or len(frame) != param['dim'] or np.sum([f is None for f in frame]) >0 :
        frame = []
        for k in range(param['dim']):
            frame.append(np.linspace(-10, 10, 20))

    if len(frame) != param['dim']:
        print("Size of frame_spec not consostant with dimension ")
        raise IrrelevantForDim

    if len([filename for filename in os.listdir('.') if filename == stamp+'_eig.npy']) >0 and not force :
        eig = np.load(stamp+"_eig.npy")
        saved_frame = np.load(stamp+"_eig_frame.npy", allow_pickle = True)

        if len(frame) == len(saved_frame) and \
            np.prod([len(frame[k]) == len(saved_frame[k]) and np.min(np.abs(frame[k]-saved_frame[k])) < 1e-10  for k in range(len(frame))]) == 1 :

            if talk :
                print("re-use spec")
            frame = saved_frame

    else :
        if talk :
            print("re-create spec")
        eig = _mk_spec_from_param(param, frame, z_min, z_max, bool_plot=False, typ=typ)
        np.save(stamp+"_eig_frame", frame)
        np.save(stamp+"_eig", eig)

    if bool_plot:
        _plot_spec(eig, frame = frame, z_min = z_min ,z_max = z_max, typ = typ, talk = talk)
        if bool_show :
            plt.show()

    return frame, eig


## Simulation
def simu(param, bool_save = False, force = True, talk = True, freq_mess = .1):

    #if not force : will re-use data if the stamp has already been used to compute,


    if not 'stamp' in param.keys():
        stamp = mk_stamp(param['dim'])
        param['stamp'] = stamp
    stamp = param['stamp']

    if len([filename for filename in os.listdir('.') if filename == stamp+'_data.npy']) >0 and not force :
        data = np.load(stamp+"_data.npy", allow_pickle=True)
        state = data[-1]

    elif not bool_save and len([filename for filename in os.listdir('.') if filename == stamp+'_final.npy']) >0 and not force:
        state = np.load(stamp+"_final.npy", allow_pickle=True)

    else :
        state = param['init']
        unitary = unitary_from_order(param, talk = talk)
        if bool_save:
            data = [state]
        deb = time.time()
        for i in range(param['Nt']):
            if talk :
                extra = "Current size (in space units) : "
                extra = extra+ 'x'.join([str(round(len(state)*param['dxs'][k], 1)) for k in range(param['dim'])])
                timer(i, param['Nt'], freq_mess, deb, extra_mess = extra )

            try :
                new = unitary(state)
                state=new
                if bool_save :
                    data.append(np.copy(state))
            except OverflowError :
                print("Broke at", i, 'th step, i.e. at', 100*np.round(i/param['Nt'], 1), "%... ")
                print("max_width is ", param['max_width'])
                print("Nt is ", param['Nt'], "\n\n")
                break

    if bool_save:
        data = np.array(data, dtype=object)
        np.save(stamp+"_data", data, allow_pickle=True)
        np.save(stamp+"_param", param, allow_pickle = True)
        np.save(stamp+"_final", state)
        if talk :
            print("Data list is ", len(data), " long (!)")
            if len(data) == param['Nt']:
                print("Went until the end !")
            else:
                print("Stopped short by ", param['Nt']- len(data), " steps !")

    return state, param


## 1.11 Save
def save(stamp, fps=30, keep=1/5, frame_eig = None, eig_typ = 'real'):
    param = np.load(stamp+"_param.npy", allow_pickle=True).item()
    param['stamp']=stamp
    data = np.load(stamp+"_data.npy", allow_pickle=True)
    if not [filename for filename in os.listdir('.') if filename == stamp] :
        os.mkdir(stamp)

    np.save(stamp+"\\"+stamp+"_data", data, allow_pickle=True)
    np.save(stamp+"\\"+stamp+"_final", data[-1])
    np.save(stamp+"\\"+stamp+"_param", param, allow_pickle=True)


    plt.ioff()
    plt.close('all')



    frame_eig, eig = mk_spec(stamp, frame = frame_eig, bool_plot = False, typ = eig_typ)
    np.save(stamp+"\\"+stamp+"_eig_frame", frame_eig)
    np.save(stamp+"\\"+stamp+"_eig", eig)
    try:
        _plot_spec(eig, frame_eig, typ = eig_typ)
        plt.savefig(stamp+"\\"+stamp+'_final.pdf')
        plt.close()
    except IrrelevantForDim:
        None


    try:
        plot_state(data[-1], param['dxs'])
        plt.savefig(stamp+"\\"+stamp+'_final.pdf')
        plt.close()
    except IrrelevantForDim:
        None


    try:
        mat, X, Y = mk_spread_map(stamp, bool_plot = False)
        np.save(stamp+"\\"+stamp+"_spread_mat", mat)
        np.save(stamp+"\\"+stamp+"_spread_mat_X", X)
        np.save(stamp+"\\"+stamp+"_spread_mat_Y", Y)
        _plot_spread_map(mat, X, Y)
        plt.savefig(stamp+"\\"+stamp+'_spread_map.pdf')
        plt.close()
    except IrrelevantForDim:
        None

    try:
        mk_gif(stamp, fps=fps, keep=keep, path=stamp+"\\")
        plt.close()
    except IrrelevantForDim:
        None


    absc, ordo = repr_std(stamp, bool_plot = False)
    np.save(stamp+"\\"+stamp+"_sigma_ord", ordo)
    np.save(stamp+"\\"+stamp+"_sigma_absc", absc)
    _plot_std_from_std(absc, ordo)
    plt.savefig(stamp+"\\"+stamp+'_std.pdf')
    plt.close()

    plt.ion()

    print("\nStamp  :", stamp, " ---> Done !")
