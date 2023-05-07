import numpy as np

from math import *
import scipy
from scipy.linalg import logm

from utils import *
from utils import _mk_param_edit_fct, _next, _analyze_axis


## Basic quantum

sig0 = np.diag([1,1])+0j
id2 = sig0
sigx = np.array([[0,1], [1,0]])
sigy = np.array([[0, -1j],[1j,0]])
sigz = np.diag([1,-1])

def comm(a,b):
    return a.dot(b) - b.dot(a)
def anti_comm(a,b):
    return a.dot(b) + b.dot(a)

def rot(ax, angle):
    if ax==1:
        sig=sigx
    elif ax==2:
        sig=sigy
    elif ax==3:
        sig=sigz
    else:
        print("Axis \' " + str(ax)+"\' not supported, please put in 1->x  2->y or 3->z")
        raise ValueError

    return expm(-1.j/2. * angle * sig)

def normalize(mat):
    return mat/np.linalg.norm(mat)


def norm_last_axis(state):
    '''takes the norm along the last axis -> to translate arry of spins into arrray of proba'''
    res = np.zeros(state.shape[:-1])
    tpl = (0,)*len(res.shape)
    while not tpl is None :
        res[tpl] = np.sqrt(np.sum([np.abs(x)**2 for x in state[tpl]]))
        tpl = _next(tpl, res.shape)
    return res

def bloch(th, ph):
    if th > np.pi or th < 0 or ph > 2*np.pi or ph < 0 :
        print("Il faut avoir : 0≤𝜃≤𝜋 ; 0≤𝜙≤2𝜋")
        raise ValueError
    return normalize([np.cos(th/2), np.exp(1j * ph)*np.sin(th/2)])

def get_xyz(ths, phs):

    # create the sphere surface
    XX = 10 * np.outer( np.cos( phs ), np.sin( ths ) )
    YY = 10 * np.outer( np.sin( phs ), np.sin( ths ) )
    ZZ = 10 * np.outer( np.ones( np.size( phs ) ), np.cos( ths ) )

    return XX, YY, ZZ


## deal with states
def proba(state, dxs, frame=None, tol = 1e-12):
    res = norm_last_axis(state)**2

    if frame is None:
        frame = [None]*len(res.shape)

    goal_shape = np.array(res.shape)
    for k in range(len(frame)):
        if not frame[k] is None :
            goal_shape[k] = len(frame[k])

    goal_shape = tuple(goal_shape)
    res = resize_nD(res, goal_shape, tol = tol)

    if np.abs(np.sum(res) - 1) > 1e-5 :
        print("Erreur dans proba, ça ne somme que à ", np.sum(res))
    return res

def _i_ax(i, state, dxs):
    ''' gives the proper i axis to display the state'''
    return np.arange(-(state.shape[i]-1)/2, (state.shape[i]+1)/2)*dxs[i]

def frame(state, dxs):
    '''gives the whole set of coordinate axis to display the state'''
    return [_i_ax(i, state, dxs) for i in range(len(dxs))]


def display(state, dxs, frame=None, displ_typ = 'pb', fct = None, tol = 1e-12):
    '''returns the axis and the data as prompted, ready to be displayed'''
    if displ_typ == 'pb':
        Z = proba(state, dxs, frame, tol)

    elif displ_typ == 'up':
        dim = len(dxs)
        state_up = state[(slice(None),)*dim , 1, None]
        state_up = state_up/np.sum(norm_last_axis(state_up)**2 * np.prod(dxs))
        Z = proba(state_up, dxs, frame, tol)

    elif displ_typ == 'down':
        dim = len(dxs)
        state_down = state[(slice(None),)*dim , 0, None]
        state_down = state_down/np.sum(norm_last_axis(state_down)**2 * np.prod(dxs))
        Z = proba(state_down, dxs, frame, tol)

    elif displ_typ == 'spe':
        if not fct is None :
            state_spe = fct(state, dxs)
            Z = proba(state_down, dxs, frame, tol)

    else:
        print("displ_typ value not recognized ")
        raise ValueError


    if frame is None:
        frame = [None]*len(dxs)

    for k in range(len(frame)):
        if frame[k] is None:
            frame[k] = _i_ax(k, state, dxs)

    return frame, Z

def occupied(state):
    '''converts the percentage of occupied positions in the state'''
    pr = proba(state, dxs = [1]*(len(state)-1))
    return np.round(np.sum(pr>0)/np.sum(pr >=0), 2)*100

## Walks

def coin(delta, theta, phi, zeta):
    return np.exp(1j*delta) * rot(3, zeta).dot(rot(2, theta)).dot(rot(3, phi))

_hadamard_pm = (np.pi/2, np.pi/2, np.pi, 0)
def hadamard_coin():
    return coin(*_hadamard_pm)

def _spin_mat_to_op_nD(mat):
    def op(state):
        res = np.zeros_like(state)*1.j
        dim = len(res.shape)-1
        if dim == 1:
            for k in range(len(state)):
                res[k] = mat.dot(state[k])
        elif dim == 2:
            for k in range(len(state)):
                for kk in range(len(state[0])):
                    res[k,kk] = mat.dot(state[k,kk])
        else :
            for k in range(len(state)):
                res[k] = op(state[k])

        return res
    return op

def _operator_composition(ops):
    '''starts from the end of the list'''
    def op(state):
        for k in range(len(ops)-1, -1, -1):
            state = ops[k](state)
        return state
    return op

def coin_op_mk(delta, theta, phi, zeta):
    c = coin(delta, theta, phi, zeta)
    return _spin_mat_to_op_nD(c)

def shift_op_mk_nD(axis, overflow = None):
    # ??
    ax = axis
    ov = overflow
    def shift_op(state):
        overflow = ov
        axis = ax
        if not overflow is None and type(overflow)==float :
            overflow = [overflow]*(len(state.shape)-1)
        dim = len(overflow)
        if not overflow is None and np.max([state.shape[k] +2 > overflow[k] for k in range(dim)]) == True :
            print("Attention, taille max atteinte, fin des shifts !")
            raise OverflowError

        shape = np.array(state.shape[:dim])

        test_too_small = [ (
                        np.max([np.linalg.norm(sp) for sp in state[(slice(None),)*k + (0,)]])**2  >1e-15 or
                        np.max([np.linalg.norm(sp) for sp in state[(slice(None),)*k + (-1,)]])**2 >1e-15
                                 )
                        for k in range(dim)]

        for k in range(dim) :
            if test_too_small[k]:
                shape[k] +=2

        shape = tuple(shape)
        state = resize_nD(state, shape)
        res = np.zeros(shape + state.shape[dim:])*1.j

        axis = _analyze_axis(axis)

        if type(axis)==int and axis < dim :
            for k in range(state.shape[axis]):
                if k+1<state.shape[axis]:
                    res[(slice(None),)*axis + (k+1 ,) + (slice(None),)*(dim-axis-1) + (1,)] += \
                                state[(slice(None),)*axis + (k ,) + (slice(None),)*(dim-axis-1) + (1,)]
                if k-1>= 0:
                    res[(slice(None),)*axis + (k-1 ,) + (slice(None),)*(dim-axis-1) + (0,)] +=  \
                                state[(slice(None),)*axis + (k ,) + (slice(None),)*(dim-axis-1) + (0,)]
        else:
            print("axis variable not recognized !, it shouls be 'x' or 'y' or an int < dim ; it was :" + axis)
            raise ValueError

        return res

    return shift_op

def unitary_mk(ops):
    def unit_op(state):
        state = _operator_composition(ops)(state)
        return state
    return unit_op