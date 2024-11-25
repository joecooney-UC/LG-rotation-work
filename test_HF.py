# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:03:42 2024

@author: joe cooney

for the gagliardi group rotation
"""

import pyscf
import numpy as np
import scipy

# define molecule

mol = pyscf.gto.Mole()

mol.atom= '''
O          0.00000     0.00000     0.11779
H          0.00000     0.75545    -0.47116
H          0.00000    -0.75545    -0.47116
'''

mol.spin=0
mol.charge=0
mol.basis='CC-PVDZ'
mol.build()

# run the code
energy = scf_procedure(mol)[0]

#use as initial guess:
h1 = get_hcore(mol)
mo_coeff = np.zeroes_like(h1) #dimension (nbasis * nbasis)
dm = construct_dm(mol, mo_coeff)

# run pyscf code for reference energy
mf = pyscf.scf.RHF(mol)
refenergy = mf.kernel()

# test:
if np.allclose():
    print("it worked")

def get_hcore(mol):
    '''
    Parameters
    ----------
    mol : Mole object

    Returns
    -------
    hcore: t+v
    '''
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    hcore = t + v
    return hcore


def get_eri(mol):
    '''
    Parameters
    ----------
    mol : Mole object

    Returns
    -------
    eri: electron repulsion integrals
    '''
    eri = mol.intor('int2e')
    return eri


def get_veff(mol, dm):
    '''
    Parameters
    ----------
    mol : Mole object
    dm : Density matrix

    Returns
    -------
    J-0.5*K
    '''
    eri = get_eri(mol)
    J = np.einsum('pqrs, qp->rs', eri, dm)
    K = np.einsum('pqrs, rq->ps', eri, dm)
    return J - 0.5*K


def construct_fock(mol, dm):
    '''
    Parameters
    ----------
    mol : Mole object
    dm : density matrix

    Returns
    -------
    fock: Fock matrix
    '''   
    
    hcore = get_hcore(mol)
    veff = get_veff(mol, dm)
    fock = hcore + veff
    return fock


def construct_dm(mol, mo_coeff):
    '''
    Parameters
    ----------
    mol : Mole object
    mo_coeff : molecular orbital coefficients
    
    Returns
    -------
    dm: density matrix
    '''
    
    nocc = int(mol.nelectron//2) #dont think int does anything here.
    dm = 2. * np.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
    return dm

    
def get_energy(mol, dm):
    '''
    Parameters
    ----------
    mol : Mole object
    dm : density matrix

    Returns
    -------
    energy: energy
    '''
    
    h1 = get_hcore(mol)
    veff = get_veff(mol, dm)
    energy = np.einsum('pq, qp->', h1, dm) \
            + 0.5 * np.einsum('pq,qp->', veff, dm) \
            + mol.energy_nuc()
            
    return energy


def generalized_eigval(fock, s):
    '''
    Parameters
    ----------
    fock : Fock matrix
    s : overlap matrix

    Returns
    -------
    mo_energy
    mo_coeff
    '''
    
    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s)
    return mo_energy, mo_coeff

def scf_procedure(mol, ethresh=1e-7, dmthresh=1e-7, maxiter=100):
    '''
    Parameters
    ----------
    mol :Mole object
    ethresh : float, optional
        DESCRIPTION. The default is 1e-7.
    maxiter : int, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    energy: energy
    mo_coeff: mol orb coefficients
    '''
    
    # get overlap matrix
    s = mol.intor_symmetric('int1e_ovlp')
    
    # init guess
    mo_coeff = np.zeroes_like(h1)
    dm = construct_dm(mol, mo_coeff)
    
    # SCF procedure
    converge = False
    energy = 0
    for i in range(maxiter):
        fock = construct_fock(mol, dm)
        mo_energy, mo_coeff = generalized_eigval(fock, s)
        new_dm = construct_dm(mol, mo_coeff)
        new_energy = get_energy(mol, new_dm)
        print('iteration', i, 'energy:', new_energy)
        if np.abs(energy-new_energy) < ethresh and np.linalg.norm(new_dm-dm) < dmthresh:
            print('converged')
            converge=True
            break
        
    dm = new_dm
    energy = new_energy
    if not converge:
        print("hasnt converged")
        
    return energy, mo_coeff

