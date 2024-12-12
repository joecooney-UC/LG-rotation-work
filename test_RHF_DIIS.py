"""
Created on Mon Nov 25 10:17:42 2024

@author: joe cooney

for the gagliardi group rotation
"""

import pyscf
import numpy as np
from numpy import dot
from scipy import linalg

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
    return J - 0.5 * K


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

    nocc = int(mol.nelectron // 2)  # dont think int does anything here.
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

    mo_energy, mo_coeff = linalg.eigh(fock, s)
    return mo_energy, mo_coeff


def scf_procedure(mol, ethresh=1e-7, dmthresh=1e-7, maxiter=5):
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
    # threshhold because some of these are zero.
    # this is not the issue - it matches test_HF.py.
    a = np.power(s, -0.5, where=s>1.e-16)
    # init guess
    h1 = get_hcore(mol)
    fock_p = a.dot(h1).dot(a)
    # mo_energy, mo_coeff_p = generalized_eigval(fock_p, s)
    mo_coeff_p = np.zeros_like(h1)
    mo_coeff = a.dot(mo_coeff_p)
    dm = construct_dm(mol, mo_coeff)

    # Trial and residual vector lists
    fock_List = []
    diis_Resid = []

    # SCF procedure
    converge = False
    energy = 0

    for i in range(maxiter):
        # build Fock Matrix
        fock = construct_fock(mol, dm)

        # build residuals
        diis_r = a.dot(fock.dot(dm).dot(s) - s.dot(dm).dot(fock)).dot(a)

        # append to lists
        fock_List.append(fock)
        diis_Resid.append(diis_r)

        # compute energy
        mo_energy, mo_coeff = generalized_eigval(fock, s)
        new_dm = construct_dm(mol, mo_coeff)
        new_energy = get_energy(mol, new_dm)
        dRMS = np.mean(diis_r ** 2) ** 0.5
        print('iteration', i, 'energy:', new_energy, 'dRMS:', dRMS)

        # check convergence
        if np.abs(energy - new_energy) < ethresh and dRMS < dmthresh:
            print('converged')
            converge = True
            break

        if i >= 1:
            # Build the B matrix
            B_dim = len(fock_List) + 1
            B = np.empty((B_dim, B_dim))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for j in range(len(fock_List)):
                for k in range(len(fock_List)):
                    B[j, k] = np.einsum('jk,jk->', diis_Resid[j], diis_Resid[k], optimize=True)

            # Build RHS of Pulay equation
            rhs = np.zeros((B_dim))
            rhs[-1] = -1

            # Solve Pulay equation for c_i's with NumPy
            coeff = np.linalg.solve(B, rhs)

            # Build DIIS Fock matrix
            fock = np.zeros_like(fock)
            for x in range(coeff.shape[0] - 1):
                fock += coeff[x] * fock_List[x]

            print(fock)

        # Compute new orbital guess with DIIS Fock matrix
        fock_p = a.dot(fock).dot(a)
        mo_energy, mo_coeff_p = generalized_eigval(fock_p, s)
        mo_coeff = a.dot(mo_coeff_p)
        new_dm = construct_dm(mol, mo_coeff)
        new_energy = get_energy(mol, new_dm)

        dm = new_dm
        energy = new_energy
        
    if not converge:
        print("hasnt converged")

    return energy, mo_coeff

#use as initial guess:
h1 = get_hcore(mol)
mo_coeff = np.zeros_like(h1) #dimension (nbasis * nbasis)
dm = construct_dm(mol, mo_coeff)

# DIIS initial guesses
# I think this is the same.

# run the code
energy = scf_procedure(mol)[0]

# run pyscf code for reference energy
mf = pyscf.scf.RHF(mol)
refenergy = mf.kernel()

# test:
if np.allclose():
    print("it worked")
