#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Tue Nov 12 13:03:42 2024

@author: joe cooney

for the gagliardi group rotation
"""

import pyscf
import numpy
import scipy


# In[2]:


# define molecule
# same as RHF - will have 5 doubly occupied orbitals (5 up, 5 down electrons)

mol = pyscf.gto.Mole()

mol.atom= '''
O          0.00000     0.00000     0.11779
H          0.00000     0.75545    -0.47116
H          0.00000    -0.75545    -0.47116
'''

mo.spin=0
mol.charge=0
mol.basis='CC-PVDZ'
mol.build()


# In[16]:


# this not the same - need to include spin of orbitals and do separately
def get_hcore(mol):
    '''
    Parameters
    ----------
    mol : Mole object

    Returns
    -------
    hcore: t+v
    '''
    t = mol.intor_symmetric('int1e_kin') # this is the part that is different?
    v = mol.intor_symmetric('int1e_nuc')
    hcore = t + v
    return hcore
    


# In[17]:


# same as RHF
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


# In[7]:


# same but do it for both dm's 
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
    eri = get_eri(mole)
    J = np.einsum('pqrs, qp->rs', eri, dm)
    K = np.einsum('pqrs, rq->ps', eri, dm)
    return J - 0.5*K


# In[8]:


# same but do it for both dm's
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


# In[14]:


# use this for each one.
def construct_dm(mol, mo_coeff, up_or_dn):
    '''
    Parameters
    ----------
    mol : Mole object
    mo_coeff : molecular orbital coefficients
    
    Returns
    -------
    dm: density matrix
    '''
    
    nocc_up, nocc_down = mol.nelec()
    
    if up_or_dn == "up":
        dm_up = 2. * np.dot(mo_coeff[:, :nocc_up], mo_coeff[:, :nocc].T)
        return dm_up
    
    else:
        dm_down = 2. * np.dot(mo_coeff[:, :nocc_down], mo_coeff[:, :nocc].T)
        return dm_down


# In[10]:


# do for both dm's
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


# In[11]:


# do for both dm's
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


# In[2]:


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
    mo_coeff = np.zeros_like(h1)
    nocc_up, nocc_down = mol.nelec()

    mo_up_coeff = mo_coeff[:, :nocc_up]
    mo_down_coeff = mo_coeff[:, :nocc_down]
    
    dm_up = construct_dm(mol, mo_up_coeff, "up")
    dm_down = construct_dm(mol, mo_down_coeff, "down")
    
    # SCF procedure
    converge = False
    energy = 0
    for i in range(maxiter):
        # up electrons
        fock_up = construct_fock(mol, dm_up)
        mo_up_energy, mo_up_coeff = generalized_eigval(fock_up, s)
        new_dm_up = construct_dm(mol, mo_up_coeff, "up")

        # down electrons
        fock_down = construct_fock(mol, dm_down)
        mo_down_energy, mo_down_coeff = generalized_eigval(fock_down, s)
        new_dm_down = construct_dm(mol, mo_down_coeff, "down")

        # get energy
        new_energy = get_energy(mol, new_dm_down + new_dm_up)
        new_dm = new_dm_up + new_dm_down
        
        print('iteration', i, 'energy:', new_energy)
        if np.abs(energy-new_energy) < ethresh and np.linalg.norm(new_dm-dm) < dmthresh:
            print('converged')
            converge=True
            break
        
    dm_up = new_dm_up
    dm_down = new_dm_down
    dm = new_dm
    energy = new_energy
    
    if not converge:
        print("hasnt converged")
        
    return energy, mo_coeff


# In[ ]:


# run the code
# remain mostly the same as RHF - change the methods and need 2 density matrices
energy = scf_procedure(mol)[0]

# run pyscf code for reference energy
mf = pyscf.scf.UHF(mol)
refenergy = mf.kernel()

# test:
if np.allclose():
    print("it worked")

#use as initial guess:
h1 = get_hcore(mol)
mo_coeff = np.zeros_like(h1) #dimension (nbasis * nbasis_up)

mo_up_coeff = mo_coeff[:, :nocc_up]
mo_down_coeff = mo_coeff[:, :nocc_down]

dm_up = construct_dm(mol, mo_up_coeff, "up")
dm_down = construct_dm(mol, mo_down_coeff, "down")
dm = dm_up + dm_down

