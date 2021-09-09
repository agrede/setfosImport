"""
setfos mobility models

Copyright (C) 2020 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of setfosImport
"""

import numpy as np
import scipy.constants as PC


def egdm_lowfield(T, mu0, sigma, c2):
    sighat = sigma*PC.e/(PC.k*T)
    return mu0*np.exp(-c2*np.power(sighat, 2))


def mu_egdm(F, p, T, mu0, sigma, c2, N0):
    sh = sigma*PC.e/(PC.k*T)
    sh2 = np.power(sh, 2)
    muT = mu0*np.exp(-c2*sh2)
    poN0 = p/N0
    poN0[poN0 > 0.1] = 0.1
    dlta = 2.*(np.log(sh2-sh)-np.log(np.log(4.0)))/sh2
    g1 = np.exp(0.5*(sh2-sh)*np.power(2.*poN0, dlta))
    a = np.power(N0, -1./3.)
    f = F*a/(sigma)
    f[f > 2.0] = 2.0
    sh32 = np.power(sh, 3./2.)
    g2 = np.exp(0.44*(sh32-2.2)*(np.sqrt(1.0+0.8*np.power(f, 2))-1.0))
    return muT*g1*g2
