#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Principal component function.

.. module:: princomp.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

from numpy import mean, cov, dot, linalg
import numpy as np
from sklearn.decomposition import FastICA


def get(A, opt=None):
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - mean(A.T)).T  # subtract the mean (along columns)
    [latent, coeff] = linalg.eig(cov(M))
    score = dot(coeff.T, M)  # projection of the data in the new space

    if(opt is None):
        pass
    else:
        rng = np.random.RandomState(42)
        ica = FastICA(random_state=rng)
        score = ica.fit(A).transform(A)
        score /= score.std(axis=0)
        score = score.T

    return coeff, score, latent
