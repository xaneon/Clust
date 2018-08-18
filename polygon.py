#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try to catch version depending conflicts caused by an older nxutils library.

.. module:: polygon.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

try:
    from matplotlib import nxutils
except ImportError:
    import nxutils
import numpy as np


def pts_in_polygon(xpoints, ypoints, polygon):
    isin = list()
    for i in xrange(0, len(xpoints)):
        isin.append(nxutils.pnpoly(xpoints[i], ypoints[i], polygon))

    ispol = np.array(isin) == 1
    return ispol == 1
