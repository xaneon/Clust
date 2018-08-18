#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper class to acount for new polygon package.

.. module:: nxutils.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

from matplotlib.path import Path


def pnpoly(x, y, polygon):
    # Path(verts).contains_points()
    out = Path(polygon).contains_point((x, y))
    return out
