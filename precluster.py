#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precluster a signal. Consists of determining the threshold, getting spike
times and spike waveforms, determining principal components.

.. module:: precluster.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

import numpy as np
import princomp
import thresh  # @UnresolvedImport
import spktimes  # @UnresolvedImport
import get_stimtrigavs  # @UnresolvedImport
import spkwaveform  # @UnresolvedImport
import filter_spikes  # @UnresolvedImport


def get(signal, Fs, dt_swf):
    signal_copy = signal
    signal = filter_spikes.high_pass(signal, Fs, 1000, 3)  # HP filter signal
    signal = filter_spikes.low_pass(signal, Fs,  Fs/float(2),
                                    1)  # LP filter signal, cf. Nyquist
    thres = thresh.detect(signal,
                          'manual')  # determine threshold automatically
    isspike = spktimes.detect(signal,
                              thres)  # get all spikes based on threshold
    idcs = np.where(isspike is True)  # get indices
    spiketimes = idcs[0] / float(Fs)  # get the actual spiketimes
    swfs = get_stimtrigavs.get_avs(signal, Fs, spiketimes, dt_swf,
                                   'zeros')  # get spike waveforms
    # heights = swfs.max(1)  # Spike amplitude
    # swfs = spkwaveform.upsample(swfs,
    #                             10), upsampling for alignment does not help
    spiketimes, swfs = spkwaveform.align(spiketimes, swfs, dt_swf,
                                         Fs)  # align SWFs to maximum

    spiketimes, swfs = spkwaveform.window_detect(spiketimes, swfs,
                                                 [0.0005, 0.001], Fs * 10,
                                                 'manual')  # Detect spk window
    PCs = dict()
    (PCs['coeff'], PCs['score'],
     PCs['latent']) = princomp.get(swfs)  # Calculate PCs
    swfs = get_stimtrigavs.get_avs(signal_copy, Fs, spiketimes, dt_swf,
                                   'zeros')  # get spike waveforms
    heights = swfs.max(1)  # Spike amplitude
    spiketimes, swfs = spkwaveform.align(spiketimes, swfs, dt_swf,
                                         Fs)  # align SWFs to maximum

    return thres, spiketimes, heights, swfs, PCs
