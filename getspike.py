#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Get Spike, main method for spike sorting routines.

.. module:: getspike.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

import numpy as np
import os
import Tkinter
import setpath
F = setpath.setpath('y')
import readsp  # @UnresolvedImport
import ISIH  # @UnresolvedImport
import visualize
import precluster
import simple_psth  # @UnresolvedImport
import template
import artefact  # @UnresolvedImport
from savefile import Dialog
import matplotlib.pyplot as plt

#  ######################### USER SERVIABLE PARAMETER ######################  #
Fs = 25000
# dt_swf = [0.00025, 0.00075]  # Spike Waveform Window
dt_swf = [0.0015, 0.0015]  # Spike Waveform Window
isi_vec = np.linspace(0, 1, 1000)
binsize = 0.001  # for PSTH
# savefigure = 'yes'  # save the summary figure
do_match = False  # Template matching
# do_artefact_removal = True  # Additional manual artefact removal
do_artefact_removal = False  # Additional manual artefact removal
showtemplate = False  # show the template
#  ############# NO USER SERVIABLE PARAMETER BEYOND THIS POINT #############  #


def SaveFile(E1, E2):
    print 'file saved'
    currfn = E1.get()
    print currfn


def cluster(spikes):

    monkname = filter(str.isalpha, spikes['sessionid']).lower()
    spikes['redo'] = False

    (spikes['clu_num'], spikes['spiketimes'], spikes['waveforms'],
     spikes['pc1'], spikes['pc2'], spikes['pc3'], spikes['heights'],
     spikes['isis'], spikes['hist'], spikes['center'], spikes['width'],
     spikes['t_sec'], spikes['frs']) = (list(), list(), list(), list(),
                                        list(), list(), list(), list(),
                                        list(), list(), list(),  list(),
                                        list())

    if(do_artefact_removal):
        sp = artefact.removeMarker(spikes['sessionid'], spikes['recid'],
                                   spikes['chnum'], Fs, 0.0025, [0.001, 0.005],
                                   F['data_' + monkname])
    else:
        sp = readsp.readsp(F['data_' + monkname] + os.sep + spikes['sessionid']
                           + os.sep + spikes['recid'] + os.sep
                           + spikes['recid']
                           + '-' + spikes['chnum'] + '.sp') # read data

    spikes['nsamples'] = len(sp)
    spikes['Fs'] = Fs
    spikes['T'] = spikes['nsamples'] / float(Fs)

    (thres, spktimes, heights, swfs,
     PCs) = precluster.get(sp, Fs, dt_swf)  # precluster

    which_pc = visualize.showPCs(PCs, heights)

    if(which_pc == 1):
        isclus = visualize.showPCS(PCs['score'][0], PCs['score'][1], PCs)
    elif(which_pc == 2): isclus = visualize.showPCS(PCs['score'][1], PCs['score'][2], PCs)
    elif(which_pc == 3): isclus = visualize.showPCS(PCs['score'][2], PCs['score'][3], PCs)
    elif(which_pc == 4): isclus = visualize.showPCS(PCs['score'][0], heights, PCs)

    spikes['pc1all'] = PCs['score'][0]
    spikes['pc2all'] = PCs['score'][1]
    spikes['pc3all'] = PCs['score'][2]

    num_clus = len(isclus)

    for i in xrange(0, num_clus):
        if(do_match):  # Fill area in between to show +/- STD
            idcs = template.get(swfs[isclus[i]], showtemplate)
        else:
            idcs = range(0, len(spktimes[isclus[i]]))
        spikes['clu_num'].append(i)
        spikes['spiketimes'].append(spktimes[isclus[i]][idcs])
        # swfs[isclus[i]][idcs] = spkwaveform.align(swfs[isclus[i]][idcs], dt_swf, Fs)
        spikes['waveforms'].append(swfs[isclus[i]][idcs])
        spikes['pc1'].append(PCs['score'][0][isclus[i]][idcs])
        spikes['pc2'].append(PCs['score'][1][isclus[i]][idcs])
        spikes['pc3'].append(PCs['score'][2][isclus[i]][idcs])
        spikes['heights'].append(heights[isclus[i]][idcs])
        isis, hist, center, width = ISIH.get(spikes['spiketimes'][i], isi_vec)
        spikes['isis'].append(isis)
        spikes['hist'].append(hist)
        spikes['center'].append(center)
        spikes['width'].append(width)
        t_sec, fr = simple_psth.simple_psth(spikes['spiketimes'][i], binsize)
        spikes['t_sec'].append(t_sec)
        spikes['frs'].append(fr)

        del isis, hist, center, width, t_sec, fr  # clear up temp vars

    figsum = visualize.showSummary(spikes)

    questionlist = 'stable', 'SU', 'PTN', 'M1'
    spikes = visualize.getAnswers(spikes, questionlist)

    if(spikes['redo']):  # redo the clustering
        spikes['redo'] = False
        cluster(spikes)
    else:
        pass

    savedir = F['hdf5']
    currfn = (spikes['sessionid'] + '_' + spikes['recid'] +
              '_' + spikes['chnum'])

    root = Tkinter.Tk()
    defaultclus = 0
    dia = Dialog(root, currfn, defaultclus, savedir, spikes)
    root.mainloop()
    root.destroy()
    del root

    fid = open('curr_clus.dat', 'r')
    clusno = int(fid.read())
    fid.close()

    plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_'
                + spikes['recid'] + '_' + spikes['chnum'] + '_' +
                'clus_' + str(clusno) + '_summary' + '.pdf')
    plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_' +
                spikes['recid'] + '_' + spikes['chnum'] + '_'
                + 'clus_' + str(clusno) + '_summary' + '.png')
    plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_' +
                spikes['recid'] + '_' + spikes['chnum'] + '_' +
                'clus_' + str(clusno) + '_summary' + '.eps')

    if(dia.redo):
        spikes['redo'] = False
        cluster(spikes)
        del dia
    else:
        del dia
        pass

    # fid = tables.openFile(savefn, mode = 'w'), hdf5pickle changed the structure:
    # Now it is not necessary to use tables library, hdf5pickle will open the file.
    # Changed on 28/08/2014
