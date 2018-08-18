#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise cluster program and GUI.

.. module:: visualize.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

import matplotlib.pyplot as plt
import Tkinter
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.colors as colors
import tkMessageBox
import numpy as np
import polygon
import setpath
import os
F = setpath.setpath('y')
import simple_psth  # @UnresolvedImport

global clist, chosen, showdensity, nbins_density, fac_scalpha

#  ######################### USER SERVIABLE PARAMETER ######################  #
clist = 'k', 'b', 'c', 'm', 'y'  # Colour list for visualisation
showdensity = True  # Show density for choosing principal components
showdensity2 = False  # density option just for summary plot
nbins_density = 300  # number of bins for first density plot
nbins_density2 = 100  # and for second one
fac_scalpha = 1000  # Factor for alpha overlay-scatter in sum plot, prev: 200
show_nspikes = 100  # how many spikes raw data should be shown?
Rs = np.linspace(0.5, 0, 11)  # general colour settings / vectors
Gs = np.linspace(0.8, 0, 11)
Bs = np.linspace(0.5, 0, 11)
greens = [(np.linspace(0.5, 0, 11)[i], np.linspace(0.8, 0, 11)[i],
           np.linspace(0.5, 0, 11)[i]) for i in xrange(0, len(Rs))]
blacks = [(np.linspace(0.4, 0, 11)[i], np.linspace(0.4, 0, 11)[i],
           np.linspace(0.4, 0, 11)[i]) for i in xrange(0, len(Rs))]
blues = [(np.linspace(0.2, 0, 11)[i], np.linspace(0.2, 0, 11)[i],
          np.linspace(0.9, 0, 11)[i]) for i in xrange(0, len(Rs))]
#  ############# NO USER SERVIABLE PARAMETER BEYOND THIS POINT #############  #
cmap = colors.ListedColormap(greens)  # default colourmap for PCs
cmap_black = colors.ListedColormap(blacks)
cmap_blue = colors.ListedColormap(blues)
cmap_list = [cmap_black, cmap_blue]


def pdense(x, y, sigma, M=1000):  # plot probability density of y with known SD
    N = len(x)
    ymin, ymax = min(y - 2 * sigma), max(y + 2 * sigma)
    yy = np.linspace(ymin, ymax, M)
    a = [np.exp(-((Y - yy) / s) ** 2) / s for Y, s in zip(y, sigma)]
    A = np.array(a)
    A = A.reshape(N, M)
    return A, ymin, ymax


def sayyes():  # If yes clicked in dialog
    figure_window.quit()
    global currans
    currans = True


def sayno():
    figure_window.quit()
    global currans
    currans = False


def getAnswers(spikes, questionlist):
    num_clus = len(spikes['waveforms'])
    fig42 = showSummary(spikes)

    global figure_window

    # CHANGED: Choose multiple cluster
    textstr = 'Do you want to choose the cluster for this recording again?'
    figure_window = Tkinter.Tk()
    figure_window.title('Cluster Summary')
    question_label = Tkinter.Label(figure_window, text=textstr,
                                   font=("Helvetica", 20))
    question_label.grid(row=1, column=1, padx=2, pady=2, columnspan=30,
                        rowspan=1)
    canvas = tkagg.FigureCanvasTkAgg(fig42, master=figure_window)
    canvas.get_tk_widget().grid(row=2, column=1, rowspan=4)
    fig_btn = Tkinter.Button(figure_window, text='YES', command=sayyes)
    fig_btn2 = Tkinter.Button(figure_window, text='NO', command=sayno)
    fig_btn.grid(row=1, column=2, sticky=Tkinter.W, pady=2, padx=5)
    fig_btn2.grid(row=1, column=3, sticky=Tkinter.W, pady=2, padx=5)
    figure_window.mainloop()

    if(currans):
        figure_window.destroy()
        plt.close()
        spikes['redo'] = True
        return spikes

    else:
        figure_window.destroy()

        spikes['info'] = dict()
        for j in xrange(0, len(questionlist)):
            spikes['info'][questionlist[j]] = list()
            for i in xrange(0, num_clus):
                textstr = ('Is Clu' + str(i) + ' ' + questionlist[j] +
                           ' (' + spikes['sessionid'] + ') ?')
                figure_window = Tkinter.Tk()
                figure_window.title('Cluster Summary')
                question_label = Tkinter.Label(figure_window, text=textstr, font=("Helvetica", 20))
                question_label.grid(row=1, column=1, padx=2, pady=2,
                                    columnspan=30, rowspan=1)
                canvas = tkagg.FigureCanvasTkAgg(fig42, master=figure_window)
                canvas.get_tk_widget().grid(row=2, column=1, rowspan=4)
                fig_btn = Tkinter.Button(figure_window, text=questionlist[j],
                                         command=sayyes)
                fig_btn2 = Tkinter.Button(figure_window, text='NOT ' +
                                          questionlist[j], command=sayno)
                fig_btn.grid(row=1, column=2, sticky=Tkinter.W, pady=2, padx=5)
                fig_btn2.grid(row=1, column=3, sticky=Tkinter.W,
                              pady=2, padx=5)
                figure_window.mainloop()
                spikes['info'][questionlist[j]].append(currans)
                figure_window.destroy()
        plt.close()
        return spikes


def on_click(event):  # Click on sub-plot to choose principal component space
    global chosen
    if(event.inaxes == ax1):
        chosen = 1
        plt.close()
    elif(event.inaxes == ax2):
        chosen = 2
        plt.close()
    elif(event.inaxes == ax3):
        chosen = 3
        plt.close()
    elif(event.inaxes == ax4):
        chosen = 4
        plt.close()


def askCellTypes(num_clus):  # Meta information: Cell types, PTN and M1/S1
    qual, celltype, loc = list(), list(), list()
    for i in xrange(0, num_clus):
        isSU = tkMessageBox.askyesno('Cluster',
                                     'Is cluster number %i a single unit?' % i)
        isPTN = tkMessageBox.askyesno('Cluster',
                                      'Is cluster number %i a PTN?' % i)
        isM1 = tkMessageBox.askyesno('Cluster',
                                     'Is cluster number %i recorded in M1?'
                                     % i)

        if(isSU):
            qual.append('SU')
        else:
            qual.append('MU')

        if(isPTN):
            celltype.append('PTN')
        else:
            celltype.append('UID')

        if(isM1):
            loc.append('M1')
        else:
            loc.append('S1')

        return qual, celltype, loc


def drawPolygon():  # Draw Polygon and return points
    cnt, button, xpoints, ypoints = 0, False, list(), list()
    while(button is False):
        tmp = plt.ginput(n=1, show_clicks=True, mouse_stop=2)
        if(not tmp):
            pass
            button = True
            xvec = np.linspace(xpoints[0], xpoints[cnt - 1], 100)
            yvec = np.linspace(ypoints[0], ypoints[cnt - 1], 100)
            plt.plot(xvec, yvec, 'r--')
            plt.draw()
        else:
            xpoints.append(tmp[0][0])
            ypoints.append(tmp[0][1])
            plt.hold(True)
            plt.plot(xpoints[cnt], ypoints[cnt], 'k*')
            if(cnt > 0):
                xvec = np.linspace(xpoints[cnt - 1], xpoints[cnt], 100)
                yvec = np.linspace(ypoints[cnt - 1], ypoints[cnt], 100)
                plt.plot(xvec, yvec, 'r--')
                plt.draw()
            cnt += 1
    polygon_points = [[xpoints[i], ypoints[i]] for i in xrange(0,
                                                               len(xpoints))]
    return polygon_points


def showSummary(spikes, savefig=None):  # Summary plot for cluster chosen
    num_clus = len(spikes['waveforms'])  # How many cluster?
    fig = plt.figure(num=None, figsize=(16, 10))

    ax11 = plt.subplot2grid((4, 4), (1, 0), colspan=1, rowspan=1)

    ylim_max, ylim_min = 0, 0
    for i in xrange(0, num_clus):

        nspikes = len(spikes['waveforms'][i])
        if(nspikes > show_nspikes):
            ax11.plot(spikes['waveforms'][i][0:show_nspikes].T, clist[i])
        else:
            ax11.plot(spikes['waveforms'][i].T, clist[i])

        ax11.hold(True)

        tmp_max = max(np.mean(spikes['waveforms'][i], axis=0)) * 1.5
        tmp_min = min(np.mean(spikes['waveforms'][i], axis=0)) * 1.7
        if(ylim_min > tmp_min):
            ylim_min = tmp_min
        else:
            pass
        if(ylim_max < tmp_max):
            ylim_max = tmp_max
        else:
            pass

    plt.ylim(ylim_min, ylim_max)

    ax11.axes.get_xaxis().set_visible(False)
    ax11.axes.get_yaxis().set_visible(False)

    plt.title('Raw data (100 / ' + str(len(spikes['waveforms'][0])) + ')')

    ax11 = plt.subplot2grid((4, 4), (0, 0), colspan=1, rowspan=1)
    ylim = 0
    for i in xrange(0, num_clus):

        if(i == 0):  # only for 1st cluster show probability density of SWF
            curr_mean = np.mean(spikes['waveforms'][i], axis=0)
            curr_std = np.std(spikes['waveforms'][i], axis=0)
            xvec = np.linspace(0, len(curr_mean), len(curr_mean))
            A, ymin, ymax = pdense(xvec, curr_mean, curr_std, M=1000)
            ax11.imshow(-A.T, cmap='gray', aspect='auto',
                        origin='lower', extent=(min(xvec), max(xvec),
                                                ymin, ymax))
            ax11.hold(True)

    plt.ylim(ylim_min, ylim_max)

    ax11.axes.get_xaxis().set_visible(False)
    ax11.axes.get_yaxis().set_visible(False)

    plt.title('SWFs for ' + spikes['sessionid'])
    ax12 = plt.subplot2grid((4, 4), (0, 1), colspan=1, rowspan=1)
    for i in xrange(0, num_clus):
        n = len(spikes['waveforms'][i])
        ax12.plot(np.mean(spikes['waveforms'][i][0:int(n / 2)], axis=0),
                  clist[i], label='Clu' + str(i))
        plt.legend(loc=1, ncol=2, prop={'size': 10})

        curr_mean = np.mean(spikes['waveforms'][i][0:int(n / 2)], axis=0)
        curr_std = np.std(spikes['waveforms'][i][0:int(n / 2)], axis=0)
        xvec = np.linspace(0, len(curr_mean), len(curr_mean))
        ax12.fill_between(xvec, curr_mean + curr_std, curr_mean - curr_std,
                          facecolor=clist[i],
                          alpha=0.5)  # Fill area in between to show +/- STD

    plt.title(spikes['recid'] + ', ChNo: ' + spikes['chnum'])
    plt.ylim(ylim_min, ylim_max)
    ax12.axes.get_xaxis().set_visible(False)
    ax12.axes.get_yaxis().set_visible(False)
    ax13 = plt.subplot2grid((4, 4), (1, 1), colspan=1, rowspan=1)

    for i in xrange(0, num_clus):
        n = len(spikes['waveforms'][i])
        ax13.plot(np.mean(spikes['waveforms'][i][int(n / 2):-1], axis=0),
                  clist[i])

        curr_mean = np.mean(spikes['waveforms'][i][int(n / 2):-1], axis=0)
        curr_std = np.std(spikes['waveforms'][i][int(n / 2):-1], axis=0)
        xvec = np.linspace(0, len(curr_mean), len(curr_mean))
        ax13.fill_between(xvec, curr_mean + curr_std, curr_mean - curr_std,
                          facecolor=clist[i], alpha=0.5)

    plt.ylim(ylim_min, ylim_max)
    ax13.axes.get_xaxis().set_visible(False)
    ax13.axes.get_yaxis().set_visible(False)

    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    for i in xrange(0, num_clus):
        if(showdensity2):
            H, xedges, yedges = np.histogram2d(spikes['pc1'][i],
                                               spikes['pc2'][i],
                                               bins=nbins_density2)
            H = np.rot90(H)  # needs to be rotated and flipped
            H = np.flipud(H)
            Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with zero
            ax2.pcolormesh(xedges, yedges, Hmasked, cmap=cmap_list[i])

        else:
            if(i == 0):  # only the first time show all unselected PCs
                print 'len of pc2all: ', len(spikes['pc2all'])
                if(len(spikes['pc2all']) > 1500):
                    ax2.scatter(spikes['pc1all'], spikes['pc2all'],
                                alpha=((1 / float(len(spikes['pc2all']))) *
                                       fac_scalpha),
                                color = (0.65, 0.73, 0.73), marker = '.')
                else:
                    ax2.scatter(spikes['pc1all'], spikes['pc2all'],
                                color=(0.65, 0.73, 0.73), marker='.')
            else:
                pass
            if(len(spikes['pc2all']) > 1500):
                ax2.scatter(spikes['pc1'][i], spikes['pc2'][i],
                            alpha=((1 / float(len(spikes['pc2'][i]))) *
                                   fac_scalpha),
                            color = clist[i], marker = '.')
            else:
                ax2.scatter(spikes['pc1'][i], spikes['pc2'][i],
                            color=clist[i], marker='.')

        ax2.hold(True)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    plt.title('PC1 vs PC2')

    ax31 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=1)
    for i in xrange(0, num_clus):
        ax31.plot(spikes['spiketimes'][i],
                  spikes['heights'][i], clist[i] + '.')
        ax31.hold(True)
        plt.ylim(-max(spikes['heights'][i]), max(spikes['heights'][i]) * 5)
    plt.xlim(0, max(spikes['t_sec'][0]))
    ax31.axes.get_xaxis().set_visible(False)
    ax31.axes.get_yaxis().set_visible(False)
    plt.title('Stability, Spike Amplitude')

    ax32 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=1)
    for i in xrange(0, num_clus):
        t_sec, fr = simple_psth.simple_psth(spikes['spiketimes'][i],
                                            10)  # better binsize for vis
        ax32.plot(t_sec, fr, clist[i])
        ax32.hold(True)
        plt.xlim(0, max(spikes['t_sec'][0]))
        ax32.fill_between(t_sec, 0, fr, facecolor=clist[i], alpha=0.5)
    plt.xlabel('Time [sec]')
    plt.ylabel('FR [Hz]')

    ax41 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=1)

    num_tmp, perc_tmp = list(), list()
    for i in xrange(0, num_clus):
        ax41.plot(spikes['isis'][i], spikes['hist'][i], clist[i])
        ax41.fill_between(spikes['isis'][i], np.zeros(len(spikes['hist'][i])),
                          spikes['hist'][i],
                          spikes['hist'][i], facecolor=clist[i], alpha=0.5)
        ax41.hold(True)
        num_tmp.append('%.0f' % (spikes['hist'][i][0]))
        perc_tmp.append('%.2f' % ((spikes['hist'][i][0] /
                                   float(len(spikes['waveforms'][0]))) * 100))

    num_str = ', '.join(num_tmp) if(len(num_tmp) > 0) else num_tmp[0]
    perc_str = ', '.join(perc_tmp) if(len(perc_tmp) > 0) else perc_tmp[0]

    pre_str = 'ISIHs, Events < 1ms: '
    total_titlestr = pre_str + num_str + ' (' + perc_str + ' %)'

    plt.xlim(0, 0.1)
    ax41.axes.get_yaxis().set_visible(False)
    plt.title(total_titlestr)

    ax42 = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=1)
    for i in xrange(0, num_clus):
        print spikes['isis'][i][0], spikes['hist'][i][0]
        ax42.step(spikes['isis'][i], spikes['hist'][i], color=clist[i])
        ax42.hold(True)
        ax42.axvspan(spikes['isis'][i][0],
                     spikes['isis'][i][1], facecolor='r',
                     alpha=0.5)  # show the interval < 1ms

        ax42.hold(True)
    ax42.axes.get_yaxis().set_visible(False)
    plt.xlabel('Time [sec]')
    plt.xlim(0, 0.01)

    if(savefig is None):
        pass
    else:
        plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_' +
                    spikes['recid'] + '_' + spikes['chnum'] + '_' +
                    'cluster_summary' + '.pdf')
        plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_' +
                    spikes['recid'] + '_' + spikes['chnum'] + '_' +
                    'cluster_summary' + '.png')
        plt.savefig(F['fig_dir'] + os.sep + spikes['sessionid'] + '_' +
                    spikes['recid'] + '_' + spikes['chnum'] + '_' +
                    'cluster_summary' + '.eps')
    return fig


def showPCS(xvec, yvec, PCs):
    plt.close()
    cnt, isin = 0, list()
    fig_1 = plt.figure(num=None, figsize=(16, 10))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    plt.hold(True)

    if(showdensity):
        H, xedges, yedges = np.histogram2d(xvec, yvec, bins=nbins_density)
        H = np.rot90(H)  # needs to be rotated and flipped
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a val zero
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)

    else:
        ax1.plot(xvec, yvec, 'g.')

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.xlim(min(xvec) - abs(min(xvec)) * 0.2, max(xvec) +
             abs(max(xvec)) * 0.2)
    plt.ylim(min(yvec) - abs(min(yvec)) * 0.5, max(yvec) +
             abs(max(yvec)) * 0.2)  # CHANGED SCALING FACTOR
    polygon_points = drawPolygon()
    isin.append(polygon.pts_in_polygon(xvec, yvec, polygon_points))

    if(showdensity):
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
    else:
        ax1.plot(xvec, yvec, 'g.')

    plt.hold(True)
    plt.xlim(min(xvec) - abs(min(xvec)) * 0.2, max(xvec) +
             abs(max(xvec)) * 0.2)
    plt.ylim(min(yvec) - abs(min(yvec)) * 0.5, max(yvec) +
             abs(max(yvec)) * 0.2)  # CHANGED SCALING FACTOR

    plt.plot(xvec[isin[cnt]], yvec[isin[cnt]], clist[cnt] + '.')
    plt.draw()

    isclus = True
    while isclus:
        isclus = tkMessageBox.askyesno('Cluster',
                                       'Do you want another cluster?')
        if(isclus):
            polygon_points = drawPolygon()
            isin.append(polygon.pts_in_polygon(xvec, yvec, polygon_points))
            plt.hold(True)
            cnt += 1
            plt.plot(xvec[isin[cnt]], yvec[isin[cnt]], clist[cnt] + '.')
            plt.draw()
        elif(isclus is False):
            plt.close(fig_1)

    # NEW ADDED, 25/06/2014
    isin2, cnt2 = list(), 0
    fig_1 = plt.figure(num=None, figsize=(16, 10))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    for i in xrange(0, len(isin)):
        if(showdensity):
            H, xedges, yedges = np.histogram2d(PCs['score'][0][isin[i]],
                                               PCs['score'][1][isin[i]],
                                               bins=nbins_density)
            H = np.rot90(H)  # needs to be rotated and flipped
            H = np.flipud(H)
            Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels val zero
            ax1.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
        else:
            ax1.plot(PCs['score'][0][isin[i]], PCs['score'][1][isin[i]],
                     clist[i] + '.')

        plt.suptitle('Refine Clu' + str(i) + ':', fontsize=14)
        polygon_points = drawPolygon()
        isin2.append(polygon.pts_in_polygon(PCs['score'][0][isin[i]],
                                            PCs['score'][1][isin[i]],
                                            polygon_points))
        plt.draw()
        cnt2 += 1
        isin[i] = np.array(isin[i])
        idcs1 = np.where(isin[i] is True)
        idcs2 = np.where(isin2[i] is False)
        idcs_total = idcs1[0][idcs2[0]]
        isin[i][idcs_total] = False
        plt.hold(True)
        ax1.plot(PCs['score'][0][isin[i]], PCs['score'][1][isin[i]],
                 clist[i] + '.')
        plt.draw()
    plt.close(fig_1)
    return isin


def showPCs(PCs, heights):
    plt.close()
    # fig = plt.figure(num = None, figsize = (16, 10))
    global fig, ax1, ax2, ax3, ax4
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(16, 10))

    # fig.set_size_inches(50, 50)
    ax1 = plt.subplot(2, 2, 1)
    plt.suptitle('Please choose the PC-space for clustering:')

    if(showdensity):
        H, xedges, yedges = np.histogram2d(PCs['score'][0], PCs['score'][1],
                                           bins=nbins_density)
        H = np.rot90(H)  # needs to be rotated and flipped
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels val zero
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
    else:
        # plt.plot(PCs['score'][0], PCs['score'][1], 'g.')
        plt.scatter(PCs['score'][0], PCs['score'][1],
                    alpha=(1 / float(len(PCs['score'][1]))) * fac_scalpha,
                    color = 'g', marker = '.')

    plt.title('1: PC1 vs PC2')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2 = plt.subplot(2, 2, 2)
    if(showdensity):
        H, xedges, yedges = np.histogram2d(PCs['score'][1], PCs['score'][2],
                                           bins=nbins_density)
        H = np.rot90(H)  # needs to be rotated and flipped
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels val zero
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
    else:
        # plt.plot(PCs['score'][1], PCs['score'][2], 'g.')
        plt.scatter(PCs['score'][1], PCs['score'][2],
                    alpha=(1 / float(len(PCs['score'][2]))) * fac_scalpha,
                    color='g', marker = '.')

    plt.title('2: PC2 vs PC3')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3 = plt.subplot(2, 2, 3)

    if(showdensity):
        H, xedges, yedges = np.histogram2d(PCs['score'][2], PCs['score'][3],
                                           bins=nbins_density)
        H = np.rot90(H)  # needs to be rotated and flipped
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels val zero
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
    else:
        # plt.plot(PCs['score'][2], PCs['score'][3], 'g.')
        plt.scatter(PCs['score'][2], PCs['score'][3],
                    alpha=(1 / float(len(PCs['score'][3]))) * fac_scalpha,
                    color='g', marker='.')

    plt.title('3: PC3 vs PC4')
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    ax4 = plt.subplot(2, 2, 4)

    if(showdensity):
        H, xedges, yedges = np.histogram2d(PCs['score'][0], heights,
                                           bins=nbins_density)
        H = np.rot90(H)  # needs to be rotated and flipped
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels val zero
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
    else:
        # plt.plot(PCs['score'][0], heights, 'g.')
        # plt.scatter(PCs['score'][0], heights, s=70,
        #             alpha = 0.01, color = 'g')
        plt.scatter(PCs['score'][0], heights,
                    alpha=(1 / float(len(heights))) * fac_scalpha,
                    color='g', marker='.')

    plt.title('4: PC1 vs Height')
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)

    fig.canvas.mpl_connect('button_press_event', on_click)
    # fig.canvas.mpl_connect('figure_enter_event', on_click)
    plt.draw()
    plt.show()

    return chosen
