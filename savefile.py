#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Save current file according to path and cluster number convention.

.. module:: savefile.py
.. moduleauthor:: Bonne Habekost <b.habekost1@ncl.ac.uk>
.. modulemodified:: July 27, 2014
"""

import Tkinter as Tk
import os
import hdf5pickle


class Dialog:
    def __init__(self, master, currfn, clusnum, savedir, data):
        self.frame = Tk.Frame(master)
        self.frame.grid(row=0, column=0)
        self.savedir = savedir
        self.data = data
        self.returnval = 'no'
        self.label1 = Tk.Label(self.frame, text='Current Filename: ')
        self.label2 = Tk.Label(self.frame, text='Cluster Number: ')
        self.label3 = Tk.Label(self.frame, text='Save file? ')
        self.label4 = Tk.Label(self.frame, fg="dark green")
        self.entry1 = Tk.Entry(self.frame, bd=5)
        self.entry2 = Tk.Entry(self.frame, bd=5)
        self.entry1.insert(0, currfn)
        self.entry2.insert(0, clusnum)
        self.button_no = Tk.Button(self.frame, text="SAVE & QUIT", fg="red",
                                   command=self.save_and_quit)
        self.button_yes = Tk.Button(self.frame, text="SAVE & REDO",
                                    command=self.save_and_redo)
        self.label1.grid(row=0, column=0)
        self.entry1.grid(row=0, column=1)
        self.label2.grid(row=1, column=0)
        self.entry2.grid(row=1, column=1)
        self.label3.grid(row=2, column=0)
        self.button_yes.grid(row=2, column=1)
        self.button_no.grid(row=2, column=2)
        self.label4.grid(row=3, column=0)

    def quit(self):
        self.root.destroy()

    def save_and_redo(self):
        newfn = self.entry1.get()
        newclus = self.entry2.get()
        savefn = (self.savedir + os.sep + newfn + '_' +
                  'cluster_' + newclus + '.h5')
        hdf5pickle.dump(self.data, savefn, '/data')
        fid = open('curr_clus.dat', 'w')
        fid.write('%s' % newclus)
        fid.close()
        self.redo = True
        self.frame.quit()
        self.frame.destroy()
        del self

    def save_and_quit(self):
        newfn = self.entry1.get()
        newclus = self.entry2.get()
        savefn = (self.savedir + os.sep + newfn + '_' +
                  'cluster_' + newclus + '.h5')
        hdf5pickle.dump(self.data, savefn, '/data')

        fid = open('curr_clus.dat', 'w')
        fid.write('%s' % newclus)
        fid.close()

        self.redo = False
        self.frame.quit()
        self.frame.destroy()
        del self
