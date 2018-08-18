import matplotlib.pyplot as plt
import numpy as np


def get(signal_chunks, showtemplate):
    n_signal_chunks = len(signal_chunks)
    signal_average = np.mean(signal_chunks, axis=0)
    std_quarter = np.mean(np.std(signal_chunks[0:int(len(signal_chunks) / 4)],
                                 axis=0), axis=0)
    # upper_limit = signal_average + 5 * np.std(signal_chunks, axis=0)

    test = signal_average.copy()
    test[0:(len(test)/2)] = abs(test[0:(len(test)/2)])

    upper_limit = abs(test) + (5 * std_quarter)
    # lower_limit = signal_average - 5 * np.std(signal_chunks, axis = 0)
    lower_limit = (np.mean(signal_average, axis=0) *
                   np.ones(len(signal_average)) - (7 * std_quarter))

    tmp = upper_limit
    new = []
    for k in xrange(0, len(tmp)):
        new.append(upper_limit[k])
        new.append(tmp[k])
        new.append(upper_limit[k])
        new.append(tmp[k])
    start_idx = int(len(new) / (4 / 1.5)) + int(4 * 1.5)
    upper_limit = new[start_idx:start_idx + len(tmp)]

    if(showtemplate):
        plt.plot(signal_average)
        plt.hold(True)
        plt.plot(upper_limit, 'r--')
        plt.plot(lower_limit, 'r--')
        plt.show()

    intemp = []
    for i in xrange(0, n_signal_chunks):
        curr_trial = signal_chunks[i]
        gthan = (curr_trial > lower_limit)
        sthan = (curr_trial < upper_limit)
        total = list(gthan) and list(sthan)
        tmp = True
        for k in total:
            tmp = tmp and k
        intemp.append(tmp)
    intemp = np.array(intemp)
    idcs = np.where(intemp is True)
    print ('%.0f signals have been reduced to %.0f.\n'
           % (len(signal_chunks), len(signal_chunks[idcs])))
    return idcs
