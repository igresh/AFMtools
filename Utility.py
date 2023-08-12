import numpy as np
import glob
import matplotlib.pyplot as plt
import os

def consecutive(data, stepsize=1):
    "Return groups of consecutive elements"
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def parseDirectory(Directory):
    """
    Used to open force curves exported as text files from Asylum.

    Opens all the files in a directory, returns an iterable such that you can use something like

    for Xfilename, Yfilename in openDirectory('CleanSilica2/'):

    """

    Xfilenames = glob.glob(os.path.join(Directory, '*ZSnsr.txt'))
    # Yfilenames = glob.glob(os.path.join(Directory, '*DeflV.txt'))
    Yfilenames = []
    for xfn in Xfilenames:
        Yfilenames.append(xfn.replace('ZSnsr', 'DeflV'))
    # assert len(Xfilenames) == len(Yfilenames)

    return zip(Xfilenames, Yfilenames)


class plotdebug (object):
    def __init__(self, debug):
        self.debug = debug

        if self.debug:
            self.fig, self.ax = plt.subplots()
            self.axt = self.ax.twinx()

    def plot(self, curves=[None], labels=[None], clear=False, ax=1, **kwrds):

        if self.debug == True:
            if ax==1:
                ax = self.ax
            else:
                ax = self.axt

            if clear:
                ax.cla()
            for c, l in zip(curves, labels):
                try:
                    ax.plot(c[0], c[1], label=l, **kwrds)
                    self.make_legend()
                except:
                    print(f"plotdebug could not plot curve with label: {l}, len: {len(c[0])}")

        return


    def scatter(self, curves=[None], labels=[None], clear=False, ax=1, **kwrds):
    
        if self.debug == True:
            if ax==1:
                ax = self.ax
            else:
                ax = self.axt
    
            if clear:
                ax.cla()
            for c, l in zip(curves, labels):
                try:
                    ax.scatter(c[0], c[1], label=l, **kwrds)
                    self.make_legend()
                except:
                    print(f"plotdebug could not scatter curve with label: {l}")

        return

    def show_plot(self):
        self.fig.show()

    def clear_plot(self):
        self.ax.cla()


    def make_legend(self):
        handles1, labels1 = self.ax.get_legend_handles_labels()
        handles2, labels2 = self.axt.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())