import numpy as np
import glob
import matplotlib.pyplot as plt

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

    def plot(self, curves=[None], labels=[None], clear=False, **kwrds):
        if self.debug:
            if clear:
                self.ax.cla()
            for c, l in zip(curves, labels):
                try:
                    self.ax.plot(c[0], c[1], label=l, **kwrds)
                    self.make_legend()
                except:
                    print(f"plotdebug could not plot curve with label: {l}, c: {c}")

        return


    def scatter(self, curves=[None], labels=[None], clear=False, **kwrds):
        if self.debug:
            if clear:
                self.ax.cla()
            for c, l in zip(curves, labels):
                try:
                    self.ax.scatter(c[0], c[1], label=l, **kwrds)
                    self.make_legend()
                except:
                    print(f"plotdebug could not scatter curve with label: {l}")

        return

    def show_plot(self):
        self.fig.show()

    def clear_plot(self):
        self.ax.cla()


    def make_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())