import numpy as np
import glob


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
