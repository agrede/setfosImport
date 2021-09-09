"""
setfos data importer

Copyright (C) 2020 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of setfosImport
"""

import numpy as np
import pandas as pd
import scipy.constants as PC
import re
import pint

RSWEEP = re.compile(r'#\sSweep\s(\d+):\s+((.+)\s+\(([^\)]+)\))')
RCOL = re.compile(r'(.+)\(([^/\n]*)/?(.*)\)')
NA_VALUES = np.array(["-nan(ind)", "NaN"])
ureg = pint.UnitRegistry()


def read_output(pth):
    """
    Read setfos output data
    Input
    -----
    path: path to datafile
    Output
    ------
    tuple containing
    dta: pandas dataframe
    cunits: units for columns
    sweeps: column name for sweep variables
    """
    sweeps = []
    with open(pth, 'r') as f:
        pline = ""
        for line in f:
            if (m := RSWEEP.match(line)):
                sweeps.append(m.group(3))
            elif (line[0] != "#" and len(line) > 1):
                break
            pline = line
    cnames = []
    cunits = {}
    for m in [RCOL.match(x) for x in (pline[2:-2]).split("\t")]:
        cnm = m.group(1)
        cnames.append(cnm)
        if len(m.group(2)) > 0:
            unt = m.group(2)
            if len(m.group(3)) > 0:
                unt += "/({})".format(m.group(3))
                unt = unt.replace("Vs", "V*s")
            cunits[cnm] = ureg.parse_expression(unt).u
        else:
            cunits[cnm] = ureg.parse_expression("").u
    if cnames[len(sweeps)] == "x":  # electric figures has uniform x spacing
        sweeps.insert(0, "x")
    dta = pd.read_table(
        pth, comment="#",
        na_values=NA_VALUES, header=None,
        index_col=False,
        names=cnames)
    N = dta.nunique()
    Nshp = tuple([N[x] for x in sweeps])
    return (dta, cunits, sweeps, Nshp)


def to_numpy(cname, dta, units, Nshp):
    """
    Quick change to reshaped numpy array for a column with units
    """
    return dta[cname].to_numpy().reshape(Nshp, order='F')*units[cname]
