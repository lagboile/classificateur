# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

from enum import Enum, IntEnum


class MESolverDepth(IntEnum):
    Linear = 0,
    Quadratic = 1


class MESolverOutput(IntEnum):
    RAW = 0,
    CUMULATIVE = 1,
    LOGISTIC = 2,
    CLOGLOG = 3


class MaxEntSolver:

    def __init__(self):
        self.__beta = 1.0e-4
        self.__tau = 0.5
        self.__iters = 1000
        self.__cnvg = 1.0e-5
        self.__depth = MESolverDepth.Linear
        self.__output = MESolverOutput.RAW

    def setprobparams(self, beta, tau):
        self.__beta = beta
        self.__tau = tau

    def setexecparams(self, iters, cnvg):
        self.__iters = iters
        self.__cnvg = cnvg

    def setotptparams(self, depth, output):
        self.__depth = depth
        self.__output = output

    def solvegrid(self, listgrid):
        pass

    @property
    def beta(self):
        return self.__beta

    @property
    def tau(self):
        return self.__tau

    @property
    def iters(self):
        return self.__iters

    @property
    def cnvg(self):
        return self.__cnvg