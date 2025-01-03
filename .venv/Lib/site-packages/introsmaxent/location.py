# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

from introsmaxent.solver import MESolverOutput
from enum import Enum
import numpy


class DistType(Enum):
    OBSERVED = 0,
    PREDICTED = 1,
    BACKGROUND = 2


class Location:
    """
    This class represents a single location. For the moment, no projections or
    geospatial transformations will be assumed. In terms of the specific
    projection, we require WGS84.
    """
    def __init__(self, lat, lon, fdata=numpy.array([])):
        self.__lat = lat
        self.__lon = lon
        self.__presence = []
        # Numerical data for MaxEnt. In order to be efficient, we will allow
        # these objects to be accessible. Better design is needed here.
        self.z = fdata
        self.lbd = numpy.array([])
        self.p_obs = numpy.array([])
        self.p_prd = numpy.array([])
        self.q_bgd = numpy.array([])
        self.ror = 0.0
        self.cumul = 0.0
        self.logistic = 0.0
        self.cloglog = 0.0

    @property
    def location(self):
        return self.__lat, self.__lon

    @property
    def featurecount(self):
        return self.z.size

    @property
    def obscount(self):
        return len(self.__presence)

    def torecord(self, output):
        result = [self.__lat, self.__lon, self, {
            MESolverOutput.RAW: self.ror,
            MESolverOutput.CUMULATIVE: self.cumul,
            MESolverOutput.LOGISTIC: self.logistic,
            MESolverOutput.CLOGLOG: self.cloglog
        }[output]]

        return tuple(result)

    def getdist(self, dtype, feature):
        if 0 <= feature < len(self.z):
            return {
                DistType.OBSERVED: self.p_obs,
                DistType.PREDICTED: self.p_prd,
                DistType.BACKGROUND: self.q_bgd
            }[dtype][feature]
        else:
            return None

    def addporecord(self, po):
        self.__presence.append(po)

    def sendtokml(self, kmlobj):
        pass

    def diff(self, otherloc):
        # This makes sure
        assert isinstance(otherloc, Location)

        # Do this only with the same coordinates
        if self.location == otherloc.location:
            newloc = Location(self.__lat, self.__lon, self.z)
            newloc.lbd = numpy.subtract(self.lbd, otherloc.lbd)
            newloc.p_obs = numpy.subtract(self.p_obs, otherloc.p_obs)
            newloc.p_prd = numpy.subtract(self.p_prd, otherloc.p_prd)
            newloc.q_bgd = numpy.subtract(self.q_bgd, otherloc.q_bgd)
            newloc.ror = self.ror - otherloc.ror
            newloc.cumul = self.cumul - otherloc.cumul
            newloc.logistic = self.logistic - otherloc.logistic
            newloc.cloglog = self.cloglog - self.cloglog
        else:
            return None
