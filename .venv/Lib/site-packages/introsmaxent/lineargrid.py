# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import numpy
import introsmaxent.solver as sv
from introsmaxent.podata import POData
from introsmaxent.location import Location


class MaxEntGrid:
    """
    We wish to work ideally with a rectangular grid.
    """
    def __init__(self, llc=None, urc=None, gd=0):
        self.__locations = []
        self.__features = []
        self.__observations = []
        self.__lowerleftcorner = llc
        self.__upperrightcorner = urc
        self.__griddistance = gd
        self.__loaded = False
        self.__solved = False
        self.__solver = sv.MaxEntSolver()

    # Test loaded state
    @property
    def isloaded(self):
        return self.__loaded

    # Test solved state
    @property
    def issolved(self):
        return self.__loaded

    # Set new locations
    def setlocations(self, locs):
        self.__locations = locs

    # Set new features
    def setfeatures(self, fts):
        self.__features = fts

    # Set observations
    def setobservations(self, obs):
        self.__observations = obs

    # Count the number of locations in the grid
    @property
    def size(self):
        return len(self.__locations)

    @property
    def locations(self):
        return self.__locations

    @property
    def llbounds(self):
        return self.__lowerleftcorner

    @property
    def urboudns(self):
        return self.__upperrightcorner

    def setbounds(self, llc, urc):
        self.__lowerleftcorner = llc
        self.__upperrightcorner = urc

    # Consume data and build the linear grid
    def consume(self):
        if len(self.__observations) == 0:
            self.__loaded = False
        else:
            if len(self.__locations) == 0:
                # In this case, we need to manufacture the locations from the upper left
                # and lower right corner.
                (lllat, lllon) = self.__lowerleftcorner
                (urlat, urlon) = self.__upperrightcorner

                # We assume that the grid is uniform in two directions
                latlist = numpy.arange(lllat, urlat, self.__griddistance).tolist()
                lonlist = numpy.arange(lllon, urlon, self.__griddistance).tolist()

                for lat in latlist:
                    for lon in lonlist:
                        self.__locations.append(Location(lat, lon, []))

            # When ready, put the data in the right bins.
            for pod in self.__observations:
                self.__searchandplace(pod)

            self.__loaded = True

    # For one observation, find the closes point and assign itself to it
    # Future work: use quadtrees and improve the algoritm
    def __searchandplace(self, tpodata):
        assert(isinstance(tpodata, POData)), 'Datum must be of presence-only type.'
        distances = [tpodata.distancefrom(x) for x in self.__locations]
        index = distances.index(min(distances))
        self.__locations[index].addporecord(tpodata)

    # Setup probability parameters of the model
    def setprobparams(self, beta, tau):
        self.__solver.setprobparams(beta, tau)

    # Setup execution parameters of the model
    def setexecparams(self, iters, cnvg):
        self.__solver.setexecparams(iters, cnvg)

    # Setup output parameters of the model
    def setotptparams(self, depth, output):
        self.__solver.setotptparams(depth, output)

    # Solve the MaxEnt model
    def solvemaxent(self):
        if not self.__loaded:
            return False
        else:
            self.__solver.solvegrid(self.__locations)
            self.__solved = True

    # Perform a diff with other grid computed from another model
    def diff(self, otherlgrid):
        assert isinstance(otherlgrid, MaxEntGrid)

        if self.size == otherlgrid.size:
            result = []

            for (a, b) in zip(self.locations, otherlgrid.locations):
                result.append(a.diff(b))

            return result
        else:
            return None

    # Represent the output set as a list of records
    def torecords(self, output):
        if self.issolved:
            return [loc.torecord(output) for loc in self.__locations]
        else:
            return None
