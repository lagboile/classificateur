# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.
import introsmaxent.utils as utils
import numpy
import pandas
import introsmaxent.location as loct


class Background:
    """
    This class implements the minimal environment based on the pandas data
    processing library for CSV files. In the future, we expect to consume this
    from a DataONE Python object. Each CVS line, at the moment, is expected to
    contain only one record per location (i.e. no degeneracy).

    The expected data format is as follows (in terms of columns)

    LAT LON F1  F2  ... FK

    Where Fi are numerical features. Headers per column are **mandatory**. When
    more than one register exists per unique LAT,LON combination, only the
    first one is kept.

    We expect geographical features to be recorded for a polygon area with a
    regular spatial sampling between them. In particular, we also expect that
    the amount of locations exceeds the amount of presence only data.

    Decimation is an important feature. For a dense dataset, it allows remotion
    of n - 1 rows and columns per every n blocks. This is useful to play with
    the density of points in relation to computational efficiency.
    """

    def __init__(self):
        self.__document = ''
        self.__decimation = 0
        self.__featureNames = []
        self.__rawdata = numpy.array([])
        self.__reduced = numpy.array([])
        self.__loaded = False

    @property
    def isfileset(self):
        return self.__document != ''

    def setfile(self, csvfile):
        self.__document = csvfile

    @property
    def isloaded(self):
        return self.__loaded

    def setdecimation(self, dec):
        if dec < 0:
            print("Error: decimation must be a non negative number")
        elif 10000 > self.__rawdata / dec:  # If few points are available, do not sample
            self.__decimation = 0
        else:
            self.__decimation = dec

    def setfeaturenames(self, features):
        self.__featureNames = features

    def consume(self):
        if self.isfileset:
            df = pandas.read_csv(self.__document)
            df.drop_duplicates(['LAT', 'LON'], keep='first')
            df.sort_values(by=['LAT', 'LON'])
            self.__rawdata = df[self.__featureNames].values
            self.__reduced = self.__rawdata
            self.__loaded = True

    def consumedataone(self):
        raise Exception("Method not yet implemented.")

    # In the following two functions we use negative numbers to signal
    # absent data. The latter is ensured through
    def mindelta(self):
        if len(self.__featureNames) != 0:
            return self.computediff(10000.0, False)
        else:
            return -1.0

    def maxdelta(self):
        if self.isloaded:
            return self.computediff(0, True)
        else:
            return -1.0

    @property
    def minlat(self):
        return self.__reduced[0].min

    @property
    def maxlat(self):
        return self.__reduced[0].max

    @property
    def minlon(self):
        return self.__reduced[1].min

    @property
    def maxlon(self):
        return self.__reduced[1].max

    def computediff(self, limit, order):
        """
        order = true -> limit is for a sup operation
        order = false -> limit is for an inf operation
        """
        result = limit
        (rows, _) = self.__rawdata.shape

        for i in range(0, rows):
            for j in range(0, rows):
                if i != j:
                    distance = utils.eucldist((self.__rawdata[i][0], self.__rawdata[i][1]),
                                              (self.__rawdata[j][0], self.__rawdata[i][1]))
                    if order:
                        if distance > limit:
                            result = distance
                    else:
                        if distance < limit:
                            result = distance
        return result

    def decimate(self):
        if self.isloaded:
            rawtranspose = self.__rawdata.T
            uniquelats = numpy.unique(rawtranspose[0])
            uniquelons = numpy.unique(rawtranspose[1])
            coords = numpy.zeros((uniquelats.size, uniquelons.size), dtype=numpy.int16)

            (rows, _) = self.__rawdata.shape

            k = 0
            for i in range(0, uniquelats.size):
                for j in range(0, uniquelons.size):
                    coords[i][j] = k
                    k = k + 1
            # Now, using numpy's tricks on views, we simulate simplification of a
            # quadtree by prunning in the following way (e.g. decimate one up and
            # down):
            #
            #   X   X   X          X       X
            #
            #   X   X   X   ->
            #
            #   X   X   X          X       X
            # Syntax explanation: transpose, sample each (decimate + 1), transpose
            #                     sample each (decimate + 1), unravel the matrix
            #                     to use it as a filter
            nplist = coords.T[0::(self.__decimation + 1)].T[0::(self.__decimation + 1)].copy().ravel()
            coordfilter = map(int, nplist.tolist())
            self.__reduced = self.__rawdata[coordfilter]

    @property
    def locations(self):
        if self.isloaded:
            locations = []
            (rows, _) = self.__reduced.shape

            for i in range(0, rows):
                locations.append(loct.Location(self.__reduced[i][0], self.__reduced[1], self.__reduced[2:]))

            return locations
        else:
            return []
