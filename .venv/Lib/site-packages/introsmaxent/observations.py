# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import numpy
import pandas
import introsmaxent.geocode as gc
from introsmaxent.darwincore import DarwinCoreSemantics as DCS
from introsmaxent.podata import POData


class PresenceDataReader:
    def __init__(self, geocode=False):
        self.__document = ''
        self.__data = None
        self.__idfilter = ''
        self.__scifilter = ''
        self.__latfilter = ''
        self.__lonfilter = ''
        self.__afilter = ''
        self.__geocode = geocode
        self.__loaded = False

    @property
    def isfileset(self):
        return self.__document != ''

    def setfile(self, csvfile):
        self.__document = csvfile

    @property
    def arefiltersset(self):
        # Note that we allow the process to run even if the extra annotation filters
        # are not specified.
        return (self.__latfilter != '' and
                self.__lonfilter != '' and
                self.__idfilter != '' and
                self.__scifilter != '')

    def setfilters(self, idfilter=DCS.ID, scifilter=DCS.SCINAME,
                   geoglat=DCS.LAT, geoglon=DCS.LON, annotsfilter=None):
        self.__idfilter = idfilter
        self.__scifilter = scifilter
        self.__latfilter = geoglat
        self.__lonfilter = geoglon

        if annotsfilter is None:
            self.__afilter = []
        else:
            self.__afilter = annotsfilter

    def consume(self, grid, joins):
        assert isinstance(grid, gc.GeoCode)

        if self.isfileset and self.arefiltersset:
            self.__data = pandas.read_csv(self.__document)[[self.__idfilter,
                                                            self.__scifilter,
                                                            self.__latfilter,
                                                            self.__lonfilter] + self.__afilter]

            # This takes care of geocoding with a reference file
            # Use case: only geographic locations are given without lat lon data
            if self.__geocode:
                if grid.isloaded:
                    self.__data = grid.georefjoin(self.__data, joins,
                                                  [self.__latfilter, self.__lonfilter], '', False)
                    self.__loaded = True
                else:
                    self.__loaded = False
            else:
                self.__loaded = True

    @property
    def idfilter(self):
        return self.__idfilter

    @property
    def scifilter(self):
        return self.__scifilter

    @property
    def latfilter(self):
        return self.__latfilter

    @property
    def lonfilter(self):
        return self.__lonfilter

    @property
    def annotfilter(self):
        return self.__afilter

    @property
    def isloaded(self):
        return self.__loaded

    @property
    def features(self):
        if self.isloaded:
            return self.__data.keys().tolist()
        else:
            return []

    @property
    def records(self):
        if self.isloaded:
            return self.__data.values
        else:
            return numpy.ndarray([])

    @property
    def dims(self):
        if self.isloaded:
            return self.__data.values.shape
        else:
            return 0, 0

    @property
    def observations(self):
        if self.isloaded:
            obslist = []

            for index, row in self.__data.iterrows():
                obslist.append(POData(row[self.__idfilter],
                                      row[self.__scifilter],
                                      row[self.__latfilter],
                                      row[self.__lonfilter],
                                      self.__afilter,
                                      row.filter(items=self.__afilter).tolist()))

            return obslist
        else:
            return []
