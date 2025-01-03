# University of Illinois at Urbana Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import pandas


class GeoCode:
    def __init__(self):
        self.__reference = ''
        self.__df = None
        self.__loaded = False

    @property
    def isfileset(self):
        return self.__reference != ''

    def setfile(self, ref):
        self.__reference = ref

    @property
    def isloaded(self):
        return self.__loaded

    @property
    def geoattributes(self):
        if self.isloaded:
            return self.__df.keys().tolist()

    def consume(self):
        if self.isfileset:
            self.__df = pandas.read_csv(self.__reference)
            self.__loaded = True

    def georeflist(self, features, tuples):
        if self.isloaded:
            wkcopy = self.__df.filter(items=features).copy()
            result = wkcopy.iloc[0:0]

            for t in tuples:
                (state, county) = t
                result = pandas.concat([result,
                                        wkcopy.loc[wkcopy[features[0]] ==
                                                   state].loc[wkcopy[features[1]] ==
                                                              county].copy()], sort=False)

            return result
        else:
            return []

    def georefjoin(self, target, joins, priorgeog, outfile, saved):
        if self.isloaded:
            toref = pandas.read_csv(target).drop(labels=priorgeog, axis=1)
            result = pandas.merge(toref, self.__df, on=joins)

            if saved:
                result.to_csv(outfile)

            return result
        else:
            return None
