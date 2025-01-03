# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import introsmaxent.utils as utils


class POData:
    def __init__(self, poid, name, lat, lon, atts, vals):
        self.__obsid = poid
        self.__lat = lat
        self.__lon = lon
        self.__name = name
        self.__atts = atts
        self.__vals = vals

    @property
    def location(self):
        return self.__lat, self.__lon

    @property
    def name(self):
        return self.__name

    @property
    def attributes(self):
        return self.__atts, self.__vals

    def distancefrom(self, point):
        return utils.eucldist(self.location, point)

    def describe(self):
        print("  Observation record " + self.__obsid)
        print("  --------------------------------------------------------------------------")
        print("  Latitude: " + str(self.__lat))
        print("  Longitude: " + str(self.__lon))
        print("  Name: " + self.__name)
        print("  Features: " + str(self.__atts))
        print("  Values:   " + str(self.__vals))
        print("")
