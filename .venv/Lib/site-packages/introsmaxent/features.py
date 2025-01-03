# University of Illinois at Urbana Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.
import pandas
import requests


class Feature:
    """
    This class creates a representation of environmental features for MaxEnt.
    Features can be used for provenance annotations and also for visualization.

    In this code, doi's are included as a means to query literature that may
    relate taxa to features as a type of sanity check.
    """
    def __init__(self, name, desc='', doi=''):
        self.__name = name
        self.__desc = desc
        self.__doi = doi

    @property
    def name(self):
        return self.__name

    @property
    def desc(self):
        return self.__desc

    @property
    def doi(self):
        return self.__doi

    def citation(self):
        url = 'http://dx.doi.org/' + self.__doi
        headers = {'accept': 'text/x-bibliography; style=apa'}
        r = requests.get(url, headers=headers)

        return r.text

    def describe(self):
        print('  Feature:' + self.name)
        print('  Description:' + self.desc)
        print('  Reference: ' + self.citation())


class FeaturesReader:
    def __init__(self):
        self.__document = ''
        self.__features = []
        self.__fnames = []
        self.__loaded = False

    @property
    def isfileset(self):
        return self.__document != ''

    def setfile(self, cvsfile):
        self.__document = cvsfile

    def consume(self):
        if self.isfileset:
            fdata = pandas.read_csv(self.__document)
            df = fdata.values

            (rows, _) = df.shape

            for i in range(0, rows):
                ft = Feature(df[i][0], df[i][1], df[i][2])
                self.__features.append(ft)

            self.__fnames = fdata['Feature'].tolist()
            self.__loaded = True
        else:
            return False

    @property
    def features(self):
        return self.__features

    @property
    def names(self):
        return self.__fnames

    @property
    def isloaded(self):
        return self.__loaded

    def describe(self):
        print('Model Features')
        print('================================')
        print('')
        for f in self.__features:
            f.describe()
