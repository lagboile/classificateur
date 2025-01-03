# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import io
import os
import json
import copy
import pandas
import shutil
import tarfile
import simplekml

import introsmaxent.solver as sv
import introsmaxent.geocode as gcode
import introsmaxent.lineargrid as lgrd
import introsmaxent.background as bgrd
import introsmaxent.features as fprov
import introsmaxent.observations as obs

from enum import Enum, IntEnum
from introsmaxent.darwincore import DarwinCoreSemantics as DCS


class ModelState(IntEnum):
    EMPTY = 1,
    INITIALIZED = 2,
    LOADED_NOENV = 3,
    LOADED_FULL = 4,
    COMPUTED = 5


class ModelOperation(Enum):
    CREATEMODEL = 0,
    DESCRIBEEXP = 1,
    SETINPUTFILES = 2,
    SETOUTPUTFILES = 3,
    SETPROBMODEL = 4,
    SETEXECMODEL = 5,
    SETOUTPUTMODEL = 6,
    SETGEOPARAMS = 7,
    SETGEOBOUNDS = 8,
    SETGEOJOIN = 9,
    SETORGANISM = 10,
    PARAMSTOJSON = 11,
    PARAMSFROMJSON = 12,
    LOGTOCSV = 13,
    LOGFROMCSV = 14,
    CONSUMEDATAENV = 15,
    CONSUMEDATAFTR = 16,
    CONSUMEDATAPO = 17,
    CONSUMEALL = 18,
    MAKEGRID = 19,
    COMPUTE = 20,
    RECOMPUTE = 21,
    DIFF = 22,
    FORK = 23,
    EXPRPACKAGE = 24,
    EMPTYTOINIT = 25,
    INITTOLOAD = 26,
    LOADTOCOMP = 27,
    REPRESENTCSV = 28,
    REPRESENTKML = 29,
    REPRESENTMAP = 30


class ModelPrintVerbosity(IntEnum):
    NONE = 0,
    ERROR = 1
    WARN = 2,
    INFO = 3


class MaxEntModel:
    def __init__(self):
        # We model the parameters through a dictionary, which can
        # later be dumped into a JSON file or consumed from one.
        self.__pars = {
            'experiment': {
                'shortname': '',
                'description': '',
                'author': '',
                'email': '',
                'organization': '',
                'date': '',
                'version': '',
                'species': '',
                'yearcollected': 0
            },
            'input': {
                'envfeatures': '',
                'envdata': '',
                'podata': ''
            },
            'geographic': {
                'needsref': False,
                'reffile': '',
                'joins': [],
                'datum': 'EPSG:4326',
                'datumfilter': DCS.DATUM,
                'latfilter': DCS.LAT,
                'lonfilter': DCS.LON,
                'decimation': 1,
                'lowerleft': (0.0, 0.0),
                'upperright': (0.0, 0.0),
                'delta': 0.0
            },
            'organism': {
                'idfilter': DCS.ID,
                'scinfilter': DCS.SCINAME,
                'attfilter': [DCS.L1TAXON,
                              DCS.L2TAXON,
                              DCS.L3TAXON,
                              DCS.L4TAXON]
            },
            'solver': {
                'beta': 0.0,
                'tau': 0.0,
                'iters': 1000,
                'cnvg': 1.0e-5,
                'depth': sv.MESolverDepth.Linear,
                'output': sv.MESolverOutput.RAW
            },
            'output': {
                'csv': '',
                'png': '',
                'kml': '',
                'log': ''
            }
        }
        # Logging facilities implemented as a list of steps represented by tuples
        self.__loglist = []
        self.__logstep = 0
        self.__logverbosity = ModelPrintVerbosity.INFO
        # Finite state machine pointer
        self.__machinestate = ModelState.EMPTY
        self.__opoutcome = False

        # Data handler objects
        # --------------------

        # Geocoder
        self.__gcoder = gcode.GeoCode()
        # Background environmental data
        self.__background = bgrd.Background()
        # Features handler
        self.__feats = fprov.FeaturesReader()
        # Presence-only data handler
        self.__podata = obs.PresenceDataReader()
        # Grid handler
        self.__grid = lgrd.MaxEntGrid()
        # Self-update the log
        self.__log(self.__logstep, ModelPrintVerbosity.INFO, ModelOperation.CREATEMODEL, 'New model created')
        self.__logstep += 1

    # Describe an experiment
    def describeexperiment(self, modname, desc, author, email, org, date, version, sp, yc):
        self.assertstate(ModelOperation.DESCRIBEEXP)

        self.__pars['experiment']['shortname'] = modname
        self.__pars['experiment']['description'] = desc
        self.__pars['experiment']['author'] = author
        self.__pars['experiment']['email'] = email
        self.__pars['experiment']['organization'] = org
        self.__pars['experiment']['date'] = date
        self.__pars['experiment']['version'] = version
        self.__pars['experiment']['species'] = sp
        self.__pars['experiment']['yearcollected'] = yc

        self.__sig_ok()

        self.refreshstate(ModelOperation.DESCRIBEEXP)

    # Functions that take care of initialization
    # ==========================================
    def setinputfiles(self, csvenvf, csvenvd, csvpodt):
        self.assertstate(ModelOperation.SETINPUTFILES)

        self.__pars['input']['envfeatures'] = csvenvf
        self.__pars['input']['envdata'] = csvenvd
        self.__pars['input']['podata'] = csvpodt

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETINPUTFILES)

    def setoutputfiles(self, csvfile, pngfile, kmlfile, logfile):
        self.assertstate(ModelOperation.SETOUTPUTFILES)

        self.__pars['output']['csv'] = csvfile
        self.__pars['output']['png'] = pngfile
        self.__pars['output']['kml'] = kmlfile
        self.__pars['output']['log'] = logfile

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETOUTPUTFILES)

    def setgeoparams(self, nref=False, rfile='', datum='WGS84',
                     datumf=DCS.DATUM,
                     latf=DCS.LAT,
                     lonf=DCS.LON, decim=1):
        self.assertstate(ModelOperation.SETGEOPARAMS)

        self.__pars['geographic']['datum'] = datum
        self.__pars['geographic']['datumfilter'] = datumf
        self.__pars['geographic']['latfilter'] = latf
        self.__pars['geographic']['lonfilter'] = lonf
        self.__pars['geographic']['decimation'] = decim

        if nref:
            self.__pars['geographic']['needsref'] = True

            if rfile != '':
                self.__pars['geographic']['reffile'] = rfile

                self.__sig_ok()
            else:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.SETGEOPARAMS, 'No reference file specified')
                self.__sig_fail()
        else:
            self.__sig_ok()

        self.refreshstate(ModelOperation.SETGEOPARAMS)

    def setgeojoinfields(self, fields=None):
        self.assertstate(ModelOperation.SETGEOJOIN)

        self.__pars['geographic']['joins'] = fields

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETGEOJOIN)

    # Set geographic bounds when there is no environment data
    def setgeobounds(self, llpos, urpos, delta):
        self.assertstate(ModelOperation.SETGEOBOUNDS)

        if self.__pars['input']['envfeatures'] != '':
            self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                       ModelOperation.SETGEOBOUNDS, 'Ignoring bounds, background data present')
        else:
            self.__pars['geographic']['lowerleft'] = llpos
            self.__pars['geographic']['upperright'] = urpos
            self.__pars['geographic']['delta'] = delta

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETGEOBOUNDS)

    # Set organism filters that help consume presence-only data
    def setorganismatts(self, idfilter, scinfilter, attfilter):
        self.assertstate(ModelOperation.SETORGANISM)

        self.__pars['organism']['idfilter'] = idfilter
        self.__pars['organism']['scinfilter'] = scinfilter
        self.__pars['organism']['attfilter'] = attfilter

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETORGANISM)

    # Setup the probability model in the solver.
    def setprobmodel(self, beta, tau):
        self.assertstate(ModelOperation.SETPROBMODEL)

        self.__pars['solver']['beta'] = beta
        self.__pars['solver']['tau'] = tau

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETPROBMODEL)

    # Set the execution model in the solver
    def setexecmodel(self, iters, cnvg):
        self.assertstate(ModelOperation.SETEXECMODEL)

        self.__pars['solver']['iters'] = iters
        self.__pars['solver']['cnvg'] = cnvg

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETEXECMODEL)

    # Set the output model in the solver
    def setotptmodel(self, depth, output):
        self.assertstate(ModelOperation.SETOUTPUTMODEL)

        self.__pars['solver']['depth'] = depth
        self.__pars['solver']['output'] = output

        self.__sig_ok()

        self.refreshstate(ModelOperation.SETOUTPUTMODEL)

    # Consume features data
    def __consumefeatures(self):
        self.assertstate(ModelOperation.CONSUMEDATAFTR)

        if self.__opoutcome:
            if self.__pars['input']['envfeatures'] != '':
                self.__feats.setfile(self.__pars['input']['envfeatures'])
                self.__feats.consume()
                self.__sig_ok()
            else:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.CONSUMEDATAFTR, 'Model does not contain environmental data')
                self.__sig_ok()

        self.refreshstate(ModelOperation.CONSUMEDATAFTR)

    # Consume environmental data
    def __consumeenvironment(self):
        self.assertstate(ModelOperation.CONSUMEDATAENV)

        # We assume that envfeatures has been properly loaded
        if self.__opoutcome:
            if self.__pars['input']['envdata'] != '':
                self.__background.setfile(self.__pars['input']['envdata'])
                self.__background.setdecimation(self.__pars['geographic']['decimation'])
                self.__background.setfeaturenames(self.__feats.features)
                self.__background.consume()
            else:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.CONSUMEDATAFTR, 'Model does not contain environmental data')
            self.__sig_ok()

        self.refreshstate(ModelOperation.CONSUMEDATAENV)

    # Consume presence-only data
    def __consumepodata(self):
        self.assertstate(ModelOperation.CONSUMEDATAPO)

        if self.__opoutcome:
            self.__podata.setfile(self.__pars['input']['podata'])
            self.__podata.setfilters(self.__pars['organism']['idfilter'],
                                     self.__pars['organism']['scifilter'],
                                     self.__pars['geographic']['latfilter'],
                                     self.__pars['geographic']['lonfilter'],
                                     self.__pars['organism']['attfilter'])
            self.__podata.consume(self.__grid, self.__pars['geographic']['joins'])

            self.__sig_ok()

        self.refreshstate(ModelOperation.CONSUMEDATAPO)

    # Make the grid using existing data
    def __makegrid(self):
        self.assertstate(ModelOperation.MAKEGRID)

        if self.__opoutcome:
            self.__grid.setfeatures(self.__feats.features)
            self.__grid.setlocations(self.__background.locations)
            self.__grid.setobservations(self.__podata.observations)

            if self.__background.locations:
                self.__grid.setbounds((self.__background.minlat,
                                       self.__background.minlon),
                                      (self.__background.maxlat,
                                       self.__background.maxlon))
            self.__grid.consume()

            self.__sig_ok()

        self.refreshstate(ModelOperation.MAKEGRID)

    # Consume all available data and assemble the grid
    def consume(self):
        self.assertstate(ModelOperation.CONSUMEALL)

        if self.__opoutcome:
            self.__consumefeatures()
            self.__consumeenvironment()
            self.__consumepodata()
            self.__makegrid()

            self.__sig_ok()

        self.refreshstate(ModelOperation.CONSUMEALL)

    # Functions that compute the current model based on existing data
    # ===============================================================

    def compute(self):
        self.assertstate(ModelOperation.COMPUTE)

        if self.__opoutcome:
            self.__grid.setprobparams(self.__pars['solver']['beta'],
                                      self.__pars['solver']['tau'])
            self.__grid.setexecparams(self.__pars['solver']['iters'],
                                      self.__pars['solver']['cnvg'])
            self.__grid.solvemaxent()
            self.__sig_ok()

        self.refreshstate(ModelOperation.COMPUTE)

    def recompute(self):
        self.assertstate(ModelOperation.RECOMPUTE)

        if self.__opoutcome:
            self.__grid.solvemaxent()
            self.__sig_ok()

        self.refreshstate(ModelOperation.RECOMPUTE)

    # Functions that represent the outcome through various visualization formats
    # ==========================================================================

    # Represent the output of the model as a CSV file
    def representcsv(self):
        self.assertstate(ModelOperation.REPRESENTCSV)

        if self.__opoutcome:
            data = self.__computedtodataframe()
            data.to_csv()

            self.__sig_ok()

        self.refreshstate(ModelOperation.REPRESENTCSV)

    # Represent the result of the model as a KML file
    def representkml(self):
        self.assertstate(ModelOperation.REPRESENTKML)

        if self.__opoutcome:
            data = self.__computedtodataframe()
            kml = simplekml.Kml(name=self.__pars['experiment']['species'] +
                                     ' collected during ' + self.__pars['experiment']['yearcollected'])
            data.apply(lambda x: kml.newpoint(name=x[self.__outlabel],
                                              coords=[(x[DCS.LAT], x[DCS.LON])]), axis=1)
            kml.save(path=self.__pars['output']['kml'])

            self.__sig_ok()

        self.refreshstate(ModelOperation.REPRESENTKML)

    # Represent the distribution as the PNG of a map
    def representmap(self):
        self.assertstate(ModelOperation.REPRESENTMAP)

        if self.__opoutcome:
            data = self.__computedtodataframe()

            (lllat, lllon) = self.__grid.llbounds
            (urlat, urlon) = self.__grid.urboudns

            # Mapping handling here / to be done

            self.__sig_ok()

        self.refreshstate(ModelOperation.REPRESENTMAP)

    # Helper function: model state to pandas dataframe
    def __computedtodataframe(self):
        labels = [DCS.LAT, DCS.LON, self.__outlabel]

        data = pandas.DataFrame.from_records(self.__grid.torecords(self.__pars['solver']['output']), columns=labels)

        return data

    # Helper function to get current output type label
    @property
    def __outlabel(self):
        return {
            sv.MESolverOutput.RAW: 'ROR',
            sv.MESolverOutput.CUMULATIVE: 'Cumulative',
            sv.MESolverOutput.LOGISTIC: 'Logistic',
            sv.MESolverOutput.CLOGLOG: 'Cloglog'
        }[self.__pars['solver']['output']]

    # Functions that take care of saving and loading an experiment to storage
    # =======================================================================

    # Initialize parameters from a JSON file
    # TODO: a JSON verifier is needed here. Pending contingent on dev time.
    def initfromjson(self, jsonfile):
        self.assertstate(ModelOperation.PARAMSFROMJSON)

        with open(jsonfile) as jfile:
            self.__pars = json.load(jfile)

        self.refreshstate(ModelOperation.PARAMSFROMJSON)

    # Save current model parametrization to JSON file
    def paramstojson(self, directory=''):
        self.assertstate(ModelOperation.PARAMSTOJSON)

        if self.__opoutcome:
            with io.open(directory + self.__pars['experiment']['shortname'] + '.json',
                         'w', encoding='utf-8') as f:
                json.dump(self.__pars, f, ensure_ascii=False)

            self.__sig_ok()

        self.refreshstate(ModelOperation.PARAMSTOJSON)

    # Package the experiment including the parameters and the input files
    def package(self):
        self.assertstate(ModelOperation.EXPRPACKAGE)

        # We use the name of the experiment to create a compressed file
        if self.__opoutcome:
            basepath = './' + self.__pars['experiment']['short'] + '/'

            shutil.rmtree(basepath)
            os.makedirs(basepath)

            self.paramstojson(basepath)
            self.logtocsv()

            # Important: this assumes relative file names
            if self.__pars['input']['envfeatures'] != '':
                shutil.copyfile(self.__pars['input']['envfeatures'], basepath + self.__pars['input']['envfeatures'])
                shutil.copyfile(self.__pars['input']['envdata'], basepath + self.__pars['input']['envdata'])

            shutil.copyfile(self.__pars['input']['podata'], basepath + self.__pars['input']['podata'])
            shutil.copyfile(self.__pars['output']['log'], basepath + self.__pars['output']['log'])

            tarname = self.__pars['experiment']['short'] + '.tar.gz'
            tar = tarfile.open(tarname, 'w:gz')
            tar.add(basepath, arcname=tarname)
            tar.close()

            shutil.rmtree(basepath)

            self.__sig_ok()

        self.refreshstate(ModelOperation.EXPRPACKAGE)

    # Functions that take care of making copies of a model
    # ====================================================

    # Create a carbon copy of the current model. Semantically equal to a
    # self documented fork.
    def carboncopy(self):
        self.__log(self.__logstep, ModelPrintVerbosity.INFO,
                   ModelOperation.FORK, 'Carbon copy created')
        return copy.deepcopy(self)

    # Functions that compare between two models
    # =========================================

    def diff(self, model):
        self.assertstate(ModelOperation.DIFF)

        assert isinstance(model, lgrd.MaxEntGrid)

        if self.__opoutcome:
            result = self.__grid.diff(model)

            self.__sig_ok()
        else:
            result = None

        self.refreshstate(ModelOperation.DIFF)

        return result

    # Functions that inspect features of the model
    # ============================================
    # Note that these inspection structures do not require guards since
    # no modification of the state exists

    # Inspect the properties of presence only data
    @property
    def inspectpoattributes(self):
        print(self.__podata.features)
        return None

    @property
    def inspectgeorefattributes(self):
        print(self.__gcoder.geoattributes)
        return None

    @property
    def inspectstate(self):
        print('Model state: ' + self.__translatestate(self.__machinestate))
        return None

    @property
    def inspectpars(self):
        print(json.dumps(self.__pars, indent=4, separators=(',', ': ')))
        return None

    @property
    def grid(self):
        return self.__grid

    # Functions that take care of logs
    # =================================
    def __log(self, step, logverb, operation, outcome):
        self.__loglist.append((step, logverb, self.__machinestate, operation, outcome))
        self.__printlog(step, self.__logtyperep(logverb), self.__translatestate, operation, outcome)

    def logtocsv(self):
        self.assertstate(ModelOperation.LOGTOCSV)

        labels = ['Step', 'Event type', 'Machine state', 'Operation', 'Input', 'Output', 'Outcome']
        data = pandas.DataFrame.from_records(self.__loglist, columns=labels)
        data.to_csv()

        self.refreshstate(ModelOperation.LOGTOCSV)

    def logfromcvs(self):
        self.assertstate(ModelOperation.LOGFROMCSV)

        self.__loglist = pandas.read_csv(self.__pars['input']['log']).to_records()
        self.__logstep = 0
        self.__log(self.__logstep, ModelPrintVerbosity.WARN, ModelOperation.LOGFROMCSV,
                   'Irreversible log load from csv')

        self.__sig_ok()

        self.refreshstate(ModelOperation.LOGFROMCSV)

    @property
    def logstep(self):
        return self.__logstep

    @staticmethod
    def __prettyprint(i, tp, st, op, oc):
        print('| {:4} | {:12} | {:12} | {:30}| {}'.format(i, tp, st, op, oc))
        return None

    def setlogprintverbosity(self, verbosity):
        self.__logverbosity = verbosity

    def __printlog(self, i, ltype, st, op, outcome):
        if ltype <= self.__logverbosity:
            self.__prettyprint(i, self.__translateverbosity(ltype),
                               self.__translatestate(st), self.__translateop(op), outcome)

    @staticmethod
    def __logtyperep(ltype):
        return {
            0: 'None',
            2: 'Error',
            3: 'Warning',
            4: 'Information'
        }.get(ltype, 'NA')

    def __transcriptentry(self, logentry):
        (i, ltype, st, op, outcome) = logentry

        if ltype <= self.__logverbosity:
            self.__prettyprint(i, self.__translateverbosity(ltype),
                               self.__translatestate(st), self.__translateop(op),
                               outcome)

    # Render the current state of the log
    @property
    def transcript(self):
        self.__prettyprint('Step', 'Level', 'State', 'Operation', 'Outcome')
        print(
            "|------|--------------|--------------|-------------------------------|---------------------------------------")

        for tl in self.__loglist:
            self.__transcriptentry(tl)

        return None

    # Functions that take care of the finite state machine
    # ====================================================
    # Opening guard for testing preconditions
    def assertstate(self, op):
        # Opening guard for describing an experiment
        if op == ModelOperation.DESCRIBEEXP:
            # Experiment descriptions have no state requirements. They may happen at
            # any state in the finite state machine.
            pass

        # Opening guard for setting input files
        elif op == ModelOperation.SETINPUTFILES:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETINPUTFILES, 'Input file references previously used to load data exist')

        # Opening guard for setting output files
        elif op == ModelOperation.SETOUTPUTFILES:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETOUTPUTFILES, 'Output file references previously used to load data exist')

        # Opening guard for setting output files
        elif op == ModelOperation.SETOUTPUTFILES:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETOUTPUTFILES,
                           'Output file references previously used to load data exist')

        # Opening guard for setting geoparameters
        elif op == ModelOperation.SETGEOPARAMS:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETGEOPARAMS,
                           'Geographic parameters were already defined')

        # Opening guard for setting geoparameters
        elif op == ModelOperation.SETGEOBOUNDS:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETGEOBOUNDS,
                           'Geographic bounds already exist')

        # Opening guard for setting geoparameters
        elif op == ModelOperation.SETORGANISM:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETORGANISM,
                           'Organism attributes were already defined')

        # Opening guard for setting geoparameters
        elif op == ModelOperation.SETGEOJOIN:
            # Potential inconsistency when file references are updated
            if self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETGEOJOIN,
                           'Geographic joins already defined')

        # Opening guard for setting the probability model in the solver
        elif op == ModelOperation.SETPROBMODEL:
            # Potential inconsistency when already computed
            if self.__machinestate == ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETPROBMODEL,
                           'Changing probability solver model after computation')

        # Opening guard for setting the execution model in the solver
        elif op == ModelOperation.SETEXECMODEL:
            # Potential inconsistency when already computed
            if self.__machinestate == ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETEXECMODEL,
                           'Changing execution solver model after computation')

        # Opening guard for setting the output model in the solver
        elif op == ModelOperation.SETOUTPUTMODEL:
            # Potential inconsistency when already computed
            if self.__machinestate == ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.SETEXECMODEL,
                           'Changing output solver model after computation')

        # Opening guard for consuming the features
        elif op == ModelOperation.CONSUMEDATAFTR:
            if self.__machinestate < ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.CONSUMEDATAFTR, 'Incomplete initialization prevents loading feature data')
                self.__opoutcome = False
            elif self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.CONSUMEDATAFTR,
                           'Overwriting existing feature data')
                self.__opoutcome = True
            else:
                pass

        # Opening guard for consuming the features
        elif op == ModelOperation.CONSUMEDATAENV:
            if self.__machinestate < ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.CONSUMEDATAENV, 'Incomplete initialization prevents loading background data')
                self.__opoutcome = False
            elif self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.CONSUMEDATAENV,
                           'Overwriting existing background data')
                self.__opoutcome = True
            else:
                pass

        # Opening guard for consuming the features
        elif op == ModelOperation.CONSUMEDATAPO:
            if self.__machinestate < ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.CONSUMEDATAPO,
                           'Incomplete initialization prevents loading presence-only data')
                self.__opoutcome = False
            elif self.__machinestate > ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                           ModelOperation.CONSUMEDATAPO,
                           'Overwriting existing presence-only data')
                self.__opoutcome = True
            else:
                pass

        # Opening guard for consuming the features
        elif op == ModelOperation.MAKEGRID:
            if self.__machinestate < ModelState.LOADED_NOENV:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.MAKEGRID,
                           'Data has not been loaded, unable to make grid')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for consuming the features
        elif op == ModelOperation.CONSUMEALL:
            if self.__machinestate < ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.MAKEGRID,
                           'Data has not been loaded, unable to make grid')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for computing the model
        elif op == ModelOperation.COMPUTE:
            if self.__machinestate < ModelState.LOADED_NOENV:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.CONSUMEDATAENV,
                           'Incomplete initialization making grid')
                self.__opoutcome = False
            else:
                if self.__machinestate == ModelState.LOADED_NOENV:
                    self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                               ModelOperation.COMPUTE,
                               'Computing model with presence-only data')
                self.__opoutcome = True

        # Opening guard for recomputing the model
        elif op == ModelOperation.RECOMPUTE:
            if self.__machinestate < ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.RECOMPUTE,
                           'Unable to recompute uncomputed model')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for recomputing the model
        elif op == ModelOperation.REPRESENTCSV:
            if self.__machinestate < ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.REPRESENTCSV,
                           'Unable to represent uncomputed model as CSV file')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for recomputing the model
        elif op == ModelOperation.REPRESENTKML:
            if self.__machinestate < ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.REPRESENTKML,
                           'Unable to represent uncomputed model as KML file')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for recomputing the model
        elif op == ModelOperation.REPRESENTMAP:
            if self.__machinestate < ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.REPRESENTKML,
                           'Unable to represent uncomputed model as map plot')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for saving the parameters to a JSON file
        elif op == ModelOperation.PARAMSTOJSON:
            if self.__machinestate < ModelState.INITIALIZED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.PARAMSTOJSON, 'Incomplete initialization prevents loading feature data')
                self.__opoutcome = False
            else:
                self.__opoutcome = True

        # Opening guard for loading the parameters from a JSON file
        elif op == ModelOperation.PARAMSFROMJSON:
            self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                       ModelOperation.PARAMSFROMJSON, 'Starting a new configuration')

        # Opening guard for packaging experiments
        elif op == ModelOperation.EXPRPACKAGE:
            if self.__machinestate < ModelState.LOADED_NOENV:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.EXPRPACKAGE, 'Possibly inconsistent states cannot be packaged')
                self.__opoutcome = False
            else:
                if self.__machinestate == ModelState.LOADED_NOENV:
                    self.__log(self.__logstep, ModelPrintVerbosity.WARN,
                               ModelOperation.EXPRPACKAGE, 'Packaging model without background environment data')
                self.__opoutcome = True

        # Opening guard for computing the model
        elif op == ModelOperation.DIFF:
            if self.__machinestate < ModelState.COMPUTED:
                self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                           ModelOperation.RECOMPUTE,
                           'Unable to compare an uncomputed model')
                self.__opoutcome = False

        else:
            self.__log(self.__logstep, ModelPrintVerbosity.ERROR,
                       ModelOperation.EXPRPACKAGE, 'Unknown operation')
            self.__opoutcome = False

    # Closing guard
    def refreshstate(self, op):
        # Handle the case if the last operation failed with a generic message
        # and do not update the step counter. No need to test for state
        # transitions either.
        if not self.__opoutcome:
            self.__log(self.__logstep, ModelPrintVerbosity.ERROR, op, 'Operation failed.')
        else:
            # We factor the operation and the content it generates from the
            # final logging call and step update.
            # Closing guard for describing an experiment
            level = ModelPrintVerbosity.INFO

            if op == ModelOperation.DESCRIBEEXP:
                message = 'Experiment description updated'

            elif op == ModelOperation.SETINPUTFILES:
                message = 'Input file references updated'
                if self.__pars['input']['envfeatures'] == '':
                    self.__log(self.__logstep, level, op,
                               'Model contains no background data')

            elif op == ModelOperation.SETOUTPUTFILES:
                message = 'Output file references updated'

            elif op == ModelOperation.SETGEOPARAMS:
                message = 'Geographical parameters set'

            elif op == ModelOperation.SETGEOJOIN:
                message = 'Geographical join fields set'

            elif op == ModelOperation.SETPROBMODEL:
                message = 'Solver probability model set'

            elif op == ModelOperation.SETEXECMODEL:
                message = 'Solver execution model set'

            elif op == ModelOperation.SETOUTPUTMODEL:
                message = 'Solver probability model set'

            elif op == ModelOperation.SETORGANISM:
                message = 'Organism attributes set'

            elif op == ModelOperation.CONSUMEDATAFTR:
                message = 'Environmental features data consumed'

            elif op == ModelOperation.CONSUMEDATAENV:
                message = 'Background environmental data consumed'

            elif op == ModelOperation.CONSUMEDATAPO:
                message = 'Presence-only data consumed'

            elif op == ModelOperation.MAKEGRID:
                message = 'MaxEnt grid constructed'

            elif op == ModelOperation.CONSUMEALL:
                message = 'All data consumed'

            elif op == ModelOperation.COMPUTE:
                message = 'Model has been computed'

            elif op == ModelOperation.COMPUTE:
                message = 'Model has been recomputed'

            elif op == ModelOperation.RECOMPUTE:
                message = 'Model has been recomputed'

            elif op == ModelOperation.REPRESENTCSV:
                message = 'Model has been save to CSV file'

            elif op == ModelOperation.PARAMSTOJSON:
                message = 'Parameters saved to JSON file'

            elif op == ModelOperation.PARAMSFROMJSON:
                message = 'Parameters loaded from JSON file'

            elif op == ModelOperation.EXPRPACKAGE:
                message = 'Experiment packaged'

            elif op == ModelOperation.DIFF:
                message = 'Comparing against another model'

            else:
                message = 'Unknown operation'

            # Update the log, increase the step, test for state change
            # and reset the operation signal
            self.__log(self.__logstep, level, op, message)
            self.__logstep += 1
            self.__testandtransition()
            self.__sig_reset()

    # Obtain the text representation for the machine state
    @staticmethod
    def __translatestate(state):
        return {
            ModelState.EMPTY: 'Empty',
            ModelState.INITIALIZED: 'Initialized',
            ModelState.LOADED_NOENV: 'Partial load',
            ModelState.LOADED_FULL: 'Full load',
            ModelState.COMPUTED: 'Computed'
        }[state]

    # Text representation for an operation
    @staticmethod
    def __translateop(op):
        return {
            ModelOperation.CREATEMODEL: 'Creating a new model',
            ModelOperation.DESCRIBEEXP: 'Describing experiment',
            ModelOperation.SETINPUTFILES: 'Setting input files',
            ModelOperation.SETOUTPUTFILES: 'Setting output files',
            ModelOperation.SETPROBMODEL: 'Configuring probability model',
            ModelOperation.SETEXECMODEL: 'Configuring execution model',
            ModelOperation.SETOUTPUTMODEL: 'Configuring output model',
            ModelOperation.SETGEOPARAMS: 'Setting geographic parameters',
            ModelOperation.SETGEOBOUNDS: 'Setting geographic bounds',
            ModelOperation.PARAMSTOJSON: 'Saving parameters to JSON file',
            ModelOperation.PARAMSFROMJSON: 'Loading parameters from JSON file',
            ModelOperation.LOGTOCSV: 'Saving log to CSV file',
            ModelOperation.LOGFROMCSV: 'Loading log from CSV file',
            ModelOperation.CONSUMEDATAENV: 'Consuming environmental data',
            ModelOperation.CONSUMEDATAFTR: 'Consuming features data',
            ModelOperation.CONSUMEDATAPO: 'Consuming presence-only data',
            ModelOperation.CONSUMEALL: 'Consuming model data',
            ModelOperation.MAKEGRID: 'Creating the grid with PO data',
            ModelOperation.COMPUTE: 'Computing the model',
            ModelOperation.RECOMPUTE: 'Recomputing the model',
            ModelOperation.DIFF: 'Analysis model diffs',
            ModelOperation.FORK: 'Forking the model',
            ModelOperation.EXPRPACKAGE: 'Packaging the experiment',
            ModelOperation.SETGEOJOIN: 'Setting geo join attributes',
            ModelOperation.SETORGANISM: 'Setting organism attributes',
            ModelOperation.EMPTYTOINIT: 'Transitioning from Empty state',
            ModelOperation.INITTOLOAD: 'Transitioning from Initialized state',
            ModelOperation.LOADTOCOMP: 'Transitioning from Loaded state',
            ModelOperation.REPRESENTCSV: 'Representing output as CSV file',
            ModelOperation.REPRESENTKML: 'Representing output as KML file',
            ModelOperation.REPRESENTMAP: 'Representing output as a map plot'
        }[op]

    # Text representation for the verbosity level
    @staticmethod
    def __translateverbosity(lvb):
        return {
            ModelPrintVerbosity.INFO: 'Information',
            ModelPrintVerbosity.WARN: 'Warning',
            ModelPrintVerbosity.ERROR: 'Error',
            ModelPrintVerbosity.NONE: 'None'
        }[lvb]

    # Indicate success after a certain operation
    def __sig_ok(self):
        self.__opoutcome = True

    # Indicate success after a certain operation
    def __sig_fail(self):
        self.__opoutcome = False

    # Reset the operation result flag. This is the same than fail, yet
    # we want the code to remain readable.
    def __sig_reset(self):
        self.__opoutcome = False

    # Test the state transition
    def __testandtransition(self):
        if self.__machinestate == ModelState.EMPTY:
            if self.__testemptytoinitialized:
                self.__machinestate = ModelState.INITIALIZED
                print('State transitioned from Empty to Initialized')
                self.__log(self.__logstep, ModelPrintVerbosity.INFO,
                           ModelOperation.EMPTYTOINIT, 'State transitioned from Empty to Initialized')
        if self.__machinestate == ModelState.INITIALIZED:
            if self.__testinitializedtoloadednobgnd():
                self.__machinestate = ModelState.LOADED_NOENV
                print('State transitioned from Initialized to Loaded w/o presence-only data')
                self.__log(self.__logstep, ModelPrintVerbosity.INFO,
                           ModelOperation.INITTOLOAD,
                           'State transitioned from Initialized to Loaded w/o presence-only data')
            elif self.__testinitializedtoloadedfull():
                self.__machinestate = ModelState.LOADED_FULL
                print('State transitioned from Initialized to fully Loaded')
                self.__log(self.__logstep, ModelPrintVerbosity.INFO,
                           ModelOperation.INITTOLOAD, 'State transitioned from Initialized to fully Loaded')
        if self.__machinestate == ModelState.LOADED_NOENV or self.__machinestate == ModelState.LOADED_FULL:
            if self.__testloadedtocomputed():
                self.__machinestate = ModelState.COMPUTED
                print('State transitioned from Loaded to Computed')
                self.__log(self.__logstep, ModelPrintVerbosity.INFO,
                           ModelOperation.LOADTOCOMP, 'State transitioned from Loaded to Computed')

    # Test for the empty to initialized transition
    @property
    def __testemptytoinitialized(self):
        result = True
        result &= self.__pars['experiment']['shortname'] != ''
        result &= self.__pars['experiment']['description'] != ''
        result &= self.__pars['experiment']['author'] != ''
        result &= self.__pars['experiment']['email'] != ''
        result &= self.__pars['experiment']['organization'] != ''
        result &= self.__pars['experiment']['date'] != ''
        result &= self.__pars['experiment']['version'] != ''
        result &= self.__pars['experiment']['species'] != ''
        result &= self.__pars['experiment']['yearcollected'] != 0

        if self.__pars['input']['envdata'] == '' and (not self.__pars['geographic']['needsref']):
            result &= self.__pars['geographic']['lowerleft'] != (0.0, 0.0)
            result &= self.__pars['geographic']['upperright'] != (0.0, 0.0)
            result &= self.__pars['geographic']['delta'] != 0

        if self.__pars['geographic']['needsref'] and self.__pars['input']['envdata'] != '':
            result &= self.__pars['geographic']['reffile'] != ''
            result &= self.__pars['geographic']['joins'] != []

        result &= self.__pars['organism']['idfilter'] != ''
        result &= self.__pars['organism']['scinfilter'] != ''

        result &= self.__pars['output']['log'] != ''
        result &= self.__pars['output']['csv'] != ''
        result &= self.__pars['output']['png'] != ''
        result &= self.__pars['output']['kml'] != ''

        return result

    # Test for the initialized to loaded_nodata transition
    def __testinitializedtoloadednobgnd(self):
        result = True
        result &= not self.__feats.isloaded
        result &= not self.__background.isloaded
        result &= self.__grid.isloaded
        result &= not self.__grid.issolved
        return result

    # Test for the initialized to loaded_full transition
    def __testinitializedtoloadedfull(self):
        result = True
        result &= self.__feats.isloaded
        result &= self.__background.isloaded
        result &= self.__grid.isloaded
        result &= not self.__grid.issolved
        return result

    # Test for loaded (in any case) to computed
    def __testloadedtocomputed(self):
        return self.__grid.issolved
