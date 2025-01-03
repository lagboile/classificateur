# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import model
from darwincore import DarwinCoreSemantics as dcs

def main():
    # Create a new model
    mymodel = model.MaxEntModel()

    mymodel.setlogprintverbosity(model.ModelPrintVerbosity.INFO)

    # Describe the model
    mymodel.describeexperiment('magnolias', 'A test for MaxEnt',
                               'Santiago Nunez-Corrales', 'nunezco2@illinois,edu',
                               'UIUC/NCSA/iSchool', '2018-06-22', '0.1', 'Magnolia', 1984)

    # Set the input and output files
    mymodel.setinputfiles('features_test.csv', 'env_test.csv', 'weakley2015.test')
    mymodel.setoutputfiles('magnolia.csv', 'magnolia.png', 'magnolia.kml', 'magnolia.log')
    mymodel.setgeoparams(True, 'counties_curated.csv')
    mymodel.setgeojoinfields(['stateProvince', 'countyName'])
    mymodel.setorganismatts(dcs.ID, dcs.SCINAME,
                            [dcs.L1TAXON,  dcs.L2TAXON])
    mymodel.setprobmodel(1.0e-3, 1.0e-2)

#    # Consume the data
#    mymodel.consume()

    # Run the model
#    mymodel.compute()

    # Maybe, we want to remove all references to the environment in a new model and recompute it
    # We would like to preserve the log. Carbon copies can be made in any moment.
#    mynewmodel =  mymodel.carboncopy()

#    # Roughly, 0.01 degrees corresponds to 1km
#    mynewmodel.setinputfiles('', '', 'weakley2015.test')
#    mynewmodel.setgeobounds((30.91, -94.94), (39.78, -76.36), 0.01)
#    mynewmodel.recompute()

    # Compare against the previous model
#    mydiff = mynewmodel.diff(mymodel)

    # Show the comparison in a KML
#    mydiff.representKML()

    # See the history of operations and outcomes that we have obtained through the logs
    print mymodel.transcript

    exit()

if __name__ == '__main__':
    main()
