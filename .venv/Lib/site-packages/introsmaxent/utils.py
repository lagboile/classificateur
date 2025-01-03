# University of Illinois at Urbana-Champaign
# The Whole Tale
# Summer Internship 2018
#
# Santiago Nunez-Corrales <nunezco2@illinois.edu>
# Copyright @ 2018, National Center for Supercomputing Applications. All rights reserved.

import numpy

def eucldist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return numpy.sqrt(numpy.power(x2 - x1, 2) + numpy.power(y2 - y1, 2))
