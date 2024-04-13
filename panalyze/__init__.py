"""
panalyze: A tool to analyze DSC data
====================================

Provides a set of objects to analyze DSC data.


Documentation
-------------

Documentation is available in the docstrings. In addition, a few examples are
provided at: `https://github.com/wjbg/panalyze`.


Available objects
-----------------

Measurement
  Class to represent a DSC measurement file.
Segment
  Class to represent a DSC measurement segment.


Available global functions
--------------------------

SAPPHIRE_CP
  Returns the specific heat of sapphire at provided temperature.

"""

__version__ = "0.0.1"


from .panalyze import *
