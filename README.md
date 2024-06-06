# panalyze - A Package To Analyze DSC Data

This is a work in progress and in its current form, the package is
extremely limited. It does have:
- A `Segment` class that represents one measurement segment (e.g. a
  heating run)
- A `Measurement` class that represents a full measurements that
  includes multiple `Segments`
- A `Baseline` class to represent a baseline used to determine the
  enthalpy under a peak.

The `Segment` and `Measurement` have a method to load data from a CSV
file exported from the TA TRIOS software. These CSV files should have
four columns, in this particular order: time, temperature, heat flow,
normalized heat flow. In addition, the exported file should include
the measurement parameters.

## Install

You can install panalyze using `pip`:

```
pip install git+https://github.com/wjbg/panalyze.git
```

Alternatively, you can also clone or download this repository as zip
file.

## License

Free as defined in the [MIT](https://choosealicense.com/licenses/mit/)
license.
