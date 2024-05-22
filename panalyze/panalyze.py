"""
Contains core of panalyze: Measurement, Segment

"""

from __future__ import annotations
from typing import Callable, Union, Optional
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from datetime import datetime
from io import StringIO
import importlib.resources as impresources
from . import materials

vector = matrix = npt.NDArray[np.float_]  # Annotation shorthand

# Sapphire heat capacity (there is probably a more elegant way to do this)
fn = impresources.files(materials) / 'sapphire_cp.csv'
CP_SAPPHIRE_DATA = np.loadtxt(fn,
                              dtype=np.float32, delimiter=",",
                              skiprows=1, usecols=(0, 2))
CP_SAPPHIRE = CubicSpline(CP_SAPPHIRE_DATA[:, 0], CP_SAPPHIRE_DATA[:, 1])


class Measurement():
    """Class to represent a DSC measurement.

    Attributes
    ----------
    sample : str
        Sample name.
    mass : float
        Sample mass in mg.
    segments : list[Segment]
        List with measurement segments.
    operator : str
        Name of the operator.
    date : datetime.date
        Run date.
    instrument : str
        Instrument name.
    fn : str
        Filename.

    Methods
    -------
    read_trios_csv(fn)
        Reads TRIOS csv file and updates attributes.

    """

    def __init__(self, fn: str):
        """Initialize Measurement instance.

        Parameters
        ----------
        fn : str
            Filename.

        Note: For now only CSV files exported from TRIOS are supported.

        """
        self.read_trios_csv(fn)

    def read_trios_csv(self, fn: str):
        """Read csv file exported frmo TRIOS software by TA.

        Parameters
        ----------
        fn : str
            Filename.

        Note
        ----
        The datafile should include the parameters and the steps. The following
        columns (in order) need to be exported: time [s], temperature [°C],
        heat flow [mW], norm. heat flow [W/g].

        """
        with open(fn, 'r', encoding='utf-8-sig') as f:
            head, *segments = f.read().split("[step]\n")
            for info in head.strip().split("\n"):
                keyw, *val = info.lower().split(",")
                if keyw == "filename":
                    self.fn = val[0] + ".csv"
                elif keyw == "operator":
                    self.operator = val[0]
                elif keyw == "rundate":
                    self.date = datetime.strptime(val[0], "%m/%d/%Y")
                elif keyw == "sample name":
                    self.sample = val[0]
                elif keyw == "sample mass":
                    self.mass = float(val[0].split(" ")[0])  # mg
                elif keyw == "instrument name":
                    self.instrument = val[0]
            self.segments = []
            for segment in segments:
                name = segment.split("\n")[0]
                data = np.genfromtxt(StringIO(segment),
                                     delimiter=",", skip_header=3,
                                     autostrip=True, missing_values=np.NAN)
                data = data[~np.isnan(data[:, :3]).any(axis=1)]  # clear NaNs
                self.segments.append(Segment(data, name, self.mass))

    def __str__(self) -> str:
        """Return string for printing."""
        meta = (f"\n{self.sample}\n" +
                "------------------------------------\n" +
                f"Mass:               {self.mass} mg\n" +
                f"Date:               {self.date.date()}\n" +
                f"Operator:           {self.operator}\n" +
                f"Instrument:         {self.instrument}\n" +
                f"Filename:           {self.fn}\n" +
                f"Number of segments: {len(self.segments):<2}\n\n")
        segm = ("Segments\n"
                "-----------------------------------\n")
        for i, segment in enumerate(self.segments):
            segm += f"{i:<2} {segment.name}\n"
        return (meta + segm)

    def __getitem__(self, index) -> Segment:
        """Return segment using index."""
        return self.segments[index]


class Segment():
    """Class to represent a DSC measurement segment.

    Attributes
    ----------
    data : np.ndarray(dim=2, type=float)
        Matrix with DSC data in the following columns:
          - Time [s]
          - Temperature [°C]
          - Heat flow [mW]
          - Normalized heat flow [J/g]
    name : str
        Description of segment
    N : int
        Number of data points.
    t_interval : (float, float)
        Start and end time.
    m : float
        Specimen mass in mg.

    Methods
    -------
    calculate_cp(baseline, sapphire, cp_sapphire)
        Calculate heat capacity based on Segment data.

    """

    def __init__(self, data: matrix, name: str = "", mass: float = 0.0):
        """Initialize Segment instance.

        Parameters
        ----------
        data : np.ndarray(dim=2, dtype=float)
            Matrix with:
              - Time [s]
              - Temperature [°C]
              - Heat flow [W]
              - Normalized heat flow [W/g]
        name : str (defaults to "")
            Description of the segment.
        mass : float (defaults to 0.0)
            Specimen mass in mg.

        """
        self.name = name
        self.data = data
        self.mass = mass
        self.N = len(data)
        self.t_interval = (data[0, 0], data[-1, 0])

    def calculate_cp(self, baseline: Segment, sapphire: Segment,
                     T: Optional[vector] = None,
                     cp_sapphire: Callable = CP_SAPPHIRE) -> matrix:
        """Calculate specific heat.

        Parameters
        ----------
        baseline : Segment
            Baseline DSC data.
        sapphire : Segment
            Sapphire DSC data.
        T : np.ndarray(dim=1, dtype=float), optional (defaults to None)
            Temperature range to calculate specific heat. If not provided, the
            temperature range will equal to `np.arange(T_min+5, T_max-5)`.
        cp_sapphire : Callable, optional (defaults to CP_SAPPHIRE)
            Function that returns cp of sapphire at T.

        Returns
        -------
        cp : np.ndarray(dim=2, dtype=float)
            Matrix with temperature and specific heat.

        """
        if T is None:
            T_min = self.data[:, 1].min().round() + 5
            T_max = self.data[:, 1].max().round() - 5
            T = np.arange(T_min, T_max)
        Dst = sapphire._interp_H(T) - baseline._interp_H(T)
        Ds = self._interp_H(T) - baseline._interp_H(T)
        cp = cp_sapphire(T) * (Ds/Dst) * (sapphire.mass/self.mass)
        return np.column_stack((T, cp))

    def calculate_enthalpy(self) -> float:
        """Return
        """
        if self.baseline is None:
            raise RuntimeError("Baseline not yet defined")
        else:
            pass

    def baseline_from_lims(self, lims: tuple[int, int]):
        """
        """
        self.baseline = Baseline(self.data, lims)

    def _interp_H(self, T: float) -> float:
        """Return heat flow at T based on linear interpolation.

        Parameter
        ---------
        T : float | np.ndarray(dim=1 dtype=float)
            Temperature in °C.

        Return
        ------
        H : np.ndarray(dim=1, dtype=float)
            Heat flow at T in mW.

        """
        if all(np.diff(self.data[:, 1]) > 0):
            return np.interp(T, self.data[:, 1], self.data[:, 2])
        elif all(np.diff(self.data[:, 1]) < 0):
            return np.interp(T, self.data[::-1, 1], self.data[::-1, 2])
        else:
            raise Exception("Temperature data must be strictly " +
                            "increasing or decreasing for interpolation.")

    def __str__(self) -> str:
        """Return string for printing."""
        meta = (f"\n{self.name}\n" +
                "--------------------------------\n" +
                f"Number of data points:   {self.N:>5}\n" +
                f"Start time:              {int(self.t_interval[0]):>5} s\n"
                f"End time:                {int(self.t_interval[1]):>5} s\n\n")
        hdr = "Data\n" + 55*"-" + "\n"
        return (meta + hdr + self.data.__str__())

    def __getitem__(self, i):
        """Return segment using index."""
        return self.data[i]


class Baseline:
    """Class to represent a baseline.

    For now only a linear baseline is implemented (which does follow
    the ASTM standard). More sophisticated baselines, e.g. based on
    splines, can of course be added.

    Attributes
    ----------
    lims : (int, int)
        Indices to indicate start and end of the baseline window.
    func : Callable(t: float) -> float
        Function that returns normalized heat flow as function of time.

    Methods
    -------
    __call__(t: float) -> flaot
        The class can be called as a function that provides the
        interpolated normalized heat flow at the provided time(s) `t`.

    """
    def __init__(self, data: matrix, lims: tuple):
        """Initialize Baseline instance.

        Parameters
        ----------
        data : np.ndarray(dim=2, dtype=float)
            Matrix with:
              - Time [s]
              - Temperature [°C]
              - Heat flow [W]
              - Normalized heat flow [W/g]
        lims : (int, int)
            Indices to indicate start and end of linear baseline.

        """
        self.lims = None  # limit of the baseline window, also end
                          # points for linear interpolation
        self.func = self._linear_baseline(data, lims)

    def _linear_baseline(self, data: matrix, lims: tuple) -> Callable:
        """Return linear interpolation function."""
        t_ = data[lims[0], 0], data[lims[1], 0]
        H = data[lims[0], 3], data[lims[1], 3]
        def func(t):
            return np.interp(t, t_, H)
        return func

    def __call__(self, t: float) -> float:
        return self.func(t)
