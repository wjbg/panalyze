"""
Contains core of panalyze: Measurement, Segment, Baseline

"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO
import importlib.resources as impresources
from . import materials

vector = matrix = npt.NDArray[np.float_]  # Annotation shorthand

# Sapphire heat capacity (there is probably a more elegant way to do this)
fn = impresources.files(materials) / 'sapphire_cp.csv'
CP_SAPPHIRE_DATA = np.loadtxt(fn, dtype=np.float32, delimiter=",",
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
    data : np.ndarray(dim=2, dtype=float)
        Matrix with DSC data in the following columns:
          - Time [s]
          - Temperature [°C]
          - Heat flow [mW]
          - Normalized heat flow [J/g]
    baseline : np.ndarray(dim=2, dtype=float) (optional)
        Matrix with baseline data.
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
        Return heat capacity based on Segment data.
    calculate_enthalpy()
        Return enthalpy; required definition of a baseline.
    calculate_normalized_enthalpy()
        Return normalized enthalpy; required definition of a baseline.

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

    def calculate_normalized_enthalpy(self) -> float:
        """Return normalized enthalpy of peak using defined Baseline."""
        if self.baseline is None:
            raise TypeError("Baseline function is not yet defined")
        lims = [np.nonzero(self.data[:, 0] == self.baseline[0, 0])[0][0],
                np.nonzero(self.data[:, 0] == self.baseline[-1, 0])[0][0] + 1]
        ix = np.arange(*lims)
        peak = self.data[ix, 3] - self.baseline[:, 3]
        h_tot = abs(trapezoid(peak, self.baseline[:, 0]))
        return h_tot

    def calculate_enthalpy(self) -> float:
        """Return normalized enthalpy of peak using defined Baseline."""
        if self.baseline is None:
            raise TypeError("Baseline function is not yet defined")
        lims = [np.nonzero(self.data[:, 0] == self.baseline[0, 0])[0][0],
                np.nonzero(self.data[:, 0] == self.baseline[-1, 0])[0][0] + 1]
        ix = np.arange(*lims)
        peak = self.data[ix, 2] - self.baseline[:, 2]
        H_tot = abs(trapezoid(peak, self.baseline[:, 0]))
        return H_tot

    def degree_of_conversion(self) -> vector:
        """Return degree of conversion using defined baseline."""
        pass

    def create_linear_baseline(self, knots: list[int, int],
                               lims: Optional[list[int, int]] = None):
        """Create linear baseline using start and end indices.

        Parameters
        ----------
        knots : List[int, int]
            Indices of the two points that define linear baseline.
        lims : List[int, int] (optional)
            Indices that indicate the limits of the baseline. If not provided
            the limit indices equals the knot indices.

        """
        if lims is None: lims = knots
        baseline = self.data[lims[0]:lims[1], :].copy()
        tp = self.data[knots, 0]  # Time values at two knots
        for i in [2, 3]:  # Heat flow & Normalized heat flow
            yp = self.data[knots, i]
            fit = np.polyfit(tp, yp, 1)
            baseline[:, i] = np.polyval(fit, baseline[:, 0])
        self.baseline = baseline

    def plot(self, x: str = "t", y: str = "h",
             ax: Optional[plt.Axes] = None,
             plot_bline: Optional[bool] = False) -> tuple[plt.Axes, list]:
        """Plot data in a Figure.

        Parameters
        ----------
        x : str (defaults to "t" for time)
            Data to plot on x-axis. Options include:
            - "t" : time
            - "T" : Temperature
        y : str (defaults to "h" for normalized heat flow)
            - "T" : Temperature
            - "H" : Heat flow
            - "h" : Normailzed heat flow
        ax : plt.Axes (optional)
            Axes for plotting.
        plot_bline : bool (optional, defaults to None)
            Plots baseline if True.

        Returns
        -------
        ax : plt.Axes
            Axes handle.
        lines : list with plt.lines.Line2d
            Line handles.

        """
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots()
        if isinstance(ax, plt.Axes):
            idx = {"t": 0, "T": 1, "H": 2, "h": 3}
            line = ax.plot(self.data[:, idx[x]], self.data[:, idx[y]])
            if plot_bline:
                ax.plot(self.baseline[:, idx[x]], self.baseline[:, idx[y]]))
        return ax, lines

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
