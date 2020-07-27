"""
This module deals with LAT analysis: calculus, statistics, and data visualization.
"""

from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.signal import argrelextrema
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import trange, tqdm
from numba import jit, njit
import seaborn as sns
from functools import lru_cache

from pathlib import Path
import pickle
import datetime
import os

from IPython.display import Math, Latex
from matplotlib.ticker import FormatStrFormatter

import logging
logging.basicConfig(level=logging.WARNING)


class Colname:
    """Provide unified column names"""
    ENERGY = 'ENERGY'
    EIN = 'Ein'
    TIME = 'TIME'
    TOBS = 'Tobs'
    KAPPA = 'kappa'
    TPEAK = 'Tpeak'
    DTOBS = 'DTobs'
    DTTSF = 'DTtsf'
    NAME = 'NAME'
    KAPPA = 'kappa'
    Z = 'z'
    COLS = ['ENERGY', 'TIME']


class Rand(Colname):
    """Generate random samples.

    Parameters
    ----------
    Set : astropy.table.table.Table
        The set of GRBs.
    Emax : float
        Maximum of the energy range.
    Emin : float
        Minimum of the energy range.
    power : float
        The parameter in the power law.
    E_LVs : numpy.ndarray
        Candidate Lorentz Violation parameters.
    rho : int, optional
        Constant rho in the test function, by default 5.
    repeat : int, optional
        Size of random samples, by default 100000.
    E_LV_sz : int, optional
        Size of random candidate Lorentz Violation parameters, by default 1000.
    """
    @staticmethod
    def Calc_kappa(Eobs_repeat, z_repeat, Omega_m: float=0.315):
        r"""Return kappa caculated from observed energy and red-shift.

        Parameters
        ----------
        Eobs_repeat : numpy.ndarray
            Observed energy shape in (n, m), where n is the number of photon events, and m is the repeat time (m = 1 for non-repeat).
        z_repeat : numpy.ndarray
            red shift of the same shape as Eobs_repeat.
        Omega_m : float, optional
            matter density parameter in the :math:`\Lambda\text{CDM}` model, by default 0.315, from [2]_.

        Returns
        -------
        numpy.ndarray
            kappa of the same shape as z_repeat (and Eobs_repeat).

        Note
        ----
        .. math::
            :nowrap:

            \begin{equation}
                \kappa = s \frac{E_\text{h} - E_\text{l}}{H_0} \frac{1}{(1+z)} \int_0^z \frac{(1 + z') \mathrm{d} z'}{\sqrt{\Omega_m(1 + z')^3 + \Omega_\Lambda}}
            \end{equation}

        See [2]_.

        References
        ----------
        .. [1] Planck Collaboration, Aghanim, N., Akrami, Y., Ashdown, M., Aumont, J., Baccigalupi, C., Ballardini, M., Banday, A. J., Barreiro, R. B., Bartolo, N., Basak, S., Battye, R., Benabed, K., Bernard, J.-P., Bersanelli, M., Bielewicz, P., Bock, J. J., Bond, J. R., Borrill, J., … Zonca, A. (2020). Planck 2018 results. VI. Cosmological parameters. Astronomy & Astrophysics. https://doi.org/10.1051/0004-6361/201833910
        .. [2] Xu, H., & Ma, B.-Q. (2018). Regularity of high energy photon events from gamma ray bursts. Journal of Cosmology and Astroparticle Physics, 2018(01), 050–050. https://doi.org/10.1088/1475-7516/2018/01/050
        """
        KAPPA: str = Colname.KAPPA
        Z: str = Colname.Z
        ELOW = 0. * u.GeV
        H0 = (67.4 * u.km / u.s / u.Mpc).to(1/u.s)
        s: float = 1.
        Omega_L: float = 1 - Omega_m
        
        
        factor1 = s * (Eobs_repeat - ELOW) / H0
    
        @jit
        def integrand (z):
            return (1 + z) / np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

        @lru_cache(maxsize=50)
        @jit(forceobj=True)
        def _kappa_z(z):
            return integrate.quad(integrand, 0, z)[0] / (1 + z)

        @jit(forceobj=True, parallel=True)
        def kappa_z(z_repeat):
            shape = z_repeat.shape
            factor2 = np.zeros_like(z_repeat)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    factor2[i, j] = _kappa_z(z_repeat[i, j])
            return factor2
        
        factor2 = kappa_z(z_repeat)
        kappa = factor1 * factor2
        return kappa
    
    @staticmethod
    def Calc_Ts(row_len: int, sz: int, rho: int, DTtsf, kappa, E_LVs):
        """Return the test function T of shape (row_len, sz).

        Parameters
        ----------
        row_len : int
            Number of total candidate Lorentz Violation parameters; also T.shape[0].
        sz : int
            Number of photon events; also T.shape[1].
        rho : int
            constant rho in the test function T.
        DTtsf : numpy.ndarray
            observed time over (1 + z), i.e., :math:`\Delta T_{obs} / (1 + z)`.
        kappa : numpy.ndarray
            kappa of the same shape as DTtsf.
        E_LVs : numpy.ndarray
            candidate Lorentz Violation parameters

        Returns
        -------
        numpy.ndarray
            Values of the test function
        """        
#         @lru_cache(maxsize=50)    # unhashable type: np.ndarray
        @jit(forceobj=True)
        def calcT(N, rho, DTtsf, kappa, E_LV):
            l = N - rho
            t_in = DTtsf - kappa / E_LV
            t_in.sort()        # sort ascending
            tau = np.sum(t_in[:, -rho:] - t_in[:, :rho], axis=1) / l
            res1 = np.log(tau)
            res2 = np.sum(np.log(t_in[:, rho:] - t_in[:, :-rho]), axis=1) / l
            result = res1 - res2
            return result

        @jit(forceobj=True, parallel=True)        
        def _calcTs(row_len: int, sz: int, rho: int, DTtsf, kappa, E_LVs):
            N = E_LVs.size
            Ts = np.zeros([row_len, N])
            for i in range(N):
                Ts[:, i] = calcT(sz, rho, DTtsf, kappa, E_LVs[i])
            return Ts
        
        Ts = _calcTs(row_len, sz, rho, DTtsf, kappa, E_LVs)
        return Ts

    def __init__(self, Set, Emax: float, Emin: float, power: float, E_LVs, rho: int=5, repeat: int=100000, E_LV_sz: int=1000):
        Set_np = Set[['z', self.DTOBS]].to_pandas().to_numpy()
        z = Set_np[:, 0]        
        DTobs = Set_np[:, 1]
        sz = z.size
        
        @njit
        def Rand_DTtsf(DTtsf):
            return np.random.permutation(DTtsf)

        @jit(forceobj=True)
        def Repeat_ndarray(a, repeat: int):
            '''Repeat ndarray repeat times'''
            l = a.tolist()
            return l * repeat    

        @jit(forceobj=True, parallel=True)
        def Get_DTtsf(DTtsf, z, repeat):
            rand_DTtsf = np.zeros([repeat, DTtsf.shape[0]])

            for i in range(repeat):
                rand_DTtsf[i] = Rand_DTtsf(DTtsf)

            return rand_DTtsf
        
        @jit(forceobj=True)
        def Rand_E(Emax: float, Emin: float, power: float, E_LV_sz: int, sz: list):
            E = np.random.random(E_LV_sz) * (Emax - Emin) + Emin
            p = np.power(E, power)
            p /= p.sum()
            E_in_repeat = np.random.choice(E, size=sz, p=p)
            return E_in_repeat
            
        self.E_LVs = E_LVs

        DTtsf = DTobs / (1 + z)
        kappa = Set[self.KAPPA]        
        l_E_LV = E_LVs.size
        
        rand_DTtsf = Get_DTtsf(DTtsf, z, repeat) * u.s
        
        E_in_repeat = Rand_E(Emax=Emax, Emin=Emin, power=power, E_LV_sz=E_LV_sz, sz=[repeat, sz]) * u.GeV
        z_repeat = np.array(Repeat_ndarray(z, repeat)).reshape(E_in_repeat.shape)
        
        Eobs_repeat = E_in_repeat / (1 + z_repeat)
        
        E_LV_repeat = Repeat_ndarray(E_LVs.value, repeat) * u.GeV
        
        rand_kappa = Rand.Calc_kappa(Eobs_repeat, z_repeat)
            
        Calc_Ts = Rand.Calc_Ts
        rand_T = Calc_Ts(repeat, sz, rho, rand_DTtsf.value, rand_kappa.value, E_LVs.value)
        
        rand_set = pd.DataFrame({'T': np.array(rand_T).flatten(),
                                'E_LV': np.array(E_LV_repeat)})
        grouped = rand_set.groupby('E_LV')['T']
        TE = grouped.agg(['mean', 'std'])
        sigmas = [0.682689, 0.997300, 0.999999]
        
        self.ci = [stats.norm.interval(sigma, loc=TE['mean'], scale=TE['std'])
                   for sigma in sigmas]

    def randPlot(self):
        """Plot n-sigma regions, where n's are 1, 3, 5.
        """        
        plt.fill_between(self.E_LVs, self.ci[2][0], self.ci[2][1], alpha=0.4, label=r'$5 \sigma$')
        plt.fill_between(self.E_LVs, self.ci[1][0], self.ci[1][1], alpha=0.6, label=r'$3 \sigma$')
        plt.fill_between(self.E_LVs, self.ci[0][0], self.ci[0][1], alpha=0.8, label=r'$1 \sigma$')

class FITSLoad(Colname):
    """Load data of given GRBs from in_dir.

    Parameters
    ----------
    grbs : pandas.core.frame.DataFrame
        Information of GRBs, including GCNNAME, ENERGY and TIME.
    in_dir : pathlib.Path or str
        Input directory that contains GRB FITS files.
    """
    def __init__(self, grbs, in_dir):
        @jit(forceobj=True)
        def fileNames(grbs, in_dir):
            files = []
            for row, grb in grbs.iterrows():
                name_digit = grb['GCNNAME'][:6]
                files.extend([name_digit + "/" + f.rsplit("/", 1)[1] for f in grb['urls']])

            event_dir = Path(in_dir)
            ph_all = [event_dir / event for event in files if 'PH' in event]
            return ph_all
        
        def Filter(ph, period: float=90.):
            '''period: 90 seconds'''
            # ? or better to add to units?
            energy_filter = lambda ph: ph[ph[self.EIN] >= 1e3]    # >= 1e3 MeV, or 1GeV
            time_filter = lambda ph: ph[(ph[self.DTOBS]<=period) & (ph[self.DTOBS]>=0)]
            ph = energy_filter(ph)    # >= 1e3 MeV, or 1GeV
            ph = time_filter(ph)
                        
            return ph

        def PH_DF(ph):
            name_index = str(ph).rsplit('/', 2)[1]

            with fits.open(ph) as f:
                data = f[1].data
            if data.size == 0:
                return pd.DataFrame()

            ph_table = Table(data)
            ph_table.keep_columns(self.COLS)
            ph_df = ph_table.to_pandas()

            tpeak = grbs.at[name_index, 'tpeak_ref']
            trigger_time = grbs.at[name_index, 'GBM_MET']

            # ? not sure if this is faster than 'df[] =' assignment
            ph_df = ph_df.assign(**{self.NAME: name_index,
                                    Z: grbs.at[name_index, Z]})
            # ph_df = ph_df.assign(**{self.EIN: (1 + ph_df.z) * ph_df.ENERGY,
                                    # self.TOBS: ph_df.TIME - trigger_time})
            ph_df = ph_df.assign(**{self.DTOBS: ph_df.Tobs - tpeak})
            ph_df = Filter(ph_df)

            return ph_df
        
        # ! AVOID use jit here, which would result in fewer photon events!
        def collectPH(ph_all: list):
            # ! assume that ph_all[0] contains valid photon events
            ph_df_all = PH_DF(ph_all[0])
            
            for ph in ph_all[1:]:
                ph_df = PH_DF(ph)
                if not ph_df.empty:
                    ph_df_all = ph_df_all.append(ph_df, ignore_index=True)
                    
            return ph_df_all
        
        ph_all = fileNames(grbs, in_dir)
        ph_df_all = collectPH(ph_all)
        self.ph_table_all = Table.from_pandas(ph_df_all)



class PHTable(FITSLoad, Rand):
    """Load GRBs and do calculations and convertings.

    Parameters
    ----------
    grbs : pandas.core.frame.DataFrame
        Information of GRBs, including GCNNAME, ENERGY and TIME.
    in_dir : pathlib.Path or str
        Input directory that contains GRB FITS files.
    erange : astropy.units.quantity.Quantity
        Array of 5 end points for 4 energy ranges.
    """
    def __init__(self, grbs, in_dir, erange=[1., 10., 20., 40., 150.] * u.GeV):
        def tableCalc(ph_table_all):
            '''Convert MeV to GeV of ENERGY'''
            Calc_kappa = Rand.Calc_kappa

            ph_table_all[self.TIME] *= u.s
            # ph_table_all[self.TOBS] *= u.s
            ph_table_all[self.DTOBS] *= u.s
            ph_table_all[self.ENERGY] *= u.MeV
            ph_table_all[self.EIN] *= u.MeV
            ph_table_all[self.ENERGY].convert_unit_to(u.GeV)
            ph_table_all[self.EIN].convert_unit_to(u.GeV)

            sz = len(ph_table_all)
            Eobs_repeat = ph_table_all[self.ENERGY].reshape([1, sz])
            z_repeat = ph_table_all[self.Z].reshape([1, sz])

            # ph_table_all[DTTSF] = ph_table_all[self.DTOBS] / (1 + ph_table_all[self.Z])
            ph_table_all[self.KAPPA] = Calc_kappa(Eobs_repeat=Eobs_repeat, z_repeat=z_repeat)[0]

        FITSLoad.__init__(self, grbs, in_dir)
        ph_table_all = self.ph_table_all
        tableCalc(ph_table_all)
        
        non_missing = set(ph_table_all[self.NAME])
        self.missing = set(grbs.index).difference(non_missing)
        self.erange = erange
        self.POWER = np.nan
        
        self.table = ph_table_all
        self.classI  = ph_table_all[ph_table_all[self.EIN] <  erange[1]]
        self.classII = ph_table_all[ph_table_all[self.EIN] >= erange[1]]
        self.classIII = ph_table_all[ph_table_all[self.EIN]>= erange[2]]
        self.classIV = ph_table_all[ph_table_all[self.EIN] >= erange[3]]
        self.classIV_less = self.classIV.copy()
        
        self.E_LVs = np.logspace(15, 20, num=1000) * u.GeV
                
    def _repr_html_(self):
        return self.table._repr_html_()
        
    def save(self, out_dir: str='../data/', fname: str='ph-table-all', date: bool=True):
        """Save the instance as pickle format, in order to reuse the instance without repeated calculations.

        Parameters
        ----------
        out_dir : str, optional
            Output directory, by default '../data/'
        fname : str, optional
            Output file name, by default 'ph-table-all'
        date : bool, optional
            Whether to include date in the file name, by default True
        """        
        filename = out_dir + fname
        if date:
            filename += '-' + str(datetime.date.today())

        with open(filename) as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        try:
            logging.info(f'Table saved as: {filename}')
            print(f'Table saved as: {filename}')
        except Exception as e:
            print(e)
            
    def regPlot(self, binmethod=np.logspace(0., 1.0), figname: str=''):
        """Plot the histogram of Data Set I and the regression line.

        Parameters
        ----------
        binmethod : numpy.ndarray, optional
            Bin method of the histogram plot, by default np.logspace(0., 1.0).
        figname : str, optional
            Output file name of the figure, by default '' (and not output to file).
        """        
        fig, ax = plt.subplots()
        weight = binmethod[1:] - binmethod[:-1]
        # ‘doane’: An improved version of Sturges’ estimator that works better with non-normal datasets.
        # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        hist, bin_edges = np.histogram(self.classI[self.EIN], bins=binmethod)

        null = [0]
        n = hist / (bin_edges[1:] - bin_edges[:-1])
        x = np.concatenate([null, bin_edges])
        y = np.concatenate([null, n, null])
        ax.fill_between(x, y, step='post', alpha=0.25)

        # Following https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
        E_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        fit_func = lambda x, a, b: a * x ** b
        popt, pcov = curve_fit(fit_func, E_center, n)

        plt.xscale("log")
        plt.yscale('log')
        plt.xlabel(r"$E_{in}$ (GeV)")
        plt.ylabel(r"counts / $\Delta E_\mathrm{in}$ (1/GeV)")
        plt.ylim(ymin=1.5)
#         ax.plot(E_center, n, 'x', label=label)

        a = popt[0]
        b = popt[1]
        fit_line = r"fit line: $\log_{{10}} (y) = {b:.2f} \log_{{10}} (x) + {a:.2f}$".format(b=b, a=a)
        ax.plot(bin_edges, fit_func(bin_edges, *popt), label=fit_line)
        sns.scatterplot(E_center, n, label='upper center of each histogram')
        
        # Folowing https://stackoverflow.com/questions/12493809/minor-ticks-on-logarithmic-axis-in-matplotlib
        ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.legend()

        if figname != '':
            os.makedirs(figname, exist_ok=True)
            plt.savefig(figname)
        
        self.POWER = popt[1]
        return fig, popt, pcov
    
    def scatterPlot_(self, ph_class, label: str, alpha: float=0.75):
        """Individual scatter plot for each data set.

        Parameters
        ----------
        ph_class : astropy.table.table.Table
            All photon events.
        label : str
            Label in the plot.
        alpha : float, optional
            Alpha of the plot, by default 0.75

        Returns
        -------
        matplotlib.figure.Figure
            :math:`\Delta t_{obs} / (1 + z)` versus :math:`\kappa` plot.
        """        
        fig, ax = plt.subplots()
        
        ax.scatter(x=ph_class[self.KAPPA], y=ph_class[self.DTTSF], label=label, alpha=alpha)
        ax.set_xlabel(r'$\kappa$ (s $\cdot$ GeV)')
        ax.set_ylabel(r'$\Delta \tau_z$ (s)')
        ax.legend()
        plt.tight_layout()

        return fig

    def scatterPlots(self, labels = ['Data Set ' + i for i in ['I', 'II', 'III', 'IV', "IV'"]]):        
        """Scatter plots of all data sets.

        Parameters
        ----------
        labels : list of str, optional
            Labels in the plots, by default ['Data Set ' + i for i in ['I', 'II', 'III', 'IV', "IV'"]]

        Returns
        -------
        matplotlib.figure.Figure
            :math:`\Delta t_{obs} / (1 + z)` versus :math:`\kappa` plots.

        See Also
        --------
        scatterPlot_ : Individual scatter plot for each data set.
        """        
        figs = []
        figs.append(self.scatterPlot_(self.classI, labels[0], alpha=0.5))
        figs.append(self.scatterPlot_(self.classII, labels[1]))
        figs.append(self.scatterPlot_(self.classIII, labels[2]))
        figs.append(self.scatterPlot_(self.classIV, labels[3]))
        figs.append(self.scatterPlot_(self.classIV_less, labels[4]))
        return figs
            
    def checkPower(self) -> bool:        
        r"""Check if the power has been assigned.

        Returns
        -------
        bool
            True if self.POWER has been assigned; False otherwide.

        Note
        ----
        Power Law:

        .. math::
        
            \frac{\mathrm{d} N}{\mathrm{d} E} \propto \left( \frac{E}{E_\text{ref}} \right)^{-\alpha}, \\

            E_\text{ref} = 1.0 \text{GeV}
        """        
        if self.POWER is np.nan:
            print("POWER not available! Please run .regplot() first.")
            return False
        else:
            return True
        
    def T(self, Set, rho: int, return_extr: bool=False):
        r"""Calculate the test function.

        Parameters
        ----------
        Set : astropy.table.table.Table
            The set of all photon events.
        rho : int
            Constant parameter in the test function.
        return_extr : bool, optional
            Whether to return the extrema, by default False.

        Returns
        -------
        numpy.ndarray
            values of the test function.

        Note
        ----
        .. math::

            T_\rho = \frac{\sum_{i=1}^{N-\rho} \log[\bar{t}_\rho / (t_{i+\rho} - t_i)]}{N - \rho}, \\

            \bar{t}_\rho = \frac{\sum_{i=1}^{N-\rho} (t_{t+\rho} - t_i)}{N - \rho}.

        See Also
        --------
        Rand.Calc_Ts : Return the test function T of shape (row_len, sz).
        """        
        Set_np = Set[[self.DTTSF, self.KAPPA]].to_pandas().to_numpy()
        DTtsf = Set_np[:, 0]
        kappa = Set_np[:, 1]
        sz = kappa.size
        
        def _T(sz, rho, DTtsf, kappa, E_LVs):
            Calc_Ts = Rand.Calc_Ts
            Ts = Calc_Ts(1, sz, rho, DTtsf.reshape([1, sz]), kappa.reshape([1, sz]), E_LVs)[0]
            return Ts
        
        Ts = _T(sz, rho, DTtsf, kappa, self.E_LVs.value)
        
        if return_extr:
            E_LV_ind = argrelextrema(Ts, np.greater)
            E_LV = self.E_LVs[E_LV_ind]
            eps_p = self.E_LVs[E_LV_ind[0]+1] - self.E_LVs[E_LV_ind[0]]
            eps_m = self.E_LVs[E_LV_ind[0]-1] - self.E_LVs[E_LV_ind[0]]
            
            return (Ts, {'E_LV': E_LV, 'eps': [eps_p, eps_m]})
        else:
            return (Ts,)
        
    def TEPlot(self, Set, rho: int, repeat: int, Emax: float, Emin: float, label: str, return_extr: bool):
        """Test function versus Energy plot.

        Parameters
        ----------
        Set : astropy.table.table.Table
            All photon events.
        rho : int
            Constant parameter in the test function.
        repeat : int
            Number of random samples.
        Emax : float
            Maximum of the energy range.
        Emin : float
            Minimum of the energy range.
        label : str
            Label in the T-E plot.
        return_extr : bool
            Whether to return the extrama.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, None or dict)
            return fig, and dict of extrama (if any).

        See Also
        --------
        T : Test function.
        """        
        if not self.checkPower():
            return np.nan
        
        Ts = self.T(Set, rho, return_extr)
        rand = Rand(Set, Emax=Emax, Emin=Emin, E_LVs=self.E_LVs,
                    power=self.POWER, rho=rho, repeat=repeat)
        
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$E_\mathrm{LV}$ (GeV)')
        ax.set_ylabel(r'$T$')
        ax.set_xscale('log')
        rand.randPlot()
        ax.plot(self.E_LVs, Ts[0], label=label)
        plt.legend()
        plt.tight_layout()
        
        return (fig, Ts[1:])

    def randPlotI(self, rho: int=5, repeat: int=100000, label: str='Data Set I', return_extr: bool=False):
        """T versus E plot for Data Set I.

        Parameters
        ----------
        rho : int, optional
            Constant parameter in the test function, by default 5.
        repeat : int, optional
            Number of random samples, by default 100000.
        label : str, optional
            Label in the plot, by default 'Data Set I'.
        return_extr : bool, optional
            Whether to return the extrama, by default False.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, None or dict)
            return fig, and dict of extrama (if any).

        See Also
        --------
        TEPlot : T versus E plot for individual data set.
        """        
        Emax = self.erange[1]
        Emin = self.erange[0]
        Set = self.classI
        return self.TEPlot(Set=Set, rho=rho, repeat=repeat, Emax=Emax, Emin=Emin, label=label, return_extr=return_extr)
            
    def randPlotII(self, rho: int=5, repeat: int=100000, label: str='Data Set II', return_extr: bool=False):
        """T versus E plot for Data Set II.

        Parameters
        ----------
        rho : int, optional
            Constant parameter in the test function, by default 5.
        repeat : int, optional
            Number of random samples, by default 100000.
        label : str, optional
            Label in the plot, by default 'Data Set II'.
        return_extr : bool, optional
            Whether to return the extrama, by default False.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, None or dict)
            return fig, and dict of extrama (if any).

        See Also
        --------
        TEPlot : T versus E plot for individual data set.
        """
        Emax = self.erange[4]
        Emin = self.erange[1]
        Set = self.classII
        return self.TEPlot(Set=Set, rho=rho, repeat=repeat, Emax=Emax, Emin=Emin, label=label, return_extr=return_extr)
            
    def randPlotIII(self, rho: int=5, repeat: int=100000, label: str='Data Set III', return_extr: bool=False):
        """T versus E plot for Data Set III.

        Parameters
        ----------
        rho : int, optional
            Constant parameter in the test function, by default 5.
        repeat : int, optional
            Number of random samples, by default 100000.
        label : str, optional
            Label in the plot, by default 'Data Set III'.
        return_extr : bool, optional
            Whether to return the extrama, by default False.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, None or dict)
            return fig, and dict of extrama (if any).

        See Also
        --------
        TEPlot : T versus E plot for individual data set.
        """
        Emax = self.erange[4]
        Emin = self.erange[2]
        Set = self.classIII
        return self.TEPlot(Set=Set, rho=rho, repeat=repeat, Emax=Emax, Emin=Emin, label=label, return_extr=return_extr)
        
    def randPlotIV(self, rho: int=5, repeat: int=100000,
                   label: str='Data Set IV',
                   less: bool=False, return_extr: bool=False):
        """T versus E plot for Data Set IV or IV' (IV_less).

        Parameters
        ----------
        rho : int, optional
            Constant parameter in the test function, by default 5.
        repeat : int, optional
            Number of random samples, by default 100000.
        label : str, optional
            Label in the plot, by default 'Data Set IV'.
        less : bool, optional
            Plot for Data Set IV_less or IV.
        return_extr : bool, optional
            Whether to return the extrama, by default False.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, None or dict)
            return fig, and dict of extrama (if any).

        See Also
        --------
        TEPlot : T versus E plot for individual data set.
        """
        if not self.checkPower():
            return np.nan
        
        Emax = self.erange[4]
        Emin = self.erange[3]
        if less:
            Set = self.classIV_less
        else:
            Set = self.classIV
            
        return self.TEPlot(Set=Set, rho=rho, repeat=repeat, Emax=Emax, Emin=Emin, label=label, return_extr=return_extr)