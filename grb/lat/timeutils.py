"""
This module provides utils to convert UTCMET between UTC and Fermi Mission Elapsed Time (MET).
"""

from astropy.time import Time
import astropy.units as u

class UTCMET():
    """Convert time between UTC and Fermi MET.
    """    
    t0 = Time('2001-01-01 0:0:0')   # The origion of Fermi MET UTCMET

    @staticmethod
    def utc2met(utc):
        """Convert UTC time to Fermi MET.

        Parameters
        ----------
        utc : str, or list of str
            UTC time formated like `yyyy-mm-dd hh:mm:ss' or so.

        Returns
        -------
        astropy.units.quantity.Quantity
            Fermi MET time corresponds to the the input UTC time.
        
        Examples
        --------
        
        >>> utc = '2008-08-28T10:46:30.271448'
        >>> met = UTCMET.utc2met(utc)
        >>> met
        <Quantity 2.41613191e+08 s>
        >>> met.value
        241613191.27144802
        """
        t0 = UTCMET.t0
        t1 = Time(utc)
        dt = (t1 - t0).to(u.s)
        return dt

    @staticmethod
    def met2utc(met):
        """Convert Fermi MET to UTC time.

        Parameters
        ----------
        met : float or array-like
            Fermi Mission Elapsed Time in second.

        Returns
        -------
        astropy.time.core.Time
            Time object correspond to the input MET.
        
        Examples
        --------

        >>> met = 488587220.275348
        >>> utc = UTCMET.met2utc(met)
        >>> utc
        <Time object: scale='utc' format='iso' value=2016-06-25 22:40:16.275>

        To retrive the value:
        >>> utc.value
        '2016-06-25 22:40:16.275'
        """
        t0 = UTCMET.t0
        dt = met * u.s
        t1 = t0 + dt
        return t1