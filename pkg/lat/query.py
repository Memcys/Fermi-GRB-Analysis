# ## Query data

# Scientific Library

import numpy as np
import pandas as pd

# Requests Urls and Manupilate Files

from astropy.utils.data import download_files_in_parallel, download_file
from astroquery import fermi
from tqdm import tqdm
import requests
import shutil
import os
# import pathlib
import functools

# Logging

import logging
# logger_info = logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.WARNING)

# from retry import retry
from retry.api import retry_call

# See https://stackoverflow.com/questions/492519/timeout-on-a-function-call
import signal


class Download:
    MISSING = pd.NA
    NAME = 'GCNNAME'
    DONE = 'DONE'
    WAIT = 2
    TBUFF = 30.
    INFOPRE = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/QueryResults.cgi?id="
    TRIES = 3
    
    def __init__(self, grbs):
        self.grbs = grbs
            
    @staticmethod
    # Transform a method into a static method. A static method does not receive an implicit first argument.
    # https://docs.python.org/3/library/functions.html#staticmethod
    def Filename(path, sep="/"):
        """
        input:
            path: file path string
            sep: seprator for directories, default to "/" (as in Linux)

        return filename of the path without directories
        """
        return path.rsplit(sep, 1)[1]

    def GRB_record(self, row: int, col: str, value):
        '''Record information for the for the given grb'''
        try:
            self.grbs.at[row, col] = value
        except Exception as e:
            print("Fail to record: ", e)
            print(type(value), value)
        
    def Missing(self, row: int, col: str):
        res = pd.isna(self.grbs.at[row, col])
        return np.any(res)
    
    def Urls_resolve(self, row):
        '''Retrive urls from record'''
        try:
            urls = eval(self.grbs.at[row, 'urls'])
            return urls
        except Exception as e:
            print("Fail to resolve urls: ", e)
            print(self.grbs._repr_html_)
        
        
    '''
    functions for single DataFrame GRB
    '''
    def Query_url(self, row: int, period: float, E_MeV_i: float, E_MeV_f: float, trigtime: str, tpeak: str, timeout: float=-1.):
        """Query downloading urls
        input:
            grbs: (DataFrame) GRBs
            row: row index of the given GRB
            period: period after initial time in second
            E_MeV_i: start energy in MeV
            E_MeV_fs: end energy in MeV
            trigtime: mission elapsed time in second
            tpeak: first peak time in second
        """
        col = 'urls'
        timesys = 'MET'
        name = self.grbs.at[row, self.NAME]
        
        missing = self.Missing(row, col)
        if not missing:
            logging.info('{}query already done'.format(' ' * 9))
            return self.DONE
        
        grb_name = 'GRB' + name
        met_i = self.grbs.at[row, trigtime]
        delta_t = self.grbs.at[row, tpeak]
        met_f = met_i + delta_t + period    # "window of 90 seconds" as apears in XHW2018.
        
        start = met_i - self.TBUFF
        stop = met_f + self.TBUFF
        met = '{}, {}'.format(start, stop)
        E_MeV = '{}, {}'.format(E_MeV_i, E_MeV_f)
        
        if timeout > 0:
            signal.alarm(timeout)
        try:
            fits_urls = retry_call(
                fermi.FermiLAT.query_object,
                fargs=[grb_name],
                fkwargs={
                    'energyrange_MeV': E_MeV,
                    'obsdates': met,
                    'timesys': timesys},
                tries=self.TRIES)
        except Exception as e:
            self.GRB_record(row, col, self.MISSING)
            logging.warning('{}Query_url failed while receiving:\n{}'.format(' ' * 9, e))
            return self.MISSING
        
        #! save urls (list) as str; Please extract urls with eval() later
        self.GRB_record(row, col, str(fits_urls))
        print(self)
        logging.info('{}query finished'.format(' ' * 9))
#         return self.DONE
        
    def Download_fits(self, row: int, out_dir, timeout: float=-1.):
        """Download fits files provided in urls to out_dir
        
        return:
            DONE: if succeeded
            MISSING: if failed
        """
        urls = self.Urls_resolve(row)
#         urls = eval(self.grbs.at[row, 'urls'])
        col = 'fits'
        name = self.grbs.at[row, self.NAME]
        if not self.Missing(row, col):
            logging.info('{}fits already saved'.format(' ' * 9))
            return self.DONE
        
        if timeout > 0:
            signal.alarm(timeout)
#         for url in urls:
        try:
            file_list = retry_call(
                # astropy.utils.data.download_files_in_parallel
                download_files_in_parallel,
                fargs=[urls],
                tries=self.TRIES)
        except:
            try:
                file_list = []
                for url in urls:
                    file_list.append(
                        retry_call(
                        # astropy.utils.data.download_file
                        download_file,
                        fargs=[url],
                        tries=self.TRIES)
                    )
            except Exception as e:
                self.GRB_record(row, col, self.MISSING)
                logging.warning("{}while downloading fits got:\n{}".format(' ' * 9, e))
                print("urls failed to download: ", urls)
                return self.MISSING

        # Following https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
        os.makedirs(out_dir, exist_ok=True)

        for i, url in enumerate(urls):
            filename = self.Filename(url)
            filename = out_dir / filename
            # filename = out_dir + "/" + filename

            if filename.exists():
                continue

            try:
#                 filename2 = wget.download(url, out_dir)
                # Following https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
                shutil.copyfile(file_list[i], filename)
                logging.info(filename.as_posix() + ' saved')
#                 logging.info(filename + ' saved as ' + filename2)
            except Exception as e:
                logging.warning(e)
        
        self.GRB_record(row, col, self.DONE)
        return self.DONE
    
    def Download_info(self, row, out_dir, pre=INFOPRE, wait=WAIT, timeout: float=-1.):
        """Request query page in url, and save tables to out_dir
        
        return:
            DONE: if succeeded
            MISSING: if failed
        """
        col = 'info'
        if not self.Missing(row, col):
            name = self.grbs.at[row, self.NAME]
            logging.info('{}info already saved'.format(' ' * 9))
            return self.DONE
        
        urls = self.Urls_resolve(row)
#         urls = self.grbs.at[row, 'urls']
        try:
            url = urls[0]
        except:
            logging.info("{}urls missing".format(' ' * 9))
            return self.MISSING
        
        ID = self.Filename(url).split("_")[0]
        query_url = pre + ID
        wait_times = 0
        
        if timeout > 0:
            signal.alarm(timeout)
        try:
            r = retry_call(
                requests.get,
                fargs=[query_url],
                tries=self.TRIES)
        except:
            self.GRB_record(row, col, self.MISSING)
            logging.info("{}query page downloading failed".format(' ' * 9))
            return self.MISSING

        query_info = r.text
        dfs = pd.read_html(query_info)
        status = dfs[1]
        position_in_queue = status['Position in Queue']
        
        if any(position_in_queue != 'Query complete'):
            logging.info("{}Query incomplete.".format(' ' * 9))
            return self.MISSING
        
        else:        
            criteria = dfs[0]
            filename = out_dir / 'criteria.csv'
            # filename = out_dir + '/criteria.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            criteria.to_csv(filename)

            info = dfs[2]
            filename = out_dir / 'info.csv'
            # filename = out_dir + '/info.csv'
            info.to_csv(filename)
            self.GRB_record(row, col, self.DONE)
            logging.info("{}query page downloaded".format(' ' * 9))
            
            return self.DONE


class Query(Download):
    FAKE = 'fake'
    PERIOD = 90.
    EMIN = 1e2    # 1e3 MeV / 10, as 1 + z < 10
    EMAX = 5e5    # The highest energy available in LAT
    TIMESYS = 'MET'
    
    def __init__(self, grbs, out_dir, init=False, retry=True, timeout: float=-1.):
        self.grbs = grbs
        self.out_dir = out_dir
        self.init = init
        self.timeout = timeout

        if self.init != False:
            self.Reset(self.init)
            
        self.Main_loop(outer_dir=out_dir)
        
        if retry and np.sum(self._Which_missing()) > 0:
            logging.info("Querying for missing information")
            self.Requery()
    
    def _repr_html_(self):
        return self.grbs._repr_html_()
    
    def Row_index(self, name):
        '''Return index of the given name'''
        index_np = self.grbs[self.grbs[self.NAME] == name].index
        index = index_np[0]
        return index
            
    def _Which_missing(self):
        """print missing information"""
        urls = self.grbs['urls'].isna()
        fits = self.grbs['fits'].isna()
        info = self.grbs['info'].isna()
        
        where = functools.reduce(np.logical_or, [urls, fits, info])
        num = np.sum(where)
        
        if num > 0:
            print("{}\n{} GRB{} missing".format('-' * 15, num, 's' if num > 1 else ''))
            print("Please Run .Requery() with(out) .Reset(init) for several times.\nIf those do not help, please download missing files manually.")
        
        return where
    
    def Which_missing(self):
        return self.grbs[self._Which_missing()]
        
    def Main_loop(self, outer_dir):
        """main loop to download all required data"""        
        period = self.PERIOD
        E_MeV_i = self.EMIN
        E_MeV_f = self.EMAX
        timesys = self.TIMESYS
        row_index = self.grbs.index
        
        for row in tqdm(row_index):
            name = self.grbs.at[row, self.NAME]
            urls = self.grbs.at[row, 'urls']
            out_dir = outer_dir / name[:6]
            # out_dir = outer_dir + '/' + name[:6]
            timeout = self.timeout
            logging.info(name + ':')
#             status = self.Query_url(row=row, period=period, E_MeV_i=E_MeV_i, E_MeV_f=E_MeV_f, trigtime='GBM_MET', tpeak='tpeak_ref')
            self.Query_url(row=row, period=period, E_MeV_i=E_MeV_i, E_MeV_f=E_MeV_f, trigtime='GBM_MET', tpeak='tpeak_ref', timeout=timeout)
            status = self.grbs.at[row, 'urls']
        
            if status is not self.MISSING:
                self.Download_info(row=row, out_dir=out_dir, timeout=timeout)
                self.Download_fits(row=row, out_dir=out_dir, timeout=timeout)
        
        rows = self._Which_missing()
        if np.sum(rows) > 0:
            # pretty printing in jupyter-notebook, following https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
            display(self.grbs.loc[rows])
        else:
            logging.info("Congratulations! All information downloaded successfully")
    
    def Reset(self, init=False):
        """reset urls of grbs with missing fits or info"""
        rows = self._Which_missing() if init==False else self.grbs.index
        self.grbs.loc[rows, ('urls', 'fits', 'info')] = self.MISSING
        
    def Requery(self):
        """remove queried urls and run Main_loop for missing grbs"""
        self.Main_loop(outer_dir=self.out_dir)