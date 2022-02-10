import numpy as np
import mne
import scot
import mne_connectivity as mnecon
from epoch_connectivity.conn_epoch import EpochConnection
from epoch_connectivity.utils import only_EEG_channels
import scipy


"""

Object oriented API to integrate Scot with the project and speedup the process
of transforming EEGs into connectivity matrices

"""

class EEGConnection(object):

    def __init__(self, epoched_eeg):
        #self.epoched_eeg = epoched_eeg
        self.ch_names = only_EEG_channels(epoched_eeg.ch_names)
        self.sfreq = epoched_eeg.info['sfreq']

        self.epoch_matrix = epoched_eeg.get_data(picks=['eeg'])
        self.n_epochs = len(self.epoch_matrix)


    def fit_MVAR_on_epochs(self, model_order=20):
        """
        Fit on MVAR on each epoch. The MVAR order is calculated for each epoch. The ScoT library fits
        the model with defined order and optimize the regularization.

        -----------
        PARAMETERS
        -----------
        model_order (int | None) : MVAR model order to be used, if None finds the best order

        -----------
        RETURNS
        -----------
        None
        
        """
        varx = scot.var.VAR(model_order=model_order)
        self.result = scot.varica.mvarica(x=self.epoch_matrix, var=varx, reducedim='no_pca', optimize_var=True, varfit='trial')

    
    def integrated_measure_epochs(self, measure_name, frequency_band = None, nfft = None):
        """
        Calculate connectivity metrics (the MVAR model must be fitted first)

        -----------
        PARAMETERS
        -----------
        measure_name (str) : name of the measure to be used {'dDTF', 'DTF', 'PDC'}
        frequency_band (list) : frequency band where the measure will be integrated if None
          then the measure is integrated over all the frequencies (broadband value) the list
          has the format [lower_bound, higher_bound]
        nfft (int) : number of frequency bins on the metrics resultant matrix if None then
          nfft = 2 * sfreq

        -----------
        RETURNS
        -----------
        connectivity matrix
        """
        if nfft is None:
            nfft = 2 * self.sfreq

        self.conn = scot.Connectivity(b = self.result.a.coef, c = self.result.a.rescov, nfft=nfft)

        if measure_name == 'dDTF':
            measure = self.conn.dDTF()
        elif measure_name == 'DTF':
            measure = self.conn.DTF()
        elif measure_name == 'PDC':
            measure = self.conn.PDC()
        
        x = np.linspace(0, self.sfreq/2, nfft)

        freq_resolution = (self.sfreq/2)/(nfft - 1)
        i, j = 0, nfft

        if frequency_band is not None:
            low, high = frequency_band[0], frequency_band[1]
            i, j = int(low/freq_resolution), int(high/freq_resolution)

        integrated_measure = scipy.integrate.simps(y=measure[:,:,i:j+1], x=x[i:j+1])
        
        return integrated_measure
        
