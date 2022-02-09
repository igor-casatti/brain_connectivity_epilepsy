import numpy as np
import mne
import scot
import mne_connectivity as mnecon
from epoch_connectivity.conn_epoch import EpochConnection
from epoch_connectivity.utils import only_EEG_channels


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

        self.epoch_connecs = []
        for epoch in range(self.n_epochs):
            self.epoch_connecs.append(EpochConnection(epoch_data = self.epoch_matrix[epoch], sfreq = self.sfreq))


    def fit_MVAR_on_epochs(self, model_order = None, order_criterion = None, max_order = 30):
        """
        Fit on MVAR on each epoch. The MVAR order is calculated for each epoch

        -----------
        PARAMETERS
        -----------
        model_order (int | None) : MVAR model order to be used, if None finds the best order
        order_criterion (str) : criterion the be used in the order selection, if None uses the default (BIC)
        max_order (int) : maximum order to try on model order selection

        -----------
        RETURNS
        -----------
        None
        
        """

        for epoch in range(self.n_epochs):
            (self.epoch_connecs[epoch]).MVAR_fit(model_order=model_order, order_criterion=order_criterion, max_order=max_order)
    
    def integrated_measure_epochs(self, measure_name, average_values=True, frequency_band = None, nfft = None):
        """
        Calculate connectivity metrics (the MVAR model must be fitted first)

        -----------
        PARAMETERS
        -----------
        measure_name (str) : name of the measure to be used {'dDTF', 'DTF', 'PDC'}
        average_values (bool) : if true average the metrics obtained over the epochs, returning
          a matrix with diemnsions (n_channels, n_channels). If False returns a matrix with
          dimensions (n_epochs, n_channels, n_channesl) where each first axis component represents
          the connectivity over an epoch
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
        result = []

        for epoch in range(self.n_epochs):
            epoch_connectivity = (self.epoch_connecs[epoch]).integrated_connection_measure(measure_name=measure_name, frequency_band=frequency_band, nfft=nfft)
            result.append(epoch_connectivity)
        result = np.array(result)
        
        if average_values:
            return np.average(result, axis=0)
        else:
            return result
        
