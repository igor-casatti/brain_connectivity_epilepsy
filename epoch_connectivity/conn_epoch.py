import numpy as np
import mne
import scot
import mne_connectivity as mnecon
import scipy


"""

Object oriented API to integrate Scot with the project

"""

class EpochConnection(object):


    def __init__(self, epoch_data, sfreq):

        """
        epoch_data (numpy array) : array with dimensions (channels, time_samples)
        sfreq (int) : sampling frequency
        """

        self.epoch_data = epoch_data
        self.sfreq = int(sfreq)
        self.Connec = None
    

    def _MVAR_order_estimation(self, max_order=30, criterion=None):

        """
        Estimate the MVAR model order using some criterion.

        -----------
        PARAMETERS
        -----------
        max_order (int): maximum order to try
        criterion (str): criterion usde to find the best order {"aic", "bic", "hqic", "fpe"}

        -----------
        RETURNS
        -----------
        selected_order (int): best model order according to the selected criterion
        """

        if criterion is None:
            criterion = 'bic'

        c = np.array([self.epoch_data]) # To make it compatible with the mne_connectivity order estimator
        X, Y = scot.varbase._construct_var_eqns(data=c, p=1)
        selected_orders = mnecon.vector_ar.select_order(X, maxlags=max_order)
        selected_order = selected_orders[criterion]
        self.model_order = selected_order

        return selected_order

    def MVAR_fit(self, model_order = None, order_criterion = None, max_order = 30):

        """
        Fits the MVAR model into the current epoch

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

        if model_order is None:
            model_order = self._MVAR_order_estimation(max_order = max_order, criterion = order_criterion)
        
        self.MVAR = scot.var.VAR(model_order=model_order)
        self.MVAR.fit(data=self.epoch_data)


    def integrated_connection_measure(self, measure_name, frequency_band = None, nfft = None):
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
            nfft = self.sfreq

        MVAR_model = self.MVAR

        if self.Connec is None:
            self.connect_build(nfft=nfft)
            #self.Connec = scot.Connectivity(b = MVAR_model.coef, c = MVAR_model.rescov, nfft = nfft)

        if measure_name == 'dDTF':
            f_measure = self.Connec.dDTF()
        elif measure_name == 'DTF':
            f_measure = self.Connec.DTF()
        elif measure_name == 'PDC':
            f_measure = self.Connec.PDC()
        
        if frequency_band is not None: # Integrate on an specific interval
            freq_resolution = (self.sfreq/2)/(nfft - 1)
            low, high = frequency_band[0], frequency_band[1]
            i, j = int(low/freq_resolution), int(high/freq_resolution)

            x = np.linspace(0, self.sfreq/2, nfft)
            #x_i = x[i:j+1]

            #return np.sum(f_measure[:,:,i:j+1], axis=2)
            return scipy.integrate.simps(y=f_measure[:,:,i:j+1], x=x[i:j+1])
        
        elif frequency_band is None:
            return np.sum(f_measure, axis=2)
    
    def connect_build(self, nfft):

        """
        Builds the ScoT connectivity class

        -----------
        PARAMETERS
        -----------
        nfft (int) : number of frequency bins to use on discretization

        -----------
        RETURNS
        -----------
        None
        """

        MVAR_model = self.MVAR
        self.Connec = scot.Connectivity(b = MVAR_model.coef, c = MVAR_model.rescov, nfft = nfft)
