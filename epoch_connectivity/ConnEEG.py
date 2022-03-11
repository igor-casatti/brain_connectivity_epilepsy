import numpy as np
import mne
import scot
import mne_connectivity as mnecon
from epoch_connectivity.utils import only_EEG_channels, FrequencyBand
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EEG_DirectedConnection(object):

    """
    Class to fit MVAR models on epochs and obtain directed connectivity measures.
    Supported measures: see Scot documentation.

    -----------
    PARAMETERS
    -----------
    epoched_eeg (mne epoched EEG) : the epoched EEG as MNE format

    -----------
    METHODS
    -----------
    self.fit_MVAR_on_epochs(self, model_order=20, epoch_cv_division=5, get_REV = False)
    self.integrated_measure_epochs(self, measure_name, frequency_band = None, nfft = None)
    self.plot_connectivity_hmp(self, measure = 'dDTF', frequency_band = None, hide_diag = True)

    """

    def __init__(self, epoched_eeg):

        ordered_channel_names = epoched_eeg.ch_names.copy()
        ordered_channel_names.sort()
        self.epoched_eeg = epoched_eeg.reorder_channels(ordered_channel_names)

        channel_indices = mne.pick_types(self.epoched_eeg.info, eeg=True)
        self.ch_names = [self.epoched_eeg.ch_names[index] for index in channel_indices]
        self.ch_names = only_EEG_channels(self.ch_names)

        self.sfreq = self.epoched_eeg.info['sfreq']
        self.info = self.epoched_eeg.info

        self.epoch_matrix = self.epoched_eeg.get_data(picks=['eeg'])
        self.n_epochs = len(self.epoch_matrix)

        self.VARs = []

    
    def fit_MVAR_on_epochs(self, model_order=20, epoch_cv_division=5, get_REV = False):

        """
        Fit on MVAR on each epoch. Each MVAR is fitted with the Scot Library.
        Each epoch is subdivided into 5 epoch_cv_divisions for the Scot optimization.
        The model_order is fixed and the MVAR is fitted with a ridge regression
        and the penalty term is optimized. Also, it is possivle to get the
        Relative Error Variace for each epoch (get_REV = True).

        -----------
        PARAMETERS
        -----------
        model_order (int) : Model order to be used, it is important that the model order
          is big enough, since the penalty term will be optimized
        epoch_cv_division (int) : subdivision of the epochs to optimize the penalty term
        get_REV (bool) : If true calculate the REV for each epoch

        -----------
        RETURNS
        -----------
        None
        """

        coefficients = []
        rescovs = []
        self.REV = []

        for epoch in range(self.n_epochs):               # for each epoch
            b = self.epoch_matrix[epoch]                 # get the data of the epoch
            _, n_samples = np.shape(b)                   # number of samples
            to_remove = n_samples % epoch_cv_division    # number of samples to be removed to make n_samples a multiple of ecd

            b_ = np.delete(b, to_remove, axis=1)
            b_ = np.array(np.split(b_, epoch_cv_division, axis=1)) # Split the epoch in ecd parts

            varx = scot.var.VAR(model_order=model_order, n_jobs=4)            # MVAR for current epoch
            result = scot.varica.mvarica(x=b_, var=varx, reducedim='no_pca', optimize_var=True, varfit='ensemble')
        
            coefficients.append(result.a.coef)        # Append the current coefficients to the coefficients matrix
            rescovs.append(result.a.rescov)

            if get_REV:
                ep_REV = self._fit_quality_(epoch_data=b_, model_order=model_order, varx=varx)
                self.REV.append(ep_REV)
        
        self.coefficients = np.array(coefficients)
        self.rescovs = np.array(rescovs)
        self.REV = np.array(self.REV)

    def _fit_quality_(self, epoch_data, model_order, varx):
        """
        Auxiliar method to calculate the REV for the current epoch

        -----------
        PARAMETERS
        -----------
        epoch_data (numpy array) : Array containing the current epoch subdivided.
        model_order (int) : Model order used to fit the MVAR model.
        varx (scot var object) : Var object used in the fitting

        -----------
        RETURNS
        -----------
        REV (numpy array)

        -----------
        RETURNS
        -----------
        [1] THE ELECTROENCEPHALOGRAM AND THE ADAPTIVE AUTOREGRESSIVE MODEL: THEORY AND APPLICATIONS. Alois Schlögl. pp 18 - 19.
        """

        r = epoch_data[:,:,model_order:] - varx.predict(epoch_data)[:,:,model_order:]
        r_conc = np.concatenate((r), axis=1)
        testing_conc = np.concatenate((epoch_data[:,:,model_order:]), axis=1)
        MSE = np.mean((r_conc**2), axis=0)
        MSS = np.mean((testing_conc**2), axis=0)
        REV = MSE/MSS

        return REV

    def fit_quality_diagnostic(self):
        """
        Return a pandas dataframe containinf the average REV for each epoch
        table columns: {'REV mean', 'REV std', 'REV max', 'REV min'}.
        The option get_REV must be set to True on the fitting procedure.

        -----------
        PARAMETERS
        -----------
        epoch_data (numpy array) : Array containing the current epoch subdivided.
        model_order (int) : Model order used to fit the MVAR model.
        varx (scot var object) : Var object used in the fitting

        -----------
        RETURNS
        -----------
        

        -----------
        RETURNS
        -----------
        [1] THE ELECTROENCEPHALOGRAM AND THE ADAPTIVE AUTOREGRESSIVE MODEL: THEORY AND APPLICATIONS. Alois Schlögl. pp 18 - 19.
        """
        rev = self.REV
        rev_std, rev_mean, rev_max, rev_min = np.std(rev, axis=1), np.mean(rev, axis=1), np.max(rev, axis=1), np.min(rev, axis=1)
        quality = {'REV mean':rev_mean, 'REV std':rev_std, 'REV max':rev_max, 'REV min':rev_min}
        return pd.DataFrame(data=quality)


    def integrated_measure_epochs(self, measures_names, frequency_bands, nfft = None, surrogate_test = True):
        
        """
        Calculate connectivity metrics (the MVAR model must be fitted first)

        -----------
        PARAMETERS
        -----------
        measure_name (str) : name of the measure to be used ex. {'dDTF', 'DTF', 'PDC'}
        frequency_band (list of strings) : frequency band where the measure will be integrated if None
          then the measure is integrated over all the frequencies (broadband value) the list
          has the format ['delta', 'beta', 'theta', 'broadband']
        nfft (int) : number of frequency bins on the metrics resultant matrix if None then
          nfft = 2 * sfreq
        surrogate_test (boolean) : wheter to perform or not the surrogate data test on the connection metrics

        -----------
        RETURNS
        -----------
        connectivity matrix (numpy array) : a matrix of shape (m, m). The first dimension is
          the sink, the second dimension is the source.

        """

        if nfft is None:
            nfft = int(2 * self.sfreq)
        freq_resolution = (self.sfreq/2)/(nfft - 1)

        measures_dict = {}
        for measure_name in measures_names:
            measures_dict[measure_name] = []


        integrated_measure = []
        for epoch in range(self.n_epochs):
            b, rescov = self.coefficients[epoch], self.rescovs[epoch]
            con = scot.Connectivity(b = b, c = rescov, nfft=nfft)
            # Get the surrogate connectivity
            if surrogate_test:
                varx = scot.var.VAR(model_order=20)
                surrogate = scot.connectivity_statistics.surrogate_connectivity(measure_names=measures_names, data=self.epoch_matrix[epoch], var=varx, nfft=nfft, repeats=100, n_jobs=4)
            # Get the connectivity measure as a function of frequency for the current epoch
            for measure_name in measures_names:
                epoch_connec = getattr(con, measure_name)()
                if surrogate_test:
                    # Get the 95th percentile of the surrogate distribution
                    percentile = np.percentile(surrogate[measure_name], 95, axis=0)
                    # Any value below the 95th percentile is set as 0, because it is non significative
                    aux = np.clip(epoch_connec-percentile, a_min=0, a_max=None)
                    aux[aux>0] = 1
                    epoch_connec = np.multiply(aux, epoch_connec)
                measures_dict[measure_name].append(epoch_connec)
        
        return_dict = {}
        for measure_name in measures_names:
            measures_dict[measure_name] = np.mean(measures_dict[measure_name], axis=0)
            return_dict[measure_name] = []

        for measure_name in measures_names:
            f_bands = FrequencyBand()
            for frequency_band in frequency_bands:
                f = getattr(f_bands, frequency_band)
                low, high = f[0], f[1]
                i, j = int(low/freq_resolution), int(high/freq_resolution)
                connectivity = measures_dict[measure_name]
                return_dict[measure_name].append(np.mean(connectivity[:,:,i:j+1], axis=2))

        return return_dict

    def plot_connectivity_hmp(self, measure = 'dDTF', frequency_band = None, hide_diag = True):

        """
        Plot a heatmap of the connectivity measure specified.

        -----------
        PARAMETERS
        -----------
        measure (str) : name of the measure to be used.
        frequency_band (list) : frequency band where the measure will be integrated if None
          then the measure is integrated over all the frequencies (broadband value) the list
          has the format [lower_bound, higher_bound]
        hide_diag (bool) : if True hide the diagonal on the plot

        -----------
        RETURNS
        -----------
        plot
        """

        con_matrix = self.integrated_measure_epochs(measure_name=measure, frequency_band = frequency_band)

        if hide_diag:
            np.fill_diagonal(con_matrix, np.nan, wrap=False)

        info = self.info
        ch_names = self.ch_names

        channel_indices = mne.pick_types(info, eeg=True)
        #ch_eeg_names = [ch_names[index] for index in channel_indices]

        #ch_eeg_names = only_EEG_channels(ch_eeg_names)

        ax = sns.heatmap(con_matrix, linewidths=.5, cmap='viridis', xticklabels=self.ch_names, yticklabels=self.ch_names)

        if frequency_band is not None:
            title = measure + ' on frequency range ' + '[' + str(frequency_band[0]) + ', ' + str(frequency_band[1]) + ']'
        else:
            title = measure + 'on broadband'

        ax.set_title(title)
        plt.show()


class EEG_SpectralConnection(object):

    """
    Class to get the spectral connection between the channels.
    Supported measures: see mne_connectivity documentation

    -----------
    PARAMETERS
    -----------
    epoched_eeg (mne epoched EEG) : the epoched EEG as MNE format

    -----------
    METHODS
    -----------
    self.spec_connectivity(self, measure = 'wpli', frequency_band = None)
    self.plot_connectivity_hmp(self, measure = 'wpli', frequency_band = None)

    """
    
    def __init__(self, epoched_eeg):
        ordered_channel_names = epoched_eeg.ch_names.copy()
        ordered_channel_names.sort()
        self.epoched_eeg = epoched_eeg.reorder_channels(ordered_channel_names)
    
    def spec_connectivity(self, measure = 'wpli', frequency_band = None):

        """
        Get the spectral connectivity of the epoched EEG.

        -----------
        PARAMETERS
        -----------
        measure (str) : measure to be used
        frequency_band (list | array) : frequency band specified as [l_frequency, h_frequency]

        -----------
        RETURNS
        -----------
        connectivity matrix
        """

        if frequency_band is None:
            f_min, f_max = 0.5, np.inf
        else:
            f_min, f_max = frequency_band[0], frequency_band[1]
        
        con = mnecon.spectral_connectivity_epochs(data=self.epoched_eeg, method=measure, mode='multitaper',
                                              fmin=f_min, fmax=f_max, faverage=True, verbose=False)
        
        con_matrix = con.get_data(output='dense')[:,:,0] + np.transpose(con.get_data(output='dense')[:,:,0])

        # Get only channels that are EEG:
        channel_indices = mne.pick_types(self.epoched_eeg.info, eeg=True) # Indices of EEG channels
        # Return the connective matrix just with the EEG channels connectivity
        return con_matrix[np.ix_(channel_indices, channel_indices)]
    
    def plot_connectivity_hmp(self, measure = 'wpli', frequency_band = None, hide_diag = True):

        """
        Plot a heatmap of the connectivity measure specified.

        -----------
        PARAMETERS
        -----------
        measure (str) : name of the measure to be used.
        frequency_band (list) : frequency band where the measure will be integrated if None
          then the measure is integrated over all the frequencies (broadband value) the list
          has the format [lower_bound, higher_bound]
        hide_diag (bool) : if True hide the diagonal on the plot

        -----------
        RETURNS
        -----------
        plot
        """

        con_matrix = self.spec_connectivity(measure=measure, frequency_band=frequency_band)

        if hide_diag:
            np.fill_diagonal(con_matrix, np.nan, wrap=False)

        info = self.epoched_eeg.info
        ch_names = self.epoched_eeg.ch_names

        #channel_indices = mne.pick_types(info, eeg=True)
        #ch_eeg_names = [ch_names[index] for index in channel_indices]

        ch_eeg_names = only_EEG_channels(ch_names)

        ax = sns.heatmap(con_matrix, linewidths=.5, cmap='viridis', xticklabels=ch_eeg_names, yticklabels=ch_eeg_names)

        if frequency_band is not None:
            title = measure + ' on frequency range ' + '[' + str(frequency_band[0]) + ', ' + str(frequency_band[1]) + ']'
        else:
            title = measure + 'on broadband'

        ax.set_title(title)
        plt.show()