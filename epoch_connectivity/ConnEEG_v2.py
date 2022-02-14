import numpy as np
import mne
import scot
import mne_connectivity as mnecon
from epoch_connectivity.conn_epoch import EpochConnection
from epoch_connectivity.utils import only_EEG_channels
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.vector_ar.var_model as smvar

class EEG_DirectedConnection(object):

    def __init__(self, epoched_eeg):
        self.epoched_eeg = epoched_eeg
        #self.ch_names = only_EEG_channels(epoched_eeg.ch_names)
        self.ch_names = epoched_eeg.ch_names
        self.sfreq = epoched_eeg.info['sfreq']
        self.info = epoched_eeg.info

        self.epoch_matrix = epoched_eeg.get_data(picks=['eeg'])
        self.n_epochs = len(self.epoch_matrix)
    
    def fit_MVAR_on_epochs(self, ic = 'aic'):

        """
        Fit on MVAR on each epoch. Each MVAR is fitted with the Statsmodel library.
        For each epoch the best model order is estimated using the information criteria specified.

        -----------
        PARAMETERS
        -----------
        ic (str) : Information criteria to be used for the estimation of the model orders

        -----------
        RETURNS
        -----------
        None
        """

        coefficients = []
        rescovs = []

        for epoch in range(self.n_epochs):               # for each epoch
            b = self.epoch_matrix[epoch]                 # get the data of the epoch

            epoch_MVAR = smvar.VAR(endog=b.T)            # MVAR for current epoch
            results = epoch_MVAR.fit(ic=ic, trend='n')   # Fit MVAR on current epoch
        
            coefficients.append(results.params.T)        # Append the current coefficients to the coefficients matrix
            rescovs.append(results.sigma_u)
        
        self.coefficients = coefficients
        self.rescovs = rescovs


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
            nfft = int(2 * self.sfreq)

        freq_resolution = (self.sfreq/2)/(nfft - 1)
        i, j = 0, nfft

        if frequency_band is not None:
            low, high = frequency_band[0], frequency_band[1]
            i, j = int(low/freq_resolution), int(high/freq_resolution)


        integrated_measure = []
        for epoch in range(self.n_epochs):
            b, rescov = self.coefficients[epoch], self.rescovs[epoch]
            con = scot.Connectivity(b = b, c = rescov, nfft=nfft)
            # Get the connectivity measure as a function of frequency for the current epoch
            epoch_connec = getattr(con, measure_name)()
            # Array containing the frequency intervals
            x = np.linspace(0, self.sfreq/2, nfft)
            # Calculate the integrated connectivity measure to the current epoch on the frequency band
            integrated_epoch_measure = scipy.integrate.simps(y=epoch_connec[:,:,i:j+1], x=x[i:j+1])
            # Append the current epoch measure to the array containing all the measures
            integrated_measure.append(integrated_epoch_measure)
        
        integrated_measure = np.array(integrated_measure)
        return np.mean(integrated_measure, axis=0) # Return the mean of the integrated measure over all epochs

    def plot_connectivity_hmp(self, measure = 'dDTF', frequency_band = None):

        con_matrix = self.integrated_measure_epochs(measure_name=measure, frequency_band = frequency_band)

        info = self.info
        ch_names = self.ch_names

        channel_indices = mne.pick_types(info, eeg=True)
        ch_eeg_names = [ch_names[index] for index in channel_indices]

        ch_eeg_names = only_EEG_channels(ch_eeg_names)

        ax = sns.heatmap(con_matrix, linewidths=.5, cmap='viridis', xticklabels=ch_eeg_names, yticklabels=ch_eeg_names)

        if frequency_band is not None:
            title = measure + ' on frequency range ' + '[' + str(frequency_band[0]) + ', ' + str(frequency_band[1]) + ']'
        else:
            title = measure + 'on broadband'

        ax.set_title(title)
        plt.show()


class EEG_SpectralConnection(object):
    
    def __init__(self, epoched_eeg):
        self.epoched_eeg = epoched_eeg
        #self.ch_names = only_EEG_channels(epoched_eeg.ch_names)
        #self.sfreq = epoched_eeg.info['sfreq']
        #self.epoch_matrix = epoched_eeg.get_data(picks=['eeg'])
        #self.n_epochs = len(self.epoch_matrix)
    
    def spec_connectivity(self, measure = 'wpli', frequency_band = None):
        if frequency_band is None:
            f_min, f_max = 0, np.inf
        else:
            f_min, f_max = frequency_band[0], frequency_band[1]
        
        con = mnecon.spectral_connectivity_epochs(data=self.epoched_eeg, method=measure, mode='multitaper',
                                              fmin=f_min, fmax=f_max, faverage=True, verbose=False)
        
        con_matrix = con.get_data(output='dense')[:,:,0] + np.transpose(con.get_data(output='dense')[:,:,0])

        # Get only channels that are EEG:
        channel_indices = mne.pick_types(self.epoched_eeg.info, eeg=True) # Indices of EEG channels
        # Return the connective matrix just with the EEG channels connectivity
        return con_matrix[np.ix_(channel_indices, channel_indices)]
    
    def plot_connectivity_hmp(self, measure = 'wpli', frequency_band = None):
        con_matrix = self.spec_connectivity(measure=measure, frequency_band=frequency_band)

        info = self.epoched_eeg.info
        ch_names = self.epoched_eeg.ch_names

        channel_indices = mne.pick_types(info, eeg=True)
        ch_eeg_names = [ch_names[index] for index in channel_indices]

        ch_eeg_names = only_EEG_channels(ch_eeg_names)

        ax = sns.heatmap(con_matrix, linewidths=.5, cmap='viridis', xticklabels=ch_eeg_names, yticklabels=ch_eeg_names)

        if frequency_band is not None:
            title = measure + ' on frequency range ' + '[' + str(frequency_band[0]) + ', ' + str(frequency_band[1]) + ']'
        else:
            title = measure + 'on broadband'

        ax.set_title(title)
        plt.show()

        


