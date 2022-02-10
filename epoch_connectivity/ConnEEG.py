import numpy as np
import mne
import scot
import mne_connectivity as mnecon
from epoch_connectivity.conn_epoch import EpochConnection
from epoch_connectivity.utils import only_EEG_channels
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

"""

Object oriented API to integrate Scot with the project and speedup the process
of transforming EEGs into connectivity matrices

"""

class EEG_DirectedConnection(object):

    def __init__(self, epoched_eeg):
        #self.epoched_eeg = epoched_eeg
        self.ch_names = only_EEG_channels(epoched_eeg.ch_names)
        self.sfreq = epoched_eeg.info['sfreq']

        self.epoch_matrix = epoched_eeg.get_data(picks=['eeg'])
        self.n_epochs = len(self.epoch_matrix)


    def fit_MVAR_on_epochs(self, model_order=20):
        """
        Fit on MVAR on each epoch. The ScoT library fits the model with defined
        order and optimize the regularization.

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
        result = scot.varica.mvarica(x=self.epoch_matrix, var=varx, reducedim='no_pca', optimize_var=True, varfit='trial')

        # By getting result.a.coef we get the coefficients on EEG space, result.b.coef are the coefs in source space (ICA)
        self.coefs = result.a.coef
        self.rescov = result.a.rescov

    
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

        self.conn = scot.Connectivity(b = self.coefs, c = self.rescov, nfft=nfft)

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
        
#TODO https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

#TODO save function
# pickle.dump(myConn, open( "save.p", "wb" ))
# myConnLoaded = pickle.load( open( "save.p", "rb" ) )

#TODO another plot (?) <- for later

#TODO wPLI https://mne.tools/mne-connectivity/stable/auto_examples/cwt_sensor_connectivity.html


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
        
        con_matrix = con.get_data(output='dense')[:,:,0]

        # Get only channels that are EEG:
        channel_indices = mne.pick_types(self.epoched_eeg.info, eeg=True) # Indices of EEG channels
        # Return the connective matrix just with the EEG channels connectivity
        return con_matrix[np.ix_(channel_indices, channel_indices)]
    
    def plot_spec_connectivity_hmp(self, measure = 'wpli', frequency_band = None):
        con_matrix = self.spec_connectivity(measure=measure, frequency_band=frequency_band)
        ax = sns.heatmap(con_matrix, linewidth=0.1)
        plt.show()
        