import epoch_connectivity.ConnEEG as cnegg
from epoch_connectivity.utils import FrequencyBand, only_EEG_channels
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Connections(object):
    def __init__(self, epoched_eeg, f_bands = ['delta', 'theta', 'alpha', 'beta', 'broadband']):
        self.epoched_eeg = epoched_eeg
        self.f_bands = f_bands
        self.directed = cnegg.EEG_DirectedConnection(epoched_eeg)
        self.spectral = cnegg.EEG_SpectralConnection(epoched_eeg)
        self.directed.fit_MVAR_on_epochs(model_order=20, get_REV = True)
        self.fit_quality = self.directed.fit_quality_diagnostic()
    
    def Connectivity(self, directed_measures=['PDC', 'DTF'], undirected_measures=['wpli'], surrogate_test=False):
        connectivity_matrices = self.directed.integrated_measure_epochs(measures_names=directed_measures, frequency_bands=self.f_bands, nfft=None, surrogate_test=surrogate_test)
        for measure in undirected_measures:
            connectivity_matrices[measure] = []
            for f_ in self.f_bands:
                frequency_bands = FrequencyBand()
                freq = getattr(frequency_bands, f_)
                connectivity_matrices[measure].append(self.spectral.spec_connectivity(measure, freq))
        return connectivity_matrices

class ConnectionsMatrices(object):
    def __init__(self, epoched_eeg, patient_number, rec_status, f_bands = ['delta', 'theta', 'alpha', 'beta', 'broadband'], surrogate_test=False):
        self.epoched_eeg, self.patient, self.rec_status = epoched_eeg, patient_number, rec_status
        self.f_bands = f_bands
        self.connections = Connections(epoched_eeg, f_bands)
        self.fit_quality = self.connections.fit_quality
        connec_matrices = self.connections.Connectivity(directed_measures=['PDC', 'DTF'], undirected_measures=['wpli'], surrogate_test=surrogate_test)
        self.wPLI = connec_matrices['wpli']
        self.PDC = connec_matrices['PDC']
        self.DTF = connec_matrices['DTF']
    
    def plot_fbands_hmaps(self, measure, hide_diag = True, savefigure = False, fformat = '.pdf', directory=None):
        to_plot = getattr(self, measure)
        n_plot = len(self.f_bands)
        v, h = 3, int(np.ceil(n_plot/3))
        axes = []
        
        eeg_channels_names = only_EEG_channels(self.epoched_eeg.ch_names)
        
        fig = plt.figure(figsize=[h*6.4, v*4.8])

        # To create a common scale to the patient
        t_max, t_min = 0, 1
        for i in range(n_plot):
            #f = self.f_bands[i]
            matrix = to_plot[i].copy()
            if hide_diag:
                np.fill_diagonal(matrix, np.nan)
            a_max, a_min = np.nanmax(matrix), np.nanmin(matrix)
            if a_max > t_max:
                t_max = a_max
            if a_min < t_min:
                t_min = a_min


        for i in range(n_plot):
            axes.append(fig.add_subplot(v*100+h*10+i+1))
        plt.subplots_adjust(top=0.90)
        for i in range(n_plot):
            f = self.f_bands[i]
            matrix = to_plot[i].copy()
            if hide_diag:
                np.fill_diagonal(matrix, np.nan)
            #a_max, a_min = np.nanmax(matrix), np.nanmin(matrix)
            sns.heatmap(matrix, linewidth=0.1, cmap='viridis', ax=axes[i],
                        xticklabels=eeg_channels_names, yticklabels=eeg_channels_names, vmin=t_min, vmax=t_max)
            axes[i].set_title(f)
        fig.suptitle(measure + ' on patient ' + str(self.patient) + ' while ' + self.rec_status, fontsize=16, y=0.94)
        if savefigure:
            save_file_name = 'PAT' + str(self.patient) + '_' + self.rec_status + '_' + measure + fformat
            plt.savefig(directory+save_file_name)
        plt.show()