import numpy as np
import mne
import matplotlib
import glob, os
matplotlib.use('Qt5Agg')

def open_events(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()
        get_info, info = False, []
        for line in lines:
            if get_info and len(line) != 0: info.append(line)
            if line == '[EVENT]': get_info = True
    
    events_array, events_dict = [], {}
    for events_counter in range(len(info)):
        current_event = info[events_counter].split(",")
        events_array.append([int(current_event[0]), int(0), int((events_counter+1))])
        events_dict[events_counter+1] = current_event[1]

    return (np.array(events_array), events_dict)

def open_filter_rereference(filepath, events_path):
    """
    Open the file and return the eeg with the channels specified in to_select
    """
    a = mne.io.read_raw_edf(filepath, preload=True)

    events_f, sfreq_i = open_events(events_path), a.info['sfreq']
    annotations = mne.annotations_from_events(events_f[0], sfreq_i, events_f[1], first_samp=0, orig_time=None, verbose=True)
    a.set_annotations(annotations)

    all_channels = set(a.ch_names)
    to_select = {'EEG C4', 'EEG O2', 'EEG Fp2', 'EEG F3', 'EEG T6', 'EEG F7', 'EEG Pz', 'ECG2', 'EEG P4', 'EEG P3',
     'EEG F8', 'EEG O1', 'EEG Fp1', 'EEG T5', 'EEG F4', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG Fz', 'EEG Cz', 'ECG1'}
    to_drop = list(all_channels-to_select)

    a.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})          # set the ecg channels as ecg type

    a.drop_channels(to_drop)                                     # drop the channels that are not in to_select

    a.set_eeg_reference(ref_channels='average', ch_type='eeg')   # re-reference to common average

    a.filter(l_freq=0.5, h_freq=30, picks=['eeg'], method='iir') # band-pass filter

    a.notch_filter(freqs=50, picks=['eeg'], method='iir') # notch filter

    return a

def make_epoched_data(eeg):
    """
    Make epoched data from anotated eeg
    """
    events = mne.events_from_annotations(eeg)
    events_name = {'E'+str(x) for x in range(1,11)}
    events_name = events_name.intersection(set(events[1]))
    selected = []
    for name in events_name:
        numb = events[1][name]
        for row in events[0]:
            if row[2] == numb:
                selected.append(row)
    selected = np.array(selected)
    epochs = mne.Epochs(eeg, selected, tmin=0, tmax=10, baseline=None, picks='all')
    return epochs, events

def plot_full_eeg(eeg):
    """
    Plot the full eeg
    """
    scales = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=500e-6,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2)
    fig = mne.viz.plot_raw(eeg, start=0, duration=100, use_opengl=True, precompute=True, scalings=scales)
    return

def plot_epoched_eeg(epoched_eeg, events = None):
    """
    Plot the epoched eeg
    """
    scales = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=500e-6,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2)
    if events is not None:
        epoched_eeg.plot(picks=['eeg', 'ecg'], events=events[0], event_id=events[1], scalings=scales)
    else:
        epoched_eeg.plot(picks=['eeg', 'ecg'], scalings=scales)

def fully_preprocess_eeg(patient_number, weaning = False, overwrite = False):
    """
    fully preprocess and save a copy on the file preprocessed
    """

    def full_cycle(root_directory, patient_number, patient_suffix=None):
        if patient_suffix is not None:
            sleep = root_directory + f"/PAT{patient_number}/EEG_SLEEP/{patient_suffix}/"
            awake = root_directory + f"/PAT{patient_number}/EEG_AWAKE/{patient_suffix}/"
            save_as = str(patient_number) + '_' + patient_suffix
        else:
            sleep, awake = root_directory + f"/PAT{patient_number}/EEG_SLEEP/", root_directory + f"/PAT{patient_number}/EEG_AWAKE/"
            save_as = str(patient_number)


        os.chdir(sleep)
        for file in glob.glob("*.edf"):
            patient_sleep = sleep + file
        for file in glob.glob("*.txt"):
            patient_sleep_events = sleep + file
            
        os.chdir(awake)
        for file in glob.glob("*.edf"):
            patient_awake = awake + file
        for file in glob.glob("*.txt"):
            patient_awake_events = awake + file

        a = open_filter_rereference(patient_sleep, patient_sleep_events)
        b, b_events = make_epoched_data(a)
        os.chdir(root_directory)
        b.save(f'preprocessed/PAT{save_as}_SLEEP-epo.fif', overwrite=overwrite)

        a = open_filter_rereference(patient_awake, patient_awake_events)
        b, b_events = make_epoched_data(a)
        os.chdir(root_directory)
        b.save(f'preprocessed/PAT{save_as}_AWAKE-epo.fif', overwrite=overwrite)

    root_directory = "C:/Users/igorc/OneDrive/Desktop/POLI/UCLouvain/3º Semestre/Thesis/Data/"

    if not weaning:
        full_cycle(root_directory, patient_number)

    else:
        full_cycle(root_directory, patient_number, 'H1')
        full_cycle(root_directory, patient_number, 'SEVRAGE')

def open_preprocessed_epoched(patient_number, weaning = False):

    def file_dir(patient_number, status, suffix = None):
        common_dir = "C:/Users/igorc/OneDrive/Desktop/POLI/UCLouvain/3º Semestre/Thesis/Data/preprocessed/"
        if suffix is not None:
            dir = f'PAT{patient_number}_{suffix}_{status}-epo.fif'
        else:
            dir = f'PAT{patient_number}_{status}-epo.fif'
        return common_dir+dir

    if weaning:
        epoched = []
        a_file, b_file, c_file, d_file = file_dir(patient_number, status = 'AWAKE',suffix = 'H1'), file_dir(patient_number, status = 'SLEEP',suffix = 'H1'), file_dir(patient_number, status = 'AWAKE',suffix = 'SEVRAGE'), file_dir(patient_number, status = 'SLEEP',suffix = 'SEVRAGE')
        epoched.append(mne.read_epochs(a_file, proj=False, preload=True, verbose=None))
        epoched.append(mne.read_epochs(b_file, proj=False, preload=True, verbose=None))
        epoched.append(mne.read_epochs(c_file, proj=False, preload=True, verbose=None))
        epoched.append(mne.read_epochs(d_file, proj=False, preload=True, verbose=None))
    
    else:
        epoched = []
        a_file, b_file = file_dir(patient_number, status = 'AWAKE'), file_dir(patient_number, status = 'SLEEP')
        epoched.append(mne.read_epochs(a_file, proj=False, preload=True, verbose=None))
        epoched.append(mne.read_epochs(b_file, proj=False, preload=True, verbose=None))
    
    return epoched
