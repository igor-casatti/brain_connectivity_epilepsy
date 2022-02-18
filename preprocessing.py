import numpy as np
import mne
import matplotlib
import glob, os
from scipy import signal

def bandpass_butterworth(cut_low, cut_high, sampling_freq, order=2):
  nyquist_freq = sampling_freq/2
  normalized_low = cut_low/nyquist_freq
  normalized_high = cut_high/nyquist_freq
  b, a = signal.butter(order, [normalized_low, normalized_high], btype='bandpass')
  return b, a

def notch_butterworth(cut_low, cut_high, sampling_freq, order=2):
  nyquist_freq = sampling_freq/2
  normalized_low = cut_low/nyquist_freq
  normalized_high = cut_high/nyquist_freq
  b, a = signal.butter(order, [normalized_low, normalized_high], btype='bandstop')
  return b, a

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
     'EEG F8', 'EEG O1', 'EEG Fp1', 'EEG T5', 'EEG F4', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG Fz', 'EEG Cz', 'ECG1', 'ECG2', 'ECG'}
    to_drop = list(all_channels-to_select)
    try:
        a.set_channel_types({'ECG1': 'ecg', 'ECG2': 'ecg'})          # set the ecg channels as ecg type
    except:
        a.set_channel_types({'ECG': 'ecg'})          # set the ecg channels as ecg type

    a.drop_channels(to_drop)                                     # drop the channels that are not in to_select

    a.set_eeg_reference(ref_channels='average', ch_type='eeg')   # re-reference to common average

    #a.filter(l_freq=0.5, h_freq=30, picks=['eeg'], method='iir', iir_params=dict(order=2, ftype='butter', output='sos')) # band-pass filter
    filt_b, filt_a = bandpass_butterworth(0.5, 30, a.info['sfreq'], order=2)
    a.filter(l_freq=0.5, h_freq=30, picks=['eeg'], method='iir', iir_params=dict(b=filt_b, a=filt_a, padlen=0))

    #a.notch_filter(freqs=50, picks=['eeg'], method='iir', iir_params=dict(order=2, ftype='butter', output='sos')) # notch filter
    filt_b, filt_a = notch_butterworth(49.5, 50.5, a.info['sfreq'], order=2)  # notch band = (freqs / 200) * 2 (MNE PYTHON CODE on github)
    a.notch_filter(freqs=50, picks=['eeg'], method='iir', iir_params=dict(b=filt_b, a=filt_a, padlen=0))

    return a

def remove_duplicate_from_events(events):
    last_column = events[:, 2]
    idx_sort = np.argsort(last_column)
    sorted_last_column = last_column[idx_sort]
    vals, idx_start, count = np.unique(sorted_last_column, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    index_to_remove = []
    for val in vals:
        print(val)
        index_to_remove.append(np.where(last_column == val)[0][-1])
    events_new = np.delete(events, index_to_remove, 0)
    return events_new

def make_epoched_data(eeg):
    """
    Make epoched data from anotated eeg
    """
    events = mne.events_from_annotations(eeg)
    events_name = {'E'+str(x) for x in range(1,11)}
    events_name = events_name.intersection(set(events[1]))
    selected, e_ids = [], {}
    for name in events_name:
        numb = events[1][name]
        for row in events[0]:
            if row[2] == numb:
                selected.append(row)
                e_ids[name] = row[2]
    selected = np.array(selected)
    selected = selected[selected[:, 0].argsort()]
    new_e_ids = {}
    for num_id in selected[:,2]:
        match = list(e_ids.keys())[list(e_ids.values()).index(num_id)]
        new_e_ids[match] = num_id
    
    if len(selected) > 10: # if len bigger than expected we remove the duplicates
        selected = remove_duplicate_from_events(selected)

    #while len(selected) > 10: # if more than 10 epochs delete the last ones
    #    selected = np.delete(selected, 10, axis=0)

    epochs = mne.Epochs(raw=eeg, events=selected, event_id=new_e_ids, tmin=0, tmax=10, baseline=None, picks='all')
    return epochs, events

def plot_full_eeg(eeg):
    """
    Plot the full eeg
    """
    scales = dict(mag=1e-12, grad=4e-11, eeg=30e-6, eog=150e-6, ecg=500e-6,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2)
    fig = mne.viz.plot_raw(eeg, start=0, duration=100, use_opengl=True, precompute=True, scalings=scales)
    return

def plot_epoched_eeg(epoched_eeg, events = None):
    """
    Plot the epoched eeg
    """
    scales = dict(mag=1e-12, grad=4e-11, eeg=30e-6, eog=150e-6, ecg=500e-6,
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
            sleep = root_directory + f"PAT{patient_number}//EEG_SLEEP//{patient_suffix}//"
            awake = root_directory + f"PAT{patient_number}//EEG_AWAKE//{patient_suffix}//"
            save_as = str(patient_number) + '_' + patient_suffix
        else:
            sleep, awake = root_directory + f"PAT{patient_number}//EEG_SLEEP//", root_directory + f"PAT{patient_number}//EEG_AWAKE//"
            save_as = str(patient_number)


        os.chdir(sleep)
        for file in glob.glob("*.edf"):
            patient_sleep = sleep + file
            print('Penis')
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
        b.save(f'preprocessed//PAT{save_as}_SLEEP-epo.fif', overwrite=overwrite)

        a = open_filter_rereference(patient_awake, patient_awake_events)
        b, b_events = make_epoched_data(a)
        os.chdir(root_directory)
        b.save(f'preprocessed//PAT{save_as}_AWAKE-epo.fif', overwrite=overwrite)

    root_directory = "C://Users//igorc//OneDrive//Desktop//POLI//UCLouvain//3º Semestre//Thesis//Data//"

    if not weaning:
        full_cycle(root_directory, patient_number)

    else:
        full_cycle(root_directory, patient_number, 'H1')
        full_cycle(root_directory, patient_number, 'SEVRAGE')

def open_preprocessed_epoched(patient_number, weaning = False):

    def file_dir(patient_number, status, suffix = None):
        common_dir = "C://Users//igorc//OneDrive//Desktop//POLI//UCLouvain//3º Semestre//Thesis//Data//preprocessed//"
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

def open_raw_eeg(patient_number, weaning = False):

    def file_dir(patient_number, status, suffix = None):
        common_dir = "C://Users//igorc//OneDrive//Desktop//POLI//UCLouvain//3º Semestre//Thesis//Data//"
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
    
def get_path(root_directory, patient_number, patient_suffix=None):
    if patient_suffix is not None:
        sleep = root_directory + f"PAT{patient_number}//EEG_SLEEP//{patient_suffix}//"
        awake = root_directory + f"PAT{patient_number}//EEG_AWAKE//{patient_suffix}//"
    else:
        sleep, awake = root_directory + f"PAT{patient_number}//EEG_SLEEP//", root_directory + f"PAT{patient_number}//EEG_AWAKE//"
    
    os.chdir(sleep)
    for file in glob.glob("*.edf"):
        sleep += file
    os.chdir(awake)
    for file in glob.glob("*.edf"):
        awake += file
        
    return sleep, awake
    
def open_raw_eeg(patient_number, weaning = False):
    root = 'C:/Users/igorc/OneDrive/Desktop/POLI/UCLouvain/3º Semestre/Thesis/Data/'

    def pre_process_this(eeg):
        eeg.set_eeg_reference(ref_channels='average', ch_type='eeg')   # re-reference to common average

        #a.filter(l_freq=0.5, h_freq=30, picks=['eeg'], method='iir', iir_params=dict(order=2, ftype='butter', output='sos')) # band-pass filter
        filt_b, filt_a = bandpass_butterworth(0.5, 30, eeg.info['sfreq'], order=2)
        eeg.filter(l_freq=0.5, h_freq=30, picks=['eeg'], method='iir', iir_params=dict(b=filt_b, a=filt_a, padlen=0))

        #a.notch_filter(freqs=50, picks=['eeg'], method='iir', iir_params=dict(order=2, ftype='butter', output='sos')) # notch filter
        filt_b, filt_a = notch_butterworth(49.5, 50.5, eeg.info['sfreq'], order=2)  # notch band = (freqs / 200) * 2 (MNE PYTHON CODE on github)
        eeg.notch_filter(freqs=50, picks=['eeg'], method='iir', iir_params=dict(b=filt_b, a=filt_a, padlen=0))
        return eeg

    if weaning:
        sleep_h1, awake_h1 = get_path(root, patient_number, 'H1')
        sleep_sev, awake_sev = get_path(root, patient_number, 'SEVRAGE')
        sleep_h1, awake_h1 = mne.io.read_raw_edf(sleep_h1, preload=True), mne.io.read_raw_edf(awake_h1, preload=True)
        sleep_sev, awake_sev = mne.io.read_raw_edf(sleep_sev, preload=True), mne.io.read_raw_edf(awake_sev, preload=True)
        return [pre_process_this(sleep_h1), pre_process_this(awake_h1), pre_process_this(sleep_sev), pre_process_this(awake_sev)]
        
    else:
        sleep, awake = get_path(root, patient_number)
        sleep, awake = mne.io.read_raw_edf(sleep, preload=True), mne.io.read_raw_edf(awake, preload=True)
        return [pre_process_this(awake), pre_process_this(sleep)]