def only_EEG_channels(ch_names_list):
    """
    Given a list of strings (channel names), return a reduced list with the channels
    with a name beginning with EEG.
    """
    reduced_list=[]
    for ch_name in ch_names_list:
        if ch_name.count('EEG') == 1:
            reduced_list.append(ch_name)
    return reduced_list

my_test = ['EEG Fp1', 'EEG F3', 'EEG F7', 'EEG C3', 'EEG T3', 'EEG T5', 'EEG P3', 'EEG O1', 'EEG Fp2', 'EEG F4', 'EEG F8', 'EEG C4', 'EEG T4', 'EEG T6', 'EEG P4', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz', 'ECG1', 'ECG2']
asdf = only_EEG_channels(my_test)
