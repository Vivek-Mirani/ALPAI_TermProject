import mne

# Load the EDF file
raw = mne.io.read_raw_edf('openbmi_data/123-5555-19.02.16.19.51.19.edf', preload=True)

# Access data
data, times = raw[:]

# data is a NumPy array of shape (n_channels, n_times)
print(data.shape)
print(times.shape)

# Access channel names
print(raw.ch_names)
print(data)