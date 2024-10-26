"""
This script compares the performance of vectorized MI calculation approaches
"""

import numpy as np
import time
import torch


data_length=30_000
filtered_low=np.random.uniform(0, np.pi, size=data_length) # Generate random values uniformly distributed between 0 and pi
filtered_high=np.abs(np.random.randn(data_length))

n_low = 1
n_high = 30
n_shifts = 100
N_BINS_TORT = 18

amplitude=filtered_high

n_bins = N_BINS_TORT
# make sure edge case for right edge is included
eps = np.finfo(filtered_low.dtype).eps * 2
phase_bins = np.linspace(-np.pi, np.pi + eps, n_bins + 1)
# get the indices of the bins to which each value in input belongs
phase_preprocessed = np.digitize(filtered_low, phase_bins) - 1


def _one_modulation_index(amplitude, phase_preprocessed, norm_a=None, method='tort', shift=0,
                          ax_special=None):
    """
    Compute one modulation index.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    # shift for the surrogate analysis
    if shift != 0:
        phase_preprocessed = np.roll(phase_preprocessed, shift)

    

    # Modulation index as in [Tort & al 2010]
    if method == 'tort':
        # mean amplitude distribution along phase bins
        n_bins = N_BINS_TORT
        amplitude_dist = np.ones(n_bins)  # default is 1 to avoid log(0)
        for b in np.unique(phase_preprocessed):
            selection = amplitude[phase_preprocessed == b]
            amplitude_dist[b] = np.mean(selection)

        # Kullback-Leibler divergence of the distribution vs uniform
        amplitude_dist /= np.sum(amplitude_dist)
        divergence_kl = np.sum(
            amplitude_dist * np.log(amplitude_dist * n_bins))

        MI = divergence_kl / np.log(n_bins)


    else:
        raise ValueError("Unknown method: %s" % (method, ))

    return MI


def _modulation_index_vectorized(amplitude, phase_preprocessed, norm_a=None, method='tort', shifts=None, ax_special=None):
    """
    Compute modulation indices for multiple shift values in a vectorized manner.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    if method != 'tort':
        raise ValueError("Incompatible method for vectorized calculation: %s" % method)
        
    # Prepare phase-shifted versions of phase_preprocessed using broadcasting
    shifted_phases = np.array([np.roll(phase_preprocessed, shift) for shift in shifts])

    # Broadcast amplitude to match the shape of shifted_phases
    amplitude_expanded = np.tile(amplitude[:, None], (1, len(shifts)))  # Shape: (len(phase_preprocessed), len(shifts))

    # Vectorized computation of the mean amplitude distribution along phase bins
    n_bins = N_BINS_TORT
    
    # Initialize amplitude distributions with ones to avoid log(0)
    amplitude_dist = np.ones((len(shifts), n_bins))
    
    for b in np.unique(phase_preprocessed):
        # Mask for selecting amplitude values corresponding to each bin in each shift
        mask = (shifted_phases == b)
        selected_amplitudes = np.where(mask.T, amplitude_expanded, 0)
        amplitude_dist[:, b] = selected_amplitudes.sum(axis=0) / np.maximum(mask.sum(axis=1), 1)  # Avoid division by zero

    # Calculate the KL divergence for each shift value
    amplitude_dist /= amplitude_dist.sum(axis=1, keepdims=True)
    divergence_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins), axis=1)

    # Compute Modulation Index for each shift
    MI = divergence_kl / np.log(n_bins)

    return MI

def _modulation_index_vectorized_gpu(amplitude, phase_preprocessed, norm_a=None, method='tort', shifts=None, ax_special=None):
    """
    Compute modulation indices for multiple shift values in a vectorized manner using GPU.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    if method != 'tort':
        raise ValueError("Incompatible method for vectorized calculation: %s" % method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move inputs to GPU
    amplitude = torch.tensor(amplitude, dtype=torch.float64, device=device)
    phase_preprocessed = torch.tensor(phase_preprocessed, dtype=torch.int32, device=device)
    
    # Prepare phase-shifted versions of phase_preprocessed
    shifted_phases = torch.stack([torch.roll(phase_preprocessed, shift) for shift in shifts], dim=0)

    # Broadcast amplitude to match the shape of shifted_phases
    amplitude_expanded = amplitude.unsqueeze(1).repeat(1, len(shifts))  # Shape: (len(phase_preprocessed), len(shifts))

    # Vectorized computation of the mean amplitude distribution along phase bins
    n_bins = N_BINS_TORT
    amplitude_dist = torch.ones((len(shifts), n_bins), dtype=torch.float64, device=device)  # Initialize with ones

    unique_bins = torch.unique(phase_preprocessed)

    for b in unique_bins:
        # Mask for selecting amplitude values corresponding to each bin in each shift
        mask = (shifted_phases == b).transpose(0, 1)  # Shape: (len(phase_preprocessed), len(shifts))
        selected_amplitudes = torch.where(mask, amplitude_expanded, torch.tensor(0.0, device=device))
        amplitude_dist[:, b] = selected_amplitudes.sum(dim=0) / torch.maximum(mask.sum(dim=0), torch.tensor(1.0, device=device))

    # Calculate the KL divergence for each shift value
    amplitude_dist /= amplitude_dist.sum(dim=1, keepdims=True)
    divergence_kl = torch.sum(amplitude_dist * torch.log(amplitude_dist * n_bins), dim=1)
    
    # Compute Modulation Index for each shift
    MI = divergence_kl / torch.log(torch.tensor(n_bins, dtype=torch.float64, device=device))

    return MI.cpu().numpy()  # Return to CPU if further processing on CPU is needed

def _modulation_index_vectorized_v2(amplitudes, phase_preprocessed, norm_a=None, method='tort', shifts=None, ax_special=None):
    """
    Compute modulation indices for multiple shift values in a vectorized manner.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    if method != 'tort':
        raise ValueError("Incompatible method for vectorized calculation: %s" % method)
        
    # Prepare phase-shifted versions of phase_preprocessed using broadcasting
    shifted_phases = np.array([np.roll(phase_preprocessed, shift) for shift in shifts])

    # Broadcast amplitudes to match the shape of shifted_phases across multiple rows
    amplitudes_expanded = np.repeat(amplitudes[:, :, None], len(shifts), axis=2)  # Shape: (n_amplitudes, len(phase_preprocessed), len(shifts))

    # Vectorized computation of the mean amplitude distribution along phase bins
    n_bins = N_BINS_TORT
    
    # Initialize amplitude distributions for each shift and each amplitude series
    amplitude_dist = np.ones((amplitudes.shape[0], len(shifts), n_bins))
    
    for b in np.unique(phase_preprocessed):
        # Mask for selecting amplitude values corresponding to each bin in each shift
        mask = (shifted_phases == b) # Shape: (len(phase_preprocessed), len(shifts))
        selected_amplitudes = np.where(mask.T[None, :, :], amplitudes_expanded, 0)  # Shape: (n_amplitudes, len(phase_preprocessed), len(shifts))
        amplitude_dist[:, :, b] = selected_amplitudes.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)  # Avoid division by zero

    # Calculate the KL divergence for each shift value
    amplitude_dist /= amplitude_dist.sum(axis=2, keepdims=True)
    divergence_kl = np.sum(amplitude_dist * np.log(amplitude_dist * n_bins), axis=2)

    # Compute Modulation Index for each shift
    MI = divergence_kl / np.log(n_bins)

    return MI

def _modulation_index_vectorized_v2_gpu(amplitudes, phase_preprocessed, norm_a=None, method='tort', shifts=None, ax_special=None):
    """
    Compute modulation indices for multiple shift values in a vectorized manner using GPU.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    if method != 'tort':
        raise ValueError("Incompatible method for vectorized calculation: %s" % method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to GPU
    amplitudes = torch.tensor(amplitudes, dtype=torch.float64, device=device)
    phase_preprocessed = torch.tensor(phase_preprocessed, dtype=torch.int32, device=device)
    #shifts = torch.tensor(np.array(shifts), device=device)
    
    # Prepare phase-shifted versions of phase_preprocessed
    shifted_phases = torch.stack([torch.roll(phase_preprocessed, shift) for shift in shifts], dim=0)

    # Broadcast amplitudes to match the shape of shifted_phases
    amplitudes_expanded = amplitudes.unsqueeze(-1).repeat(1, 1, len(shifts))  # Shape: (n_amplitudes, len(phase_preprocessed), len(shifts))

    # Vectorized computation of the mean amplitude distribution along phase bins
    n_bins = N_BINS_TORT
    
    # Initialize amplitude distributions for each shift and each amplitude series
    amplitude_dist = torch.ones((amplitudes.shape[0], len(shifts), n_bins), dtype=torch.float64, device=device)
    
    unique_bins = torch.unique(phase_preprocessed)
    
    for b in unique_bins:
        # Mask for selecting amplitude values corresponding to each bin in each shift
        mask = (shifted_phases == b).transpose(0, 1)  # Shape: (len(phase_preprocessed), len(shifts))
        selected_amplitudes = torch.where(mask.unsqueeze(0), amplitudes_expanded, torch.tensor(0.0, device=device))
        amplitude_dist[:, :, b] = selected_amplitudes.sum(dim=1) / torch.maximum(mask.sum(dim=0).unsqueeze(0), torch.tensor(1.0, device=device))
    
    # Calculate the KL divergence for each shift value
    amplitude_dist /= amplitude_dist.sum(dim=2, keepdims=True)
    divergence_kl = (amplitude_dist * torch.log(amplitude_dist * n_bins)).sum(dim=2)
    
    # Compute Modulation Index for each shift
    MI = divergence_kl / torch.log(torch.tensor(n_bins, dtype=torch.float64, device=device))

    return MI.cpu().numpy()  # Move back to CPU if needed for further processing

shifts=np.arange(0,n_shifts,1).tolist()

#%%
# Start time
start_time = time.time()
for i in range(n_high):
    a=[]
    for s in shifts:
        c=_one_modulation_index(amplitude,phase_preprocessed,shift=s)
        a.append(c)
# End time
end_time = time.time()

# Execution time
execution_time = end_time - start_time
print(f"Execution time baseline: {execution_time} seconds")

#%%
# Start time
start_time = time.time()
for i in range(n_high):
    b=_modulation_index_vectorized(amplitude,phase_preprocessed,shifts=shifts)
# End time
end_time = time.time()

# Execution time
execution_time = end_time - start_time
print(f"Execution time vectorized: {execution_time} seconds")

print(np.max(np.abs(b-a)))

#%%
# Start time
start_time = time.time()
for i in range(n_high):
    b=_modulation_index_vectorized_gpu(amplitude,phase_preprocessed,shifts=shifts)
# End time
end_time = time.time()

# Execution time
execution_time = end_time - start_time
print(f"Execution time vectorized gpu: {execution_time} seconds")

print(np.max(np.abs(b-a)))
#%%
amplitudes=np.repeat(filtered_high[np.newaxis, :], n_high, axis=0) #np.vstack((filtered_high,filtered_high))
# Start time
start_time = time.time()
b2=_modulation_index_vectorized_v2(amplitudes,phase_preprocessed,shifts=shifts)
# End time
end_time = time.time()

# Execution time
execution_time = end_time - start_time
print(f"Execution time vectorized_v2: {execution_time} seconds")
print(np.max(np.abs(b2[1,:]-a)))
#%%
# Start time
start_time = time.time()
b2=_modulation_index_vectorized_v2_gpu(amplitudes,phase_preprocessed,shifts=shifts)
# End time
end_time = time.time()

# Execution time
execution_time = end_time - start_time
print(f"Execution time vectorized_v2 gpu: {execution_time} seconds")
print(np.max(np.abs(b2[1,:]-a)))



def _modulation_index_vectorized_final(amplitudes, phase_preprocessed, method='tort', shifts=None):
    """
    Compute modulation indices for multiple shift values in a vectorized manner using GPU.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    if method != 'tort':
        raise ValueError("Incompatible method for vectorized calculation: %s" % method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to GPU
    amplitudes = torch.tensor(amplitudes, device=device)
    phase_preprocessed = torch.tensor(phase_preprocessed, device=device)
    
    # Prepare phase-shifted versions of phase_preprocessed
    shifted_phases = torch.stack([torch.roll(phase_preprocessed, shift) for shift in shifts], dim=0)

    # Broadcast amplitudes to match the shape of shifted_phases
    amplitudes_expanded = amplitudes.unsqueeze(-1).repeat(1, 1, len(shifts))  # Shape: (n_amplitudes, len(phase_preprocessed), len(shifts))

    # Vectorized computation of the mean amplitude distribution along phase bins
    n_bins = N_BINS_TORT
    
    # Initialize amplitude distributions for each shift and each amplitude series
    amplitude_dist = torch.ones((amplitudes.shape[0], len(shifts), n_bins), device=device)
    
    unique_bins = torch.unique(phase_preprocessed)
    
    for b in unique_bins:
        # Mask for selecting amplitude values corresponding to each bin in each shift
        mask = (shifted_phases == b).transpose(0, 1)  # Shape: (len(phase_preprocessed), len(shifts))
        selected_amplitudes = torch.where(mask.unsqueeze(0), amplitudes_expanded, torch.tensor(0.0, device=device))
        amplitude_dist[:, :, b] = selected_amplitudes.sum(dim=1) / torch.maximum(mask.sum(dim=0).unsqueeze(0), torch.tensor(1.0, device=device))
    
    # Calculate the KL divergence for each shift value
    amplitude_dist /= amplitude_dist.sum(dim=2, keepdims=True)
    divergence_kl = (amplitude_dist * torch.log(amplitude_dist * n_bins)).sum(dim=2)
    
    # Compute Modulation Index for each shift
    MI = divergence_kl / torch.log(torch.tensor(n_bins, device=device))

    return MI.cpu().numpy()  # Move back to CPU if needed for further processing