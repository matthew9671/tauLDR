import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import dct
import pandas as pd
import os

def count_mistakes(samples):
    """
    sample: (N, L, S)
    """
    num_positions = samples.shape[0] * samples.shape[1]
    return (np.sum(((samples[:,:-1] - samples[:,1:]) != 1) * (samples[:,:-1] != 0))) / num_positions

def get_dist(seq, S):
    L = seq.shape[0]
    one_hot = np.zeros((L, S))
    seq = np.array(seq, dtype=int)
    one_hot[np.arange(L), seq] = 1
    return np.sum(one_hot, axis=0) / L

def hellinger(seq1, seq2, S):
    d1, d2 = get_dist(seq1, S), get_dist(seq2, S)
    return np.sqrt(.5 * np.sum((d1 ** .5 - d2 ** .5) ** 2))

# This is for pianoroll only
def get_mask(seq, S=129):
    L = seq.shape[0]
    one_hot = np.zeros((L, S))
    seq = np.array(seq, dtype=int)
    one_hot[np.arange(L), seq] = 1
    return 1 - np.prod(1 - one_hot, axis=0)

def outliers(ref, sample, S=129):
    ref_mask = get_mask(ref, S)
    sample_dist = get_dist(sample, S)
    return np.sum((1 - ref_mask) * sample_dist)

def eval_mse_stats(dataset, samples, S):
    
    true_samples = dataset.data.cpu().numpy()
    samples = np.clip(samples, 0, S-2)

    true_samples_scaled = dataset.rescale(true_samples)
    samples_scaled = dataset.rescale(samples)

    out = {
        "mean": np.mean(
            (np.mean(samples_scaled, axis=0) 
           - np.mean(true_samples_scaled, axis=0))**2
        ),
        "variance": np.mean(
            (np.var(samples_scaled, axis=0) 
           - np.var(true_samples_scaled, axis=0))**2
        ),
        "skewness": np.mean(
            (skew(samples_scaled) 
           - skew(true_samples_scaled))**2
        ),
        "kurtosis": np.mean(
            (kurtosis(samples_scaled) 
           - kurtosis(true_samples_scaled))**2
        ),
    }
    return out

def eval_dct_mse_stats(dataset, samples, S):
    samples = np.clip(samples, 0, S-2)
    scaled_samples = dataset.rescale(samples)
    scaled_data = dataset.rescale(dataset.data.cpu()).numpy()
    
    true_samples_scaled = dct(scaled_data, type=2, norm='ortho')
    samples_scaled = dct(scaled_samples, type=2, norm='ortho')

    out = {
        "mean": np.mean(
            (np.mean(samples_scaled, axis=0) 
           - np.mean(true_samples_scaled, axis=0))**2
        ),
        "variance": np.mean(
            (np.var(samples_scaled, axis=0) 
           - np.var(true_samples_scaled, axis=0))**2
        ),
        "skewness": np.mean(
            (skew(samples_scaled) 
           - skew(true_samples_scaled))**2
        ),
        "kurtosis": np.mean(
            (kurtosis(samples_scaled) 
           - kurtosis(true_samples_scaled))**2
        ),
    }
    return out

def save_results(new_results, file_name):

    # Convert the new results to a DataFrame
    new_results_df = pd.DataFrame(new_results)

    # Check if the results file already exists
    if os.path.exists(file_name):
        # If the file exists, read the existing data
        existing_results_df = pd.read_csv(file_name)
        # Append the new results to the existing data
        updated_results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
    else:
        # If the file does not exist, the new results are the updated results
        updated_results_df = new_results_df

    # Save the updated results to the file
    updated_results_df.to_csv(file_name, index=False)

    print("Experiment results saved to ", file_name)