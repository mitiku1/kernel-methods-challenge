import pandas as pd 
import numpy as np 
from collections import defaultdict
import os
import argparse
from utils.util import string2code, get_dna_encoding, sequence2kmers, code2index
from utils.datastructures import KM_MismatchDS

def get_sequene_features(seq_kmers, mismatch_ds, k):
    output = np.zeros((4 ** k))
    for kmer in seq_kmers:
        for neighbor in mismatch_ds[kmer]:
            output[neighbor]+=1
    return output

def save_features(sequences_kmers, mismatch_ds, k, output_path):
    output = np.zeros((len(sequences_kmers), 4 ** k))

    for index, row in sequences_kmers.iterrows():
        output[index] = get_sequene_features(row.values, mismatch_ds, k)

    output = pd.DataFrame(output)
    output.to_csv(output_path, header=None, index=False)



def main(args):
    k = args.k
    m = args.m


    training_kmers_path = "logs/train_kmers-k-{}.csv".format(k)
    test_kmers_path = "logs/test_kmers-k-{}.csv".format(k)
    mismatch_data_path = "logs/k-{}-m-{}.csv".format(k, m)

    mismatch_ds = KM_MismatchDS(mismatch_data_path)

    training_kmers = pd.read_csv(training_kmers_path, header=None)    
    test_kmers = pd.read_csv(test_kmers_path, header=None)   

    training_output_path = "logs/train_features-k-{}-m-{}.csv".format(k, m)
    test_output_path = "logs/test_features-k-{}-m-{}.csv".format(k, m)

    save_features(training_kmers, mismatch_ds, k, training_output_path)
    save_features(test_kmers, mismatch_ds, k, test_output_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    args = parser.parse_args()
    main(args)

    