import pandas as pd 
import numpy as np 
from collections import defaultdict
import os
import argparse
from utils.util import string2code, get_dna_encoding, sequence2kmers, code2index



def save_set_kmers(meta_data, k):
    seqs = pd.read_csv(meta_data["path"])
    name = meta_data["name"]
    output_path = meta_data["output_path"]
    encoding = get_dna_encoding()

    

    with open(output_path, "w+") as output_file:
        for index, row in seqs.iterrows():
            kmers = sequence2kmers(row["seq"], k)
            codes = [code2index( string2code(kmer, encoding), 4) for kmer in kmers]
            output_file.write(",".join(list(map(str, codes))))
            output_file.write("\n")



def main(args):
    k = args.k
    # m = args.m

    # distances_path = "logs/k-{}.csv".format(k)
    # outputpath = "logs/k-{}-m-{}.csv".format(k, m)

    sequences_metas = [
        {
            "name":"train_sequences",
            "path": "dataset/Xtr.csv",
            "output_path": "logs/train_kmers-k-{}.csv".format(k)
        },
        {
            "name":"test_sequences",
            "path": "dataset/Xte.csv",
            "output_path": "logs/test_kmers-k-{}.csv".format(k)
        }
    ]
    for seq_meta in sequences_metas:
        print("processing", seq_meta["name"])
        save_set_kmers(seq_meta, k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True)
    args = parser.parse_args()
    main(args)

    