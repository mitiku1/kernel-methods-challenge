import pandas as pd 
import numpy as np 
from collections import defaultdict
import os
import argparse
def main(args):
    k = args.k
    m = args.m
    distances_path = "logs/k-{}.csv".format(k)
    outputpath = "logs/k-{}-m-{}.csv".format(k, m)
    distances = pd.read_csv(distances_path, header=None).values

    close_kmers = defaultdict(set)
    
    close_set = np.argwhere(distances <= m)
    for (i, j) in close_set:
      
        close_kmers[i].add(j)
        close_kmers[j].add(i)

    with open(outputpath, "w+") as output_file:
        for kmer in close_kmers:
            output_file.write(str(kmer))
            output_file.write(",")
            
            output_file.write(",".join(map(str, close_kmers[kmer])))
            output_file.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    args = parser.parse_args()
    main(args)

    