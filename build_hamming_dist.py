from utils.util import build_dist_matrix, index2code
import numpy as np
import pandas as pd
import argparse
def main(args):
    k = args.k
    output_path = "logs/k-{}.csv".format(k)
    matrix = build_dist_matrix(4, k=k)
    
    dist_df = pd.DataFrame(matrix)
    dist_df.to_csv(output_path, index=False, header=None)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", required=True, type=int)
    args = parser.parse_args()
    main(args)
