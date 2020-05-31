python sequence2kmers.py -k 3;
python sequence2kmers.py -k 4;
python sequence2kmers.py -k 5;
python sequence2kmers.py -k 6;
python sequence2kmers.py -k 7;


python build_hamming_dist.py -k 3
python build_hamming_dist.py -k 4
python build_hamming_dist.py -k 5
python build_hamming_dist.py -k 6
python build_hamming_dist.py -k 7


python build_mismatch_kmers.py -k 3 -m 1
python build_mismatch_kmers.py -k 4 -m 1
python build_mismatch_kmers.py -k 5 -m 1
python build_mismatch_kmers.py -k 5 -m 2
python build_mismatch_kmers.py -k 5 -m 3
python build_mismatch_kmers.py -k 6 -m 1
python build_mismatch_kmers.py -k 6 -m 2
python build_mismatch_kmers.py -k 6 -m 3
python build_mismatch_kmers.py -k 7 -m 1



python build_single_k_feature.py -k 3 -m 1
python build_single_k_feature.py -k 4 -m 1
python build_single_k_feature.py -k 5 -m 1
python build_single_k_feature.py -k 5 -m 2
python build_single_k_feature.py -k 5 -m 3
python build_single_k_feature.py -k 6 -m 1
python build_single_k_feature.py -k 6 -m 2
python build_single_k_feature.py -k 6 -m 3
python build_single_k_feature.py -k 7 -m 1

python train_script.py