import numpy as np

def encode_kmer(encoding, kmer):
    return np.array([encoding[c] for c in kmer])


def hamming_distance(x_code, y_code):
    return np.sum(x_code!=y_code)
def str_hamming_dist(str1, str2):
    assert len(str1)==len(str2)
    output = 0
    for i in range(len(str1)):
        if str1[i]!=str2:
            output+=1
    return output

    
def code2index(code, alpha_size=4):
    powers = np.array([alpha_size**i for i in range(len(code) -1, -1, -1)])
    return code.dot(powers)
def get_dna_encoding():
    return {
        "A" : 0,
        "C" : 1,
        "G" : 2,
        "T" : 3
    }

def string2code(string, encoding):
    return np.array([encoding[c] for c in string])

def index2code(index, length, alpha_size):
    output = []
    for i in range(length):
        output.append(index%alpha_size)
        index = index//alpha_size
    return np.array(output[::-1])
def indices2code(indices, k, l):
    output = np.zeros((indices.shape[0], k))
    for i in range(output.shape[0]):
        output[i] = index2code(indices[i], k, l)
    return output




def build_dist_matrix(l, k ):
    codes = indices2code(np.arange(l**k), k, l)
    output = np.array([[hamming_distance(codes[i], codes[j]) for i in range(len(codes))] for j in range(len(codes)) ])
    print(output.shape)
    return output

def sequence2kmers(seq, k):
    return [seq[i:i+k]  for i in range(len(seq) - k+1)]

