import argparse
import pandas as pd
import numpy as np
from numpy import logsumexp

def build_tau_forward(p, k, num_states, index_b1, index_b2, index_b_end):

    tau = [[0 for j in range(num_states)] for i in range(num_states)]
    tau[index_b1][index_b1] = 1 - p
    tau[index_b1][0] = p
    tau[index_b2][index_b2] = 1 - p
    tau[index_b2][index_b_end] = p
    for i in range(k - 1):
        tau[i][i + 1] = 1
    tau[k - 1][index_b2] = 1
    return tau

def forward(seq,initial_emission, p, q):

    emission_table = pd.read_csv(initial_emission, sep='\t')
    k = len(emission_table['A'])
    num_rows = k + 3
    emission_table['$'] = [0 for i in range(k)]
    emission_table_bottom = [[0 for j in range(5)] for i in range(num_rows - k)]
    for j in range(4):
        for i in range(2):
            emission_table_bottom[i][j] = 0.25
    emission_table_bottom[2][4] = 1
    emission_table_bottom = pd.DataFrame(emission_table_bottom)
    emission_table_bottom.columns = emission_table.columns
    emission_table_bottom.index = [i + k for i in range (num_rows - k)]

    #print(emission_table)
    emission_table = emission_table.append(emission_table_bottom)
    #print(emission_table)

    #k = len(emission_table['A'])
    #num_rows = k + 3
    num_cols = len(seq)         #todo : has a column to B_end
    index_b1 = num_rows - 3
    index_b2 = num_rows - 2
    index_b_end = num_rows -1
    forward_table = [[0 for j in range(num_cols)] for i in range(num_rows)]
    tau = build_tau_forward(p, k, num_rows, index_b1, index_b2, index_b_end)

    #print(forward_table)
    #print(tau)

    forward_table[index_b1][0] = np.log(q * 0.25)
    forward_table[index_b2][0] = (1 - q) * 0.25
    for j in range(1, num_cols):   #todo : without te last column B_end
        for i in range(num_rows):
            for k in range(num_rows):
                forward_table[i][j] = np.logaddexp(forward_table[k][j-1] * tau[k][i]  * emission_table.loc[i][seq[j]],
                                                   forward_table[i][j])
    print(forward_table)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    if args.alg == 'viterbi':
        raise NotImplementedError

    elif args.alg == 'forward':
        forward(args.seq + '$', args.initial_emission, args.p, args.q)

    elif args.alg == 'backward':
        raise NotImplementedError

    elif args.alg == 'posterior':
        raise NotImplementedError


if __name__ == '__main__':
    main()
