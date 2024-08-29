import glob
import json
import numpy as np
import os
import pandas as pd

import message_utils


def process_logs(meets_criteria, scores_reducer, reducer_args):
    rollouts = []
    for logfile in meets_criteria:
        with open(logfile, 'r') as fid:
            buffer = json.load(fid)
        messages = buffer['messages']
        rollouts.append(messages)

    results = []
    n = 0
    for fake_payload in rollouts:
        iteration_scores = message_utils.extract_results_from_messages(fake_payload)
        if len(iteration_scores[0]) == 0:
            iteration_scores = iteration_scores[1:]
        scs = message_utils.sorted_cumulative_scores(iteration_scores)
        kbd = scores_reducer(scs, **reducer_args)
        results.append(kbd)
        if len(scs) > n:
            n = len(scs)

    counts_arr = np.zeros((len(rollouts), n)) * np.nan
    for j, fake_payload in enumerate(rollouts):
        iteration_scores = message_utils.extract_results_from_messages(fake_payload)
        if len(iteration_scores[0]) == 0:
            iteration_scores = iteration_scores[1:]
        k = 0
        for i, s in enumerate(iteration_scores):
            k += len(s)
            counts_arr[j, i] = k

    results_iter = np.zeros((len(rollouts), n)) * np.nan
    for i, r in enumerate(results):
        results_iter[i, :len(r)] = r

    nc = int(np.nanmax(counts_arr))
    results_resampled = np.zeros((len(rollouts), nc)) * np.nan
    for i, r in enumerate(results_iter):
        for j in range(nc):
            results_resampled[i, j] = np.interp(j + 1, counts_arr[i], r,
                                                left=np.nan, right=np.nan)

    return results_iter, results_resampled


def main():
    base_dir = os.path.join('data', 'llm-logs')
    named_bools = {True: 'seeded', False: 'unseeded'}

    experiment_names = ['oracle', 'scientific', 'active', 'evolutionary', 'random']
    arch_morphs = ['liquid', 'string', 'membrane', 'vesicle', 'wormlike micelle', 'spherical micelle']

    reducers = [(message_utils.k_below_d, dict(d=1.34), 'kltd'),
                (message_utils.top_k_d, dict(top_k=3), 'topkd')]

    for (reducer, reducer_args, reducer_name) in reducers:

        big_df = []

        experiment_results = {}
        for experiment in experiment_names:
            for seed_type in [True, False]:
                arch_results = []
                for morph in arch_morphs:
                    experiment_longname = experiment
                    target_dir = os.path.join(base_dir, named_bools[seed_type], experiment_longname, morph)
                    logfiles = sorted(
                        glob.glob(os.path.join(target_dir, '*.json')))
                    if len(logfiles) < 5:
                        continue
                    results_iter, results_seq = process_logs(logfiles, reducer, reducer_args)
                    if results_seq.shape[1] > 50:  # truncate at 50 labeled sequences
                        results_seq = results_seq[:, :50]
                    if results_seq.shape[1] < 50:
                        print(f'Warning! {target_dir} gave less than 50 unique sequences')
                    arch_results.append(np.nanmax(results_seq, axis=1).flatten())
                    for i, it in enumerate(results_seq):
                        big_df.append([experiment, named_bools[seed_type], morph, i, *it.tolist()])
                experiment_results[(experiment, seed_type)] = arch_results

        data = pd.DataFrame(big_df, columns=['Experiment', 'Seed Type', 'Morphology', 'Replica',
                                             *[f'{reducer_name} after {i + 1} Labels' for i in range(50)]])
        data.to_csv(os.path.join('data', f'all-rollouts-{reducer_name}.csv'))


if __name__ == "__main__":
    main()
