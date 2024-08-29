import glob
import json
import message_utils, model_utils
import numpy as np
import os
import pandas as pd
import torch


def get_suffix(logfile):
    return int(logfile.split('.json')[0][-1])


def main():
    base_dir = os.path.join('data', 'llm-logs')
    experiment_names = ['oracle', 'scientific', 'active', 'evolutionary', 'random']

    # load the model ensemble
    model_ensemble = []
    model_path = os.path.join('models', 'gru-opt-cv10-sym')
    for i in range(10):
        model = torch.jit.load(os.path.join(model_path, f'fold-{i:02d}-scripted.pt'), map_location='cpu')
        model.eval()
        model_ensemble.append(model)

    ridx = 1
    for experiment in experiment_names:
        experiment_longname = experiment

        logfiles = glob.glob(os.path.join(base_dir, 'unseeded', experiment_longname, 'membrane', f'*{ridx}.json'))
        log = sorted(logfiles, key=os.path.getmtime)[0]
        if get_suffix(log) != ridx:
            raise ValueError(f'Expected to get replica 3 (#4), got {log} with suffix {get_suffix(log)}')
        with open(log, 'r') as fid:
            messages = json.load(fid)['messages']

        agg_data = []

        # for LLM
        all_seq = []
        all_iter = np.zeros(0)
        all_preds = np.zeros((0, 2))

        if experiment == "active":
            r = list(range(1, len(messages)))
            m = {i: i for i in r}
        elif experiment == "evolutionary":
            r = list(range(1, len(messages)))
            m = {i: i for i in r}
        elif (experiment == "scientific") or (experiment == "oracle"):
            r = [0] + list(range(1, 19, 2))
            m = {i: i // 2 + 1 for i in r}
        elif experiment == "random":
            continue
        else:
            raise ValueError()

        for i in r:
            sequences = message_utils.extract_AB_substrings(messages[i]['content'][0]['text'])
            sequences = [it for it in sequences if len(it) == 20]
            preds = model_utils.run_sequences(sequences, model_ensemble)
            preds = [it.split(':')[1] for it in preds.split('\n')]
            preds = np.array([np.array([float(it) for it in x.split(',')]) for x in preds])
            all_preds = np.vstack([all_preds, preds])
            this_iter = np.ones(len(preds)) * m[i]
            if i == 0:
                this_iter *= 0
            all_iter = np.hstack([all_iter, this_iter])
            all_seq.append(sequences)
        all_seq = np.hstack(all_seq)

        data = pd.DataFrame(np.hstack([all_seq.reshape(-1, 1), all_iter.reshape(-1, 1), all_preds]),
                            columns=['Sequence', 'Iteration', 'Z0', 'Z1'])
        data.to_csv(os.path.join('data', f'rollout-membrane-{experiment}-{ridx}.csv'))


if __name__ == "__main__":
    main()
