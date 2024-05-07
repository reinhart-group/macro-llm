import argparse
import numpy as np
import os
import torch
import anthropic
import hashlib
import json
import time
import message_utils, model_utils
import multiprocessing
from target_defs import archetype_predictions, archetype_sequences, archetype_plaintext
import yaml


def hash_dict(dictionary, n=8):
    """
    Converts a dictionary to a string and hashes it using SHA-256.
    Returns the hexadecimal digest of the hash.
    """
    # Convert the dictionary to a string sorted by keys
    dict_string = str(sorted(dictionary.items()))

    # Hash the string using SHA-256
    sha256 = hashlib.sha256()
    sha256.update(dict_string.encode('utf-8'))

    # Return the hexadecimal digest
    return sha256.hexdigest()[:n]


number_words = {'three': 3, 'five': 5, 'ten': 10}


def run_rollout(morph, n_batch, n_iter, use_oracle, use_seed, prompt_yml, gen_random, pad_random, temperature=0.0, suffix=""):
    # load the model ensemble
    model_ensemble = []
    model_path = os.path.join('models', 'gru-opt-cv10-sym')
    for i in range(10):
        model = torch.jit.load(os.path.join(model_path, f'fold-{i:02d}-scripted.pt'), map_location='cpu')
        model.eval()
        model_ensemble.append(model)

    # set up the LLM client
    with open('credentials-anthropic.txt', 'r') as fid:
        api_key = fid.read().strip()

    client = anthropic.Anthropic(api_key=api_key, timeout=30, max_retries=1)
    llm_model = "claude-3-opus-20240229"

    target = archetype_predictions[morph]

    # set up the prompts

    if use_oracle:
        with open(prompt_yml, 'r') as fid:
            buffer = yaml.safe_load(fid)
        initial_prompt = buffer['initial_prompt'].format(n_batch=n_batch, n_iter=n_iter+1)
        batch_prompt = buffer['batch_prompt'].format(n_batch=n_batch, n_iter=n_iter+1)

    else:
        target_description = archetype_plaintext[morph]
        with open(prompt_yml, 'r') as fid:
            buffer = yaml.safe_load(fid)
        initial_prompt = buffer['initial_prompt'].format(n_batch=n_batch, target_description=target_description, n_iter=n_iter+1)
        batch_prompt = buffer['batch_prompt'].format(n_batch=n_batch, target_description=target_description, n_iter=n_iter+1)

    if use_seed:
        initial_prompt += f" Hint: I already know that {archetype_sequences[morph]} gives a good result."
    if len(suffix) > 0 and temperature < 0.1 and gen_random is not True:  # only required for very low temperature
        initial_prompt += f" Here is a unique seed to differentiate subsequent rollouts (ignore this): {suffix}"

    max_tries = 5

    params = {'llm_model': llm_model,
              'target': target.tolist(),
              'initial_prompt': initial_prompt,
              'batch_prompt': batch_prompt,
              'temperature': temperature,
              'prompt_yml': prompt_yml,
              'gen_random': gen_random,
              'pad_random': pad_random}

    param_hash = hash_dict(params)

    ridx = int(suffix)
    rng = np.random.RandomState(ridx)

    # generate all sequences by composition
    n = 20
    all_seq_by_frac = {k: set() for k in range(n + 1)}
    limit = 2 ** n
    for i in range(limit):
        sequence = bin(i)[2:].zfill(n)
        mirror_sequence = sequence[::-1]
        if sequence <= mirror_sequence:
            all_seq_by_frac[sequence.count('1')].add(sequence)

    # create a master list of all possible sequences
    all_sequences = []
    for k, v in all_seq_by_frac.items():
        all_sequences += v
    possible_sequences = np.array(list(all_seq_by_frac[8]))

    if gen_random:
        print(f'choosing randomly from {len(possible_sequences)} sequences')
        init_idx = rng.choice(np.arange(len(possible_sequences)), 5, replace=False)
        init_bitstr = [possible_sequences[it] for it in init_idx]
        if use_seed:
            init_bitstr[0] = archetype_sequences[morph]
        init_sequences = [it.replace('0', 'A').replace('1', 'B') for it in init_bitstr]
        out = model_utils.evaluate_sequences(init_sequences, target, model_ensemble)
        initial_prompt += f"\nHere is an initial batch selected completely at random to get you started:\n{out}\n"

    # create initial message
    messages = [{"role": "user", "content": [{"type": "text", "text": initial_prompt}]}]

    start_time = int(time.time())
    logfile = f'logs/claude-test-{param_hash}-{start_time}{suffix}.json'
    unique_seq = []
    for _ in range(n_iter * 2):

        if len(unique_seq) >= (n_iter * number_words[n_batch]):
            break

        tries = 0
        response = None
        while tries < max_tries and response is None:
            try:
                response = message_utils.send_message(messages, client, llm_model, temperature=temperature)
            except anthropic.InternalServerError:
                print('received InternalServerError, trying again...')
                tries += 1
        messages.append(message_utils.build_llm_message(response.content[0].text))
        result = message_utils.extract_AB_substrings(response.content[0].text)
        # these_sequences = list(set(message_utils.postproc_sequences(result)))
        good_seq = message_utils.postproc_and_pad_sequences(result, number_words[n_batch], unique_seq, rng, possible_sequences, pad_random=pad_random)
        # these_sequences = list(set(good_seq))
        these_sequences = []
        for it in good_seq:  # this is to preserve ordering
            if it not in these_sequences:
                these_sequences.append(it)
        unique_seq = sorted(set(unique_seq + these_sequences))
        next_message = message_utils.build_user_message(batch_prompt,
                                                        model_utils.evaluate_sequences(these_sequences, target,
                                                                                       model_ensemble, sort=False))
        messages.append(next_message)
        print(next_message['content'][0]['text'])

        buffer = {'params': params, 'messages': messages}
        with open(logfile, 'w') as fid:
            json.dump(buffer, fid)

    print(f'Wrote results to {logfile}')
    return logfile


def main():

    parser = argparse.ArgumentParser(description='Example argument parser')

    parser.add_argument('morph', type=str, help='Target morphology (required)')
    parser.add_argument('prompt', type=str, help='Which prompt to use, specified as a yml file (required)')
    parser.add_argument('--use_oracle', action='store_true', default=False, help='Whether to use oracle prompt (otherwise contextualized prompt)')
    parser.add_argument('--nproc', type=int, default=5, help='Number of processes (default: 5)')
    parser.add_argument('--n_batch', type=str, default='five', help='Batch size (default: "five")')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations to run (default: 10)')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for LLM (default: 0)')
    parser.add_argument('--use_seed', action='store_true', default=False, help='Whether to seed with one good solution (default: False)')
    parser.add_argument('--gen_random', action='store_true', default=False, help='Whether to seed with random solutions same as AL (default: False)')
    parser.add_argument('--pad_random', action='store_true', default=False, help='Whether to pad batches with random solutions (default: False)')

    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=args.nproc)
    # submit the tasks to the pool for parallel execution
    results = {}
    for i in range(args.nproc):
        result = pool.apply_async(run_rollout, args=(args.morph, args.n_batch, args.n_iter, args.use_oracle, args.use_seed, args.prompt, args.gen_random, args.pad_random, args.temperature, str(i)))
        results[i] = result

    # wait for all tasks to complete and collect the results
    pool.close()
    pool.join()

    # retrieve the results from the dictionary using bitstrings as keys
    final_results = {}
    for i, result in results.items():
        try:
            final_results[i] = result.get()
        except multiprocessing.TimeoutError:
            print(f"Warning: computing {i} timed out.")


if __name__ == "__main__":
    main()
