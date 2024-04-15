import hashlib
import matplotlib.pyplot as plt
import numpy as np
import random


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


def extract_AB_substrings(input_string):
    substrings = []
    current_substring = ''

    for char in input_string:
        if char in {'A', 'B'}:
            current_substring += char
        else:
            if current_substring:
                substrings.append(current_substring)
                current_substring = ''

    if current_substring:
        substrings.append(current_substring)

    return substrings


def correct_sequence(s, margin=2):
    if len(s) < (20-margin) or len(s) > (20+margin):
        return None
    while len(s) > 20:
        if random.random() < 0.5:
            s = s[1:]
        else:
            s = s[:-1]
    while len(s) < 20:
        # choose character
        if random.random() < 0.5:
            new_char = 'A'
        else:
            new_char = 'B'
        # choose position
        if random.random() < 0.5:
            s = new_char + s
        else:
            s = s + new_char

    return s


def postproc_sequences(substrings):
    good_seq = []
    for s in substrings:
        result = correct_sequence(s, margin=2)
        if result is None:
            continue
        good_seq.append(result)
    return good_seq


def random_bitflip(seq):
    i = random.randint(0, len(seq) - 1)
    if seq[i] == 'A':
        seq = seq[:i] + 'B' + seq[(i+1):]
    else:
        seq = seq[:i] + 'A' + seq[(i+1):]
    return seq


def postproc_and_pad_sequences(substrings, n_batch, old_sequences, rng, possible_sequences, pad_random=False):
    rec_seq = []
    for s in substrings:
        result = correct_sequence(s, margin=0)
        if result is None:
            continue
        rec_seq.append(result)
    good_seq = [it for it in rec_seq if it not in old_sequences]
    non_unique_seq = [it for it in rec_seq if it in old_sequences]
    print(f'Received {len(substrings)} from the LLM...Filtered to {len(good_seq)} new sequences')
    if len(good_seq) < n_batch:
        for result in non_unique_seq:
            # do a random bit flip if this sequence is already present
            while (result in good_seq) or (result in old_sequences):
                result = random_bitflip(result)
            good_seq.append(result)
        print(f'Got {len(good_seq)} new sequences with 1st round bit flipping')
    while len(good_seq) < n_batch:
        for result in good_seq:
            # do a random bit flip if this sequence is already present
            while (result in good_seq) or (result in old_sequences):
                result = random_bitflip(result)
            good_seq.append(result)
        print(f'Got {len(good_seq)} new sequences with 2nd round bit flipping')
    # add random sequences to fill out the batch if < n_batch were given
    while (len(good_seq) < n_batch) and pad_random:
        new_seq = rng.choice(possible_sequences, 1)
        if (new_seq in old_sequences) or (new_seq in good_seq):
            continue
        good_seq.append(new_seq)
    # down-select to only the number we wanted if the LLM gave more
    if len(good_seq) > n_batch:
        print(f'Got {len(good_seq)} in this batch, wanted {n_batch}')
        # good_seq = rng.choice(good_seq, n_batch, replace=False)
        good_seq = good_seq[:n_batch]  # take them in the order they were suggested
    return good_seq


def send_message(messages, client, model,
                 temperature=0):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=temperature,
        messages=messages,
    )

    return message


def build_user_message(batch_prompt, result):
    text = batch_prompt.replace("<result>", result)
    message = {"role": "user", "content": [{"type": "text", "text": text}]}
    return message


def build_llm_message(result):
    text = result
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    return message


def extract_seq_from_init_message(message):
    try:
        seq = message['content'][0]['text'].replace('\\n', '\n').split('started:\n', 1)[1]
    except:
        seq = ""
    return seq


def extract_seq_from_user_message(message):
    try:
        seq = message['content'][0]['text'].replace('\\n', '\n').split('\n', 1)[1].split('Note', 1)[0]
    except:
        seq = ""
    return seq


# def extract_results_from_messages(messages):
#     seq_init = [extract_seq_from_init_message(messages[0])]
#     user_messages = [it for it in messages if it['role'] == 'user']
#     seq_lines = seq_init
#     for it in user_messages[1:]:
#         seq_lines += [extract_seq_from_user_message(it)]
#     iteration_scores = []
#     for sl in seq_lines:
#         scores = [float(it.split(':')[1]) for it in sl.split('\n')[:-1]]
#         iteration_scores.append(scores)
#     return iteration_scores

def extract_results_from_messages(messages):
    old_sequences = []
    seq_init = [extract_seq_from_init_message(messages[0])]
    user_messages = [it for it in messages if it['role'] == 'user']
    seq_lines = seq_init
    for it in user_messages[1:]:
        seq_lines += [extract_seq_from_user_message(it)]
    iteration_scores = []
    for sl in seq_lines:
        seq = [it.split(':')[0] for it in sl.split('\n')[:-1]]
        is_new = [it not in old_sequences for it in seq]
        scores = [float(it.split(':')[1]) for it in sl.split('\n')[:-1]]
        scores = [it for i, it in enumerate(scores) if is_new[i]]
        old_sequences += seq
        iteration_scores.append(scores)
    return iteration_scores


def plot_results_from_messages(messages):
    iteration_scores = extract_results_from_messages(messages)
    fig, ax = plt.subplots()
    top_range = 0
    for i, it in enumerate(extract_results_from_messages(messages)):
        ax.plot(np.ones(len(it))*(i+1)+np.random.randn(len(it))*0.05, it, '.')
        top_range = max(top_range, max(it))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance from Target')
    ax.set_ylim(0, top_range*1.05)
    return fig, ax


def sorted_cumulative_scores(iteration_scores):
    out = [sorted(iteration_scores[0])]
    for it in iteration_scores[1:]:
        out.append(sorted(it+out[-1]))
    return out


def k_below_d(scores, d):
    out = []
    for iter in scores:
        out.append(len([it for it in iter if it < d]))
    return out
