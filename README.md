# Codes and data for "Large language models design sequence-defined macromolecules via evolutionary optimization"

Note this repository contains codes and data files for the manuscript. A snapshot of this repository, frozen at the time of submission, can be found on Zenodo (INSERT DOI) 

# Codes

## LLM codes
- `run_claude.py` - the routine for performing LLM-based rollouts; intended for command line execution using argparse
- `message_utils.py` - utilities for constructing and parsing messages for LLM I/O
- `model_utils.py` - lightweight utilities for retrieving formatted predictions from the RNN ensemble
- `target_defs.py` - defines the sequence, locations, and natural language descriptions of the target structures
- `ask_about_oracle.ipynb` - asks the LLM to speculate about the nature of the optimization task

## other algorithms
- `active_learning.ipynb` - use EI acquisition with RF surrogate to label new sequences; includes an unused tokenization scheme
- `evolutionary_algorithm.ipynb` - use DEAP library to perform evolutionary optimization
- `random_sampling.ipynb` - sample sequences randomly from all possible sequences

## postprocessing
- `process_aggregated_logs.py` - reads data from the raw log files and prepares them for visualization
- `process_sample_rollouts.py` - reads data from the raw log files and prepares individual rollouts

## visualization
- `figure1b.ipynb` - renders panel b of Fig. 1
- `figure1efg.ipynb` - renders the last row of Fig. 1 (panels e-g)
- `figure2.ipynb` - renders all of Fig. 2
- `figure_si.ipynb` - renders Figs. S1 and S2
- `figure_md_validation.ipynb` - renders Fig. S3

# Data files

- `prompts/`
  - `prompt-scientific-v4.4.yml` - the full text of the scientific prompt, to be read by `run_claude.py`
  - `prompt-oracle-v4.4.yml` - the full text of the oracle prompt, to be read by `run_claude.py`
- `models/` - the TorchScript RNN models used to make predictions
- `data/`
  - `embeddings` - calculated embeddings for a collection of sequences from our prior work
  - `llm-logs` - the raw logs obtained from the Claude 3.5 Sonnet LLM (other algorithms made to look like the LLM logs after the fact)
  - `llm-logs-opus` - the raw logs obtained from the Claude 3.0 Opus LLM (used in the first draft of the article, replaced by Claude 3.5 Sonnet) 
  - `all-rollouts-kltd.csv` - postprocessed logs for all the rollouts using the "top $k < d^*$" metric
  - `all-rollouts-topkd.csv` - postprocessed logs for all the rollouts using the "mean $d$ for top $k$" metric
  - `sample-rollout-membranes-x-3.csv` - postprocessed logs for a single rollout replica, `x` = each algorithm type
  - `snapshots` - png snapshots of MD simulation results at different locations in the manifold
  