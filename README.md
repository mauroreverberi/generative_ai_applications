# Generative AI Applications Project: Generating Company Names with a Character-Level Transformer

**GitHub repository:** https://github.com/mauroreverberi/generative_ai_applications
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I train a small decoder-only Transformer from scratch with PyTorch and the
Hugging Face Transformers library. It writes new UK company names, one
character at a time. It learns from 80,142 names of
real private limited companies on the Companies House register. The
evaluation checks whether the names are plausible and varied, and whether
they copy training names or match companies that already exist.

This project connects to my earlier capstone projects, which worked with a
company register, company registrations, a bankruptcy score and negative
company news. A later capstone project will build a due diligence agent that
matches company names against registers and lists. Generated names could
serve as test inputs for that step, names that look real but should not match
anything.

## What Was Built
- A Jupyter notebook (`generative_model.ipynb`) with the whole workflow: data
  checks, a character n-gram reference model, the Transformer built with the
  GPT-2 classes of Hugging Face Transformers (random weights, no
  pretraining), my own training loop with a loss curve, the generation at four
  temperatures, the checks against the training names and the register, the
  choice of the temperature, a qualitative evaluation of new names, collisions
  and failure cases, and a short summary. The project overview calls it
  `generative_ai.ipynb`, I use the file name from the submission instructions.
- A written report (`Generative_AI_Analysis_Report.pdf`, with the Word source
  `Generative_AI_Analysis_Report.docx`) that explains the task, the data, the
  model, the training, the evaluation, the limitations and the ethical risks.
  The project overview calls it `module_summary.pdf`, I use the file name from
  the submission instructions.
- A data preparation script (`prepare_dataset.py`) with its settings and
  name rules (`constants.py`). It turns the Companies House snapshot into the
  sample `company_names.csv` and the register keys `register_key_hashes.npy`.
- A tokenizer module (`tokenizer.py`) with the character tokenizer of the
  notebook, adapted from the course.
- A reference model (`ngram.py`), a character n-gram model without deep
  learning, used in the notebook as a baseline.
- A folder `reference_run` with the weights of the trained model
  (`model_weights.pt`), its loss history (`training_history.csv`) and the
  generated names (`generated_names.csv`). The CSV holds the raw outputs,
  including names with a register-key match and malformed names, so it is a
  record of the run and not a list of approved test names.
- A reproducibility file (`requirements.txt`), generated with `pip freeze`
  inside the project's own virtual environment.

## Dataset
Companies House, Free Company Data Product, the monthly list of all live
companies on the UK register, snapshot of 2026-10-01:
https://download.companieshouse.gov.uk/en_output.html

Companies House does not limit how the information on the public register may
be used. Possible rights of third parties in the data still apply. The names
are real register entries, nothing is synthetic or generated, and I did not
use this source in my earlier capstone projects.

The snapshot (494 MB) is not included. `prepare_dataset.py` reads it once and
writes two files:

- `company_names.csv` (3.6 MB), a random sample of 100,000 names of private
  limited companies with seed 42 and the columns `name`, `group` and `split`.
  Similar names like ACME HOLDINGS LTD and ACME HOLDINGS (UK) LIMITED form
  one group, and the groups are split about 80/10/10 into train, validation
  and test.
- `register_key_hashes.npy` (45 MB), a 64-bit hash of a name key for every
  live company. The key is based on the "same as" rules in Schedule 3 of the
  Names and Trading Disclosures Regulations 2015
  (https://www.legislation.gov.uk/uksi/2015/17/schedule/3/made), as the
  Companies House guidance "Incorporation and names" describes them. It is my
  own implementation and not the official test. It does not apply every rule
  in the same order, for example ACME L.T.D. and ACME LTD get different keys.
  So a match is only a possible collision with a real company, and a name
  without a match can still be the same as a registered name.

Both files are included. To build them from the original file:
```
python prepare_dataset.py --download
```

The download page lists only the latest snapshot. If the file of 2026-10-01
is no longer online, a newer one can be passed with `--source`, but then the
results will differ. The file of 2026-10-01 has the SHA-256
`29d54c17e5bf39862a08d5ebf1ae83abc079c8591c8c6d1bab0a29879accb4ee`, and from
it the script writes `company_names.csv` with SHA-256
`b37e08c9ac6750aba2e5c12b95f75f88802593ff7300535d922be1a50786336b` and
`register_key_hashes.npy` with SHA-256
`427970c7b750a5d57ab9a647f126c6b5872495660941be70644047ad89a1855a`.

## Main Results
The model is a `GPT2LMHeadModel` of the Hugging Face Transformers library,
created from a `GPT2Config` with random weights. Nothing is pretrained or
downloaded. It has 3.2 million parameters, 4 blocks with 8 attention heads and
an embedding size of 256. I trained it with my own PyTorch loop for 20 epochs
on the CPU and kept epoch 20, which had the lowest validation loss. On the
test split it reaches 1.1513 nats per character (perplexity 3.16), against
1.3819 for the best character 5-gram model. The names are generated with
`model.generate` of the library.

I generated 2,000 names at each of the temperatures 0.5, 0.8, 1.0 and 1.2.
Between 99.45% and 100% of them end with LTD, LTD. or LIMITED, and 16 of the
8,000 names are exact copies of a training name. The register check matters
more. The key of 44.25% of the names at temperature 0.5 matches a company in
the snapshot, 21.15% at 0.8, 10.15% at 1.0 and 5.35% at 1.2. Almost none of
these names has the key of a training name, so most matches seem to be common
names that real companies already use. Without further checks the names are
not safe as test names that should not match real companies. At low
temperature the model repeats generic names and overuses words like
CONSULTING, at high temperature it writes more misspelled words, wrong endings and open
brackets.

A rule on the validation names chose temperature 1.0 for the detailed
examples. This is an exploratory choice. After removing invalid, repeated
and register-matching names, 1.0 is still the closest on the same measure,
but I added this check only after the rule, so it does not prove which
temperature is best for the names I would use.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment (Python 3.12 or newer, tested
   with 3.14):
   ```
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start Jupyter:
   ```
   jupyter lab
   ```
5. Open `generative_model.ipynb` and run all cells
   (Kernel > Restart Kernel and Run All Cells).

The notebook uses CUDA if it is available, otherwise the CPU. I do not use the
GPU of a Mac (MPS), because two runs with the same seed gave slightly
different results there. On the CPU of my Mac (Apple M1 Max) one epoch took
about 3 to 5 minutes, so the training took about 75 minutes and a full run
took about 80 to 85 minutes. Three training runs on the CPU gave the same loss
in every epoch and the same weights.
I could not test CUDA. If PyTorch does not find an NVIDIA GPU after the
install, install it with the command from https://pytorch.org for your system.

The notebook, this README and the report describe the reference run in
`reference_run`.
With the default settings (`TRAIN_MODEL = True`,
`ANALYSE_REFERENCE_RUN = True`) the notebook trains a new model into
`new_run` but analyses the saved reference run. To analyse the new model, set
`ANALYSE_REFERENCE_RUN = False`. With `TRAIN_MODEL = False` the notebook skips
the training, and a full run took about 3 minutes. After a training run the
notebook prints whether the new run has the same weights and losses as the
reference run. If the generated names differ from the saved ones, for example
on other hardware, the notebook prints this, keeps the new names in
`new_run/names_from_reference_weights.csv` and analyses the saved names.

## Dependencies
`requirements.txt` was created inside the project's own virtual
environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
I stay in the domain of company data and compliance checks. The planned due
diligence agent needs to match company names, and generated names can test
that step with inputs that look real. Such a name should only be used if its
key matches no company in the register, and even then it needs further
checks, for example against similar names, other countries and trademarks.
