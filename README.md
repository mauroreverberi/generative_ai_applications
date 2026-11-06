# Generative AI Applications Project: Generating Company Names with a Character-Level Transformer

**GitHub repository:** https://github.com/mauroreverberi/generative_ai_applications
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I train a small decoder-only Transformer from scratch with PyTorch and the
Hugging Face Transformers library. It writes new UK company names, one
character at a time. It learns from 80,142 names of
real private limited companies on the Companies House register. The
evaluation checks whether the names are plausible and varied, how often
they copy a complete training name and how often they have the key of a
company in the register.

This project connects to my earlier capstone projects, which worked with a
company register, company registrations, a bankruptcy score and negative
company news. A later capstone project will build a due diligence agent that
matches company names against registers and lists. Generated names could
serve as test inputs for that step, names that look real but should not match
anything.

## What Was Built
- A Jupyter notebook (`generative_model.ipynb`) with the whole workflow. It
  starts with data checks, the character tokenizer (adapted from the course)
  and a character n-gram reference model. Then it builds the Transformer with
  the GPT-2 classes of Hugging Face Transformers (random weights, no
  pretraining), defines the same architecture once more in my own PyTorch
  classes and compares the logits of both, and trains the model with my own
  training loop and a loss curve. After that it generates names at four temperatures, checks them
  against the training names and the register, chooses the temperature,
  evaluates new names, collisions and failure cases, and ends with a short
  summary.
  The project overview calls it `generative_ai.ipynb`, I use the file name
  from the submission instructions.
- A written report (`Generative_AI_Analysis_Report.pdf`) that explains the
  task, the data, the model, the training, the evaluation, the limitations and
  the ethical risks.
  The project overview calls it `module_summary.pdf`, I use the file name from
  the submission instructions.
- A data preparation script (`prepare_dataset.py`) with its settings and
  name rules (`constants.py`). It turns the Companies House snapshot into the
  sample `company_names.csv` and the register keys `register_key_hashes.npy`.
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

I downloaded the snapshot (494 MB) on 2026-10-16. It is not included,
`prepare_dataset.py` reads it once and writes two files:

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
`29d54c17e5bf39862a08d5ebf1ae83abc079c8591c8c6d1bab0a29879accb4ee`.

This is the output of the script for that file, without the progress lines.
The two SHA-256 values in the last lines are the same as the ones of the files
in this repository, which you can check with `shasum -a 256 company_names.csv`
(on Windows `certutil -hashfile company_names.csv SHA256`).
```
5,704,711 live companies in the snapshot, 1 empty or broken rows skipped
removed, other category: 428,049
removed, empty name: 0
removed, character outside the allowed set: 6,984
removed, longer than 60 characters: 2,415
removed, other ending: 7,971
5,259,292 names kept
11 repeated names removed
95,078 name groups, 1,901 of them with more than one name (6,823 names)
train: 80,142 names (80.1%)
validation: 9,835 names (9.8%)
test: 10,023 names (10.0%)
5,673,818 different keys in the register
wrote company_names.csv, SHA-256 b37e08c9ac6750aba2e5c12b95f75f88802593ff7300535d922be1a50786336b
wrote register_key_hashes.npy, SHA-256 427970c7b750a5d57ab9a647f126c6b5872495660941be70644047ad89a1855a
```

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
the snapshot, 21.15% at 0.8, 10.15% at 1.0 and 5.35% at 1.2. Of the 1,618
generated names whose key matched the register, only 37 had the key of a
training name and 16 were exact copies. I did not test whether the model
reproduced parts of other names. Without further checks the names are not
safe as test names that should not match real companies. At low temperature
the model repeats generic names and overuses words like
CONSULTING. At 1.2, 11 names have a wrong ending, and some words in the
examples are hard to read or brackets stay open.

I used temperature 1.0 for the examples because its similarity scores were
closest to those of the validation names. I wrote this rule after I had
already looked at the test names, so it is an exploratory choice.

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

The notebook uses CUDA if it is available, otherwise the CPU. On the CPU of my
Mac (Apple M1 Max) one epoch took about 4 minutes in the saved run, so the
training took about 80 minutes and the full run about 81 minutes. Other runs
had epochs between 3.5 and 6 minutes and full runs between 69 and 100 minutes.
The notebook trains the model in every full run. Repeated runs on the CPU gave the same loss in every
epoch and the same generated names.

The notebook, this README and the report describe my run on the CPU. On other
hardware, for example with CUDA, the losses and the generated names can
differ.

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
