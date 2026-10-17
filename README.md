# Generative AI Applications Project: Generating Company Names with a Character-Level Transformer

## Project Description
TBD

This project connects to my earlier capstone projects, which worked with a
company register, company registrations, a bankruptcy score and negative
company news. A later capstone project will build a due diligence agent that
matches company names against registers and lists. Generated names could
serve as test inputs for that step, names that look real but should not match
anything.

## What Was Built
- A data preparation script (`prepare_dataset.py`) with its settings and
  name rules (`constants.py`). It turns the Companies House snapshot into the
  sample `company_names.csv` and the register keys `register_key_hashes.npy`.
- A tokenizer module (`tokenizer.py`) with the character tokenizer of the
  notebook, adapted from the course.
- A reference model (`ngram.py`), a character n-gram model without deep
  learning, used in the notebook as a baseline.
- A reproducibility file (`requirements.txt`), generated with `pip freeze`
  inside the project's own virtual environment.
- TBD

## Dataset
Companies House, Free Company Data Product, the monthly list of all live
companies on the UK register, snapshot of 2026-10-01:
https://download.companieshouse.gov.uk/en_output.html

Companies House imposes no rules on how the information on the public
register is used, but rights of third parties in the data remain. The names
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
  live company. The key follows the "same as" rules in Schedule 3 of the
  Names and Trading Disclosures Regulations 2015
  (https://www.legislation.gov.uk/uksi/2015/17/schedule/3/made), as the
  Companies House guidance "Incorporation and names" describes them. It is my
  own implementation and not the official test, so a match is only a possible
  collision with a real company.

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
TBD

## How to Run the Project
TBD

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
