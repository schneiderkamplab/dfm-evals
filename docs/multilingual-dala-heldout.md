# Multilingual DaLA Held-Out Tasks (2026-09-30)

CPU-only implementation for nineteen non-Danish/non-English languages plus
English. No scheduler, headline-average, existing task, or training files were
modified by this implementation.

## Interface

- Config: `config/dfm_evals_multilingual_dala_heldout.yaml` in HRM-Text.
- Scheduler entries: `config/dfm_dala_heldout_registry_20260930.json`.
- Data evidence: `config/dfm_dala_heldout_20260930.json`.
- Actual task-load/context receipt: `config/dfm_dala_heldout_preflight_20260930.json`.
- Task keys and suite names: `dala_<language>` and `gec_dala_<language>`.
- Absolute file selectors load `dala_heldout` and `gec_dala_heldout` without
  changing the shared plugin registry. The installed `dfm-evals/.venv` is needed;
  the HRM training environment does not contain `inspect_ai`.
- Acceptability scorer `linguistic-acceptability` reuses existing macro-F1/MCC
  implementations. Prompts require literal `yes` or `no`; other output is invalid.
- Correction reuses `gec_dala_scorer`, `exact_match` with mean/stderr. It retains
  the existing leading/trailing-whitespace normalization, not semantic matching.

Languages: nb, nn, sv, is, fo, nl, pl, de, fr, es, it, cs, pt_pt, fi, et,
ca, el, ro, uk. English is separate, for the English headline. Danish is excluded.

## Held-Out Evidence

All twenty local sources have completed test files. Northern-language artifacts
come from the producer's finalized six-language recovery release. EN/NL and the
twelve additional European languages use their checksummed published package
provenance. Upstream repository revisions are recorded for all languages; the
six local-only held-out releases have no invented Hugging Face release revision.

The preparation scans every producer train and test pair, comparing pair IDs,
document SHA256 and original/corrupted text normalized with NFC, casefold and
whitespace collapse. All twenty scans found zero overlaps. Every input file and
producer manifest is hashed; complete selected pair IDs and test sizes are saved.
The runtime only reads pinned test files; missing, changed, wrong-split or
wrong-language data fails closed. It never falls back to train or validation.

This is **same-producer train/test exclusion**, not proof of exclusion from every
DFM11/DFM12 pretraining, translation or synthetic source. No fuzzy/semantic or
full inherited-corpus decontamination claim is made.

## Selection And Prompts

Provisional routine cap: 2,000 samples per task/language, selected as 1,000 complete
pairs by SHA256 ranking of `4242:language:pair_id`. The same pairs feed both tasks;
both clean and corrupted controls are retained and kept together across four
shards. The selected IDs are fixed and do not depend on model outcomes.

Producer-native prompts are unchanged for NB/NN/SV/IS/FO/NL/PL/EN. The twelve newer
packages originally used English instructions naming the language; new native
evaluation translations are stored separately, with the original prompts retained
in the manifest. These translations are agent-authored, not human-certified.
Portuguese explicitly requests European Portuguese, not Brazilian Portuguese.

Temperature is zero. Output budgets are 32 tokens for acceptability and 512 for
correction. Actual CPU loading of all forty tasks and raw Gemma training-template
tokenization checked 80,000 samples: maximum prompt 236 tokens, maximum reference
146 tokens, zero truncation under a 4,096-token context. No model was queried.

## Quality Limitations

The twelve European package manifests describe automated pair-audit-pass data.
The six northern sources remain dictionary/rule-screened candidate labels, not
newly pair-audited held-out labels. Dutch uses checker screening plus review
exclusions. The English published manifest explicitly records
`checker_screened_with_known_source_label_errors`. None of these distinctions
constitutes native-speaker certification. Results should retain these benchmark
quality caveats; training-row acceptance does not establish held-out label quality.

Preparation and context-check entry points are the new scripts
`dfm-evals/scripts/prepare_dala_heldout_manifest.py` and
`dfm-evals/scripts/check_dala_heldout_tasks.py`. Both refuse to overwrite their
completed output artifacts. Future selection changes require a new manifest and
new validation receipt rather than silently changing this baseline.
