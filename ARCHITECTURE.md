# Project Architecture

This repository is organized into focused layers so training, inference, experiments, and UI are separated.

## Folder layout

```text
natvox-ai/
  app.py                       # Streamlit product UI
  main.py                      # Training entrypoint
  inference.py                 # Single-file inference entrypoint
  vocalis_pipeline.py          # TTS -> embedding adaptation demo pipeline
  model.py                     # NATVOXAdapter network
  train.py                     # Training loop and optimizer logic
  dataset.py                   # Paired dataset loading + embedding cache
  audio_utils.py               # Embedding extraction + audio utilities
  config.py                    # Shared project paths/constants
  experiments/
    vocalis_experiment.py      # Notebook-style local experiment runner
    full_pipeline.py           # Full report/ablation/graph generator
  tools/
    validate_dataset.py        # Validate synthetic/natural file pairing
    cleanup_project.py         # Remove temp files and normalize folders
  paper/
    fig_vocalis_arch.tex       # Adapter architecture figure (LaTeX/TikZ)
  voice_data/
    my_pairs/
      synthetic/               # Synthetic reference audios
      natural/                 # Natural reference audios
  artifacts/
    experiment/                # Experiment outputs
    reports/                   # Full report outputs
```

## Data flow

1. Put paired data in `voice_data/my_pairs/synthetic` and `voice_data/my_pairs/natural`.
2. Run training using `main.py` or Streamlit `Train` page.
3. Model checkpoint and metrics are saved in `artifacts/`.
4. Run `experiments/full_pipeline.py` to generate graphs + report JSON in `artifacts/reports/`.
5. Use `app.py` dashboard for interactive training and inference checks.

## Conventions

- Keep generated outputs inside `artifacts/` only.
- Keep temporary files out of root (`tools/cleanup_project.py` can remove them).
- Keep reusable constants in `config.py` to avoid hardcoded paths.
