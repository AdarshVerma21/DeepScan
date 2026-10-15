# DeepScan conventions

- Use Python 3.11, type annotations, small testable functions, and structured logging.
- Never add model training or fine-tuning code. Pretrained inference and classical forensic signals only.
- Keep thresholds and weights in `app/config.py`.
- Handle optional model, face, audio, and metadata failures explicitly and preserve user-facing limitations.
- Do not hardcode API keys or other secrets.
