# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Updated code that records the fine-grained perplexity metrics per subdomain to also include perplexity over words, characters, bytes, and also bits per byte
- Added option to track avg logit per token type
- Added script that uses the tango steps as functions, and bypasses the tango caching mechanism, for simpler execution
- minimal example of how to run Paloma from HF hub as well as step to output results in jsonl.gz format

### Fixed

- Fixed incorrect paths in readme
- Fixed model names written to gsheet by run_lm_eval.py
- Fixed hf_olmo module and function name that has changed

### Changed

- Updated default image in tango-in-beaker.yml
