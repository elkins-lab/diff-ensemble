# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-06-30

### Added
- Added `readme` metadata to `pyproject.toml` to correctly render the project description on PyPI.
- Integrated `hypothesis` property-based testing and parameterized core testing for robust shape and dimension contract verification.
- Added explicit shape validation checks to functions in `diff_ensemble/model.py`.
- Added mock PED ensemble parity test for structurally valid verification in the test suite.

### Fixed
- Fixed `Type` syntax error with PyPI `numpy` stubs under Python 3.12 by setting `python_version = "3.12"` in mypy configuration.
- Fixed Biotite `AtomArrayStack` strict numpy dtype type-checking assignment errors in `diff_ensemble/io.py`.

### Documentation
- Updated URLs and badges in `README.md` to point to the correct GitHub actions and PyPI configurations.

## [0.1.2] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.
