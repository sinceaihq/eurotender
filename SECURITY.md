# Security Policy

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting:**
**[Report a vulnerability](https://github.com/sinceaihq/eurotender/security/advisories/new)**

If GitHub reporting is unavailable to you, email **builders@sinceai.fi**.
Please do not open a public issue for a suspected vulnerability.

We aim to acknowledge reports within a few working days. EuroTender is
maintained by a small team, so we do not promise a fixed remediation deadline.
You will be credited in the advisory unless you prefer otherwise.

## Scope

In scope:

- **Code execution through model files.** Loading a model must never execute
  code. Models are stored as `safetensors` plus JSON and loaded without
  `pickle`; any way to make `CpvModel.load` run code is a vulnerability.
- **Personal data in published artifacts.** If you find e-mail addresses,
  phone numbers or other direct identifiers of individuals in the published
  dataset, report it here (or email us) and we will remove it.
- **Supply chain.** Problems in how the package, model or dataset are built
  or published.

Out of scope: wrong CPV predictions (please open a normal issue), and the
content of TED notices themselves, which is published by contracting
authorities through the Publications Office of the European Union.

## Supported versions

Only the latest release receives fixes.
