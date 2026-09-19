# Security policy

This repository is public. It contains demo code, a synthetic sample corpus and
written design notes; it does not contain credentials, client material or
copyrighted industry standards.

## Reporting a vulnerability

Use **Security → Report a vulnerability** on GitHub (private vulnerability
reporting is enabled on this repository). That opens a private thread with me
and keeps the details out of public view.

Please do not open a public issue for a security problem. If you cannot use the
GitHub flow, send a message on [LinkedIn](https://linkedin.com/in/nadya-nadya-404309104)
asking for a private channel, without including the details.

Useful reports include the affected version or commit, a reproduction, and the
impact you can see. This is a personal portfolio project, so expect an
acknowledgement within a few days rather than an enterprise SLA.

### In scope

- The application code in `demo1_doc_intelligence/` (ingestion, retrieval,
  citation validation, the Streamlit interface).
- The deployment configuration in this repository (`.streamlit/`, the
  Streamlit Community Cloud setup, the GitHub Actions workflow).
- Dependency or supply-chain issues that affect how this project is built.

### Out of scope

- Vulnerabilities in Groq, Streamlit, Hugging Face, ChromaDB or any other
  upstream service. Report those to the provider.
- The content of the documents a user chooses to index. The application treats
  retrieved text as untrusted data, but it cannot vouch for what a document
  says.
- Findings that require a leaked key of your own to exploit.

## How this project handles secrets and data

**Secrets.** No credential is stored in the repository. The application reads
`GROQ_API_KEY` from the environment: `.env` locally and Streamlit secrets on the
hosted demo. `.gitignore` excludes `.env`, `secrets.toml`, key files and every
PDF, and GitHub secret scanning with push protection is enabled, so a key would
be blocked before it landed. If you believe a key of yours was exposed here,
rotate it first and then tell me: a leaked key is leaked even if the commit is
deleted afterwards.

**What leaves your machine.** Retrieval runs locally. Only the question and the
retrieved excerpts are sent to Groq for answer generation. The documents you
index are never uploaded by this application.

**What the hosted demo holds.** The deployed demo indexes the synthetic sample
corpus in `demo1_doc_intelligence/sample_docs/`, two short documents written for
this repository. It is not connected to any real standard or client document.

**Prompt injection.** Document text is treated as data, not instructions. Every
excerpt handed to the model is wrapped in `<untrusted_source>` tags with an
explicit instruction to ignore anything inside them that looks like a command,
and no answer is shown unless each claim quotes text that is actually present in
a retrieved excerpt.

## Repository protections

Enabled on `nadyapribadi/og-ai-portfolio`:

| Control | State |
|---|---|
| Secret scanning | enabled |
| Secret scanning push protection | enabled |
| Dependabot alerts | enabled |
| Dependabot security updates | enabled |
| CodeQL default setup | configured |
| Private vulnerability reporting | enabled |

Not enabled: secret scanning for non-provider patterns and validity checks.
GitHub only offers those with GitHub Advanced Security, so they are left off
rather than paid for on a portfolio repository.

## Dependencies

Dependencies are pinned in `demo1_doc_intelligence/requirements.txt`, with the
full transitive graph in `requirements.lock.txt`. Dependabot opens the update
pull requests; the test suite in `demo1_doc_intelligence/tests/` is the gate.
