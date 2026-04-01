# Security Policy

## Supported Versions

We actively support the following versions of Fennel with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.3.x   | :x:                |
| < 1.3   | :x:                |

## Reporting a Vulnerability

We take the security of Fennel seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Email**: Send details to stephan.meighenberger@gmail.com
2. **Private Security Advisory**: Use GitHub's [private vulnerability reporting](https://github.com/MeighenBergerS/fennel/security/advisories/new)

### What to Include

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We will acknowledge your report within 48 hours
- **Status Update**: We will provide a detailed response within 7 days indicating next steps
- **Fix Timeline**: We aim to release a fix within 30 days for critical vulnerabilities
- **Disclosure**: We will coordinate with you on public disclosure timing

### What to Expect

After reporting a vulnerability:

1. We will confirm receipt of your report
2. We will investigate and validate the vulnerability
3. We will develop and test a fix
4. We will release a security patch
5. We will publicly disclose the vulnerability (with credit to you, if desired)

## Security Best Practices

When using Fennel:

- Always use the latest version
- Keep dependencies up to date
- Follow secure coding practices when integrating Fennel
- Validate all user inputs before passing to Fennel functions
- Run Fennel in a sandboxed environment when processing untrusted data

## Known Security Considerations

Fennel is primarily a scientific computing library. However, be aware:

- **Input Validation**: While v2.0+ includes comprehensive input validation, always validate data from untrusted sources
- **Resource Usage**: Large energy values or extensive wavelength arrays can consume significant memory
- **File Operations**: Fennel reads parametrization data from pickle files - ensure these files are from trusted sources

## Updates and Patches

Security updates will be:

- Released as patch versions (e.g., 2.0.1)
- Documented in [CHANGELOG.md](CHANGELOG.md)
- Announced in GitHub releases
- Tagged with the `security` label

## Contact

For any security concerns, contact:
- **Email**: stephan.meighenberger@gmail.com
- **GitHub**: [@MeighenBergerS](https://github.com/MeighenBergerS)

---

Thank you for helping keep Fennel and its users safe!
