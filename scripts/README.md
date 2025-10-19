# Scripts

Environment bootstrap helpers used by contributors and CI:

- `setup_env.sh` — POSIX shell script that creates a virtual environment, installs the
  project (editable) and dependencies, and downloads optional extras when available.
- `setup_env.ps1` — Windows PowerShell variant performing the same actions.

Typical usage:

```bash
./scripts/setup_env.sh        # Linux/macOS
powershell -File scripts\setup_env.ps1   # Windows
```

After running either script, activate the reported virtual environment and invoke
`pytest` or the demo programs as needed.
