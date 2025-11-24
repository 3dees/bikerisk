# API Key Setup for BikeRisk Harmonization Layer

Your Anthropic API key has been securely stored in `.env` at the project root.

## How It Works

- **Storage**: `.env` file (Git-ignored by default)
- **Loading**: Python scripts automatically read `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`
- **Security**: Never commit `.env` to version control (already in `.gitignore`)

## Usage

### Automatic (Recommended)
Scripts like `run_phase2_consolidation.py` will load the key automatically:

```powershell
python run_phase2_consolidation.py -i "C:\path\to\data.csv" -o outputs
```

### Manual (Terminal Session)
If you prefer to set it per-session:

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = (Get-Content .env | Select-String "ANTHROPIC_API_KEY" | ForEach-Object { $_ -replace "ANTHROPIC_API_KEY=", "" })

# Or load from .env directly
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Verifying Setup

Check if the key is loaded:

```powershell
python -c "import os; print('Key loaded:', bool(os.getenv('ANTHROPIC_API_KEY')))"
```

## Security Notes

- **Never** share your `.env` file
- **Never** commit API keys to Git
- Rotate keys regularly via Anthropic Console
- If leaked, revoke immediately at https://console.anthropic.com/

## Related Files

- `.gitignore` - Already excludes `.env` and `.env.local`
- `harmonization/anthropic_client.py` - Centralized API client
- `run_phase2_consolidation.py` - Phase 2 consolidation runner
