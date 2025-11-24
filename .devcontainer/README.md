# BikeRisk DevContainer with Claude Code

This devcontainer configuration enables Claude Code to work with remote development in VS Code.

## What's Included

- Python 3.11 base image
- Node.js LTS (required for Claude Code)
- Claude Code CLI installed globally
- Python dependencies from requirements.txt
- VS Code extensions: Claude Code, Python, Pylance
- Streamlit port forwarding (8501)

## Setup Instructions

### Option 1: GitHub Codespaces

1. **Set up your API key:**
   - Go to https://console.anthropic.com/keys and get your API key
   - Go to https://github.com/settings/codespaces
   - Click "New secret"
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your API key
   - Choose which repositories can access it (or select all)

2. **Launch Codespace:**
   - Go to your GitHub repository
   - Click "Code" → "Codespaces" → "Create codespace on [branch]"
   - Wait for the container to build
   - You should see "✅ ANTHROPIC_API_KEY is set" in the terminal

3. **Use Claude Code:**
   - Open the Claude Code extension in VS Code
   - Or run `claude` in the terminal

### Option 2: VS Code Remote - Containers (Local)

1. **Prerequisites:**
   - Install Docker Desktop
   - Install "Dev Containers" extension in VS Code

2. **Set up API key in your environment:**
   ```bash
   # Add to your ~/.bashrc or ~/.zshrc
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

3. **Open in container:**
   - Open this repository in VS Code
   - Press F1 → "Dev Containers: Reopen in Container"
   - Wait for container to build
   - You should see "✅ ANTHROPIC_API_KEY is set" in the terminal

### Option 3: VS Code Remote - SSH

**Note:** This setup works best with Codespaces or local containers. For Remote SSH:
- The devcontainer needs to be on the remote machine
- Node.js and npm must be installed on the remote server
- You may need to manually install Claude Code on the remote server

## Troubleshooting

### "ANTHROPIC_API_KEY is not set"
- For Codespaces: Check GitHub Codespaces secrets settings
- For Local: Ensure the environment variable is set before launching VS Code
- Rebuild the container after adding the API key

### Claude Code not found
- The container installation might have failed
- Try rebuilding: F1 → "Dev Containers: Rebuild Container"

### Python dependencies not installed
- Run manually: `pip install -r requirements.txt`
- Or rebuild the container

## Testing

To verify everything works:
1. Open terminal in VS Code
2. Run: `claude --version`
3. You should see the Claude Code version
4. Open Claude Code extension or run `claude` to start chatting
