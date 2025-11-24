#!/bin/bash

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY is not set!"
    echo ""
    echo "To use Claude Code, you need to set your Anthropic API key:"
    echo ""
    echo "1. Get your API key from: https://console.anthropic.com/keys"
    echo ""
    echo "2a. For GitHub Codespaces:"
    echo "    - Go to: https://github.com/settings/codespaces"
    echo "    - Add ANTHROPIC_API_KEY as a repository or user secret"
    echo "    - Rebuild the container"
    echo ""
    echo "2b. For local development:"
    echo "    - Set it in your environment before starting VS Code"
    echo "    - Or run: export ANTHROPIC_API_KEY=your-api-key-here"
    echo ""
else
    echo "✅ ANTHROPIC_API_KEY is set"
    echo "You can now use Claude Code! Try running 'claude' in the terminal."
fi
