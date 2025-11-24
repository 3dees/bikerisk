"""Quick diagnostic to check Anthropic API key configuration."""
import os
from pathlib import Path

print("=" * 80)
print("ANTHROPIC API KEY DIAGNOSTIC")
print("=" * 80)

# Step 1: Check if .env file exists
env_file = Path(".env")
print(f"\n[1] .env file exists: {env_file.exists()}")
if env_file.exists():
    print(f"    Path: {env_file.absolute()}")
    content = env_file.read_text(encoding='utf-8')
    if "ANTHROPIC_API_KEY" in content:
        # Extract just the key part (don't print full key for security)
        for line in content.split('\n'):
            if line.strip().startswith('ANTHROPIC_API_KEY'):
                key_preview = line.split('=')[1][:20] + "..." if '=' in line else "ERROR"
                print(f"    Found in .env: {key_preview}")
    else:
        print("    WARNING: ANTHROPIC_API_KEY not found in .env file")

# Step 2: Check environment variable BEFORE loading .env
print(f"\n[2] Environment variable (before dotenv):")
key_before = os.environ.get("ANTHROPIC_API_KEY")
if key_before:
    print(f"    ANTHROPIC_API_KEY: {key_before[:20]}... (length: {len(key_before)})")
else:
    print("    ANTHROPIC_API_KEY: NOT SET")

# Step 3: Load .env with dotenv
print(f"\n[3] Loading .env with python-dotenv...")
from dotenv import load_dotenv
result = load_dotenv()
print(f"    load_dotenv() returned: {result}")

# Step 4: Check environment variable AFTER loading .env
print(f"\n[4] Environment variable (after dotenv):")
key_after = os.environ.get("ANTHROPIC_API_KEY")
if key_after:
    print(f"    ANTHROPIC_API_KEY: {key_after[:20]}... (length: {len(key_after)})")
    
    # Validate key format
    if key_after.startswith("sk-ant-"):
        print(f"    ✓ Key format looks valid (starts with 'sk-ant-')")
    else:
        print(f"    ✗ WARNING: Key does not start with 'sk-ant-' (got: {key_after[:10]}...)")
else:
    print("    ANTHROPIC_API_KEY: STILL NOT SET")
    print("    ✗ ERROR: dotenv did not load the key!")

# Step 5: Test Anthropic client initialization
print(f"\n[5] Testing Anthropic client initialization...")
try:
    from anthropic import Anthropic
    client = Anthropic(api_key=key_after)
    print(f"    ✓ Anthropic client created successfully")
    print(f"    Using API key: {key_after[:20]}...")
except Exception as e:
    print(f"    ✗ ERROR creating client: {e}")

# Step 6: Test actual API call (minimal)
print(f"\n[6] Testing minimal API call...")
try:
    from harmonization.anthropic_client import messages_create_with_retries
    
    response = messages_create_with_retries(
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        temperature=0,
        messages=[{
            "role": "user",
            "content": "Say 'OK'"
        }]
    )
    
    reply = response.content[0].text
    print(f"    ✓ API call successful!")
    print(f"    Response: {reply}")
    
except Exception as e:
    print(f"    ✗ API call failed: {e}")
    import traceback
    print(f"\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
