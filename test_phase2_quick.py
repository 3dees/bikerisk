"""Quick test of Phase 2 prompt structure."""
import sys
from harmonization.models import Clause
from harmonization.consolidate import build_llm_prompt_for_group

# Create test clauses
c1 = Clause(
    clause_number='5.1.101',
    text='Battery instructions shall include charging temperature range of 0-45°C.',
    standard_name='UL 2271',
    category='user_documentation_and_instructions',
    requirement_type='user_instructions'
)

c2 = Clause(
    clause_number='7.3.2',
    text='User manual must specify charge temperature limits between 5-40°C.',
    standard_name='EN 50604-1',
    category='user_documentation_and_instructions',
    requirement_type='user_instructions'
)

try:
    print("[TEST] Building prompts...")
    system_prompt, user_prompt = build_llm_prompt_for_group([c1, c2], 0)
    
    print(f"\n[TEST] ✓ Prompt generation successful!")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  User prompt: {len(user_prompt)} chars")
    
    print("\n[TEST] System prompt preview:")
    print(system_prompt[:200] + "...")
    
    print("\n[TEST] User prompt preview:")
    print(user_prompt[:500] + "...")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n[TEST] ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
