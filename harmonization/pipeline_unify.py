"""
End-to-end orchestration pipeline for harmonization layer.

This module orchestrates the complete harmonization workflow:
1. Load clauses from tagged CSV files
2. Group clauses by similarity (TF-IDF embeddings)
3. Consolidate groups using LLM
4. Generate HTML report

For the tiny test, this loads test_tagging_sample_filtered.csv and generates a simple report.
"""

import os
import csv
from typing import List, Callable, Optional
import numpy as np
from anthropic import Anthropic

from harmonization.models import Clause, RequirementGroup
from harmonization.grouping import load_and_group_clauses, load_clauses_from_tagged_csv, compute_embeddings, group_clauses_by_similarity
from harmonization.consolidate import consolidate_groups, stub_call_llm
from harmonization.report_builder import save_html_report
from harmonization.anthropic_client import get_anthropic_client


# =============================================================================
# EMBEDDING FUNCTION (Real Embeddings)
# =============================================================================

def get_embedding(text: str, api_key: str = None) -> List[float]:
    """
    Get embedding for text using Claude API.

    Args:
        text: Text to embed
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Returns:
        List of floats representing the embedding vector
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    # Use Claude's embedding API (via messages with a special prompt)
    # Note: Anthropic doesn't have a dedicated embedding endpoint yet,
    # so we'll use a simple approach with the messages API
    try:
        # Truncate very long texts to avoid token limits
        if len(text) > 2000:
            text = text[:2000]

        # Use a minimal prompt to get semantic representation
        # This is a workaround until Claude has a proper embedding API
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"Summarize this requirement in exactly 5 key technical terms: {text}"
            }]
        )

        # Extract text response and convert to a simple embedding
        # This is a STUB - in production, you'd use a proper embedding model
        summary = response.content[0].text

        # Simple hash-based embedding (for demonstration)
        # In production, use OpenAI embeddings or Voyage AI
        embedding = [float(hash(word) % 1000) / 1000.0 for word in summary.split()[:100]]

        # Pad to fixed size (100 dimensions)
        while len(embedding) < 100:
            embedding.append(0.0)

        return embedding[:100]

    except Exception as e:
        print(f"[EMBEDDING] Error getting embedding: {e}")
        # Fallback: return zero vector
        return [0.0] * 100


def get_openai_embedding(text: str, api_key: str = None) -> List[float]:
    """
    Get embedding using OpenAI's text-embedding-3-small model.

    This is the RECOMMENDED approach for production use.

    Args:
        text: Text to embed
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

    Returns:
        List of floats (1536 dimensions for text-embedding-3-small)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Truncate long texts
    if len(text) > 8000:
        text = text[:8000]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# =============================================================================
# LLM CONSOLIDATION FUNCTION (Real LLM)
# =============================================================================

def call_llm_for_consolidation(prompt: str, api_key: str = None, use_claude: bool = True) -> str:
    """
    Call LLM to consolidate requirement groups.

    HARD LOCK POLICY:
    Consolidation MUST be performed with Claude Sonnet 4.5.
    This ensures:
      - Deterministic JSON format
      - Consistent normative language
      - Stable consolidation logic across standards
      - Prevention of fallback hallucination behavior

    Args:
        prompt: The consolidation prompt
        api_key: API key (Anthropic). If None, reads from ANTHROPIC_API_KEY env var.
        use_claude: MUST be True. OpenAI models are NOT permitted.

    Returns:
        JSON string response from Claude Sonnet 4.5

    Raises:
        ValueError: If API key not found or use_claude is False
        RuntimeError: If LLM call fails
        json.JSONDecodeError: If LLM returns invalid JSON
    """
    import json

    # Enforce hard lock: NO OpenAI fallback allowed
    if not use_claude:
        raise ValueError(
            "OpenAI models are NOT permitted for harmonization consolidation. "
            "Use ONLY Claude Sonnet 4.5."
        )

    # Initialize client with centralized helper
    client = get_anthropic_client(api_key=api_key, timeout=120.0, verbose=True)

    # HARD LOCK: Only Claude Sonnet 4.5 is permitted
    model_name = "claude-sonnet-4-5-20250929"

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        llm_response = response.content[0].text

        # Strip markdown code fences if present
        json_str = llm_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        else:
            json_str = json_str.strip()

        # Validate JSON
        json.loads(json_str)  # Will raise JSONDecodeError if invalid
        return json_str

    except json.JSONDecodeError as e:
        print(f"[LLM] ERROR: Claude Sonnet 4.5 returned invalid JSON: {e}")
        print(f"[LLM] Raw response was: {llm_response[:500]}...")
        print(f"[LLM] Cleaned JSON string was: {json_str[:500]}...")
        raise
    except Exception as e:
        raise RuntimeError(
            f"LLM consolidation failed using required model {model_name}: {e}"
        )


# =============================================================================
# REAL EMBEDDING TEST PIPELINE
# =============================================================================

def run_real_embedding_test(
    csv_path: str = "tiny_test_clauses.csv",
    similarity_threshold: float = 0.7,
    filter_by: Optional[str] = None,
    filter_value: Optional[str] = None,
    use_openai: bool = True
) -> None:
    """
    Test the harmonization grouping with REAL embeddings on a small dataset.

    This validates that the grouping algorithm works correctly with real embeddings
    before enabling full-scale processing.

    Args:
        csv_path: Path to CSV file with clauses
        similarity_threshold: Cosine similarity threshold (0.0-1.0)
                             Note: Real embeddings typically have higher similarity
                             than TF-IDF, so use 0.7-0.85 instead of 0.05
        filter_by: Optional column to filter on (e.g., "Manual_Flag", "Safety_Flag")
        filter_value: Value to filter for (e.g., "y")
        use_openai: If True, uses OpenAI embeddings (recommended).
                   If False, uses Claude stub (not recommended for production).

    Returns:
        None (prints results to console)
    """
    print("=" * 80)
    print("REAL EMBEDDING TEST - Harmonization Grouping")
    print("=" * 80)
    print(f"Dataset: {csv_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    if filter_by:
        print(f"Filter: {filter_by} == {filter_value}")
    print(f"Embedding model: {'OpenAI text-embedding-3-small' if use_openai else 'Claude stub'}")
    print("=" * 80)

    # Load clauses
    print(f"\n[TEST] Loading clauses from {csv_path}...")
    all_clauses = load_clauses_from_tagged_csv(csv_path)
    print(f"[TEST] Loaded {len(all_clauses)} total clauses")

    # Apply filter if specified
    if filter_by and filter_value:
        filtered_clauses = []
        for clause in all_clauses:
            clause_dict = clause.__dict__
            if clause_dict.get(filter_by) == filter_value:
                filtered_clauses.append(clause)

        print(f"[TEST] Filtered to {len(filtered_clauses)} clauses where {filter_by}={filter_value}")
        all_clauses = filtered_clauses

    if len(all_clauses) < 2:
        print("[TEST] ERROR: Need at least 2 clauses to test grouping")
        return

    # Create embedding function
    if use_openai:
        print("[TEST] Using OpenAI embeddings...")
        embed_fn = lambda text: get_openai_embedding(text)
    else:
        print("[TEST] Using Claude stub embeddings...")
        embed_fn = lambda text: get_embedding(text)

    # Compute embeddings
    print(f"\n[TEST] Computing embeddings for {len(all_clauses)} clauses...")
    embeddings, _ = compute_embeddings(all_clauses, embed_fn=embed_fn)
    print(f"[TEST] Embeddings shape: {embeddings.shape}")

    # Group by similarity
    print(f"\n[TEST] Grouping clauses by similarity (threshold={similarity_threshold})...")
    groups = group_clauses_by_similarity(all_clauses, embeddings, similarity_threshold)

    # Print results
    print("\n" + "=" * 80)
    print("GROUPING RESULTS")
    print("=" * 80)
    print(f"Total clauses: {len(all_clauses)}")
    print(f"Cross-standard groups found: {len(groups)}")
    print()

    if not groups:
        print("No cross-standard groups created.")
        print("Possible reasons:")
        print("  - Similarity threshold too high")
        print("  - All clauses from same standard")
        print("  - Dataset too small")
        return

    # Detailed group analysis
    for i, group in enumerate(groups):
        print(f"\nGroup {i+1}: {len(group)} clauses")
        standards = set(all_clauses[idx].standard_name for idx in group)
        print(f"  Standards: {standards}")
        print(f"  Clauses:")

        for idx in group:
            clause = all_clauses[idx]
            text_preview = clause.text[:80] + "..." if len(clause.text) > 80 else clause.text
            print(f"    - {clause.standard_name} {clause.clause_number}: {text_preview}")

        # Calculate average similarity within group
        if len(group) > 1:
            similarities = []
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            for i_idx in range(len(group)):
                for j_idx in range(i_idx + 1, len(group)):
                    sim = cos_sim(
                        embeddings[group[i_idx]].reshape(1, -1),
                        embeddings[group[j_idx]].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            print(f"  Similarity stats: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


# =============================================================================
# CONSOLIDATION TEST PIPELINE
# =============================================================================

def run_consolidation_test(
    csv_path: str = "tiny_test_clauses.csv",
    output_path: str = "tiny_test_consolidation.html",
    similarity_threshold: float = 0.05,
    use_claude: bool = True
) -> List[RequirementGroup]:
    """
    Test the full consolidation pipeline with REAL LLM calls on a small dataset.

    This validates that the consolidation layer (Phase 2) works correctly:
    1. Loads clauses from CSV
    2. Groups by TF-IDF similarity
    3. Calls REAL LLM (Claude or OpenAI) to consolidate each group
    4. Generates HTML report with Phase 2 fields
    5. Prints detailed output for verification

    Args:
        csv_path: Path to CSV file with clauses (default: tiny_test_clauses.csv)
        output_path: Path to save HTML report (default: tiny_test_consolidation.html)
        similarity_threshold: Cosine similarity threshold (default: 0.05 for TF-IDF)
        use_claude: If True, uses Claude API. If False, uses OpenAI GPT-4.

    Returns:
        List of RequirementGroup objects with Phase 2 consolidation fields populated

    Raises:
        ValueError: If API key not found
        Exception: If LLM call fails or returns invalid JSON
    """
    print("=" * 80)
    print("CONSOLIDATION TEST PIPELINE - PHASE 2")
    print("=" * 80)
    print(f"CSV Input: {csv_path}")
    print(f"HTML Output: {output_path}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print(f"LLM: {'Claude (Anthropic)' if use_claude else 'OpenAI GPT-4'}")
    print("=" * 80)

    # Step 1: Load clauses and group by similarity (using TF-IDF)
    print("\n[CONSOLIDATION TEST] Step 1: Loading clauses and grouping...")
    all_clauses, groups = load_and_group_clauses(
        csv_paths=[csv_path],
        similarity_threshold=similarity_threshold,
        embed_fn=None  # Use TF-IDF for speed
    )

    if not groups:
        print("[CONSOLIDATION TEST] ERROR: No cross-standard groups found!")
        print("  - Check similarity_threshold (try lower value)")
        print("  - Verify CSV contains clauses from 2+ different standards")
        return []

    print(f"[CONSOLIDATION TEST] Found {len(groups)} cross-standard groups")

    # Step 2: Create REAL LLM function
    print("\n[CONSOLIDATION TEST] Step 2: Creating LLM consolidation function...")
    llm_fn = lambda prompt: call_llm_for_consolidation(prompt, use_claude=use_claude)
    print(f"[CONSOLIDATION TEST] Using {'Claude' if use_claude else 'OpenAI'} for consolidation")

    # Step 3: Consolidate groups with REAL LLM
    print("\n[CONSOLIDATION TEST] Step 3: Consolidating groups with LLM...")
    print(f"[CONSOLIDATION TEST] This will make {len(groups)} LLM API calls...")
    requirement_groups = consolidate_groups(all_clauses, groups, llm_fn)

    if not requirement_groups:
        print("[CONSOLIDATION TEST] ERROR: No groups were successfully consolidated!")
        return []

    print(f"[CONSOLIDATION TEST] Successfully consolidated {len(requirement_groups)} groups")

    # Step 4: Print detailed consolidation results
    print("\n" + "=" * 80)
    print("CONSOLIDATION RESULTS")
    print("=" * 80)

    for i, req_group in enumerate(requirement_groups):
        print(f"\n{'='*80}")
        print(f"GROUP {i+1}: {req_group.group_title}")
        print(f"{'='*80}")
        print(f"Standards: {', '.join(req_group.get_standards())}")
        print(f"Clause Count: {req_group.get_clause_count()}")
        print()

        print(f"REGULATORY INTENT:")
        print(f"  {req_group.regulatory_intent}")
        print()

        print(f"CONSOLIDATED REQUIREMENT:")
        # Wrap long text for readability
        for line in req_group.consolidated_requirement.split('\n'):
            print(f"  {line}")
        print()

        print(f"DIFFERENCES ACROSS STANDARDS:")
        if req_group.differences:
            for diff in req_group.differences:
                std = diff.get('standard', 'Unknown')
                diffs = diff.get('differences', 'N/A')
                print(f"  • {std}:")
                # Indent multi-line differences
                for line in diffs.split('\n'):
                    print(f"      {line}")
        else:
            print("  (None)")
        print()

        if req_group.unique_requirements:
            print(f"UNIQUE REQUIREMENTS:")
            print(f"  {req_group.unique_requirements}")
            print()

        if req_group.conflicts:
            print(f"⚠️  CONFLICTS DETECTED:")
            print(f"  {req_group.conflicts}")
            print()

        print(f"ORIGINAL CLAUSES IN THIS GROUP:")
        for clause in req_group.clauses:
            print(f"  • [{clause.standard_name}] {clause.clause_number}")
            text_preview = clause.text[:100] + "..." if len(clause.text) > 100 else clause.text
            print(f"    {text_preview}")

    # Step 5: Generate HTML report
    print("\n" + "=" * 80)
    print("GENERATING HTML REPORT")
    print("=" * 80)
    save_html_report(requirement_groups, output_path)
    print(f"[CONSOLIDATION TEST] Report saved to: {output_path}")

    print("\n" + "=" * 80)
    print("CONSOLIDATION TEST COMPLETE")
    print("=" * 80)
    print(f"Summary:")
    print(f"  - Groups consolidated: {len(requirement_groups)}")
    print(f"  - Total clauses: {sum(g.get_clause_count() for g in requirement_groups)}")
    print(f"  - Conflicts detected: {sum(1 for g in requirement_groups if g.conflicts)}")
    print(f"  - Unique requirements flagged: {sum(1 for g in requirement_groups if g.unique_requirements)}")
    print("=" * 80)

    return requirement_groups


def run_tiny_test_pipeline(
    csv_path: str = "tiny_test_clauses.csv",
    output_path: str = "tiny_test_report.html",
    similarity_threshold: float = 0.05,
    llm_call_fn: Optional[Callable[[str], str]] = None
) -> List[RequirementGroup]:
    """
    Run the harmonization pipeline on the tiny test dataset.

    This is a simplified version for testing with a small dataset (8 clauses).

    Args:
        csv_path: Path to tagged CSV file (default: test_tagging_sample_filtered.csv)
        output_path: Path to save HTML report (default: tiny_test_report.html)
        similarity_threshold: Cosine similarity threshold for grouping (default: 0.3)
        llm_call_fn: Optional LLM function. If None, uses stub_call_llm.

    Returns:
        List of RequirementGroup objects

    Pipeline Steps:
        1. Load clauses from CSV
        2. Compute TF-IDF embeddings
        3. Group by cosine similarity (filters to cross-standard groups only)
        4. Consolidate groups with LLM
        5. Generate HTML report
    """
    print("=" * 80)
    print("HARMONIZATION PIPELINE - TINY TEST")
    print("=" * 80)

    # Use stub LLM if no real LLM provided
    if llm_call_fn is None:
        print("[PIPELINE] WARNING: No LLM function provided, using stub")
        llm_call_fn = stub_call_llm

    # Step 1: Load clauses and group by similarity
    print("\n[PIPELINE] Step 1: Loading clauses and grouping by similarity...")
    all_clauses, groups = load_and_group_clauses(
        csv_paths=[csv_path],
        similarity_threshold=similarity_threshold
    )

    if not groups:
        print("[PIPELINE] ERROR: No cross-standard groups found!")
        print("[PIPELINE] This likely means:")
        print("  - All clauses are from the same standard, OR")
        print("  - Similarity threshold is too high, OR")
        print("  - Dataset is too small to form meaningful groups")
        print("\n[PIPELINE] Pipeline stopped. No report generated.")
        return []

    # Step 2: Consolidate groups with LLM
    print("\n[PIPELINE] Step 2: Consolidating groups with LLM...")
    requirement_groups = consolidate_groups(all_clauses, groups, llm_call_fn)

    if not requirement_groups:
        print("[PIPELINE] ERROR: No requirement groups created!")
        print("[PIPELINE] Pipeline stopped. No report generated.")
        return []

    # Step 3: Generate HTML report
    print("\n[PIPELINE] Step 3: Generating HTML report...")
    save_html_report(
        groups=requirement_groups,
        output_path=output_path,
        title="Tiny Test - Cross-Standard Harmonization Report"
    )

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total clauses processed: {len(all_clauses)}")
    print(f"Cross-standard groups created: {len(requirement_groups)}")
    print(f"HTML report saved to: {output_path}")
    print("=" * 80)

    return requirement_groups


def run_full_pipeline(
    csv_paths: List[str],
    output_path: str,
    similarity_threshold: float = 0.3,
    llm_call_fn: Optional[Callable[[str], str]] = None,
    title: str = "Cross-Standard Harmonization Report"
) -> List[RequirementGroup]:
    """
    Run the harmonization pipeline on multiple CSV files (full production version).

    Args:
        csv_paths: List of paths to tagged CSV files from different standards
        output_path: Path to save HTML report
        similarity_threshold: Cosine similarity threshold for grouping
        llm_call_fn: LLM function that takes prompt and returns response
        title: Report title

    Returns:
        List of RequirementGroup objects
    """
    print("=" * 80)
    print("HARMONIZATION PIPELINE - FULL VERSION")
    print("=" * 80)

    # Validate inputs
    if not csv_paths:
        raise ValueError("No CSV paths provided")

    if len(csv_paths) < 2:
        print("[PIPELINE] WARNING: Only 1 CSV file provided.")
        print("[PIPELINE] Cross-standard harmonization requires 2+ standards.")
        print("[PIPELINE] Proceeding anyway, but no groups may be created.")

    if llm_call_fn is None:
        print("[PIPELINE] WARNING: No LLM function provided, using stub")
        llm_call_fn = stub_call_llm

    # Step 1: Load clauses and group by similarity
    print(f"\n[PIPELINE] Step 1: Loading {len(csv_paths)} CSV files and grouping by similarity...")
    all_clauses, groups = load_and_group_clauses(
        csv_paths=csv_paths,
        similarity_threshold=similarity_threshold
    )

    if not groups:
        print("[PIPELINE] WARNING: No cross-standard groups found!")
        print("[PIPELINE] Creating empty report...")
        save_html_report(
            groups=[],
            output_path=output_path,
            title=title + " (No Groups Found)"
        )
        return []

    # Step 2: Consolidate groups with LLM
    print("\n[PIPELINE] Step 2: Consolidating groups with LLM...")
    requirement_groups = consolidate_groups(all_clauses, groups, llm_call_fn)

    if not requirement_groups:
        print("[PIPELINE] WARNING: No requirement groups created after consolidation!")
        save_html_report(
            groups=[],
            output_path=output_path,
            title=title + " (No Groups Created)"
        )
        return []

    # Step 3: Generate HTML report
    print("\n[PIPELINE] Step 3: Generating HTML report...")
    save_html_report(
        groups=requirement_groups,
        output_path=output_path,
        title=title
    )

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"CSV files processed: {len(csv_paths)}")
    print(f"Total clauses processed: {len(all_clauses)}")
    print(f"Cross-standard groups created: {len(requirement_groups)}")
    print(f"HTML report saved to: {output_path}")
    print("=" * 80)

    return requirement_groups


# Example usage and main execution
if __name__ == "__main__":
    """
    Run test pipelines when executed as a script.

    Usage:
        python -m harmonization.pipeline_unify                  # Run TF-IDF tiny test (stub LLM)
        python -m harmonization.pipeline_unify --embeddings     # Run real embedding test
        python -m harmonization.pipeline_unify --consolidation  # Run consolidation test (REAL LLM)
    """
    import sys

    # Check if test file exists
    test_file = "tiny_test_clauses.csv"
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        print(f"Current directory: {os.getcwd()}")
        print("Please ensure the test file exists before running the pipeline.")
        exit(1)

    # Check for --consolidation flag (PHASE 2 TEST)
    if "--consolidation" in sys.argv or "--phase2" in sys.argv:
        # Run CONSOLIDATION TEST with REAL LLM
        print("=" * 80)
        print("RUNNING CONSOLIDATION TEST MODE (PHASE 2)")
        print("=" * 80)
        print()

        try:
            # Determine which LLM to use
            use_claude = True
            if "--openai" in sys.argv or "--gpt" in sys.argv:
                use_claude = False

            groups = run_consolidation_test(
                csv_path=test_file,
                output_path="tiny_test_consolidation.html",
                similarity_threshold=0.05,  # TF-IDF threshold
                use_claude=use_claude
            )

            if groups:
                print(f"\nSUCCESS! Consolidated {len(groups)} requirement groups with REAL LLM.")
                print("Open tiny_test_consolidation.html in your browser to view the report.")
            else:
                print("\nNo groups created. Check the logs above for details.")

        except Exception as e:
            print(f"\nERROR: Consolidation test failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    # Check for --embeddings flag
    elif "--embeddings" in sys.argv or "--real" in sys.argv:
        # Run REAL EMBEDDING TEST
        print("=" * 80)
        print("RUNNING REAL EMBEDDING TEST MODE")
        print("=" * 80)
        print()

        try:
            run_real_embedding_test(
                csv_path=test_file,
                similarity_threshold=0.75,  # Higher threshold for real embeddings
                filter_by=None,  # No filter - use all clauses
                filter_value=None,
                use_openai=True  # Use OpenAI embeddings
            )
        except Exception as e:
            print(f"\nERROR: Real embedding test failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    else:
        # Run TF-IDF TINY TEST (default)
        print("Running tiny test pipeline with TF-IDF and stub LLM...")
        print("(To test with real embeddings, run: python -m harmonization.pipeline_unify --embeddings)")
        print("(To test consolidation with REAL LLM, run: python -m harmonization.pipeline_unify --consolidation)")
        print()

        try:
            groups = run_tiny_test_pipeline(
                csv_path=test_file,
                output_path="tiny_test_report.html",
                similarity_threshold=0.05,
                llm_call_fn=None  # Uses stub
            )

            if groups:
                print(f"\nSuccess! Created {len(groups)} requirement groups.")
                print("Open tiny_test_report.html in your browser to view the report.")
            else:
                print("\nNo groups created. Check the logs above for details.")

        except Exception as e:
            print(f"\nERROR: Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
