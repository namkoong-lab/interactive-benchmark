#!/usr/bin/env python3
"""
Test script to get full reasoning from consistency evaluator for specific answers.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.core.llm_providers import chat_completion
import json
import re

# The 10 answers from consistency_trajectory_1_results.json
question = "Do you have a budget preference for the cross-stitch kit?"

answers = [
    "I don't have a strict budget, but I generally look for quality kits that offer good value, usually in the mid-range price.",
    "I prefer to keep it under $50, as I like to balance quality with affordability while being mindful of my budget.",
    "I prefer to keep it under $50, as I like to find a good balance between quality and affordability.",
    "I prefer to keep it under $50, as I want something of good quality without overspending.",
    "I prefer to keep it under $50, as it fits comfortably within my budget while allowing me to indulge in my hobby without breaking the bank.",
    "I prefer to keep my budget moderate, aiming for quality without going overboard, usually around $30 to $50.",
    "I generally look for kits that are reasonably priced, ideally under $50, as they offer good value without breaking the bank.",
    "I usually aim to keep it under $50, as it fits comfortably within my budget while still allowing for quality materials and designs.",
    "I don't have a strict budget, but I generally look for kits that are reasonably priced and offer good value, as I try to be mindful of my spending while still enjoying my hobbies.",
    "I usually aim for something moderately priced that offers good quality, as I like to ensure I'm getting a kit that's worth the investment and will last."
]

# Format answers for prompt
answers_text = ""
for i, answer in enumerate(answers, 1):
    answers_text += f"Answer {i}: {answer}\n"

# Create evaluation prompt with reasoning request
prompt = f"""You are evaluating the consistency of multiple statements.

**Statements (from 10 separate runs):**
{answers_text}

**Task:**
1. Analyze all 10 statements together and identify what the MAJORITY position is (what most statements are saying/expressing).
2. For each statement, determine if it aligns with the majority position or deviates from it.
3. Count how many statements are aligned with the majority vs misaligned.

**Guidelines:**
- The majority position is what most statements express (could be a preference, value, need, opinion, etc.)
- A statement is "Aligned" if it matches or is consistent with the majority position
- A statement is "Misaligned" if it contradicts, conflicts with, or significantly deviates from the majority position
- Consider both explicit statements and implied meanings
- If statements are split evenly or there's no clear majority, identify the most common position

**Output Format:**
Respond with a JSON object containing:
{{
  "majority_position": "description of what the majority of statements express",
  "aligned_with_majority_count": <number>,
  "misaligned_with_majority_count": <number>,
  "breakdown": [
    {{"answer_num": 1, "aligned_with_majority": true}},
    {{"answer_num": 2, "aligned_with_majority": false}},
    ...
  ],
  "reasoning": "Detailed explanation of how you identified the majority position and classified each statement"
}}

**Critical JSON Rules:**
- Use only valid JSON syntax
- Escape all quotes in string values with backslash: \\"
- Use true/false (lowercase, not True/False or strings)
- Keep majority_position description concise
- Ensure all 10 answers are included in breakdown
- Include detailed reasoning explaining your analysis"""

print("=" * 70)
print("CONSISTENCY EVALUATION WITH FULL REASONING")
print("=" * 70)
print(f"\nQuestion: {question}\n")
print("10 Answers:")
for i, answer in enumerate(answers, 1):
    print(f"  {i}. {answer}")
print("\n" + "=" * 70)
print("Calling Claude Sonnet 4...\n")

try:
    # Call Claude with request for reasoning
    response = chat_completion(
        model="claude-sonnet-4-20250514",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of statement consistency. You must respond with valid JSON only. Ensure all strings are properly escaped."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000,
        json_mode=False
    )
    
    print("Raw Response:")
    print("-" * 70)
    print(response)
    print("-" * 70)
    print()
    
    # Parse JSON response
    response_text = response.strip()
    
    # Try to extract JSON from markdown code blocks
    if response_text.startswith("```"):
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
    elif "{" in response_text:
        start_idx = response_text.find("{")
        if start_idx >= 0:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if brace_count == 0:
                response_text = response_text[start_idx:end_idx]
    
    result = json.loads(response_text)
    
    print("=" * 70)
    print("PARSED RESULTS")
    print("=" * 70)
    print(f"\nMajority Position:")
    print(f"  {result.get('majority_position', 'N/A')}")
    print(f"\nCounts:")
    print(f"  Aligned with majority: {result.get('aligned_with_majority_count', 0)}")
    print(f"  Misaligned with majority: {result.get('misaligned_with_majority_count', 0)}")
    print(f"\nBreakdown:")
    for item in result.get('breakdown', []):
        status = "✓ Aligned" if item.get('aligned_with_majority', False) else "✗ Misaligned"
        print(f"  Answer {item.get('answer_num', '?')}: {status}")
    
    if 'reasoning' in result:
        print(f"\n{'=' * 70}")
        print("DETAILED REASONING")
        print("=" * 70)
        print(f"\n{result['reasoning']}")
    
    print(f"\n{'=' * 70}")
    print("FULL JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nRaw response was:")
    print(response if 'response' in locals() else "No response received")

