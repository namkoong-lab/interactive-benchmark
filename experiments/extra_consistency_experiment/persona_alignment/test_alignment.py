#!/usr/bin/env python3
"""
Test script to evaluate a specific persona-answer alignment and get reasoning.
Uses the exact same prompt format as persona_alignment_evaluator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.core.llm_providers import chat_completion

# Manually set persona description (exactly as provided)
persona_description = "Meet Everett Thompson, a 68-year-old retired high school history teacher who has lived in rural New Hampshire for most of his life. Born and raised in the small town of Peterborough, Everett grew up surrounded by the picturesque landscape of the Monadnock Region. He is a white male of English and Scottish descent, with a strong sense of community and tradition.\n\nEverett stands at 5'10\" with a slightly stooping posture, a testament to his years of bending over to help students with their assignments. His thinning grey hair is always neatly combed, and his bright blue eyes sparkle with warmth and kindness. He has a bushy grey mustache that he's had since his college days, which adds to his gentle, avuncular demeanor.\n\nAfter graduating from the University of New Hampshire with a degree in history, Everett spent 35 years teaching at the local high school. He was known for his engaging storytelling and ability to make history come alive for his students. His love for the subject is evident in the way he can rattle off dates, events, and anecdotes with ease.\n\nEverett is a widower, having lost his wife of 40 years, Margaret, to cancer five years ago. They had two children together, both of whom have moved away to pursue their careers. His son, James, is a software engineer in California, and his daughter, Emily, is a doctor in Boston. Everett is proud of their accomplishments but misses the regular family gatherings they used to have.\n\nIn his free time, Everett enjoys woodworking, gardening, and reading historical fiction. He's an avid fan of authors like Ken Follett and Bernard Cornwell and has a vast collection of their books. He's also an active member of the local historical society, where he helps with research and gives occasional lectures on the region's history.\n\nEverett's politics are moderate, leaning slightly to the left. He values education, healthcare, and social justice, but also believes in personal responsibility and limited government intervention. He's a fan of the New England Patriots and the Boston Red Sox, and he loves watching sports with his friends at the local pub.\n\nDespite his retirement, Everett remains engaged with his community. He volunteers at the local library, helps with the town's annual festivals, and participates in the regional senior center's activities. He's a proud Granite Stater who loves the state's motto, \"Live Free or Die,\" and strives to live up to its spirit of independence and resilience.\n\nEverett's life is a testament to the values of hard work, community, and tradition. He's a kind, wise, and witty individual who has earned the respect and admiration of his friends and neighbors."

# The answer from the alignment check (exactly as in the JSON)
answer = "I prefer a medium-sized cross-stitch kit, as it offers a good balance between detail and manageability, allowing me to enjoy the process without feeling overwhelmed."

# Use EXACT same prompt format as persona_alignment_evaluator.py
prompt = f"""You are evaluating whether a persona agent's response aligns with their persona description.

**Persona Description:**
{persona_description}

**Persona Agent's Answer:**
{answer}

**Task:**
Classify the persona agent's answer as one of:
- "Aligned": The answer is clearly consistent with the persona description. The response directly reflects the persona's explicitly stated characteristics, preferences, lifestyle, or needs.
- "Conflict": The answer clearly contradicts or conflicts with the persona description. There must be explicit evidence of contradiction (e.g., persona says they hate X but answer expresses love for X, or persona explicitly states they don't do Y but answer assumes they do Y). Do NOT mark as Conflict just because something isn't mentioned - absence of information is not conflict.
- "Neutral": The answer is neither clearly aligned nor in conflict. 

**Output Format:**
Respond with ONLY one word: "Aligned", "Conflict", or "Neutral"
Do not include any explanation or additional text."""

print("=" * 70)
print("TESTING PERSONA-ANSWER ALIGNMENT")
print("=" * 70)
print("\nPersona Description:")
print("-" * 70)
print(persona_description)
print("-" * 70)
print(f"\nAnswer to Evaluate:")
print("-" * 70)
print(answer)
print("-" * 70)
print("\n" + "=" * 70)
print("PROMPT SENT TO CLAUDE (exact format from experiment):")
print("=" * 70)
print(prompt)
print("=" * 70)
print("\n" + "=" * 70)
print("CLAUDE RESPONSE:")
print("=" * 70 + "\n")

try:
    # Use EXACT same call format as persona_alignment_evaluator.py
    response = chat_completion(
        model="claude-sonnet-4-20250514",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of persona consistency. Respond with only one word: Aligned, Conflict, or Neutral."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=10,  # Same as in evaluator
        json_mode=False
    )
    
    print(f"Response: {response}")
    print("\n" + "=" * 70)
    print("Now asking for reasoning...")
    print("=" * 70 + "\n")
    
    # Now ask for reasoning separately
    reasoning_prompt = f"""You just classified this answer as "{response.strip()}". 

Please explain your reasoning in detail. Why did you classify it this way? What specific aspects of the persona description support or conflict with the answer?

**Persona Description:**
{persona_description}

**Persona Agent's Answer:**
{answer}

**Your Classification:** {response.strip()}

**Your Reasoning:**"""
    
    reasoning_response = chat_completion(
        model="claude-sonnet-4-20250514",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of persona consistency. Provide detailed reasoning for your classification."},
            {"role": "user", "content": reasoning_prompt}
        ],
        temperature=0.3,
        max_tokens=500,
        json_mode=False
    )
    
    print(reasoning_response)
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

