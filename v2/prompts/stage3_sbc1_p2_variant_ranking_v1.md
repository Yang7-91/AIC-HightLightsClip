You are comparing three candidate versions of the same highlight segment boundary.

Three short video clips are attached, in this fixed order:
- First clip: KEEP variant — the original candidate boundary is preserved as-is.
- Second clip: TRIM variant — the boundary is moved inward by about {variant_shift_sec} seconds, so the segment is shorter.
- Third clip: EXPAND variant — the boundary is moved outward by about {variant_shift_sec} seconds, so the segment is longer.

All three clips come from the same video and differ only at this boundary position.

Context:
- side = {side}
- The coarse candidate segment is about {candidate_duration_sec} seconds long.
- {side_direction}

Choose the variant whose segment:
1. most completely covers the highlight event (action onset, main action, reaction/result), and
2. contains the least unrelated context (setup, transition, trailing footage).

Rules:
- If completeness is uncertain, prefer KEEP.
- If EXPAND clearly includes the action onset or the continued reaction, choose EXPAND.
- If TRIM clearly removes unrelated setup/trailing content without losing event content, choose TRIM.

Return strict JSON only, no markdown, in exactly this shape:
{"best_variant": "KEEP|TRIM|EXPAND", "confidence": 0.0, "ranking": ["KEEP", "TRIM", "EXPAND"], "rationale_short": "<=20 words"}
