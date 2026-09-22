You are judging whether footage just outside a candidate boundary belongs to the same highlight event as the candidate core.

Two video clips are attached, in this fixed order:
- First clip: the CORE clip — sampled from the middle of the coarse highlight candidate. Treat this as the reference content of the highlight event.
- Second clip: the BOUNDARY clip — spans the candidate boundary region (about {window_before_sec} seconds before and {window_after_sec} seconds after the boundary). The boundary is located at {boundary_offset_in_clip_sec} seconds from the start of this second clip.

Context:
- side = {side}
- The coarse candidate segment is about {candidate_duration_sec} seconds long.
- {side_direction}

Question: Does the footage on the OUTSIDE of the boundary (the part of the second clip beyond the boundary) still belong to the same event as the core clip?

Decide whether this boundary should move inward, stay, or move outward:

- TRIM: the outside footage is clearly a different event, transition, or unrelated context, AND the boundary-near inside footage is only padding; the boundary should move inward.
- KEEP: the outside footage is ambiguous, mixed, or partially related; or the boundary is already acceptable. Use KEEP when uncertain.
- EXPAND: the outside footage clearly continues the same event as the core clip; the boundary should move outward.

Return strict JSON only, no markdown, in exactly this shape:
{"action": "TRIM|KEEP|EXPAND", "confidence": 0.0, "same_event_outside": false, "boundary_context_is_redundant": false, "rationale_short": "<=20 words"}
