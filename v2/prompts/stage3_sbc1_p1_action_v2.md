You are judging one boundary of a coarse highlight segment in a short video clip.

A video clip is attached. It spans {window_before_sec} seconds before and {window_after_sec} seconds after one candidate boundary of a coarse highlight segment. The boundary is located at {boundary_offset_in_clip_sec} seconds from the start of this clip. The clip duration is {clip_duration_sec} seconds.

Context:
- side = {side}
- The coarse candidate segment is about {candidate_duration_sec} seconds long.
- {side_direction}

Decide whether this boundary should move inward, stay, or move outward.

Definitions:
- TRIM: the boundary currently includes preparation, trailing footage, or unrelated context that is not part of the highlight event; the boundary should move inward to remove it.
- KEEP: the boundary already covers the event well, or the evidence is ambiguous. Use KEEP when uncertain.
- EXPAND: the highlight event clearly starts before this boundary (for the left side) or continues after this boundary (for the right side); the boundary should move outward to include it.

Key questions to check:
1. Does the event continue outside the candidate beyond this boundary (suggesting EXPAND)?
2. Does the inside of the candidate near this boundary contain only setup / transition / unrelated context (suggesting TRIM)?
3. Does the boundary keep the action onset, the main action, and the reaction/result inside the segment?

Return strict JSON only, no markdown, in exactly this shape:
{"action": "TRIM|KEEP|EXPAND", "confidence": 0.0, "evidence": {"event_continues_outside": false, "boundary_contains_context_only": false, "core_event_visible": false}, "rationale_short": "<=20 words"}
