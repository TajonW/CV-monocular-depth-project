# Shared comparison videos

Small exports for collaborators to inspect without GPU inference:

- `rollercoaster_70_rgb_da2_vda.mp4`: original 24 FPS, 70 frames,
  2.916667 seconds; RGB / DA V2-Small / VDA-Small.
- `rollercoaster_70_review_4x_slower.mp4`: labeled review copy at 6 FPS,
  the same 70 frames over 11.666667 seconds; original source timestamps retained.

Both models use the same palette and fixed whole-clip normalization per method.
Colors do not represent a shared metric scale. Raw predictions remain in ignored
outputs. Provenance: `../tables/comparison/` and `../tables/video_analysis/`.
Analysis: `../../docs/video_qualitative_analysis.md`.

The source has 70 frames although the course asks for 100; that scope difference
still needs clarification. Slow playback does not add source frames.
