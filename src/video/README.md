# Part 2: Video depth

Implement the Video Depth Anything-Small wrapper here.
Use the relative-depth checkpoint, offline inference, and the repository's
default FP16. Start with --encoder vits --max_len 100 --max_res 640 --save_npz.
Compare exactly the same frames with the frame-wise model. Use the same palette
and a fixed normalization for each method across the whole clip. Record input
settings, runtime, peak GPU memory, and visible failure cases.
