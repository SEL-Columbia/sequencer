## 0.0.5

- Performance improvements:
    - Removed dependency on adjacency matrix materializing (memory reduction)
    - Re-wrote accumulate function to be iterative rather than recursive (stack size limit reached)
- Made logging and versioning more consistent with networker
- Added regression test
- Issues addressed:
    #42, 47


