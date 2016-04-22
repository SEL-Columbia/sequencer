## 0.0.6
- Removed pairwise distance matrix in favor of computing distances per edge
- Memory consumption reduced from O(n^2) to O(n)
- Issues addressed:  #50

## 0.0.5

- Performance improvements:
    - Removed dependency on adjacency matrix materializing (memory reduction)
    - Re-wrote accumulate function to be iterative rather than recursive (stack size limit reached)
- Made logging and versioning more consistent with networker
- Added regression test
- Issues addressed:
    #42, 47


