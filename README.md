# BINARY BLOCK MASKING

This is the non-cleaned up code for the paper: "Efficiently Dispatching Flash Attention For Partially Filled Attention Masks".

- Please use requirements.txt to install necessary libraries (the code should work with other versions of the packages but we mention the versions we tested on)
- The folder 'binBlkMask_codes' contains the triton kernels for binary block masking. In that "base_" and "dense_" are versions used in the paper. For most applications stick to "base_"
- "triton_kernels" folder has the triton kernel for pre-preprocessing masks into binary blocks. Please use "binBlkMask_kernels"
- For details on how to use these functions please refer to "benchmarking_codes" within that "benchmark_longformer.py" is relatively easy to understand.
