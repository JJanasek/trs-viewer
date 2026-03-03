import sys
import numpy as np
import trsfile
from tqdm import tqdm
import os
import gc
import mmap

class ttest(object):
    def __init__(self, Ns):
        self.Ns = Ns
        self.Nt = np.zeros(2, dtype=int)
        self.Nt_total = 0
        
        # CRITICAL FIX 1: Use float64 for accumulators to prevent precision loss
        # with large N or DC offsets.
        self.t_sum = np.zeros((2, Ns), dtype=np.float64)
        self.t_sum2 = np.zeros((2, Ns), dtype=np.float64)
    

    def calculate_and_save(self, traceset, out_path, disable_tqdm=False, trace_file_path=None):
        # Local optimization references (views into the same arrays)
        t_sum0, t_sum1 = self.t_sum[0], self.t_sum[1]
        t_sum20, t_sum21 = self.t_sum2[0], self.t_sum2[1]
        Nt = self.Nt

        # --- PHASE 1: SAFE ACCUMULATION LOOP ---
        total_traces = len(traceset)
        print(f"Accumulating {total_traces} traces...")

        # Setup mmap if file path is provided
        mm = None
        f_obj = None
        if trace_file_path and os.path.exists(trace_file_path):
            try:
                f_obj = open(trace_file_path, "rb")
                mm = mmap.mmap(f_obj.fileno(), 0, access=mmap.ACCESS_READ)
                print("Memory mapping enabled for efficient reading.")
            except Exception as e:
                print(f"Warning: Could not mmap file ({e}). Falling back to standard reading.")

        # Iterate by INDEX to prevent holding references to many traces
        chunk_size = 10_000_000 # 10M samples per chunk to keep peak memory low
        
        for i in tqdm(range(self.Nt_total, total_traces), disable=disable_tqdm):
            trace = traceset[i]
            g = trace.parameters['ttest'].value[0]

            # Try to get data offset for mmap
            data_offset = None
            if mm is not None:
                # Attempt to find the data offset from the trace object
                # Common attributes in trsfile implementations
                if hasattr(trace, '_data_offset'):
                    data_offset = trace._data_offset
                elif hasattr(trace, 'data_offset'):
                    data_offset = trace.data_offset
            
            # Process in chunks to avoid loading full trace into memory
            for start_idx in range(0, self.Ns, chunk_size):
                end_idx = min(start_idx + chunk_size, self.Ns)
                current_chunk_len = end_idx - start_idx
                
                t_chunk = None
                
                # OPTIMIZATION: Read from mmap if possible
                if mm is not None and data_offset is not None:
                    # Calculate byte offset for this chunk
                    # Assuming float32 samples (4 bytes)
                    chunk_byte_offset = data_offset + (start_idx * 4)
                    
                    # Create numpy array from buffer (zero-copy view)
                    try:
                        t_chunk = np.frombuffer(mm, dtype=np.int8, count=current_chunk_len, offset=chunk_byte_offset)
                        t_chunk = t_chunk_raw.astype(np.float32)
                    except Exception:
                        # Fallback if something goes wrong (e.g. alignment or bounds)
                        t_chunk = None

                # Fallback to standard reading
                if t_chunk is None:
                    # Load only the chunk
                    # Assuming trace.samples supports slicing for partial reads
                    t_chunk = np.asarray(trace.samples[start_idx:end_idx], dtype=np.float32)

                if g == 0:
                    self.t_sum[0, start_idx:end_idx] += t_chunk
                    self.t_sum2[0, start_idx:end_idx] += np.square(t_chunk)
                elif g == 1:
                    self.t_sum[1, start_idx:end_idx] += t_chunk
                    self.t_sum2[1, start_idx:end_idx] += np.square(t_chunk)
                
                del t_chunk

            if g == 0:
                Nt[0] += 1
            elif g == 1:
                Nt[1] += 1

            # Free memory immediately
            del trace
            
        # Close mmap
        if mm is not None:
            mm.close()
        if f_obj is not None:
            f_obj.close()

        print(f"Group counts: Group0={Nt[0]}, Group1={Nt[1]}")
        self.Nt_total = int(Nt[0] + Nt[1])

        # Sanity check: need at least 2 traces in each group for variance
        if Nt[0] < 2 or Nt[1] < 2:
            raise ValueError(
                f"Not enough traces per group for variance: Nt0={Nt[0]}, Nt1={Nt[1]}"
            )

        # --- PHASE 2: OPTIMIZED CALCULATION (IN-PLACE) ---
        print("Calculating statistics (In-Place)...")

        # 1. Means (overwrite t_sum)
        # self.t_sum[g] becomes mean_g
        self.t_sum[0] /= Nt[0]
        self.t_sum[1] /= Nt[1]

        # 2. Variances (overwrite t_sum2)
        # var = (sum2 - N * mean^2) / (N - 1)

        # Group 0 variance
        temp_sq = np.square(self.t_sum[0])   # mean0^2
        temp_sq *= Nt[0]                     # mean0^2 * N0
        self.t_sum2[0] -= temp_sq            # sum20 - mean0^2 * N0
        del temp_sq
        self.t_sum2[0] /= (Nt[0] - 1)        # var0

        # Group 1 variance
        temp_sq = np.square(self.t_sum[1])   # mean1^2
        temp_sq *= Nt[1]                     # mean1^2 * N1
        self.t_sum2[1] -= temp_sq            # sum21 - mean1^2 * N1
        del temp_sq
        self.t_sum2[1] /= (Nt[1] - 1)        # var1

        # 3. Denominator for Welch t: sqrt(var0/N0 + var1/N1)
        # Do this before freeing self.t_sum2
        var0 = self.t_sum2[0]   # alias, no copy
        var1 = self.t_sum2[1]   # alias, no copy
        denom = np.sqrt(var0 / Nt[0] + var1 / Nt[1])

        # Now we can free the variance arrays
        self.t_sum2 = None
        gc.collect()

        # 4. Numerator (reuse t_sum[0]): mean0 - mean1
        self.t_sum[0] -= self.t_sum[1]

        # 5. T-Stat: (mean0 - mean1) / sqrt(var0/N0 + var1/N1)
        np.divide(self.t_sum[0], denom, out=self.t_sum[0], where=denom != 0)

        # 6. Save as Float32
        tstat = self.t_sum[0].astype(np.float32)

        # Final cleanup
        del denom
        self.t_sum = None
        gc.collect()

        print(f"Saving to {out_path}...")
        np.save(out_path, tstat)
        print("Done.")


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <trace_file.trs> <output_file.npy>")
        sys.exit(1)

    traceFileName = sys.argv[1]
    outputFileName = sys.argv[2]

    # Open the trace set with lazy engine
    # 'TrsEngine' is typically the lazy engine in trsfile
    try:
        traceset = trsfile.open(traceFileName, 'r', engine='TrsEngine')
        print("Opened trace file with TrsEngine (Lazy).")
    except TypeError:
        # Fallback if engine argument is not supported
        print("Warning: 'engine' argument not supported. Opening with default settings.")
        traceset = trsfile.open(traceFileName, 'r')
    
    # Initialize
    num_samples = traceset.get_header(trsfile.Header.NUMBER_SAMPLES)
    ttest_object = ttest(num_samples)
    
    # Run and Save
    ttest_object.calculate_and_save(traceset, outputFileName, trace_file_path=traceFileName)
    
    traceset.close()

