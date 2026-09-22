------------------------------------------------------------------------
-- STEP ENTROPY C API SPECIFICATION v2.0
--
-- Thin C-callable wrapper layer over the Ada Step_Entropy package.
-- All pointer parameters use System.Address (= C void*) — the correct
-- Ada 2022 pattern for C FFI (Ada RM B.3, §13.3).
--
-- Naming: every exported symbol uses single underscore to match Python
-- ctypes expectations (e.g. step_entropy_calculate_token_entropy).
--
-- Error contract: every function returns -1.0 (float) or -1 (int) on
-- any error, logs to stderr, and never raises an Ada exception to the
-- caller. Callers must check return codes.
--
-- Output buffers: caller allocates; size documented per function.
--
-- Introspection:
--   step_entropy_get_version       → major*10000 + minor*100 + patch
--   step_entropy_get_max_vocab_size → Max_Vocab_Size constant
--   step_entropy_get_max_tokens_per_step → Max_Tokens_Per_Step constant
--
-- Status: Production v2.0  |  Last Updated: 2026-04-25
------------------------------------------------------------------------

with Interfaces.C;
with System;

package Step_Entropy_C_API is

   use Interfaces.C;

   subtype C_Float is Interfaces.C.C_float;
   subtype C_Int   is Interfaces.C.int;

   -----------------------------------------------------------------------
   -- CORE: TOKEN-LEVEL ENTROPY
   -----------------------------------------------------------------------

   -- H(t) = -Σ p(w) log_base(p(w))
   -- Logits: float* (length = Logits_Len, 0-indexed, max Max_Vocab_Size)
   -- Base:   log base (2.0 = bits default, 2.718... = nats)
   -- Returns entropy >= 0.0, or -1.0 on error.
   --
   function C_Calculate_Token_Entropy
      (Logits     : System.Address;
       Logits_Len : C_Int;
       Base       : C_Float) return C_Float
      with Export, Convention => C,
           External_Name => "step_entropy_calculate_token_entropy";

   -- Softmax probability for token_id (0-indexed).
   -- Returns probability in [0,1], or -1.0 on error.
   --
   function C_Calculate_Token_Probability
      (Logits     : System.Address;
       Logits_Len : C_Int;
       Token_Id   : C_Int) return C_Float
      with Export, Convention => C,
           External_Name => "step_entropy_calculate_token_probability";

   -----------------------------------------------------------------------
   -- CORE: ENTROPY CLASSIFICATION
   -----------------------------------------------------------------------

   -- Classify entropy: 0=LOW, 1=MEDIUM, 2=HIGH, -1=error
   -- Requires Threshold_Low < Threshold_High and Entropy >= 0.
   --
   function C_Classify_Entropy
      (Entropy        : C_Float;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float) return C_Int
      with Export, Convention => C,
           External_Name => "step_entropy_classify_entropy";

   -----------------------------------------------------------------------
   -- CORE: ROUTING DECISIONS
   -----------------------------------------------------------------------

   -- Route a step based on its entropy statistics.
   -- Writes routing path (0=Fast/1=Normal/2=Slow) to *Out_Path.
   -- Writes retrieve flag (0/1) to *Out_Retrieve.
   -- Returns 0=success, -1=error.
   --
   function C_Route_Step
      (Avg_Entropy    : C_Float;
       Total_Entropy  : C_Float;
       Max_Entropy    : C_Float;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Enable_Memory  : C_Int;
       Out_Path       : System.Address;   -- int*
       Out_Retrieve   : System.Address) return C_Int  -- int*
      with Export, Convention => C,
           External_Name => "step_entropy_route_step";

   -- Batch route: reads float[Count] entropy values, writes int[Count].
   -- Returns 0=success, -1=error.
   --
   function C_Batch_Route
      (Step_Entropies : System.Address;   -- float[Count]
       Count          : C_Int;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Out_Decisions  : System.Address) return C_Int  -- int[Count]
      with Export, Convention => C,
           External_Name => "step_entropy_batch_route";

   -----------------------------------------------------------------------
   -- CORE: CHAIN ANALYSIS
   -----------------------------------------------------------------------

   -- Analyze chain compression opportunity.
   -- Out_Buffer: float[8] = [step_count, compressible, informative,
   --   low_count, medium_count, high_count, compression_ratio, avg_entropy]
   -- Returns 0=success, -1=error.
   --
   function C_Analyze_Chain
      (Step_Entropies : System.Address;   -- float[Step_Count]
       Step_Count     : C_Int;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Out_Buffer     : System.Address) return C_Int  -- float[8]
      with Export, Convention => C,
           External_Name => "step_entropy_analyze_chain";

   -- Recommend pruning: reads float[Step_Count] avg entropies,
   -- writes sorted ascending int[*Out_Count] prune indices.
   -- Out_Count: int* (single output integer).
   -- Returns 0=success, -1=error.
   --
   function C_Recommend_Pruning
      (Step_Entropies : System.Address;   -- float[Step_Count]
       Step_Count     : C_Int;
       Pruning_Ratio  : C_Float;
       Out_Indices    : System.Address;   -- int[Step_Count]
       Out_Count      : System.Address) return C_Int  -- int*
      with Export, Convention => C,
           External_Name => "step_entropy_recommend_pruning";

   -----------------------------------------------------------------------
   -- SOTA+++: ENTROPY HISTOGRAM
   -----------------------------------------------------------------------

   -- Compute token-entropy histogram from float[Count] avg entropies.
   --
   -- Out_Buffer layout: float[5 + Bin_Count]
   --   [0] bin_width
   --   [1] min_entropy
   --   [2] max_entropy
   --   [3] mean_entropy
   --   [4] std_dev
   --   [5..5+Bin_Count-1] bin counts (as floats for uniform buffer type)
   --
   -- Returns 0=success, -1=error.
   --
   function C_Compute_Histogram
      (Entropies  : System.Address;       -- float[Count]
       Count      : C_Int;
       Bin_Count  : C_Int;
       Out_Buffer : System.Address) return C_Int  -- float[5 + Bin_Count]
      with Export, Convention => C,
           External_Name => "step_entropy_compute_histogram";

   -----------------------------------------------------------------------
   -- SOTA+++: Z-SCORE NORMALIZATION
   -----------------------------------------------------------------------

   -- Normalize entropy to z-score: (Entropy - Mean) / Std_Dev.
   -- Returns raw Entropy when Std_Dev = 0 (no variance yet).
   -- Returns -999.0 on error (sentinel; not a valid z-score in context).
   --
   function C_Normalize_Entropy
      (Entropy : C_Float;
       Mean    : C_Float;
       Std_Dev : C_Float) return C_Float
      with Export, Convention => C,
           External_Name => "step_entropy_normalize_entropy";

   -----------------------------------------------------------------------
   -- SOTA+++: GRPO REWARD FUNCTION (Paper Eq.13-15, Table 3)
   -----------------------------------------------------------------------

   -- Compute inference-time GRPO reward components.
   -- R_correctness (oracle) is NOT computed here.
   --
   -- Out_Rewards: float[4] = [r_skip_ratio, r_skip_num, r_response_len, r_total]
   --   r_skip_ratio ∈ {0.0, 0.5, 1.0}
   --   r_skip_num   ∈ {-1.0, 0.0}
   --   r_response_len ∈ {-1.0, 0.0}
   --   r_total = sum of 3 above
   --
   -- Returns 0=success, -1=error.
   --
   function C_Compute_GRPO_Rewards
      (Skip_Ratio      : C_Float;
       Skip_Count      : C_Int;
       Response_Length : C_Int;
       K_High          : C_Float;
       K_Low           : C_Float;
       Max_Skip        : C_Int;
       Max_Length      : C_Int;
       Out_Rewards     : System.Address) return C_Int  -- float[4]
      with Export, Convention => C,
           External_Name => "step_entropy_compute_grpo_rewards";

   -----------------------------------------------------------------------
   -- SOTA+++: ADAPTIVE THRESHOLD STATE
   -----------------------------------------------------------------------

   -- Adaptive threshold state buffer: float[5]
   --   [0] running_mean      (Welford running mean)
   --   [1] running_m2        (Welford M2 accumulator)
   --   [2] sample_count      (stored as float for buffer uniformity)
   --   [3] low_sigma_factor  (threshold_low  = mean + factor * std)
   --   [4] high_sigma_factor (threshold_high = mean + factor * std)
   --
   -- Initialize: zero the buffer and set [3]=-1.0, [4]=1.0 before first call.

   -- Welford online update with one new entropy observation.
   -- Reads and writes State_Buf in-place.
   -- Returns 0=success, -1=error.
   --
   function C_Update_Adaptive_State
      (Entropy   : C_Float;
       State_Buf : System.Address) return C_Int  -- float[5], in-out
      with Export, Convention => C,
           External_Name => "step_entropy_update_adaptive_state";

   -- Compute current adaptive thresholds from state buffer.
   -- Writes threshold_low to *Out_Low, threshold_high to *Out_High.
   -- Returns  0=success
   --         -1=error
   --         -2=insufficient samples (sample_count < 2)
   --
   function C_Get_Adaptive_Thresholds
      (State_Buf         : System.Address;   -- float[5]
       Out_Threshold_Low : System.Address;   -- float*
       Out_Threshold_High: System.Address) return C_Int  -- float*
      with Export, Convention => C,
           External_Name => "step_entropy_get_adaptive_thresholds";

   -----------------------------------------------------------------------
   -- LIBRARY INTROSPECTION
   -----------------------------------------------------------------------

   -- Version: major*10000 + minor*100 + patch (v2.0.0 → 20000)
   --
   function C_Get_Version return C_Int
      with Export, Convention => C,
           External_Name => "step_entropy_get_version";

   function C_Get_Max_Vocab_Size return C_Int
      with Export, Convention => C,
           External_Name => "step_entropy_get_max_vocab_size";

   function C_Get_Max_Tokens_Per_Step return C_Int
      with Export, Convention => C,
           External_Name => "step_entropy_get_max_tokens_per_step";

end Step_Entropy_C_API;
