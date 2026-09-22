------------------------------------------------------------------------
-- STEP ENTROPY - ADA SPECIFICATION v2.1 (arXiv:2508.03346)
--
-- "Compressing Chain-of-Thought in LLMs via Step Entropy"
--
-- Paper-Accurate + SOTA+++ Implementation:
--
--   Core (Paper §2-3):
--     H(t)   = -Σ_{w∈V} p(w|ctx) log₂ p(w|ctx)   [Token Entropy, Eq.1]
--     H(S_i) = (1/M_i) Σ_j H(t_{i,j})             [Step Entropy, mean-norm]
--     Routing: Low→Fast(compress), Medium→Normal, High→Slow(HSGM)
--
--   SOTA+++ Extensions:
--     GRPO Rewards       — Paper Eq.13-15, Table 3 (inference-time)
--     Adaptive Thresholds — Welford online (Paper §5 future work)
--     Entropy Histogram   — Rich distribution telemetry
--     Z-Score Normalization — Cross-model entropy portability
--
-- Aggregation note (see MEMORY.md AD-003):
--   Paper Eq.2 writes H(S_i) = Σ H(t) (sum notation).  Implementation
--   uses MEAN (Avg_Entropy) for routing threshold comparisons because sum
--   is length-dependent and cannot be compared against fixed bit thresholds
--   across variable-length reasoning steps. INTENTIONAL DEVIATION.
--
-- Type correctness:
--   Dynamic_Predicate is used (not Static_Predicate) for Float-derived
--   types.  Static_Predicate is valid only for discrete types (Ada RM
--   3.2.4).  Dynamic_Predicate enforces the invariant at runtime when
--   assertions are enabled (-gnata).
--
-- Status: SOTA+++ v2.1  |  Last Updated: 2026-05-24
------------------------------------------------------------------------

-- Assertion policy: make all contract layers explicit and active.
--
-- DO-178C guidance (and MISRA Ada 2012 Rule 14.3) requires that assertion
-- checking be traceable at the source level, not inferred from compiler
-- flags alone.  pragma Assertion_Policy here supplements the -gnata build
-- flag in garlic_core.gpr: if the project file is replaced or the flag
-- is dropped, these contracts remain enforced at the language level.
--
-- DO-NOT-REMOVE: removing this pragma downgrades these checks from
-- language-level contracts to advisory-only documentation.
--
pragma Assertion_Policy (
   Dynamic_Predicate => Check,   -- Entropy_Value >= 0, Probability_Value in [0,1]
   Pre               => Check,   -- All preconditions on public subprograms
   Post              => Check    -- All postconditions on public subprograms
);

package Step_Entropy is

   -----------------------------------------------------------------------
   -- PAPER CONSTANTS (arXiv:2508.03346, Table 3, §3.2-3.3)
   -----------------------------------------------------------------------

   -- GRPO reward thresholds
   K_Skip_High           : constant Float   := 0.80; -- κ_high → R_skip_ratio = 1.0
   K_Skip_Low            : constant Float   := 0.50; -- κ_low  → R_skip_ratio = 0.5
   Tau_Skip_Max          : constant Natural := 100;  -- τ_skip: skip-count penalty gate
   Tau_Length_Max        : constant Natural := 3_500;-- τ_length: response-length gate
   R_Correctness_Value   : constant Float   := 2.0;  -- Eq.14: oracle reward (external)
   Optimal_Pruning_Ratio : constant Float   := 0.80; -- κ: §3.2 empirical sweet spot

   -- Default routing thresholds (bits, log₂)
   Default_Threshold_Low  : constant Float := 2.0;   -- < 2 bits → LOW  (compress)
   Default_Threshold_High : constant Float := 4.0;   -- > 4 bits → HIGH (retrieve)

   -----------------------------------------------------------------------
   -- ENTROPY LEVEL
   -----------------------------------------------------------------------

   type Entropy_Level is (Low, Medium, High);

   -----------------------------------------------------------------------
   -- MATHEMATICAL TYPES WITH VERIFIED INVARIANTS
   -----------------------------------------------------------------------

   -- Dynamic_Predicate (not Static_Predicate) is correct for Float subtypes.
   -- Static_Predicate is restricted to discrete types only (Ada RM 3.2.4).

   type Entropy_Value is new Float
      with Dynamic_Predicate => Entropy_Value >= 0.0;

   type Probability_Value is new Float
      with Dynamic_Predicate =>
         Probability_Value >= 0.0 and Probability_Value <= 1.0;

   -- Compression ratio: enforces [0.0, 1.0] on every assignment, not
   -- just at the Analyze_Chain Post-condition check.  This catches any
   -- arithmetic overflow in the compression-ratio calculation at the
   -- type boundary before it can propagate into routing decisions.
   type Compression_Ratio_Value is new Float
      with Dynamic_Predicate =>
         Compression_Ratio_Value >= 0.0 and Compression_Ratio_Value <= 1.0;

   -----------------------------------------------------------------------
   -- VOCABULARY / STEP / HISTOGRAM BOUNDS
   -----------------------------------------------------------------------

   subtype Token_ID is Integer range 0 .. 999_999;

   Max_Tokens_Per_Step : constant := 512;
   Max_Vocab_Size      : constant := 262_144;  -- model-agnostic ceiling (covers GPT-2, Llama-3+, Gemma 3, future)
   Max_Histogram_Bins  : constant := 32;

   -----------------------------------------------------------------------
   -- CORE ARRAY TYPES
   -----------------------------------------------------------------------

   type Logit_Array        is array (1 .. Max_Vocab_Size)      of Float;
   type Entropy_Array      is array (1 .. Max_Tokens_Per_Step)  of Entropy_Value;
   type Probability_Array  is array (1 .. Max_Tokens_Per_Step)  of Probability_Value;
   type Token_ID_Array     is array (1 .. Max_Tokens_Per_Step)  of Token_ID;
   type Integer_Array      is array (Positive range <>)         of Integer;
   type Natural_Array      is array (Positive range <>)         of Natural;

   Max_Token_Text_Length : constant := 64;
   subtype Token_Text    is String (1 .. Max_Token_Text_Length);
   type Token_Text_Array is array (1 .. Max_Tokens_Per_Step) of Token_Text;

   -----------------------------------------------------------------------
   -- TOKEN-LEVEL ENTROPY RECORD
   -----------------------------------------------------------------------

   type Token_Entropy is record
      Token_Index      : Token_ID;
      Token_String     : Token_Text;
      Entropy          : Entropy_Value;
      Probability      : Probability_Value;
      Context_Position : Natural;
   end record;

   type Token_Entropy_Array is array (1 .. Max_Tokens_Per_Step) of Token_Entropy;

   type Token_Logits_Array is array (Positive range <>) of Logit_Array;
   subtype Window_Logits_Array is Token_Logits_Array (1 .. Max_Tokens_Per_Step);

   -----------------------------------------------------------------------
   -- STEP-LEVEL ENTROPY RECORD
   -----------------------------------------------------------------------

   type Step_Entropy is record
      Step_Id         : Natural;
      Step_Text       : Token_Text;
      Token_Count     : Natural;
      Token_Entropies : Token_Entropy_Array;
      Total_Entropy   : Entropy_Value;
      Avg_Entropy     : Entropy_Value;
      Max_Entropy     : Entropy_Value;
      Level           : Entropy_Level;
   end record;

   type Step_Entropy_Array is array (Positive range <>) of Step_Entropy;

   -----------------------------------------------------------------------
   -- COMPRESSION ANALYSIS RECORD
   -----------------------------------------------------------------------

   type Compression_Analysis is record
      Total_Steps          : Natural;
      Compressible_Steps   : Natural;
      Informative_Steps    : Natural;
      Low_Entropy_Steps    : Natural;
      Medium_Entropy_Steps : Natural;
      High_Entropy_Steps   : Natural;
      Compression_Ratio    : Compression_Ratio_Value;   -- [0.0, 1.0] enforced by type
      Avg_Entropy          : Entropy_Value;
   end record;

   -----------------------------------------------------------------------
   -- ROUTING DECISION RECORD
   -----------------------------------------------------------------------

   type Routing_Path is (Fast, Normal, Slow);

   type Routing_Decision is record
      Path            : Routing_Path;
      Retrieve_Memory : Boolean;
      Total_Entropy   : Entropy_Value;
      Avg_Entropy     : Entropy_Value;
      Max_Entropy     : Entropy_Value;
      Level           : Entropy_Level;
   end record;

   -----------------------------------------------------------------------
   -- SOTA+++: ENTROPY HISTOGRAM
   --
   -- Rich per-step token-entropy distribution for telemetry, adaptive
   -- threshold learning, and cross-model analysis.
   -- Single-pass Welford algorithm: mean and std_dev computed in one loop.
   -----------------------------------------------------------------------

   type Histogram_Bin_Array is array (1 .. Max_Histogram_Bins) of Natural;

   type Entropy_Histogram is record
      Bin_Count    : Positive range 1 .. Max_Histogram_Bins;
      Bin_Width    : Float;                -- entropy range per bin (bits)
      Min_Entropy  : Float;                -- minimum observed entropy
      Max_Entropy  : Float;                -- maximum observed entropy
      Bins         : Histogram_Bin_Array;  -- token count per bin
      Total_Tokens : Natural;              -- total samples
      Mean_Entropy : Float;                -- Welford-computed mean (bits)
      Std_Dev      : Float;                -- Welford-computed std dev >= 0
   end record;

   -----------------------------------------------------------------------
   -- SOTA+++: ADAPTIVE THRESHOLD STATE (Welford Online Algorithm)
   --
   -- Paper §5: "adaptive thresholds for task-aware compression".
   -- threshold_low  = Running_Mean + Low_Sigma_Factor  * std
   -- threshold_high = Running_Mean + High_Sigma_Factor * std
   -- Default: low = mean - 1σ, high = mean + 1σ
   -----------------------------------------------------------------------

   type Threshold_State is record
      Running_Mean      : Float   := 0.0;  -- Welford running mean
      Running_M2        : Float   := 0.0;  -- Welford M2 (sum of squared deltas)
      Sample_Count      : Natural := 0;
      Low_Sigma_Factor  : Float   := -1.0; -- multiplier for low threshold
      High_Sigma_Factor : Float   :=  1.0; -- multiplier for high threshold
   end record;

   -----------------------------------------------------------------------
   -- SOTA+++: GRPO REWARD RECORD (Paper Eq.13-15, Table 3)
   --
   -- Kernel computes inference-time reward components.
   -- R_correctness = R_Correctness_Value/0.0 requires external oracle.
   -----------------------------------------------------------------------

   type GRPO_Rewards is record
      R_Skip_Ratio   : Float;  -- 0.0 | 0.5 | 1.0  (Eq.15 tiered)
      R_Skip_Num     : Float;  -- 0.0 | -1.0        (R_sn penalty)
      R_Response_Len : Float;  -- 0.0 | -1.0        (R_rl penalty)
      R_Total        : Float;  -- R_Skip_Ratio + R_Skip_Num + R_Response_Len
   end record;

   -----------------------------------------------------------------------
   -- TOKEN-LEVEL ENTROPY
   -----------------------------------------------------------------------

   --  H(t) = -Σ_{w∈V} p(w|ctx) log₂ p(w|ctx)
   --  Logits are raw language-model outputs; softmax is applied internally.
   --
   --  Pre:  Base > 0.0
   --  Post: Result >= 0.0
   --
   function Calculate_Token_Entropy (
      Logits : Logit_Array;
      Base   : Float := 2.0
   ) return Entropy_Value
      with Pre  => Base > 0.0,
           Post => Calculate_Token_Entropy'Result >= 0.0;

   --  Softmax probability p(token_index | logits).
   --  Post: Result ∈ [0.0, 1.0]
   --
   function Calculate_Token_Probability (
      Logits      : Logit_Array;
      Token_Index : Token_ID
   ) return Probability_Value;

   -----------------------------------------------------------------------
   -- STEP-LEVEL ENTROPY
   -----------------------------------------------------------------------

   --  H(S_i) = (1/M_i) Σ_j H(t_{i,j})  — mean over M_i tokens.
   --  Total_Entropy = Σ H(t) is stored but routing uses Avg_Entropy.
   --
   --  Pre:  0 < Token_Count <= Max_Tokens_Per_Step
   --        Threshold_Low < Threshold_High
   --  Post: Result.Token_Count = Token_Count
   --        Result.Total_Entropy >= 0
   --        Result.Avg_Entropy   >= 0
   --
   function Calculate_Step_Entropy (
      Token_Logits   : Token_Logits_Array;
      Token_Ids      : Token_ID_Array;
      Token_Texts    : Token_Text_Array;
      Token_Count    : Positive;
      Step_Id        : Natural;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Step_Entropy
      with Pre  => (Token_Count > 0
               and  Token_Count   <= Max_Tokens_Per_Step
               and  Threshold_Low <  Threshold_High),
           Post => (Calculate_Step_Entropy'Result.Token_Count  = Token_Count
               and  Calculate_Step_Entropy'Result.Total_Entropy >= 0.0
               and  Calculate_Step_Entropy'Result.Avg_Entropy   >= 0.0);

   -----------------------------------------------------------------------
   -- ENTROPY LEVEL CLASSIFICATION
   -----------------------------------------------------------------------

   --  LOW    iff Entropy < Threshold_Low
   --  HIGH   iff Entropy > Threshold_High
   --  MEDIUM otherwise
   --
   --  Post is a formal proof of the classification invariant.
   --
   function Classify_Entropy (
      Entropy        : Entropy_Value;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Entropy_Level
      with Pre  => Threshold_Low < Threshold_High,
           Post => (
              (Entropy <  Threshold_Low  and Classify_Entropy'Result = Low)   or
              (Entropy >  Threshold_High and Classify_Entropy'Result = High)  or
              (Entropy >= Threshold_Low  and Entropy <= Threshold_High
               and Classify_Entropy'Result = Medium)
           );

   -----------------------------------------------------------------------
   -- STEP QUERIES
   -----------------------------------------------------------------------

   function Is_Compressible (
      Step      : Step_Entropy;
      Threshold : Entropy_Value
   ) return Boolean
      with Post => (Is_Compressible'Result = (Step.Avg_Entropy < Threshold));

   function Is_Informative (
      Step      : Step_Entropy;
      Threshold : Entropy_Value
   ) return Boolean
      with Post => (Is_Informative'Result = (Step.Avg_Entropy > Threshold));

   -----------------------------------------------------------------------
   -- COMPRESSION ANALYSIS
   -----------------------------------------------------------------------

   --  Analyze an array of Step_Entropy records for compression opportunity.
   --
   --  Post: Total_Steps = Step_Count
   --        Compression_Ratio ∈ [0.0, 1.0]
   --        Compressible_Steps, Informative_Steps ≤ Step_Count
   --
   function Analyze_Chain (
      Steps          : Step_Entropy_Array;
      Step_Count     : Positive;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Compression_Analysis
      with Pre  => (Step_Count > 0 and Threshold_Low < Threshold_High),
           Post => (Analyze_Chain'Result.Total_Steps        = Step_Count
               and  Analyze_Chain'Result.Compressible_Steps <= Step_Count
               and  Analyze_Chain'Result.Informative_Steps  <= Step_Count);
               -- Compression_Ratio bounds [0.0, 1.0] are enforced by
               -- Compression_Ratio_Value's Dynamic_Predicate — no Post
               -- repetition needed (would be dead code after type check).

   -----------------------------------------------------------------------
   -- ROUTING DECISIONS
   -----------------------------------------------------------------------

   --  Low  → Fast  (skip/compress; no memory retrieval)
   --  High → Slow  (memory retrieval governed by Enable_Memory_Retrieval)
   --  Med  → Normal (standard generation; no retrieval)
   --
   --  Post is a disjunctive proof of the routing invariant.
   --
   function Route_Step (
      Step                    : Step_Entropy;
      Enable_Memory_Retrieval : Boolean := True
   ) return Routing_Decision
      with Post => (
         (Step.Level = Low    and Route_Step'Result.Path = Fast
                              and not Route_Step'Result.Retrieve_Memory)        or
         (Step.Level = High   and Route_Step'Result.Path = Slow
                              and Route_Step'Result.Retrieve_Memory
                                     = Enable_Memory_Retrieval)                 or
         (Step.Level = Medium and Route_Step'Result.Path = Normal
                              and not Route_Step'Result.Retrieve_Memory)
      );

   --  Batch routing: returns Integer_Array (1..Count) with 0=Fast/1=Normal/2=Slow.
   --
   function Batch_Route (
      Step_Entropies : Entropy_Array;
      Count          : Positive;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Integer_Array
      with Pre  => (Count > 0
               and  Count <= Max_Tokens_Per_Step
               and  Threshold_Low < Threshold_High),
           Post => Batch_Route'Result'Length = Count;

   -----------------------------------------------------------------------
   -- PRUNING RECOMMENDATIONS (Paper §3.2)
   -----------------------------------------------------------------------

   --  Rank steps by ascending entropy; select bottom Pruning_Ratio × N.
   --  Prune_Indices is returned sorted ascending.
   --
   --  Post: Count = floor(Step_Count * Pruning_Ratio)
   --        Count ≤ Step_Count
   --
   procedure Recommend_Pruning (
      Steps         : Step_Entropy_Array;
      Step_Count    : Positive;
      Pruning_Ratio : Float;
      Prune_Indices : out Natural_Array;
      Count         : out Natural
   )
      with Pre  => (Step_Count > 0
               and  Pruning_Ratio >= 0.0
               and  Pruning_Ratio <= 1.0),
           Post => (Count <= Step_Count
               and  Count = Natural (Float (Step_Count) * Pruning_Ratio));

   -----------------------------------------------------------------------
   -- REAL-TIME TOKEN WINDOW
   -----------------------------------------------------------------------

   type Token_Window is record
      Logits        : Window_Logits_Array;
      Token_Ids     : Token_ID_Array;
      Token_Texts   : Token_Text_Array;
      Current_Count : Natural;
      Window_Size   : Positive;
      Step_Counter  : Natural;  -- incremented on every successful Flush_Window
   end record;

   --  Initialize an empty window of given size.
   --  Post: Current_Count = 0, Step_Counter = 0, Window_Size = Window_Size
   --
   procedure Initialize_Window (
      Window      : out Token_Window;
      Window_Size : Positive
   )
      with Post => (Window.Current_Count = 0
               and  Window.Window_Size   = Window_Size
               and  Window.Step_Counter  = 0);

   --  Append one token to the window.
   --  Returns True iff the window is now full.
   --  Post: count incremented by 1; result = (new count >= window size)
   --
   function Add_Token_To_Window (
      Window           : in out Token_Window;
      Logits           : Logit_Array;
      Token_Index      : Token_ID;
      Token_Text_Value : Token_Text
   ) return Boolean
      with Pre  => Window.Current_Count < Window.Window_Size,
           Post => (Window.Current_Count = Window.Current_Count'Old + 1
               and  Add_Token_To_Window'Result =
                       (Window.Current_Count >= Window.Window_Size));

   function Is_Window_Full (Window : Token_Window) return Boolean
      with Post => (Is_Window_Full'Result =
                   (Window.Current_Count >= Window.Window_Size));

   function Get_Window_Count (Window : Token_Window) return Natural
      with Post => Get_Window_Count'Result = Window.Current_Count;

   --  Number of complete steps flushed from this window since Initialize.
   --
   function Get_Step_Counter (Window : Token_Window) return Natural
      with Post => Get_Step_Counter'Result = Window.Step_Counter;

   --  Process accumulated tokens into a Step_Entropy and reset the window.
   --  Post: Current_Count = 0, Step_Counter incremented by 1.
   --
   procedure Flush_Window (
      Window         : in out Token_Window;
      Result         : out Step_Entropy;
      Step_Id        : Natural;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   )
      with Post => (Window.Current_Count = 0
               and  Window.Step_Counter  = Window.Step_Counter'Old + 1);

   -----------------------------------------------------------------------
   -- SOTA+++: ENTROPY HISTOGRAM
   -----------------------------------------------------------------------

   --  Compute entropy distribution histogram from token entropy array.
   --  Uses Welford single-pass for mean and std_dev, then assigns bins.
   --
   --  C API output buffer layout:
   --    [bin_width, min_entropy, max_entropy, mean, std_dev, bin_0 .. bin_{N-1}]
   --    Total floats = 5 + Bin_Count
   --
   --  Pre:  Count > 0 and <= Max_Tokens_Per_Step
   --        Bin_Count <= Max_Histogram_Bins
   --  Post: Result.Total_Tokens = Count, Result.Bin_Count = Bin_Count, Std_Dev >= 0
   --
   function Compute_Histogram (
      Entropies : Entropy_Array;
      Count     : Positive;
      Bin_Count : Positive := 16
   ) return Entropy_Histogram
      with Pre  => (Count     > 0
               and  Count     <= Max_Tokens_Per_Step
               and  Bin_Count <= Max_Histogram_Bins),
           Post => (Compute_Histogram'Result.Total_Tokens = Count
               and  Compute_Histogram'Result.Bin_Count    = Bin_Count
               and  Compute_Histogram'Result.Std_Dev      >= 0.0);

   -----------------------------------------------------------------------
   -- SOTA+++: ADAPTIVE THRESHOLD (Welford Online Algorithm)
   -----------------------------------------------------------------------

   --  Initialize state with custom sigma factors.
   --  Pre: Low_Sigma_Factor < High_Sigma_Factor
   --  Post: Sample_Count = 0; sigma factors stored.
   --
   procedure Initialize_Threshold_State (
      State             : out Threshold_State;
      Low_Sigma_Factor  : Float := -1.0;
      High_Sigma_Factor : Float :=  1.0
   )
      with Pre  => Low_Sigma_Factor < High_Sigma_Factor,
           Post => (State.Sample_Count      = 0
               and  State.Low_Sigma_Factor  = Low_Sigma_Factor
               and  State.High_Sigma_Factor = High_Sigma_Factor);

   --  Welford online update — O(1), numerically stable, no overflow risk.
   --  Post: Sample_Count incremented by 1.
   --
   procedure Update_Threshold_State (
      State   : in out Threshold_State;
      Entropy : Entropy_Value
   )
      with Post => State.Sample_Count = State.Sample_Count'Old + 1;

   --  Compute adaptive thresholds from current running statistics.
   --  threshold_low  = mean + Low_Sigma_Factor  * std
   --  threshold_high = mean + High_Sigma_Factor * std
   --  Both are clamped to >= 0 and separated by at least 0.01 bits.
   --
   --  Pre: Sample_Count >= 2 (variance estimate requires at least 2 samples)
   --
   procedure Get_Adaptive_Thresholds (
      State          : Threshold_State;
      Threshold_Low  : out Entropy_Value;
      Threshold_High : out Entropy_Value
   )
      with Pre => State.Sample_Count >= 2;

   -----------------------------------------------------------------------
   -- SOTA+++: Z-SCORE NORMALIZATION
   -----------------------------------------------------------------------

   --  Normalize entropy to z-score for cross-model portability.
   --  Returns Float(Entropy) unchanged when Std_Dev = 0 (no variance yet).
   --
   --  Pre: Std_Dev >= 0.0
   --
   function Normalize_Entropy (
      Entropy : Entropy_Value;
      Mean    : Float;
      Std_Dev : Float
   ) return Float
      with Pre => Std_Dev >= 0.0;

   -----------------------------------------------------------------------
   -- SOTA+++: GRPO REWARD FUNCTION (Paper Eq.13-15, Table 3)
   -----------------------------------------------------------------------

   --  Computes inference-time kernel reward components.
   --  R_correctness (external oracle) is NOT computed here.
   --
   --  R_skip_ratio (Eq.15):
   --    1.0  if Skip_Ratio >= K_High   (strong compression)
   --    0.5  if K_Low <= Skip_Ratio < K_High
   --    0.0  if Skip_Ratio < K_Low     (insufficient compression)
   --  where Skip_Ratio = N_skip / max(1, N_think + N_skip)
   --
   --  R_skip_num:  -1.0 if Skip_Count > Max_Skip, else 0.0
   --  R_response_len: -1.0 if Response_Len > Max_Length, else 0.0
   --  R_Total: R_skip_ratio + R_skip_num + R_response_len
   --
   --  Pre:  0 <= Skip_Ratio <= 1, K_Low < K_High
   --  Post: R_Skip_Ratio ∈ {0.0, 0.5, 1.0}
   --        R_Skip_Num ∈ {-1.0, 0.0}
   --        R_Response_Len ∈ {-1.0, 0.0}
   --
   function Compute_GRPO_Rewards (
      Skip_Ratio   : Float;
      Skip_Count   : Natural;
      Response_Len : Natural;
      K_High       : Float   := K_Skip_High;
      K_Low        : Float   := K_Skip_Low;
      Max_Skip     : Natural := Tau_Skip_Max;
      Max_Length   : Natural := Tau_Length_Max
   ) return GRPO_Rewards
      with Pre  => (Skip_Ratio >= 0.0
               and  Skip_Ratio <= 1.0
               and  K_Low < K_High),
           Post => (Compute_GRPO_Rewards'Result.R_Skip_Ratio   in 0.0 | 0.5 | 1.0
               and  Compute_GRPO_Rewards'Result.R_Skip_Num     in -1.0 | 0.0
               and  Compute_GRPO_Rewards'Result.R_Response_Len in -1.0 | 0.0);

end Step_Entropy;
