------------------------------------------------------------------------
-- STEP ENTROPY - ADA IMPLEMENTATION BODY v2.0
--
-- Paper-Accurate + SOTA+++ Implementation of arXiv:2508.03346
-- "Compressing Chain-of-Thought in LLMs via Step Entropy"
--
-- Status: SOTA+++ v2.1  |  Last Updated: 2026-05-24
------------------------------------------------------------------------

with Ada.Numerics;
with Ada.Numerics.Elementary_Functions;
with Ada.Containers.Generic_Array_Sort;

package body Step_Entropy is

   use Ada.Numerics.Elementary_Functions;
   use Ada.Numerics;

   -----------------------------------------------------------------------
   -- MATHEMATICAL UTILITIES (private)
   -----------------------------------------------------------------------

   -- Epsilon for numerical stability — matches Python reference (1e-10)
   Epsilon : constant := 1.0e-10;

   -- Log base 2 cached to avoid recomputation in tight loops
   Log_2_Base : constant Float := Log (2.0);

   -- Minimum separation between adaptive thresholds (bits)
   Min_Threshold_Gap : constant Float := 0.01;

   --  Safe logarithm: clamps argument to [Epsilon, +∞) before computing.
   --  Base parameter: 2.0 = bits (default), e = nats, other = general.
   --  2.0 is exactly representable in IEEE 754, so equality is safe here.
   --
   function Safe_Log (X : Float; Base : Float := 2.0) return Float is
      Clamped : constant Float := Float'Max (X, Epsilon);
   begin
      if Base = 2.0 then
         return Log (Clamped) / Log_2_Base;
      elsif Base = e then
         return Log (Clamped);
      else
         return Log (Clamped) / Log (Float'Max (Base, Epsilon));
      end if;
   end Safe_Log;

   --  Numerically stable softmax with max-subtraction trick.
   --  Handles degenerate case (all identical logits or underflow) by
   --  returning the uniform distribution.
   --
   function Softmax (Logits : Logit_Array) return Logit_Array is
      Result    : Logit_Array := Logits;
      Max_Logit : Float       := Logits (Logits'First);
      Sum_Exp   : Float       := 0.0;
      Exp_Val   : Float;
   begin
      -- Step 1: Find maximum logit (stability)
      for I in Logits'Range loop
         if Logits (I) > Max_Logit then
            Max_Logit := Logits (I);
         end if;
      end loop;

      -- Step 2: Compute exp(logit - max) and accumulate
      for I in Logits'Range loop
         Exp_Val    := Exp (Logits (I) - Max_Logit);
         Result (I) := Exp_Val;
         Sum_Exp    := Sum_Exp + Exp_Val;
      end loop;

      -- Step 3: Normalize
      if Sum_Exp > Epsilon then
         for I in Result'Range loop
            Result (I) := Result (I) / Sum_Exp;
         end loop;
      else
         -- Degenerate: return uniform distribution
         Result := (others => 1.0 / Float (Logits'Length));
      end if;

      return Result;
   end Softmax;

   -----------------------------------------------------------------------
   -- TOKEN-LEVEL ENTROPY
   -----------------------------------------------------------------------

   function Calculate_Token_Entropy (
      Logits : Logit_Array;
      Base   : Float := 2.0
   ) return Entropy_Value is

      Probs   : Logit_Array;
      Entropy : Float := 0.0;

   begin
      if Base <= 0.0 then
         raise Constraint_Error with "Base must be positive";
      end if;

      Probs := Softmax (Logits);

      for I in Probs'Range loop
         if Probs (I) > Epsilon then
            Entropy := Entropy - (Probs (I) * Safe_Log (Probs (I), Base));
         end if;
      end loop;

      return Entropy_Value (Float'Max (Entropy, 0.0));

   end Calculate_Token_Entropy;

   function Calculate_Token_Probability (
      Logits      : Logit_Array;
      Token_Index : Token_ID
   ) return Probability_Value is

      Probs : Logit_Array;
      Index : Integer;

   begin
      if Token_Index < 0 or Token_Index >= Max_Vocab_Size then
         raise Constraint_Error with
            "Token_Index" & Integer'Image (Token_Index) & " out of vocabulary range";
      end if;

      Probs := Softmax (Logits);
      Index := Integer (Token_Index) + 1;  -- 0-indexed → 1-indexed

      if Index < Probs'First or Index > Probs'Last then
         return Probability_Value (0.0);
      end if;

      return Probability_Value (
         Float'Min (Float'Max (Probs (Index), 0.0), 1.0)
      );

   end Calculate_Token_Probability;

   -----------------------------------------------------------------------
   -- STEP-LEVEL ENTROPY CALCULATION
   -----------------------------------------------------------------------

   function Calculate_Step_Entropy (
      Token_Logits   : Token_Logits_Array;
      Token_Ids      : Token_ID_Array;
      Token_Texts    : Token_Text_Array;
      Token_Count    : Positive;
      Step_Id        : Natural;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Step_Entropy is

      Result          : Step_Entropy;
      Token_Ent_Val   : Entropy_Value;
      Token_Prob      : Probability_Value;
      Total           : Float := 0.0;
      Max_Val         : Float := 0.0;
      Step_Text_Idx   : Positive := 1;

   begin
      if Token_Count = 0 or Token_Count > Max_Tokens_Per_Step then
         raise Constraint_Error with
            "Token_Count must be in [1," & Natural'Image (Max_Tokens_Per_Step) & "]";
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         raise Constraint_Error with "Threshold_Low must be < Threshold_High";
      end if;

      Result.Step_Id         := Step_Id;
      Result.Token_Count     := Token_Count;
      Result.Total_Entropy   := Entropy_Value (0.0);
      Result.Avg_Entropy     := Entropy_Value (0.0);
      Result.Max_Entropy     := Entropy_Value (0.0);
      Result.Step_Text       := (others => ' ');

      for I in 1 .. Token_Count loop
         if I < Token_Logits'First or I > Token_Logits'Last then
            raise Constraint_Error with "Token index out of range in Token_Logits";
         end if;
         if I < Token_Ids'First or I > Token_Ids'Last then
            raise Constraint_Error with "Token index out of range in Token_Ids";
         end if;
         if I < Token_Texts'First or I > Token_Texts'Last then
            raise Constraint_Error with "Token index out of range in Token_Texts";
         end if;

         Token_Ent_Val := Calculate_Token_Entropy (
            Token_Logits (Token_Logits'First + I - 1), 2.0);

         Token_Prob := Calculate_Token_Probability (
            Token_Logits (Token_Logits'First + I - 1),
            Token_Ids (Token_Ids'First + I - 1));

         Result.Token_Entropies (I) := (
            Token_Index      => Token_Ids (Token_Ids'First + I - 1),
            Token_String     => Token_Texts (Token_Texts'First + I - 1),
            Entropy          => Token_Ent_Val,
            Probability      => Token_Prob,
            Context_Position => I - 1
         );

         Total := Total + Float (Token_Ent_Val);

         if Float (Token_Ent_Val) > Max_Val then
            Max_Val := Float (Token_Ent_Val);
         end if;

         -- Accumulate step text (first 64 chars of concatenated tokens)
         declare
            Token_Str : constant Token_Text :=
               Token_Texts (Token_Texts'First + I - 1);
            Last_Non_Space : Natural := 0;
         begin
            for J in Token_Str'Range loop
               if Token_Str (J) /= ' ' then
                  Last_Non_Space := J - Token_Str'First + 1;
               end if;
            end loop;

            for J in 1 .. Last_Non_Space loop
               if Step_Text_Idx <= Result.Step_Text'Last then
                  Result.Step_Text (Step_Text_Idx) :=
                     Token_Str (Token_Str'First + J - 1);
                  Step_Text_Idx := Step_Text_Idx + 1;
               end if;
            end loop;
         end;
      end loop;

      Result.Total_Entropy := Entropy_Value (Total);
      Result.Max_Entropy   := Entropy_Value (Max_Val);
      Result.Avg_Entropy   := Entropy_Value (Total / Float (Token_Count));

      -- Classify routing level
      if Result.Avg_Entropy < Threshold_Low then
         Result.Level := Low;
      elsif Result.Avg_Entropy > Threshold_High then
         Result.Level := High;
      else
         Result.Level := Medium;
      end if;

      return Result;

   end Calculate_Step_Entropy;

   -----------------------------------------------------------------------
   -- ENTROPY LEVEL CLASSIFICATION
   -----------------------------------------------------------------------

   function Classify_Entropy (
      Entropy        : Entropy_Value;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Entropy_Level is
   begin
      if Entropy < Threshold_Low then
         return Low;
      elsif Entropy > Threshold_High then
         return High;
      else
         return Medium;
      end if;
   end Classify_Entropy;

   -----------------------------------------------------------------------
   -- STEP QUERIES
   -----------------------------------------------------------------------

   function Is_Compressible (
      Step      : Step_Entropy;
      Threshold : Entropy_Value
   ) return Boolean is
   begin
      return Step.Avg_Entropy < Threshold;
   end Is_Compressible;

   function Is_Informative (
      Step      : Step_Entropy;
      Threshold : Entropy_Value
   ) return Boolean is
   begin
      return Step.Avg_Entropy > Threshold;
   end Is_Informative;

   -----------------------------------------------------------------------
   -- COMPRESSION ANALYSIS
   -----------------------------------------------------------------------

   function Analyze_Chain (
      Steps          : Step_Entropy_Array;
      Step_Count     : Positive;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Compression_Analysis is

      Result             : Compression_Analysis;
      Compressible_Count : Natural := 0;
      Informative_Count  : Natural := 0;
      Low_Count          : Natural := 0;
      Medium_Count       : Natural := 0;
      High_Count         : Natural := 0;
      Avg_Entropy_Sum    : Float   := 0.0;

   begin
      if Step_Count = 0 then
         raise Constraint_Error with "Step_Count must be positive";
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         raise Constraint_Error with "Threshold_Low must be < Threshold_High";
      end if;

      for I in Steps'First .. Steps'First + Step_Count - 1 loop
         if Is_Compressible (Steps (I), Threshold_Low) then
            Compressible_Count := Compressible_Count + 1;
         end if;

         if Is_Informative (Steps (I), Threshold_High) then
            Informative_Count := Informative_Count + 1;
         end if;

         case Steps (I).Level is
            when Low    => Low_Count    := Low_Count    + 1;
            when Medium => Medium_Count := Medium_Count + 1;
            when High   => High_Count   := High_Count   + 1;
         end case;

         Avg_Entropy_Sum := Avg_Entropy_Sum + Float (Steps (I).Avg_Entropy);
      end loop;

      Result.Total_Steps          := Step_Count;
      Result.Compressible_Steps   := Compressible_Count;
      Result.Informative_Steps    := Informative_Count;
      Result.Low_Entropy_Steps    := Low_Count;
      Result.Medium_Entropy_Steps := Medium_Count;
      Result.High_Entropy_Steps   := High_Count;
      Result.Compression_Ratio    :=
         Compression_Ratio_Value (Float (Compressible_Count) / Float (Step_Count));
         -- Cast is safe: 0 <= Compressible_Count <= Step_Count by construction,
         -- so the ratio is always in [0.0, 1.0].  The Dynamic_Predicate on
         -- Compression_Ratio_Value will fire (with -gnata) if arithmetic ever
         -- produces an out-of-range value, turning a silent logic error into a
         -- loud assertion failure.
      Result.Avg_Entropy          :=
         Entropy_Value (Avg_Entropy_Sum / Float (Step_Count));

      return Result;

   end Analyze_Chain;

   -----------------------------------------------------------------------
   -- ROUTING DECISIONS
   -----------------------------------------------------------------------

   function Route_Step (
      Step                    : Step_Entropy;
      Enable_Memory_Retrieval : Boolean := True
   ) return Routing_Decision is

      Decision : Routing_Decision;

   begin
      case Step.Level is
         when Low =>
            Decision.Path           := Fast;
            Decision.Retrieve_Memory := False;

         when High =>
            Decision.Path           := Slow;
            Decision.Retrieve_Memory := Enable_Memory_Retrieval;

         when Medium =>
            Decision.Path           := Normal;
            Decision.Retrieve_Memory := False;
      end case;

      Decision.Total_Entropy := Step.Total_Entropy;
      Decision.Avg_Entropy   := Step.Avg_Entropy;
      Decision.Max_Entropy   := Step.Max_Entropy;
      Decision.Level         := Step.Level;

      return Decision;

   end Route_Step;

   -----------------------------------------------------------------------
   -- BATCH ROUTING
   -----------------------------------------------------------------------

   procedure Batch_Route_Internal (
      Step_Entropies : Entropy_Array;
      Count          : Positive;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value;
      Decisions      : out Integer_Array
   ) is
      Level : Entropy_Level;
   begin
      if Count = 0 or Count > Max_Tokens_Per_Step then
         raise Constraint_Error with "Invalid Count for Batch_Route";
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         raise Constraint_Error with "Threshold_Low must be < Threshold_High";
      end if;

      for I in 1 .. Count loop
         Level := Classify_Entropy (
            Step_Entropies (I), Threshold_Low, Threshold_High);

         case Level is
            when Low    => Decisions (I) := 0;
            when Medium => Decisions (I) := 1;
            when High   => Decisions (I) := 2;
         end case;
      end loop;
   end Batch_Route_Internal;

   function Batch_Route (
      Step_Entropies : Entropy_Array;
      Count          : Positive;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) return Integer_Array is
      Decisions : Integer_Array (1 .. Count);
   begin
      Batch_Route_Internal (
         Step_Entropies => Step_Entropies,
         Count          => Count,
         Threshold_Low  => Threshold_Low,
         Threshold_High => Threshold_High,
         Decisions      => Decisions
      );
      return Decisions;
   end Batch_Route;

   -----------------------------------------------------------------------
   -- PRUNING RECOMMENDATIONS
   -----------------------------------------------------------------------

   type Indexed_Entropy is record
      Index   : Natural;
      Entropy : Entropy_Value;
   end record;

   type Indexed_Entropy_Array is array (Natural range <>) of Indexed_Entropy;

   function Less_Indexed_Entropy (
      Left  : Indexed_Entropy;
      Right : Indexed_Entropy
   ) return Boolean is
   begin
      return Left.Entropy < Right.Entropy;
   end Less_Indexed_Entropy;

   procedure Sort_By_Entropy is new Ada.Containers.Generic_Array_Sort (
      Index_Type   => Natural,
      Element_Type => Indexed_Entropy,
      Array_Type   => Indexed_Entropy_Array,
      "<"          => Less_Indexed_Entropy
   );

   -- Second instantiation: sorts output Natural_Array indices ascending.
   -- Replaces the O(n²) bubble sort that was used to re-order prune indices
   -- after Sort_By_Entropy selected the lowest-entropy steps.
   --
   -- Complexity: O(n log n) average and worst-case.
   -- GNAT implements Generic_Array_Sort as introsort (quicksort with
   -- heapsort fallback at depth > 2·log₂(n)), which guarantees O(n log n)
   -- worst-case and uses in-place insertion sort for n ≤ 16.
   --
   -- At the maximum input size (n = Max_Tokens_Per_Step = 512):
   --   Introsort: ≤ 512 × log₂(512) × 2 ≈ 9,216 comparisons (typical)
   --   Bubble sort (prior): up to 512² / 2 = 131,072 comparisons (worst)
   --
   function Less_Natural (Left, Right : Natural) return Boolean is
   begin
      return Left < Right;
   end Less_Natural;

   procedure Natural_Index_Sort is new Ada.Containers.Generic_Array_Sort (
      Index_Type   => Positive,
      Element_Type => Natural,
      Array_Type   => Natural_Array,
      "<"          => Less_Natural
   );

   procedure Recommend_Pruning (
      Steps         : Step_Entropy_Array;
      Step_Count    : Positive;
      Pruning_Ratio : Float;
      Prune_Indices : out Natural_Array;
      Count         : out Natural
   ) is

      Indexed_Steps : Indexed_Entropy_Array (1 .. Step_Count);
      Num_To_Prune  : Natural;

   begin
      if Step_Count = 0 then
         raise Constraint_Error with "Step_Count must be positive";
      end if;

      if Pruning_Ratio < 0.0 or Pruning_Ratio > 1.0 then
         raise Constraint_Error with
            "Pruning_Ratio must be in [0.0, 1.0], got:"
            & Float'Image (Pruning_Ratio);
      end if;

      for I in 1 .. Step_Count loop
         Indexed_Steps (I) := (
            Index   => I - 1,
            Entropy => Steps (Steps'First + I - 1).Avg_Entropy
         );
      end loop;

      Sort_By_Entropy (Indexed_Steps);

      Num_To_Prune := Natural (Float (Step_Count) * Pruning_Ratio);

      if Num_To_Prune > Prune_Indices'Length then
         Num_To_Prune := Prune_Indices'Length;
      end if;

      for I in 1 .. Num_To_Prune loop
         Prune_Indices (I) := Indexed_Steps (I).Index;
      end loop;

      Count := Num_To_Prune;

      -- Re-sort the selected prune indices ascending by original position
      -- so callers receive a predictable, position-ordered list.
      -- O(n log n) via Natural_Index_Sort (introsort).
      if Count > 1 then
         Natural_Index_Sort (Prune_Indices (1 .. Count));
      end if;

   end Recommend_Pruning;

   -----------------------------------------------------------------------
   -- TOKEN WINDOW MANAGEMENT
   -----------------------------------------------------------------------

   procedure Initialize_Window (
      Window      : out Token_Window;
      Window_Size : Positive
   ) is
   begin
      if Window_Size > Max_Tokens_Per_Step then
         raise Constraint_Error with
            "Window_Size" & Positive'Image (Window_Size) &
            " exceeds Max_Tokens_Per_Step =" &
            Natural'Image (Max_Tokens_Per_Step);
      end if;

      Window.Window_Size   := Window_Size;
      Window.Current_Count := 0;
      Window.Step_Counter  := 0;
   end Initialize_Window;

   function Add_Token_To_Window (
      Window           : in out Token_Window;
      Logits           : Logit_Array;
      Token_Index      : Token_ID;
      Token_Text_Value : Token_Text
   ) return Boolean is

      New_Pos : Positive;

   begin
      if Window.Current_Count >= Window.Window_Size then
         raise Constraint_Error with "Window is already full";
      end if;

      New_Pos := Window.Current_Count + 1;

      if New_Pos > Max_Tokens_Per_Step then
         raise Constraint_Error with "Would exceed Max_Tokens_Per_Step";
      end if;

      Window.Logits      (New_Pos) := Logits;
      Window.Token_Ids   (New_Pos) := Token_Index;
      Window.Token_Texts (New_Pos) := Token_Text_Value;
      Window.Current_Count         := New_Pos;

      return Window.Current_Count >= Window.Window_Size;

   end Add_Token_To_Window;

   function Is_Window_Full (Window : Token_Window) return Boolean is
   begin
      return Window.Current_Count >= Window.Window_Size;
   end Is_Window_Full;

   function Get_Window_Count (Window : Token_Window) return Natural is
   begin
      return Window.Current_Count;
   end Get_Window_Count;

   function Get_Step_Counter (Window : Token_Window) return Natural is
   begin
      return Window.Step_Counter;
   end Get_Step_Counter;

   procedure Flush_Window (
      Window         : in out Token_Window;
      Result         : out Step_Entropy;
      Step_Id        : Natural;
      Threshold_Low  : Entropy_Value;
      Threshold_High : Entropy_Value
   ) is
      Token_Count : Natural;
   begin
      if Window.Current_Count = 0 then
         raise Constraint_Error with "Cannot flush empty window";
      end if;

      Token_Count := Window.Current_Count;

      Result := Calculate_Step_Entropy (
         Token_Logits   => Window.Logits (1 .. Token_Count),
         Token_Ids      => Window.Token_Ids (1 .. Token_Count),
         Token_Texts    => Window.Token_Texts (1 .. Token_Count),
         Token_Count    => Token_Count,
         Step_Id        => Step_Id,
         Threshold_Low  => Threshold_Low,
         Threshold_High => Threshold_High
      );

      Window.Current_Count := 0;
      Window.Step_Counter  := Window.Step_Counter + 1;

   end Flush_Window;

   -----------------------------------------------------------------------
   -- SOTA+++: ENTROPY HISTOGRAM
   -----------------------------------------------------------------------

   function Compute_Histogram (
      Entropies : Entropy_Array;
      Count     : Positive;
      Bin_Count : Positive := 16
   ) return Entropy_Histogram is

      Result : Entropy_Histogram;

      -- Welford running stats
      W_Mean  : Float := 0.0;
      W_M2    : Float := 0.0;
      W_Count : Float := 0.0;
      Delta1, Delta2 : Float;

      Min_E : Float := Float (Entropies (1));
      Max_E : Float := Float (Entropies (1));

      Variance : Float;
      Bin_Idx  : Integer;

   begin
      if Count = 0 or Count > Max_Tokens_Per_Step then
         raise Constraint_Error with "Count out of range for Compute_Histogram";
      end if;

      if Bin_Count > Max_Histogram_Bins then
         raise Constraint_Error with "Bin_Count exceeds Max_Histogram_Bins";
      end if;

      -- Initialize result
      Result.Bin_Count    := Bin_Count;
      Result.Total_Tokens := Count;
      Result.Bins         := (others => 0);

      -- Single-pass Welford for min, max, mean, variance
      for I in 1 .. Count loop
         declare
            Val : constant Float := Float (Entropies (I));
         begin
            W_Count := W_Count + 1.0;
            Delta1  := Val - W_Mean;
            W_Mean  := W_Mean + Delta1 / W_Count;
            Delta2  := Val - W_Mean;
            W_M2    := W_M2 + Delta1 * Delta2;

            if Val < Min_E then Min_E := Val; end if;
            if Val > Max_E then Max_E := Val; end if;
         end;
      end loop;

      Result.Mean_Entropy := W_Mean;
      Result.Min_Entropy  := Min_E;
      Result.Max_Entropy  := Max_E;

      if Count > 1 then
         Variance    := W_M2 / Float (Count - 1);
         Result.Std_Dev := Sqrt (Float'Max (Variance, 0.0));
      else
         Result.Std_Dev := 0.0;
      end if;

      -- Assign bins
      declare
         Bin_Range : constant Float := Max_E - Min_E;
      begin
         if Bin_Range <= 0.0 then
            -- All values identical: put everything in bin 1
            Result.Bin_Width := 1.0;
            Result.Bins (1)  := Count;
         else
            Result.Bin_Width := Bin_Range / Float (Bin_Count);

            for I in 1 .. Count loop
               Bin_Idx := Integer (
                  Float'Floor (
                     (Float (Entropies (I)) - Min_E) / Result.Bin_Width
                  )
               ) + 1;
               -- Clamp: the maximum-value sample falls exactly on the upper edge
               Bin_Idx := Integer'Max (1, Integer'Min (Bin_Count, Bin_Idx));
               Result.Bins (Bin_Idx) := Result.Bins (Bin_Idx) + 1;
            end loop;
         end if;
      end;

      return Result;

   end Compute_Histogram;

   -----------------------------------------------------------------------
   -- SOTA+++: ADAPTIVE THRESHOLD STATE
   -----------------------------------------------------------------------

   procedure Initialize_Threshold_State (
      State             : out Threshold_State;
      Low_Sigma_Factor  : Float := -1.0;
      High_Sigma_Factor : Float :=  1.0
   ) is
   begin
      if Low_Sigma_Factor >= High_Sigma_Factor then
         raise Constraint_Error with
            "Low_Sigma_Factor must be < High_Sigma_Factor";
      end if;

      State.Running_Mean      := 0.0;
      State.Running_M2        := 0.0;
      State.Sample_Count      := 0;
      State.Low_Sigma_Factor  := Low_Sigma_Factor;
      State.High_Sigma_Factor := High_Sigma_Factor;
   end Initialize_Threshold_State;

   procedure Update_Threshold_State (
      State   : in out Threshold_State;
      Entropy : Entropy_Value
   ) is
      Val    : constant Float := Float (Entropy);
      Delta1 : Float;
      Delta2 : Float;
   begin
      State.Sample_Count := State.Sample_Count + 1;

      Delta1              := Val - State.Running_Mean;
      State.Running_Mean  := State.Running_Mean +
                             Delta1 / Float (State.Sample_Count);
      Delta2              := Val - State.Running_Mean;
      State.Running_M2    := State.Running_M2 + Delta1 * Delta2;
   end Update_Threshold_State;

   procedure Get_Adaptive_Thresholds (
      State          : Threshold_State;
      Threshold_Low  : out Entropy_Value;
      Threshold_High : out Entropy_Value
   ) is
      Variance : Float;
      Std_Dev  : Float;
      Low_Val  : Float;
      High_Val : Float;
   begin
      if State.Sample_Count < 2 then
         raise Constraint_Error with
            "At least 2 samples required for variance estimate";
      end if;

      Variance := State.Running_M2 / Float (State.Sample_Count - 1);
      Std_Dev  := Sqrt (Float'Max (Variance, 0.0));

      Low_Val  := State.Running_Mean + State.Low_Sigma_Factor  * Std_Dev;
      High_Val := State.Running_Mean + State.High_Sigma_Factor * Std_Dev;

      -- Both thresholds must be non-negative (entropy is >= 0)
      Low_Val  := Float'Max (Low_Val,  0.0);
      High_Val := Float'Max (High_Val, 0.0);

      -- Guarantee minimum separation to maintain Pre on Classify_Entropy
      if High_Val - Low_Val < Min_Threshold_Gap then
         High_Val := Low_Val + Min_Threshold_Gap;
      end if;

      Threshold_Low  := Entropy_Value (Low_Val);
      Threshold_High := Entropy_Value (High_Val);
   end Get_Adaptive_Thresholds;

   -----------------------------------------------------------------------
   -- SOTA+++: Z-SCORE NORMALIZATION
   -----------------------------------------------------------------------

   function Normalize_Entropy (
      Entropy : Entropy_Value;
      Mean    : Float;
      Std_Dev : Float
   ) return Float is
   begin
      if Std_Dev <= 0.0 then
         -- No variance: return raw value (cannot normalize)
         return Float (Entropy);
      else
         return (Float (Entropy) - Mean) / Std_Dev;
      end if;
   end Normalize_Entropy;

   -----------------------------------------------------------------------
   -- SOTA+++: GRPO REWARD FUNCTION (Paper Eq.13-15, Table 3)
   -----------------------------------------------------------------------

   function Compute_GRPO_Rewards (
      Skip_Ratio   : Float;
      Skip_Count   : Natural;
      Response_Len : Natural;
      K_High       : Float   := K_Skip_High;
      K_Low        : Float   := K_Skip_Low;
      Max_Skip     : Natural := Tau_Skip_Max;
      Max_Length   : Natural := Tau_Length_Max
   ) return GRPO_Rewards is

      Result : GRPO_Rewards;

   begin
      if Skip_Ratio < 0.0 or Skip_Ratio > 1.0 then
         raise Constraint_Error with
            "Skip_Ratio must be in [0.0, 1.0], got:" & Float'Image (Skip_Ratio);
      end if;

      if Float (K_Low) >= Float (K_High) then
         raise Constraint_Error with "K_Low must be < K_High";
      end if;

      -- R_skip_ratio (Eq.15): tiered reward based on compression ratio
      if Skip_Ratio >= K_High then
         Result.R_Skip_Ratio := 1.0;
      elsif Skip_Ratio >= K_Low then
         Result.R_Skip_Ratio := 0.5;
      else
         Result.R_Skip_Ratio := 0.0;
      end if;

      -- R_skip_num: penalty for excessive skip token count
      if Skip_Count > Max_Skip then
         Result.R_Skip_Num := -1.0;
      else
         Result.R_Skip_Num := 0.0;
      end if;

      -- R_response_length: penalty for overly long responses
      if Response_Len > Max_Length then
         Result.R_Response_Len := -1.0;
      else
         Result.R_Response_Len := 0.0;
      end if;

      -- R_Total excludes R_correctness (requires external oracle)
      Result.R_Total :=
         Result.R_Skip_Ratio + Result.R_Skip_Num + Result.R_Response_Len;

      return Result;

   end Compute_GRPO_Rewards;

end Step_Entropy;
