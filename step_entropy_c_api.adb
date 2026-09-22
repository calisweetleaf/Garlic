------------------------------------------------------------------------
-- STEP ENTROPY C API — Implementation Body v2.0
--
-- Marshaling layer: C pointers (System.Address) → Ada arrays/records
--
-- All pointer parameters arrive as System.Address (= C void*).
-- The address overlay + pragma Import (Ada, X) pattern is the
-- official Ada 2022 mechanism for accessing C-allocated memory
-- without copying (LRM §13.3, §B.3).
--
-- Error handling: every function catches all Ada exceptions,
-- logs to stderr, returns -1/-1.0. Zero means success.
--
-- Status: Production v2.0
-- Last Updated: 2026-04-25
------------------------------------------------------------------------

with Step_Entropy;
with Ada.Text_IO;
with Ada.Exceptions;
with Ada.Numerics.Elementary_Functions;
with Interfaces.C;
with System;
with System.Storage_Elements;

package body Step_Entropy_C_API is

   use Step_Entropy;
   use Interfaces.C;
   use Ada.Exceptions;
   use System.Storage_Elements;
   use type System.Address;

   -----------------------------------------------------------------------
   -- INTERNAL: Error logging
   -----------------------------------------------------------------------

   procedure Log_Error (Context : String; Message : String) is
   begin
      Ada.Text_IO.Put_Line
         (Ada.Text_IO.Standard_Error,
          "[ADA-STEP-ENTROPY ERROR] " & Context & ": " & Message);
   end Log_Error;

   -----------------------------------------------------------------------
   -- INTERNAL: Address-offset read/write helpers
   --
   -- These are the canonical Ada 2022 zero-copy overlay helpers.
   -- The pragma Import (Ada, X) on each overlay variable suppresses
   -- default initialization so the compiler does not generate a spurious
   -- write before the read (LRM §B.1, §13.3(19)).
   --
   -- GNAT requires that for-address-use expressions reference System.Address
   -- arithmetic via a pre-computed constant (not an inline expression) when
   -- the "+" operator visibility is constrained by the private type rules.
   -- We therefore compute the target address into a constant first.
   -----------------------------------------------------------------------

   --  Float element stride: bytes per C_Float element.
   C_Float_Stride : constant Storage_Offset :=
      Storage_Offset (C_Float'Size / System.Storage_Unit);

   --  Int element stride: bytes per C_Int element.
   C_Int_Stride : constant Storage_Offset :=
      Storage_Offset (C_Int'Size / System.Storage_Unit);

   function Read_Float_At
      (Base   : System.Address;
       Offset : Natural) return C_Float
   is
      Target_Addr : constant System.Address :=
         Base + Storage_Offset (Offset) * C_Float_Stride;
      F : C_Float;
      for F'Address use Target_Addr;
      pragma Import (Ada, F);
   begin
      return F;
   end Read_Float_At;

   procedure Write_Float_At
      (Base   : System.Address;
       Offset : Natural;
       Value  : C_Float)
   is
      Target_Addr : constant System.Address :=
         Base + Storage_Offset (Offset) * C_Float_Stride;
      F : C_Float;
      for F'Address use Target_Addr;
      pragma Import (Ada, F);
   begin
      F := Value;
   end Write_Float_At;

   function Read_Int_At
      (Base   : System.Address;
       Offset : Natural) return C_Int
   is
      Target_Addr : constant System.Address :=
         Base + Storage_Offset (Offset) * C_Int_Stride;
      V : C_Int;
      for V'Address use Target_Addr;
      pragma Import (Ada, V);
   begin
      return V;
   end Read_Int_At;

   procedure Write_Int_At
      (Base   : System.Address;
       Offset : Natural;
       Value  : C_Int)
   is
      Target_Addr : constant System.Address :=
         Base + Storage_Offset (Offset) * C_Int_Stride;
      V : C_Int;
      for V'Address use Target_Addr;
      pragma Import (Ada, V);
   begin
      V := Value;
   end Write_Int_At;

   -----------------------------------------------------------------------
   -- INTERNAL: Build_Logit_Array
   --
   -- Copies Len C floats from the C-side memory at Addr into a
   -- fully-initialized Ada Logit_Array (1..Max_Vocab_Size).
   -- Elements beyond Len are zero so the softmax sees a valid
   -- distribution (extra zeros do not contribute to entropy).
   -----------------------------------------------------------------------

   procedure Build_Logit_Array
      (Addr   : System.Address;
       Len    : C_Int;
       Result : out Logit_Array)
   is
      Copy_Len : constant Integer :=
         Integer'Min (Integer (Len), Max_Vocab_Size);
   begin
      Result := (others => 0.0);
      for I in 0 .. Copy_Len - 1 loop
         Result (I + 1) := Float (Read_Float_At (Addr, I));
      end loop;
   end Build_Logit_Array;

   -----------------------------------------------------------------------
   -- INTERNAL: Epsilon for numerical stability (matches Python 1e-10)
   -----------------------------------------------------------------------

   Epsilon : constant Float := 1.0e-10;

   -----------------------------------------------------------------------
   -- TOKEN-LEVEL ENTROPY
   --
   -- These functions compute directly from C memory using address
   -- arithmetic to AVOID allocating Ada Logit_Array (50257 × 8 = 402 KB)
   -- on the stack.  Calling the high-level Ada Calculate_Token_Entropy
   -- caused stack corruption when three nested frames each held a copy.
   -----------------------------------------------------------------------

   function C_Calculate_Token_Entropy
      (Logits     : System.Address;
       Logits_Len : C_Int;
       Base       : C_Float) return C_Float
   is
      use Ada.Numerics.Elementary_Functions;

      N       : constant Integer := Integer (Logits_Len);
      Log_Base : Float;
      Max_L    : Float;
      Sum_Exp  : Float := 0.0;
      Entropy  : Float := 0.0;
      L, P     : Float;
   begin
      if Logits = System.Null_Address then
         Log_Error ("C_Calculate_Token_Entropy", "Logits address is null");
         return -1.0;
      end if;

      if N <= 0 or N > Max_Vocab_Size then
         Log_Error ("C_Calculate_Token_Entropy",
            "logits_len out of [1," & Natural'Image (Max_Vocab_Size) &
            "]: " & C_Int'Image (Logits_Len));
         return -1.0;
      end if;

      if Float (Base) <= 0.0 then
         Log_Error ("C_Calculate_Token_Entropy",
            "Base must be > 0, got: " & C_Float'Image (Base));
         return -1.0;
      end if;

      -- Precompute log of base for base conversion
      Log_Base := Log (Float (Base));

      -- Pass 1: find maximum logit (numerical stability trick)
      Max_L := Float (Read_Float_At (Logits, 0));
      for I in 1 .. N - 1 loop
         L := Float (Read_Float_At (Logits, I));
         if L > Max_L then Max_L := L; end if;
      end loop;

      -- Pass 2: accumulate sum of exp(logit - max)
      for I in 0 .. N - 1 loop
         Sum_Exp := Sum_Exp + Exp (Float (Read_Float_At (Logits, I)) - Max_L);
      end loop;

      if Sum_Exp <= Epsilon then
         -- Degenerate: uniform distribution over N classes
         return C_Float (Log (Float (N)) / Log_Base);
      end if;

      -- Pass 3: compute -Σ p·log(p) directly from softmax probabilities
      for I in 0 .. N - 1 loop
         P := Exp (Float (Read_Float_At (Logits, I)) - Max_L) / Sum_Exp;
         if P > Epsilon then
            Entropy := Entropy - P * (Log (P) / Log_Base);
         end if;
      end loop;

      return C_Float (Float'Max (Entropy, 0.0));

   exception
      when E : others =>
         Log_Error ("C_Calculate_Token_Entropy",
            "Ada exception: " & Exception_Message (E));
         return -1.0;
   end C_Calculate_Token_Entropy;

   -----------------------------------------------------------------------

   function C_Calculate_Token_Probability
      (Logits     : System.Address;
       Logits_Len : C_Int;
       Token_Id   : C_Int) return C_Float
   is
      use Ada.Numerics.Elementary_Functions;

      N          : constant Integer := Integer (Logits_Len);
      Max_L      : Float;
      Sum_Exp    : Float := 0.0;
      Target_Exp : Float;
      L          : Float;
   begin
      if Logits = System.Null_Address then
         Log_Error ("C_Calculate_Token_Probability", "Logits address is null");
         return -1.0;
      end if;

      if N <= 0 or N > Max_Vocab_Size then
         Log_Error ("C_Calculate_Token_Probability",
            "logits_len out of [1," & Natural'Image (Max_Vocab_Size) &
            "]: " & C_Int'Image (Logits_Len));
         return -1.0;
      end if;

      if Token_Id < 0 or Token_Id >= Logits_Len then
         Log_Error ("C_Calculate_Token_Probability",
            "Token_Id " & C_Int'Image (Token_Id) &
            " out of [0," & C_Int'Image (Logits_Len - 1) & "]");
         return -1.0;
      end if;

      -- Pass 1: find max logit
      Max_L := Float (Read_Float_At (Logits, 0));
      for I in 1 .. N - 1 loop
         L := Float (Read_Float_At (Logits, I));
         if L > Max_L then Max_L := L; end if;
      end loop;

      -- Pass 2: accumulate sum of exp
      for I in 0 .. N - 1 loop
         Sum_Exp := Sum_Exp + Exp (Float (Read_Float_At (Logits, I)) - Max_L);
      end loop;

      -- Extract probability for requested token (0-indexed)
      Target_Exp := Exp (Float (Read_Float_At (Logits, Natural (Token_Id))) - Max_L);

      if Sum_Exp <= Epsilon then
         return C_Float (1.0 / Float (N));  -- uniform fallback
      end if;

      return C_Float (Float'Min (Float'Max (Target_Exp / Sum_Exp, 0.0), 1.0));

   exception
      when E : others =>
         Log_Error ("C_Calculate_Token_Probability",
            "Ada exception: " & Exception_Message (E));
         return -1.0;
   end C_Calculate_Token_Probability;

   -----------------------------------------------------------------------
   -- ENTROPY CLASSIFICATION
   -----------------------------------------------------------------------

   function C_Classify_Entropy
      (Entropy        : C_Float;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float) return C_Int
   is
      Level : Entropy_Level;
   begin
      if Float (Threshold_Low) >= Float (Threshold_High) then
         Log_Error ("C_Classify_Entropy",
            "Threshold_Low must be < Threshold_High; got " &
            C_Float'Image (Threshold_Low) & " >= " &
            C_Float'Image (Threshold_High));
         return -1;
      end if;

      if Float (Entropy) < 0.0 then
         Log_Error ("C_Classify_Entropy",
            "Entropy must be >= 0.0, got:" & C_Float'Image (Entropy));
         return -1;
      end if;

      Level := Classify_Entropy
         (Entropy_Value (Float (Entropy)),
          Entropy_Value (Float (Threshold_Low)),
          Entropy_Value (Float (Threshold_High)));

      case Level is
         when Low    => return 0;
         when Medium => return 1;
         when High   => return 2;
      end case;

   exception
      when E : others =>
         Log_Error ("C_Classify_Entropy",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Classify_Entropy;

   -----------------------------------------------------------------------
   -- STEP ROUTING
   -----------------------------------------------------------------------

   function C_Route_Step
      (Avg_Entropy    : C_Float;
       Total_Entropy  : C_Float;
       Max_Entropy    : C_Float;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Enable_Memory  : C_Int;
       Out_Path       : System.Address;
       Out_Retrieve   : System.Address) return C_Int
   is
      Step     : Step_Entropy.Step_Entropy;
      Decision : Routing_Decision;

      --  Scalar output overlays (Pattern 2).
      --  Declare as local constants for the address, then overlay.
      Path_Addr     : constant System.Address := Out_Path;
      Retrieve_Addr : constant System.Address := Out_Retrieve;

      Path_Out : C_Int;
      for Path_Out'Address use Path_Addr;
      pragma Import (Ada, Path_Out);

      Retrieve_Out : C_Int;
      for Retrieve_Out'Address use Retrieve_Addr;
      pragma Import (Ada, Retrieve_Out);

   begin
      if Out_Path = System.Null_Address then
         Log_Error ("C_Route_Step", "Out_Path pointer is null");
         return -1;
      end if;

      if Out_Retrieve = System.Null_Address then
         Log_Error ("C_Route_Step", "Out_Retrieve pointer is null");
         return -1;
      end if;

      if Float (Avg_Entropy) < 0.0 then
         Log_Error ("C_Route_Step",
            "Avg_Entropy must be >= 0.0, got:" &
            C_Float'Image (Avg_Entropy));
         return -1;
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         Log_Error ("C_Route_Step",
            "Threshold_Low must be < Threshold_High; got " &
            C_Float'Image (Threshold_Low) & " >= " &
            C_Float'Image (Threshold_High));
         return -1;
      end if;

      --  Build a minimal Step_Entropy record from the scalar arguments.
      --  Token_Entropies and Step_Text are unused by Route_Step (it reads
      --  only Avg_Entropy, Total_Entropy, Max_Entropy, and Level).
      Step.Step_Id     := 0;
      Step.Token_Count := 1;
      Step.Step_Text   := (others => ' ');

      --  Clamp negative values that may arrive due to float imprecision.
      Step.Total_Entropy :=
         Entropy_Value (Float'Max (Float (Total_Entropy), 0.0));
      Step.Avg_Entropy   :=
         Entropy_Value (Float'Max (Float (Avg_Entropy), 0.0));
      Step.Max_Entropy   :=
         Entropy_Value (Float'Max (Float (Max_Entropy), 0.0));

      --  Classify level so Route_Step has a consistent record.
      if Step.Avg_Entropy < Entropy_Value (Float (Threshold_Low)) then
         Step.Level := Low;
      elsif Step.Avg_Entropy > Entropy_Value (Float (Threshold_High)) then
         Step.Level := High;
      else
         Step.Level := Medium;
      end if;

      Decision := Route_Step (Step, Enable_Memory /= 0);

      case Decision.Path is
         when Fast   => Path_Out := 0;
         when Normal => Path_Out := 1;
         when Slow   => Path_Out := 2;
      end case;

      Retrieve_Out := (if Decision.Retrieve_Memory then 1 else 0);

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Route_Step",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Route_Step;

   -----------------------------------------------------------------------
   -- BATCH ROUTING
   -----------------------------------------------------------------------

   function C_Batch_Route
      (Step_Entropies : System.Address;
       Count          : C_Int;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Out_Decisions  : System.Address) return C_Int
   is
      Level   : Entropy_Level;
      Ent_Val : Float;
   begin
      if Step_Entropies = System.Null_Address then
         Log_Error ("C_Batch_Route", "Step_Entropies pointer is null");
         return -1;
      end if;

      if Out_Decisions = System.Null_Address then
         Log_Error ("C_Batch_Route", "Out_Decisions pointer is null");
         return -1;
      end if;

      if Count <= 0 then
         Log_Error ("C_Batch_Route",
            "Count must be positive, got:" & C_Int'Image (Count));
         return -1;
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         Log_Error ("C_Batch_Route",
            "Threshold_Low must be < Threshold_High; got " &
            C_Float'Image (Threshold_Low) & " >= " &
            C_Float'Image (Threshold_High));
         return -1;
      end if;

      for I in 0 .. Integer (Count) - 1 loop
         Ent_Val := Float (Read_Float_At (Step_Entropies, I));

         --  Clamp negative entropy values from noisy C callers.
         if Ent_Val < 0.0 then
            Ent_Val := 0.0;
         end if;

         Level := Classify_Entropy
            (Entropy_Value (Ent_Val),
             Entropy_Value (Float (Threshold_Low)),
             Entropy_Value (Float (Threshold_High)));

         case Level is
            when Low    => Write_Int_At (Out_Decisions, I, 0);
            when Medium => Write_Int_At (Out_Decisions, I, 1);
            when High   => Write_Int_At (Out_Decisions, I, 2);
         end case;
      end loop;

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Batch_Route",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Batch_Route;

   -----------------------------------------------------------------------
   -- COMPRESSION ANALYSIS
   -----------------------------------------------------------------------

   function C_Analyze_Chain
      (Step_Entropies : System.Address;
       Step_Count     : C_Int;
       Threshold_Low  : C_Float;
       Threshold_High : C_Float;
       Out_Buffer     : System.Address) return C_Int
   is
      Compressible_Count : Natural := 0;
      Informative_Count  : Natural := 0;
      Low_Count          : Natural := 0;
      Medium_Count       : Natural := 0;
      High_Count         : Natural := 0;
      Entropy_Sum        : Float   := 0.0;
      Level              : Entropy_Level;
      Ent_Val            : Float;
      N                  : constant Integer := Integer (Step_Count);
   begin
      if Step_Entropies = System.Null_Address then
         Log_Error ("C_Analyze_Chain", "Step_Entropies pointer is null");
         return -1;
      end if;

      if Out_Buffer = System.Null_Address then
         Log_Error ("C_Analyze_Chain", "Out_Buffer pointer is null");
         return -1;
      end if;

      if Step_Count <= 0 then
         Log_Error ("C_Analyze_Chain",
            "Step_Count must be positive, got:" &
            C_Int'Image (Step_Count));
         return -1;
      end if;

      if Float (Threshold_Low) >= Float (Threshold_High) then
         Log_Error ("C_Analyze_Chain",
            "Threshold_Low must be < Threshold_High; got " &
            C_Float'Image (Threshold_Low) & " >= " &
            C_Float'Image (Threshold_High));
         return -1;
      end if;

      for I in 0 .. N - 1 loop
         Ent_Val := Float (Read_Float_At (Step_Entropies, I));

         if Ent_Val < 0.0 then
            Ent_Val := 0.0;
         end if;

         Level := Classify_Entropy
            (Entropy_Value (Ent_Val),
             Entropy_Value (Float (Threshold_Low)),
             Entropy_Value (Float (Threshold_High)));

         --  Compressible = below low threshold (low entropy = redundant)
         if Ent_Val < Float (Threshold_Low) then
            Compressible_Count := Compressible_Count + 1;
         end if;

         --  Informative = above high threshold (high entropy = uncertain)
         if Ent_Val > Float (Threshold_High) then
            Informative_Count := Informative_Count + 1;
         end if;

         case Level is
            when Low    => Low_Count    := Low_Count    + 1;
            when Medium => Medium_Count := Medium_Count + 1;
            when High   => High_Count   := High_Count   + 1;
         end case;

         Entropy_Sum := Entropy_Sum + Ent_Val;
      end loop;

      --  Pack 8-float output buffer:
      --    [0] step_count       [1] compressible     [2] informative
      --    [3] low_count        [4] medium_count      [5] high_count
      --    [6] compression_ratio                      [7] avg_entropy
      Write_Float_At (Out_Buffer, 0, C_Float (Float (N)));
      Write_Float_At (Out_Buffer, 1, C_Float (Float (Compressible_Count)));
      Write_Float_At (Out_Buffer, 2, C_Float (Float (Informative_Count)));
      Write_Float_At (Out_Buffer, 3, C_Float (Float (Low_Count)));
      Write_Float_At (Out_Buffer, 4, C_Float (Float (Medium_Count)));
      Write_Float_At (Out_Buffer, 5, C_Float (Float (High_Count)));
      Write_Float_At (Out_Buffer, 6,
         C_Float (Float (Compressible_Count) / Float (N)));
      Write_Float_At (Out_Buffer, 7,
         C_Float (Entropy_Sum / Float (N)));

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Analyze_Chain",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Analyze_Chain;

   -----------------------------------------------------------------------
   -- PRUNING RECOMMENDATIONS
   -----------------------------------------------------------------------

   function C_Recommend_Pruning
      (Step_Entropies : System.Address;
       Step_Count     : C_Int;
       Pruning_Ratio  : C_Float;
       Out_Indices    : System.Address;
       Out_Count      : System.Address) return C_Int
   is
      --  Max_Tokens_Per_Step (512) is the hard bound for Step_Count.
      --  Values beyond this are rejected so the stack arrays stay bounded.
      Max_Steps : constant Integer := Max_Tokens_Per_Step;

      type Indexed_Entry is record
         Original_Index : Integer;
         Entropy        : Float;
      end record;

      --  Fixed-size stack arrays — no heap allocation.
      type Indexed_Array is array (0 .. Max_Steps - 1) of Indexed_Entry;
      type Result_Index_Array is array (0 .. Max_Steps - 1) of Integer;

      Indexed        : Indexed_Array;
      Result_Indices : Result_Index_Array;
      N              : Integer;
      Num_To_Prune   : Integer;

      --  Scalar overlay for Out_Count write.
      Count_Out_Addr : constant System.Address := Out_Count;
      Count_Out : C_Int;
      for Count_Out'Address use Count_Out_Addr;
      pragma Import (Ada, Count_Out);

      --  Insertion sort: Indexed(0..N-1) ascending by Entropy.
      procedure Sort_By_Entropy_Ascending is
         Temp : Indexed_Entry;
         J    : Integer;
      begin
         for I in 1 .. N - 1 loop
            Temp := Indexed (I);
            J    := I - 1;
            while J >= 0 and then Indexed (J).Entropy > Temp.Entropy loop
               Indexed (J + 1) := Indexed (J);
               J := J - 1;
            end loop;
            Indexed (J + 1) := Temp;
         end loop;
      end Sort_By_Entropy_Ascending;

      --  Insertion sort: Result_Indices(0..M-1) ascending by value.
      procedure Sort_Indices_Ascending (M : Integer) is
         Temp : Integer;
         J    : Integer;
      begin
         for I in 1 .. M - 1 loop
            Temp := Result_Indices (I);
            J    := I - 1;
            while J >= 0 and then Result_Indices (J) > Temp loop
               Result_Indices (J + 1) := Result_Indices (J);
               J := J - 1;
            end loop;
            Result_Indices (J + 1) := Temp;
         end loop;
      end Sort_Indices_Ascending;

   begin
      if Step_Entropies = System.Null_Address then
         Log_Error ("C_Recommend_Pruning",
            "Step_Entropies pointer is null");
         return -1;
      end if;

      if Out_Indices = System.Null_Address then
         Log_Error ("C_Recommend_Pruning", "Out_Indices pointer is null");
         return -1;
      end if;

      if Out_Count = System.Null_Address then
         Log_Error ("C_Recommend_Pruning", "Out_Count pointer is null");
         return -1;
      end if;

      if Step_Count <= 0 then
         Log_Error ("C_Recommend_Pruning",
            "Step_Count must be positive, got:" &
            C_Int'Image (Step_Count));
         return -1;
      end if;

      if Integer (Step_Count) > Max_Steps then
         Log_Error ("C_Recommend_Pruning",
            "Step_Count" & C_Int'Image (Step_Count) &
            " exceeds maximum" & Integer'Image (Max_Steps));
         return -1;
      end if;

      if Float (Pruning_Ratio) < 0.0 or Float (Pruning_Ratio) > 1.0 then
         Log_Error ("C_Recommend_Pruning",
            "Pruning_Ratio must be in [0.0, 1.0], got:" &
            C_Float'Image (Pruning_Ratio));
         return -1;
      end if;

      N := Integer (Step_Count);

      --  Build indexed array from C memory.
      for I in 0 .. N - 1 loop
         declare
            Ent : Float := Float (Read_Float_At (Step_Entropies, I));
         begin
            if Ent < 0.0 then
               Ent := 0.0;
            end if;
            Indexed (I) := (Original_Index => I, Entropy => Ent);
         end;
      end loop;

      --  Sort ascending by entropy — lowest entropy = most compressible.
      Sort_By_Entropy_Ascending;

      --  Select bottom floor(N * Pruning_Ratio) indices.
      Num_To_Prune := Integer (Float (N) * Float (Pruning_Ratio));

      --  Collect the indices from the sorted list, then re-sort by
      --  original position for a predictable, ascending output order.
      for I in 0 .. Num_To_Prune - 1 loop
         Result_Indices (I) := Indexed (I).Original_Index;
      end loop;

      if Num_To_Prune > 1 then
         Sort_Indices_Ascending (Num_To_Prune);
      end if;

      --  Write prune indices to C output buffer.
      for I in 0 .. Num_To_Prune - 1 loop
         Write_Int_At (Out_Indices, I, C_Int (Result_Indices (I)));
      end loop;

      --  Write count scalar directly through the overlay.
      Count_Out := C_Int (Num_To_Prune);

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Recommend_Pruning",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Recommend_Pruning;

   -----------------------------------------------------------------------
   -- LIBRARY INTROSPECTION
   -----------------------------------------------------------------------

   function C_Get_Version return C_Int is
   begin
      --  v2.0.0 = 2*10000 + 0*100 + 0
      return 20000;
   end C_Get_Version;

   function C_Get_Max_Vocab_Size return C_Int is
   begin
      return C_Int (Max_Vocab_Size);
   end C_Get_Max_Vocab_Size;

   function C_Get_Max_Tokens_Per_Step return C_Int is
   begin
      return C_Int (Max_Tokens_Per_Step);
   end C_Get_Max_Tokens_Per_Step;

   -----------------------------------------------------------------------
   -- SOTA+++: ENTROPY HISTOGRAM
   -----------------------------------------------------------------------

   function C_Compute_Histogram
      (Entropies  : System.Address;
       Count      : C_Int;
       Bin_Count  : C_Int;
       Out_Buffer : System.Address) return C_Int
   is
      Ada_Entropies : Entropy_Array;
      Histogram     : Entropy_Histogram;
   begin
      if Entropies = System.Null_Address then
         Log_Error ("C_Compute_Histogram", "Entropies is null");
         return -1;
      end if;

      if Out_Buffer = System.Null_Address then
         Log_Error ("C_Compute_Histogram", "Out_Buffer is null");
         return -1;
      end if;

      if Count <= 0 or Count > C_Int (Max_Tokens_Per_Step) then
         Log_Error ("C_Compute_Histogram",
            "Count out of [1," & Natural'Image (Max_Tokens_Per_Step) &
            "], got:" & C_Int'Image (Count));
         return -1;
      end if;

      if Bin_Count <= 0 or Bin_Count > C_Int (Max_Histogram_Bins) then
         Log_Error ("C_Compute_Histogram",
            "Bin_Count out of [1," & Natural'Image (Max_Histogram_Bins) &
            "], got:" & C_Int'Image (Bin_Count));
         return -1;
      end if;

      -- Build Ada Entropy_Array from C float pointer
      Ada_Entropies := (others => Entropy_Value (0.0));
      for I in 0 .. Integer (Count) - 1 loop
         declare
            Val : Float := Float (Read_Float_At (Entropies, I));
         begin
            if Val < 0.0 then Val := 0.0; end if;
            Ada_Entropies (I + 1) := Entropy_Value (Val);
         end;
      end loop;

      Histogram := Compute_Histogram (
         Entropies => Ada_Entropies,
         Count     => Positive (Count),
         Bin_Count => Positive (Bin_Count));

      -- Out_Buffer layout: [bin_width, min, max, mean, std, bin_0..bin_{N-1}]
      -- Total floats = 5 + Bin_Count
      Write_Float_At (Out_Buffer, 0, C_Float (Histogram.Bin_Width));
      Write_Float_At (Out_Buffer, 1, C_Float (Histogram.Min_Entropy));
      Write_Float_At (Out_Buffer, 2, C_Float (Histogram.Max_Entropy));
      Write_Float_At (Out_Buffer, 3, C_Float (Histogram.Mean_Entropy));
      Write_Float_At (Out_Buffer, 4, C_Float (Histogram.Std_Dev));

      for I in 1 .. Integer (Bin_Count) loop
         Write_Float_At (Out_Buffer, 4 + I,
            C_Float (Float (Histogram.Bins (I))));
      end loop;

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Compute_Histogram",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Compute_Histogram;

   -----------------------------------------------------------------------
   -- SOTA+++: Z-SCORE NORMALIZATION
   -----------------------------------------------------------------------

   function C_Normalize_Entropy
      (Entropy : C_Float;
       Mean    : C_Float;
       Std_Dev : C_Float) return C_Float
   is
   begin
      if Float (Entropy) < 0.0 then
         Log_Error ("C_Normalize_Entropy",
            "Entropy must be >= 0, got:" & C_Float'Image (Entropy));
         return -999.0;
      end if;

      if Float (Std_Dev) < 0.0 then
         Log_Error ("C_Normalize_Entropy",
            "Std_Dev must be >= 0, got:" & C_Float'Image (Std_Dev));
         return -999.0;
      end if;

      return C_Float (Normalize_Entropy (
         Entropy => Entropy_Value (Float (Entropy)),
         Mean    => Float (Mean),
         Std_Dev => Float (Std_Dev)));

   exception
      when E : others =>
         Log_Error ("C_Normalize_Entropy",
            "Ada exception: " & Exception_Message (E));
         return -999.0;
   end C_Normalize_Entropy;

   -----------------------------------------------------------------------
   -- SOTA+++: GRPO REWARD FUNCTION (Paper Eq.13-15)
   -----------------------------------------------------------------------

   function C_Compute_GRPO_Rewards
      (Skip_Ratio      : C_Float;
       Skip_Count      : C_Int;
       Response_Length : C_Int;
       K_High          : C_Float;
       K_Low           : C_Float;
       Max_Skip        : C_Int;
       Max_Length      : C_Int;
       Out_Rewards     : System.Address) return C_Int
   is
      Rewards : GRPO_Rewards;
   begin
      if Out_Rewards = System.Null_Address then
         Log_Error ("C_Compute_GRPO_Rewards", "Out_Rewards is null");
         return -1;
      end if;

      if Float (Skip_Ratio) < 0.0 or Float (Skip_Ratio) > 1.0 then
         Log_Error ("C_Compute_GRPO_Rewards",
            "Skip_Ratio out of [0,1]:" & C_Float'Image (Skip_Ratio));
         return -1;
      end if;

      if Float (K_Low) >= Float (K_High) then
         Log_Error ("C_Compute_GRPO_Rewards",
            "K_Low must be < K_High; got" & C_Float'Image (K_Low) &
            " >= " & C_Float'Image (K_High));
         return -1;
      end if;

      Rewards := Compute_GRPO_Rewards (
         Skip_Ratio   => Float (Skip_Ratio),
         Skip_Count   => Natural (Skip_Count),
         Response_Len => Natural (Response_Length),
         K_High       => Float (K_High),
         K_Low        => Float (K_Low),
         Max_Skip     => Natural (Max_Skip),
         Max_Length   => Natural (Max_Length));

      -- Out_Rewards[0..3] = [r_skip_ratio, r_skip_num, r_response_len, r_total]
      Write_Float_At (Out_Rewards, 0, C_Float (Rewards.R_Skip_Ratio));
      Write_Float_At (Out_Rewards, 1, C_Float (Rewards.R_Skip_Num));
      Write_Float_At (Out_Rewards, 2, C_Float (Rewards.R_Response_Len));
      Write_Float_At (Out_Rewards, 3, C_Float (Rewards.R_Total));

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Compute_GRPO_Rewards",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Compute_GRPO_Rewards;

   -----------------------------------------------------------------------
   -- SOTA+++: ADAPTIVE THRESHOLD STATE
   --
   -- State_Buf layout: float[5]
   --   [0] running_mean
   --   [1] running_m2
   --   [2] sample_count (as float for uniform buffer type)
   --   [3] low_sigma_factor   (threshold_low  = mean + factor * std)
   --   [4] high_sigma_factor  (threshold_high = mean + factor * std)
   -----------------------------------------------------------------------

   function C_Update_Adaptive_State
      (Entropy   : C_Float;
       State_Buf : System.Address) return C_Int
   is
      State : Threshold_State;
   begin
      if State_Buf = System.Null_Address then
         Log_Error ("C_Update_Adaptive_State", "State_Buf is null");
         return -1;
      end if;

      if Float (Entropy) < 0.0 then
         Log_Error ("C_Update_Adaptive_State",
            "Entropy must be >= 0, got:" & C_Float'Image (Entropy));
         return -1;
      end if;

      -- Deserialize state from buffer
      State.Running_Mean      := Float (Read_Float_At (State_Buf, 0));
      State.Running_M2        := Float (Read_Float_At (State_Buf, 1));
      State.Sample_Count      := Natural (Float (Read_Float_At (State_Buf, 2)));
      State.Low_Sigma_Factor  := Float (Read_Float_At (State_Buf, 3));
      State.High_Sigma_Factor := Float (Read_Float_At (State_Buf, 4));

      if State.Low_Sigma_Factor >= State.High_Sigma_Factor then
         Log_Error ("C_Update_Adaptive_State",
            "State_Buf sigma factors invalid: [3] must be < [4]");
         return -1;
      end if;

      -- Welford online update
      Update_Threshold_State (State, Entropy_Value (Float (Entropy)));

      -- Serialize updated state back to buffer
      Write_Float_At (State_Buf, 0, C_Float (State.Running_Mean));
      Write_Float_At (State_Buf, 1, C_Float (State.Running_M2));
      Write_Float_At (State_Buf, 2, C_Float (Float (State.Sample_Count)));
      Write_Float_At (State_Buf, 3, C_Float (State.Low_Sigma_Factor));
      Write_Float_At (State_Buf, 4, C_Float (State.High_Sigma_Factor));

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Update_Adaptive_State",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Update_Adaptive_State;

   -----------------------------------------------------------------------

   function C_Get_Adaptive_Thresholds
      (State_Buf         : System.Address;
       Out_Threshold_Low : System.Address;
       Out_Threshold_High: System.Address) return C_Int
   is
      State    : Threshold_State;
      Low_Val  : Entropy_Value;
      High_Val : Entropy_Value;
   begin
      if State_Buf = System.Null_Address then
         Log_Error ("C_Get_Adaptive_Thresholds", "State_Buf is null");
         return -1;
      end if;

      if Out_Threshold_Low = System.Null_Address then
         Log_Error ("C_Get_Adaptive_Thresholds", "Out_Threshold_Low is null");
         return -1;
      end if;

      if Out_Threshold_High = System.Null_Address then
         Log_Error ("C_Get_Adaptive_Thresholds", "Out_Threshold_High is null");
         return -1;
      end if;

      -- Deserialize state
      State.Running_Mean      := Float (Read_Float_At (State_Buf, 0));
      State.Running_M2        := Float (Read_Float_At (State_Buf, 1));
      State.Sample_Count      := Natural (Float (Read_Float_At (State_Buf, 2)));
      State.Low_Sigma_Factor  := Float (Read_Float_At (State_Buf, 3));
      State.High_Sigma_Factor := Float (Read_Float_At (State_Buf, 4));

      if State.Sample_Count < 2 then
         Log_Error ("C_Get_Adaptive_Thresholds",
            "Need >= 2 samples; have" & Natural'Image (State.Sample_Count));
         return -2;   -- distinct code: caller should use defaults
      end if;

      Get_Adaptive_Thresholds (State, Low_Val, High_Val);

      -- Write outputs via address overlay (Pattern B)
      declare
         LV : C_Float;
         for LV'Address use Out_Threshold_Low;
         pragma Import (Ada, LV);
      begin
         LV := C_Float (Float (Low_Val));
      end;

      declare
         HV : C_Float;
         for HV'Address use Out_Threshold_High;
         pragma Import (Ada, HV);
      begin
         HV := C_Float (Float (High_Val));
      end;

      return 0;

   exception
      when E : others =>
         Log_Error ("C_Get_Adaptive_Thresholds",
            "Ada exception: " & Exception_Message (E));
         return -1;
   end C_Get_Adaptive_Thresholds;

end Step_Entropy_C_API;
