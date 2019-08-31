\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.001 L[4]
   + 0.001 L[5] + 0.002 R[0] + 0.002 R[1] + 0.002 R[2] + 0.002 R[3]
   + 0.002 R[4] + 0.002 R[5] + 0.001 T[0] + 0.001 T[1] + 0.001 T[2]
   + 0.001 T[3] + 0.001 T[4] + 0.001 T[5] + 0.002 B[0] + 0.002 B[1]
   + 0.002 B[2] + 0.002 B[3] + 0.002 B[4] + 0.002 B[5] - 0.001 H[0]
   - 0.001 H[1] - 0.001 H[2] - 0.001 H[3] - 0.001 H[4] - 0.001 H[5]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] - 0.001 W[4]
   - 0.001 W[5] + 0 ABOVE[0,0] + 0 ABOVE[1,1] + 0 ABOVE[2,2] + 0 ABOVE[3,3]
   + 0 ABOVE[4,4] + 0 ABOVE[5,5] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2]
   + 0 LEFT[3,3] + 0 LEFT[4,4] + 0 LEFT[5,5] + 2 LAG[0] + 2 LAG[1]
   + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + 2 LAG[5] + RAG[0] + RAG[1] + RAG[2]
   + RAG[3] + RAG[4] + RAG[5] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3]
   + 2 TAG[4] + 2 TAG[5] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4]
   + BAG[5]
Subject To
 R0: - R[0] + maxX >= 0
 R1: - B[0] + maxY >= 0
 R2: - R[1] + maxX >= 0
 R3: - B[1] + maxY >= 0
 R4: - R[2] + maxX >= 0
 R5: - B[2] + maxY >= 0
 R6: - R[3] + maxX >= 0
 R7: - B[3] + maxY >= 0
 R8: - R[4] + maxX >= 0
 R9: - B[4] + maxY >= 0
 R10: - R[5] + maxX >= 0
 R11: - B[5] + maxY >= 0
 R12: 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + 2 LAG[5]
   + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + 2 TAG[0]
   + 2 TAG[1] + 2 TAG[2] + 2 TAG[3] + 2 TAG[4] + 2 TAG[5] + BAG[0] + BAG[1]
   + BAG[2] + BAG[3] + BAG[4] + BAG[5] >= 10
 PrespecifiedXOfElement(0): L[0] = 0
 PrespecifiedYOfElement(0): T[0] = 0
 PrespecifiedXOfElement(3): L[3] = 128
 PrespecifiedYOfElement(3): T[3] = 64
 PrespecifiedXOfElement(4): L[4] = 0
 DisableNon-PermittedBasedOnPrespecified: = 0
 R-L=W(0): L[0] - R[0] + W[0] = 0
 B-T=H(0): T[0] - B[0] + H[0] = 0
 R-L=W(1): L[1] - R[1] + W[1] = 0
 B-T=H(1): T[1] - B[1] + H[1] = 0
 R-L=W(2): L[2] - R[2] + W[2] = 0
 B-T=H(2): T[2] - B[2] + H[2] = 0
 R-L=W(3): L[3] - R[3] + W[3] = 0
 B-T=H(3): T[3] - B[3] + H[3] = 0
 R-L=W(4): L[4] - R[4] + W[4] = 0
 B-T=H(4): T[4] - B[4] + H[4] = 0
 R-L=W(5): L[5] - R[5] + W[5] = 0
 B-T=H(5): T[5] - B[5] + H[5] = 0
 NoOverlap(01): ABOVE[0,1] + ABOVE[1,0] + LEFT[0,1] + LEFT[1,0] >= 1
 UpperLimOfQuadrants(01): ABOVE[0,1] + ABOVE[1,0] + LEFT[0,1] + LEFT[1,0]
   <= 2
 Anti-symmetryABOVE(01): ABOVE[0,1] + ABOVE[1,0] <= 1
 Anti-symmetryLEFT(01): LEFT[0,1] + LEFT[1,0] <= 1
 NoOverlap(02): ABOVE[0,2] + ABOVE[2,0] + LEFT[0,2] + LEFT[2,0] >= 1
 UpperLimOfQuadrants(02): ABOVE[0,2] + ABOVE[2,0] + LEFT[0,2] + LEFT[2,0]
   <= 2
 Anti-symmetryABOVE(02): ABOVE[0,2] + ABOVE[2,0] <= 1
 Anti-symmetryLEFT(02): LEFT[0,2] + LEFT[2,0] <= 1
 NoOverlap(03): ABOVE[0,3] + ABOVE[3,0] + LEFT[0,3] + LEFT[3,0] >= 1
 UpperLimOfQuadrants(03): ABOVE[0,3] + ABOVE[3,0] + LEFT[0,3] + LEFT[3,0]
   <= 2
 Anti-symmetryABOVE(03): ABOVE[0,3] + ABOVE[3,0] <= 1
 Anti-symmetryLEFT(03): LEFT[0,3] + LEFT[3,0] <= 1
 NoOverlap(04): ABOVE[0,4] + ABOVE[4,0] + LEFT[0,4] + LEFT[4,0] >= 1
 UpperLimOfQuadrants(04): ABOVE[0,4] + ABOVE[4,0] + LEFT[0,4] + LEFT[4,0]
   <= 2
 Anti-symmetryABOVE(04): ABOVE[0,4] + ABOVE[4,0] <= 1
 Anti-symmetryLEFT(04): LEFT[0,4] + LEFT[4,0] <= 1
 NoOverlap(05): ABOVE[0,5] + ABOVE[5,0] + LEFT[0,5] + LEFT[5,0] >= 1
 UpperLimOfQuadrants(05): ABOVE[0,5] + ABOVE[5,0] + LEFT[0,5] + LEFT[5,0]
   <= 2
 Anti-symmetryABOVE(05): ABOVE[0,5] + ABOVE[5,0] <= 1
 Anti-symmetryLEFT(05): LEFT[0,5] + LEFT[5,0] <= 1
 NoOverlap(12): ABOVE[1,2] + ABOVE[2,1] + LEFT[1,2] + LEFT[2,1] >= 1
 UpperLimOfQuadrants(12): ABOVE[1,2] + ABOVE[2,1] + LEFT[1,2] + LEFT[2,1]
   <= 2
 Anti-symmetryABOVE(12): ABOVE[1,2] + ABOVE[2,1] <= 1
 Anti-symmetryLEFT(12): LEFT[1,2] + LEFT[2,1] <= 1
 NoOverlap(13): ABOVE[1,3] + ABOVE[3,1] + LEFT[1,3] + LEFT[3,1] >= 1
 UpperLimOfQuadrants(13): ABOVE[1,3] + ABOVE[3,1] + LEFT[1,3] + LEFT[3,1]
   <= 2
 Anti-symmetryABOVE(13): ABOVE[1,3] + ABOVE[3,1] <= 1
 Anti-symmetryLEFT(13): LEFT[1,3] + LEFT[3,1] <= 1
 NoOverlap(14): ABOVE[1,4] + ABOVE[4,1] + LEFT[1,4] + LEFT[4,1] >= 1
 UpperLimOfQuadrants(14): ABOVE[1,4] + ABOVE[4,1] + LEFT[1,4] + LEFT[4,1]
   <= 2
 Anti-symmetryABOVE(14): ABOVE[1,4] + ABOVE[4,1] <= 1
 Anti-symmetryLEFT(14): LEFT[1,4] + LEFT[4,1] <= 1
 NoOverlap(15): ABOVE[1,5] + ABOVE[5,1] + LEFT[1,5] + LEFT[5,1] >= 1
 UpperLimOfQuadrants(15): ABOVE[1,5] + ABOVE[5,1] + LEFT[1,5] + LEFT[5,1]
   <= 2
 Anti-symmetryABOVE(15): ABOVE[1,5] + ABOVE[5,1] <= 1
 Anti-symmetryLEFT(15): LEFT[1,5] + LEFT[5,1] <= 1
 NoOverlap(23): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2] >= 1
 UpperLimOfQuadrants(23): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2]
   <= 2
 Anti-symmetryABOVE(23): ABOVE[2,3] + ABOVE[3,2] <= 1
 Anti-symmetryLEFT(23): LEFT[2,3] + LEFT[3,2] <= 1
 NoOverlap(24): ABOVE[2,4] + ABOVE[4,2] + LEFT[2,4] + LEFT[4,2] >= 1
 UpperLimOfQuadrants(24): ABOVE[2,4] + ABOVE[4,2] + LEFT[2,4] + LEFT[4,2]
   <= 2
 Anti-symmetryABOVE(24): ABOVE[2,4] + ABOVE[4,2] <= 1
 Anti-symmetryLEFT(24): LEFT[2,4] + LEFT[4,2] <= 1
 NoOverlap(25): ABOVE[2,5] + ABOVE[5,2] + LEFT[2,5] + LEFT[5,2] >= 1
 UpperLimOfQuadrants(25): ABOVE[2,5] + ABOVE[5,2] + LEFT[2,5] + LEFT[5,2]
   <= 2
 Anti-symmetryABOVE(25): ABOVE[2,5] + ABOVE[5,2] <= 1
 Anti-symmetryLEFT(25): LEFT[2,5] + LEFT[5,2] <= 1
 NoOverlap(34): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3] >= 1
 UpperLimOfQuadrants(34): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3]
   <= 2
 Anti-symmetryABOVE(34): ABOVE[3,4] + ABOVE[4,3] <= 1
 Anti-symmetryLEFT(34): LEFT[3,4] + LEFT[4,3] <= 1
 NoOverlap(35): ABOVE[3,5] + ABOVE[5,3] + LEFT[3,5] + LEFT[5,3] >= 1
 UpperLimOfQuadrants(35): ABOVE[3,5] + ABOVE[5,3] + LEFT[3,5] + LEFT[5,3]
   <= 2
 Anti-symmetryABOVE(35): ABOVE[3,5] + ABOVE[5,3] <= 1
 Anti-symmetryLEFT(35): LEFT[3,5] + LEFT[5,3] <= 1
 NoOverlap(45): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4] >= 1
 UpperLimOfQuadrants(45): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4]
   <= 2
 Anti-symmetryABOVE(45): ABOVE[4,5] + ABOVE[5,4] <= 1
 Anti-symmetryLEFT(45): LEFT[4,5] + LEFT[5,4] <= 1
 0(ToLeftOf)1: - L[1] + R[0] + 1048 LEFT[0,1] <= 1016
 0(Above)1: - T[1] + B[0] + 933 ABOVE[0,1] <= 901
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 1048 LEFT[0,1] <= 32
 0(ConverseOfAboveOf)1: T[1] - B[0] - 933 ABOVE[0,1] <= 32
 0(ToLeftOf)2: - L[2] + R[0] + 1048 LEFT[0,2] <= 1016
 0(Above)2: - T[2] + B[0] + 933 ABOVE[0,2] <= 901
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 1048 LEFT[0,2] <= 32
 0(ConverseOfAboveOf)2: T[2] - B[0] - 933 ABOVE[0,2] <= 32
 0(ToLeftOf)3: - L[3] + R[0] + 1048 LEFT[0,3] <= 1016
 0(Above)3: - T[3] + B[0] + 933 ABOVE[0,3] <= 901
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 1048 LEFT[0,3] <= 32
 0(ConverseOfAboveOf)3: T[3] - B[0] - 933 ABOVE[0,3] <= 32
 0(ToLeftOf)4: - L[4] + R[0] + 1048 LEFT[0,4] <= 1016
 0(Above)4: - T[4] + B[0] + 933 ABOVE[0,4] <= 901
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 1048 LEFT[0,4] <= 32
 0(ConverseOfAboveOf)4: T[4] - B[0] - 933 ABOVE[0,4] <= 32
 0(ToLeftOf)5: - L[5] + R[0] + 1048 LEFT[0,5] <= 1016
 0(Above)5: - T[5] + B[0] + 933 ABOVE[0,5] <= 901
 0(ConverseOfToLeftOf)5: L[5] - R[0] - 1048 LEFT[0,5] <= 32
 0(ConverseOfAboveOf)5: T[5] - B[0] - 933 ABOVE[0,5] <= 32
 1(ToLeftOf)0: - L[0] + R[1] + 1048 LEFT[1,0] <= 1016
 1(Above)0: - T[0] + B[1] + 933 ABOVE[1,0] <= 901
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 1048 LEFT[1,0] <= 32
 1(ConverseOfAboveOf)0: T[0] - B[1] - 933 ABOVE[1,0] <= 32
 1(ToLeftOf)2: - L[2] + R[1] + 1048 LEFT[1,2] <= 1016
 1(Above)2: - T[2] + B[1] + 933 ABOVE[1,2] <= 901
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 1048 LEFT[1,2] <= 32
 1(ConverseOfAboveOf)2: T[2] - B[1] - 933 ABOVE[1,2] <= 32
 1(ToLeftOf)3: - L[3] + R[1] + 1048 LEFT[1,3] <= 1016
 1(Above)3: - T[3] + B[1] + 933 ABOVE[1,3] <= 901
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 1048 LEFT[1,3] <= 32
 1(ConverseOfAboveOf)3: T[3] - B[1] - 933 ABOVE[1,3] <= 32
 1(ToLeftOf)4: - L[4] + R[1] + 1048 LEFT[1,4] <= 1016
 1(Above)4: - T[4] + B[1] + 933 ABOVE[1,4] <= 901
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 1048 LEFT[1,4] <= 32
 1(ConverseOfAboveOf)4: T[4] - B[1] - 933 ABOVE[1,4] <= 32
 1(ToLeftOf)5: - L[5] + R[1] + 1048 LEFT[1,5] <= 1016
 1(Above)5: - T[5] + B[1] + 933 ABOVE[1,5] <= 901
 1(ConverseOfToLeftOf)5: L[5] - R[1] - 1048 LEFT[1,5] <= 32
 1(ConverseOfAboveOf)5: T[5] - B[1] - 933 ABOVE[1,5] <= 32
 2(ToLeftOf)0: - L[0] + R[2] + 1048 LEFT[2,0] <= 1016
 2(Above)0: - T[0] + B[2] + 933 ABOVE[2,0] <= 901
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 1048 LEFT[2,0] <= 32
 2(ConverseOfAboveOf)0: T[0] - B[2] - 933 ABOVE[2,0] <= 32
 2(ToLeftOf)1: - L[1] + R[2] + 1048 LEFT[2,1] <= 1016
 2(Above)1: - T[1] + B[2] + 933 ABOVE[2,1] <= 901
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 1048 LEFT[2,1] <= 32
 2(ConverseOfAboveOf)1: T[1] - B[2] - 933 ABOVE[2,1] <= 32
 2(ToLeftOf)3: - L[3] + R[2] + 1048 LEFT[2,3] <= 1016
 2(Above)3: - T[3] + B[2] + 933 ABOVE[2,3] <= 901
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 1048 LEFT[2,3] <= 32
 2(ConverseOfAboveOf)3: T[3] - B[2] - 933 ABOVE[2,3] <= 32
 2(ToLeftOf)4: - L[4] + R[2] + 1048 LEFT[2,4] <= 1016
 2(Above)4: - T[4] + B[2] + 933 ABOVE[2,4] <= 901
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 1048 LEFT[2,4] <= 32
 2(ConverseOfAboveOf)4: T[4] - B[2] - 933 ABOVE[2,4] <= 32
 2(ToLeftOf)5: - L[5] + R[2] + 1048 LEFT[2,5] <= 1016
 2(Above)5: - T[5] + B[2] + 933 ABOVE[2,5] <= 901
 2(ConverseOfToLeftOf)5: L[5] - R[2] - 1048 LEFT[2,5] <= 32
 2(ConverseOfAboveOf)5: T[5] - B[2] - 933 ABOVE[2,5] <= 32
 3(ToLeftOf)0: - L[0] + R[3] + 1048 LEFT[3,0] <= 1016
 3(Above)0: - T[0] + B[3] + 933 ABOVE[3,0] <= 901
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 1048 LEFT[3,0] <= 32
 3(ConverseOfAboveOf)0: T[0] - B[3] - 933 ABOVE[3,0] <= 32
 3(ToLeftOf)1: - L[1] + R[3] + 1048 LEFT[3,1] <= 1016
 3(Above)1: - T[1] + B[3] + 933 ABOVE[3,1] <= 901
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 1048 LEFT[3,1] <= 32
 3(ConverseOfAboveOf)1: T[1] - B[3] - 933 ABOVE[3,1] <= 32
 3(ToLeftOf)2: - L[2] + R[3] + 1048 LEFT[3,2] <= 1016
 3(Above)2: - T[2] + B[3] + 933 ABOVE[3,2] <= 901
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 1048 LEFT[3,2] <= 32
 3(ConverseOfAboveOf)2: T[2] - B[3] - 933 ABOVE[3,2] <= 32
 3(ToLeftOf)4: - L[4] + R[3] + 1048 LEFT[3,4] <= 1016
 3(Above)4: - T[4] + B[3] + 933 ABOVE[3,4] <= 901
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 1048 LEFT[3,4] <= 32
 3(ConverseOfAboveOf)4: T[4] - B[3] - 933 ABOVE[3,4] <= 32
 3(ToLeftOf)5: - L[5] + R[3] + 1048 LEFT[3,5] <= 1016
 3(Above)5: - T[5] + B[3] + 933 ABOVE[3,5] <= 901
 3(ConverseOfToLeftOf)5: L[5] - R[3] - 1048 LEFT[3,5] <= 32
 3(ConverseOfAboveOf)5: T[5] - B[3] - 933 ABOVE[3,5] <= 32
 4(ToLeftOf)0: - L[0] + R[4] + 1048 LEFT[4,0] <= 1016
 4(Above)0: - T[0] + B[4] + 933 ABOVE[4,0] <= 901
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 1048 LEFT[4,0] <= 32
 4(ConverseOfAboveOf)0: T[0] - B[4] - 933 ABOVE[4,0] <= 32
 4(ToLeftOf)1: - L[1] + R[4] + 1048 LEFT[4,1] <= 1016
 4(Above)1: - T[1] + B[4] + 933 ABOVE[4,1] <= 901
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 1048 LEFT[4,1] <= 32
 4(ConverseOfAboveOf)1: T[1] - B[4] - 933 ABOVE[4,1] <= 32
 4(ToLeftOf)2: - L[2] + R[4] + 1048 LEFT[4,2] <= 1016
 4(Above)2: - T[2] + B[4] + 933 ABOVE[4,2] <= 901
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 1048 LEFT[4,2] <= 32
 4(ConverseOfAboveOf)2: T[2] - B[4] - 933 ABOVE[4,2] <= 32
 4(ToLeftOf)3: - L[3] + R[4] + 1048 LEFT[4,3] <= 1016
 4(Above)3: - T[3] + B[4] + 933 ABOVE[4,3] <= 901
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 1048 LEFT[4,3] <= 32
 4(ConverseOfAboveOf)3: T[3] - B[4] - 933 ABOVE[4,3] <= 32
 4(ToLeftOf)5: - L[5] + R[4] + 1048 LEFT[4,5] <= 1016
 4(Above)5: - T[5] + B[4] + 933 ABOVE[4,5] <= 901
 4(ConverseOfToLeftOf)5: L[5] - R[4] - 1048 LEFT[4,5] <= 32
 4(ConverseOfAboveOf)5: T[5] - B[4] - 933 ABOVE[4,5] <= 32
 5(ToLeftOf)0: - L[0] + R[5] + 1048 LEFT[5,0] <= 1016
 5(Above)0: - T[0] + B[5] + 933 ABOVE[5,0] <= 901
 5(ConverseOfToLeftOf)0: L[0] - R[5] - 1048 LEFT[5,0] <= 32
 5(ConverseOfAboveOf)0: T[0] - B[5] - 933 ABOVE[5,0] <= 32
 5(ToLeftOf)1: - L[1] + R[5] + 1048 LEFT[5,1] <= 1016
 5(Above)1: - T[1] + B[5] + 933 ABOVE[5,1] <= 901
 5(ConverseOfToLeftOf)1: L[1] - R[5] - 1048 LEFT[5,1] <= 32
 5(ConverseOfAboveOf)1: T[1] - B[5] - 933 ABOVE[5,1] <= 32
 5(ToLeftOf)2: - L[2] + R[5] + 1048 LEFT[5,2] <= 1016
 5(Above)2: - T[2] + B[5] + 933 ABOVE[5,2] <= 901
 5(ConverseOfToLeftOf)2: L[2] - R[5] - 1048 LEFT[5,2] <= 32
 5(ConverseOfAboveOf)2: T[2] - B[5] - 933 ABOVE[5,2] <= 32
 5(ToLeftOf)3: - L[3] + R[5] + 1048 LEFT[5,3] <= 1016
 5(Above)3: - T[3] + B[5] + 933 ABOVE[5,3] <= 901
 5(ConverseOfToLeftOf)3: L[3] - R[5] - 1048 LEFT[5,3] <= 32
 5(ConverseOfAboveOf)3: T[3] - B[5] - 933 ABOVE[5,3] <= 32
 5(ToLeftOf)4: - L[4] + R[5] + 1048 LEFT[5,4] <= 1016
 5(Above)4: - T[4] + B[5] + 933 ABOVE[5,4] <= 901
 5(ConverseOfToLeftOf)4: L[4] - R[5] - 1048 LEFT[5,4] <= 32
 5(ConverseOfAboveOf)4: T[4] - B[5] - 933 ABOVE[5,4] <= 32
 OneLAGForElement[0]: zLAG[0,0] + zLAG[0,1] + zLAG[0,2] + zLAG[0,3]
   + zLAG[0,4] + zLAG[0,5] = 1
 OneTAGForElement[0]: zTAG[0,0] + zTAG[0,1] + zTAG[0,2] + zTAG[0,3]
   + zTAG[0,4] + zTAG[0,5] = 1
 OneBAGForElement[0]: zBAG[0,0] + zBAG[0,1] + zBAG[0,2] + zBAG[0,3]
   + zBAG[0,4] + zBAG[0,5] = 1
 OneRAGForElement[0]: zRAG[0,0] + zRAG[0,1] + zRAG[0,2] + zRAG[0,3]
   + zRAG[0,4] + zRAG[0,5] = 1
 OneLAGForElement[1]: zLAG[1,0] + zLAG[1,1] + zLAG[1,2] + zLAG[1,3]
   + zLAG[1,4] + zLAG[1,5] = 1
 OneTAGForElement[1]: zTAG[1,0] + zTAG[1,1] + zTAG[1,2] + zTAG[1,3]
   + zTAG[1,4] + zTAG[1,5] = 1
 OneBAGForElement[1]: zBAG[1,0] + zBAG[1,1] + zBAG[1,2] + zBAG[1,3]
   + zBAG[1,4] + zBAG[1,5] = 1
 OneRAGForElement[1]: zRAG[1,0] + zRAG[1,1] + zRAG[1,2] + zRAG[1,3]
   + zRAG[1,4] + zRAG[1,5] = 1
 OneLAGForElement[2]: zLAG[2,0] + zLAG[2,1] + zLAG[2,2] + zLAG[2,3]
   + zLAG[2,4] + zLAG[2,5] = 1
 OneTAGForElement[2]: zTAG[2,0] + zTAG[2,1] + zTAG[2,2] + zTAG[2,3]
   + zTAG[2,4] + zTAG[2,5] = 1
 OneBAGForElement[2]: zBAG[2,0] + zBAG[2,1] + zBAG[2,2] + zBAG[2,3]
   + zBAG[2,4] + zBAG[2,5] = 1
 OneRAGForElement[2]: zRAG[2,0] + zRAG[2,1] + zRAG[2,2] + zRAG[2,3]
   + zRAG[2,4] + zRAG[2,5] = 1
 OneLAGForElement[3]: zLAG[3,0] + zLAG[3,1] + zLAG[3,2] + zLAG[3,3]
   + zLAG[3,4] + zLAG[3,5] = 1
 OneTAGForElement[3]: zTAG[3,0] + zTAG[3,1] + zTAG[3,2] + zTAG[3,3]
   + zTAG[3,4] + zTAG[3,5] = 1
 OneBAGForElement[3]: zBAG[3,0] + zBAG[3,1] + zBAG[3,2] + zBAG[3,3]
   + zBAG[3,4] + zBAG[3,5] = 1
 OneRAGForElement[3]: zRAG[3,0] + zRAG[3,1] + zRAG[3,2] + zRAG[3,3]
   + zRAG[3,4] + zRAG[3,5] = 1
 OneLAGForElement[4]: zLAG[4,0] + zLAG[4,1] + zLAG[4,2] + zLAG[4,3]
   + zLAG[4,4] + zLAG[4,5] = 1
 OneTAGForElement[4]: zTAG[4,0] + zTAG[4,1] + zTAG[4,2] + zTAG[4,3]
   + zTAG[4,4] + zTAG[4,5] = 1
 OneBAGForElement[4]: zBAG[4,0] + zBAG[4,1] + zBAG[4,2] + zBAG[4,3]
   + zBAG[4,4] + zBAG[4,5] = 1
 OneRAGForElement[4]: zRAG[4,0] + zRAG[4,1] + zRAG[4,2] + zRAG[4,3]
   + zRAG[4,4] + zRAG[4,5] = 1
 OneLAGForElement[5]: zLAG[5,0] + zLAG[5,1] + zLAG[5,2] + zLAG[5,3]
   + zLAG[5,4] + zLAG[5,5] = 1
 OneTAGForElement[5]: zTAG[5,0] + zTAG[5,1] + zTAG[5,2] + zTAG[5,3]
   + zTAG[5,4] + zTAG[5,5] = 1
 OneBAGForElement[5]: zBAG[5,0] + zBAG[5,1] + zBAG[5,2] + zBAG[5,3]
   + zBAG[5,4] + zBAG[5,5] = 1
 OneRAGForElement[5]: zRAG[5,0] + zRAG[5,1] + zRAG[5,2] + zRAG[5,3]
   + zRAG[5,4] + zRAG[5,5] = 1
 R235: - LAG[0] + zLAG[0,0] <= 0
 R236: - RAG[0] + zRAG[0,0] <= 0
 R237: - TAG[0] + zTAG[0,0] <= 0
 R238: - BAG[0] + zBAG[0,0] <= 0
 R239: - LAG[0] + zLAG[1,0] <= 0
 R240: - RAG[0] + zRAG[1,0] <= 0
 R241: - TAG[0] + zTAG[1,0] <= 0
 R242: - BAG[0] + zBAG[1,0] <= 0
 R243: - LAG[0] + zLAG[2,0] <= 0
 R244: - RAG[0] + zRAG[2,0] <= 0
 R245: - TAG[0] + zTAG[2,0] <= 0
 R246: - BAG[0] + zBAG[2,0] <= 0
 R247: - LAG[0] + zLAG[3,0] <= 0
 R248: - RAG[0] + zRAG[3,0] <= 0
 R249: - TAG[0] + zTAG[3,0] <= 0
 R250: - BAG[0] + zBAG[3,0] <= 0
 R251: - LAG[0] + zLAG[4,0] <= 0
 R252: - RAG[0] + zRAG[4,0] <= 0
 R253: - TAG[0] + zTAG[4,0] <= 0
 R254: - BAG[0] + zBAG[4,0] <= 0
 R255: - LAG[0] + zLAG[5,0] <= 0
 R256: - RAG[0] + zRAG[5,0] <= 0
 R257: - TAG[0] + zTAG[5,0] <= 0
 R258: - BAG[0] + zBAG[5,0] <= 0
 R259: - LAG[1] + zLAG[0,1] <= 0
 R260: - RAG[1] + zRAG[0,1] <= 0
 R261: - TAG[1] + zTAG[0,1] <= 0
 R262: - BAG[1] + zBAG[0,1] <= 0
 R263: - LAG[1] + zLAG[1,1] <= 0
 R264: - RAG[1] + zRAG[1,1] <= 0
 R265: - TAG[1] + zTAG[1,1] <= 0
 R266: - BAG[1] + zBAG[1,1] <= 0
 R267: - LAG[1] + zLAG[2,1] <= 0
 R268: - RAG[1] + zRAG[2,1] <= 0
 R269: - TAG[1] + zTAG[2,1] <= 0
 R270: - BAG[1] + zBAG[2,1] <= 0
 R271: - LAG[1] + zLAG[3,1] <= 0
 R272: - RAG[1] + zRAG[3,1] <= 0
 R273: - TAG[1] + zTAG[3,1] <= 0
 R274: - BAG[1] + zBAG[3,1] <= 0
 R275: - LAG[1] + zLAG[4,1] <= 0
 R276: - RAG[1] + zRAG[4,1] <= 0
 R277: - TAG[1] + zTAG[4,1] <= 0
 R278: - BAG[1] + zBAG[4,1] <= 0
 R279: - LAG[1] + zLAG[5,1] <= 0
 R280: - RAG[1] + zRAG[5,1] <= 0
 R281: - TAG[1] + zTAG[5,1] <= 0
 R282: - BAG[1] + zBAG[5,1] <= 0
 R283: - LAG[2] + zLAG[0,2] <= 0
 R284: - RAG[2] + zRAG[0,2] <= 0
 R285: - TAG[2] + zTAG[0,2] <= 0
 R286: - BAG[2] + zBAG[0,2] <= 0
 R287: - LAG[2] + zLAG[1,2] <= 0
 R288: - RAG[2] + zRAG[1,2] <= 0
 R289: - TAG[2] + zTAG[1,2] <= 0
 R290: - BAG[2] + zBAG[1,2] <= 0
 R291: - LAG[2] + zLAG[2,2] <= 0
 R292: - RAG[2] + zRAG[2,2] <= 0
 R293: - TAG[2] + zTAG[2,2] <= 0
 R294: - BAG[2] + zBAG[2,2] <= 0
 R295: - LAG[2] + zLAG[3,2] <= 0
 R296: - RAG[2] + zRAG[3,2] <= 0
 R297: - TAG[2] + zTAG[3,2] <= 0
 R298: - BAG[2] + zBAG[3,2] <= 0
 R299: - LAG[2] + zLAG[4,2] <= 0
 R300: - RAG[2] + zRAG[4,2] <= 0
 R301: - TAG[2] + zTAG[4,2] <= 0
 R302: - BAG[2] + zBAG[4,2] <= 0
 R303: - LAG[2] + zLAG[5,2] <= 0
 R304: - RAG[2] + zRAG[5,2] <= 0
 R305: - TAG[2] + zTAG[5,2] <= 0
 R306: - BAG[2] + zBAG[5,2] <= 0
 R307: - LAG[3] + zLAG[0,3] <= 0
 R308: - RAG[3] + zRAG[0,3] <= 0
 R309: - TAG[3] + zTAG[0,3] <= 0
 R310: - BAG[3] + zBAG[0,3] <= 0
 R311: - LAG[3] + zLAG[1,3] <= 0
 R312: - RAG[3] + zRAG[1,3] <= 0
 R313: - TAG[3] + zTAG[1,3] <= 0
 R314: - BAG[3] + zBAG[1,3] <= 0
 R315: - LAG[3] + zLAG[2,3] <= 0
 R316: - RAG[3] + zRAG[2,3] <= 0
 R317: - TAG[3] + zTAG[2,3] <= 0
 R318: - BAG[3] + zBAG[2,3] <= 0
 R319: - LAG[3] + zLAG[3,3] <= 0
 R320: - RAG[3] + zRAG[3,3] <= 0
 R321: - TAG[3] + zTAG[3,3] <= 0
 R322: - BAG[3] + zBAG[3,3] <= 0
 R323: - LAG[3] + zLAG[4,3] <= 0
 R324: - RAG[3] + zRAG[4,3] <= 0
 R325: - TAG[3] + zTAG[4,3] <= 0
 R326: - BAG[3] + zBAG[4,3] <= 0
 R327: - LAG[3] + zLAG[5,3] <= 0
 R328: - RAG[3] + zRAG[5,3] <= 0
 R329: - TAG[3] + zTAG[5,3] <= 0
 R330: - BAG[3] + zBAG[5,3] <= 0
 R331: - LAG[4] + zLAG[0,4] <= 0
 R332: - RAG[4] + zRAG[0,4] <= 0
 R333: - TAG[4] + zTAG[0,4] <= 0
 R334: - BAG[4] + zBAG[0,4] <= 0
 R335: - LAG[4] + zLAG[1,4] <= 0
 R336: - RAG[4] + zRAG[1,4] <= 0
 R337: - TAG[4] + zTAG[1,4] <= 0
 R338: - BAG[4] + zBAG[1,4] <= 0
 R339: - LAG[4] + zLAG[2,4] <= 0
 R340: - RAG[4] + zRAG[2,4] <= 0
 R341: - TAG[4] + zTAG[2,4] <= 0
 R342: - BAG[4] + zBAG[2,4] <= 0
 R343: - LAG[4] + zLAG[3,4] <= 0
 R344: - RAG[4] + zRAG[3,4] <= 0
 R345: - TAG[4] + zTAG[3,4] <= 0
 R346: - BAG[4] + zBAG[3,4] <= 0
 R347: - LAG[4] + zLAG[4,4] <= 0
 R348: - RAG[4] + zRAG[4,4] <= 0
 R349: - TAG[4] + zTAG[4,4] <= 0
 R350: - BAG[4] + zBAG[4,4] <= 0
 R351: - LAG[4] + zLAG[5,4] <= 0
 R352: - RAG[4] + zRAG[5,4] <= 0
 R353: - TAG[4] + zTAG[5,4] <= 0
 R354: - BAG[4] + zBAG[5,4] <= 0
 R355: - LAG[5] + zLAG[0,5] <= 0
 R356: - RAG[5] + zRAG[0,5] <= 0
 R357: - TAG[5] + zTAG[0,5] <= 0
 R358: - BAG[5] + zBAG[0,5] <= 0
 R359: - LAG[5] + zLAG[1,5] <= 0
 R360: - RAG[5] + zRAG[1,5] <= 0
 R361: - TAG[5] + zTAG[1,5] <= 0
 R362: - BAG[5] + zBAG[1,5] <= 0
 R363: - LAG[5] + zLAG[2,5] <= 0
 R364: - RAG[5] + zRAG[2,5] <= 0
 R365: - TAG[5] + zTAG[2,5] <= 0
 R366: - BAG[5] + zBAG[2,5] <= 0
 R367: - LAG[5] + zLAG[3,5] <= 0
 R368: - RAG[5] + zRAG[3,5] <= 0
 R369: - TAG[5] + zTAG[3,5] <= 0
 R370: - BAG[5] + zBAG[3,5] <= 0
 R371: - LAG[5] + zLAG[4,5] <= 0
 R372: - RAG[5] + zRAG[4,5] <= 0
 R373: - TAG[5] + zTAG[4,5] <= 0
 R374: - BAG[5] + zBAG[4,5] <= 0
 R375: - LAG[5] + zLAG[5,5] <= 0
 R376: - RAG[5] + zRAG[5,5] <= 0
 R377: - TAG[5] + zTAG[5,5] <= 0
 R378: - BAG[5] + zBAG[5,5] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 1048 zLAG[0,0] <= 1048
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 1048 zRAG[0,0] <= 1048
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 933 zTAG[0,0] <= 933
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 933 zBAG[0,0] <= 933
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 1048 zLAG[0,0] >= -1048
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 1048 zRAG[0,0] >= -1048
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 933 zTAG[0,0] >= -933
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 933 zBAG[0,0] >= -933
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 1048 zLAG[1,0] <= 1048
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 1048 zRAG[1,0] <= 1048
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 933 zTAG[1,0] <= 933
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 933 zBAG[1,0] <= 933
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 1048 zLAG[1,0] >= -1048
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 1048 zRAG[1,0] >= -1048
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 933 zTAG[1,0] >= -933
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 933 zBAG[1,0] >= -933
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 1048 zLAG[2,0] <= 1048
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 1048 zRAG[2,0] <= 1048
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 933 zTAG[2,0] <= 933
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 933 zBAG[2,0] <= 933
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 1048 zLAG[2,0] >= -1048
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 1048 zRAG[2,0] >= -1048
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 933 zTAG[2,0] >= -933
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 933 zBAG[2,0] >= -933
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 1048 zLAG[3,0] <= 1048
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 1048 zRAG[3,0] <= 1048
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 933 zTAG[3,0] <= 933
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 933 zBAG[3,0] <= 933
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 1048 zLAG[3,0] >= -1048
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 1048 zRAG[3,0] >= -1048
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 933 zTAG[3,0] >= -933
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 933 zBAG[3,0] >= -933
 MinsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] + 1048 zLAG[4,0] <= 1048
 MinsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] + 1048 zRAG[4,0] <= 1048
 MinsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] + 933 zTAG[4,0] <= 933
 MinsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] + 933 zBAG[4,0] <= 933
 MaxsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] - 1048 zLAG[4,0] >= -1048
 MaxsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] - 1048 zRAG[4,0] >= -1048
 MaxsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] - 933 zTAG[4,0] >= -933
 MaxsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] - 933 zBAG[4,0] >= -933
 MinsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] + 1048 zLAG[5,0] <= 1048
 MinsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] + 1048 zRAG[5,0] <= 1048
 MinsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] + 933 zTAG[5,0] <= 933
 MinsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] + 933 zBAG[5,0] <= 933
 MaxsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] - 1048 zLAG[5,0] >= -1048
 MaxsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] - 1048 zRAG[5,0] >= -1048
 MaxsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] - 933 zTAG[5,0] >= -933
 MaxsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] - 933 zBAG[5,0] >= -933
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 1048 zLAG[0,1] <= 1048
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 1048 zRAG[0,1] <= 1048
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 933 zTAG[0,1] <= 933
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 933 zBAG[0,1] <= 933
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 1048 zLAG[0,1] >= -1048
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 1048 zRAG[0,1] >= -1048
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 933 zTAG[0,1] >= -933
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 933 zBAG[0,1] >= -933
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 1048 zLAG[1,1] <= 1048
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 1048 zRAG[1,1] <= 1048
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 933 zTAG[1,1] <= 933
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 933 zBAG[1,1] <= 933
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 1048 zLAG[1,1] >= -1048
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 1048 zRAG[1,1] >= -1048
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 933 zTAG[1,1] >= -933
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 933 zBAG[1,1] >= -933
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 1048 zLAG[2,1] <= 1048
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 1048 zRAG[2,1] <= 1048
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 933 zTAG[2,1] <= 933
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 933 zBAG[2,1] <= 933
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 1048 zLAG[2,1] >= -1048
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 1048 zRAG[2,1] >= -1048
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 933 zTAG[2,1] >= -933
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 933 zBAG[2,1] >= -933
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 1048 zLAG[3,1] <= 1048
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 1048 zRAG[3,1] <= 1048
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 933 zTAG[3,1] <= 933
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 933 zBAG[3,1] <= 933
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 1048 zLAG[3,1] >= -1048
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 1048 zRAG[3,1] >= -1048
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 933 zTAG[3,1] >= -933
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 933 zBAG[3,1] >= -933
 MinsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] + 1048 zLAG[4,1] <= 1048
 MinsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] + 1048 zRAG[4,1] <= 1048
 MinsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] + 933 zTAG[4,1] <= 933
 MinsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] + 933 zBAG[4,1] <= 933
 MaxsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] - 1048 zLAG[4,1] >= -1048
 MaxsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] - 1048 zRAG[4,1] >= -1048
 MaxsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] - 933 zTAG[4,1] >= -933
 MaxsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] - 933 zBAG[4,1] >= -933
 MinsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] + 1048 zLAG[5,1] <= 1048
 MinsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] + 1048 zRAG[5,1] <= 1048
 MinsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] + 933 zTAG[5,1] <= 933
 MinsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] + 933 zBAG[5,1] <= 933
 MaxsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] - 1048 zLAG[5,1] >= -1048
 MaxsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] - 1048 zRAG[5,1] >= -1048
 MaxsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] - 933 zTAG[5,1] >= -933
 MaxsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] - 933 zBAG[5,1] >= -933
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 1048 zLAG[0,2] <= 1048
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 1048 zRAG[0,2] <= 1048
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 933 zTAG[0,2] <= 933
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 933 zBAG[0,2] <= 933
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 1048 zLAG[0,2] >= -1048
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 1048 zRAG[0,2] >= -1048
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 933 zTAG[0,2] >= -933
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 933 zBAG[0,2] >= -933
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 1048 zLAG[1,2] <= 1048
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 1048 zRAG[1,2] <= 1048
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 933 zTAG[1,2] <= 933
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 933 zBAG[1,2] <= 933
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 1048 zLAG[1,2] >= -1048
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 1048 zRAG[1,2] >= -1048
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 933 zTAG[1,2] >= -933
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 933 zBAG[1,2] >= -933
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 1048 zLAG[2,2] <= 1048
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 1048 zRAG[2,2] <= 1048
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 933 zTAG[2,2] <= 933
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 933 zBAG[2,2] <= 933
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 1048 zLAG[2,2] >= -1048
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 1048 zRAG[2,2] >= -1048
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 933 zTAG[2,2] >= -933
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 933 zBAG[2,2] >= -933
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 1048 zLAG[3,2] <= 1048
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 1048 zRAG[3,2] <= 1048
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 933 zTAG[3,2] <= 933
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 933 zBAG[3,2] <= 933
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 1048 zLAG[3,2] >= -1048
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 1048 zRAG[3,2] >= -1048
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 933 zTAG[3,2] >= -933
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 933 zBAG[3,2] >= -933
 MinsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] + 1048 zLAG[4,2] <= 1048
 MinsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] + 1048 zRAG[4,2] <= 1048
 MinsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] + 933 zTAG[4,2] <= 933
 MinsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] + 933 zBAG[4,2] <= 933
 MaxsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] - 1048 zLAG[4,2] >= -1048
 MaxsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] - 1048 zRAG[4,2] >= -1048
 MaxsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] - 933 zTAG[4,2] >= -933
 MaxsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] - 933 zBAG[4,2] >= -933
 MinsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] + 1048 zLAG[5,2] <= 1048
 MinsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] + 1048 zRAG[5,2] <= 1048
 MinsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] + 933 zTAG[5,2] <= 933
 MinsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] + 933 zBAG[5,2] <= 933
 MaxsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] - 1048 zLAG[5,2] >= -1048
 MaxsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] - 1048 zRAG[5,2] >= -1048
 MaxsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] - 933 zTAG[5,2] >= -933
 MaxsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] - 933 zBAG[5,2] >= -933
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 1048 zLAG[0,3] <= 1048
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 1048 zRAG[0,3] <= 1048
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 933 zTAG[0,3] <= 933
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 933 zBAG[0,3] <= 933
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 1048 zLAG[0,3] >= -1048
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 1048 zRAG[0,3] >= -1048
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 933 zTAG[0,3] >= -933
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 933 zBAG[0,3] >= -933
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 1048 zLAG[1,3] <= 1048
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 1048 zRAG[1,3] <= 1048
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 933 zTAG[1,3] <= 933
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 933 zBAG[1,3] <= 933
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 1048 zLAG[1,3] >= -1048
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 1048 zRAG[1,3] >= -1048
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 933 zTAG[1,3] >= -933
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 933 zBAG[1,3] >= -933
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 1048 zLAG[2,3] <= 1048
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 1048 zRAG[2,3] <= 1048
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 933 zTAG[2,3] <= 933
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 933 zBAG[2,3] <= 933
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 1048 zLAG[2,3] >= -1048
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 1048 zRAG[2,3] >= -1048
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 933 zTAG[2,3] >= -933
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 933 zBAG[2,3] >= -933
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 1048 zLAG[3,3] <= 1048
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 1048 zRAG[3,3] <= 1048
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 933 zTAG[3,3] <= 933
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 933 zBAG[3,3] <= 933
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 1048 zLAG[3,3] >= -1048
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 1048 zRAG[3,3] >= -1048
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 933 zTAG[3,3] >= -933
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 933 zBAG[3,3] >= -933
 MinsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] + 1048 zLAG[4,3] <= 1048
 MinsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] + 1048 zRAG[4,3] <= 1048
 MinsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] + 933 zTAG[4,3] <= 933
 MinsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] + 933 zBAG[4,3] <= 933
 MaxsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] - 1048 zLAG[4,3] >= -1048
 MaxsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] - 1048 zRAG[4,3] >= -1048
 MaxsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] - 933 zTAG[4,3] >= -933
 MaxsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] - 933 zBAG[4,3] >= -933
 MinsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] + 1048 zLAG[5,3] <= 1048
 MinsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] + 1048 zRAG[5,3] <= 1048
 MinsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] + 933 zTAG[5,3] <= 933
 MinsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] + 933 zBAG[5,3] <= 933
 MaxsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] - 1048 zLAG[5,3] >= -1048
 MaxsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] - 1048 zRAG[5,3] >= -1048
 MaxsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] - 933 zTAG[5,3] >= -933
 MaxsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] - 933 zBAG[5,3] >= -933
 MinsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] + 1048 zLAG[0,4] <= 1048
 MinsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] + 1048 zRAG[0,4] <= 1048
 MinsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] + 933 zTAG[0,4] <= 933
 MinsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] + 933 zBAG[0,4] <= 933
 MaxsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] - 1048 zLAG[0,4] >= -1048
 MaxsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] - 1048 zRAG[0,4] >= -1048
 MaxsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] - 933 zTAG[0,4] >= -933
 MaxsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] - 933 zBAG[0,4] >= -933
 MinsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] + 1048 zLAG[1,4] <= 1048
 MinsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] + 1048 zRAG[1,4] <= 1048
 MinsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] + 933 zTAG[1,4] <= 933
 MinsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] + 933 zBAG[1,4] <= 933
 MaxsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] - 1048 zLAG[1,4] >= -1048
 MaxsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] - 1048 zRAG[1,4] >= -1048
 MaxsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] - 933 zTAG[1,4] >= -933
 MaxsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] - 933 zBAG[1,4] >= -933
 MinsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] + 1048 zLAG[2,4] <= 1048
 MinsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] + 1048 zRAG[2,4] <= 1048
 MinsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] + 933 zTAG[2,4] <= 933
 MinsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] + 933 zBAG[2,4] <= 933
 MaxsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] - 1048 zLAG[2,4] >= -1048
 MaxsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] - 1048 zRAG[2,4] >= -1048
 MaxsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] - 933 zTAG[2,4] >= -933
 MaxsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] - 933 zBAG[2,4] >= -933
 MinsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] + 1048 zLAG[3,4] <= 1048
 MinsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] + 1048 zRAG[3,4] <= 1048
 MinsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] + 933 zTAG[3,4] <= 933
 MinsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] + 933 zBAG[3,4] <= 933
 MaxsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] - 1048 zLAG[3,4] >= -1048
 MaxsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] - 1048 zRAG[3,4] >= -1048
 MaxsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] - 933 zTAG[3,4] >= -933
 MaxsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] - 933 zBAG[3,4] >= -933
 MinsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] + 1048 zLAG[4,4] <= 1048
 MinsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] + 1048 zRAG[4,4] <= 1048
 MinsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] + 933 zTAG[4,4] <= 933
 MinsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] + 933 zBAG[4,4] <= 933
 MaxsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] - 1048 zLAG[4,4] >= -1048
 MaxsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] - 1048 zRAG[4,4] >= -1048
 MaxsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] - 933 zTAG[4,4] >= -933
 MaxsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] - 933 zBAG[4,4] >= -933
 MinsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] + 1048 zLAG[5,4] <= 1048
 MinsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] + 1048 zRAG[5,4] <= 1048
 MinsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] + 933 zTAG[5,4] <= 933
 MinsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] + 933 zBAG[5,4] <= 933
 MaxsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] - 1048 zLAG[5,4] >= -1048
 MaxsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] - 1048 zRAG[5,4] >= -1048
 MaxsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] - 933 zTAG[5,4] >= -933
 MaxsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] - 933 zBAG[5,4] >= -933
 MinsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] + 1048 zLAG[0,5] <= 1048
 MinsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] + 1048 zRAG[0,5] <= 1048
 MinsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] + 933 zTAG[0,5] <= 933
 MinsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] + 933 zBAG[0,5] <= 933
 MaxsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] - 1048 zLAG[0,5] >= -1048
 MaxsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] - 1048 zRAG[0,5] >= -1048
 MaxsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] - 933 zTAG[0,5] >= -933
 MaxsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] - 933 zBAG[0,5] >= -933
 MinsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] + 1048 zLAG[1,5] <= 1048
 MinsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] + 1048 zRAG[1,5] <= 1048
 MinsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] + 933 zTAG[1,5] <= 933
 MinsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] + 933 zBAG[1,5] <= 933
 MaxsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] - 1048 zLAG[1,5] >= -1048
 MaxsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] - 1048 zRAG[1,5] >= -1048
 MaxsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] - 933 zTAG[1,5] >= -933
 MaxsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] - 933 zBAG[1,5] >= -933
 MinsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] + 1048 zLAG[2,5] <= 1048
 MinsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] + 1048 zRAG[2,5] <= 1048
 MinsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] + 933 zTAG[2,5] <= 933
 MinsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] + 933 zBAG[2,5] <= 933
 MaxsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] - 1048 zLAG[2,5] >= -1048
 MaxsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] - 1048 zRAG[2,5] >= -1048
 MaxsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] - 933 zTAG[2,5] >= -933
 MaxsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] - 933 zBAG[2,5] >= -933
 MinsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] + 1048 zLAG[3,5] <= 1048
 MinsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] + 1048 zRAG[3,5] <= 1048
 MinsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] + 933 zTAG[3,5] <= 933
 MinsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] + 933 zBAG[3,5] <= 933
 MaxsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] - 1048 zLAG[3,5] >= -1048
 MaxsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] - 1048 zRAG[3,5] >= -1048
 MaxsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] - 933 zTAG[3,5] >= -933
 MaxsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] - 933 zBAG[3,5] >= -933
 MinsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] + 1048 zLAG[4,5] <= 1048
 MinsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] + 1048 zRAG[4,5] <= 1048
 MinsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] + 933 zTAG[4,5] <= 933
 MinsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] + 933 zBAG[4,5] <= 933
 MaxsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] - 1048 zLAG[4,5] >= -1048
 MaxsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] - 1048 zRAG[4,5] >= -1048
 MaxsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] - 933 zTAG[4,5] >= -933
 MaxsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] - 933 zBAG[4,5] >= -933
 MinsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] + 1048 zLAG[5,5] <= 1048
 MinsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] + 1048 zRAG[5,5] <= 1048
 MinsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] + 933 zTAG[5,5] <= 933
 MinsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] + 933 zBAG[5,5] <= 933
 MaxsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] - 1048 zLAG[5,5] >= -1048
 MaxsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] - 1048 zRAG[5,5] >= -1048
 MaxsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] - 933 zTAG[5,5] >= -933
 MaxsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] - 933 zBAG[5,5] >= -933
Bounds
 L[0] <= 105
 L[1] <= 720
 L[2] <= 611
 L[3] <= 680
 L[4] <= 952
 L[5] <= 544
 943 <= R[0] <= 1048
 328 <= R[1] <= 1048
 437 <= R[2] <= 1048
 368 <= R[3] <= 1048
 96 <= R[4] <= 1048
 504 <= R[5] <= 1048
 T[0] <= 901
 T[1] <= 453
 T[2] <= 558
 T[3] <= 613
 T[4] <= 321
 T[5] <= 581
 32 <= B[0] <= 933
 480 <= B[1] <= 933
 375 <= B[2] <= 933
 320 <= B[3] <= 933
 612 <= B[4] <= 933
 352 <= B[5] <= 933
 H[0] = 32
 H[1] = 480
 H[2] = 375
 H[3] = 320
 H[4] = 612
 H[5] = 352
 W[0] = 943
 W[1] = 328
 W[2] = 437
 W[3] = 368
 W[4] = 96
 W[5] = 504
 vLAG[0] <= 1047
 vLAG[1] <= 1047
 vLAG[2] <= 1047
 vLAG[3] <= 1047
 vLAG[4] <= 1047
 vLAG[5] <= 1047
 1 <= vRAG[0] <= 1048
 1 <= vRAG[1] <= 1048
 1 <= vRAG[2] <= 1048
 1 <= vRAG[3] <= 1048
 1 <= vRAG[4] <= 1048
 1 <= vRAG[5] <= 1048
 vTAG[0] <= 932
 vTAG[1] <= 932
 vTAG[2] <= 932
 vTAG[3] <= 932
 vTAG[4] <= 932
 vTAG[5] <= 932
 1 <= vBAG[0] <= 933
 1 <= vBAG[1] <= 933
 1 <= vBAG[2] <= 933
 1 <= vBAG[3] <= 933
 1 <= vBAG[4] <= 933
 1 <= vBAG[5] <= 933
Binaries
 ABOVE[0,0] ABOVE[0,1] ABOVE[0,2] ABOVE[0,3] ABOVE[0,4] ABOVE[0,5]
 ABOVE[1,0] ABOVE[1,1] ABOVE[1,2] ABOVE[1,3] ABOVE[1,4] ABOVE[1,5]
 ABOVE[2,0] ABOVE[2,1] ABOVE[2,2] ABOVE[2,3] ABOVE[2,4] ABOVE[2,5]
 ABOVE[3,0] ABOVE[3,1] ABOVE[3,2] ABOVE[3,3] ABOVE[3,4] ABOVE[3,5]
 ABOVE[4,0] ABOVE[4,1] ABOVE[4,2] ABOVE[4,3] ABOVE[4,4] ABOVE[4,5]
 ABOVE[5,0] ABOVE[5,1] ABOVE[5,2] ABOVE[5,3] ABOVE[5,4] ABOVE[5,5]
 LEFT[0,0] LEFT[0,1] LEFT[0,2] LEFT[0,3] LEFT[0,4] LEFT[0,5] LEFT[1,0]
 LEFT[1,1] LEFT[1,2] LEFT[1,3] LEFT[1,4] LEFT[1,5] LEFT[2,0] LEFT[2,1]
 LEFT[2,2] LEFT[2,3] LEFT[2,4] LEFT[2,5] LEFT[3,0] LEFT[3,1] LEFT[3,2]
 LEFT[3,3] LEFT[3,4] LEFT[3,5] LEFT[4,0] LEFT[4,1] LEFT[4,2] LEFT[4,3]
 LEFT[4,4] LEFT[4,5] LEFT[5,0] LEFT[5,1] LEFT[5,2] LEFT[5,3] LEFT[5,4]
 LEFT[5,5] LAG[0] LAG[1] LAG[2] LAG[3] LAG[4] LAG[5] RAG[0] RAG[1] RAG[2]
 RAG[3] RAG[4] RAG[5] TAG[0] TAG[1] TAG[2] TAG[3] TAG[4] TAG[5] BAG[0]
 BAG[1] BAG[2] BAG[3] BAG[4] BAG[5] zLAG[0,0] zLAG[0,1] zLAG[0,2] zLAG[0,3]
 zLAG[0,4] zLAG[0,5] zLAG[1,0] zLAG[1,1] zLAG[1,2] zLAG[1,3] zLAG[1,4]
 zLAG[1,5] zLAG[2,0] zLAG[2,1] zLAG[2,2] zLAG[2,3] zLAG[2,4] zLAG[2,5]
 zLAG[3,0] zLAG[3,1] zLAG[3,2] zLAG[3,3] zLAG[3,4] zLAG[3,5] zLAG[4,0]
 zLAG[4,1] zLAG[4,2] zLAG[4,3] zLAG[4,4] zLAG[4,5] zLAG[5,0] zLAG[5,1]
 zLAG[5,2] zLAG[5,3] zLAG[5,4] zLAG[5,5] zRAG[0,0] zRAG[0,1] zRAG[0,2]
 zRAG[0,3] zRAG[0,4] zRAG[0,5] zRAG[1,0] zRAG[1,1] zRAG[1,2] zRAG[1,3]
 zRAG[1,4] zRAG[1,5] zRAG[2,0] zRAG[2,1] zRAG[2,2] zRAG[2,3] zRAG[2,4]
 zRAG[2,5] zRAG[3,0] zRAG[3,1] zRAG[3,2] zRAG[3,3] zRAG[3,4] zRAG[3,5]
 zRAG[4,0] zRAG[4,1] zRAG[4,2] zRAG[4,3] zRAG[4,4] zRAG[4,5] zRAG[5,0]
 zRAG[5,1] zRAG[5,2] zRAG[5,3] zRAG[5,4] zRAG[5,5] zTAG[0,0] zTAG[0,1]
 zTAG[0,2] zTAG[0,3] zTAG[0,4] zTAG[0,5] zTAG[1,0] zTAG[1,1] zTAG[1,2]
 zTAG[1,3] zTAG[1,4] zTAG[1,5] zTAG[2,0] zTAG[2,1] zTAG[2,2] zTAG[2,3]
 zTAG[2,4] zTAG[2,5] zTAG[3,0] zTAG[3,1] zTAG[3,2] zTAG[3,3] zTAG[3,4]
 zTAG[3,5] zTAG[4,0] zTAG[4,1] zTAG[4,2] zTAG[4,3] zTAG[4,4] zTAG[4,5]
 zTAG[5,0] zTAG[5,1] zTAG[5,2] zTAG[5,3] zTAG[5,4] zTAG[5,5] zBAG[0,0]
 zBAG[0,1] zBAG[0,2] zBAG[0,3] zBAG[0,4] zBAG[0,5] zBAG[1,0] zBAG[1,1]
 zBAG[1,2] zBAG[1,3] zBAG[1,4] zBAG[1,5] zBAG[2,0] zBAG[2,1] zBAG[2,2]
 zBAG[2,3] zBAG[2,4] zBAG[2,5] zBAG[3,0] zBAG[3,1] zBAG[3,2] zBAG[3,3]
 zBAG[3,4] zBAG[3,5] zBAG[4,0] zBAG[4,1] zBAG[4,2] zBAG[4,3] zBAG[4,4]
 zBAG[4,5] zBAG[5,0] zBAG[5,1] zBAG[5,2] zBAG[5,3] zBAG[5,4] zBAG[5,5]
Generals
 L[0] L[1] L[2] L[3] L[4] L[5] R[0] R[1] R[2] R[3] R[4] R[5] T[0] T[1] T[2]
 T[3] T[4] T[5] B[0] B[1] B[2] B[3] B[4] B[5] H[0] H[1] H[2] H[3] H[4] H[5]
 W[0] W[1] W[2] W[3] W[4] W[5] vLAG[0] vLAG[1] vLAG[2] vLAG[3] vLAG[4]
 vLAG[5] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vRAG[5] vTAG[0] vTAG[1]
 vTAG[2] vTAG[3] vTAG[4] vTAG[5] vBAG[0] vBAG[1] vBAG[2] vBAG[3] vBAG[4]
 vBAG[5] maxX maxY
End
