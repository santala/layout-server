\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.001 L[4]
   + 0.001 L[5] + 0.001 L[6] + 0.002 R[0] + 0.002 R[1] + 0.002 R[2]
   + 0.002 R[3] + 0.002 R[4] + 0.002 R[5] + 0.002 R[6] + 0.001 T[0]
   + 0.001 T[1] + 0.001 T[2] + 0.001 T[3] + 0.001 T[4] + 0.001 T[5]
   + 0.001 T[6] + 0.002 B[0] + 0.002 B[1] + 0.002 B[2] + 0.002 B[3]
   + 0.002 B[4] + 0.002 B[5] + 0.002 B[6] - 0.001 H[0] - 0.001 H[1]
   - 0.001 H[2] - 0.001 H[3] - 0.001 H[4] - 0.001 H[5] - 0.001 H[6]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] - 0.001 W[4]
   - 0.001 W[5] - 0.001 W[6] + 0 ABOVE[0,0] + 0 ABOVE[1,1] + 0 ABOVE[2,2]
   + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 ABOVE[5,5] + 0 ABOVE[6,6]
   + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2] + 0 LEFT[3,3] + 0 LEFT[4,4]
   + 0 LEFT[5,5] + 0 LEFT[6,6] + 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3]
   + 2 LAG[4] + 2 LAG[5] + 2 LAG[6] + RAG[0] + RAG[1] + RAG[2] + RAG[3]
   + RAG[4] + RAG[5] + RAG[6] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3]
   + 2 TAG[4] + 2 TAG[5] + 2 TAG[6] + BAG[0] + BAG[1] + BAG[2] + BAG[3]
   + BAG[4] + BAG[5] + BAG[6]
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
 R12: - R[6] + maxX >= 0
 R13: - B[6] + maxY >= 0
 R14: 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + 2 LAG[5]
   + 2 LAG[6] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5]
   + RAG[6] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3] + 2 TAG[4]
   + 2 TAG[5] + 2 TAG[6] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4]
   + BAG[5] + BAG[6] >= 12
 PrespecifiedXOfElement(: L[0] = 800
 PrespecifiedYOfElement(: T[0] = 100
 PrespecifiedXOfElement(: L[1] = 0
 PrespecifiedYOfElement(: T[1] = 500
 PrespecifiedXOfElement(: L[2] = 400
 PrespecifiedYOfElement(: T[2] = 500
 PrespecifiedXOfElement(: L[3] = 400
 PrespecifiedYOfElement(: T[3] = 100
 PrespecifiedXOfElement(: L[4] = 0
 PrespecifiedYOfElement(: T[4] = 300
 PrespecifiedXOfElement(: L[5] = 0
 PrespecifiedYOfElement(: T[5] = 100
 PrespecifiedXOfElement(: L[6] = 0
 PrespecifiedYOfElement(: T[6] = 800
 Disable_non-permitted_based_on_prespecified: ABOVE[6,0] + ABOVE[6,1]
   + ABOVE[6,2] + ABOVE[6,3] + ABOVE[6,4] + ABOVE[6,5] = 0
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
 R-L=W(6): L[6] - R[6] + W[6] = 0
 B-T=H(6): T[6] - B[6] + H[6] = 0
 NoOverlap(10): ABOVE[0,1] + ABOVE[1,0] + LEFT[0,1] + LEFT[1,0] >= 1
 UpperLimOfQuadrants(10): ABOVE[0,1] + ABOVE[1,0] + LEFT[0,1] + LEFT[1,0]
   <= 2
 Anti-symmetryABOVE(10): ABOVE[0,1] + ABOVE[1,0] <= 1
 Anti-symmetryLEFT(10): LEFT[0,1] + LEFT[1,0] <= 1
 NoOverlap(20): ABOVE[0,2] + ABOVE[2,0] + LEFT[0,2] + LEFT[2,0] >= 1
 UpperLimOfQuadrants(20): ABOVE[0,2] + ABOVE[2,0] + LEFT[0,2] + LEFT[2,0]
   <= 2
 Anti-symmetryABOVE(20): ABOVE[0,2] + ABOVE[2,0] <= 1
 Anti-symmetryLEFT(20): LEFT[0,2] + LEFT[2,0] <= 1
 NoOverlap(21): ABOVE[1,2] + ABOVE[2,1] + LEFT[1,2] + LEFT[2,1] >= 1
 UpperLimOfQuadrants(21): ABOVE[1,2] + ABOVE[2,1] + LEFT[1,2] + LEFT[2,1]
   <= 2
 Anti-symmetryABOVE(21): ABOVE[1,2] + ABOVE[2,1] <= 1
 Anti-symmetryLEFT(21): LEFT[1,2] + LEFT[2,1] <= 1
 NoOverlap(30): ABOVE[0,3] + ABOVE[3,0] + LEFT[0,3] + LEFT[3,0] >= 1
 UpperLimOfQuadrants(30): ABOVE[0,3] + ABOVE[3,0] + LEFT[0,3] + LEFT[3,0]
   <= 2
 Anti-symmetryABOVE(30): ABOVE[0,3] + ABOVE[3,0] <= 1
 Anti-symmetryLEFT(30): LEFT[0,3] + LEFT[3,0] <= 1
 NoOverlap(31): ABOVE[1,3] + ABOVE[3,1] + LEFT[1,3] + LEFT[3,1] >= 1
 UpperLimOfQuadrants(31): ABOVE[1,3] + ABOVE[3,1] + LEFT[1,3] + LEFT[3,1]
   <= 2
 Anti-symmetryABOVE(31): ABOVE[1,3] + ABOVE[3,1] <= 1
 Anti-symmetryLEFT(31): LEFT[1,3] + LEFT[3,1] <= 1
 NoOverlap(32): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2] >= 1
 UpperLimOfQuadrants(32): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2]
   <= 2
 Anti-symmetryABOVE(32): ABOVE[2,3] + ABOVE[3,2] <= 1
 Anti-symmetryLEFT(32): LEFT[2,3] + LEFT[3,2] <= 1
 NoOverlap(40): ABOVE[0,4] + ABOVE[4,0] + LEFT[0,4] + LEFT[4,0] >= 1
 UpperLimOfQuadrants(40): ABOVE[0,4] + ABOVE[4,0] + LEFT[0,4] + LEFT[4,0]
   <= 2
 Anti-symmetryABOVE(40): ABOVE[0,4] + ABOVE[4,0] <= 1
 Anti-symmetryLEFT(40): LEFT[0,4] + LEFT[4,0] <= 1
 NoOverlap(41): ABOVE[1,4] + ABOVE[4,1] + LEFT[1,4] + LEFT[4,1] >= 1
 UpperLimOfQuadrants(41): ABOVE[1,4] + ABOVE[4,1] + LEFT[1,4] + LEFT[4,1]
   <= 2
 Anti-symmetryABOVE(41): ABOVE[1,4] + ABOVE[4,1] <= 1
 Anti-symmetryLEFT(41): LEFT[1,4] + LEFT[4,1] <= 1
 NoOverlap(42): ABOVE[2,4] + ABOVE[4,2] + LEFT[2,4] + LEFT[4,2] >= 1
 UpperLimOfQuadrants(42): ABOVE[2,4] + ABOVE[4,2] + LEFT[2,4] + LEFT[4,2]
   <= 2
 Anti-symmetryABOVE(42): ABOVE[2,4] + ABOVE[4,2] <= 1
 Anti-symmetryLEFT(42): LEFT[2,4] + LEFT[4,2] <= 1
 NoOverlap(43): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3] >= 1
 UpperLimOfQuadrants(43): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3]
   <= 2
 Anti-symmetryABOVE(43): ABOVE[3,4] + ABOVE[4,3] <= 1
 Anti-symmetryLEFT(43): LEFT[3,4] + LEFT[4,3] <= 1
 NoOverlap(50): ABOVE[0,5] + ABOVE[5,0] + LEFT[0,5] + LEFT[5,0] >= 1
 UpperLimOfQuadrants(50): ABOVE[0,5] + ABOVE[5,0] + LEFT[0,5] + LEFT[5,0]
   <= 2
 Anti-symmetryABOVE(50): ABOVE[0,5] + ABOVE[5,0] <= 1
 Anti-symmetryLEFT(50): LEFT[0,5] + LEFT[5,0] <= 1
 NoOverlap(51): ABOVE[1,5] + ABOVE[5,1] + LEFT[1,5] + LEFT[5,1] >= 1
 UpperLimOfQuadrants(51): ABOVE[1,5] + ABOVE[5,1] + LEFT[1,5] + LEFT[5,1]
   <= 2
 Anti-symmetryABOVE(51): ABOVE[1,5] + ABOVE[5,1] <= 1
 Anti-symmetryLEFT(51): LEFT[1,5] + LEFT[5,1] <= 1
 NoOverlap(52): ABOVE[2,5] + ABOVE[5,2] + LEFT[2,5] + LEFT[5,2] >= 1
 UpperLimOfQuadrants(52): ABOVE[2,5] + ABOVE[5,2] + LEFT[2,5] + LEFT[5,2]
   <= 2
 Anti-symmetryABOVE(52): ABOVE[2,5] + ABOVE[5,2] <= 1
 Anti-symmetryLEFT(52): LEFT[2,5] + LEFT[5,2] <= 1
 NoOverlap(53): ABOVE[3,5] + ABOVE[5,3] + LEFT[3,5] + LEFT[5,3] >= 1
 UpperLimOfQuadrants(53): ABOVE[3,5] + ABOVE[5,3] + LEFT[3,5] + LEFT[5,3]
   <= 2
 Anti-symmetryABOVE(53): ABOVE[3,5] + ABOVE[5,3] <= 1
 Anti-symmetryLEFT(53): LEFT[3,5] + LEFT[5,3] <= 1
 NoOverlap(54): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4] >= 1
 UpperLimOfQuadrants(54): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4]
   <= 2
 Anti-symmetryABOVE(54): ABOVE[4,5] + ABOVE[5,4] <= 1
 Anti-symmetryLEFT(54): LEFT[4,5] + LEFT[5,4] <= 1
 NoOverlap(60): ABOVE[0,6] + ABOVE[6,0] + LEFT[0,6] + LEFT[6,0] >= 1
 UpperLimOfQuadrants(60): ABOVE[0,6] + ABOVE[6,0] + LEFT[0,6] + LEFT[6,0]
   <= 2
 Anti-symmetryABOVE(60): ABOVE[0,6] + ABOVE[6,0] <= 1
 Anti-symmetryLEFT(60): LEFT[0,6] + LEFT[6,0] <= 1
 NoOverlap(61): ABOVE[1,6] + ABOVE[6,1] + LEFT[1,6] + LEFT[6,1] >= 1
 UpperLimOfQuadrants(61): ABOVE[1,6] + ABOVE[6,1] + LEFT[1,6] + LEFT[6,1]
   <= 2
 Anti-symmetryABOVE(61): ABOVE[1,6] + ABOVE[6,1] <= 1
 Anti-symmetryLEFT(61): LEFT[1,6] + LEFT[6,1] <= 1
 NoOverlap(62): ABOVE[2,6] + ABOVE[6,2] + LEFT[2,6] + LEFT[6,2] >= 1
 UpperLimOfQuadrants(62): ABOVE[2,6] + ABOVE[6,2] + LEFT[2,6] + LEFT[6,2]
   <= 2
 Anti-symmetryABOVE(62): ABOVE[2,6] + ABOVE[6,2] <= 1
 Anti-symmetryLEFT(62): LEFT[2,6] + LEFT[6,2] <= 1
 NoOverlap(63): ABOVE[3,6] + ABOVE[6,3] + LEFT[3,6] + LEFT[6,3] >= 1
 UpperLimOfQuadrants(63): ABOVE[3,6] + ABOVE[6,3] + LEFT[3,6] + LEFT[6,3]
   <= 2
 Anti-symmetryABOVE(63): ABOVE[3,6] + ABOVE[6,3] <= 1
 Anti-symmetryLEFT(63): LEFT[3,6] + LEFT[6,3] <= 1
 NoOverlap(64): ABOVE[4,6] + ABOVE[6,4] + LEFT[4,6] + LEFT[6,4] >= 1
 UpperLimOfQuadrants(64): ABOVE[4,6] + ABOVE[6,4] + LEFT[4,6] + LEFT[6,4]
   <= 2
 Anti-symmetryABOVE(64): ABOVE[4,6] + ABOVE[6,4] <= 1
 Anti-symmetryLEFT(64): LEFT[4,6] + LEFT[6,4] <= 1
 NoOverlap(65): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5] >= 1
 UpperLimOfQuadrants(65): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5]
   <= 2
 Anti-symmetryABOVE(65): ABOVE[5,6] + ABOVE[6,5] <= 1
 Anti-symmetryLEFT(65): LEFT[5,6] + LEFT[6,5] <= 1
 0(ToLeftOf)1: - L[1] + R[0] + 1000 LEFT[0,1] <= 1000
 0(Above)1: - T[1] + B[0] + 950 ABOVE[0,1] <= 950
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 1000 LEFT[0,1] <= 0
 0(ConverseOfAboveOf)1: T[1] - B[0] - 950 ABOVE[0,1] <= 0
 0(ToLeftOf)2: - L[2] + R[0] + 1000 LEFT[0,2] <= 1000
 0(Above)2: - T[2] + B[0] + 950 ABOVE[0,2] <= 950
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 1000 LEFT[0,2] <= 0
 0(ConverseOfAboveOf)2: T[2] - B[0] - 950 ABOVE[0,2] <= 0
 0(ToLeftOf)3: - L[3] + R[0] + 1000 LEFT[0,3] <= 1000
 0(Above)3: - T[3] + B[0] + 950 ABOVE[0,3] <= 950
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 1000 LEFT[0,3] <= 0
 0(ConverseOfAboveOf)3: T[3] - B[0] - 950 ABOVE[0,3] <= 0
 0(ToLeftOf)4: - L[4] + R[0] + 1000 LEFT[0,4] <= 1000
 0(Above)4: - T[4] + B[0] + 950 ABOVE[0,4] <= 950
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 1000 LEFT[0,4] <= 0
 0(ConverseOfAboveOf)4: T[4] - B[0] - 950 ABOVE[0,4] <= 0
 0(ToLeftOf)5: - L[5] + R[0] + 1000 LEFT[0,5] <= 1000
 0(Above)5: - T[5] + B[0] + 950 ABOVE[0,5] <= 950
 0(ConverseOfToLeftOf)5: L[5] - R[0] - 1000 LEFT[0,5] <= 0
 0(ConverseOfAboveOf)5: T[5] - B[0] - 950 ABOVE[0,5] <= 0
 0(ToLeftOf)6: - L[6] + R[0] + 1000 LEFT[0,6] <= 1000
 0(Above)6: - T[6] + B[0] + 950 ABOVE[0,6] <= 950
 0(ConverseOfToLeftOf)6: L[6] - R[0] - 1000 LEFT[0,6] <= 0
 0(ConverseOfAboveOf)6: T[6] - B[0] - 950 ABOVE[0,6] <= 0
 1(ToLeftOf)0: - L[0] + R[1] + 1000 LEFT[1,0] <= 1000
 1(Above)0: - T[0] + B[1] + 950 ABOVE[1,0] <= 950
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 1000 LEFT[1,0] <= 0
 1(ConverseOfAboveOf)0: T[0] - B[1] - 950 ABOVE[1,0] <= 0
 1(ToLeftOf)2: - L[2] + R[1] + 1000 LEFT[1,2] <= 1000
 1(Above)2: - T[2] + B[1] + 950 ABOVE[1,2] <= 950
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 1000 LEFT[1,2] <= 0
 1(ConverseOfAboveOf)2: T[2] - B[1] - 950 ABOVE[1,2] <= 0
 1(ToLeftOf)3: - L[3] + R[1] + 1000 LEFT[1,3] <= 1000
 1(Above)3: - T[3] + B[1] + 950 ABOVE[1,3] <= 950
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 1000 LEFT[1,3] <= 0
 1(ConverseOfAboveOf)3: T[3] - B[1] - 950 ABOVE[1,3] <= 0
 1(ToLeftOf)4: - L[4] + R[1] + 1000 LEFT[1,4] <= 1000
 1(Above)4: - T[4] + B[1] + 950 ABOVE[1,4] <= 950
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 1000 LEFT[1,4] <= 0
 1(ConverseOfAboveOf)4: T[4] - B[1] - 950 ABOVE[1,4] <= 0
 1(ToLeftOf)5: - L[5] + R[1] + 1000 LEFT[1,5] <= 1000
 1(Above)5: - T[5] + B[1] + 950 ABOVE[1,5] <= 950
 1(ConverseOfToLeftOf)5: L[5] - R[1] - 1000 LEFT[1,5] <= 0
 1(ConverseOfAboveOf)5: T[5] - B[1] - 950 ABOVE[1,5] <= 0
 1(ToLeftOf)6: - L[6] + R[1] + 1000 LEFT[1,6] <= 1000
 1(Above)6: - T[6] + B[1] + 950 ABOVE[1,6] <= 950
 1(ConverseOfToLeftOf)6: L[6] - R[1] - 1000 LEFT[1,6] <= 0
 1(ConverseOfAboveOf)6: T[6] - B[1] - 950 ABOVE[1,6] <= 0
 2(ToLeftOf)0: - L[0] + R[2] + 1000 LEFT[2,0] <= 1000
 2(Above)0: - T[0] + B[2] + 950 ABOVE[2,0] <= 950
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 1000 LEFT[2,0] <= 0
 2(ConverseOfAboveOf)0: T[0] - B[2] - 950 ABOVE[2,0] <= 0
 2(ToLeftOf)1: - L[1] + R[2] + 1000 LEFT[2,1] <= 1000
 2(Above)1: - T[1] + B[2] + 950 ABOVE[2,1] <= 950
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 1000 LEFT[2,1] <= 0
 2(ConverseOfAboveOf)1: T[1] - B[2] - 950 ABOVE[2,1] <= 0
 2(ToLeftOf)3: - L[3] + R[2] + 1000 LEFT[2,3] <= 1000
 2(Above)3: - T[3] + B[2] + 950 ABOVE[2,3] <= 950
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 1000 LEFT[2,3] <= 0
 2(ConverseOfAboveOf)3: T[3] - B[2] - 950 ABOVE[2,3] <= 0
 2(ToLeftOf)4: - L[4] + R[2] + 1000 LEFT[2,4] <= 1000
 2(Above)4: - T[4] + B[2] + 950 ABOVE[2,4] <= 950
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 1000 LEFT[2,4] <= 0
 2(ConverseOfAboveOf)4: T[4] - B[2] - 950 ABOVE[2,4] <= 0
 2(ToLeftOf)5: - L[5] + R[2] + 1000 LEFT[2,5] <= 1000
 2(Above)5: - T[5] + B[2] + 950 ABOVE[2,5] <= 950
 2(ConverseOfToLeftOf)5: L[5] - R[2] - 1000 LEFT[2,5] <= 0
 2(ConverseOfAboveOf)5: T[5] - B[2] - 950 ABOVE[2,5] <= 0
 2(ToLeftOf)6: - L[6] + R[2] + 1000 LEFT[2,6] <= 1000
 2(Above)6: - T[6] + B[2] + 950 ABOVE[2,6] <= 950
 2(ConverseOfToLeftOf)6: L[6] - R[2] - 1000 LEFT[2,6] <= 0
 2(ConverseOfAboveOf)6: T[6] - B[2] - 950 ABOVE[2,6] <= 0
 3(ToLeftOf)0: - L[0] + R[3] + 1000 LEFT[3,0] <= 1000
 3(Above)0: - T[0] + B[3] + 950 ABOVE[3,0] <= 950
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 1000 LEFT[3,0] <= 0
 3(ConverseOfAboveOf)0: T[0] - B[3] - 950 ABOVE[3,0] <= 0
 3(ToLeftOf)1: - L[1] + R[3] + 1000 LEFT[3,1] <= 1000
 3(Above)1: - T[1] + B[3] + 950 ABOVE[3,1] <= 950
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 1000 LEFT[3,1] <= 0
 3(ConverseOfAboveOf)1: T[1] - B[3] - 950 ABOVE[3,1] <= 0
 3(ToLeftOf)2: - L[2] + R[3] + 1000 LEFT[3,2] <= 1000
 3(Above)2: - T[2] + B[3] + 950 ABOVE[3,2] <= 950
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 1000 LEFT[3,2] <= 0
 3(ConverseOfAboveOf)2: T[2] - B[3] - 950 ABOVE[3,2] <= 0
 3(ToLeftOf)4: - L[4] + R[3] + 1000 LEFT[3,4] <= 1000
 3(Above)4: - T[4] + B[3] + 950 ABOVE[3,4] <= 950
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 1000 LEFT[3,4] <= 0
 3(ConverseOfAboveOf)4: T[4] - B[3] - 950 ABOVE[3,4] <= 0
 3(ToLeftOf)5: - L[5] + R[3] + 1000 LEFT[3,5] <= 1000
 3(Above)5: - T[5] + B[3] + 950 ABOVE[3,5] <= 950
 3(ConverseOfToLeftOf)5: L[5] - R[3] - 1000 LEFT[3,5] <= 0
 3(ConverseOfAboveOf)5: T[5] - B[3] - 950 ABOVE[3,5] <= 0
 3(ToLeftOf)6: - L[6] + R[3] + 1000 LEFT[3,6] <= 1000
 3(Above)6: - T[6] + B[3] + 950 ABOVE[3,6] <= 950
 3(ConverseOfToLeftOf)6: L[6] - R[3] - 1000 LEFT[3,6] <= 0
 3(ConverseOfAboveOf)6: T[6] - B[3] - 950 ABOVE[3,6] <= 0
 4(ToLeftOf)0: - L[0] + R[4] + 1000 LEFT[4,0] <= 1000
 4(Above)0: - T[0] + B[4] + 950 ABOVE[4,0] <= 950
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 1000 LEFT[4,0] <= 0
 4(ConverseOfAboveOf)0: T[0] - B[4] - 950 ABOVE[4,0] <= 0
 4(ToLeftOf)1: - L[1] + R[4] + 1000 LEFT[4,1] <= 1000
 4(Above)1: - T[1] + B[4] + 950 ABOVE[4,1] <= 950
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 1000 LEFT[4,1] <= 0
 4(ConverseOfAboveOf)1: T[1] - B[4] - 950 ABOVE[4,1] <= 0
 4(ToLeftOf)2: - L[2] + R[4] + 1000 LEFT[4,2] <= 1000
 4(Above)2: - T[2] + B[4] + 950 ABOVE[4,2] <= 950
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 1000 LEFT[4,2] <= 0
 4(ConverseOfAboveOf)2: T[2] - B[4] - 950 ABOVE[4,2] <= 0
 4(ToLeftOf)3: - L[3] + R[4] + 1000 LEFT[4,3] <= 1000
 4(Above)3: - T[3] + B[4] + 950 ABOVE[4,3] <= 950
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 1000 LEFT[4,3] <= 0
 4(ConverseOfAboveOf)3: T[3] - B[4] - 950 ABOVE[4,3] <= 0
 4(ToLeftOf)5: - L[5] + R[4] + 1000 LEFT[4,5] <= 1000
 4(Above)5: - T[5] + B[4] + 950 ABOVE[4,5] <= 950
 4(ConverseOfToLeftOf)5: L[5] - R[4] - 1000 LEFT[4,5] <= 0
 4(ConverseOfAboveOf)5: T[5] - B[4] - 950 ABOVE[4,5] <= 0
 4(ToLeftOf)6: - L[6] + R[4] + 1000 LEFT[4,6] <= 1000
 4(Above)6: - T[6] + B[4] + 950 ABOVE[4,6] <= 950
 4(ConverseOfToLeftOf)6: L[6] - R[4] - 1000 LEFT[4,6] <= 0
 4(ConverseOfAboveOf)6: T[6] - B[4] - 950 ABOVE[4,6] <= 0
 5(ToLeftOf)0: - L[0] + R[5] + 1000 LEFT[5,0] <= 1000
 5(Above)0: - T[0] + B[5] + 950 ABOVE[5,0] <= 950
 5(ConverseOfToLeftOf)0: L[0] - R[5] - 1000 LEFT[5,0] <= 0
 5(ConverseOfAboveOf)0: T[0] - B[5] - 950 ABOVE[5,0] <= 0
 5(ToLeftOf)1: - L[1] + R[5] + 1000 LEFT[5,1] <= 1000
 5(Above)1: - T[1] + B[5] + 950 ABOVE[5,1] <= 950
 5(ConverseOfToLeftOf)1: L[1] - R[5] - 1000 LEFT[5,1] <= 0
 5(ConverseOfAboveOf)1: T[1] - B[5] - 950 ABOVE[5,1] <= 0
 5(ToLeftOf)2: - L[2] + R[5] + 1000 LEFT[5,2] <= 1000
 5(Above)2: - T[2] + B[5] + 950 ABOVE[5,2] <= 950
 5(ConverseOfToLeftOf)2: L[2] - R[5] - 1000 LEFT[5,2] <= 0
 5(ConverseOfAboveOf)2: T[2] - B[5] - 950 ABOVE[5,2] <= 0
 5(ToLeftOf)3: - L[3] + R[5] + 1000 LEFT[5,3] <= 1000
 5(Above)3: - T[3] + B[5] + 950 ABOVE[5,3] <= 950
 5(ConverseOfToLeftOf)3: L[3] - R[5] - 1000 LEFT[5,3] <= 0
 5(ConverseOfAboveOf)3: T[3] - B[5] - 950 ABOVE[5,3] <= 0
 5(ToLeftOf)4: - L[4] + R[5] + 1000 LEFT[5,4] <= 1000
 5(Above)4: - T[4] + B[5] + 950 ABOVE[5,4] <= 950
 5(ConverseOfToLeftOf)4: L[4] - R[5] - 1000 LEFT[5,4] <= 0
 5(ConverseOfAboveOf)4: T[4] - B[5] - 950 ABOVE[5,4] <= 0
 5(ToLeftOf)6: - L[6] + R[5] + 1000 LEFT[5,6] <= 1000
 5(Above)6: - T[6] + B[5] + 950 ABOVE[5,6] <= 950
 5(ConverseOfToLeftOf)6: L[6] - R[5] - 1000 LEFT[5,6] <= 0
 5(ConverseOfAboveOf)6: T[6] - B[5] - 950 ABOVE[5,6] <= 0
 6(ToLeftOf)0: - L[0] + R[6] + 1000 LEFT[6,0] <= 1000
 6(Above)0: - T[0] + B[6] + 950 ABOVE[6,0] <= 950
 6(ConverseOfToLeftOf)0: L[0] - R[6] - 1000 LEFT[6,0] <= 0
 6(ConverseOfAboveOf)0: T[0] - B[6] - 950 ABOVE[6,0] <= 0
 6(ToLeftOf)1: - L[1] + R[6] + 1000 LEFT[6,1] <= 1000
 6(Above)1: - T[1] + B[6] + 950 ABOVE[6,1] <= 950
 6(ConverseOfToLeftOf)1: L[1] - R[6] - 1000 LEFT[6,1] <= 0
 6(ConverseOfAboveOf)1: T[1] - B[6] - 950 ABOVE[6,1] <= 0
 6(ToLeftOf)2: - L[2] + R[6] + 1000 LEFT[6,2] <= 1000
 6(Above)2: - T[2] + B[6] + 950 ABOVE[6,2] <= 950
 6(ConverseOfToLeftOf)2: L[2] - R[6] - 1000 LEFT[6,2] <= 0
 6(ConverseOfAboveOf)2: T[2] - B[6] - 950 ABOVE[6,2] <= 0
 6(ToLeftOf)3: - L[3] + R[6] + 1000 LEFT[6,3] <= 1000
 6(Above)3: - T[3] + B[6] + 950 ABOVE[6,3] <= 950
 6(ConverseOfToLeftOf)3: L[3] - R[6] - 1000 LEFT[6,3] <= 0
 6(ConverseOfAboveOf)3: T[3] - B[6] - 950 ABOVE[6,3] <= 0
 6(ToLeftOf)4: - L[4] + R[6] + 1000 LEFT[6,4] <= 1000
 6(Above)4: - T[4] + B[6] + 950 ABOVE[6,4] <= 950
 6(ConverseOfToLeftOf)4: L[4] - R[6] - 1000 LEFT[6,4] <= 0
 6(ConverseOfAboveOf)4: T[4] - B[6] - 950 ABOVE[6,4] <= 0
 6(ToLeftOf)5: - L[5] + R[6] + 1000 LEFT[6,5] <= 1000
 6(Above)5: - T[5] + B[6] + 950 ABOVE[6,5] <= 950
 6(ConverseOfToLeftOf)5: L[5] - R[6] - 1000 LEFT[6,5] <= 0
 6(ConverseOfAboveOf)5: T[5] - B[6] - 950 ABOVE[6,5] <= 0
 OneLAGForElement[0]: zLAG[0,0] + zLAG[0,1] + zLAG[0,2] + zLAG[0,3]
   + zLAG[0,4] + zLAG[0,5] + zLAG[0,6] = 1
 OneTAGForElement[0]: zTAG[0,0] + zTAG[0,1] + zTAG[0,2] + zTAG[0,3]
   + zTAG[0,4] + zTAG[0,5] + zTAG[0,6] = 1
 OneBAGForElement[0]: zBAG[0,0] + zBAG[0,1] + zBAG[0,2] + zBAG[0,3]
   + zBAG[0,4] + zBAG[0,5] + zBAG[0,6] = 1
 OneRAGForElement[0]: zRAG[0,0] + zRAG[0,1] + zRAG[0,2] + zRAG[0,3]
   + zRAG[0,4] + zRAG[0,5] + zRAG[0,6] = 1
 OneLAGForElement[1]: zLAG[1,0] + zLAG[1,1] + zLAG[1,2] + zLAG[1,3]
   + zLAG[1,4] + zLAG[1,5] + zLAG[1,6] = 1
 OneTAGForElement[1]: zTAG[1,0] + zTAG[1,1] + zTAG[1,2] + zTAG[1,3]
   + zTAG[1,4] + zTAG[1,5] + zTAG[1,6] = 1
 OneBAGForElement[1]: zBAG[1,0] + zBAG[1,1] + zBAG[1,2] + zBAG[1,3]
   + zBAG[1,4] + zBAG[1,5] + zBAG[1,6] = 1
 OneRAGForElement[1]: zRAG[1,0] + zRAG[1,1] + zRAG[1,2] + zRAG[1,3]
   + zRAG[1,4] + zRAG[1,5] + zRAG[1,6] = 1
 OneLAGForElement[2]: zLAG[2,0] + zLAG[2,1] + zLAG[2,2] + zLAG[2,3]
   + zLAG[2,4] + zLAG[2,5] + zLAG[2,6] = 1
 OneTAGForElement[2]: zTAG[2,0] + zTAG[2,1] + zTAG[2,2] + zTAG[2,3]
   + zTAG[2,4] + zTAG[2,5] + zTAG[2,6] = 1
 OneBAGForElement[2]: zBAG[2,0] + zBAG[2,1] + zBAG[2,2] + zBAG[2,3]
   + zBAG[2,4] + zBAG[2,5] + zBAG[2,6] = 1
 OneRAGForElement[2]: zRAG[2,0] + zRAG[2,1] + zRAG[2,2] + zRAG[2,3]
   + zRAG[2,4] + zRAG[2,5] + zRAG[2,6] = 1
 OneLAGForElement[3]: zLAG[3,0] + zLAG[3,1] + zLAG[3,2] + zLAG[3,3]
   + zLAG[3,4] + zLAG[3,5] + zLAG[3,6] = 1
 OneTAGForElement[3]: zTAG[3,0] + zTAG[3,1] + zTAG[3,2] + zTAG[3,3]
   + zTAG[3,4] + zTAG[3,5] + zTAG[3,6] = 1
 OneBAGForElement[3]: zBAG[3,0] + zBAG[3,1] + zBAG[3,2] + zBAG[3,3]
   + zBAG[3,4] + zBAG[3,5] + zBAG[3,6] = 1
 OneRAGForElement[3]: zRAG[3,0] + zRAG[3,1] + zRAG[3,2] + zRAG[3,3]
   + zRAG[3,4] + zRAG[3,5] + zRAG[3,6] = 1
 OneLAGForElement[4]: zLAG[4,0] + zLAG[4,1] + zLAG[4,2] + zLAG[4,3]
   + zLAG[4,4] + zLAG[4,5] + zLAG[4,6] = 1
 OneTAGForElement[4]: zTAG[4,0] + zTAG[4,1] + zTAG[4,2] + zTAG[4,3]
   + zTAG[4,4] + zTAG[4,5] + zTAG[4,6] = 1
 OneBAGForElement[4]: zBAG[4,0] + zBAG[4,1] + zBAG[4,2] + zBAG[4,3]
   + zBAG[4,4] + zBAG[4,5] + zBAG[4,6] = 1
 OneRAGForElement[4]: zRAG[4,0] + zRAG[4,1] + zRAG[4,2] + zRAG[4,3]
   + zRAG[4,4] + zRAG[4,5] + zRAG[4,6] = 1
 OneLAGForElement[5]: zLAG[5,0] + zLAG[5,1] + zLAG[5,2] + zLAG[5,3]
   + zLAG[5,4] + zLAG[5,5] + zLAG[5,6] = 1
 OneTAGForElement[5]: zTAG[5,0] + zTAG[5,1] + zTAG[5,2] + zTAG[5,3]
   + zTAG[5,4] + zTAG[5,5] + zTAG[5,6] = 1
 OneBAGForElement[5]: zBAG[5,0] + zBAG[5,1] + zBAG[5,2] + zBAG[5,3]
   + zBAG[5,4] + zBAG[5,5] + zBAG[5,6] = 1
 OneRAGForElement[5]: zRAG[5,0] + zRAG[5,1] + zRAG[5,2] + zRAG[5,3]
   + zRAG[5,4] + zRAG[5,5] + zRAG[5,6] = 1
 OneLAGForElement[6]: zLAG[6,0] + zLAG[6,1] + zLAG[6,2] + zLAG[6,3]
   + zLAG[6,4] + zLAG[6,5] + zLAG[6,6] = 1
 OneTAGForElement[6]: zTAG[6,0] + zTAG[6,1] + zTAG[6,2] + zTAG[6,3]
   + zTAG[6,4] + zTAG[6,5] + zTAG[6,6] = 1
 OneBAGForElement[6]: zBAG[6,0] + zBAG[6,1] + zBAG[6,2] + zBAG[6,3]
   + zBAG[6,4] + zBAG[6,5] + zBAG[6,6] = 1
 OneRAGForElement[6]: zRAG[6,0] + zRAG[6,1] + zRAG[6,2] + zRAG[6,3]
   + zRAG[6,4] + zRAG[6,5] + zRAG[6,6] = 1
 R324: - LAG[0] + zLAG[0,0] <= 0
 R325: - RAG[0] + zRAG[0,0] <= 0
 R326: - TAG[0] + zTAG[0,0] <= 0
 R327: - BAG[0] + zBAG[0,0] <= 0
 R328: - LAG[0] + zLAG[1,0] <= 0
 R329: - RAG[0] + zRAG[1,0] <= 0
 R330: - TAG[0] + zTAG[1,0] <= 0
 R331: - BAG[0] + zBAG[1,0] <= 0
 R332: - LAG[0] + zLAG[2,0] <= 0
 R333: - RAG[0] + zRAG[2,0] <= 0
 R334: - TAG[0] + zTAG[2,0] <= 0
 R335: - BAG[0] + zBAG[2,0] <= 0
 R336: - LAG[0] + zLAG[3,0] <= 0
 R337: - RAG[0] + zRAG[3,0] <= 0
 R338: - TAG[0] + zTAG[3,0] <= 0
 R339: - BAG[0] + zBAG[3,0] <= 0
 R340: - LAG[0] + zLAG[4,0] <= 0
 R341: - RAG[0] + zRAG[4,0] <= 0
 R342: - TAG[0] + zTAG[4,0] <= 0
 R343: - BAG[0] + zBAG[4,0] <= 0
 R344: - LAG[0] + zLAG[5,0] <= 0
 R345: - RAG[0] + zRAG[5,0] <= 0
 R346: - TAG[0] + zTAG[5,0] <= 0
 R347: - BAG[0] + zBAG[5,0] <= 0
 R348: - LAG[0] + zLAG[6,0] <= 0
 R349: - RAG[0] + zRAG[6,0] <= 0
 R350: - TAG[0] + zTAG[6,0] <= 0
 R351: - BAG[0] + zBAG[6,0] <= 0
 R352: - LAG[1] + zLAG[0,1] <= 0
 R353: - RAG[1] + zRAG[0,1] <= 0
 R354: - TAG[1] + zTAG[0,1] <= 0
 R355: - BAG[1] + zBAG[0,1] <= 0
 R356: - LAG[1] + zLAG[1,1] <= 0
 R357: - RAG[1] + zRAG[1,1] <= 0
 R358: - TAG[1] + zTAG[1,1] <= 0
 R359: - BAG[1] + zBAG[1,1] <= 0
 R360: - LAG[1] + zLAG[2,1] <= 0
 R361: - RAG[1] + zRAG[2,1] <= 0
 R362: - TAG[1] + zTAG[2,1] <= 0
 R363: - BAG[1] + zBAG[2,1] <= 0
 R364: - LAG[1] + zLAG[3,1] <= 0
 R365: - RAG[1] + zRAG[3,1] <= 0
 R366: - TAG[1] + zTAG[3,1] <= 0
 R367: - BAG[1] + zBAG[3,1] <= 0
 R368: - LAG[1] + zLAG[4,1] <= 0
 R369: - RAG[1] + zRAG[4,1] <= 0
 R370: - TAG[1] + zTAG[4,1] <= 0
 R371: - BAG[1] + zBAG[4,1] <= 0
 R372: - LAG[1] + zLAG[5,1] <= 0
 R373: - RAG[1] + zRAG[5,1] <= 0
 R374: - TAG[1] + zTAG[5,1] <= 0
 R375: - BAG[1] + zBAG[5,1] <= 0
 R376: - LAG[1] + zLAG[6,1] <= 0
 R377: - RAG[1] + zRAG[6,1] <= 0
 R378: - TAG[1] + zTAG[6,1] <= 0
 R379: - BAG[1] + zBAG[6,1] <= 0
 R380: - LAG[2] + zLAG[0,2] <= 0
 R381: - RAG[2] + zRAG[0,2] <= 0
 R382: - TAG[2] + zTAG[0,2] <= 0
 R383: - BAG[2] + zBAG[0,2] <= 0
 R384: - LAG[2] + zLAG[1,2] <= 0
 R385: - RAG[2] + zRAG[1,2] <= 0
 R386: - TAG[2] + zTAG[1,2] <= 0
 R387: - BAG[2] + zBAG[1,2] <= 0
 R388: - LAG[2] + zLAG[2,2] <= 0
 R389: - RAG[2] + zRAG[2,2] <= 0
 R390: - TAG[2] + zTAG[2,2] <= 0
 R391: - BAG[2] + zBAG[2,2] <= 0
 R392: - LAG[2] + zLAG[3,2] <= 0
 R393: - RAG[2] + zRAG[3,2] <= 0
 R394: - TAG[2] + zTAG[3,2] <= 0
 R395: - BAG[2] + zBAG[3,2] <= 0
 R396: - LAG[2] + zLAG[4,2] <= 0
 R397: - RAG[2] + zRAG[4,2] <= 0
 R398: - TAG[2] + zTAG[4,2] <= 0
 R399: - BAG[2] + zBAG[4,2] <= 0
 R400: - LAG[2] + zLAG[5,2] <= 0
 R401: - RAG[2] + zRAG[5,2] <= 0
 R402: - TAG[2] + zTAG[5,2] <= 0
 R403: - BAG[2] + zBAG[5,2] <= 0
 R404: - LAG[2] + zLAG[6,2] <= 0
 R405: - RAG[2] + zRAG[6,2] <= 0
 R406: - TAG[2] + zTAG[6,2] <= 0
 R407: - BAG[2] + zBAG[6,2] <= 0
 R408: - LAG[3] + zLAG[0,3] <= 0
 R409: - RAG[3] + zRAG[0,3] <= 0
 R410: - TAG[3] + zTAG[0,3] <= 0
 R411: - BAG[3] + zBAG[0,3] <= 0
 R412: - LAG[3] + zLAG[1,3] <= 0
 R413: - RAG[3] + zRAG[1,3] <= 0
 R414: - TAG[3] + zTAG[1,3] <= 0
 R415: - BAG[3] + zBAG[1,3] <= 0
 R416: - LAG[3] + zLAG[2,3] <= 0
 R417: - RAG[3] + zRAG[2,3] <= 0
 R418: - TAG[3] + zTAG[2,3] <= 0
 R419: - BAG[3] + zBAG[2,3] <= 0
 R420: - LAG[3] + zLAG[3,3] <= 0
 R421: - RAG[3] + zRAG[3,3] <= 0
 R422: - TAG[3] + zTAG[3,3] <= 0
 R423: - BAG[3] + zBAG[3,3] <= 0
 R424: - LAG[3] + zLAG[4,3] <= 0
 R425: - RAG[3] + zRAG[4,3] <= 0
 R426: - TAG[3] + zTAG[4,3] <= 0
 R427: - BAG[3] + zBAG[4,3] <= 0
 R428: - LAG[3] + zLAG[5,3] <= 0
 R429: - RAG[3] + zRAG[5,3] <= 0
 R430: - TAG[3] + zTAG[5,3] <= 0
 R431: - BAG[3] + zBAG[5,3] <= 0
 R432: - LAG[3] + zLAG[6,3] <= 0
 R433: - RAG[3] + zRAG[6,3] <= 0
 R434: - TAG[3] + zTAG[6,3] <= 0
 R435: - BAG[3] + zBAG[6,3] <= 0
 R436: - LAG[4] + zLAG[0,4] <= 0
 R437: - RAG[4] + zRAG[0,4] <= 0
 R438: - TAG[4] + zTAG[0,4] <= 0
 R439: - BAG[4] + zBAG[0,4] <= 0
 R440: - LAG[4] + zLAG[1,4] <= 0
 R441: - RAG[4] + zRAG[1,4] <= 0
 R442: - TAG[4] + zTAG[1,4] <= 0
 R443: - BAG[4] + zBAG[1,4] <= 0
 R444: - LAG[4] + zLAG[2,4] <= 0
 R445: - RAG[4] + zRAG[2,4] <= 0
 R446: - TAG[4] + zTAG[2,4] <= 0
 R447: - BAG[4] + zBAG[2,4] <= 0
 R448: - LAG[4] + zLAG[3,4] <= 0
 R449: - RAG[4] + zRAG[3,4] <= 0
 R450: - TAG[4] + zTAG[3,4] <= 0
 R451: - BAG[4] + zBAG[3,4] <= 0
 R452: - LAG[4] + zLAG[4,4] <= 0
 R453: - RAG[4] + zRAG[4,4] <= 0
 R454: - TAG[4] + zTAG[4,4] <= 0
 R455: - BAG[4] + zBAG[4,4] <= 0
 R456: - LAG[4] + zLAG[5,4] <= 0
 R457: - RAG[4] + zRAG[5,4] <= 0
 R458: - TAG[4] + zTAG[5,4] <= 0
 R459: - BAG[4] + zBAG[5,4] <= 0
 R460: - LAG[4] + zLAG[6,4] <= 0
 R461: - RAG[4] + zRAG[6,4] <= 0
 R462: - TAG[4] + zTAG[6,4] <= 0
 R463: - BAG[4] + zBAG[6,4] <= 0
 R464: - LAG[5] + zLAG[0,5] <= 0
 R465: - RAG[5] + zRAG[0,5] <= 0
 R466: - TAG[5] + zTAG[0,5] <= 0
 R467: - BAG[5] + zBAG[0,5] <= 0
 R468: - LAG[5] + zLAG[1,5] <= 0
 R469: - RAG[5] + zRAG[1,5] <= 0
 R470: - TAG[5] + zTAG[1,5] <= 0
 R471: - BAG[5] + zBAG[1,5] <= 0
 R472: - LAG[5] + zLAG[2,5] <= 0
 R473: - RAG[5] + zRAG[2,5] <= 0
 R474: - TAG[5] + zTAG[2,5] <= 0
 R475: - BAG[5] + zBAG[2,5] <= 0
 R476: - LAG[5] + zLAG[3,5] <= 0
 R477: - RAG[5] + zRAG[3,5] <= 0
 R478: - TAG[5] + zTAG[3,5] <= 0
 R479: - BAG[5] + zBAG[3,5] <= 0
 R480: - LAG[5] + zLAG[4,5] <= 0
 R481: - RAG[5] + zRAG[4,5] <= 0
 R482: - TAG[5] + zTAG[4,5] <= 0
 R483: - BAG[5] + zBAG[4,5] <= 0
 R484: - LAG[5] + zLAG[5,5] <= 0
 R485: - RAG[5] + zRAG[5,5] <= 0
 R486: - TAG[5] + zTAG[5,5] <= 0
 R487: - BAG[5] + zBAG[5,5] <= 0
 R488: - LAG[5] + zLAG[6,5] <= 0
 R489: - RAG[5] + zRAG[6,5] <= 0
 R490: - TAG[5] + zTAG[6,5] <= 0
 R491: - BAG[5] + zBAG[6,5] <= 0
 R492: - LAG[6] + zLAG[0,6] <= 0
 R493: - RAG[6] + zRAG[0,6] <= 0
 R494: - TAG[6] + zTAG[0,6] <= 0
 R495: - BAG[6] + zBAG[0,6] <= 0
 R496: - LAG[6] + zLAG[1,6] <= 0
 R497: - RAG[6] + zRAG[1,6] <= 0
 R498: - TAG[6] + zTAG[1,6] <= 0
 R499: - BAG[6] + zBAG[1,6] <= 0
 R500: - LAG[6] + zLAG[2,6] <= 0
 R501: - RAG[6] + zRAG[2,6] <= 0
 R502: - TAG[6] + zTAG[2,6] <= 0
 R503: - BAG[6] + zBAG[2,6] <= 0
 R504: - LAG[6] + zLAG[3,6] <= 0
 R505: - RAG[6] + zRAG[3,6] <= 0
 R506: - TAG[6] + zTAG[3,6] <= 0
 R507: - BAG[6] + zBAG[3,6] <= 0
 R508: - LAG[6] + zLAG[4,6] <= 0
 R509: - RAG[6] + zRAG[4,6] <= 0
 R510: - TAG[6] + zTAG[4,6] <= 0
 R511: - BAG[6] + zBAG[4,6] <= 0
 R512: - LAG[6] + zLAG[5,6] <= 0
 R513: - RAG[6] + zRAG[5,6] <= 0
 R514: - TAG[6] + zTAG[5,6] <= 0
 R515: - BAG[6] + zBAG[5,6] <= 0
 R516: - LAG[6] + zLAG[6,6] <= 0
 R517: - RAG[6] + zRAG[6,6] <= 0
 R518: - TAG[6] + zTAG[6,6] <= 0
 R519: - BAG[6] + zBAG[6,6] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 1000 zLAG[0,0] <= 1000
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 1000 zRAG[0,0] <= 1000
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 950 zTAG[0,0] <= 950
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 950 zBAG[0,0] <= 950
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 1000 zLAG[0,0] >= -1000
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 1000 zRAG[0,0] >= -1000
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 950 zTAG[0,0] >= -950
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 950 zBAG[0,0] >= -950
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 1000 zLAG[1,0] <= 1000
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 1000 zRAG[1,0] <= 1000
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 950 zTAG[1,0] <= 950
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 950 zBAG[1,0] <= 950
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 1000 zLAG[1,0] >= -1000
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 1000 zRAG[1,0] >= -1000
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 950 zTAG[1,0] >= -950
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 950 zBAG[1,0] >= -950
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 1000 zLAG[2,0] <= 1000
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 1000 zRAG[2,0] <= 1000
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 950 zTAG[2,0] <= 950
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 950 zBAG[2,0] <= 950
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 1000 zLAG[2,0] >= -1000
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 1000 zRAG[2,0] >= -1000
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 950 zTAG[2,0] >= -950
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 950 zBAG[2,0] >= -950
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 1000 zLAG[3,0] <= 1000
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 1000 zRAG[3,0] <= 1000
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 950 zTAG[3,0] <= 950
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 950 zBAG[3,0] <= 950
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 1000 zLAG[3,0] >= -1000
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 1000 zRAG[3,0] >= -1000
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 950 zTAG[3,0] >= -950
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 950 zBAG[3,0] >= -950
 MinsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] + 1000 zLAG[4,0] <= 1000
 MinsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] + 1000 zRAG[4,0] <= 1000
 MinsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] + 950 zTAG[4,0] <= 950
 MinsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] + 950 zBAG[4,0] <= 950
 MaxsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] - 1000 zLAG[4,0] >= -1000
 MaxsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] - 1000 zRAG[4,0] >= -1000
 MaxsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] - 950 zTAG[4,0] >= -950
 MaxsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] - 950 zBAG[4,0] >= -950
 MinsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] + 1000 zLAG[5,0] <= 1000
 MinsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] + 1000 zRAG[5,0] <= 1000
 MinsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] + 950 zTAG[5,0] <= 950
 MinsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] + 950 zBAG[5,0] <= 950
 MaxsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] - 1000 zLAG[5,0] >= -1000
 MaxsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] - 1000 zRAG[5,0] >= -1000
 MaxsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] - 950 zTAG[5,0] >= -950
 MaxsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] - 950 zBAG[5,0] >= -950
 MinsideConnectL[6]ToLAG[0]: L[6] - vLAG[0] + 1000 zLAG[6,0] <= 1000
 MinsideConnectR[6]ToRAG[0]: R[6] - vRAG[0] + 1000 zRAG[6,0] <= 1000
 MinsideConnectT[6]ToTAG[0]: T[6] - vTAG[0] + 950 zTAG[6,0] <= 950
 MinsideConnectB[6]ToBAG[0]: B[6] - vBAG[0] + 950 zBAG[6,0] <= 950
 MaxsideConnectL[6]ToLAG[0]: L[6] - vLAG[0] - 1000 zLAG[6,0] >= -1000
 MaxsideConnectR[6]ToRAG[0]: R[6] - vRAG[0] - 1000 zRAG[6,0] >= -1000
 MaxsideConnectT[6]ToTAG[0]: T[6] - vTAG[0] - 950 zTAG[6,0] >= -950
 MaxsideConnectB[6]ToBAG[0]: B[6] - vBAG[0] - 950 zBAG[6,0] >= -950
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 1000 zLAG[0,1] <= 1000
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 1000 zRAG[0,1] <= 1000
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 950 zTAG[0,1] <= 950
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 950 zBAG[0,1] <= 950
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 1000 zLAG[0,1] >= -1000
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 1000 zRAG[0,1] >= -1000
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 950 zTAG[0,1] >= -950
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 950 zBAG[0,1] >= -950
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 1000 zLAG[1,1] <= 1000
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 1000 zRAG[1,1] <= 1000
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 950 zTAG[1,1] <= 950
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 950 zBAG[1,1] <= 950
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 1000 zLAG[1,1] >= -1000
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 1000 zRAG[1,1] >= -1000
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 950 zTAG[1,1] >= -950
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 950 zBAG[1,1] >= -950
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 1000 zLAG[2,1] <= 1000
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 1000 zRAG[2,1] <= 1000
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 950 zTAG[2,1] <= 950
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 950 zBAG[2,1] <= 950
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 1000 zLAG[2,1] >= -1000
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 1000 zRAG[2,1] >= -1000
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 950 zTAG[2,1] >= -950
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 950 zBAG[2,1] >= -950
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 1000 zLAG[3,1] <= 1000
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 1000 zRAG[3,1] <= 1000
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 950 zTAG[3,1] <= 950
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 950 zBAG[3,1] <= 950
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 1000 zLAG[3,1] >= -1000
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 1000 zRAG[3,1] >= -1000
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 950 zTAG[3,1] >= -950
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 950 zBAG[3,1] >= -950
 MinsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] + 1000 zLAG[4,1] <= 1000
 MinsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] + 1000 zRAG[4,1] <= 1000
 MinsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] + 950 zTAG[4,1] <= 950
 MinsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] + 950 zBAG[4,1] <= 950
 MaxsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] - 1000 zLAG[4,1] >= -1000
 MaxsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] - 1000 zRAG[4,1] >= -1000
 MaxsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] - 950 zTAG[4,1] >= -950
 MaxsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] - 950 zBAG[4,1] >= -950
 MinsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] + 1000 zLAG[5,1] <= 1000
 MinsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] + 1000 zRAG[5,1] <= 1000
 MinsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] + 950 zTAG[5,1] <= 950
 MinsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] + 950 zBAG[5,1] <= 950
 MaxsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] - 1000 zLAG[5,1] >= -1000
 MaxsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] - 1000 zRAG[5,1] >= -1000
 MaxsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] - 950 zTAG[5,1] >= -950
 MaxsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] - 950 zBAG[5,1] >= -950
 MinsideConnectL[6]ToLAG[1]: L[6] - vLAG[1] + 1000 zLAG[6,1] <= 1000
 MinsideConnectR[6]ToRAG[1]: R[6] - vRAG[1] + 1000 zRAG[6,1] <= 1000
 MinsideConnectT[6]ToTAG[1]: T[6] - vTAG[1] + 950 zTAG[6,1] <= 950
 MinsideConnectB[6]ToBAG[1]: B[6] - vBAG[1] + 950 zBAG[6,1] <= 950
 MaxsideConnectL[6]ToLAG[1]: L[6] - vLAG[1] - 1000 zLAG[6,1] >= -1000
 MaxsideConnectR[6]ToRAG[1]: R[6] - vRAG[1] - 1000 zRAG[6,1] >= -1000
 MaxsideConnectT[6]ToTAG[1]: T[6] - vTAG[1] - 950 zTAG[6,1] >= -950
 MaxsideConnectB[6]ToBAG[1]: B[6] - vBAG[1] - 950 zBAG[6,1] >= -950
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 1000 zLAG[0,2] <= 1000
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 1000 zRAG[0,2] <= 1000
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 950 zTAG[0,2] <= 950
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 950 zBAG[0,2] <= 950
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 1000 zLAG[0,2] >= -1000
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 1000 zRAG[0,2] >= -1000
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 950 zTAG[0,2] >= -950
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 950 zBAG[0,2] >= -950
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 1000 zLAG[1,2] <= 1000
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 1000 zRAG[1,2] <= 1000
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 950 zTAG[1,2] <= 950
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 950 zBAG[1,2] <= 950
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 1000 zLAG[1,2] >= -1000
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 1000 zRAG[1,2] >= -1000
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 950 zTAG[1,2] >= -950
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 950 zBAG[1,2] >= -950
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 1000 zLAG[2,2] <= 1000
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 1000 zRAG[2,2] <= 1000
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 950 zTAG[2,2] <= 950
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 950 zBAG[2,2] <= 950
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 1000 zLAG[2,2] >= -1000
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 1000 zRAG[2,2] >= -1000
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 950 zTAG[2,2] >= -950
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 950 zBAG[2,2] >= -950
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 1000 zLAG[3,2] <= 1000
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 1000 zRAG[3,2] <= 1000
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 950 zTAG[3,2] <= 950
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 950 zBAG[3,2] <= 950
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 1000 zLAG[3,2] >= -1000
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 1000 zRAG[3,2] >= -1000
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 950 zTAG[3,2] >= -950
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 950 zBAG[3,2] >= -950
 MinsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] + 1000 zLAG[4,2] <= 1000
 MinsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] + 1000 zRAG[4,2] <= 1000
 MinsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] + 950 zTAG[4,2] <= 950
 MinsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] + 950 zBAG[4,2] <= 950
 MaxsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] - 1000 zLAG[4,2] >= -1000
 MaxsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] - 1000 zRAG[4,2] >= -1000
 MaxsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] - 950 zTAG[4,2] >= -950
 MaxsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] - 950 zBAG[4,2] >= -950
 MinsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] + 1000 zLAG[5,2] <= 1000
 MinsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] + 1000 zRAG[5,2] <= 1000
 MinsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] + 950 zTAG[5,2] <= 950
 MinsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] + 950 zBAG[5,2] <= 950
 MaxsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] - 1000 zLAG[5,2] >= -1000
 MaxsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] - 1000 zRAG[5,2] >= -1000
 MaxsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] - 950 zTAG[5,2] >= -950
 MaxsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] - 950 zBAG[5,2] >= -950
 MinsideConnectL[6]ToLAG[2]: L[6] - vLAG[2] + 1000 zLAG[6,2] <= 1000
 MinsideConnectR[6]ToRAG[2]: R[6] - vRAG[2] + 1000 zRAG[6,2] <= 1000
 MinsideConnectT[6]ToTAG[2]: T[6] - vTAG[2] + 950 zTAG[6,2] <= 950
 MinsideConnectB[6]ToBAG[2]: B[6] - vBAG[2] + 950 zBAG[6,2] <= 950
 MaxsideConnectL[6]ToLAG[2]: L[6] - vLAG[2] - 1000 zLAG[6,2] >= -1000
 MaxsideConnectR[6]ToRAG[2]: R[6] - vRAG[2] - 1000 zRAG[6,2] >= -1000
 MaxsideConnectT[6]ToTAG[2]: T[6] - vTAG[2] - 950 zTAG[6,2] >= -950
 MaxsideConnectB[6]ToBAG[2]: B[6] - vBAG[2] - 950 zBAG[6,2] >= -950
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 1000 zLAG[0,3] <= 1000
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 1000 zRAG[0,3] <= 1000
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 950 zTAG[0,3] <= 950
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 950 zBAG[0,3] <= 950
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 1000 zLAG[0,3] >= -1000
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 1000 zRAG[0,3] >= -1000
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 950 zTAG[0,3] >= -950
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 950 zBAG[0,3] >= -950
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 1000 zLAG[1,3] <= 1000
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 1000 zRAG[1,3] <= 1000
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 950 zTAG[1,3] <= 950
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 950 zBAG[1,3] <= 950
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 1000 zLAG[1,3] >= -1000
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 1000 zRAG[1,3] >= -1000
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 950 zTAG[1,3] >= -950
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 950 zBAG[1,3] >= -950
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 1000 zLAG[2,3] <= 1000
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 1000 zRAG[2,3] <= 1000
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 950 zTAG[2,3] <= 950
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 950 zBAG[2,3] <= 950
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 1000 zLAG[2,3] >= -1000
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 1000 zRAG[2,3] >= -1000
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 950 zTAG[2,3] >= -950
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 950 zBAG[2,3] >= -950
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 1000 zLAG[3,3] <= 1000
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 1000 zRAG[3,3] <= 1000
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 950 zTAG[3,3] <= 950
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 950 zBAG[3,3] <= 950
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 1000 zLAG[3,3] >= -1000
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 1000 zRAG[3,3] >= -1000
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 950 zTAG[3,3] >= -950
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 950 zBAG[3,3] >= -950
 MinsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] + 1000 zLAG[4,3] <= 1000
 MinsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] + 1000 zRAG[4,3] <= 1000
 MinsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] + 950 zTAG[4,3] <= 950
 MinsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] + 950 zBAG[4,3] <= 950
 MaxsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] - 1000 zLAG[4,3] >= -1000
 MaxsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] - 1000 zRAG[4,3] >= -1000
 MaxsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] - 950 zTAG[4,3] >= -950
 MaxsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] - 950 zBAG[4,3] >= -950
 MinsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] + 1000 zLAG[5,3] <= 1000
 MinsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] + 1000 zRAG[5,3] <= 1000
 MinsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] + 950 zTAG[5,3] <= 950
 MinsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] + 950 zBAG[5,3] <= 950
 MaxsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] - 1000 zLAG[5,3] >= -1000
 MaxsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] - 1000 zRAG[5,3] >= -1000
 MaxsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] - 950 zTAG[5,3] >= -950
 MaxsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] - 950 zBAG[5,3] >= -950
 MinsideConnectL[6]ToLAG[3]: L[6] - vLAG[3] + 1000 zLAG[6,3] <= 1000
 MinsideConnectR[6]ToRAG[3]: R[6] - vRAG[3] + 1000 zRAG[6,3] <= 1000
 MinsideConnectT[6]ToTAG[3]: T[6] - vTAG[3] + 950 zTAG[6,3] <= 950
 MinsideConnectB[6]ToBAG[3]: B[6] - vBAG[3] + 950 zBAG[6,3] <= 950
 MaxsideConnectL[6]ToLAG[3]: L[6] - vLAG[3] - 1000 zLAG[6,3] >= -1000
 MaxsideConnectR[6]ToRAG[3]: R[6] - vRAG[3] - 1000 zRAG[6,3] >= -1000
 MaxsideConnectT[6]ToTAG[3]: T[6] - vTAG[3] - 950 zTAG[6,3] >= -950
 MaxsideConnectB[6]ToBAG[3]: B[6] - vBAG[3] - 950 zBAG[6,3] >= -950
 MinsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] + 1000 zLAG[0,4] <= 1000
 MinsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] + 1000 zRAG[0,4] <= 1000
 MinsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] + 950 zTAG[0,4] <= 950
 MinsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] + 950 zBAG[0,4] <= 950
 MaxsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] - 1000 zLAG[0,4] >= -1000
 MaxsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] - 1000 zRAG[0,4] >= -1000
 MaxsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] - 950 zTAG[0,4] >= -950
 MaxsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] - 950 zBAG[0,4] >= -950
 MinsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] + 1000 zLAG[1,4] <= 1000
 MinsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] + 1000 zRAG[1,4] <= 1000
 MinsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] + 950 zTAG[1,4] <= 950
 MinsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] + 950 zBAG[1,4] <= 950
 MaxsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] - 1000 zLAG[1,4] >= -1000
 MaxsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] - 1000 zRAG[1,4] >= -1000
 MaxsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] - 950 zTAG[1,4] >= -950
 MaxsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] - 950 zBAG[1,4] >= -950
 MinsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] + 1000 zLAG[2,4] <= 1000
 MinsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] + 1000 zRAG[2,4] <= 1000
 MinsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] + 950 zTAG[2,4] <= 950
 MinsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] + 950 zBAG[2,4] <= 950
 MaxsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] - 1000 zLAG[2,4] >= -1000
 MaxsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] - 1000 zRAG[2,4] >= -1000
 MaxsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] - 950 zTAG[2,4] >= -950
 MaxsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] - 950 zBAG[2,4] >= -950
 MinsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] + 1000 zLAG[3,4] <= 1000
 MinsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] + 1000 zRAG[3,4] <= 1000
 MinsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] + 950 zTAG[3,4] <= 950
 MinsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] + 950 zBAG[3,4] <= 950
 MaxsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] - 1000 zLAG[3,4] >= -1000
 MaxsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] - 1000 zRAG[3,4] >= -1000
 MaxsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] - 950 zTAG[3,4] >= -950
 MaxsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] - 950 zBAG[3,4] >= -950
 MinsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] + 1000 zLAG[4,4] <= 1000
 MinsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] + 1000 zRAG[4,4] <= 1000
 MinsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] + 950 zTAG[4,4] <= 950
 MinsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] + 950 zBAG[4,4] <= 950
 MaxsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] - 1000 zLAG[4,4] >= -1000
 MaxsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] - 1000 zRAG[4,4] >= -1000
 MaxsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] - 950 zTAG[4,4] >= -950
 MaxsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] - 950 zBAG[4,4] >= -950
 MinsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] + 1000 zLAG[5,4] <= 1000
 MinsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] + 1000 zRAG[5,4] <= 1000
 MinsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] + 950 zTAG[5,4] <= 950
 MinsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] + 950 zBAG[5,4] <= 950
 MaxsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] - 1000 zLAG[5,4] >= -1000
 MaxsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] - 1000 zRAG[5,4] >= -1000
 MaxsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] - 950 zTAG[5,4] >= -950
 MaxsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] - 950 zBAG[5,4] >= -950
 MinsideConnectL[6]ToLAG[4]: L[6] - vLAG[4] + 1000 zLAG[6,4] <= 1000
 MinsideConnectR[6]ToRAG[4]: R[6] - vRAG[4] + 1000 zRAG[6,4] <= 1000
 MinsideConnectT[6]ToTAG[4]: T[6] - vTAG[4] + 950 zTAG[6,4] <= 950
 MinsideConnectB[6]ToBAG[4]: B[6] - vBAG[4] + 950 zBAG[6,4] <= 950
 MaxsideConnectL[6]ToLAG[4]: L[6] - vLAG[4] - 1000 zLAG[6,4] >= -1000
 MaxsideConnectR[6]ToRAG[4]: R[6] - vRAG[4] - 1000 zRAG[6,4] >= -1000
 MaxsideConnectT[6]ToTAG[4]: T[6] - vTAG[4] - 950 zTAG[6,4] >= -950
 MaxsideConnectB[6]ToBAG[4]: B[6] - vBAG[4] - 950 zBAG[6,4] >= -950
 MinsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] + 1000 zLAG[0,5] <= 1000
 MinsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] + 1000 zRAG[0,5] <= 1000
 MinsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] + 950 zTAG[0,5] <= 950
 MinsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] + 950 zBAG[0,5] <= 950
 MaxsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] - 1000 zLAG[0,5] >= -1000
 MaxsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] - 1000 zRAG[0,5] >= -1000
 MaxsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] - 950 zTAG[0,5] >= -950
 MaxsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] - 950 zBAG[0,5] >= -950
 MinsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] + 1000 zLAG[1,5] <= 1000
 MinsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] + 1000 zRAG[1,5] <= 1000
 MinsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] + 950 zTAG[1,5] <= 950
 MinsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] + 950 zBAG[1,5] <= 950
 MaxsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] - 1000 zLAG[1,5] >= -1000
 MaxsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] - 1000 zRAG[1,5] >= -1000
 MaxsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] - 950 zTAG[1,5] >= -950
 MaxsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] - 950 zBAG[1,5] >= -950
 MinsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] + 1000 zLAG[2,5] <= 1000
 MinsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] + 1000 zRAG[2,5] <= 1000
 MinsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] + 950 zTAG[2,5] <= 950
 MinsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] + 950 zBAG[2,5] <= 950
 MaxsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] - 1000 zLAG[2,5] >= -1000
 MaxsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] - 1000 zRAG[2,5] >= -1000
 MaxsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] - 950 zTAG[2,5] >= -950
 MaxsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] - 950 zBAG[2,5] >= -950
 MinsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] + 1000 zLAG[3,5] <= 1000
 MinsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] + 1000 zRAG[3,5] <= 1000
 MinsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] + 950 zTAG[3,5] <= 950
 MinsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] + 950 zBAG[3,5] <= 950
 MaxsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] - 1000 zLAG[3,5] >= -1000
 MaxsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] - 1000 zRAG[3,5] >= -1000
 MaxsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] - 950 zTAG[3,5] >= -950
 MaxsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] - 950 zBAG[3,5] >= -950
 MinsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] + 1000 zLAG[4,5] <= 1000
 MinsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] + 1000 zRAG[4,5] <= 1000
 MinsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] + 950 zTAG[4,5] <= 950
 MinsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] + 950 zBAG[4,5] <= 950
 MaxsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] - 1000 zLAG[4,5] >= -1000
 MaxsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] - 1000 zRAG[4,5] >= -1000
 MaxsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] - 950 zTAG[4,5] >= -950
 MaxsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] - 950 zBAG[4,5] >= -950
 MinsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] + 1000 zLAG[5,5] <= 1000
 MinsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] + 1000 zRAG[5,5] <= 1000
 MinsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] + 950 zTAG[5,5] <= 950
 MinsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] + 950 zBAG[5,5] <= 950
 MaxsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] - 1000 zLAG[5,5] >= -1000
 MaxsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] - 1000 zRAG[5,5] >= -1000
 MaxsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] - 950 zTAG[5,5] >= -950
 MaxsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] - 950 zBAG[5,5] >= -950
 MinsideConnectL[6]ToLAG[5]: L[6] - vLAG[5] + 1000 zLAG[6,5] <= 1000
 MinsideConnectR[6]ToRAG[5]: R[6] - vRAG[5] + 1000 zRAG[6,5] <= 1000
 MinsideConnectT[6]ToTAG[5]: T[6] - vTAG[5] + 950 zTAG[6,5] <= 950
 MinsideConnectB[6]ToBAG[5]: B[6] - vBAG[5] + 950 zBAG[6,5] <= 950
 MaxsideConnectL[6]ToLAG[5]: L[6] - vLAG[5] - 1000 zLAG[6,5] >= -1000
 MaxsideConnectR[6]ToRAG[5]: R[6] - vRAG[5] - 1000 zRAG[6,5] >= -1000
 MaxsideConnectT[6]ToTAG[5]: T[6] - vTAG[5] - 950 zTAG[6,5] >= -950
 MaxsideConnectB[6]ToBAG[5]: B[6] - vBAG[5] - 950 zBAG[6,5] >= -950
 MinsideConnectL[0]ToLAG[6]: L[0] - vLAG[6] + 1000 zLAG[0,6] <= 1000
 MinsideConnectR[0]ToRAG[6]: R[0] - vRAG[6] + 1000 zRAG[0,6] <= 1000
 MinsideConnectT[0]ToTAG[6]: T[0] - vTAG[6] + 950 zTAG[0,6] <= 950
 MinsideConnectB[0]ToBAG[6]: B[0] - vBAG[6] + 950 zBAG[0,6] <= 950
 MaxsideConnectL[0]ToLAG[6]: L[0] - vLAG[6] - 1000 zLAG[0,6] >= -1000
 MaxsideConnectR[0]ToRAG[6]: R[0] - vRAG[6] - 1000 zRAG[0,6] >= -1000
 MaxsideConnectT[0]ToTAG[6]: T[0] - vTAG[6] - 950 zTAG[0,6] >= -950
 MaxsideConnectB[0]ToBAG[6]: B[0] - vBAG[6] - 950 zBAG[0,6] >= -950
 MinsideConnectL[1]ToLAG[6]: L[1] - vLAG[6] + 1000 zLAG[1,6] <= 1000
 MinsideConnectR[1]ToRAG[6]: R[1] - vRAG[6] + 1000 zRAG[1,6] <= 1000
 MinsideConnectT[1]ToTAG[6]: T[1] - vTAG[6] + 950 zTAG[1,6] <= 950
 MinsideConnectB[1]ToBAG[6]: B[1] - vBAG[6] + 950 zBAG[1,6] <= 950
 MaxsideConnectL[1]ToLAG[6]: L[1] - vLAG[6] - 1000 zLAG[1,6] >= -1000
 MaxsideConnectR[1]ToRAG[6]: R[1] - vRAG[6] - 1000 zRAG[1,6] >= -1000
 MaxsideConnectT[1]ToTAG[6]: T[1] - vTAG[6] - 950 zTAG[1,6] >= -950
 MaxsideConnectB[1]ToBAG[6]: B[1] - vBAG[6] - 950 zBAG[1,6] >= -950
 MinsideConnectL[2]ToLAG[6]: L[2] - vLAG[6] + 1000 zLAG[2,6] <= 1000
 MinsideConnectR[2]ToRAG[6]: R[2] - vRAG[6] + 1000 zRAG[2,6] <= 1000
 MinsideConnectT[2]ToTAG[6]: T[2] - vTAG[6] + 950 zTAG[2,6] <= 950
 MinsideConnectB[2]ToBAG[6]: B[2] - vBAG[6] + 950 zBAG[2,6] <= 950
 MaxsideConnectL[2]ToLAG[6]: L[2] - vLAG[6] - 1000 zLAG[2,6] >= -1000
 MaxsideConnectR[2]ToRAG[6]: R[2] - vRAG[6] - 1000 zRAG[2,6] >= -1000
 MaxsideConnectT[2]ToTAG[6]: T[2] - vTAG[6] - 950 zTAG[2,6] >= -950
 MaxsideConnectB[2]ToBAG[6]: B[2] - vBAG[6] - 950 zBAG[2,6] >= -950
 MinsideConnectL[3]ToLAG[6]: L[3] - vLAG[6] + 1000 zLAG[3,6] <= 1000
 MinsideConnectR[3]ToRAG[6]: R[3] - vRAG[6] + 1000 zRAG[3,6] <= 1000
 MinsideConnectT[3]ToTAG[6]: T[3] - vTAG[6] + 950 zTAG[3,6] <= 950
 MinsideConnectB[3]ToBAG[6]: B[3] - vBAG[6] + 950 zBAG[3,6] <= 950
 MaxsideConnectL[3]ToLAG[6]: L[3] - vLAG[6] - 1000 zLAG[3,6] >= -1000
 MaxsideConnectR[3]ToRAG[6]: R[3] - vRAG[6] - 1000 zRAG[3,6] >= -1000
 MaxsideConnectT[3]ToTAG[6]: T[3] - vTAG[6] - 950 zTAG[3,6] >= -950
 MaxsideConnectB[3]ToBAG[6]: B[3] - vBAG[6] - 950 zBAG[3,6] >= -950
 MinsideConnectL[4]ToLAG[6]: L[4] - vLAG[6] + 1000 zLAG[4,6] <= 1000
 MinsideConnectR[4]ToRAG[6]: R[4] - vRAG[6] + 1000 zRAG[4,6] <= 1000
 MinsideConnectT[4]ToTAG[6]: T[4] - vTAG[6] + 950 zTAG[4,6] <= 950
 MinsideConnectB[4]ToBAG[6]: B[4] - vBAG[6] + 950 zBAG[4,6] <= 950
 MaxsideConnectL[4]ToLAG[6]: L[4] - vLAG[6] - 1000 zLAG[4,6] >= -1000
 MaxsideConnectR[4]ToRAG[6]: R[4] - vRAG[6] - 1000 zRAG[4,6] >= -1000
 MaxsideConnectT[4]ToTAG[6]: T[4] - vTAG[6] - 950 zTAG[4,6] >= -950
 MaxsideConnectB[4]ToBAG[6]: B[4] - vBAG[6] - 950 zBAG[4,6] >= -950
 MinsideConnectL[5]ToLAG[6]: L[5] - vLAG[6] + 1000 zLAG[5,6] <= 1000
 MinsideConnectR[5]ToRAG[6]: R[5] - vRAG[6] + 1000 zRAG[5,6] <= 1000
 MinsideConnectT[5]ToTAG[6]: T[5] - vTAG[6] + 950 zTAG[5,6] <= 950
 MinsideConnectB[5]ToBAG[6]: B[5] - vBAG[6] + 950 zBAG[5,6] <= 950
 MaxsideConnectL[5]ToLAG[6]: L[5] - vLAG[6] - 1000 zLAG[5,6] >= -1000
 MaxsideConnectR[5]ToRAG[6]: R[5] - vRAG[6] - 1000 zRAG[5,6] >= -1000
 MaxsideConnectT[5]ToTAG[6]: T[5] - vTAG[6] - 950 zTAG[5,6] >= -950
 MaxsideConnectB[5]ToBAG[6]: B[5] - vBAG[6] - 950 zBAG[5,6] >= -950
 MinsideConnectL[6]ToLAG[6]: L[6] - vLAG[6] + 1000 zLAG[6,6] <= 1000
 MinsideConnectR[6]ToRAG[6]: R[6] - vRAG[6] + 1000 zRAG[6,6] <= 1000
 MinsideConnectT[6]ToTAG[6]: T[6] - vTAG[6] + 950 zTAG[6,6] <= 950
 MinsideConnectB[6]ToBAG[6]: B[6] - vBAG[6] + 950 zBAG[6,6] <= 950
 MaxsideConnectL[6]ToLAG[6]: L[6] - vLAG[6] - 1000 zLAG[6,6] >= -1000
 MaxsideConnectR[6]ToRAG[6]: R[6] - vRAG[6] - 1000 zRAG[6,6] >= -1000
 MaxsideConnectT[6]ToTAG[6]: T[6] - vTAG[6] - 950 zTAG[6,6] >= -950
 MaxsideConnectB[6]ToBAG[6]: B[6] - vBAG[6] - 950 zBAG[6,6] >= -950
Bounds
 L[0] <= 800
 L[1] <= 600
 L[2] <= 600
 L[3] <= 600
 L[4] <= 600
 L[5] <= 600
 L[6] = 0
 200 <= R[0] <= 1000
 400 <= R[1] <= 1000
 400 <= R[2] <= 1000
 400 <= R[3] <= 1000
 400 <= R[4] <= 1000
 400 <= R[5] <= 1000
 R[6] = 1000
 T[0] <= 250
 T[1] <= 650
 T[2] <= 650
 T[3] <= 550
 T[4] <= 750
 T[5] <= 750
 T[6] <= 800
 700 <= B[0] <= 950
 300 <= B[1] <= 950
 300 <= B[2] <= 950
 400 <= B[3] <= 950
 200 <= B[4] <= 950
 200 <= B[5] <= 950
 150 <= B[6] <= 950
 H[0] = 700
 H[1] = 300
 H[2] = 300
 H[3] = 400
 H[4] = 200
 H[5] = 200
 H[6] = 150
 W[0] = 200
 W[1] = 400
 W[2] = 400
 W[3] = 400
 W[4] = 400
 W[5] = 400
 W[6] = 1000
 vLAG[0] <= 999
 vLAG[1] <= 999
 vLAG[2] <= 999
 vLAG[3] <= 999
 vLAG[4] <= 999
 vLAG[5] <= 999
 vLAG[6] <= 999
 1 <= vRAG[0] <= 1000
 1 <= vRAG[1] <= 1000
 1 <= vRAG[2] <= 1000
 1 <= vRAG[3] <= 1000
 1 <= vRAG[4] <= 1000
 1 <= vRAG[5] <= 1000
 1 <= vRAG[6] <= 1000
 vTAG[0] <= 949
 vTAG[1] <= 949
 vTAG[2] <= 949
 vTAG[3] <= 949
 vTAG[4] <= 949
 vTAG[5] <= 949
 vTAG[6] <= 949
 1 <= vBAG[0] <= 950
 1 <= vBAG[1] <= 950
 1 <= vBAG[2] <= 950
 1 <= vBAG[3] <= 950
 1 <= vBAG[4] <= 950
 1 <= vBAG[5] <= 950
 1 <= vBAG[6] <= 950
Binaries
 ABOVE[0,0] ABOVE[0,1] ABOVE[0,2] ABOVE[0,3] ABOVE[0,4] ABOVE[0,5]
 ABOVE[0,6] ABOVE[1,0] ABOVE[1,1] ABOVE[1,2] ABOVE[1,3] ABOVE[1,4]
 ABOVE[1,5] ABOVE[1,6] ABOVE[2,0] ABOVE[2,1] ABOVE[2,2] ABOVE[2,3]
 ABOVE[2,4] ABOVE[2,5] ABOVE[2,6] ABOVE[3,0] ABOVE[3,1] ABOVE[3,2]
 ABOVE[3,3] ABOVE[3,4] ABOVE[3,5] ABOVE[3,6] ABOVE[4,0] ABOVE[4,1]
 ABOVE[4,2] ABOVE[4,3] ABOVE[4,4] ABOVE[4,5] ABOVE[4,6] ABOVE[5,0]
 ABOVE[5,1] ABOVE[5,2] ABOVE[5,3] ABOVE[5,4] ABOVE[5,5] ABOVE[5,6]
 ABOVE[6,0] ABOVE[6,1] ABOVE[6,2] ABOVE[6,3] ABOVE[6,4] ABOVE[6,5]
 ABOVE[6,6] LEFT[0,0] LEFT[0,1] LEFT[0,2] LEFT[0,3] LEFT[0,4] LEFT[0,5]
 LEFT[0,6] LEFT[1,0] LEFT[1,1] LEFT[1,2] LEFT[1,3] LEFT[1,4] LEFT[1,5]
 LEFT[1,6] LEFT[2,0] LEFT[2,1] LEFT[2,2] LEFT[2,3] LEFT[2,4] LEFT[2,5]
 LEFT[2,6] LEFT[3,0] LEFT[3,1] LEFT[3,2] LEFT[3,3] LEFT[3,4] LEFT[3,5]
 LEFT[3,6] LEFT[4,0] LEFT[4,1] LEFT[4,2] LEFT[4,3] LEFT[4,4] LEFT[4,5]
 LEFT[4,6] LEFT[5,0] LEFT[5,1] LEFT[5,2] LEFT[5,3] LEFT[5,4] LEFT[5,5]
 LEFT[5,6] LEFT[6,0] LEFT[6,1] LEFT[6,2] LEFT[6,3] LEFT[6,4] LEFT[6,5]
 LEFT[6,6] LAG[0] LAG[1] LAG[2] LAG[3] LAG[4] LAG[5] LAG[6] RAG[0] RAG[1]
 RAG[2] RAG[3] RAG[4] RAG[5] RAG[6] TAG[0] TAG[1] TAG[2] TAG[3] TAG[4]
 TAG[5] TAG[6] BAG[0] BAG[1] BAG[2] BAG[3] BAG[4] BAG[5] BAG[6] zLAG[0,0]
 zLAG[0,1] zLAG[0,2] zLAG[0,3] zLAG[0,4] zLAG[0,5] zLAG[0,6] zLAG[1,0]
 zLAG[1,1] zLAG[1,2] zLAG[1,3] zLAG[1,4] zLAG[1,5] zLAG[1,6] zLAG[2,0]
 zLAG[2,1] zLAG[2,2] zLAG[2,3] zLAG[2,4] zLAG[2,5] zLAG[2,6] zLAG[3,0]
 zLAG[3,1] zLAG[3,2] zLAG[3,3] zLAG[3,4] zLAG[3,5] zLAG[3,6] zLAG[4,0]
 zLAG[4,1] zLAG[4,2] zLAG[4,3] zLAG[4,4] zLAG[4,5] zLAG[4,6] zLAG[5,0]
 zLAG[5,1] zLAG[5,2] zLAG[5,3] zLAG[5,4] zLAG[5,5] zLAG[5,6] zLAG[6,0]
 zLAG[6,1] zLAG[6,2] zLAG[6,3] zLAG[6,4] zLAG[6,5] zLAG[6,6] zRAG[0,0]
 zRAG[0,1] zRAG[0,2] zRAG[0,3] zRAG[0,4] zRAG[0,5] zRAG[0,6] zRAG[1,0]
 zRAG[1,1] zRAG[1,2] zRAG[1,3] zRAG[1,4] zRAG[1,5] zRAG[1,6] zRAG[2,0]
 zRAG[2,1] zRAG[2,2] zRAG[2,3] zRAG[2,4] zRAG[2,5] zRAG[2,6] zRAG[3,0]
 zRAG[3,1] zRAG[3,2] zRAG[3,3] zRAG[3,4] zRAG[3,5] zRAG[3,6] zRAG[4,0]
 zRAG[4,1] zRAG[4,2] zRAG[4,3] zRAG[4,4] zRAG[4,5] zRAG[4,6] zRAG[5,0]
 zRAG[5,1] zRAG[5,2] zRAG[5,3] zRAG[5,4] zRAG[5,5] zRAG[5,6] zRAG[6,0]
 zRAG[6,1] zRAG[6,2] zRAG[6,3] zRAG[6,4] zRAG[6,5] zRAG[6,6] zTAG[0,0]
 zTAG[0,1] zTAG[0,2] zTAG[0,3] zTAG[0,4] zTAG[0,5] zTAG[0,6] zTAG[1,0]
 zTAG[1,1] zTAG[1,2] zTAG[1,3] zTAG[1,4] zTAG[1,5] zTAG[1,6] zTAG[2,0]
 zTAG[2,1] zTAG[2,2] zTAG[2,3] zTAG[2,4] zTAG[2,5] zTAG[2,6] zTAG[3,0]
 zTAG[3,1] zTAG[3,2] zTAG[3,3] zTAG[3,4] zTAG[3,5] zTAG[3,6] zTAG[4,0]
 zTAG[4,1] zTAG[4,2] zTAG[4,3] zTAG[4,4] zTAG[4,5] zTAG[4,6] zTAG[5,0]
 zTAG[5,1] zTAG[5,2] zTAG[5,3] zTAG[5,4] zTAG[5,5] zTAG[5,6] zTAG[6,0]
 zTAG[6,1] zTAG[6,2] zTAG[6,3] zTAG[6,4] zTAG[6,5] zTAG[6,6] zBAG[0,0]
 zBAG[0,1] zBAG[0,2] zBAG[0,3] zBAG[0,4] zBAG[0,5] zBAG[0,6] zBAG[1,0]
 zBAG[1,1] zBAG[1,2] zBAG[1,3] zBAG[1,4] zBAG[1,5] zBAG[1,6] zBAG[2,0]
 zBAG[2,1] zBAG[2,2] zBAG[2,3] zBAG[2,4] zBAG[2,5] zBAG[2,6] zBAG[3,0]
 zBAG[3,1] zBAG[3,2] zBAG[3,3] zBAG[3,4] zBAG[3,5] zBAG[3,6] zBAG[4,0]
 zBAG[4,1] zBAG[4,2] zBAG[4,3] zBAG[4,4] zBAG[4,5] zBAG[4,6] zBAG[5,0]
 zBAG[5,1] zBAG[5,2] zBAG[5,3] zBAG[5,4] zBAG[5,5] zBAG[5,6] zBAG[6,0]
 zBAG[6,1] zBAG[6,2] zBAG[6,3] zBAG[6,4] zBAG[6,5] zBAG[6,6]
Generals
 L[0] L[1] L[2] L[3] L[4] L[5] L[6] R[0] R[1] R[2] R[3] R[4] R[5] R[6] T[0]
 T[1] T[2] T[3] T[4] T[5] T[6] B[0] B[1] B[2] B[3] B[4] B[5] B[6] H[0] H[1]
 H[2] H[3] H[4] H[5] H[6] W[0] W[1] W[2] W[3] W[4] W[5] W[6] vLAG[0]
 vLAG[1] vLAG[2] vLAG[3] vLAG[4] vLAG[5] vLAG[6] vRAG[0] vRAG[1] vRAG[2]
 vRAG[3] vRAG[4] vRAG[5] vRAG[6] vTAG[0] vTAG[1] vTAG[2] vTAG[3] vTAG[4]
 vTAG[5] vTAG[6] vBAG[0] vBAG[1] vBAG[2] vBAG[3] vBAG[4] vBAG[5] vBAG[6]
 maxX maxY
End
