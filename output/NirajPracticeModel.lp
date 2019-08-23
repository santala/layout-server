\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.001 L[4]
   + 0.001 L[5] + 0.001 L[6] + 0.001 L[7] + 0.002 R[0] + 0.002 R[1]
   + 0.002 R[2] + 0.002 R[3] + 0.002 R[4] + 0.002 R[5] + 0.002 R[6]
   + 0.002 R[7] + 0.001 T[0] + 0.001 T[1] + 0.001 T[2] + 0.001 T[3]
   + 0.001 T[4] + 0.001 T[5] + 0.001 T[6] + 0.001 T[7] + 0.002 B[0]
   + 0.002 B[1] + 0.002 B[2] + 0.002 B[3] + 0.002 B[4] + 0.002 B[5]
   + 0.002 B[6] + 0.002 B[7] - 0.001 H[0] - 0.001 H[1] - 0.001 H[2]
   - 0.001 H[3] - 0.001 H[4] - 0.001 H[5] - 0.001 H[6] - 0.001 H[7]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] - 0.001 W[4]
   - 0.001 W[5] - 0.001 W[6] - 0.001 W[7] + 0 ABOVE[0,0] + 0 ABOVE[1,1]
   + 0 ABOVE[2,2] + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 ABOVE[5,5]
   + 0 ABOVE[6,6] + 0 ABOVE[7,7] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2]
   + 0 LEFT[3,3] + 0 LEFT[4,4] + 0 LEFT[5,5] + 0 LEFT[6,6] + 0 LEFT[7,7]
   + 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + 2 LAG[5]
   + 2 LAG[6] + 2 LAG[7] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4]
   + RAG[5] + RAG[6] + RAG[7] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3]
   + 2 TAG[4] + 2 TAG[5] + 2 TAG[6] + 2 TAG[7] + BAG[0] + BAG[1] + BAG[2]
   + BAG[3] + BAG[4] + BAG[5] + BAG[6] + BAG[7]
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
 R14: - R[7] + maxX >= 0
 R15: - B[7] + maxY >= 0
 R16: 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + 2 LAG[5]
   + 2 LAG[6] + 2 LAG[7] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4]
   + RAG[5] + RAG[6] + RAG[7] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3]
   + 2 TAG[4] + 2 TAG[5] + 2 TAG[6] + 2 TAG[7] + BAG[0] + BAG[1] + BAG[2]
   + BAG[3] + BAG[4] + BAG[5] + BAG[6] + BAG[7] >= 12
 PrespecifiedXOfElement(: L[0] = 0
 PrespecifiedYOfElement(: T[0] = 0
 PrespecifiedXOfElement(: L[1] = 0
 PrespecifiedYOfElement(: T[1] = 100
 PrespecifiedXOfElement(: L[2] = 600
 PrespecifiedYOfElement(: T[2] = 500
 PrespecifiedXOfElement(: L[3] = 200
 PrespecifiedYOfElement(: T[3] = 500
 PrespecifiedXOfElement(: L[4] = 600
 PrespecifiedYOfElement(: T[4] = 100
 PrespecifiedXOfElement(: L[5] = 200
 PrespecifiedYOfElement(: T[5] = 100
 PrespecifiedXOfElement(: L[6] = 200
 PrespecifiedYOfElement(: T[6] = 300
 PrespecifiedXOfElement(: L[7] = 0
 PrespecifiedYOfElement(: T[7] = 800
 Disable_non-permitted_based_on_prespecified: ABOVE[1,0] + ABOVE[2,0]
   + ABOVE[3,0] + ABOVE[4,0] + ABOVE[5,0] + ABOVE[6,0] + 2 ABOVE[7,0]
   + ABOVE[7,1] + ABOVE[7,2] + ABOVE[7,3] + ABOVE[7,4] + ABOVE[7,5]
   + ABOVE[7,6] = 0
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
 R-L=W(7): L[7] - R[7] + W[7] = 0
 B-T=H(7): T[7] - B[7] + H[7] = 0
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
 NoOverlap(06): ABOVE[0,6] + ABOVE[6,0] + LEFT[0,6] + LEFT[6,0] >= 1
 UpperLimOfQuadrants(06): ABOVE[0,6] + ABOVE[6,0] + LEFT[0,6] + LEFT[6,0]
   <= 2
 Anti-symmetryABOVE(06): ABOVE[0,6] + ABOVE[6,0] <= 1
 Anti-symmetryLEFT(06): LEFT[0,6] + LEFT[6,0] <= 1
 NoOverlap(07): ABOVE[0,7] + ABOVE[7,0] + LEFT[0,7] + LEFT[7,0] >= 1
 UpperLimOfQuadrants(07): ABOVE[0,7] + ABOVE[7,0] + LEFT[0,7] + LEFT[7,0]
   <= 2
 Anti-symmetryABOVE(07): ABOVE[0,7] + ABOVE[7,0] <= 1
 Anti-symmetryLEFT(07): LEFT[0,7] + LEFT[7,0] <= 1
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
 NoOverlap(16): ABOVE[1,6] + ABOVE[6,1] + LEFT[1,6] + LEFT[6,1] >= 1
 UpperLimOfQuadrants(16): ABOVE[1,6] + ABOVE[6,1] + LEFT[1,6] + LEFT[6,1]
   <= 2
 Anti-symmetryABOVE(16): ABOVE[1,6] + ABOVE[6,1] <= 1
 Anti-symmetryLEFT(16): LEFT[1,6] + LEFT[6,1] <= 1
 NoOverlap(17): ABOVE[1,7] + ABOVE[7,1] + LEFT[1,7] + LEFT[7,1] >= 1
 UpperLimOfQuadrants(17): ABOVE[1,7] + ABOVE[7,1] + LEFT[1,7] + LEFT[7,1]
   <= 2
 Anti-symmetryABOVE(17): ABOVE[1,7] + ABOVE[7,1] <= 1
 Anti-symmetryLEFT(17): LEFT[1,7] + LEFT[7,1] <= 1
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
 NoOverlap(26): ABOVE[2,6] + ABOVE[6,2] + LEFT[2,6] + LEFT[6,2] >= 1
 UpperLimOfQuadrants(26): ABOVE[2,6] + ABOVE[6,2] + LEFT[2,6] + LEFT[6,2]
   <= 2
 Anti-symmetryABOVE(26): ABOVE[2,6] + ABOVE[6,2] <= 1
 Anti-symmetryLEFT(26): LEFT[2,6] + LEFT[6,2] <= 1
 NoOverlap(27): ABOVE[2,7] + ABOVE[7,2] + LEFT[2,7] + LEFT[7,2] >= 1
 UpperLimOfQuadrants(27): ABOVE[2,7] + ABOVE[7,2] + LEFT[2,7] + LEFT[7,2]
   <= 2
 Anti-symmetryABOVE(27): ABOVE[2,7] + ABOVE[7,2] <= 1
 Anti-symmetryLEFT(27): LEFT[2,7] + LEFT[7,2] <= 1
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
 NoOverlap(36): ABOVE[3,6] + ABOVE[6,3] + LEFT[3,6] + LEFT[6,3] >= 1
 UpperLimOfQuadrants(36): ABOVE[3,6] + ABOVE[6,3] + LEFT[3,6] + LEFT[6,3]
   <= 2
 Anti-symmetryABOVE(36): ABOVE[3,6] + ABOVE[6,3] <= 1
 Anti-symmetryLEFT(36): LEFT[3,6] + LEFT[6,3] <= 1
 NoOverlap(37): ABOVE[3,7] + ABOVE[7,3] + LEFT[3,7] + LEFT[7,3] >= 1
 UpperLimOfQuadrants(37): ABOVE[3,7] + ABOVE[7,3] + LEFT[3,7] + LEFT[7,3]
   <= 2
 Anti-symmetryABOVE(37): ABOVE[3,7] + ABOVE[7,3] <= 1
 Anti-symmetryLEFT(37): LEFT[3,7] + LEFT[7,3] <= 1
 NoOverlap(45): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4] >= 1
 UpperLimOfQuadrants(45): ABOVE[4,5] + ABOVE[5,4] + LEFT[4,5] + LEFT[5,4]
   <= 2
 Anti-symmetryABOVE(45): ABOVE[4,5] + ABOVE[5,4] <= 1
 Anti-symmetryLEFT(45): LEFT[4,5] + LEFT[5,4] <= 1
 NoOverlap(46): ABOVE[4,6] + ABOVE[6,4] + LEFT[4,6] + LEFT[6,4] >= 1
 UpperLimOfQuadrants(46): ABOVE[4,6] + ABOVE[6,4] + LEFT[4,6] + LEFT[6,4]
   <= 2
 Anti-symmetryABOVE(46): ABOVE[4,6] + ABOVE[6,4] <= 1
 Anti-symmetryLEFT(46): LEFT[4,6] + LEFT[6,4] <= 1
 NoOverlap(47): ABOVE[4,7] + ABOVE[7,4] + LEFT[4,7] + LEFT[7,4] >= 1
 UpperLimOfQuadrants(47): ABOVE[4,7] + ABOVE[7,4] + LEFT[4,7] + LEFT[7,4]
   <= 2
 Anti-symmetryABOVE(47): ABOVE[4,7] + ABOVE[7,4] <= 1
 Anti-symmetryLEFT(47): LEFT[4,7] + LEFT[7,4] <= 1
 NoOverlap(56): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5] >= 1
 UpperLimOfQuadrants(56): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5]
   <= 2
 Anti-symmetryABOVE(56): ABOVE[5,6] + ABOVE[6,5] <= 1
 Anti-symmetryLEFT(56): LEFT[5,6] + LEFT[6,5] <= 1
 NoOverlap(57): ABOVE[5,7] + ABOVE[7,5] + LEFT[5,7] + LEFT[7,5] >= 1
 UpperLimOfQuadrants(57): ABOVE[5,7] + ABOVE[7,5] + LEFT[5,7] + LEFT[7,5]
   <= 2
 Anti-symmetryABOVE(57): ABOVE[5,7] + ABOVE[7,5] <= 1
 Anti-symmetryLEFT(57): LEFT[5,7] + LEFT[7,5] <= 1
 NoOverlap(67): ABOVE[6,7] + ABOVE[7,6] + LEFT[6,7] + LEFT[7,6] >= 1
 UpperLimOfQuadrants(67): ABOVE[6,7] + ABOVE[7,6] + LEFT[6,7] + LEFT[7,6]
   <= 2
 Anti-symmetryABOVE(67): ABOVE[6,7] + ABOVE[7,6] <= 1
 Anti-symmetryLEFT(67): LEFT[6,7] + LEFT[7,6] <= 1
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
 0(ToLeftOf)7: - L[7] + R[0] + 1000 LEFT[0,7] <= 1000
 0(Above)7: - T[7] + B[0] + 950 ABOVE[0,7] <= 950
 0(ConverseOfToLeftOf)7: L[7] - R[0] - 1000 LEFT[0,7] <= 0
 0(ConverseOfAboveOf)7: T[7] - B[0] - 950 ABOVE[0,7] <= 0
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
 1(ToLeftOf)7: - L[7] + R[1] + 1000 LEFT[1,7] <= 1000
 1(Above)7: - T[7] + B[1] + 950 ABOVE[1,7] <= 950
 1(ConverseOfToLeftOf)7: L[7] - R[1] - 1000 LEFT[1,7] <= 0
 1(ConverseOfAboveOf)7: T[7] - B[1] - 950 ABOVE[1,7] <= 0
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
 2(ToLeftOf)7: - L[7] + R[2] + 1000 LEFT[2,7] <= 1000
 2(Above)7: - T[7] + B[2] + 950 ABOVE[2,7] <= 950
 2(ConverseOfToLeftOf)7: L[7] - R[2] - 1000 LEFT[2,7] <= 0
 2(ConverseOfAboveOf)7: T[7] - B[2] - 950 ABOVE[2,7] <= 0
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
 3(ToLeftOf)7: - L[7] + R[3] + 1000 LEFT[3,7] <= 1000
 3(Above)7: - T[7] + B[3] + 950 ABOVE[3,7] <= 950
 3(ConverseOfToLeftOf)7: L[7] - R[3] - 1000 LEFT[3,7] <= 0
 3(ConverseOfAboveOf)7: T[7] - B[3] - 950 ABOVE[3,7] <= 0
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
 4(ToLeftOf)7: - L[7] + R[4] + 1000 LEFT[4,7] <= 1000
 4(Above)7: - T[7] + B[4] + 950 ABOVE[4,7] <= 950
 4(ConverseOfToLeftOf)7: L[7] - R[4] - 1000 LEFT[4,7] <= 0
 4(ConverseOfAboveOf)7: T[7] - B[4] - 950 ABOVE[4,7] <= 0
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
 5(ToLeftOf)7: - L[7] + R[5] + 1000 LEFT[5,7] <= 1000
 5(Above)7: - T[7] + B[5] + 950 ABOVE[5,7] <= 950
 5(ConverseOfToLeftOf)7: L[7] - R[5] - 1000 LEFT[5,7] <= 0
 5(ConverseOfAboveOf)7: T[7] - B[5] - 950 ABOVE[5,7] <= 0
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
 6(ToLeftOf)7: - L[7] + R[6] + 1000 LEFT[6,7] <= 1000
 6(Above)7: - T[7] + B[6] + 950 ABOVE[6,7] <= 950
 6(ConverseOfToLeftOf)7: L[7] - R[6] - 1000 LEFT[6,7] <= 0
 6(ConverseOfAboveOf)7: T[7] - B[6] - 950 ABOVE[6,7] <= 0
 7(ToLeftOf)0: - L[0] + R[7] + 1000 LEFT[7,0] <= 1000
 7(Above)0: - T[0] + B[7] + 950 ABOVE[7,0] <= 950
 7(ConverseOfToLeftOf)0: L[0] - R[7] - 1000 LEFT[7,0] <= 0
 7(ConverseOfAboveOf)0: T[0] - B[7] - 950 ABOVE[7,0] <= 0
 7(ToLeftOf)1: - L[1] + R[7] + 1000 LEFT[7,1] <= 1000
 7(Above)1: - T[1] + B[7] + 950 ABOVE[7,1] <= 950
 7(ConverseOfToLeftOf)1: L[1] - R[7] - 1000 LEFT[7,1] <= 0
 7(ConverseOfAboveOf)1: T[1] - B[7] - 950 ABOVE[7,1] <= 0
 7(ToLeftOf)2: - L[2] + R[7] + 1000 LEFT[7,2] <= 1000
 7(Above)2: - T[2] + B[7] + 950 ABOVE[7,2] <= 950
 7(ConverseOfToLeftOf)2: L[2] - R[7] - 1000 LEFT[7,2] <= 0
 7(ConverseOfAboveOf)2: T[2] - B[7] - 950 ABOVE[7,2] <= 0
 7(ToLeftOf)3: - L[3] + R[7] + 1000 LEFT[7,3] <= 1000
 7(Above)3: - T[3] + B[7] + 950 ABOVE[7,3] <= 950
 7(ConverseOfToLeftOf)3: L[3] - R[7] - 1000 LEFT[7,3] <= 0
 7(ConverseOfAboveOf)3: T[3] - B[7] - 950 ABOVE[7,3] <= 0
 7(ToLeftOf)4: - L[4] + R[7] + 1000 LEFT[7,4] <= 1000
 7(Above)4: - T[4] + B[7] + 950 ABOVE[7,4] <= 950
 7(ConverseOfToLeftOf)4: L[4] - R[7] - 1000 LEFT[7,4] <= 0
 7(ConverseOfAboveOf)4: T[4] - B[7] - 950 ABOVE[7,4] <= 0
 7(ToLeftOf)5: - L[5] + R[7] + 1000 LEFT[7,5] <= 1000
 7(Above)5: - T[5] + B[7] + 950 ABOVE[7,5] <= 950
 7(ConverseOfToLeftOf)5: L[5] - R[7] - 1000 LEFT[7,5] <= 0
 7(ConverseOfAboveOf)5: T[5] - B[7] - 950 ABOVE[7,5] <= 0
 7(ToLeftOf)6: - L[6] + R[7] + 1000 LEFT[7,6] <= 1000
 7(Above)6: - T[6] + B[7] + 950 ABOVE[7,6] <= 950
 7(ConverseOfToLeftOf)6: L[6] - R[7] - 1000 LEFT[7,6] <= 0
 7(ConverseOfAboveOf)6: T[6] - B[7] - 950 ABOVE[7,6] <= 0
 OneLAGForElement[0]: zLAG[0,0] + zLAG[0,1] + zLAG[0,2] + zLAG[0,3]
   + zLAG[0,4] + zLAG[0,5] + zLAG[0,6] + zLAG[0,7] = 1
 OneTAGForElement[0]: zTAG[0,0] + zTAG[0,1] + zTAG[0,2] + zTAG[0,3]
   + zTAG[0,4] + zTAG[0,5] + zTAG[0,6] + zTAG[0,7] = 1
 OneBAGForElement[0]: zBAG[0,0] + zBAG[0,1] + zBAG[0,2] + zBAG[0,3]
   + zBAG[0,4] + zBAG[0,5] + zBAG[0,6] + zBAG[0,7] = 1
 OneRAGForElement[0]: zRAG[0,0] + zRAG[0,1] + zRAG[0,2] + zRAG[0,3]
   + zRAG[0,4] + zRAG[0,5] + zRAG[0,6] + zRAG[0,7] = 1
 OneLAGForElement[1]: zLAG[1,0] + zLAG[1,1] + zLAG[1,2] + zLAG[1,3]
   + zLAG[1,4] + zLAG[1,5] + zLAG[1,6] + zLAG[1,7] = 1
 OneTAGForElement[1]: zTAG[1,0] + zTAG[1,1] + zTAG[1,2] + zTAG[1,3]
   + zTAG[1,4] + zTAG[1,5] + zTAG[1,6] + zTAG[1,7] = 1
 OneBAGForElement[1]: zBAG[1,0] + zBAG[1,1] + zBAG[1,2] + zBAG[1,3]
   + zBAG[1,4] + zBAG[1,5] + zBAG[1,6] + zBAG[1,7] = 1
 OneRAGForElement[1]: zRAG[1,0] + zRAG[1,1] + zRAG[1,2] + zRAG[1,3]
   + zRAG[1,4] + zRAG[1,5] + zRAG[1,6] + zRAG[1,7] = 1
 OneLAGForElement[2]: zLAG[2,0] + zLAG[2,1] + zLAG[2,2] + zLAG[2,3]
   + zLAG[2,4] + zLAG[2,5] + zLAG[2,6] + zLAG[2,7] = 1
 OneTAGForElement[2]: zTAG[2,0] + zTAG[2,1] + zTAG[2,2] + zTAG[2,3]
   + zTAG[2,4] + zTAG[2,5] + zTAG[2,6] + zTAG[2,7] = 1
 OneBAGForElement[2]: zBAG[2,0] + zBAG[2,1] + zBAG[2,2] + zBAG[2,3]
   + zBAG[2,4] + zBAG[2,5] + zBAG[2,6] + zBAG[2,7] = 1
 OneRAGForElement[2]: zRAG[2,0] + zRAG[2,1] + zRAG[2,2] + zRAG[2,3]
   + zRAG[2,4] + zRAG[2,5] + zRAG[2,6] + zRAG[2,7] = 1
 OneLAGForElement[3]: zLAG[3,0] + zLAG[3,1] + zLAG[3,2] + zLAG[3,3]
   + zLAG[3,4] + zLAG[3,5] + zLAG[3,6] + zLAG[3,7] = 1
 OneTAGForElement[3]: zTAG[3,0] + zTAG[3,1] + zTAG[3,2] + zTAG[3,3]
   + zTAG[3,4] + zTAG[3,5] + zTAG[3,6] + zTAG[3,7] = 1
 OneBAGForElement[3]: zBAG[3,0] + zBAG[3,1] + zBAG[3,2] + zBAG[3,3]
   + zBAG[3,4] + zBAG[3,5] + zBAG[3,6] + zBAG[3,7] = 1
 OneRAGForElement[3]: zRAG[3,0] + zRAG[3,1] + zRAG[3,2] + zRAG[3,3]
   + zRAG[3,4] + zRAG[3,5] + zRAG[3,6] + zRAG[3,7] = 1
 OneLAGForElement[4]: zLAG[4,0] + zLAG[4,1] + zLAG[4,2] + zLAG[4,3]
   + zLAG[4,4] + zLAG[4,5] + zLAG[4,6] + zLAG[4,7] = 1
 OneTAGForElement[4]: zTAG[4,0] + zTAG[4,1] + zTAG[4,2] + zTAG[4,3]
   + zTAG[4,4] + zTAG[4,5] + zTAG[4,6] + zTAG[4,7] = 1
 OneBAGForElement[4]: zBAG[4,0] + zBAG[4,1] + zBAG[4,2] + zBAG[4,3]
   + zBAG[4,4] + zBAG[4,5] + zBAG[4,6] + zBAG[4,7] = 1
 OneRAGForElement[4]: zRAG[4,0] + zRAG[4,1] + zRAG[4,2] + zRAG[4,3]
   + zRAG[4,4] + zRAG[4,5] + zRAG[4,6] + zRAG[4,7] = 1
 OneLAGForElement[5]: zLAG[5,0] + zLAG[5,1] + zLAG[5,2] + zLAG[5,3]
   + zLAG[5,4] + zLAG[5,5] + zLAG[5,6] + zLAG[5,7] = 1
 OneTAGForElement[5]: zTAG[5,0] + zTAG[5,1] + zTAG[5,2] + zTAG[5,3]
   + zTAG[5,4] + zTAG[5,5] + zTAG[5,6] + zTAG[5,7] = 1
 OneBAGForElement[5]: zBAG[5,0] + zBAG[5,1] + zBAG[5,2] + zBAG[5,3]
   + zBAG[5,4] + zBAG[5,5] + zBAG[5,6] + zBAG[5,7] = 1
 OneRAGForElement[5]: zRAG[5,0] + zRAG[5,1] + zRAG[5,2] + zRAG[5,3]
   + zRAG[5,4] + zRAG[5,5] + zRAG[5,6] + zRAG[5,7] = 1
 OneLAGForElement[6]: zLAG[6,0] + zLAG[6,1] + zLAG[6,2] + zLAG[6,3]
   + zLAG[6,4] + zLAG[6,5] + zLAG[6,6] + zLAG[6,7] = 1
 OneTAGForElement[6]: zTAG[6,0] + zTAG[6,1] + zTAG[6,2] + zTAG[6,3]
   + zTAG[6,4] + zTAG[6,5] + zTAG[6,6] + zTAG[6,7] = 1
 OneBAGForElement[6]: zBAG[6,0] + zBAG[6,1] + zBAG[6,2] + zBAG[6,3]
   + zBAG[6,4] + zBAG[6,5] + zBAG[6,6] + zBAG[6,7] = 1
 OneRAGForElement[6]: zRAG[6,0] + zRAG[6,1] + zRAG[6,2] + zRAG[6,3]
   + zRAG[6,4] + zRAG[6,5] + zRAG[6,6] + zRAG[6,7] = 1
 OneLAGForElement[7]: zLAG[7,0] + zLAG[7,1] + zLAG[7,2] + zLAG[7,3]
   + zLAG[7,4] + zLAG[7,5] + zLAG[7,6] + zLAG[7,7] = 1
 OneTAGForElement[7]: zTAG[7,0] + zTAG[7,1] + zTAG[7,2] + zTAG[7,3]
   + zTAG[7,4] + zTAG[7,5] + zTAG[7,6] + zTAG[7,7] = 1
 OneBAGForElement[7]: zBAG[7,0] + zBAG[7,1] + zBAG[7,2] + zBAG[7,3]
   + zBAG[7,4] + zBAG[7,5] + zBAG[7,6] + zBAG[7,7] = 1
 OneRAGForElement[7]: zRAG[7,0] + zRAG[7,1] + zRAG[7,2] + zRAG[7,3]
   + zRAG[7,4] + zRAG[7,5] + zRAG[7,6] + zRAG[7,7] = 1
 R418: - LAG[0] + zLAG[0,0] <= 0
 R419: - RAG[0] + zRAG[0,0] <= 0
 R420: - TAG[0] + zTAG[0,0] <= 0
 R421: - BAG[0] + zBAG[0,0] <= 0
 R422: - LAG[0] + zLAG[1,0] <= 0
 R423: - RAG[0] + zRAG[1,0] <= 0
 R424: - TAG[0] + zTAG[1,0] <= 0
 R425: - BAG[0] + zBAG[1,0] <= 0
 R426: - LAG[0] + zLAG[2,0] <= 0
 R427: - RAG[0] + zRAG[2,0] <= 0
 R428: - TAG[0] + zTAG[2,0] <= 0
 R429: - BAG[0] + zBAG[2,0] <= 0
 R430: - LAG[0] + zLAG[3,0] <= 0
 R431: - RAG[0] + zRAG[3,0] <= 0
 R432: - TAG[0] + zTAG[3,0] <= 0
 R433: - BAG[0] + zBAG[3,0] <= 0
 R434: - LAG[0] + zLAG[4,0] <= 0
 R435: - RAG[0] + zRAG[4,0] <= 0
 R436: - TAG[0] + zTAG[4,0] <= 0
 R437: - BAG[0] + zBAG[4,0] <= 0
 R438: - LAG[0] + zLAG[5,0] <= 0
 R439: - RAG[0] + zRAG[5,0] <= 0
 R440: - TAG[0] + zTAG[5,0] <= 0
 R441: - BAG[0] + zBAG[5,0] <= 0
 R442: - LAG[0] + zLAG[6,0] <= 0
 R443: - RAG[0] + zRAG[6,0] <= 0
 R444: - TAG[0] + zTAG[6,0] <= 0
 R445: - BAG[0] + zBAG[6,0] <= 0
 R446: - LAG[0] + zLAG[7,0] <= 0
 R447: - RAG[0] + zRAG[7,0] <= 0
 R448: - TAG[0] + zTAG[7,0] <= 0
 R449: - BAG[0] + zBAG[7,0] <= 0
 R450: - LAG[1] + zLAG[0,1] <= 0
 R451: - RAG[1] + zRAG[0,1] <= 0
 R452: - TAG[1] + zTAG[0,1] <= 0
 R453: - BAG[1] + zBAG[0,1] <= 0
 R454: - LAG[1] + zLAG[1,1] <= 0
 R455: - RAG[1] + zRAG[1,1] <= 0
 R456: - TAG[1] + zTAG[1,1] <= 0
 R457: - BAG[1] + zBAG[1,1] <= 0
 R458: - LAG[1] + zLAG[2,1] <= 0
 R459: - RAG[1] + zRAG[2,1] <= 0
 R460: - TAG[1] + zTAG[2,1] <= 0
 R461: - BAG[1] + zBAG[2,1] <= 0
 R462: - LAG[1] + zLAG[3,1] <= 0
 R463: - RAG[1] + zRAG[3,1] <= 0
 R464: - TAG[1] + zTAG[3,1] <= 0
 R465: - BAG[1] + zBAG[3,1] <= 0
 R466: - LAG[1] + zLAG[4,1] <= 0
 R467: - RAG[1] + zRAG[4,1] <= 0
 R468: - TAG[1] + zTAG[4,1] <= 0
 R469: - BAG[1] + zBAG[4,1] <= 0
 R470: - LAG[1] + zLAG[5,1] <= 0
 R471: - RAG[1] + zRAG[5,1] <= 0
 R472: - TAG[1] + zTAG[5,1] <= 0
 R473: - BAG[1] + zBAG[5,1] <= 0
 R474: - LAG[1] + zLAG[6,1] <= 0
 R475: - RAG[1] + zRAG[6,1] <= 0
 R476: - TAG[1] + zTAG[6,1] <= 0
 R477: - BAG[1] + zBAG[6,1] <= 0
 R478: - LAG[1] + zLAG[7,1] <= 0
 R479: - RAG[1] + zRAG[7,1] <= 0
 R480: - TAG[1] + zTAG[7,1] <= 0
 R481: - BAG[1] + zBAG[7,1] <= 0
 R482: - LAG[2] + zLAG[0,2] <= 0
 R483: - RAG[2] + zRAG[0,2] <= 0
 R484: - TAG[2] + zTAG[0,2] <= 0
 R485: - BAG[2] + zBAG[0,2] <= 0
 R486: - LAG[2] + zLAG[1,2] <= 0
 R487: - RAG[2] + zRAG[1,2] <= 0
 R488: - TAG[2] + zTAG[1,2] <= 0
 R489: - BAG[2] + zBAG[1,2] <= 0
 R490: - LAG[2] + zLAG[2,2] <= 0
 R491: - RAG[2] + zRAG[2,2] <= 0
 R492: - TAG[2] + zTAG[2,2] <= 0
 R493: - BAG[2] + zBAG[2,2] <= 0
 R494: - LAG[2] + zLAG[3,2] <= 0
 R495: - RAG[2] + zRAG[3,2] <= 0
 R496: - TAG[2] + zTAG[3,2] <= 0
 R497: - BAG[2] + zBAG[3,2] <= 0
 R498: - LAG[2] + zLAG[4,2] <= 0
 R499: - RAG[2] + zRAG[4,2] <= 0
 R500: - TAG[2] + zTAG[4,2] <= 0
 R501: - BAG[2] + zBAG[4,2] <= 0
 R502: - LAG[2] + zLAG[5,2] <= 0
 R503: - RAG[2] + zRAG[5,2] <= 0
 R504: - TAG[2] + zTAG[5,2] <= 0
 R505: - BAG[2] + zBAG[5,2] <= 0
 R506: - LAG[2] + zLAG[6,2] <= 0
 R507: - RAG[2] + zRAG[6,2] <= 0
 R508: - TAG[2] + zTAG[6,2] <= 0
 R509: - BAG[2] + zBAG[6,2] <= 0
 R510: - LAG[2] + zLAG[7,2] <= 0
 R511: - RAG[2] + zRAG[7,2] <= 0
 R512: - TAG[2] + zTAG[7,2] <= 0
 R513: - BAG[2] + zBAG[7,2] <= 0
 R514: - LAG[3] + zLAG[0,3] <= 0
 R515: - RAG[3] + zRAG[0,3] <= 0
 R516: - TAG[3] + zTAG[0,3] <= 0
 R517: - BAG[3] + zBAG[0,3] <= 0
 R518: - LAG[3] + zLAG[1,3] <= 0
 R519: - RAG[3] + zRAG[1,3] <= 0
 R520: - TAG[3] + zTAG[1,3] <= 0
 R521: - BAG[3] + zBAG[1,3] <= 0
 R522: - LAG[3] + zLAG[2,3] <= 0
 R523: - RAG[3] + zRAG[2,3] <= 0
 R524: - TAG[3] + zTAG[2,3] <= 0
 R525: - BAG[3] + zBAG[2,3] <= 0
 R526: - LAG[3] + zLAG[3,3] <= 0
 R527: - RAG[3] + zRAG[3,3] <= 0
 R528: - TAG[3] + zTAG[3,3] <= 0
 R529: - BAG[3] + zBAG[3,3] <= 0
 R530: - LAG[3] + zLAG[4,3] <= 0
 R531: - RAG[3] + zRAG[4,3] <= 0
 R532: - TAG[3] + zTAG[4,3] <= 0
 R533: - BAG[3] + zBAG[4,3] <= 0
 R534: - LAG[3] + zLAG[5,3] <= 0
 R535: - RAG[3] + zRAG[5,3] <= 0
 R536: - TAG[3] + zTAG[5,3] <= 0
 R537: - BAG[3] + zBAG[5,3] <= 0
 R538: - LAG[3] + zLAG[6,3] <= 0
 R539: - RAG[3] + zRAG[6,3] <= 0
 R540: - TAG[3] + zTAG[6,3] <= 0
 R541: - BAG[3] + zBAG[6,3] <= 0
 R542: - LAG[3] + zLAG[7,3] <= 0
 R543: - RAG[3] + zRAG[7,3] <= 0
 R544: - TAG[3] + zTAG[7,3] <= 0
 R545: - BAG[3] + zBAG[7,3] <= 0
 R546: - LAG[4] + zLAG[0,4] <= 0
 R547: - RAG[4] + zRAG[0,4] <= 0
 R548: - TAG[4] + zTAG[0,4] <= 0
 R549: - BAG[4] + zBAG[0,4] <= 0
 R550: - LAG[4] + zLAG[1,4] <= 0
 R551: - RAG[4] + zRAG[1,4] <= 0
 R552: - TAG[4] + zTAG[1,4] <= 0
 R553: - BAG[4] + zBAG[1,4] <= 0
 R554: - LAG[4] + zLAG[2,4] <= 0
 R555: - RAG[4] + zRAG[2,4] <= 0
 R556: - TAG[4] + zTAG[2,4] <= 0
 R557: - BAG[4] + zBAG[2,4] <= 0
 R558: - LAG[4] + zLAG[3,4] <= 0
 R559: - RAG[4] + zRAG[3,4] <= 0
 R560: - TAG[4] + zTAG[3,4] <= 0
 R561: - BAG[4] + zBAG[3,4] <= 0
 R562: - LAG[4] + zLAG[4,4] <= 0
 R563: - RAG[4] + zRAG[4,4] <= 0
 R564: - TAG[4] + zTAG[4,4] <= 0
 R565: - BAG[4] + zBAG[4,4] <= 0
 R566: - LAG[4] + zLAG[5,4] <= 0
 R567: - RAG[4] + zRAG[5,4] <= 0
 R568: - TAG[4] + zTAG[5,4] <= 0
 R569: - BAG[4] + zBAG[5,4] <= 0
 R570: - LAG[4] + zLAG[6,4] <= 0
 R571: - RAG[4] + zRAG[6,4] <= 0
 R572: - TAG[4] + zTAG[6,4] <= 0
 R573: - BAG[4] + zBAG[6,4] <= 0
 R574: - LAG[4] + zLAG[7,4] <= 0
 R575: - RAG[4] + zRAG[7,4] <= 0
 R576: - TAG[4] + zTAG[7,4] <= 0
 R577: - BAG[4] + zBAG[7,4] <= 0
 R578: - LAG[5] + zLAG[0,5] <= 0
 R579: - RAG[5] + zRAG[0,5] <= 0
 R580: - TAG[5] + zTAG[0,5] <= 0
 R581: - BAG[5] + zBAG[0,5] <= 0
 R582: - LAG[5] + zLAG[1,5] <= 0
 R583: - RAG[5] + zRAG[1,5] <= 0
 R584: - TAG[5] + zTAG[1,5] <= 0
 R585: - BAG[5] + zBAG[1,5] <= 0
 R586: - LAG[5] + zLAG[2,5] <= 0
 R587: - RAG[5] + zRAG[2,5] <= 0
 R588: - TAG[5] + zTAG[2,5] <= 0
 R589: - BAG[5] + zBAG[2,5] <= 0
 R590: - LAG[5] + zLAG[3,5] <= 0
 R591: - RAG[5] + zRAG[3,5] <= 0
 R592: - TAG[5] + zTAG[3,5] <= 0
 R593: - BAG[5] + zBAG[3,5] <= 0
 R594: - LAG[5] + zLAG[4,5] <= 0
 R595: - RAG[5] + zRAG[4,5] <= 0
 R596: - TAG[5] + zTAG[4,5] <= 0
 R597: - BAG[5] + zBAG[4,5] <= 0
 R598: - LAG[5] + zLAG[5,5] <= 0
 R599: - RAG[5] + zRAG[5,5] <= 0
 R600: - TAG[5] + zTAG[5,5] <= 0
 R601: - BAG[5] + zBAG[5,5] <= 0
 R602: - LAG[5] + zLAG[6,5] <= 0
 R603: - RAG[5] + zRAG[6,5] <= 0
 R604: - TAG[5] + zTAG[6,5] <= 0
 R605: - BAG[5] + zBAG[6,5] <= 0
 R606: - LAG[5] + zLAG[7,5] <= 0
 R607: - RAG[5] + zRAG[7,5] <= 0
 R608: - TAG[5] + zTAG[7,5] <= 0
 R609: - BAG[5] + zBAG[7,5] <= 0
 R610: - LAG[6] + zLAG[0,6] <= 0
 R611: - RAG[6] + zRAG[0,6] <= 0
 R612: - TAG[6] + zTAG[0,6] <= 0
 R613: - BAG[6] + zBAG[0,6] <= 0
 R614: - LAG[6] + zLAG[1,6] <= 0
 R615: - RAG[6] + zRAG[1,6] <= 0
 R616: - TAG[6] + zTAG[1,6] <= 0
 R617: - BAG[6] + zBAG[1,6] <= 0
 R618: - LAG[6] + zLAG[2,6] <= 0
 R619: - RAG[6] + zRAG[2,6] <= 0
 R620: - TAG[6] + zTAG[2,6] <= 0
 R621: - BAG[6] + zBAG[2,6] <= 0
 R622: - LAG[6] + zLAG[3,6] <= 0
 R623: - RAG[6] + zRAG[3,6] <= 0
 R624: - TAG[6] + zTAG[3,6] <= 0
 R625: - BAG[6] + zBAG[3,6] <= 0
 R626: - LAG[6] + zLAG[4,6] <= 0
 R627: - RAG[6] + zRAG[4,6] <= 0
 R628: - TAG[6] + zTAG[4,6] <= 0
 R629: - BAG[6] + zBAG[4,6] <= 0
 R630: - LAG[6] + zLAG[5,6] <= 0
 R631: - RAG[6] + zRAG[5,6] <= 0
 R632: - TAG[6] + zTAG[5,6] <= 0
 R633: - BAG[6] + zBAG[5,6] <= 0
 R634: - LAG[6] + zLAG[6,6] <= 0
 R635: - RAG[6] + zRAG[6,6] <= 0
 R636: - TAG[6] + zTAG[6,6] <= 0
 R637: - BAG[6] + zBAG[6,6] <= 0
 R638: - LAG[6] + zLAG[7,6] <= 0
 R639: - RAG[6] + zRAG[7,6] <= 0
 R640: - TAG[6] + zTAG[7,6] <= 0
 R641: - BAG[6] + zBAG[7,6] <= 0
 R642: - LAG[7] + zLAG[0,7] <= 0
 R643: - RAG[7] + zRAG[0,7] <= 0
 R644: - TAG[7] + zTAG[0,7] <= 0
 R645: - BAG[7] + zBAG[0,7] <= 0
 R646: - LAG[7] + zLAG[1,7] <= 0
 R647: - RAG[7] + zRAG[1,7] <= 0
 R648: - TAG[7] + zTAG[1,7] <= 0
 R649: - BAG[7] + zBAG[1,7] <= 0
 R650: - LAG[7] + zLAG[2,7] <= 0
 R651: - RAG[7] + zRAG[2,7] <= 0
 R652: - TAG[7] + zTAG[2,7] <= 0
 R653: - BAG[7] + zBAG[2,7] <= 0
 R654: - LAG[7] + zLAG[3,7] <= 0
 R655: - RAG[7] + zRAG[3,7] <= 0
 R656: - TAG[7] + zTAG[3,7] <= 0
 R657: - BAG[7] + zBAG[3,7] <= 0
 R658: - LAG[7] + zLAG[4,7] <= 0
 R659: - RAG[7] + zRAG[4,7] <= 0
 R660: - TAG[7] + zTAG[4,7] <= 0
 R661: - BAG[7] + zBAG[4,7] <= 0
 R662: - LAG[7] + zLAG[5,7] <= 0
 R663: - RAG[7] + zRAG[5,7] <= 0
 R664: - TAG[7] + zTAG[5,7] <= 0
 R665: - BAG[7] + zBAG[5,7] <= 0
 R666: - LAG[7] + zLAG[6,7] <= 0
 R667: - RAG[7] + zRAG[6,7] <= 0
 R668: - TAG[7] + zTAG[6,7] <= 0
 R669: - BAG[7] + zBAG[6,7] <= 0
 R670: - LAG[7] + zLAG[7,7] <= 0
 R671: - RAG[7] + zRAG[7,7] <= 0
 R672: - TAG[7] + zTAG[7,7] <= 0
 R673: - BAG[7] + zBAG[7,7] <= 0
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
 MinsideConnectL[7]ToLAG[0]: L[7] - vLAG[0] + 1000 zLAG[7,0] <= 1000
 MinsideConnectR[7]ToRAG[0]: R[7] - vRAG[0] + 1000 zRAG[7,0] <= 1000
 MinsideConnectT[7]ToTAG[0]: T[7] - vTAG[0] + 950 zTAG[7,0] <= 950
 MinsideConnectB[7]ToBAG[0]: B[7] - vBAG[0] + 950 zBAG[7,0] <= 950
 MaxsideConnectL[7]ToLAG[0]: L[7] - vLAG[0] - 1000 zLAG[7,0] >= -1000
 MaxsideConnectR[7]ToRAG[0]: R[7] - vRAG[0] - 1000 zRAG[7,0] >= -1000
 MaxsideConnectT[7]ToTAG[0]: T[7] - vTAG[0] - 950 zTAG[7,0] >= -950
 MaxsideConnectB[7]ToBAG[0]: B[7] - vBAG[0] - 950 zBAG[7,0] >= -950
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
 MinsideConnectL[7]ToLAG[1]: L[7] - vLAG[1] + 1000 zLAG[7,1] <= 1000
 MinsideConnectR[7]ToRAG[1]: R[7] - vRAG[1] + 1000 zRAG[7,1] <= 1000
 MinsideConnectT[7]ToTAG[1]: T[7] - vTAG[1] + 950 zTAG[7,1] <= 950
 MinsideConnectB[7]ToBAG[1]: B[7] - vBAG[1] + 950 zBAG[7,1] <= 950
 MaxsideConnectL[7]ToLAG[1]: L[7] - vLAG[1] - 1000 zLAG[7,1] >= -1000
 MaxsideConnectR[7]ToRAG[1]: R[7] - vRAG[1] - 1000 zRAG[7,1] >= -1000
 MaxsideConnectT[7]ToTAG[1]: T[7] - vTAG[1] - 950 zTAG[7,1] >= -950
 MaxsideConnectB[7]ToBAG[1]: B[7] - vBAG[1] - 950 zBAG[7,1] >= -950
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
 MinsideConnectL[7]ToLAG[2]: L[7] - vLAG[2] + 1000 zLAG[7,2] <= 1000
 MinsideConnectR[7]ToRAG[2]: R[7] - vRAG[2] + 1000 zRAG[7,2] <= 1000
 MinsideConnectT[7]ToTAG[2]: T[7] - vTAG[2] + 950 zTAG[7,2] <= 950
 MinsideConnectB[7]ToBAG[2]: B[7] - vBAG[2] + 950 zBAG[7,2] <= 950
 MaxsideConnectL[7]ToLAG[2]: L[7] - vLAG[2] - 1000 zLAG[7,2] >= -1000
 MaxsideConnectR[7]ToRAG[2]: R[7] - vRAG[2] - 1000 zRAG[7,2] >= -1000
 MaxsideConnectT[7]ToTAG[2]: T[7] - vTAG[2] - 950 zTAG[7,2] >= -950
 MaxsideConnectB[7]ToBAG[2]: B[7] - vBAG[2] - 950 zBAG[7,2] >= -950
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
 MinsideConnectL[7]ToLAG[3]: L[7] - vLAG[3] + 1000 zLAG[7,3] <= 1000
 MinsideConnectR[7]ToRAG[3]: R[7] - vRAG[3] + 1000 zRAG[7,3] <= 1000
 MinsideConnectT[7]ToTAG[3]: T[7] - vTAG[3] + 950 zTAG[7,3] <= 950
 MinsideConnectB[7]ToBAG[3]: B[7] - vBAG[3] + 950 zBAG[7,3] <= 950
 MaxsideConnectL[7]ToLAG[3]: L[7] - vLAG[3] - 1000 zLAG[7,3] >= -1000
 MaxsideConnectR[7]ToRAG[3]: R[7] - vRAG[3] - 1000 zRAG[7,3] >= -1000
 MaxsideConnectT[7]ToTAG[3]: T[7] - vTAG[3] - 950 zTAG[7,3] >= -950
 MaxsideConnectB[7]ToBAG[3]: B[7] - vBAG[3] - 950 zBAG[7,3] >= -950
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
 MinsideConnectL[7]ToLAG[4]: L[7] - vLAG[4] + 1000 zLAG[7,4] <= 1000
 MinsideConnectR[7]ToRAG[4]: R[7] - vRAG[4] + 1000 zRAG[7,4] <= 1000
 MinsideConnectT[7]ToTAG[4]: T[7] - vTAG[4] + 950 zTAG[7,4] <= 950
 MinsideConnectB[7]ToBAG[4]: B[7] - vBAG[4] + 950 zBAG[7,4] <= 950
 MaxsideConnectL[7]ToLAG[4]: L[7] - vLAG[4] - 1000 zLAG[7,4] >= -1000
 MaxsideConnectR[7]ToRAG[4]: R[7] - vRAG[4] - 1000 zRAG[7,4] >= -1000
 MaxsideConnectT[7]ToTAG[4]: T[7] - vTAG[4] - 950 zTAG[7,4] >= -950
 MaxsideConnectB[7]ToBAG[4]: B[7] - vBAG[4] - 950 zBAG[7,4] >= -950
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
 MinsideConnectL[7]ToLAG[5]: L[7] - vLAG[5] + 1000 zLAG[7,5] <= 1000
 MinsideConnectR[7]ToRAG[5]: R[7] - vRAG[5] + 1000 zRAG[7,5] <= 1000
 MinsideConnectT[7]ToTAG[5]: T[7] - vTAG[5] + 950 zTAG[7,5] <= 950
 MinsideConnectB[7]ToBAG[5]: B[7] - vBAG[5] + 950 zBAG[7,5] <= 950
 MaxsideConnectL[7]ToLAG[5]: L[7] - vLAG[5] - 1000 zLAG[7,5] >= -1000
 MaxsideConnectR[7]ToRAG[5]: R[7] - vRAG[5] - 1000 zRAG[7,5] >= -1000
 MaxsideConnectT[7]ToTAG[5]: T[7] - vTAG[5] - 950 zTAG[7,5] >= -950
 MaxsideConnectB[7]ToBAG[5]: B[7] - vBAG[5] - 950 zBAG[7,5] >= -950
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
 MinsideConnectL[7]ToLAG[6]: L[7] - vLAG[6] + 1000 zLAG[7,6] <= 1000
 MinsideConnectR[7]ToRAG[6]: R[7] - vRAG[6] + 1000 zRAG[7,6] <= 1000
 MinsideConnectT[7]ToTAG[6]: T[7] - vTAG[6] + 950 zTAG[7,6] <= 950
 MinsideConnectB[7]ToBAG[6]: B[7] - vBAG[6] + 950 zBAG[7,6] <= 950
 MaxsideConnectL[7]ToLAG[6]: L[7] - vLAG[6] - 1000 zLAG[7,6] >= -1000
 MaxsideConnectR[7]ToRAG[6]: R[7] - vRAG[6] - 1000 zRAG[7,6] >= -1000
 MaxsideConnectT[7]ToTAG[6]: T[7] - vTAG[6] - 950 zTAG[7,6] >= -950
 MaxsideConnectB[7]ToBAG[6]: B[7] - vBAG[6] - 950 zBAG[7,6] >= -950
 MinsideConnectL[0]ToLAG[7]: L[0] - vLAG[7] + 1000 zLAG[0,7] <= 1000
 MinsideConnectR[0]ToRAG[7]: R[0] - vRAG[7] + 1000 zRAG[0,7] <= 1000
 MinsideConnectT[0]ToTAG[7]: T[0] - vTAG[7] + 950 zTAG[0,7] <= 950
 MinsideConnectB[0]ToBAG[7]: B[0] - vBAG[7] + 950 zBAG[0,7] <= 950
 MaxsideConnectL[0]ToLAG[7]: L[0] - vLAG[7] - 1000 zLAG[0,7] >= -1000
 MaxsideConnectR[0]ToRAG[7]: R[0] - vRAG[7] - 1000 zRAG[0,7] >= -1000
 MaxsideConnectT[0]ToTAG[7]: T[0] - vTAG[7] - 950 zTAG[0,7] >= -950
 MaxsideConnectB[0]ToBAG[7]: B[0] - vBAG[7] - 950 zBAG[0,7] >= -950
 MinsideConnectL[1]ToLAG[7]: L[1] - vLAG[7] + 1000 zLAG[1,7] <= 1000
 MinsideConnectR[1]ToRAG[7]: R[1] - vRAG[7] + 1000 zRAG[1,7] <= 1000
 MinsideConnectT[1]ToTAG[7]: T[1] - vTAG[7] + 950 zTAG[1,7] <= 950
 MinsideConnectB[1]ToBAG[7]: B[1] - vBAG[7] + 950 zBAG[1,7] <= 950
 MaxsideConnectL[1]ToLAG[7]: L[1] - vLAG[7] - 1000 zLAG[1,7] >= -1000
 MaxsideConnectR[1]ToRAG[7]: R[1] - vRAG[7] - 1000 zRAG[1,7] >= -1000
 MaxsideConnectT[1]ToTAG[7]: T[1] - vTAG[7] - 950 zTAG[1,7] >= -950
 MaxsideConnectB[1]ToBAG[7]: B[1] - vBAG[7] - 950 zBAG[1,7] >= -950
 MinsideConnectL[2]ToLAG[7]: L[2] - vLAG[7] + 1000 zLAG[2,7] <= 1000
 MinsideConnectR[2]ToRAG[7]: R[2] - vRAG[7] + 1000 zRAG[2,7] <= 1000
 MinsideConnectT[2]ToTAG[7]: T[2] - vTAG[7] + 950 zTAG[2,7] <= 950
 MinsideConnectB[2]ToBAG[7]: B[2] - vBAG[7] + 950 zBAG[2,7] <= 950
 MaxsideConnectL[2]ToLAG[7]: L[2] - vLAG[7] - 1000 zLAG[2,7] >= -1000
 MaxsideConnectR[2]ToRAG[7]: R[2] - vRAG[7] - 1000 zRAG[2,7] >= -1000
 MaxsideConnectT[2]ToTAG[7]: T[2] - vTAG[7] - 950 zTAG[2,7] >= -950
 MaxsideConnectB[2]ToBAG[7]: B[2] - vBAG[7] - 950 zBAG[2,7] >= -950
 MinsideConnectL[3]ToLAG[7]: L[3] - vLAG[7] + 1000 zLAG[3,7] <= 1000
 MinsideConnectR[3]ToRAG[7]: R[3] - vRAG[7] + 1000 zRAG[3,7] <= 1000
 MinsideConnectT[3]ToTAG[7]: T[3] - vTAG[7] + 950 zTAG[3,7] <= 950
 MinsideConnectB[3]ToBAG[7]: B[3] - vBAG[7] + 950 zBAG[3,7] <= 950
 MaxsideConnectL[3]ToLAG[7]: L[3] - vLAG[7] - 1000 zLAG[3,7] >= -1000
 MaxsideConnectR[3]ToRAG[7]: R[3] - vRAG[7] - 1000 zRAG[3,7] >= -1000
 MaxsideConnectT[3]ToTAG[7]: T[3] - vTAG[7] - 950 zTAG[3,7] >= -950
 MaxsideConnectB[3]ToBAG[7]: B[3] - vBAG[7] - 950 zBAG[3,7] >= -950
 MinsideConnectL[4]ToLAG[7]: L[4] - vLAG[7] + 1000 zLAG[4,7] <= 1000
 MinsideConnectR[4]ToRAG[7]: R[4] - vRAG[7] + 1000 zRAG[4,7] <= 1000
 MinsideConnectT[4]ToTAG[7]: T[4] - vTAG[7] + 950 zTAG[4,7] <= 950
 MinsideConnectB[4]ToBAG[7]: B[4] - vBAG[7] + 950 zBAG[4,7] <= 950
 MaxsideConnectL[4]ToLAG[7]: L[4] - vLAG[7] - 1000 zLAG[4,7] >= -1000
 MaxsideConnectR[4]ToRAG[7]: R[4] - vRAG[7] - 1000 zRAG[4,7] >= -1000
 MaxsideConnectT[4]ToTAG[7]: T[4] - vTAG[7] - 950 zTAG[4,7] >= -950
 MaxsideConnectB[4]ToBAG[7]: B[4] - vBAG[7] - 950 zBAG[4,7] >= -950
 MinsideConnectL[5]ToLAG[7]: L[5] - vLAG[7] + 1000 zLAG[5,7] <= 1000
 MinsideConnectR[5]ToRAG[7]: R[5] - vRAG[7] + 1000 zRAG[5,7] <= 1000
 MinsideConnectT[5]ToTAG[7]: T[5] - vTAG[7] + 950 zTAG[5,7] <= 950
 MinsideConnectB[5]ToBAG[7]: B[5] - vBAG[7] + 950 zBAG[5,7] <= 950
 MaxsideConnectL[5]ToLAG[7]: L[5] - vLAG[7] - 1000 zLAG[5,7] >= -1000
 MaxsideConnectR[5]ToRAG[7]: R[5] - vRAG[7] - 1000 zRAG[5,7] >= -1000
 MaxsideConnectT[5]ToTAG[7]: T[5] - vTAG[7] - 950 zTAG[5,7] >= -950
 MaxsideConnectB[5]ToBAG[7]: B[5] - vBAG[7] - 950 zBAG[5,7] >= -950
 MinsideConnectL[6]ToLAG[7]: L[6] - vLAG[7] + 1000 zLAG[6,7] <= 1000
 MinsideConnectR[6]ToRAG[7]: R[6] - vRAG[7] + 1000 zRAG[6,7] <= 1000
 MinsideConnectT[6]ToTAG[7]: T[6] - vTAG[7] + 950 zTAG[6,7] <= 950
 MinsideConnectB[6]ToBAG[7]: B[6] - vBAG[7] + 950 zBAG[6,7] <= 950
 MaxsideConnectL[6]ToLAG[7]: L[6] - vLAG[7] - 1000 zLAG[6,7] >= -1000
 MaxsideConnectR[6]ToRAG[7]: R[6] - vRAG[7] - 1000 zRAG[6,7] >= -1000
 MaxsideConnectT[6]ToTAG[7]: T[6] - vTAG[7] - 950 zTAG[6,7] >= -950
 MaxsideConnectB[6]ToBAG[7]: B[6] - vBAG[7] - 950 zBAG[6,7] >= -950
 MinsideConnectL[7]ToLAG[7]: L[7] - vLAG[7] + 1000 zLAG[7,7] <= 1000
 MinsideConnectR[7]ToRAG[7]: R[7] - vRAG[7] + 1000 zRAG[7,7] <= 1000
 MinsideConnectT[7]ToTAG[7]: T[7] - vTAG[7] + 950 zTAG[7,7] <= 950
 MinsideConnectB[7]ToBAG[7]: B[7] - vBAG[7] + 950 zBAG[7,7] <= 950
 MaxsideConnectL[7]ToLAG[7]: L[7] - vLAG[7] - 1000 zLAG[7,7] >= -1000
 MaxsideConnectR[7]ToRAG[7]: R[7] - vRAG[7] - 1000 zRAG[7,7] >= -1000
 MaxsideConnectT[7]ToTAG[7]: T[7] - vTAG[7] - 950 zTAG[7,7] >= -950
 MaxsideConnectB[7]ToBAG[7]: B[7] - vBAG[7] - 950 zBAG[7,7] >= -950
Bounds
 L[0] = 0
 L[1] <= 800
 L[2] <= 600
 L[3] <= 600
 L[4] <= 600
 L[5] <= 600
 L[6] <= 600
 L[7] = 0
 R[0] = 1000
 200 <= R[1] <= 1000
 400 <= R[2] <= 1000
 400 <= R[3] <= 1000
 400 <= R[4] <= 1000
 400 <= R[5] <= 1000
 400 <= R[6] <= 1000
 R[7] = 1000
 T[0] <= 850
 T[1] <= 250
 T[2] <= 650
 T[3] <= 650
 T[4] <= 550
 T[5] <= 750
 T[6] <= 750
 T[7] <= 800
 100 <= B[0] <= 950
 700 <= B[1] <= 950
 300 <= B[2] <= 950
 300 <= B[3] <= 950
 400 <= B[4] <= 950
 200 <= B[5] <= 950
 200 <= B[6] <= 950
 150 <= B[7] <= 950
 H[0] = 100
 H[1] = 700
 H[2] = 300
 H[3] = 300
 H[4] = 400
 H[5] = 200
 H[6] = 200
 H[7] = 150
 W[0] = 1000
 W[1] = 200
 W[2] = 400
 W[3] = 400
 W[4] = 400
 W[5] = 400
 W[6] = 400
 W[7] = 1000
 vLAG[0] <= 999
 vLAG[1] <= 999
 vLAG[2] <= 999
 vLAG[3] <= 999
 vLAG[4] <= 999
 vLAG[5] <= 999
 vLAG[6] <= 999
 vLAG[7] <= 999
 1 <= vRAG[0] <= 1000
 1 <= vRAG[1] <= 1000
 1 <= vRAG[2] <= 1000
 1 <= vRAG[3] <= 1000
 1 <= vRAG[4] <= 1000
 1 <= vRAG[5] <= 1000
 1 <= vRAG[6] <= 1000
 1 <= vRAG[7] <= 1000
 vTAG[0] <= 949
 vTAG[1] <= 949
 vTAG[2] <= 949
 vTAG[3] <= 949
 vTAG[4] <= 949
 vTAG[5] <= 949
 vTAG[6] <= 949
 vTAG[7] <= 949
 1 <= vBAG[0] <= 950
 1 <= vBAG[1] <= 950
 1 <= vBAG[2] <= 950
 1 <= vBAG[3] <= 950
 1 <= vBAG[4] <= 950
 1 <= vBAG[5] <= 950
 1 <= vBAG[6] <= 950
 1 <= vBAG[7] <= 950
Binaries
 ABOVE[0,0] ABOVE[0,1] ABOVE[0,2] ABOVE[0,3] ABOVE[0,4] ABOVE[0,5]
 ABOVE[0,6] ABOVE[0,7] ABOVE[1,0] ABOVE[1,1] ABOVE[1,2] ABOVE[1,3]
 ABOVE[1,4] ABOVE[1,5] ABOVE[1,6] ABOVE[1,7] ABOVE[2,0] ABOVE[2,1]
 ABOVE[2,2] ABOVE[2,3] ABOVE[2,4] ABOVE[2,5] ABOVE[2,6] ABOVE[2,7]
 ABOVE[3,0] ABOVE[3,1] ABOVE[3,2] ABOVE[3,3] ABOVE[3,4] ABOVE[3,5]
 ABOVE[3,6] ABOVE[3,7] ABOVE[4,0] ABOVE[4,1] ABOVE[4,2] ABOVE[4,3]
 ABOVE[4,4] ABOVE[4,5] ABOVE[4,6] ABOVE[4,7] ABOVE[5,0] ABOVE[5,1]
 ABOVE[5,2] ABOVE[5,3] ABOVE[5,4] ABOVE[5,5] ABOVE[5,6] ABOVE[5,7]
 ABOVE[6,0] ABOVE[6,1] ABOVE[6,2] ABOVE[6,3] ABOVE[6,4] ABOVE[6,5]
 ABOVE[6,6] ABOVE[6,7] ABOVE[7,0] ABOVE[7,1] ABOVE[7,2] ABOVE[7,3]
 ABOVE[7,4] ABOVE[7,5] ABOVE[7,6] ABOVE[7,7] LEFT[0,0] LEFT[0,1] LEFT[0,2]
 LEFT[0,3] LEFT[0,4] LEFT[0,5] LEFT[0,6] LEFT[0,7] LEFT[1,0] LEFT[1,1]
 LEFT[1,2] LEFT[1,3] LEFT[1,4] LEFT[1,5] LEFT[1,6] LEFT[1,7] LEFT[2,0]
 LEFT[2,1] LEFT[2,2] LEFT[2,3] LEFT[2,4] LEFT[2,5] LEFT[2,6] LEFT[2,7]
 LEFT[3,0] LEFT[3,1] LEFT[3,2] LEFT[3,3] LEFT[3,4] LEFT[3,5] LEFT[3,6]
 LEFT[3,7] LEFT[4,0] LEFT[4,1] LEFT[4,2] LEFT[4,3] LEFT[4,4] LEFT[4,5]
 LEFT[4,6] LEFT[4,7] LEFT[5,0] LEFT[5,1] LEFT[5,2] LEFT[5,3] LEFT[5,4]
 LEFT[5,5] LEFT[5,6] LEFT[5,7] LEFT[6,0] LEFT[6,1] LEFT[6,2] LEFT[6,3]
 LEFT[6,4] LEFT[6,5] LEFT[6,6] LEFT[6,7] LEFT[7,0] LEFT[7,1] LEFT[7,2]
 LEFT[7,3] LEFT[7,4] LEFT[7,5] LEFT[7,6] LEFT[7,7] LAG[0] LAG[1] LAG[2]
 LAG[3] LAG[4] LAG[5] LAG[6] LAG[7] RAG[0] RAG[1] RAG[2] RAG[3] RAG[4]
 RAG[5] RAG[6] RAG[7] TAG[0] TAG[1] TAG[2] TAG[3] TAG[4] TAG[5] TAG[6]
 TAG[7] BAG[0] BAG[1] BAG[2] BAG[3] BAG[4] BAG[5] BAG[6] BAG[7] zLAG[0,0]
 zLAG[0,1] zLAG[0,2] zLAG[0,3] zLAG[0,4] zLAG[0,5] zLAG[0,6] zLAG[0,7]
 zLAG[1,0] zLAG[1,1] zLAG[1,2] zLAG[1,3] zLAG[1,4] zLAG[1,5] zLAG[1,6]
 zLAG[1,7] zLAG[2,0] zLAG[2,1] zLAG[2,2] zLAG[2,3] zLAG[2,4] zLAG[2,5]
 zLAG[2,6] zLAG[2,7] zLAG[3,0] zLAG[3,1] zLAG[3,2] zLAG[3,3] zLAG[3,4]
 zLAG[3,5] zLAG[3,6] zLAG[3,7] zLAG[4,0] zLAG[4,1] zLAG[4,2] zLAG[4,3]
 zLAG[4,4] zLAG[4,5] zLAG[4,6] zLAG[4,7] zLAG[5,0] zLAG[5,1] zLAG[5,2]
 zLAG[5,3] zLAG[5,4] zLAG[5,5] zLAG[5,6] zLAG[5,7] zLAG[6,0] zLAG[6,1]
 zLAG[6,2] zLAG[6,3] zLAG[6,4] zLAG[6,5] zLAG[6,6] zLAG[6,7] zLAG[7,0]
 zLAG[7,1] zLAG[7,2] zLAG[7,3] zLAG[7,4] zLAG[7,5] zLAG[7,6] zLAG[7,7]
 zRAG[0,0] zRAG[0,1] zRAG[0,2] zRAG[0,3] zRAG[0,4] zRAG[0,5] zRAG[0,6]
 zRAG[0,7] zRAG[1,0] zRAG[1,1] zRAG[1,2] zRAG[1,3] zRAG[1,4] zRAG[1,5]
 zRAG[1,6] zRAG[1,7] zRAG[2,0] zRAG[2,1] zRAG[2,2] zRAG[2,3] zRAG[2,4]
 zRAG[2,5] zRAG[2,6] zRAG[2,7] zRAG[3,0] zRAG[3,1] zRAG[3,2] zRAG[3,3]
 zRAG[3,4] zRAG[3,5] zRAG[3,6] zRAG[3,7] zRAG[4,0] zRAG[4,1] zRAG[4,2]
 zRAG[4,3] zRAG[4,4] zRAG[4,5] zRAG[4,6] zRAG[4,7] zRAG[5,0] zRAG[5,1]
 zRAG[5,2] zRAG[5,3] zRAG[5,4] zRAG[5,5] zRAG[5,6] zRAG[5,7] zRAG[6,0]
 zRAG[6,1] zRAG[6,2] zRAG[6,3] zRAG[6,4] zRAG[6,5] zRAG[6,6] zRAG[6,7]
 zRAG[7,0] zRAG[7,1] zRAG[7,2] zRAG[7,3] zRAG[7,4] zRAG[7,5] zRAG[7,6]
 zRAG[7,7] zTAG[0,0] zTAG[0,1] zTAG[0,2] zTAG[0,3] zTAG[0,4] zTAG[0,5]
 zTAG[0,6] zTAG[0,7] zTAG[1,0] zTAG[1,1] zTAG[1,2] zTAG[1,3] zTAG[1,4]
 zTAG[1,5] zTAG[1,6] zTAG[1,7] zTAG[2,0] zTAG[2,1] zTAG[2,2] zTAG[2,3]
 zTAG[2,4] zTAG[2,5] zTAG[2,6] zTAG[2,7] zTAG[3,0] zTAG[3,1] zTAG[3,2]
 zTAG[3,3] zTAG[3,4] zTAG[3,5] zTAG[3,6] zTAG[3,7] zTAG[4,0] zTAG[4,1]
 zTAG[4,2] zTAG[4,3] zTAG[4,4] zTAG[4,5] zTAG[4,6] zTAG[4,7] zTAG[5,0]
 zTAG[5,1] zTAG[5,2] zTAG[5,3] zTAG[5,4] zTAG[5,5] zTAG[5,6] zTAG[5,7]
 zTAG[6,0] zTAG[6,1] zTAG[6,2] zTAG[6,3] zTAG[6,4] zTAG[6,5] zTAG[6,6]
 zTAG[6,7] zTAG[7,0] zTAG[7,1] zTAG[7,2] zTAG[7,3] zTAG[7,4] zTAG[7,5]
 zTAG[7,6] zTAG[7,7] zBAG[0,0] zBAG[0,1] zBAG[0,2] zBAG[0,3] zBAG[0,4]
 zBAG[0,5] zBAG[0,6] zBAG[0,7] zBAG[1,0] zBAG[1,1] zBAG[1,2] zBAG[1,3]
 zBAG[1,4] zBAG[1,5] zBAG[1,6] zBAG[1,7] zBAG[2,0] zBAG[2,1] zBAG[2,2]
 zBAG[2,3] zBAG[2,4] zBAG[2,5] zBAG[2,6] zBAG[2,7] zBAG[3,0] zBAG[3,1]
 zBAG[3,2] zBAG[3,3] zBAG[3,4] zBAG[3,5] zBAG[3,6] zBAG[3,7] zBAG[4,0]
 zBAG[4,1] zBAG[4,2] zBAG[4,3] zBAG[4,4] zBAG[4,5] zBAG[4,6] zBAG[4,7]
 zBAG[5,0] zBAG[5,1] zBAG[5,2] zBAG[5,3] zBAG[5,4] zBAG[5,5] zBAG[5,6]
 zBAG[5,7] zBAG[6,0] zBAG[6,1] zBAG[6,2] zBAG[6,3] zBAG[6,4] zBAG[6,5]
 zBAG[6,6] zBAG[6,7] zBAG[7,0] zBAG[7,1] zBAG[7,2] zBAG[7,3] zBAG[7,4]
 zBAG[7,5] zBAG[7,6] zBAG[7,7]
Generals
 L[0] L[1] L[2] L[3] L[4] L[5] L[6] L[7] R[0] R[1] R[2] R[3] R[4] R[5] R[6]
 R[7] T[0] T[1] T[2] T[3] T[4] T[5] T[6] T[7] B[0] B[1] B[2] B[3] B[4] B[5]
 B[6] B[7] H[0] H[1] H[2] H[3] H[4] H[5] H[6] H[7] W[0] W[1] W[2] W[3] W[4]
 W[5] W[6] W[7] vLAG[0] vLAG[1] vLAG[2] vLAG[3] vLAG[4] vLAG[5] vLAG[6]
 vLAG[7] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vRAG[5] vRAG[6] vRAG[7]
 vTAG[0] vTAG[1] vTAG[2] vTAG[3] vTAG[4] vTAG[5] vTAG[6] vTAG[7] vBAG[0]
 vBAG[1] vBAG[2] vBAG[3] vBAG[4] vBAG[5] vBAG[6] vBAG[7] maxX maxY
End
