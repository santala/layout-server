\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.001 L[4]
   + 0.002 R[0] + 0.002 R[1] + 0.002 R[2] + 0.002 R[3] + 0.002 R[4]
   + 0.001 T[0] + 0.001 T[1] + 0.001 T[2] + 0.001 T[3] + 0.001 T[4]
   + 0.002 B[0] + 0.002 B[1] + 0.002 B[2] + 0.002 B[3] + 0.002 B[4]
   - 0.001 H[0] - 0.001 H[1] - 0.001 H[2] - 0.001 H[3] - 0.001 H[4]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] - 0.001 W[4]
   + 0 LG[0] + 0 TG[0] + 0 ABOVE[0,0] + 0 ABOVE[1,1] + 0 ABOVE[2,2]
   + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2]
   + 0 LEFT[3,3] + 0 LEFT[4,4] + 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3]
   + 2 LAG[4] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + 2 TAG[0]
   + 2 TAG[1] + 2 TAG[2] + 2 TAG[3] + 2 TAG[4] + BAG[0] + BAG[1] + BAG[2]
   + BAG[3] + BAG[4]
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
 R10: 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + 2 LAG[4] + RAG[0]
   + RAG[1] + RAG[2] + RAG[3] + RAG[4] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2]
   + 2 TAG[3] + 2 TAG[4] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4]
   >= 10
 PrespecifiedXOfElement(0): L[0] = 0
 PrespecifiedYOfElement(0): T[0] = 0
 SnapWToGrid(0): - W[0] + 8 WG[0] = 0
 SnapHToGrid(0): - H[0] + 8 HG[0] = 0
 SnapXToGrid(1): - L[1] + 8 LG[1] = 0
 SnapYToGrid(1): - T[1] + 8 TG[1] = 0
 SnapWToGrid(1): - W[1] + 8 WG[1] = 0
 SnapHToGrid(1): - H[1] + 8 HG[1] = 0
 SnapXToGrid(2): - L[2] + 8 LG[2] = 0
 SnapYToGrid(2): - T[2] + 8 TG[2] = 0
 SnapWToGrid(2): - W[2] + 8 WG[2] = 0
 SnapHToGrid(2): - H[2] + 8 HG[2] = 0
 SnapXToGrid(3): - L[3] + 8 LG[3] = 0
 SnapYToGrid(3): - T[3] + 8 TG[3] = 0
 SnapWToGrid(3): - W[3] + 8 WG[3] = 0
 SnapHToGrid(3): - H[3] + 8 HG[3] = 0
 SnapXToGrid(4): - L[4] + 8 LG[4] = 0
 SnapYToGrid(4): - T[4] + 8 TG[4] = 0
 SnapWToGrid(4): - W[4] + 8 WG[4] = 0
 SnapHToGrid(4): - H[4] + 8 HG[4] = 0
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
 NoOverlap(34): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3] >= 1
 UpperLimOfQuadrants(34): ABOVE[3,4] + ABOVE[4,3] + LEFT[3,4] + LEFT[4,3]
   <= 2
 Anti-symmetryABOVE(34): ABOVE[3,4] + ABOVE[4,3] <= 1
 Anti-symmetryLEFT(34): LEFT[3,4] + LEFT[4,3] <= 1
 0(ToLeftOf)1: - L[1] + R[0] + 1461 LEFT[0,1] <= 1429
 0(Above)1: - T[1] + B[0] + 996 ABOVE[0,1] <= 964
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 1461 LEFT[0,1] <= 32
 0(ConverseOfAboveOf)1: T[1] - B[0] - 996 ABOVE[0,1] <= 32
 0(ToLeftOf)2: - L[2] + R[0] + 1461 LEFT[0,2] <= 1429
 0(Above)2: - T[2] + B[0] + 996 ABOVE[0,2] <= 964
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 1461 LEFT[0,2] <= 32
 0(ConverseOfAboveOf)2: T[2] - B[0] - 996 ABOVE[0,2] <= 32
 0(ToLeftOf)3: - L[3] + R[0] + 1461 LEFT[0,3] <= 1429
 0(Above)3: - T[3] + B[0] + 996 ABOVE[0,3] <= 964
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 1461 LEFT[0,3] <= 32
 0(ConverseOfAboveOf)3: T[3] - B[0] - 996 ABOVE[0,3] <= 32
 0(ToLeftOf)4: - L[4] + R[0] + 1461 LEFT[0,4] <= 1429
 0(Above)4: - T[4] + B[0] + 996 ABOVE[0,4] <= 964
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 1461 LEFT[0,4] <= 32
 0(ConverseOfAboveOf)4: T[4] - B[0] - 996 ABOVE[0,4] <= 32
 1(ToLeftOf)0: - L[0] + R[1] + 1461 LEFT[1,0] <= 1429
 1(Above)0: - T[0] + B[1] + 996 ABOVE[1,0] <= 964
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 1461 LEFT[1,0] <= 32
 1(ConverseOfAboveOf)0: T[0] - B[1] - 996 ABOVE[1,0] <= 32
 1(ToLeftOf)2: - L[2] + R[1] + 1461 LEFT[1,2] <= 1429
 1(Above)2: - T[2] + B[1] + 996 ABOVE[1,2] <= 964
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 1461 LEFT[1,2] <= 32
 1(ConverseOfAboveOf)2: T[2] - B[1] - 996 ABOVE[1,2] <= 32
 1(ToLeftOf)3: - L[3] + R[1] + 1461 LEFT[1,3] <= 1429
 1(Above)3: - T[3] + B[1] + 996 ABOVE[1,3] <= 964
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 1461 LEFT[1,3] <= 32
 1(ConverseOfAboveOf)3: T[3] - B[1] - 996 ABOVE[1,3] <= 32
 1(ToLeftOf)4: - L[4] + R[1] + 1461 LEFT[1,4] <= 1429
 1(Above)4: - T[4] + B[1] + 996 ABOVE[1,4] <= 964
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 1461 LEFT[1,4] <= 32
 1(ConverseOfAboveOf)4: T[4] - B[1] - 996 ABOVE[1,4] <= 32
 2(ToLeftOf)0: - L[0] + R[2] + 1461 LEFT[2,0] <= 1429
 2(Above)0: - T[0] + B[2] + 996 ABOVE[2,0] <= 964
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 1461 LEFT[2,0] <= 32
 2(ConverseOfAboveOf)0: T[0] - B[2] - 996 ABOVE[2,0] <= 32
 2(ToLeftOf)1: - L[1] + R[2] + 1461 LEFT[2,1] <= 1429
 2(Above)1: - T[1] + B[2] + 996 ABOVE[2,1] <= 964
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 1461 LEFT[2,1] <= 32
 2(ConverseOfAboveOf)1: T[1] - B[2] - 996 ABOVE[2,1] <= 32
 2(ToLeftOf)3: - L[3] + R[2] + 1461 LEFT[2,3] <= 1429
 2(Above)3: - T[3] + B[2] + 996 ABOVE[2,3] <= 964
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 1461 LEFT[2,3] <= 32
 2(ConverseOfAboveOf)3: T[3] - B[2] - 996 ABOVE[2,3] <= 32
 2(ToLeftOf)4: - L[4] + R[2] + 1461 LEFT[2,4] <= 1429
 2(Above)4: - T[4] + B[2] + 996 ABOVE[2,4] <= 964
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 1461 LEFT[2,4] <= 32
 2(ConverseOfAboveOf)4: T[4] - B[2] - 996 ABOVE[2,4] <= 32
 3(ToLeftOf)0: - L[0] + R[3] + 1461 LEFT[3,0] <= 1429
 3(Above)0: - T[0] + B[3] + 996 ABOVE[3,0] <= 964
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 1461 LEFT[3,0] <= 32
 3(ConverseOfAboveOf)0: T[0] - B[3] - 996 ABOVE[3,0] <= 32
 3(ToLeftOf)1: - L[1] + R[3] + 1461 LEFT[3,1] <= 1429
 3(Above)1: - T[1] + B[3] + 996 ABOVE[3,1] <= 964
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 1461 LEFT[3,1] <= 32
 3(ConverseOfAboveOf)1: T[1] - B[3] - 996 ABOVE[3,1] <= 32
 3(ToLeftOf)2: - L[2] + R[3] + 1461 LEFT[3,2] <= 1429
 3(Above)2: - T[2] + B[3] + 996 ABOVE[3,2] <= 964
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 1461 LEFT[3,2] <= 32
 3(ConverseOfAboveOf)2: T[2] - B[3] - 996 ABOVE[3,2] <= 32
 3(ToLeftOf)4: - L[4] + R[3] + 1461 LEFT[3,4] <= 1429
 3(Above)4: - T[4] + B[3] + 996 ABOVE[3,4] <= 964
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 1461 LEFT[3,4] <= 32
 3(ConverseOfAboveOf)4: T[4] - B[3] - 996 ABOVE[3,4] <= 32
 4(ToLeftOf)0: - L[0] + R[4] + 1461 LEFT[4,0] <= 1429
 4(Above)0: - T[0] + B[4] + 996 ABOVE[4,0] <= 964
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 1461 LEFT[4,0] <= 32
 4(ConverseOfAboveOf)0: T[0] - B[4] - 996 ABOVE[4,0] <= 32
 4(ToLeftOf)1: - L[1] + R[4] + 1461 LEFT[4,1] <= 1429
 4(Above)1: - T[1] + B[4] + 996 ABOVE[4,1] <= 964
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 1461 LEFT[4,1] <= 32
 4(ConverseOfAboveOf)1: T[1] - B[4] - 996 ABOVE[4,1] <= 32
 4(ToLeftOf)2: - L[2] + R[4] + 1461 LEFT[4,2] <= 1429
 4(Above)2: - T[2] + B[4] + 996 ABOVE[4,2] <= 964
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 1461 LEFT[4,2] <= 32
 4(ConverseOfAboveOf)2: T[2] - B[4] - 996 ABOVE[4,2] <= 32
 4(ToLeftOf)3: - L[3] + R[4] + 1461 LEFT[4,3] <= 1429
 4(Above)3: - T[3] + B[4] + 996 ABOVE[4,3] <= 964
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 1461 LEFT[4,3] <= 32
 4(ConverseOfAboveOf)3: T[3] - B[4] - 996 ABOVE[4,3] <= 32
 OneLAGForElement[0]: zLAG[0,0] + zLAG[0,1] + zLAG[0,2] + zLAG[0,3]
   + zLAG[0,4] = 1
 OneTAGForElement[0]: zTAG[0,0] + zTAG[0,1] + zTAG[0,2] + zTAG[0,3]
   + zTAG[0,4] = 1
 OneBAGForElement[0]: zBAG[0,0] + zBAG[0,1] + zBAG[0,2] + zBAG[0,3]
   + zBAG[0,4] = 1
 OneRAGForElement[0]: zRAG[0,0] + zRAG[0,1] + zRAG[0,2] + zRAG[0,3]
   + zRAG[0,4] = 1
 OneLAGForElement[1]: zLAG[1,0] + zLAG[1,1] + zLAG[1,2] + zLAG[1,3]
   + zLAG[1,4] = 1
 OneTAGForElement[1]: zTAG[1,0] + zTAG[1,1] + zTAG[1,2] + zTAG[1,3]
   + zTAG[1,4] = 1
 OneBAGForElement[1]: zBAG[1,0] + zBAG[1,1] + zBAG[1,2] + zBAG[1,3]
   + zBAG[1,4] = 1
 OneRAGForElement[1]: zRAG[1,0] + zRAG[1,1] + zRAG[1,2] + zRAG[1,3]
   + zRAG[1,4] = 1
 OneLAGForElement[2]: zLAG[2,0] + zLAG[2,1] + zLAG[2,2] + zLAG[2,3]
   + zLAG[2,4] = 1
 OneTAGForElement[2]: zTAG[2,0] + zTAG[2,1] + zTAG[2,2] + zTAG[2,3]
   + zTAG[2,4] = 1
 OneBAGForElement[2]: zBAG[2,0] + zBAG[2,1] + zBAG[2,2] + zBAG[2,3]
   + zBAG[2,4] = 1
 OneRAGForElement[2]: zRAG[2,0] + zRAG[2,1] + zRAG[2,2] + zRAG[2,3]
   + zRAG[2,4] = 1
 OneLAGForElement[3]: zLAG[3,0] + zLAG[3,1] + zLAG[3,2] + zLAG[3,3]
   + zLAG[3,4] = 1
 OneTAGForElement[3]: zTAG[3,0] + zTAG[3,1] + zTAG[3,2] + zTAG[3,3]
   + zTAG[3,4] = 1
 OneBAGForElement[3]: zBAG[3,0] + zBAG[3,1] + zBAG[3,2] + zBAG[3,3]
   + zBAG[3,4] = 1
 OneRAGForElement[3]: zRAG[3,0] + zRAG[3,1] + zRAG[3,2] + zRAG[3,3]
   + zRAG[3,4] = 1
 OneLAGForElement[4]: zLAG[4,0] + zLAG[4,1] + zLAG[4,2] + zLAG[4,3]
   + zLAG[4,4] = 1
 OneTAGForElement[4]: zTAG[4,0] + zTAG[4,1] + zTAG[4,2] + zTAG[4,3]
   + zTAG[4,4] = 1
 OneBAGForElement[4]: zBAG[4,0] + zBAG[4,1] + zBAG[4,2] + zBAG[4,3]
   + zBAG[4,4] = 1
 OneRAGForElement[4]: zRAG[4,0] + zRAG[4,1] + zRAG[4,2] + zRAG[4,3]
   + zRAG[4,4] = 1
 R182: - LAG[0] + zLAG[0,0] <= 0
 R183: - RAG[0] + zRAG[0,0] <= 0
 R184: - TAG[0] + zTAG[0,0] <= 0
 R185: - BAG[0] + zBAG[0,0] <= 0
 R186: - LAG[0] + zLAG[1,0] <= 0
 R187: - RAG[0] + zRAG[1,0] <= 0
 R188: - TAG[0] + zTAG[1,0] <= 0
 R189: - BAG[0] + zBAG[1,0] <= 0
 R190: - LAG[0] + zLAG[2,0] <= 0
 R191: - RAG[0] + zRAG[2,0] <= 0
 R192: - TAG[0] + zTAG[2,0] <= 0
 R193: - BAG[0] + zBAG[2,0] <= 0
 R194: - LAG[0] + zLAG[3,0] <= 0
 R195: - RAG[0] + zRAG[3,0] <= 0
 R196: - TAG[0] + zTAG[3,0] <= 0
 R197: - BAG[0] + zBAG[3,0] <= 0
 R198: - LAG[0] + zLAG[4,0] <= 0
 R199: - RAG[0] + zRAG[4,0] <= 0
 R200: - TAG[0] + zTAG[4,0] <= 0
 R201: - BAG[0] + zBAG[4,0] <= 0
 R202: - LAG[1] + zLAG[0,1] <= 0
 R203: - RAG[1] + zRAG[0,1] <= 0
 R204: - TAG[1] + zTAG[0,1] <= 0
 R205: - BAG[1] + zBAG[0,1] <= 0
 R206: - LAG[1] + zLAG[1,1] <= 0
 R207: - RAG[1] + zRAG[1,1] <= 0
 R208: - TAG[1] + zTAG[1,1] <= 0
 R209: - BAG[1] + zBAG[1,1] <= 0
 R210: - LAG[1] + zLAG[2,1] <= 0
 R211: - RAG[1] + zRAG[2,1] <= 0
 R212: - TAG[1] + zTAG[2,1] <= 0
 R213: - BAG[1] + zBAG[2,1] <= 0
 R214: - LAG[1] + zLAG[3,1] <= 0
 R215: - RAG[1] + zRAG[3,1] <= 0
 R216: - TAG[1] + zTAG[3,1] <= 0
 R217: - BAG[1] + zBAG[3,1] <= 0
 R218: - LAG[1] + zLAG[4,1] <= 0
 R219: - RAG[1] + zRAG[4,1] <= 0
 R220: - TAG[1] + zTAG[4,1] <= 0
 R221: - BAG[1] + zBAG[4,1] <= 0
 R222: - LAG[2] + zLAG[0,2] <= 0
 R223: - RAG[2] + zRAG[0,2] <= 0
 R224: - TAG[2] + zTAG[0,2] <= 0
 R225: - BAG[2] + zBAG[0,2] <= 0
 R226: - LAG[2] + zLAG[1,2] <= 0
 R227: - RAG[2] + zRAG[1,2] <= 0
 R228: - TAG[2] + zTAG[1,2] <= 0
 R229: - BAG[2] + zBAG[1,2] <= 0
 R230: - LAG[2] + zLAG[2,2] <= 0
 R231: - RAG[2] + zRAG[2,2] <= 0
 R232: - TAG[2] + zTAG[2,2] <= 0
 R233: - BAG[2] + zBAG[2,2] <= 0
 R234: - LAG[2] + zLAG[3,2] <= 0
 R235: - RAG[2] + zRAG[3,2] <= 0
 R236: - TAG[2] + zTAG[3,2] <= 0
 R237: - BAG[2] + zBAG[3,2] <= 0
 R238: - LAG[2] + zLAG[4,2] <= 0
 R239: - RAG[2] + zRAG[4,2] <= 0
 R240: - TAG[2] + zTAG[4,2] <= 0
 R241: - BAG[2] + zBAG[4,2] <= 0
 R242: - LAG[3] + zLAG[0,3] <= 0
 R243: - RAG[3] + zRAG[0,3] <= 0
 R244: - TAG[3] + zTAG[0,3] <= 0
 R245: - BAG[3] + zBAG[0,3] <= 0
 R246: - LAG[3] + zLAG[1,3] <= 0
 R247: - RAG[3] + zRAG[1,3] <= 0
 R248: - TAG[3] + zTAG[1,3] <= 0
 R249: - BAG[3] + zBAG[1,3] <= 0
 R250: - LAG[3] + zLAG[2,3] <= 0
 R251: - RAG[3] + zRAG[2,3] <= 0
 R252: - TAG[3] + zTAG[2,3] <= 0
 R253: - BAG[3] + zBAG[2,3] <= 0
 R254: - LAG[3] + zLAG[3,3] <= 0
 R255: - RAG[3] + zRAG[3,3] <= 0
 R256: - TAG[3] + zTAG[3,3] <= 0
 R257: - BAG[3] + zBAG[3,3] <= 0
 R258: - LAG[3] + zLAG[4,3] <= 0
 R259: - RAG[3] + zRAG[4,3] <= 0
 R260: - TAG[3] + zTAG[4,3] <= 0
 R261: - BAG[3] + zBAG[4,3] <= 0
 R262: - LAG[4] + zLAG[0,4] <= 0
 R263: - RAG[4] + zRAG[0,4] <= 0
 R264: - TAG[4] + zTAG[0,4] <= 0
 R265: - BAG[4] + zBAG[0,4] <= 0
 R266: - LAG[4] + zLAG[1,4] <= 0
 R267: - RAG[4] + zRAG[1,4] <= 0
 R268: - TAG[4] + zTAG[1,4] <= 0
 R269: - BAG[4] + zBAG[1,4] <= 0
 R270: - LAG[4] + zLAG[2,4] <= 0
 R271: - RAG[4] + zRAG[2,4] <= 0
 R272: - TAG[4] + zTAG[2,4] <= 0
 R273: - BAG[4] + zBAG[2,4] <= 0
 R274: - LAG[4] + zLAG[3,4] <= 0
 R275: - RAG[4] + zRAG[3,4] <= 0
 R276: - TAG[4] + zTAG[3,4] <= 0
 R277: - BAG[4] + zBAG[3,4] <= 0
 R278: - LAG[4] + zLAG[4,4] <= 0
 R279: - RAG[4] + zRAG[4,4] <= 0
 R280: - TAG[4] + zTAG[4,4] <= 0
 R281: - BAG[4] + zBAG[4,4] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 1461 zLAG[0,0] <= 1461
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 1461 zRAG[0,0] <= 1461
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 996 zTAG[0,0] <= 996
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 996 zBAG[0,0] <= 996
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 1461 zLAG[0,0] >= -1461
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 1461 zRAG[0,0] >= -1461
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 996 zTAG[0,0] >= -996
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 996 zBAG[0,0] >= -996
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 1461 zLAG[1,0] <= 1461
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 1461 zRAG[1,0] <= 1461
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 996 zTAG[1,0] <= 996
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 996 zBAG[1,0] <= 996
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 1461 zLAG[1,0] >= -1461
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 1461 zRAG[1,0] >= -1461
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 996 zTAG[1,0] >= -996
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 996 zBAG[1,0] >= -996
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 1461 zLAG[2,0] <= 1461
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 1461 zRAG[2,0] <= 1461
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 996 zTAG[2,0] <= 996
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 996 zBAG[2,0] <= 996
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 1461 zLAG[2,0] >= -1461
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 1461 zRAG[2,0] >= -1461
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 996 zTAG[2,0] >= -996
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 996 zBAG[2,0] >= -996
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 1461 zLAG[3,0] <= 1461
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 1461 zRAG[3,0] <= 1461
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 996 zTAG[3,0] <= 996
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 996 zBAG[3,0] <= 996
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 1461 zLAG[3,0] >= -1461
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 1461 zRAG[3,0] >= -1461
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 996 zTAG[3,0] >= -996
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 996 zBAG[3,0] >= -996
 MinsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] + 1461 zLAG[4,0] <= 1461
 MinsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] + 1461 zRAG[4,0] <= 1461
 MinsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] + 996 zTAG[4,0] <= 996
 MinsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] + 996 zBAG[4,0] <= 996
 MaxsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] - 1461 zLAG[4,0] >= -1461
 MaxsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] - 1461 zRAG[4,0] >= -1461
 MaxsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] - 996 zTAG[4,0] >= -996
 MaxsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] - 996 zBAG[4,0] >= -996
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 1461 zLAG[0,1] <= 1461
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 1461 zRAG[0,1] <= 1461
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 996 zTAG[0,1] <= 996
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 996 zBAG[0,1] <= 996
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 1461 zLAG[0,1] >= -1461
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 1461 zRAG[0,1] >= -1461
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 996 zTAG[0,1] >= -996
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 996 zBAG[0,1] >= -996
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 1461 zLAG[1,1] <= 1461
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 1461 zRAG[1,1] <= 1461
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 996 zTAG[1,1] <= 996
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 996 zBAG[1,1] <= 996
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 1461 zLAG[1,1] >= -1461
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 1461 zRAG[1,1] >= -1461
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 996 zTAG[1,1] >= -996
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 996 zBAG[1,1] >= -996
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 1461 zLAG[2,1] <= 1461
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 1461 zRAG[2,1] <= 1461
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 996 zTAG[2,1] <= 996
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 996 zBAG[2,1] <= 996
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 1461 zLAG[2,1] >= -1461
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 1461 zRAG[2,1] >= -1461
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 996 zTAG[2,1] >= -996
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 996 zBAG[2,1] >= -996
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 1461 zLAG[3,1] <= 1461
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 1461 zRAG[3,1] <= 1461
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 996 zTAG[3,1] <= 996
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 996 zBAG[3,1] <= 996
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 1461 zLAG[3,1] >= -1461
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 1461 zRAG[3,1] >= -1461
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 996 zTAG[3,1] >= -996
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 996 zBAG[3,1] >= -996
 MinsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] + 1461 zLAG[4,1] <= 1461
 MinsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] + 1461 zRAG[4,1] <= 1461
 MinsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] + 996 zTAG[4,1] <= 996
 MinsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] + 996 zBAG[4,1] <= 996
 MaxsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] - 1461 zLAG[4,1] >= -1461
 MaxsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] - 1461 zRAG[4,1] >= -1461
 MaxsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] - 996 zTAG[4,1] >= -996
 MaxsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] - 996 zBAG[4,1] >= -996
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 1461 zLAG[0,2] <= 1461
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 1461 zRAG[0,2] <= 1461
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 996 zTAG[0,2] <= 996
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 996 zBAG[0,2] <= 996
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 1461 zLAG[0,2] >= -1461
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 1461 zRAG[0,2] >= -1461
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 996 zTAG[0,2] >= -996
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 996 zBAG[0,2] >= -996
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 1461 zLAG[1,2] <= 1461
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 1461 zRAG[1,2] <= 1461
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 996 zTAG[1,2] <= 996
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 996 zBAG[1,2] <= 996
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 1461 zLAG[1,2] >= -1461
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 1461 zRAG[1,2] >= -1461
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 996 zTAG[1,2] >= -996
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 996 zBAG[1,2] >= -996
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 1461 zLAG[2,2] <= 1461
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 1461 zRAG[2,2] <= 1461
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 996 zTAG[2,2] <= 996
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 996 zBAG[2,2] <= 996
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 1461 zLAG[2,2] >= -1461
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 1461 zRAG[2,2] >= -1461
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 996 zTAG[2,2] >= -996
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 996 zBAG[2,2] >= -996
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 1461 zLAG[3,2] <= 1461
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 1461 zRAG[3,2] <= 1461
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 996 zTAG[3,2] <= 996
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 996 zBAG[3,2] <= 996
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 1461 zLAG[3,2] >= -1461
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 1461 zRAG[3,2] >= -1461
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 996 zTAG[3,2] >= -996
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 996 zBAG[3,2] >= -996
 MinsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] + 1461 zLAG[4,2] <= 1461
 MinsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] + 1461 zRAG[4,2] <= 1461
 MinsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] + 996 zTAG[4,2] <= 996
 MinsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] + 996 zBAG[4,2] <= 996
 MaxsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] - 1461 zLAG[4,2] >= -1461
 MaxsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] - 1461 zRAG[4,2] >= -1461
 MaxsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] - 996 zTAG[4,2] >= -996
 MaxsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] - 996 zBAG[4,2] >= -996
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 1461 zLAG[0,3] <= 1461
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 1461 zRAG[0,3] <= 1461
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 996 zTAG[0,3] <= 996
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 996 zBAG[0,3] <= 996
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 1461 zLAG[0,3] >= -1461
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 1461 zRAG[0,3] >= -1461
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 996 zTAG[0,3] >= -996
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 996 zBAG[0,3] >= -996
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 1461 zLAG[1,3] <= 1461
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 1461 zRAG[1,3] <= 1461
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 996 zTAG[1,3] <= 996
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 996 zBAG[1,3] <= 996
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 1461 zLAG[1,3] >= -1461
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 1461 zRAG[1,3] >= -1461
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 996 zTAG[1,3] >= -996
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 996 zBAG[1,3] >= -996
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 1461 zLAG[2,3] <= 1461
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 1461 zRAG[2,3] <= 1461
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 996 zTAG[2,3] <= 996
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 996 zBAG[2,3] <= 996
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 1461 zLAG[2,3] >= -1461
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 1461 zRAG[2,3] >= -1461
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 996 zTAG[2,3] >= -996
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 996 zBAG[2,3] >= -996
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 1461 zLAG[3,3] <= 1461
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 1461 zRAG[3,3] <= 1461
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 996 zTAG[3,3] <= 996
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 996 zBAG[3,3] <= 996
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 1461 zLAG[3,3] >= -1461
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 1461 zRAG[3,3] >= -1461
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 996 zTAG[3,3] >= -996
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 996 zBAG[3,3] >= -996
 MinsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] + 1461 zLAG[4,3] <= 1461
 MinsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] + 1461 zRAG[4,3] <= 1461
 MinsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] + 996 zTAG[4,3] <= 996
 MinsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] + 996 zBAG[4,3] <= 996
 MaxsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] - 1461 zLAG[4,3] >= -1461
 MaxsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] - 1461 zRAG[4,3] >= -1461
 MaxsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] - 996 zTAG[4,3] >= -996
 MaxsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] - 996 zBAG[4,3] >= -996
 MinsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] + 1461 zLAG[0,4] <= 1461
 MinsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] + 1461 zRAG[0,4] <= 1461
 MinsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] + 996 zTAG[0,4] <= 996
 MinsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] + 996 zBAG[0,4] <= 996
 MaxsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] - 1461 zLAG[0,4] >= -1461
 MaxsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] - 1461 zRAG[0,4] >= -1461
 MaxsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] - 996 zTAG[0,4] >= -996
 MaxsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] - 996 zBAG[0,4] >= -996
 MinsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] + 1461 zLAG[1,4] <= 1461
 MinsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] + 1461 zRAG[1,4] <= 1461
 MinsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] + 996 zTAG[1,4] <= 996
 MinsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] + 996 zBAG[1,4] <= 996
 MaxsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] - 1461 zLAG[1,4] >= -1461
 MaxsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] - 1461 zRAG[1,4] >= -1461
 MaxsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] - 996 zTAG[1,4] >= -996
 MaxsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] - 996 zBAG[1,4] >= -996
 MinsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] + 1461 zLAG[2,4] <= 1461
 MinsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] + 1461 zRAG[2,4] <= 1461
 MinsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] + 996 zTAG[2,4] <= 996
 MinsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] + 996 zBAG[2,4] <= 996
 MaxsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] - 1461 zLAG[2,4] >= -1461
 MaxsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] - 1461 zRAG[2,4] >= -1461
 MaxsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] - 996 zTAG[2,4] >= -996
 MaxsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] - 996 zBAG[2,4] >= -996
 MinsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] + 1461 zLAG[3,4] <= 1461
 MinsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] + 1461 zRAG[3,4] <= 1461
 MinsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] + 996 zTAG[3,4] <= 996
 MinsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] + 996 zBAG[3,4] <= 996
 MaxsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] - 1461 zLAG[3,4] >= -1461
 MaxsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] - 1461 zRAG[3,4] >= -1461
 MaxsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] - 996 zTAG[3,4] >= -996
 MaxsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] - 996 zBAG[3,4] >= -996
 MinsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] + 1461 zLAG[4,4] <= 1461
 MinsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] + 1461 zRAG[4,4] <= 1461
 MinsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] + 996 zTAG[4,4] <= 996
 MinsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] + 996 zBAG[4,4] <= 996
 MaxsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] - 1461 zLAG[4,4] >= -1461
 MaxsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] - 1461 zRAG[4,4] >= -1461
 MaxsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] - 996 zTAG[4,4] >= -996
 MaxsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] - 996 zBAG[4,4] >= -996
Bounds
 L[0] <= 149
 L[1] <= 1277
 L[2] <= 1093
 L[3] <= 1093
 L[4] <= 1125
 1312 <= R[0] <= 1461
 184 <= R[1] <= 1461
 368 <= R[2] <= 1461
 368 <= R[3] <= 1461
 336 <= R[4] <= 1461
 T[0] <= 964
 T[1] <= 148
 T[2] <= 756
 T[3] <= 580
 T[4] <= 516
 32 <= B[0] <= 996
 848 <= B[1] <= 996
 240 <= B[2] <= 996
 416 <= B[3] <= 996
 480 <= B[4] <= 996
 32 <= H[0] <= 40
 848 <= H[1] <= 856
 240 <= H[2] <= 248
 416 <= H[3] <= 424
 480 <= H[4] <= 488
 1312 <= W[0] <= 1320
 184 <= W[1] <= 192
 368 <= W[2] <= 376
 368 <= W[3] <= 376
 336 <= W[4] <= 344
 vLAG[0] <= 1460
 vLAG[1] <= 1460
 vLAG[2] <= 1460
 vLAG[3] <= 1460
 vLAG[4] <= 1460
 1 <= vRAG[0] <= 1461
 1 <= vRAG[1] <= 1461
 1 <= vRAG[2] <= 1461
 1 <= vRAG[3] <= 1461
 1 <= vRAG[4] <= 1461
 vTAG[0] <= 995
 vTAG[1] <= 995
 vTAG[2] <= 995
 vTAG[3] <= 995
 vTAG[4] <= 995
 1 <= vBAG[0] <= 996
 1 <= vBAG[1] <= 996
 1 <= vBAG[2] <= 996
 1 <= vBAG[3] <= 996
 1 <= vBAG[4] <= 996
Binaries
 ABOVE[0,0] ABOVE[0,1] ABOVE[0,2] ABOVE[0,3] ABOVE[0,4] ABOVE[1,0]
 ABOVE[1,1] ABOVE[1,2] ABOVE[1,3] ABOVE[1,4] ABOVE[2,0] ABOVE[2,1]
 ABOVE[2,2] ABOVE[2,3] ABOVE[2,4] ABOVE[3,0] ABOVE[3,1] ABOVE[3,2]
 ABOVE[3,3] ABOVE[3,4] ABOVE[4,0] ABOVE[4,1] ABOVE[4,2] ABOVE[4,3]
 ABOVE[4,4] LEFT[0,0] LEFT[0,1] LEFT[0,2] LEFT[0,3] LEFT[0,4] LEFT[1,0]
 LEFT[1,1] LEFT[1,2] LEFT[1,3] LEFT[1,4] LEFT[2,0] LEFT[2,1] LEFT[2,2]
 LEFT[2,3] LEFT[2,4] LEFT[3,0] LEFT[3,1] LEFT[3,2] LEFT[3,3] LEFT[3,4]
 LEFT[4,0] LEFT[4,1] LEFT[4,2] LEFT[4,3] LEFT[4,4] LAG[0] LAG[1] LAG[2]
 LAG[3] LAG[4] RAG[0] RAG[1] RAG[2] RAG[3] RAG[4] TAG[0] TAG[1] TAG[2]
 TAG[3] TAG[4] BAG[0] BAG[1] BAG[2] BAG[3] BAG[4] zLAG[0,0] zLAG[0,1]
 zLAG[0,2] zLAG[0,3] zLAG[0,4] zLAG[1,0] zLAG[1,1] zLAG[1,2] zLAG[1,3]
 zLAG[1,4] zLAG[2,0] zLAG[2,1] zLAG[2,2] zLAG[2,3] zLAG[2,4] zLAG[3,0]
 zLAG[3,1] zLAG[3,2] zLAG[3,3] zLAG[3,4] zLAG[4,0] zLAG[4,1] zLAG[4,2]
 zLAG[4,3] zLAG[4,4] zRAG[0,0] zRAG[0,1] zRAG[0,2] zRAG[0,3] zRAG[0,4]
 zRAG[1,0] zRAG[1,1] zRAG[1,2] zRAG[1,3] zRAG[1,4] zRAG[2,0] zRAG[2,1]
 zRAG[2,2] zRAG[2,3] zRAG[2,4] zRAG[3,0] zRAG[3,1] zRAG[3,2] zRAG[3,3]
 zRAG[3,4] zRAG[4,0] zRAG[4,1] zRAG[4,2] zRAG[4,3] zRAG[4,4] zTAG[0,0]
 zTAG[0,1] zTAG[0,2] zTAG[0,3] zTAG[0,4] zTAG[1,0] zTAG[1,1] zTAG[1,2]
 zTAG[1,3] zTAG[1,4] zTAG[2,0] zTAG[2,1] zTAG[2,2] zTAG[2,3] zTAG[2,4]
 zTAG[3,0] zTAG[3,1] zTAG[3,2] zTAG[3,3] zTAG[3,4] zTAG[4,0] zTAG[4,1]
 zTAG[4,2] zTAG[4,3] zTAG[4,4] zBAG[0,0] zBAG[0,1] zBAG[0,2] zBAG[0,3]
 zBAG[0,4] zBAG[1,0] zBAG[1,1] zBAG[1,2] zBAG[1,3] zBAG[1,4] zBAG[2,0]
 zBAG[2,1] zBAG[2,2] zBAG[2,3] zBAG[2,4] zBAG[3,0] zBAG[3,1] zBAG[3,2]
 zBAG[3,3] zBAG[3,4] zBAG[4,0] zBAG[4,1] zBAG[4,2] zBAG[4,3] zBAG[4,4]
Generals
 L[0] L[1] L[2] L[3] L[4] R[0] R[1] R[2] R[3] R[4] T[0] T[1] T[2] T[3] T[4]
 B[0] B[1] B[2] B[3] B[4] H[0] H[1] H[2] H[3] H[4] W[0] W[1] W[2] W[3] W[4]
 LG[0] LG[1] LG[2] LG[3] LG[4] TG[0] TG[1] TG[2] TG[3] TG[4] WG[0] WG[1]
 WG[2] WG[3] WG[4] HG[0] HG[1] HG[2] HG[3] HG[4] vLAG[0] vLAG[1] vLAG[2]
 vLAG[3] vLAG[4] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vTAG[0] vTAG[1]
 vTAG[2] vTAG[3] vTAG[4] vBAG[0] vBAG[1] vBAG[2] vBAG[3] vBAG[4] maxX maxY
End
