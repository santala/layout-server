\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.001 L[4]
   + 0.002 R[0] + 0.002 R[1] + 0.002 R[2] + 0.002 R[3] + 0.002 R[4]
   + 0.001 T[0] + 0.001 T[1] + 0.001 T[2] + 0.001 T[3] + 0.001 T[4]
   + 0.002 B[0] + 0.002 B[1] + 0.002 B[2] + 0.002 B[3] + 0.002 B[4]
   - 0.001 H[0] - 0.001 H[1] - 0.001 H[2] - 0.001 H[3] - 0.001 H[4]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] - 0.001 W[4]
   + 0.0762195121951219 resizeWAbs[0] + 0.5434782608695652 resizeWAbs[1]
   + 0.2717391304347826 resizeWAbs[2] + 0.2717391304347826 resizeWAbs[3]
   + 0.2976190476190476 resizeWAbs[4] + 3.125 resizeHAbs[0]
   + 0.1037344398340249 resizeHAbs[1] + 0.2083333333333333 resizeHAbs[2]
   + 0.2083333333333333 resizeHAbs[3] + 0.2083333333333333 resizeHAbs[4]
   + 0 ABOVE[0,0] + 0 ABOVE[1,1] + 0 ABOVE[2,2] + 0 ABOVE[3,3]
   + 0 ABOVE[4,4] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2] + 0 LEFT[3,3]
   + 0 LEFT[4,4] + LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + RAG[0]
   + RAG[1] + RAG[2] + RAG[3] + RAG[4] + TAG[0] + TAG[1] + TAG[2] + TAG[3]
   + TAG[4] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4]
Subject To
 resizeW: - W[0] + resizeW[0] = -164
 resizeH: - H[0] + resizeH[0] = -4
 PrespecifiedXOfElement(0): L[0] = 0
 PrespecifiedYOfElement(0): T[0] = 0
 resizeW: - W[1] + resizeW[1] = -23
 resizeH: - H[1] + resizeH[1] = -121
 resizeW: - W[2] + resizeW[2] = -46
 resizeH: - H[2] + resizeH[2] = -60
 resizeW: - W[3] + resizeW[3] = -46
 resizeH: - H[3] + resizeH[3] = -60
 resizeW: - W[4] + resizeW[4] = -42
 resizeH: - H[4] + resizeH[4] = -60
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
 R163: - LAG[0] + zLAG[0,0] <= 0
 R164: - RAG[0] + zRAG[0,0] <= 0
 R165: - TAG[0] + zTAG[0,0] <= 0
 R166: - BAG[0] + zBAG[0,0] <= 0
 R167: - LAG[0] + zLAG[1,0] <= 0
 R168: - RAG[0] + zRAG[1,0] <= 0
 R169: - TAG[0] + zTAG[1,0] <= 0
 R170: - BAG[0] + zBAG[1,0] <= 0
 R171: - LAG[0] + zLAG[2,0] <= 0
 R172: - RAG[0] + zRAG[2,0] <= 0
 R173: - TAG[0] + zTAG[2,0] <= 0
 R174: - BAG[0] + zBAG[2,0] <= 0
 R175: - LAG[0] + zLAG[3,0] <= 0
 R176: - RAG[0] + zRAG[3,0] <= 0
 R177: - TAG[0] + zTAG[3,0] <= 0
 R178: - BAG[0] + zBAG[3,0] <= 0
 R179: - LAG[0] + zLAG[4,0] <= 0
 R180: - RAG[0] + zRAG[4,0] <= 0
 R181: - TAG[0] + zTAG[4,0] <= 0
 R182: - BAG[0] + zBAG[4,0] <= 0
 R183: - LAG[1] + zLAG[0,1] <= 0
 R184: - RAG[1] + zRAG[0,1] <= 0
 R185: - TAG[1] + zTAG[0,1] <= 0
 R186: - BAG[1] + zBAG[0,1] <= 0
 R187: - LAG[1] + zLAG[1,1] <= 0
 R188: - RAG[1] + zRAG[1,1] <= 0
 R189: - TAG[1] + zTAG[1,1] <= 0
 R190: - BAG[1] + zBAG[1,1] <= 0
 R191: - LAG[1] + zLAG[2,1] <= 0
 R192: - RAG[1] + zRAG[2,1] <= 0
 R193: - TAG[1] + zTAG[2,1] <= 0
 R194: - BAG[1] + zBAG[2,1] <= 0
 R195: - LAG[1] + zLAG[3,1] <= 0
 R196: - RAG[1] + zRAG[3,1] <= 0
 R197: - TAG[1] + zTAG[3,1] <= 0
 R198: - BAG[1] + zBAG[3,1] <= 0
 R199: - LAG[1] + zLAG[4,1] <= 0
 R200: - RAG[1] + zRAG[4,1] <= 0
 R201: - TAG[1] + zTAG[4,1] <= 0
 R202: - BAG[1] + zBAG[4,1] <= 0
 R203: - LAG[2] + zLAG[0,2] <= 0
 R204: - RAG[2] + zRAG[0,2] <= 0
 R205: - TAG[2] + zTAG[0,2] <= 0
 R206: - BAG[2] + zBAG[0,2] <= 0
 R207: - LAG[2] + zLAG[1,2] <= 0
 R208: - RAG[2] + zRAG[1,2] <= 0
 R209: - TAG[2] + zTAG[1,2] <= 0
 R210: - BAG[2] + zBAG[1,2] <= 0
 R211: - LAG[2] + zLAG[2,2] <= 0
 R212: - RAG[2] + zRAG[2,2] <= 0
 R213: - TAG[2] + zTAG[2,2] <= 0
 R214: - BAG[2] + zBAG[2,2] <= 0
 R215: - LAG[2] + zLAG[3,2] <= 0
 R216: - RAG[2] + zRAG[3,2] <= 0
 R217: - TAG[2] + zTAG[3,2] <= 0
 R218: - BAG[2] + zBAG[3,2] <= 0
 R219: - LAG[2] + zLAG[4,2] <= 0
 R220: - RAG[2] + zRAG[4,2] <= 0
 R221: - TAG[2] + zTAG[4,2] <= 0
 R222: - BAG[2] + zBAG[4,2] <= 0
 R223: - LAG[3] + zLAG[0,3] <= 0
 R224: - RAG[3] + zRAG[0,3] <= 0
 R225: - TAG[3] + zTAG[0,3] <= 0
 R226: - BAG[3] + zBAG[0,3] <= 0
 R227: - LAG[3] + zLAG[1,3] <= 0
 R228: - RAG[3] + zRAG[1,3] <= 0
 R229: - TAG[3] + zTAG[1,3] <= 0
 R230: - BAG[3] + zBAG[1,3] <= 0
 R231: - LAG[3] + zLAG[2,3] <= 0
 R232: - RAG[3] + zRAG[2,3] <= 0
 R233: - TAG[3] + zTAG[2,3] <= 0
 R234: - BAG[3] + zBAG[2,3] <= 0
 R235: - LAG[3] + zLAG[3,3] <= 0
 R236: - RAG[3] + zRAG[3,3] <= 0
 R237: - TAG[3] + zTAG[3,3] <= 0
 R238: - BAG[3] + zBAG[3,3] <= 0
 R239: - LAG[3] + zLAG[4,3] <= 0
 R240: - RAG[3] + zRAG[4,3] <= 0
 R241: - TAG[3] + zTAG[4,3] <= 0
 R242: - BAG[3] + zBAG[4,3] <= 0
 R243: - LAG[4] + zLAG[0,4] <= 0
 R244: - RAG[4] + zRAG[0,4] <= 0
 R245: - TAG[4] + zTAG[0,4] <= 0
 R246: - BAG[4] + zBAG[0,4] <= 0
 R247: - LAG[4] + zLAG[1,4] <= 0
 R248: - RAG[4] + zRAG[1,4] <= 0
 R249: - TAG[4] + zTAG[1,4] <= 0
 R250: - BAG[4] + zBAG[1,4] <= 0
 R251: - LAG[4] + zLAG[2,4] <= 0
 R252: - RAG[4] + zRAG[2,4] <= 0
 R253: - TAG[4] + zTAG[2,4] <= 0
 R254: - BAG[4] + zBAG[2,4] <= 0
 R255: - LAG[4] + zLAG[3,4] <= 0
 R256: - RAG[4] + zRAG[3,4] <= 0
 R257: - TAG[4] + zTAG[3,4] <= 0
 R258: - BAG[4] + zBAG[3,4] <= 0
 R259: - LAG[4] + zLAG[4,4] <= 0
 R260: - RAG[4] + zRAG[4,4] <= 0
 R261: - TAG[4] + zTAG[4,4] <= 0
 R262: - BAG[4] + zBAG[4,4] <= 0
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
 R463: - R[0] + maxX >= 0
 R464: - B[0] + maxY >= 0
 R465: - R[1] + maxX >= 0
 R466: - B[1] + maxY >= 0
 R467: - R[2] + maxX >= 0
 R468: - B[2] + maxY >= 0
 R469: - R[3] + maxX >= 0
 R470: - B[3] + maxY >= 0
 R471: - R[4] + maxX >= 0
 R472: - B[4] + maxY >= 0
 R473: LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + RAG[0] + RAG[1]
   + RAG[2] + RAG[3] + RAG[4] + TAG[0] + TAG[1] + TAG[2] + TAG[3] + TAG[4]
   + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4] >= 10
Bounds
 L[0] <= 182.5
 L[1] <= 182.5
 L[2] <= 182.5
 L[3] <= 182.5
 L[4] <= 182.5
 1 <= R[0] <= 182.625
 1 <= R[1] <= 182.625
 1 <= R[2] <= 182.625
 1 <= R[3] <= 182.625
 1 <= R[4] <= 182.625
 T[0] <= 124.375
 T[1] <= 124.375
 T[2] <= 124.375
 T[3] <= 124.375
 T[4] <= 124.375
 1 <= B[0] <= 124.5
 1 <= B[1] <= 124.5
 1 <= B[2] <= 124.5
 1 <= B[3] <= 124.5
 1 <= B[4] <= 124.5
 1 <= H[0] <= 125
 1 <= H[1] <= 125
 1 <= H[2] <= 125
 1 <= H[3] <= 125
 1 <= H[4] <= 125
 1 <= W[0] <= 183
 1 <= W[1] <= 183
 1 <= W[2] <= 183
 1 <= W[3] <= 183
 1 <= W[4] <= 183
 resizeW[0] >= -1000
 resizeW[1] >= -1000
 resizeW[2] >= -1000
 resizeW[3] >= -1000
 resizeW[4] >= -1000
 resizeH[0] >= -1000
 resizeH[1] >= -1000
 resizeH[2] >= -1000
 resizeH[3] >= -1000
 resizeH[4] >= -1000
 vLAG[0] <= 181.625
 vLAG[1] <= 181.625
 vLAG[2] <= 181.625
 vLAG[3] <= 181.625
 vLAG[4] <= 181.625
 1 <= vRAG[0] <= 182.625
 1 <= vRAG[1] <= 182.625
 1 <= vRAG[2] <= 182.625
 1 <= vRAG[3] <= 182.625
 1 <= vRAG[4] <= 182.625
 vTAG[0] <= 123.5
 vTAG[1] <= 123.5
 vTAG[2] <= 123.5
 vTAG[3] <= 123.5
 vTAG[4] <= 123.5
 1 <= vBAG[0] <= 124.5
 1 <= vBAG[1] <= 124.5
 1 <= vBAG[2] <= 124.5
 1 <= vBAG[3] <= 124.5
 1 <= vBAG[4] <= 124.5
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
 resizeW[0] resizeW[1] resizeW[2] resizeW[3] resizeW[4] resizeWAbs[0]
 resizeWAbs[1] resizeWAbs[2] resizeWAbs[3] resizeWAbs[4] resizeH[0]
 resizeH[1] resizeH[2] resizeH[3] resizeH[4] resizeHAbs[0] resizeHAbs[1]
 resizeHAbs[2] resizeHAbs[3] resizeHAbs[4] vLAG[0] vLAG[1] vLAG[2] vLAG[3]
 vLAG[4] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vTAG[0] vTAG[1] vTAG[2]
 vTAG[3] vTAG[4] vBAG[0] vBAG[1] vBAG[2] vBAG[3] vBAG[4] maxX maxY
General Constraints
 resizeWAbs: resizeWAbs[0] = ABS ( resizeW[0] )
 resizeHAbs: resizeHAbs[0] = ABS ( resizeH[0] )
 resizeWAbs: resizeWAbs[1] = ABS ( resizeW[1] )
 resizeHAbs: resizeHAbs[1] = ABS ( resizeH[1] )
 resizeWAbs: resizeWAbs[2] = ABS ( resizeW[2] )
 resizeHAbs: resizeHAbs[2] = ABS ( resizeH[2] )
 resizeWAbs: resizeWAbs[3] = ABS ( resizeW[3] )
 resizeHAbs: resizeHAbs[3] = ABS ( resizeH[3] )
 resizeWAbs: resizeWAbs[4] = ABS ( resizeW[4] )
 resizeHAbs: resizeHAbs[4] = ABS ( resizeH[4] )
End
