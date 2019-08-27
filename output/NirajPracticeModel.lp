\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.002 R[0]
   + 0.002 R[1] + 0.002 R[2] + 0.002 R[3] + 0.001 T[0] + 0.001 T[1]
   + 0.001 T[2] + 0.001 T[3] + 0.002 B[0] + 0.002 B[1] + 0.002 B[2]
   + 0.002 B[3] - 0.001 H[0] - 0.001 H[1] - 0.001 H[2] - 0.001 H[3]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3] + 0 ABOVE[0,0]
   + 0 ABOVE[1,1] + 0 ABOVE[2,2] + 0 ABOVE[3,3] + 0 LEFT[0,0] + 0 LEFT[1,1]
   + 0 LEFT[2,2] + 0 LEFT[3,3] + 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3]
   + RAG[0] + RAG[1] + RAG[2] + RAG[3] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2]
   + 2 TAG[3] + BAG[0] + BAG[1] + BAG[2] + BAG[3]
Subject To
 R0: - R[0] + maxX >= 0
 R1: - B[0] + maxY >= 0
 R2: - R[1] + maxX >= 0
 R3: - B[1] + maxY >= 0
 R4: - R[2] + maxX >= 0
 R5: - B[2] + maxY >= 0
 R6: - R[3] + maxX >= 0
 R7: - B[3] + maxY >= 0
 R8: 2 LAG[0] + 2 LAG[1] + 2 LAG[2] + 2 LAG[3] + RAG[0] + RAG[1] + RAG[2]
   + RAG[3] + 2 TAG[0] + 2 TAG[1] + 2 TAG[2] + 2 TAG[3] + BAG[0] + BAG[1]
   + BAG[2] + BAG[3] >= 8
 PrespecifiedXOfElement(0): L[0] = 10
 PrespecifiedYOfElement(0): T[0] = 81
 PrespecifiedXOfElement(1): L[1] = 285
 PrespecifiedYOfElement(1): T[1] = 81
 PrespecifiedXOfElement(2): L[2] = 28
 PrespecifiedYOfElement(2): T[2] = 34
 PrespecifiedXOfElement(3): L[3] = 29
 PrespecifiedYOfElement(3): T[3] = 135
 DisableNon-PermittedBasedOnPrespecified: = 0
 R-L=W(0): L[0] - R[0] + W[0] = 0
 B-T=H(0): T[0] - B[0] + H[0] = 0
 R-L=W(1): L[1] - R[1] + W[1] = 0
 B-T=H(1): T[1] - B[1] + H[1] = 0
 R-L=W(2): L[2] - R[2] + W[2] = 0
 B-T=H(2): T[2] - B[2] + H[2] = 0
 R-L=W(3): L[3] - R[3] + W[3] = 0
 B-T=H(3): T[3] - B[3] + H[3] = 0
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
 NoOverlap(23): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2] >= 1
 UpperLimOfQuadrants(23): ABOVE[2,3] + ABOVE[3,2] + LEFT[2,3] + LEFT[3,2]
   <= 2
 Anti-symmetryABOVE(23): ABOVE[2,3] + ABOVE[3,2] <= 1
 Anti-symmetryLEFT(23): LEFT[2,3] + LEFT[3,2] <= 1
 0(ToLeftOf)1: - L[1] + R[0] + 575 LEFT[0,1] <= 575
 0(Above)1: - T[1] + B[0] + 535 ABOVE[0,1] <= 535
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 575 LEFT[0,1] <= 0
 0(ConverseOfAboveOf)1: T[1] - B[0] - 535 ABOVE[0,1] <= 0
 0(ToLeftOf)2: - L[2] + R[0] + 575 LEFT[0,2] <= 575
 0(Above)2: - T[2] + B[0] + 535 ABOVE[0,2] <= 535
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 575 LEFT[0,2] <= 0
 0(ConverseOfAboveOf)2: T[2] - B[0] - 535 ABOVE[0,2] <= 0
 0(ToLeftOf)3: - L[3] + R[0] + 575 LEFT[0,3] <= 575
 0(Above)3: - T[3] + B[0] + 535 ABOVE[0,3] <= 535
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 575 LEFT[0,3] <= 0
 0(ConverseOfAboveOf)3: T[3] - B[0] - 535 ABOVE[0,3] <= 0
 1(ToLeftOf)0: - L[0] + R[1] + 575 LEFT[1,0] <= 575
 1(Above)0: - T[0] + B[1] + 535 ABOVE[1,0] <= 535
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 575 LEFT[1,0] <= 0
 1(ConverseOfAboveOf)0: T[0] - B[1] - 535 ABOVE[1,0] <= 0
 1(ToLeftOf)2: - L[2] + R[1] + 575 LEFT[1,2] <= 575
 1(Above)2: - T[2] + B[1] + 535 ABOVE[1,2] <= 535
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 575 LEFT[1,2] <= 0
 1(ConverseOfAboveOf)2: T[2] - B[1] - 535 ABOVE[1,2] <= 0
 1(ToLeftOf)3: - L[3] + R[1] + 575 LEFT[1,3] <= 575
 1(Above)3: - T[3] + B[1] + 535 ABOVE[1,3] <= 535
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 575 LEFT[1,3] <= 0
 1(ConverseOfAboveOf)3: T[3] - B[1] - 535 ABOVE[1,3] <= 0
 2(ToLeftOf)0: - L[0] + R[2] + 575 LEFT[2,0] <= 575
 2(Above)0: - T[0] + B[2] + 535 ABOVE[2,0] <= 535
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 575 LEFT[2,0] <= 0
 2(ConverseOfAboveOf)0: T[0] - B[2] - 535 ABOVE[2,0] <= 0
 2(ToLeftOf)1: - L[1] + R[2] + 575 LEFT[2,1] <= 575
 2(Above)1: - T[1] + B[2] + 535 ABOVE[2,1] <= 535
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 575 LEFT[2,1] <= 0
 2(ConverseOfAboveOf)1: T[1] - B[2] - 535 ABOVE[2,1] <= 0
 2(ToLeftOf)3: - L[3] + R[2] + 575 LEFT[2,3] <= 575
 2(Above)3: - T[3] + B[2] + 535 ABOVE[2,3] <= 535
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 575 LEFT[2,3] <= 0
 2(ConverseOfAboveOf)3: T[3] - B[2] - 535 ABOVE[2,3] <= 0
 3(ToLeftOf)0: - L[0] + R[3] + 575 LEFT[3,0] <= 575
 3(Above)0: - T[0] + B[3] + 535 ABOVE[3,0] <= 535
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 575 LEFT[3,0] <= 0
 3(ConverseOfAboveOf)0: T[0] - B[3] - 535 ABOVE[3,0] <= 0
 3(ToLeftOf)1: - L[1] + R[3] + 575 LEFT[3,1] <= 575
 3(Above)1: - T[1] + B[3] + 535 ABOVE[3,1] <= 535
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 575 LEFT[3,1] <= 0
 3(ConverseOfAboveOf)1: T[1] - B[3] - 535 ABOVE[3,1] <= 0
 3(ToLeftOf)2: - L[2] + R[3] + 575 LEFT[3,2] <= 575
 3(Above)2: - T[2] + B[3] + 535 ABOVE[3,2] <= 535
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 575 LEFT[3,2] <= 0
 3(ConverseOfAboveOf)2: T[2] - B[3] - 535 ABOVE[3,2] <= 0
 OneLAGForElement[0]: zLAG[0,0] + zLAG[0,1] + zLAG[0,2] + zLAG[0,3] = 1
 OneTAGForElement[0]: zTAG[0,0] + zTAG[0,1] + zTAG[0,2] + zTAG[0,3] = 1
 OneBAGForElement[0]: zBAG[0,0] + zBAG[0,1] + zBAG[0,2] + zBAG[0,3] = 1
 OneRAGForElement[0]: zRAG[0,0] + zRAG[0,1] + zRAG[0,2] + zRAG[0,3] = 1
 OneLAGForElement[1]: zLAG[1,0] + zLAG[1,1] + zLAG[1,2] + zLAG[1,3] = 1
 OneTAGForElement[1]: zTAG[1,0] + zTAG[1,1] + zTAG[1,2] + zTAG[1,3] = 1
 OneBAGForElement[1]: zBAG[1,0] + zBAG[1,1] + zBAG[1,2] + zBAG[1,3] = 1
 OneRAGForElement[1]: zRAG[1,0] + zRAG[1,1] + zRAG[1,2] + zRAG[1,3] = 1
 OneLAGForElement[2]: zLAG[2,0] + zLAG[2,1] + zLAG[2,2] + zLAG[2,3] = 1
 OneTAGForElement[2]: zTAG[2,0] + zTAG[2,1] + zTAG[2,2] + zTAG[2,3] = 1
 OneBAGForElement[2]: zBAG[2,0] + zBAG[2,1] + zBAG[2,2] + zBAG[2,3] = 1
 OneRAGForElement[2]: zRAG[2,0] + zRAG[2,1] + zRAG[2,2] + zRAG[2,3] = 1
 OneLAGForElement[3]: zLAG[3,0] + zLAG[3,1] + zLAG[3,2] + zLAG[3,3] = 1
 OneTAGForElement[3]: zTAG[3,0] + zTAG[3,1] + zTAG[3,2] + zTAG[3,3] = 1
 OneBAGForElement[3]: zBAG[3,0] + zBAG[3,1] + zBAG[3,2] + zBAG[3,3] = 1
 OneRAGForElement[3]: zRAG[3,0] + zRAG[3,1] + zRAG[3,2] + zRAG[3,3] = 1
 R114: - LAG[0] + zLAG[0,0] <= 0
 R115: - RAG[0] + zRAG[0,0] <= 0
 R116: - TAG[0] + zTAG[0,0] <= 0
 R117: - BAG[0] + zBAG[0,0] <= 0
 R118: - LAG[0] + zLAG[1,0] <= 0
 R119: - RAG[0] + zRAG[1,0] <= 0
 R120: - TAG[0] + zTAG[1,0] <= 0
 R121: - BAG[0] + zBAG[1,0] <= 0
 R122: - LAG[0] + zLAG[2,0] <= 0
 R123: - RAG[0] + zRAG[2,0] <= 0
 R124: - TAG[0] + zTAG[2,0] <= 0
 R125: - BAG[0] + zBAG[2,0] <= 0
 R126: - LAG[0] + zLAG[3,0] <= 0
 R127: - RAG[0] + zRAG[3,0] <= 0
 R128: - TAG[0] + zTAG[3,0] <= 0
 R129: - BAG[0] + zBAG[3,0] <= 0
 R130: - LAG[1] + zLAG[0,1] <= 0
 R131: - RAG[1] + zRAG[0,1] <= 0
 R132: - TAG[1] + zTAG[0,1] <= 0
 R133: - BAG[1] + zBAG[0,1] <= 0
 R134: - LAG[1] + zLAG[1,1] <= 0
 R135: - RAG[1] + zRAG[1,1] <= 0
 R136: - TAG[1] + zTAG[1,1] <= 0
 R137: - BAG[1] + zBAG[1,1] <= 0
 R138: - LAG[1] + zLAG[2,1] <= 0
 R139: - RAG[1] + zRAG[2,1] <= 0
 R140: - TAG[1] + zTAG[2,1] <= 0
 R141: - BAG[1] + zBAG[2,1] <= 0
 R142: - LAG[1] + zLAG[3,1] <= 0
 R143: - RAG[1] + zRAG[3,1] <= 0
 R144: - TAG[1] + zTAG[3,1] <= 0
 R145: - BAG[1] + zBAG[3,1] <= 0
 R146: - LAG[2] + zLAG[0,2] <= 0
 R147: - RAG[2] + zRAG[0,2] <= 0
 R148: - TAG[2] + zTAG[0,2] <= 0
 R149: - BAG[2] + zBAG[0,2] <= 0
 R150: - LAG[2] + zLAG[1,2] <= 0
 R151: - RAG[2] + zRAG[1,2] <= 0
 R152: - TAG[2] + zTAG[1,2] <= 0
 R153: - BAG[2] + zBAG[1,2] <= 0
 R154: - LAG[2] + zLAG[2,2] <= 0
 R155: - RAG[2] + zRAG[2,2] <= 0
 R156: - TAG[2] + zTAG[2,2] <= 0
 R157: - BAG[2] + zBAG[2,2] <= 0
 R158: - LAG[2] + zLAG[3,2] <= 0
 R159: - RAG[2] + zRAG[3,2] <= 0
 R160: - TAG[2] + zTAG[3,2] <= 0
 R161: - BAG[2] + zBAG[3,2] <= 0
 R162: - LAG[3] + zLAG[0,3] <= 0
 R163: - RAG[3] + zRAG[0,3] <= 0
 R164: - TAG[3] + zTAG[0,3] <= 0
 R165: - BAG[3] + zBAG[0,3] <= 0
 R166: - LAG[3] + zLAG[1,3] <= 0
 R167: - RAG[3] + zRAG[1,3] <= 0
 R168: - TAG[3] + zTAG[1,3] <= 0
 R169: - BAG[3] + zBAG[1,3] <= 0
 R170: - LAG[3] + zLAG[2,3] <= 0
 R171: - RAG[3] + zRAG[2,3] <= 0
 R172: - TAG[3] + zTAG[2,3] <= 0
 R173: - BAG[3] + zBAG[2,3] <= 0
 R174: - LAG[3] + zLAG[3,3] <= 0
 R175: - RAG[3] + zRAG[3,3] <= 0
 R176: - TAG[3] + zTAG[3,3] <= 0
 R177: - BAG[3] + zBAG[3,3] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 575 zLAG[0,0] <= 575
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 575 zRAG[0,0] <= 575
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 535 zTAG[0,0] <= 535
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 535 zBAG[0,0] <= 535
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 575 zLAG[0,0] >= -575
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 575 zRAG[0,0] >= -575
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 535 zTAG[0,0] >= -535
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 535 zBAG[0,0] >= -535
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 575 zLAG[1,0] <= 575
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 575 zRAG[1,0] <= 575
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 535 zTAG[1,0] <= 535
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 535 zBAG[1,0] <= 535
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 575 zLAG[1,0] >= -575
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 575 zRAG[1,0] >= -575
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 535 zTAG[1,0] >= -535
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 535 zBAG[1,0] >= -535
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 575 zLAG[2,0] <= 575
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 575 zRAG[2,0] <= 575
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 535 zTAG[2,0] <= 535
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 535 zBAG[2,0] <= 535
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 575 zLAG[2,0] >= -575
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 575 zRAG[2,0] >= -575
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 535 zTAG[2,0] >= -535
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 535 zBAG[2,0] >= -535
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 575 zLAG[3,0] <= 575
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 575 zRAG[3,0] <= 575
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 535 zTAG[3,0] <= 535
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 535 zBAG[3,0] <= 535
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 575 zLAG[3,0] >= -575
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 575 zRAG[3,0] >= -575
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 535 zTAG[3,0] >= -535
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 535 zBAG[3,0] >= -535
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 575 zLAG[0,1] <= 575
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 575 zRAG[0,1] <= 575
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 535 zTAG[0,1] <= 535
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 535 zBAG[0,1] <= 535
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 575 zLAG[0,1] >= -575
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 575 zRAG[0,1] >= -575
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 535 zTAG[0,1] >= -535
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 535 zBAG[0,1] >= -535
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 575 zLAG[1,1] <= 575
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 575 zRAG[1,1] <= 575
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 535 zTAG[1,1] <= 535
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 535 zBAG[1,1] <= 535
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 575 zLAG[1,1] >= -575
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 575 zRAG[1,1] >= -575
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 535 zTAG[1,1] >= -535
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 535 zBAG[1,1] >= -535
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 575 zLAG[2,1] <= 575
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 575 zRAG[2,1] <= 575
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 535 zTAG[2,1] <= 535
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 535 zBAG[2,1] <= 535
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 575 zLAG[2,1] >= -575
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 575 zRAG[2,1] >= -575
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 535 zTAG[2,1] >= -535
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 535 zBAG[2,1] >= -535
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 575 zLAG[3,1] <= 575
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 575 zRAG[3,1] <= 575
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 535 zTAG[3,1] <= 535
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 535 zBAG[3,1] <= 535
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 575 zLAG[3,1] >= -575
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 575 zRAG[3,1] >= -575
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 535 zTAG[3,1] >= -535
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 535 zBAG[3,1] >= -535
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 575 zLAG[0,2] <= 575
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 575 zRAG[0,2] <= 575
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 535 zTAG[0,2] <= 535
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 535 zBAG[0,2] <= 535
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 575 zLAG[0,2] >= -575
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 575 zRAG[0,2] >= -575
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 535 zTAG[0,2] >= -535
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 535 zBAG[0,2] >= -535
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 575 zLAG[1,2] <= 575
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 575 zRAG[1,2] <= 575
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 535 zTAG[1,2] <= 535
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 535 zBAG[1,2] <= 535
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 575 zLAG[1,2] >= -575
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 575 zRAG[1,2] >= -575
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 535 zTAG[1,2] >= -535
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 535 zBAG[1,2] >= -535
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 575 zLAG[2,2] <= 575
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 575 zRAG[2,2] <= 575
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 535 zTAG[2,2] <= 535
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 535 zBAG[2,2] <= 535
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 575 zLAG[2,2] >= -575
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 575 zRAG[2,2] >= -575
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 535 zTAG[2,2] >= -535
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 535 zBAG[2,2] >= -535
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 575 zLAG[3,2] <= 575
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 575 zRAG[3,2] <= 575
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 535 zTAG[3,2] <= 535
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 535 zBAG[3,2] <= 535
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 575 zLAG[3,2] >= -575
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 575 zRAG[3,2] >= -575
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 535 zTAG[3,2] >= -535
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 535 zBAG[3,2] >= -535
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 575 zLAG[0,3] <= 575
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 575 zRAG[0,3] <= 575
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 535 zTAG[0,3] <= 535
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 535 zBAG[0,3] <= 535
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 575 zLAG[0,3] >= -575
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 575 zRAG[0,3] >= -575
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 535 zTAG[0,3] >= -535
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 535 zBAG[0,3] >= -535
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 575 zLAG[1,3] <= 575
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 575 zRAG[1,3] <= 575
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 535 zTAG[1,3] <= 535
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 535 zBAG[1,3] <= 535
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 575 zLAG[1,3] >= -575
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 575 zRAG[1,3] >= -575
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 535 zTAG[1,3] >= -535
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 535 zBAG[1,3] >= -535
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 575 zLAG[2,3] <= 575
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 575 zRAG[2,3] <= 575
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 535 zTAG[2,3] <= 535
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 535 zBAG[2,3] <= 535
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 575 zLAG[2,3] >= -575
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 575 zRAG[2,3] >= -575
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 535 zTAG[2,3] >= -535
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 535 zBAG[2,3] >= -535
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 575 zLAG[3,3] <= 575
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 575 zRAG[3,3] <= 575
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 535 zTAG[3,3] <= 535
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 535 zBAG[3,3] <= 535
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 575 zLAG[3,3] >= -575
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 575 zRAG[3,3] >= -575
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 535 zTAG[3,3] >= -535
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 535 zBAG[3,3] >= -535
Bounds
 L[0] <= 336
 L[1] <= 303
 L[2] <= 514
 L[3] <= 47
 239 <= R[0] <= 575
 272 <= R[1] <= 575
 61 <= R[2] <= 575
 528 <= R[3] <= 575
 T[0] <= 487
 T[1] <= 487
 T[2] <= 513
 T[3] <= 143
 48 <= B[0] <= 535
 48 <= B[1] <= 535
 22 <= B[2] <= 535
 392 <= B[3] <= 535
 H[0] = 48
 H[1] = 48
 H[2] = 22
 H[3] = 392
 W[0] = 239
 W[1] = 272
 W[2] = 61
 W[3] = 528
 vLAG[0] <= 574
 vLAG[1] <= 574
 vLAG[2] <= 574
 vLAG[3] <= 574
 1 <= vRAG[0] <= 575
 1 <= vRAG[1] <= 575
 1 <= vRAG[2] <= 575
 1 <= vRAG[3] <= 575
 vTAG[0] <= 534
 vTAG[1] <= 534
 vTAG[2] <= 534
 vTAG[3] <= 534
 1 <= vBAG[0] <= 535
 1 <= vBAG[1] <= 535
 1 <= vBAG[2] <= 535
 1 <= vBAG[3] <= 535
Binaries
 ABOVE[0,0] ABOVE[0,1] ABOVE[0,2] ABOVE[0,3] ABOVE[1,0] ABOVE[1,1]
 ABOVE[1,2] ABOVE[1,3] ABOVE[2,0] ABOVE[2,1] ABOVE[2,2] ABOVE[2,3]
 ABOVE[3,0] ABOVE[3,1] ABOVE[3,2] ABOVE[3,3] LEFT[0,0] LEFT[0,1] LEFT[0,2]
 LEFT[0,3] LEFT[1,0] LEFT[1,1] LEFT[1,2] LEFT[1,3] LEFT[2,0] LEFT[2,1]
 LEFT[2,2] LEFT[2,3] LEFT[3,0] LEFT[3,1] LEFT[3,2] LEFT[3,3] LAG[0] LAG[1]
 LAG[2] LAG[3] RAG[0] RAG[1] RAG[2] RAG[3] TAG[0] TAG[1] TAG[2] TAG[3]
 BAG[0] BAG[1] BAG[2] BAG[3] zLAG[0,0] zLAG[0,1] zLAG[0,2] zLAG[0,3]
 zLAG[1,0] zLAG[1,1] zLAG[1,2] zLAG[1,3] zLAG[2,0] zLAG[2,1] zLAG[2,2]
 zLAG[2,3] zLAG[3,0] zLAG[3,1] zLAG[3,2] zLAG[3,3] zRAG[0,0] zRAG[0,1]
 zRAG[0,2] zRAG[0,3] zRAG[1,0] zRAG[1,1] zRAG[1,2] zRAG[1,3] zRAG[2,0]
 zRAG[2,1] zRAG[2,2] zRAG[2,3] zRAG[3,0] zRAG[3,1] zRAG[3,2] zRAG[3,3]
 zTAG[0,0] zTAG[0,1] zTAG[0,2] zTAG[0,3] zTAG[1,0] zTAG[1,1] zTAG[1,2]
 zTAG[1,3] zTAG[2,0] zTAG[2,1] zTAG[2,2] zTAG[2,3] zTAG[3,0] zTAG[3,1]
 zTAG[3,2] zTAG[3,3] zBAG[0,0] zBAG[0,1] zBAG[0,2] zBAG[0,3] zBAG[1,0]
 zBAG[1,1] zBAG[1,2] zBAG[1,3] zBAG[2,0] zBAG[2,1] zBAG[2,2] zBAG[2,3]
 zBAG[3,0] zBAG[3,1] zBAG[3,2] zBAG[3,3]
Generals
 L[0] L[1] L[2] L[3] R[0] R[1] R[2] R[3] T[0] T[1] T[2] T[3] B[0] B[1] B[2]
 B[3] H[0] H[1] H[2] H[3] W[0] W[1] W[2] W[3] vLAG[0] vLAG[1] vLAG[2]
 vLAG[3] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vTAG[0] vTAG[1] vTAG[2] vTAG[3]
 vBAG[0] vBAG[1] vBAG[2] vBAG[3] maxX maxY
End
