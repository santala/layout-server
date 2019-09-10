\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 L[0] + 0.001 L[1] + 0.001 L[2] + 0.001 L[3] + 0.002 R[0]
   + 0.002 R[1] + 0.002 R[2] + 0.002 R[3] + 0.001 T[0] + 0.001 T[1]
   + 0.001 T[2] + 0.001 T[3] + 0.002 B[0] + 0.002 B[1] + 0.002 B[2]
   + 0.002 B[3] - 0.001 H[0] - 0.001 H[1] - 0.001 H[2] - 0.001 H[3]
   - 0.001 W[0] - 0.001 W[1] - 0.001 W[2] - 0.001 W[3]
   + 0.4629629629629629 resizeWAbs[0] + 0.390625 resizeWAbs[1]
   + 0.4166666666666667 resizeWAbs[2] + 0.4310344827586207 resizeWAbs[3]
   + 0.3676470588235294 resizeHAbs[0] + 0.3676470588235294 resizeHAbs[1]
   + 0.4807692307692308 resizeHAbs[2] + 0.4807692307692308 resizeHAbs[3]
   + 0.0462962962962963 moveXAbs[0] + 0.0390625 moveXAbs[1]
   + 0.0416666666666667 moveXAbs[2] + 0.0431034482758621 moveXAbs[3]
   + 0.0367647058823529 moveYAbs[0] + 0.0367647058823529 moveYAbs[1]
   + 0.0480769230769231 moveYAbs[2] + 0.0480769230769231 moveYAbs[3]
   + 0 ABOVE[0,0] + 0 ABOVE[1,1] + 0 ABOVE[2,2] + 0 ABOVE[3,3]
   + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2] + 0 LEFT[3,3] + LAG[0]
   + LAG[1] + LAG[2] + LAG[3] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + TAG[0]
   + TAG[1] + TAG[2] + TAG[3] + BAG[0] + BAG[1] + BAG[2] + BAG[3]
Subject To
 resizeW: - W[0] + resizeW[0] = -27
 resizeH: - H[0] + resizeH[0] = -34
 moveX: - L[0] + moveX[0] = -59
 moveY: - T[0] + moveY[0] = -11
 resizeW: - W[1] + resizeW[1] = -32
 resizeH: - H[1] + resizeH[1] = -34
 moveX: - L[1] + moveX[1] = -3
 moveY: - T[1] + moveY[1] = -62
 resizeW: - W[2] + resizeW[2] = -30
 resizeH: - H[2] + resizeH[2] = -26
 moveX: - L[2] + moveX[2] = -56
 moveY: - T[2] + moveY[2] = -62
 resizeW: - W[3] + resizeW[3] = -29
 resizeH: - H[3] + resizeH[3] = -26
 moveX: - L[3] + moveX[3] = -11
 moveY: - T[3] + moveY[3] = -15
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
 0(ToLeftOf)1: - L[1] + R[0] + 862 LEFT[0,1] <= 830
 0(Above)1: - T[1] + B[0] + 996 ABOVE[0,1] <= 964
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 862 LEFT[0,1] <= 32
 0(ConverseOfAboveOf)1: T[1] - B[0] - 996 ABOVE[0,1] <= 32
 0(ToLeftOf)2: - L[2] + R[0] + 862 LEFT[0,2] <= 830
 0(Above)2: - T[2] + B[0] + 996 ABOVE[0,2] <= 964
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 862 LEFT[0,2] <= 32
 0(ConverseOfAboveOf)2: T[2] - B[0] - 996 ABOVE[0,2] <= 32
 0(ToLeftOf)3: - L[3] + R[0] + 862 LEFT[0,3] <= 830
 0(Above)3: - T[3] + B[0] + 996 ABOVE[0,3] <= 964
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 862 LEFT[0,3] <= 32
 0(ConverseOfAboveOf)3: T[3] - B[0] - 996 ABOVE[0,3] <= 32
 1(ToLeftOf)0: - L[0] + R[1] + 862 LEFT[1,0] <= 830
 1(Above)0: - T[0] + B[1] + 996 ABOVE[1,0] <= 964
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 862 LEFT[1,0] <= 32
 1(ConverseOfAboveOf)0: T[0] - B[1] - 996 ABOVE[1,0] <= 32
 1(ToLeftOf)2: - L[2] + R[1] + 862 LEFT[1,2] <= 830
 1(Above)2: - T[2] + B[1] + 996 ABOVE[1,2] <= 964
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 862 LEFT[1,2] <= 32
 1(ConverseOfAboveOf)2: T[2] - B[1] - 996 ABOVE[1,2] <= 32
 1(ToLeftOf)3: - L[3] + R[1] + 862 LEFT[1,3] <= 830
 1(Above)3: - T[3] + B[1] + 996 ABOVE[1,3] <= 964
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 862 LEFT[1,3] <= 32
 1(ConverseOfAboveOf)3: T[3] - B[1] - 996 ABOVE[1,3] <= 32
 2(ToLeftOf)0: - L[0] + R[2] + 862 LEFT[2,0] <= 830
 2(Above)0: - T[0] + B[2] + 996 ABOVE[2,0] <= 964
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 862 LEFT[2,0] <= 32
 2(ConverseOfAboveOf)0: T[0] - B[2] - 996 ABOVE[2,0] <= 32
 2(ToLeftOf)1: - L[1] + R[2] + 862 LEFT[2,1] <= 830
 2(Above)1: - T[1] + B[2] + 996 ABOVE[2,1] <= 964
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 862 LEFT[2,1] <= 32
 2(ConverseOfAboveOf)1: T[1] - B[2] - 996 ABOVE[2,1] <= 32
 2(ToLeftOf)3: - L[3] + R[2] + 862 LEFT[2,3] <= 830
 2(Above)3: - T[3] + B[2] + 996 ABOVE[2,3] <= 964
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 862 LEFT[2,3] <= 32
 2(ConverseOfAboveOf)3: T[3] - B[2] - 996 ABOVE[2,3] <= 32
 3(ToLeftOf)0: - L[0] + R[3] + 862 LEFT[3,0] <= 830
 3(Above)0: - T[0] + B[3] + 996 ABOVE[3,0] <= 964
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 862 LEFT[3,0] <= 32
 3(ConverseOfAboveOf)0: T[0] - B[3] - 996 ABOVE[3,0] <= 32
 3(ToLeftOf)1: - L[1] + R[3] + 862 LEFT[3,1] <= 830
 3(Above)1: - T[1] + B[3] + 996 ABOVE[3,1] <= 964
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 862 LEFT[3,1] <= 32
 3(ConverseOfAboveOf)1: T[1] - B[3] - 996 ABOVE[3,1] <= 32
 3(ToLeftOf)2: - L[2] + R[3] + 862 LEFT[3,2] <= 830
 3(Above)2: - T[2] + B[3] + 996 ABOVE[3,2] <= 964
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 862 LEFT[3,2] <= 32
 3(ConverseOfAboveOf)2: T[2] - B[3] - 996 ABOVE[3,2] <= 32
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
 R113: - LAG[0] + zLAG[0,0] <= 0
 R114: - RAG[0] + zRAG[0,0] <= 0
 R115: - TAG[0] + zTAG[0,0] <= 0
 R116: - BAG[0] + zBAG[0,0] <= 0
 R117: - LAG[0] + zLAG[1,0] <= 0
 R118: - RAG[0] + zRAG[1,0] <= 0
 R119: - TAG[0] + zTAG[1,0] <= 0
 R120: - BAG[0] + zBAG[1,0] <= 0
 R121: - LAG[0] + zLAG[2,0] <= 0
 R122: - RAG[0] + zRAG[2,0] <= 0
 R123: - TAG[0] + zTAG[2,0] <= 0
 R124: - BAG[0] + zBAG[2,0] <= 0
 R125: - LAG[0] + zLAG[3,0] <= 0
 R126: - RAG[0] + zRAG[3,0] <= 0
 R127: - TAG[0] + zTAG[3,0] <= 0
 R128: - BAG[0] + zBAG[3,0] <= 0
 R129: - LAG[1] + zLAG[0,1] <= 0
 R130: - RAG[1] + zRAG[0,1] <= 0
 R131: - TAG[1] + zTAG[0,1] <= 0
 R132: - BAG[1] + zBAG[0,1] <= 0
 R133: - LAG[1] + zLAG[1,1] <= 0
 R134: - RAG[1] + zRAG[1,1] <= 0
 R135: - TAG[1] + zTAG[1,1] <= 0
 R136: - BAG[1] + zBAG[1,1] <= 0
 R137: - LAG[1] + zLAG[2,1] <= 0
 R138: - RAG[1] + zRAG[2,1] <= 0
 R139: - TAG[1] + zTAG[2,1] <= 0
 R140: - BAG[1] + zBAG[2,1] <= 0
 R141: - LAG[1] + zLAG[3,1] <= 0
 R142: - RAG[1] + zRAG[3,1] <= 0
 R143: - TAG[1] + zTAG[3,1] <= 0
 R144: - BAG[1] + zBAG[3,1] <= 0
 R145: - LAG[2] + zLAG[0,2] <= 0
 R146: - RAG[2] + zRAG[0,2] <= 0
 R147: - TAG[2] + zTAG[0,2] <= 0
 R148: - BAG[2] + zBAG[0,2] <= 0
 R149: - LAG[2] + zLAG[1,2] <= 0
 R150: - RAG[2] + zRAG[1,2] <= 0
 R151: - TAG[2] + zTAG[1,2] <= 0
 R152: - BAG[2] + zBAG[1,2] <= 0
 R153: - LAG[2] + zLAG[2,2] <= 0
 R154: - RAG[2] + zRAG[2,2] <= 0
 R155: - TAG[2] + zTAG[2,2] <= 0
 R156: - BAG[2] + zBAG[2,2] <= 0
 R157: - LAG[2] + zLAG[3,2] <= 0
 R158: - RAG[2] + zRAG[3,2] <= 0
 R159: - TAG[2] + zTAG[3,2] <= 0
 R160: - BAG[2] + zBAG[3,2] <= 0
 R161: - LAG[3] + zLAG[0,3] <= 0
 R162: - RAG[3] + zRAG[0,3] <= 0
 R163: - TAG[3] + zTAG[0,3] <= 0
 R164: - BAG[3] + zBAG[0,3] <= 0
 R165: - LAG[3] + zLAG[1,3] <= 0
 R166: - RAG[3] + zRAG[1,3] <= 0
 R167: - TAG[3] + zTAG[1,3] <= 0
 R168: - BAG[3] + zBAG[1,3] <= 0
 R169: - LAG[3] + zLAG[2,3] <= 0
 R170: - RAG[3] + zRAG[2,3] <= 0
 R171: - TAG[3] + zTAG[2,3] <= 0
 R172: - BAG[3] + zBAG[2,3] <= 0
 R173: - LAG[3] + zLAG[3,3] <= 0
 R174: - RAG[3] + zRAG[3,3] <= 0
 R175: - TAG[3] + zTAG[3,3] <= 0
 R176: - BAG[3] + zBAG[3,3] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 862 zLAG[0,0] <= 862
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 862 zRAG[0,0] <= 862
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 996 zTAG[0,0] <= 996
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 996 zBAG[0,0] <= 996
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 862 zLAG[0,0] >= -862
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 862 zRAG[0,0] >= -862
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 996 zTAG[0,0] >= -996
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 996 zBAG[0,0] >= -996
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 862 zLAG[1,0] <= 862
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 862 zRAG[1,0] <= 862
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 996 zTAG[1,0] <= 996
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 996 zBAG[1,0] <= 996
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 862 zLAG[1,0] >= -862
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 862 zRAG[1,0] >= -862
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 996 zTAG[1,0] >= -996
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 996 zBAG[1,0] >= -996
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 862 zLAG[2,0] <= 862
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 862 zRAG[2,0] <= 862
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 996 zTAG[2,0] <= 996
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 996 zBAG[2,0] <= 996
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 862 zLAG[2,0] >= -862
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 862 zRAG[2,0] >= -862
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 996 zTAG[2,0] >= -996
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 996 zBAG[2,0] >= -996
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 862 zLAG[3,0] <= 862
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 862 zRAG[3,0] <= 862
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 996 zTAG[3,0] <= 996
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 996 zBAG[3,0] <= 996
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 862 zLAG[3,0] >= -862
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 862 zRAG[3,0] >= -862
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 996 zTAG[3,0] >= -996
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 996 zBAG[3,0] >= -996
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 862 zLAG[0,1] <= 862
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 862 zRAG[0,1] <= 862
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 996 zTAG[0,1] <= 996
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 996 zBAG[0,1] <= 996
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 862 zLAG[0,1] >= -862
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 862 zRAG[0,1] >= -862
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 996 zTAG[0,1] >= -996
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 996 zBAG[0,1] >= -996
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 862 zLAG[1,1] <= 862
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 862 zRAG[1,1] <= 862
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 996 zTAG[1,1] <= 996
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 996 zBAG[1,1] <= 996
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 862 zLAG[1,1] >= -862
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 862 zRAG[1,1] >= -862
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 996 zTAG[1,1] >= -996
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 996 zBAG[1,1] >= -996
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 862 zLAG[2,1] <= 862
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 862 zRAG[2,1] <= 862
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 996 zTAG[2,1] <= 996
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 996 zBAG[2,1] <= 996
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 862 zLAG[2,1] >= -862
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 862 zRAG[2,1] >= -862
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 996 zTAG[2,1] >= -996
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 996 zBAG[2,1] >= -996
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 862 zLAG[3,1] <= 862
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 862 zRAG[3,1] <= 862
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 996 zTAG[3,1] <= 996
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 996 zBAG[3,1] <= 996
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 862 zLAG[3,1] >= -862
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 862 zRAG[3,1] >= -862
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 996 zTAG[3,1] >= -996
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 996 zBAG[3,1] >= -996
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 862 zLAG[0,2] <= 862
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 862 zRAG[0,2] <= 862
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 996 zTAG[0,2] <= 996
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 996 zBAG[0,2] <= 996
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 862 zLAG[0,2] >= -862
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 862 zRAG[0,2] >= -862
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 996 zTAG[0,2] >= -996
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 996 zBAG[0,2] >= -996
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 862 zLAG[1,2] <= 862
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 862 zRAG[1,2] <= 862
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 996 zTAG[1,2] <= 996
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 996 zBAG[1,2] <= 996
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 862 zLAG[1,2] >= -862
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 862 zRAG[1,2] >= -862
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 996 zTAG[1,2] >= -996
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 996 zBAG[1,2] >= -996
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 862 zLAG[2,2] <= 862
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 862 zRAG[2,2] <= 862
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 996 zTAG[2,2] <= 996
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 996 zBAG[2,2] <= 996
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 862 zLAG[2,2] >= -862
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 862 zRAG[2,2] >= -862
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 996 zTAG[2,2] >= -996
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 996 zBAG[2,2] >= -996
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 862 zLAG[3,2] <= 862
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 862 zRAG[3,2] <= 862
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 996 zTAG[3,2] <= 996
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 996 zBAG[3,2] <= 996
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 862 zLAG[3,2] >= -862
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 862 zRAG[3,2] >= -862
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 996 zTAG[3,2] >= -996
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 996 zBAG[3,2] >= -996
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 862 zLAG[0,3] <= 862
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 862 zRAG[0,3] <= 862
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 996 zTAG[0,3] <= 996
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 996 zBAG[0,3] <= 996
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 862 zLAG[0,3] >= -862
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 862 zRAG[0,3] >= -862
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 996 zTAG[0,3] >= -996
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 996 zBAG[0,3] >= -996
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 862 zLAG[1,3] <= 862
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 862 zRAG[1,3] <= 862
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 996 zTAG[1,3] <= 996
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 996 zBAG[1,3] <= 996
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 862 zLAG[1,3] >= -862
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 862 zRAG[1,3] >= -862
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 996 zTAG[1,3] >= -996
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 996 zBAG[1,3] >= -996
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 862 zLAG[2,3] <= 862
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 862 zRAG[2,3] <= 862
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 996 zTAG[2,3] <= 996
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 996 zBAG[2,3] <= 996
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 862 zLAG[2,3] >= -862
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 862 zRAG[2,3] >= -862
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 996 zTAG[2,3] >= -996
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 996 zBAG[2,3] >= -996
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 862 zLAG[3,3] <= 862
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 862 zRAG[3,3] <= 862
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 996 zTAG[3,3] <= 996
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 996 zBAG[3,3] <= 996
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 862 zLAG[3,3] >= -862
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 862 zRAG[3,3] >= -862
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 996 zTAG[3,3] >= -996
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 996 zBAG[3,3] >= -996
 R305: - R[0] + maxX >= 0
 R306: - B[0] + maxY >= 0
 R307: - R[1] + maxX >= 0
 R308: - B[1] + maxY >= 0
 R309: - R[2] + maxX >= 0
 R310: - B[2] + maxY >= 0
 R311: - R[3] + maxX >= 0
 R312: - B[3] + maxY >= 0
 R313: LAG[0] + LAG[1] + LAG[2] + LAG[3] + RAG[0] + RAG[1] + RAG[2]
   + RAG[3] + TAG[0] + TAG[1] + TAG[2] + TAG[3] + BAG[0] + BAG[1] + BAG[2]
   + BAG[3] >= 8
Bounds
 L[0] <= 107.625
 L[1] <= 107.625
 L[2] <= 107.625
 L[3] <= 107.625
 1 <= R[0] <= 107.75
 1 <= R[1] <= 107.75
 1 <= R[2] <= 107.75
 1 <= R[3] <= 107.75
 T[0] <= 124.375
 T[1] <= 124.375
 T[2] <= 124.375
 T[3] <= 124.375
 1 <= B[0] <= 124.5
 1 <= B[1] <= 124.5
 1 <= B[2] <= 124.5
 1 <= B[3] <= 124.5
 1 <= H[0] <= 125
 1 <= H[1] <= 125
 1 <= H[2] <= 125
 1 <= H[3] <= 125
 1 <= W[0] <= 108
 1 <= W[1] <= 108
 1 <= W[2] <= 108
 1 <= W[3] <= 108
 resizeW[0] >= -1000
 resizeW[1] >= -1000
 resizeW[2] >= -1000
 resizeW[3] >= -1000
 resizeH[0] >= -1000
 resizeH[1] >= -1000
 resizeH[2] >= -1000
 resizeH[3] >= -1000
 moveX[0] >= -1000
 moveX[1] >= -1000
 moveX[2] >= -1000
 moveX[3] >= -1000
 moveY[0] >= -1000
 moveY[1] >= -1000
 moveY[2] >= -1000
 moveY[3] >= -1000
 vLAG[0] <= 106.75
 vLAG[1] <= 106.75
 vLAG[2] <= 106.75
 vLAG[3] <= 106.75
 1 <= vRAG[0] <= 107.75
 1 <= vRAG[1] <= 107.75
 1 <= vRAG[2] <= 107.75
 1 <= vRAG[3] <= 107.75
 vTAG[0] <= 123.5
 vTAG[1] <= 123.5
 vTAG[2] <= 123.5
 vTAG[3] <= 123.5
 1 <= vBAG[0] <= 124.5
 1 <= vBAG[1] <= 124.5
 1 <= vBAG[2] <= 124.5
 1 <= vBAG[3] <= 124.5
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
 B[3] H[0] H[1] H[2] H[3] W[0] W[1] W[2] W[3] resizeW[0] resizeW[1]
 resizeW[2] resizeW[3] resizeWAbs[0] resizeWAbs[1] resizeWAbs[2]
 resizeWAbs[3] resizeH[0] resizeH[1] resizeH[2] resizeH[3] resizeHAbs[0]
 resizeHAbs[1] resizeHAbs[2] resizeHAbs[3] moveX[0] moveX[1] moveX[2]
 moveX[3] moveXAbs[0] moveXAbs[1] moveXAbs[2] moveXAbs[3] moveY[0] moveY[1]
 moveY[2] moveY[3] moveYAbs[0] moveYAbs[1] moveYAbs[2] moveYAbs[3] vLAG[0]
 vLAG[1] vLAG[2] vLAG[3] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vTAG[0] vTAG[1]
 vTAG[2] vTAG[3] vBAG[0] vBAG[1] vBAG[2] vBAG[3] maxX maxY
General Constraints
 resizeWAbs: resizeWAbs[0] = ABS ( resizeW[0] )
 resizeHAbs: resizeHAbs[0] = ABS ( resizeH[0] )
 moveXAbs: moveXAbs[0] = ABS ( moveX[0] )
 moveYAbs: moveYAbs[0] = ABS ( moveY[0] )
 resizeWAbs: resizeWAbs[1] = ABS ( resizeW[1] )
 resizeHAbs: resizeHAbs[1] = ABS ( resizeH[1] )
 moveXAbs: moveXAbs[1] = ABS ( moveX[1] )
 moveYAbs: moveYAbs[1] = ABS ( moveY[1] )
 resizeWAbs: resizeWAbs[2] = ABS ( resizeW[2] )
 resizeHAbs: resizeHAbs[2] = ABS ( resizeH[2] )
 moveXAbs: moveXAbs[2] = ABS ( moveX[2] )
 moveYAbs: moveYAbs[2] = ABS ( moveY[2] )
 resizeWAbs: resizeWAbs[3] = ABS ( resizeW[3] )
 resizeHAbs: resizeHAbs[3] = ABS ( resizeH[3] )
 moveXAbs: moveXAbs[3] = ABS ( moveX[3] )
 moveYAbs: moveYAbs[3] = ABS ( moveY[3] )
End
