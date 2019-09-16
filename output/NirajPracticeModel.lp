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
   - 0.001 W[5] - 0.001 W[6] + 0.2257336343115124 resizeWAbs[0]
   + 0.2257336343115124 resizeWAbs[1] + 0.390625 resizeWAbs[2]
   + 0.4166666666666667 resizeWAbs[3] + 0.4166666666666667 resizeWAbs[4]
   + 0.4166666666666667 resizeWAbs[5] + 0.390625 resizeWAbs[6]
   + 0.3257328990228013 resizeHAbs[0] + 0.3257328990228013 resizeHAbs[1]
   + 0.3676470588235294 resizeHAbs[2] + 0.4807692307692308 resizeHAbs[3]
   + 0.7751937984496124 resizeHAbs[4] + 0.7751937984496124 resizeHAbs[5]
   + 0.4149377593360996 resizeHAbs[6] + 0.0225733634311512 moveXAbs[0]
   + 0.0225733634311512 moveXAbs[1] + 0.0390625 moveXAbs[2]
   + 0.0416666666666667 moveXAbs[3] + 0.0416666666666667 moveXAbs[4]
   + 0.0416666666666667 moveXAbs[5] + 0.0390625 moveXAbs[6]
   + 0.0325732899022801 moveYAbs[0] + 0.0325732899022801 moveYAbs[1]
   + 0.0367647058823529 moveYAbs[2] + 0.0480769230769231 moveYAbs[3]
   + 0.0775193798449612 moveYAbs[4] + 0.0775193798449612 moveYAbs[5]
   + 0.04149377593361 moveYAbs[6] + 0 ABOVE[0,0] + 0 ABOVE[1,1]
   + 0 ABOVE[2,2] + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 ABOVE[5,5]
   + 0 ABOVE[6,6] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2] + 0 LEFT[3,3]
   + 0 LEFT[4,4] + 0 LEFT[5,5] + 0 LEFT[6,6] + LAG[0] + LAG[1] + LAG[2]
   + LAG[3] + LAG[4] + LAG[5] + LAG[6] + RAG[0] + RAG[1] + RAG[2] + RAG[3]
   + RAG[4] + RAG[5] + RAG[6] + TAG[0] + TAG[1] + TAG[2] + TAG[3] + TAG[4]
   + TAG[5] + TAG[6] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4] + BAG[5]
   + BAG[6]
Subject To
 resizeW: - W[0] + resizeW[0] = -56
 resizeH: - H[0] + resizeH[0] = -39
 moveX: - L[0] + moveX[0] = -40
 moveY: - T[0] + moveY[0] = -14
 resizeW: - W[1] + resizeW[1] = -56
 resizeH: - H[1] + resizeH[1] = -39
 moveX: - L[1] + moveX[1] = -34
 moveY: - T[1] + moveY[1] = -73
 resizeW: - W[2] + resizeW[2] = -32
 resizeH: - H[2] + resizeH[2] = -34
 moveX: - L[2] + moveX[2] = -3
 moveY: - T[2] + moveY[2] = -57
 resizeW: - W[3] + resizeW[3] = -30
 resizeH: - H[3] + resizeH[3] = -26
 moveX: - L[3] + moveX[3] = -66
 moveY: - T[3] + moveY[3] = -69
 resizeW: - W[4] + resizeW[4] = -30
 resizeH: - H[4] + resizeH[4] = -17
 moveX: - L[4] + moveX[4] = -32
 moveY: - T[4] + moveY[4] = -87
 resizeW: - W[5] + resizeW[5] = -30
 resizeH: - H[5] + resizeH[5] = -17
 moveX: - L[5] + moveX[5] = -8
 moveY: - T[5] + moveY[5] = -41
 resizeW: - W[6] + resizeW[6] = -32
 resizeH: - H[6] + resizeH[6] = -31
 moveX: - L[6] + moveX[6] = -3
 moveY: - T[6] + moveY[6] = -8
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
 R-L=W(6): L[6] - R[6] + W[6] = 0
 B-T=H(6): T[6] - B[6] + H[6] = 0
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
 NoOverlap(56): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5] >= 1
 UpperLimOfQuadrants(56): ABOVE[5,6] + ABOVE[6,5] + LEFT[5,6] + LEFT[6,5]
   <= 2
 Anti-symmetryABOVE(56): ABOVE[5,6] + ABOVE[6,5] <= 1
 Anti-symmetryLEFT(56): LEFT[5,6] + LEFT[6,5] <= 1
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
 0(ToLeftOf)4: - L[4] + R[0] + 862 LEFT[0,4] <= 830
 0(Above)4: - T[4] + B[0] + 996 ABOVE[0,4] <= 964
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 862 LEFT[0,4] <= 32
 0(ConverseOfAboveOf)4: T[4] - B[0] - 996 ABOVE[0,4] <= 32
 0(ToLeftOf)5: - L[5] + R[0] + 862 LEFT[0,5] <= 830
 0(Above)5: - T[5] + B[0] + 996 ABOVE[0,5] <= 964
 0(ConverseOfToLeftOf)5: L[5] - R[0] - 862 LEFT[0,5] <= 32
 0(ConverseOfAboveOf)5: T[5] - B[0] - 996 ABOVE[0,5] <= 32
 0(ToLeftOf)6: - L[6] + R[0] + 862 LEFT[0,6] <= 830
 0(Above)6: - T[6] + B[0] + 996 ABOVE[0,6] <= 964
 0(ConverseOfToLeftOf)6: L[6] - R[0] - 862 LEFT[0,6] <= 32
 0(ConverseOfAboveOf)6: T[6] - B[0] - 996 ABOVE[0,6] <= 32
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
 1(ToLeftOf)4: - L[4] + R[1] + 862 LEFT[1,4] <= 830
 1(Above)4: - T[4] + B[1] + 996 ABOVE[1,4] <= 964
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 862 LEFT[1,4] <= 32
 1(ConverseOfAboveOf)4: T[4] - B[1] - 996 ABOVE[1,4] <= 32
 1(ToLeftOf)5: - L[5] + R[1] + 862 LEFT[1,5] <= 830
 1(Above)5: - T[5] + B[1] + 996 ABOVE[1,5] <= 964
 1(ConverseOfToLeftOf)5: L[5] - R[1] - 862 LEFT[1,5] <= 32
 1(ConverseOfAboveOf)5: T[5] - B[1] - 996 ABOVE[1,5] <= 32
 1(ToLeftOf)6: - L[6] + R[1] + 862 LEFT[1,6] <= 830
 1(Above)6: - T[6] + B[1] + 996 ABOVE[1,6] <= 964
 1(ConverseOfToLeftOf)6: L[6] - R[1] - 862 LEFT[1,6] <= 32
 1(ConverseOfAboveOf)6: T[6] - B[1] - 996 ABOVE[1,6] <= 32
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
 2(ToLeftOf)4: - L[4] + R[2] + 862 LEFT[2,4] <= 830
 2(Above)4: - T[4] + B[2] + 996 ABOVE[2,4] <= 964
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 862 LEFT[2,4] <= 32
 2(ConverseOfAboveOf)4: T[4] - B[2] - 996 ABOVE[2,4] <= 32
 2(ToLeftOf)5: - L[5] + R[2] + 862 LEFT[2,5] <= 830
 2(Above)5: - T[5] + B[2] + 996 ABOVE[2,5] <= 964
 2(ConverseOfToLeftOf)5: L[5] - R[2] - 862 LEFT[2,5] <= 32
 2(ConverseOfAboveOf)5: T[5] - B[2] - 996 ABOVE[2,5] <= 32
 2(ToLeftOf)6: - L[6] + R[2] + 862 LEFT[2,6] <= 830
 2(Above)6: - T[6] + B[2] + 996 ABOVE[2,6] <= 964
 2(ConverseOfToLeftOf)6: L[6] - R[2] - 862 LEFT[2,6] <= 32
 2(ConverseOfAboveOf)6: T[6] - B[2] - 996 ABOVE[2,6] <= 32
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
 3(ToLeftOf)4: - L[4] + R[3] + 862 LEFT[3,4] <= 830
 3(Above)4: - T[4] + B[3] + 996 ABOVE[3,4] <= 964
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 862 LEFT[3,4] <= 32
 3(ConverseOfAboveOf)4: T[4] - B[3] - 996 ABOVE[3,4] <= 32
 3(ToLeftOf)5: - L[5] + R[3] + 862 LEFT[3,5] <= 830
 3(Above)5: - T[5] + B[3] + 996 ABOVE[3,5] <= 964
 3(ConverseOfToLeftOf)5: L[5] - R[3] - 862 LEFT[3,5] <= 32
 3(ConverseOfAboveOf)5: T[5] - B[3] - 996 ABOVE[3,5] <= 32
 3(ToLeftOf)6: - L[6] + R[3] + 862 LEFT[3,6] <= 830
 3(Above)6: - T[6] + B[3] + 996 ABOVE[3,6] <= 964
 3(ConverseOfToLeftOf)6: L[6] - R[3] - 862 LEFT[3,6] <= 32
 3(ConverseOfAboveOf)6: T[6] - B[3] - 996 ABOVE[3,6] <= 32
 4(ToLeftOf)0: - L[0] + R[4] + 862 LEFT[4,0] <= 830
 4(Above)0: - T[0] + B[4] + 996 ABOVE[4,0] <= 964
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 862 LEFT[4,0] <= 32
 4(ConverseOfAboveOf)0: T[0] - B[4] - 996 ABOVE[4,0] <= 32
 4(ToLeftOf)1: - L[1] + R[4] + 862 LEFT[4,1] <= 830
 4(Above)1: - T[1] + B[4] + 996 ABOVE[4,1] <= 964
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 862 LEFT[4,1] <= 32
 4(ConverseOfAboveOf)1: T[1] - B[4] - 996 ABOVE[4,1] <= 32
 4(ToLeftOf)2: - L[2] + R[4] + 862 LEFT[4,2] <= 830
 4(Above)2: - T[2] + B[4] + 996 ABOVE[4,2] <= 964
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 862 LEFT[4,2] <= 32
 4(ConverseOfAboveOf)2: T[2] - B[4] - 996 ABOVE[4,2] <= 32
 4(ToLeftOf)3: - L[3] + R[4] + 862 LEFT[4,3] <= 830
 4(Above)3: - T[3] + B[4] + 996 ABOVE[4,3] <= 964
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 862 LEFT[4,3] <= 32
 4(ConverseOfAboveOf)3: T[3] - B[4] - 996 ABOVE[4,3] <= 32
 4(ToLeftOf)5: - L[5] + R[4] + 862 LEFT[4,5] <= 830
 4(Above)5: - T[5] + B[4] + 996 ABOVE[4,5] <= 964
 4(ConverseOfToLeftOf)5: L[5] - R[4] - 862 LEFT[4,5] <= 32
 4(ConverseOfAboveOf)5: T[5] - B[4] - 996 ABOVE[4,5] <= 32
 4(ToLeftOf)6: - L[6] + R[4] + 862 LEFT[4,6] <= 830
 4(Above)6: - T[6] + B[4] + 996 ABOVE[4,6] <= 964
 4(ConverseOfToLeftOf)6: L[6] - R[4] - 862 LEFT[4,6] <= 32
 4(ConverseOfAboveOf)6: T[6] - B[4] - 996 ABOVE[4,6] <= 32
 5(ToLeftOf)0: - L[0] + R[5] + 862 LEFT[5,0] <= 830
 5(Above)0: - T[0] + B[5] + 996 ABOVE[5,0] <= 964
 5(ConverseOfToLeftOf)0: L[0] - R[5] - 862 LEFT[5,0] <= 32
 5(ConverseOfAboveOf)0: T[0] - B[5] - 996 ABOVE[5,0] <= 32
 5(ToLeftOf)1: - L[1] + R[5] + 862 LEFT[5,1] <= 830
 5(Above)1: - T[1] + B[5] + 996 ABOVE[5,1] <= 964
 5(ConverseOfToLeftOf)1: L[1] - R[5] - 862 LEFT[5,1] <= 32
 5(ConverseOfAboveOf)1: T[1] - B[5] - 996 ABOVE[5,1] <= 32
 5(ToLeftOf)2: - L[2] + R[5] + 862 LEFT[5,2] <= 830
 5(Above)2: - T[2] + B[5] + 996 ABOVE[5,2] <= 964
 5(ConverseOfToLeftOf)2: L[2] - R[5] - 862 LEFT[5,2] <= 32
 5(ConverseOfAboveOf)2: T[2] - B[5] - 996 ABOVE[5,2] <= 32
 5(ToLeftOf)3: - L[3] + R[5] + 862 LEFT[5,3] <= 830
 5(Above)3: - T[3] + B[5] + 996 ABOVE[5,3] <= 964
 5(ConverseOfToLeftOf)3: L[3] - R[5] - 862 LEFT[5,3] <= 32
 5(ConverseOfAboveOf)3: T[3] - B[5] - 996 ABOVE[5,3] <= 32
 5(ToLeftOf)4: - L[4] + R[5] + 862 LEFT[5,4] <= 830
 5(Above)4: - T[4] + B[5] + 996 ABOVE[5,4] <= 964
 5(ConverseOfToLeftOf)4: L[4] - R[5] - 862 LEFT[5,4] <= 32
 5(ConverseOfAboveOf)4: T[4] - B[5] - 996 ABOVE[5,4] <= 32
 5(ToLeftOf)6: - L[6] + R[5] + 862 LEFT[5,6] <= 830
 5(Above)6: - T[6] + B[5] + 996 ABOVE[5,6] <= 964
 5(ConverseOfToLeftOf)6: L[6] - R[5] - 862 LEFT[5,6] <= 32
 5(ConverseOfAboveOf)6: T[6] - B[5] - 996 ABOVE[5,6] <= 32
 6(ToLeftOf)0: - L[0] + R[6] + 862 LEFT[6,0] <= 830
 6(Above)0: - T[0] + B[6] + 996 ABOVE[6,0] <= 964
 6(ConverseOfToLeftOf)0: L[0] - R[6] - 862 LEFT[6,0] <= 32
 6(ConverseOfAboveOf)0: T[0] - B[6] - 996 ABOVE[6,0] <= 32
 6(ToLeftOf)1: - L[1] + R[6] + 862 LEFT[6,1] <= 830
 6(Above)1: - T[1] + B[6] + 996 ABOVE[6,1] <= 964
 6(ConverseOfToLeftOf)1: L[1] - R[6] - 862 LEFT[6,1] <= 32
 6(ConverseOfAboveOf)1: T[1] - B[6] - 996 ABOVE[6,1] <= 32
 6(ToLeftOf)2: - L[2] + R[6] + 862 LEFT[6,2] <= 830
 6(Above)2: - T[2] + B[6] + 996 ABOVE[6,2] <= 964
 6(ConverseOfToLeftOf)2: L[2] - R[6] - 862 LEFT[6,2] <= 32
 6(ConverseOfAboveOf)2: T[2] - B[6] - 996 ABOVE[6,2] <= 32
 6(ToLeftOf)3: - L[3] + R[6] + 862 LEFT[6,3] <= 830
 6(Above)3: - T[3] + B[6] + 996 ABOVE[6,3] <= 964
 6(ConverseOfToLeftOf)3: L[3] - R[6] - 862 LEFT[6,3] <= 32
 6(ConverseOfAboveOf)3: T[3] - B[6] - 996 ABOVE[6,3] <= 32
 6(ToLeftOf)4: - L[4] + R[6] + 862 LEFT[6,4] <= 830
 6(Above)4: - T[4] + B[6] + 996 ABOVE[6,4] <= 964
 6(ConverseOfToLeftOf)4: L[4] - R[6] - 862 LEFT[6,4] <= 32
 6(ConverseOfAboveOf)4: T[4] - B[6] - 996 ABOVE[6,4] <= 32
 6(ToLeftOf)5: - L[5] + R[6] + 862 LEFT[6,5] <= 830
 6(Above)5: - T[5] + B[6] + 996 ABOVE[6,5] <= 964
 6(ConverseOfToLeftOf)5: L[5] - R[6] - 862 LEFT[6,5] <= 32
 6(ConverseOfAboveOf)5: T[5] - B[6] - 996 ABOVE[6,5] <= 32
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
 R323: - LAG[0] + zLAG[0,0] <= 0
 R324: - RAG[0] + zRAG[0,0] <= 0
 R325: - TAG[0] + zTAG[0,0] <= 0
 R326: - BAG[0] + zBAG[0,0] <= 0
 R327: - LAG[0] + zLAG[1,0] <= 0
 R328: - RAG[0] + zRAG[1,0] <= 0
 R329: - TAG[0] + zTAG[1,0] <= 0
 R330: - BAG[0] + zBAG[1,0] <= 0
 R331: - LAG[0] + zLAG[2,0] <= 0
 R332: - RAG[0] + zRAG[2,0] <= 0
 R333: - TAG[0] + zTAG[2,0] <= 0
 R334: - BAG[0] + zBAG[2,0] <= 0
 R335: - LAG[0] + zLAG[3,0] <= 0
 R336: - RAG[0] + zRAG[3,0] <= 0
 R337: - TAG[0] + zTAG[3,0] <= 0
 R338: - BAG[0] + zBAG[3,0] <= 0
 R339: - LAG[0] + zLAG[4,0] <= 0
 R340: - RAG[0] + zRAG[4,0] <= 0
 R341: - TAG[0] + zTAG[4,0] <= 0
 R342: - BAG[0] + zBAG[4,0] <= 0
 R343: - LAG[0] + zLAG[5,0] <= 0
 R344: - RAG[0] + zRAG[5,0] <= 0
 R345: - TAG[0] + zTAG[5,0] <= 0
 R346: - BAG[0] + zBAG[5,0] <= 0
 R347: - LAG[0] + zLAG[6,0] <= 0
 R348: - RAG[0] + zRAG[6,0] <= 0
 R349: - TAG[0] + zTAG[6,0] <= 0
 R350: - BAG[0] + zBAG[6,0] <= 0
 R351: - LAG[1] + zLAG[0,1] <= 0
 R352: - RAG[1] + zRAG[0,1] <= 0
 R353: - TAG[1] + zTAG[0,1] <= 0
 R354: - BAG[1] + zBAG[0,1] <= 0
 R355: - LAG[1] + zLAG[1,1] <= 0
 R356: - RAG[1] + zRAG[1,1] <= 0
 R357: - TAG[1] + zTAG[1,1] <= 0
 R358: - BAG[1] + zBAG[1,1] <= 0
 R359: - LAG[1] + zLAG[2,1] <= 0
 R360: - RAG[1] + zRAG[2,1] <= 0
 R361: - TAG[1] + zTAG[2,1] <= 0
 R362: - BAG[1] + zBAG[2,1] <= 0
 R363: - LAG[1] + zLAG[3,1] <= 0
 R364: - RAG[1] + zRAG[3,1] <= 0
 R365: - TAG[1] + zTAG[3,1] <= 0
 R366: - BAG[1] + zBAG[3,1] <= 0
 R367: - LAG[1] + zLAG[4,1] <= 0
 R368: - RAG[1] + zRAG[4,1] <= 0
 R369: - TAG[1] + zTAG[4,1] <= 0
 R370: - BAG[1] + zBAG[4,1] <= 0
 R371: - LAG[1] + zLAG[5,1] <= 0
 R372: - RAG[1] + zRAG[5,1] <= 0
 R373: - TAG[1] + zTAG[5,1] <= 0
 R374: - BAG[1] + zBAG[5,1] <= 0
 R375: - LAG[1] + zLAG[6,1] <= 0
 R376: - RAG[1] + zRAG[6,1] <= 0
 R377: - TAG[1] + zTAG[6,1] <= 0
 R378: - BAG[1] + zBAG[6,1] <= 0
 R379: - LAG[2] + zLAG[0,2] <= 0
 R380: - RAG[2] + zRAG[0,2] <= 0
 R381: - TAG[2] + zTAG[0,2] <= 0
 R382: - BAG[2] + zBAG[0,2] <= 0
 R383: - LAG[2] + zLAG[1,2] <= 0
 R384: - RAG[2] + zRAG[1,2] <= 0
 R385: - TAG[2] + zTAG[1,2] <= 0
 R386: - BAG[2] + zBAG[1,2] <= 0
 R387: - LAG[2] + zLAG[2,2] <= 0
 R388: - RAG[2] + zRAG[2,2] <= 0
 R389: - TAG[2] + zTAG[2,2] <= 0
 R390: - BAG[2] + zBAG[2,2] <= 0
 R391: - LAG[2] + zLAG[3,2] <= 0
 R392: - RAG[2] + zRAG[3,2] <= 0
 R393: - TAG[2] + zTAG[3,2] <= 0
 R394: - BAG[2] + zBAG[3,2] <= 0
 R395: - LAG[2] + zLAG[4,2] <= 0
 R396: - RAG[2] + zRAG[4,2] <= 0
 R397: - TAG[2] + zTAG[4,2] <= 0
 R398: - BAG[2] + zBAG[4,2] <= 0
 R399: - LAG[2] + zLAG[5,2] <= 0
 R400: - RAG[2] + zRAG[5,2] <= 0
 R401: - TAG[2] + zTAG[5,2] <= 0
 R402: - BAG[2] + zBAG[5,2] <= 0
 R403: - LAG[2] + zLAG[6,2] <= 0
 R404: - RAG[2] + zRAG[6,2] <= 0
 R405: - TAG[2] + zTAG[6,2] <= 0
 R406: - BAG[2] + zBAG[6,2] <= 0
 R407: - LAG[3] + zLAG[0,3] <= 0
 R408: - RAG[3] + zRAG[0,3] <= 0
 R409: - TAG[3] + zTAG[0,3] <= 0
 R410: - BAG[3] + zBAG[0,3] <= 0
 R411: - LAG[3] + zLAG[1,3] <= 0
 R412: - RAG[3] + zRAG[1,3] <= 0
 R413: - TAG[3] + zTAG[1,3] <= 0
 R414: - BAG[3] + zBAG[1,3] <= 0
 R415: - LAG[3] + zLAG[2,3] <= 0
 R416: - RAG[3] + zRAG[2,3] <= 0
 R417: - TAG[3] + zTAG[2,3] <= 0
 R418: - BAG[3] + zBAG[2,3] <= 0
 R419: - LAG[3] + zLAG[3,3] <= 0
 R420: - RAG[3] + zRAG[3,3] <= 0
 R421: - TAG[3] + zTAG[3,3] <= 0
 R422: - BAG[3] + zBAG[3,3] <= 0
 R423: - LAG[3] + zLAG[4,3] <= 0
 R424: - RAG[3] + zRAG[4,3] <= 0
 R425: - TAG[3] + zTAG[4,3] <= 0
 R426: - BAG[3] + zBAG[4,3] <= 0
 R427: - LAG[3] + zLAG[5,3] <= 0
 R428: - RAG[3] + zRAG[5,3] <= 0
 R429: - TAG[3] + zTAG[5,3] <= 0
 R430: - BAG[3] + zBAG[5,3] <= 0
 R431: - LAG[3] + zLAG[6,3] <= 0
 R432: - RAG[3] + zRAG[6,3] <= 0
 R433: - TAG[3] + zTAG[6,3] <= 0
 R434: - BAG[3] + zBAG[6,3] <= 0
 R435: - LAG[4] + zLAG[0,4] <= 0
 R436: - RAG[4] + zRAG[0,4] <= 0
 R437: - TAG[4] + zTAG[0,4] <= 0
 R438: - BAG[4] + zBAG[0,4] <= 0
 R439: - LAG[4] + zLAG[1,4] <= 0
 R440: - RAG[4] + zRAG[1,4] <= 0
 R441: - TAG[4] + zTAG[1,4] <= 0
 R442: - BAG[4] + zBAG[1,4] <= 0
 R443: - LAG[4] + zLAG[2,4] <= 0
 R444: - RAG[4] + zRAG[2,4] <= 0
 R445: - TAG[4] + zTAG[2,4] <= 0
 R446: - BAG[4] + zBAG[2,4] <= 0
 R447: - LAG[4] + zLAG[3,4] <= 0
 R448: - RAG[4] + zRAG[3,4] <= 0
 R449: - TAG[4] + zTAG[3,4] <= 0
 R450: - BAG[4] + zBAG[3,4] <= 0
 R451: - LAG[4] + zLAG[4,4] <= 0
 R452: - RAG[4] + zRAG[4,4] <= 0
 R453: - TAG[4] + zTAG[4,4] <= 0
 R454: - BAG[4] + zBAG[4,4] <= 0
 R455: - LAG[4] + zLAG[5,4] <= 0
 R456: - RAG[4] + zRAG[5,4] <= 0
 R457: - TAG[4] + zTAG[5,4] <= 0
 R458: - BAG[4] + zBAG[5,4] <= 0
 R459: - LAG[4] + zLAG[6,4] <= 0
 R460: - RAG[4] + zRAG[6,4] <= 0
 R461: - TAG[4] + zTAG[6,4] <= 0
 R462: - BAG[4] + zBAG[6,4] <= 0
 R463: - LAG[5] + zLAG[0,5] <= 0
 R464: - RAG[5] + zRAG[0,5] <= 0
 R465: - TAG[5] + zTAG[0,5] <= 0
 R466: - BAG[5] + zBAG[0,5] <= 0
 R467: - LAG[5] + zLAG[1,5] <= 0
 R468: - RAG[5] + zRAG[1,5] <= 0
 R469: - TAG[5] + zTAG[1,5] <= 0
 R470: - BAG[5] + zBAG[1,5] <= 0
 R471: - LAG[5] + zLAG[2,5] <= 0
 R472: - RAG[5] + zRAG[2,5] <= 0
 R473: - TAG[5] + zTAG[2,5] <= 0
 R474: - BAG[5] + zBAG[2,5] <= 0
 R475: - LAG[5] + zLAG[3,5] <= 0
 R476: - RAG[5] + zRAG[3,5] <= 0
 R477: - TAG[5] + zTAG[3,5] <= 0
 R478: - BAG[5] + zBAG[3,5] <= 0
 R479: - LAG[5] + zLAG[4,5] <= 0
 R480: - RAG[5] + zRAG[4,5] <= 0
 R481: - TAG[5] + zTAG[4,5] <= 0
 R482: - BAG[5] + zBAG[4,5] <= 0
 R483: - LAG[5] + zLAG[5,5] <= 0
 R484: - RAG[5] + zRAG[5,5] <= 0
 R485: - TAG[5] + zTAG[5,5] <= 0
 R486: - BAG[5] + zBAG[5,5] <= 0
 R487: - LAG[5] + zLAG[6,5] <= 0
 R488: - RAG[5] + zRAG[6,5] <= 0
 R489: - TAG[5] + zTAG[6,5] <= 0
 R490: - BAG[5] + zBAG[6,5] <= 0
 R491: - LAG[6] + zLAG[0,6] <= 0
 R492: - RAG[6] + zRAG[0,6] <= 0
 R493: - TAG[6] + zTAG[0,6] <= 0
 R494: - BAG[6] + zBAG[0,6] <= 0
 R495: - LAG[6] + zLAG[1,6] <= 0
 R496: - RAG[6] + zRAG[1,6] <= 0
 R497: - TAG[6] + zTAG[1,6] <= 0
 R498: - BAG[6] + zBAG[1,6] <= 0
 R499: - LAG[6] + zLAG[2,6] <= 0
 R500: - RAG[6] + zRAG[2,6] <= 0
 R501: - TAG[6] + zTAG[2,6] <= 0
 R502: - BAG[6] + zBAG[2,6] <= 0
 R503: - LAG[6] + zLAG[3,6] <= 0
 R504: - RAG[6] + zRAG[3,6] <= 0
 R505: - TAG[6] + zTAG[3,6] <= 0
 R506: - BAG[6] + zBAG[3,6] <= 0
 R507: - LAG[6] + zLAG[4,6] <= 0
 R508: - RAG[6] + zRAG[4,6] <= 0
 R509: - TAG[6] + zTAG[4,6] <= 0
 R510: - BAG[6] + zBAG[4,6] <= 0
 R511: - LAG[6] + zLAG[5,6] <= 0
 R512: - RAG[6] + zRAG[5,6] <= 0
 R513: - TAG[6] + zTAG[5,6] <= 0
 R514: - BAG[6] + zBAG[5,6] <= 0
 R515: - LAG[6] + zLAG[6,6] <= 0
 R516: - RAG[6] + zRAG[6,6] <= 0
 R517: - TAG[6] + zTAG[6,6] <= 0
 R518: - BAG[6] + zBAG[6,6] <= 0
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
 MinsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] + 862 zLAG[4,0] <= 862
 MinsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] + 862 zRAG[4,0] <= 862
 MinsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] + 996 zTAG[4,0] <= 996
 MinsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] + 996 zBAG[4,0] <= 996
 MaxsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] - 862 zLAG[4,0] >= -862
 MaxsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] - 862 zRAG[4,0] >= -862
 MaxsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] - 996 zTAG[4,0] >= -996
 MaxsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] - 996 zBAG[4,0] >= -996
 MinsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] + 862 zLAG[5,0] <= 862
 MinsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] + 862 zRAG[5,0] <= 862
 MinsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] + 996 zTAG[5,0] <= 996
 MinsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] + 996 zBAG[5,0] <= 996
 MaxsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] - 862 zLAG[5,0] >= -862
 MaxsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] - 862 zRAG[5,0] >= -862
 MaxsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] - 996 zTAG[5,0] >= -996
 MaxsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] - 996 zBAG[5,0] >= -996
 MinsideConnectL[6]ToLAG[0]: L[6] - vLAG[0] + 862 zLAG[6,0] <= 862
 MinsideConnectR[6]ToRAG[0]: R[6] - vRAG[0] + 862 zRAG[6,0] <= 862
 MinsideConnectT[6]ToTAG[0]: T[6] - vTAG[0] + 996 zTAG[6,0] <= 996
 MinsideConnectB[6]ToBAG[0]: B[6] - vBAG[0] + 996 zBAG[6,0] <= 996
 MaxsideConnectL[6]ToLAG[0]: L[6] - vLAG[0] - 862 zLAG[6,0] >= -862
 MaxsideConnectR[6]ToRAG[0]: R[6] - vRAG[0] - 862 zRAG[6,0] >= -862
 MaxsideConnectT[6]ToTAG[0]: T[6] - vTAG[0] - 996 zTAG[6,0] >= -996
 MaxsideConnectB[6]ToBAG[0]: B[6] - vBAG[0] - 996 zBAG[6,0] >= -996
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
 MinsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] + 862 zLAG[4,1] <= 862
 MinsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] + 862 zRAG[4,1] <= 862
 MinsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] + 996 zTAG[4,1] <= 996
 MinsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] + 996 zBAG[4,1] <= 996
 MaxsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] - 862 zLAG[4,1] >= -862
 MaxsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] - 862 zRAG[4,1] >= -862
 MaxsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] - 996 zTAG[4,1] >= -996
 MaxsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] - 996 zBAG[4,1] >= -996
 MinsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] + 862 zLAG[5,1] <= 862
 MinsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] + 862 zRAG[5,1] <= 862
 MinsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] + 996 zTAG[5,1] <= 996
 MinsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] + 996 zBAG[5,1] <= 996
 MaxsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] - 862 zLAG[5,1] >= -862
 MaxsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] - 862 zRAG[5,1] >= -862
 MaxsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] - 996 zTAG[5,1] >= -996
 MaxsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] - 996 zBAG[5,1] >= -996
 MinsideConnectL[6]ToLAG[1]: L[6] - vLAG[1] + 862 zLAG[6,1] <= 862
 MinsideConnectR[6]ToRAG[1]: R[6] - vRAG[1] + 862 zRAG[6,1] <= 862
 MinsideConnectT[6]ToTAG[1]: T[6] - vTAG[1] + 996 zTAG[6,1] <= 996
 MinsideConnectB[6]ToBAG[1]: B[6] - vBAG[1] + 996 zBAG[6,1] <= 996
 MaxsideConnectL[6]ToLAG[1]: L[6] - vLAG[1] - 862 zLAG[6,1] >= -862
 MaxsideConnectR[6]ToRAG[1]: R[6] - vRAG[1] - 862 zRAG[6,1] >= -862
 MaxsideConnectT[6]ToTAG[1]: T[6] - vTAG[1] - 996 zTAG[6,1] >= -996
 MaxsideConnectB[6]ToBAG[1]: B[6] - vBAG[1] - 996 zBAG[6,1] >= -996
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
 MinsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] + 862 zLAG[4,2] <= 862
 MinsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] + 862 zRAG[4,2] <= 862
 MinsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] + 996 zTAG[4,2] <= 996
 MinsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] + 996 zBAG[4,2] <= 996
 MaxsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] - 862 zLAG[4,2] >= -862
 MaxsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] - 862 zRAG[4,2] >= -862
 MaxsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] - 996 zTAG[4,2] >= -996
 MaxsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] - 996 zBAG[4,2] >= -996
 MinsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] + 862 zLAG[5,2] <= 862
 MinsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] + 862 zRAG[5,2] <= 862
 MinsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] + 996 zTAG[5,2] <= 996
 MinsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] + 996 zBAG[5,2] <= 996
 MaxsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] - 862 zLAG[5,2] >= -862
 MaxsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] - 862 zRAG[5,2] >= -862
 MaxsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] - 996 zTAG[5,2] >= -996
 MaxsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] - 996 zBAG[5,2] >= -996
 MinsideConnectL[6]ToLAG[2]: L[6] - vLAG[2] + 862 zLAG[6,2] <= 862
 MinsideConnectR[6]ToRAG[2]: R[6] - vRAG[2] + 862 zRAG[6,2] <= 862
 MinsideConnectT[6]ToTAG[2]: T[6] - vTAG[2] + 996 zTAG[6,2] <= 996
 MinsideConnectB[6]ToBAG[2]: B[6] - vBAG[2] + 996 zBAG[6,2] <= 996
 MaxsideConnectL[6]ToLAG[2]: L[6] - vLAG[2] - 862 zLAG[6,2] >= -862
 MaxsideConnectR[6]ToRAG[2]: R[6] - vRAG[2] - 862 zRAG[6,2] >= -862
 MaxsideConnectT[6]ToTAG[2]: T[6] - vTAG[2] - 996 zTAG[6,2] >= -996
 MaxsideConnectB[6]ToBAG[2]: B[6] - vBAG[2] - 996 zBAG[6,2] >= -996
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
 MinsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] + 862 zLAG[4,3] <= 862
 MinsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] + 862 zRAG[4,3] <= 862
 MinsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] + 996 zTAG[4,3] <= 996
 MinsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] + 996 zBAG[4,3] <= 996
 MaxsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] - 862 zLAG[4,3] >= -862
 MaxsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] - 862 zRAG[4,3] >= -862
 MaxsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] - 996 zTAG[4,3] >= -996
 MaxsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] - 996 zBAG[4,3] >= -996
 MinsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] + 862 zLAG[5,3] <= 862
 MinsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] + 862 zRAG[5,3] <= 862
 MinsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] + 996 zTAG[5,3] <= 996
 MinsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] + 996 zBAG[5,3] <= 996
 MaxsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] - 862 zLAG[5,3] >= -862
 MaxsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] - 862 zRAG[5,3] >= -862
 MaxsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] - 996 zTAG[5,3] >= -996
 MaxsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] - 996 zBAG[5,3] >= -996
 MinsideConnectL[6]ToLAG[3]: L[6] - vLAG[3] + 862 zLAG[6,3] <= 862
 MinsideConnectR[6]ToRAG[3]: R[6] - vRAG[3] + 862 zRAG[6,3] <= 862
 MinsideConnectT[6]ToTAG[3]: T[6] - vTAG[3] + 996 zTAG[6,3] <= 996
 MinsideConnectB[6]ToBAG[3]: B[6] - vBAG[3] + 996 zBAG[6,3] <= 996
 MaxsideConnectL[6]ToLAG[3]: L[6] - vLAG[3] - 862 zLAG[6,3] >= -862
 MaxsideConnectR[6]ToRAG[3]: R[6] - vRAG[3] - 862 zRAG[6,3] >= -862
 MaxsideConnectT[6]ToTAG[3]: T[6] - vTAG[3] - 996 zTAG[6,3] >= -996
 MaxsideConnectB[6]ToBAG[3]: B[6] - vBAG[3] - 996 zBAG[6,3] >= -996
 MinsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] + 862 zLAG[0,4] <= 862
 MinsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] + 862 zRAG[0,4] <= 862
 MinsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] + 996 zTAG[0,4] <= 996
 MinsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] + 996 zBAG[0,4] <= 996
 MaxsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] - 862 zLAG[0,4] >= -862
 MaxsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] - 862 zRAG[0,4] >= -862
 MaxsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] - 996 zTAG[0,4] >= -996
 MaxsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] - 996 zBAG[0,4] >= -996
 MinsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] + 862 zLAG[1,4] <= 862
 MinsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] + 862 zRAG[1,4] <= 862
 MinsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] + 996 zTAG[1,4] <= 996
 MinsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] + 996 zBAG[1,4] <= 996
 MaxsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] - 862 zLAG[1,4] >= -862
 MaxsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] - 862 zRAG[1,4] >= -862
 MaxsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] - 996 zTAG[1,4] >= -996
 MaxsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] - 996 zBAG[1,4] >= -996
 MinsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] + 862 zLAG[2,4] <= 862
 MinsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] + 862 zRAG[2,4] <= 862
 MinsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] + 996 zTAG[2,4] <= 996
 MinsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] + 996 zBAG[2,4] <= 996
 MaxsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] - 862 zLAG[2,4] >= -862
 MaxsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] - 862 zRAG[2,4] >= -862
 MaxsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] - 996 zTAG[2,4] >= -996
 MaxsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] - 996 zBAG[2,4] >= -996
 MinsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] + 862 zLAG[3,4] <= 862
 MinsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] + 862 zRAG[3,4] <= 862
 MinsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] + 996 zTAG[3,4] <= 996
 MinsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] + 996 zBAG[3,4] <= 996
 MaxsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] - 862 zLAG[3,4] >= -862
 MaxsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] - 862 zRAG[3,4] >= -862
 MaxsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] - 996 zTAG[3,4] >= -996
 MaxsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] - 996 zBAG[3,4] >= -996
 MinsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] + 862 zLAG[4,4] <= 862
 MinsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] + 862 zRAG[4,4] <= 862
 MinsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] + 996 zTAG[4,4] <= 996
 MinsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] + 996 zBAG[4,4] <= 996
 MaxsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] - 862 zLAG[4,4] >= -862
 MaxsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] - 862 zRAG[4,4] >= -862
 MaxsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] - 996 zTAG[4,4] >= -996
 MaxsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] - 996 zBAG[4,4] >= -996
 MinsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] + 862 zLAG[5,4] <= 862
 MinsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] + 862 zRAG[5,4] <= 862
 MinsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] + 996 zTAG[5,4] <= 996
 MinsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] + 996 zBAG[5,4] <= 996
 MaxsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] - 862 zLAG[5,4] >= -862
 MaxsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] - 862 zRAG[5,4] >= -862
 MaxsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] - 996 zTAG[5,4] >= -996
 MaxsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] - 996 zBAG[5,4] >= -996
 MinsideConnectL[6]ToLAG[4]: L[6] - vLAG[4] + 862 zLAG[6,4] <= 862
 MinsideConnectR[6]ToRAG[4]: R[6] - vRAG[4] + 862 zRAG[6,4] <= 862
 MinsideConnectT[6]ToTAG[4]: T[6] - vTAG[4] + 996 zTAG[6,4] <= 996
 MinsideConnectB[6]ToBAG[4]: B[6] - vBAG[4] + 996 zBAG[6,4] <= 996
 MaxsideConnectL[6]ToLAG[4]: L[6] - vLAG[4] - 862 zLAG[6,4] >= -862
 MaxsideConnectR[6]ToRAG[4]: R[6] - vRAG[4] - 862 zRAG[6,4] >= -862
 MaxsideConnectT[6]ToTAG[4]: T[6] - vTAG[4] - 996 zTAG[6,4] >= -996
 MaxsideConnectB[6]ToBAG[4]: B[6] - vBAG[4] - 996 zBAG[6,4] >= -996
 MinsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] + 862 zLAG[0,5] <= 862
 MinsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] + 862 zRAG[0,5] <= 862
 MinsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] + 996 zTAG[0,5] <= 996
 MinsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] + 996 zBAG[0,5] <= 996
 MaxsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] - 862 zLAG[0,5] >= -862
 MaxsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] - 862 zRAG[0,5] >= -862
 MaxsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] - 996 zTAG[0,5] >= -996
 MaxsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] - 996 zBAG[0,5] >= -996
 MinsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] + 862 zLAG[1,5] <= 862
 MinsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] + 862 zRAG[1,5] <= 862
 MinsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] + 996 zTAG[1,5] <= 996
 MinsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] + 996 zBAG[1,5] <= 996
 MaxsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] - 862 zLAG[1,5] >= -862
 MaxsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] - 862 zRAG[1,5] >= -862
 MaxsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] - 996 zTAG[1,5] >= -996
 MaxsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] - 996 zBAG[1,5] >= -996
 MinsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] + 862 zLAG[2,5] <= 862
 MinsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] + 862 zRAG[2,5] <= 862
 MinsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] + 996 zTAG[2,5] <= 996
 MinsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] + 996 zBAG[2,5] <= 996
 MaxsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] - 862 zLAG[2,5] >= -862
 MaxsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] - 862 zRAG[2,5] >= -862
 MaxsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] - 996 zTAG[2,5] >= -996
 MaxsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] - 996 zBAG[2,5] >= -996
 MinsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] + 862 zLAG[3,5] <= 862
 MinsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] + 862 zRAG[3,5] <= 862
 MinsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] + 996 zTAG[3,5] <= 996
 MinsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] + 996 zBAG[3,5] <= 996
 MaxsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] - 862 zLAG[3,5] >= -862
 MaxsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] - 862 zRAG[3,5] >= -862
 MaxsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] - 996 zTAG[3,5] >= -996
 MaxsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] - 996 zBAG[3,5] >= -996
 MinsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] + 862 zLAG[4,5] <= 862
 MinsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] + 862 zRAG[4,5] <= 862
 MinsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] + 996 zTAG[4,5] <= 996
 MinsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] + 996 zBAG[4,5] <= 996
 MaxsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] - 862 zLAG[4,5] >= -862
 MaxsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] - 862 zRAG[4,5] >= -862
 MaxsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] - 996 zTAG[4,5] >= -996
 MaxsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] - 996 zBAG[4,5] >= -996
 MinsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] + 862 zLAG[5,5] <= 862
 MinsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] + 862 zRAG[5,5] <= 862
 MinsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] + 996 zTAG[5,5] <= 996
 MinsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] + 996 zBAG[5,5] <= 996
 MaxsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] - 862 zLAG[5,5] >= -862
 MaxsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] - 862 zRAG[5,5] >= -862
 MaxsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] - 996 zTAG[5,5] >= -996
 MaxsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] - 996 zBAG[5,5] >= -996
 MinsideConnectL[6]ToLAG[5]: L[6] - vLAG[5] + 862 zLAG[6,5] <= 862
 MinsideConnectR[6]ToRAG[5]: R[6] - vRAG[5] + 862 zRAG[6,5] <= 862
 MinsideConnectT[6]ToTAG[5]: T[6] - vTAG[5] + 996 zTAG[6,5] <= 996
 MinsideConnectB[6]ToBAG[5]: B[6] - vBAG[5] + 996 zBAG[6,5] <= 996
 MaxsideConnectL[6]ToLAG[5]: L[6] - vLAG[5] - 862 zLAG[6,5] >= -862
 MaxsideConnectR[6]ToRAG[5]: R[6] - vRAG[5] - 862 zRAG[6,5] >= -862
 MaxsideConnectT[6]ToTAG[5]: T[6] - vTAG[5] - 996 zTAG[6,5] >= -996
 MaxsideConnectB[6]ToBAG[5]: B[6] - vBAG[5] - 996 zBAG[6,5] >= -996
 MinsideConnectL[0]ToLAG[6]: L[0] - vLAG[6] + 862 zLAG[0,6] <= 862
 MinsideConnectR[0]ToRAG[6]: R[0] - vRAG[6] + 862 zRAG[0,6] <= 862
 MinsideConnectT[0]ToTAG[6]: T[0] - vTAG[6] + 996 zTAG[0,6] <= 996
 MinsideConnectB[0]ToBAG[6]: B[0] - vBAG[6] + 996 zBAG[0,6] <= 996
 MaxsideConnectL[0]ToLAG[6]: L[0] - vLAG[6] - 862 zLAG[0,6] >= -862
 MaxsideConnectR[0]ToRAG[6]: R[0] - vRAG[6] - 862 zRAG[0,6] >= -862
 MaxsideConnectT[0]ToTAG[6]: T[0] - vTAG[6] - 996 zTAG[0,6] >= -996
 MaxsideConnectB[0]ToBAG[6]: B[0] - vBAG[6] - 996 zBAG[0,6] >= -996
 MinsideConnectL[1]ToLAG[6]: L[1] - vLAG[6] + 862 zLAG[1,6] <= 862
 MinsideConnectR[1]ToRAG[6]: R[1] - vRAG[6] + 862 zRAG[1,6] <= 862
 MinsideConnectT[1]ToTAG[6]: T[1] - vTAG[6] + 996 zTAG[1,6] <= 996
 MinsideConnectB[1]ToBAG[6]: B[1] - vBAG[6] + 996 zBAG[1,6] <= 996
 MaxsideConnectL[1]ToLAG[6]: L[1] - vLAG[6] - 862 zLAG[1,6] >= -862
 MaxsideConnectR[1]ToRAG[6]: R[1] - vRAG[6] - 862 zRAG[1,6] >= -862
 MaxsideConnectT[1]ToTAG[6]: T[1] - vTAG[6] - 996 zTAG[1,6] >= -996
 MaxsideConnectB[1]ToBAG[6]: B[1] - vBAG[6] - 996 zBAG[1,6] >= -996
 MinsideConnectL[2]ToLAG[6]: L[2] - vLAG[6] + 862 zLAG[2,6] <= 862
 MinsideConnectR[2]ToRAG[6]: R[2] - vRAG[6] + 862 zRAG[2,6] <= 862
 MinsideConnectT[2]ToTAG[6]: T[2] - vTAG[6] + 996 zTAG[2,6] <= 996
 MinsideConnectB[2]ToBAG[6]: B[2] - vBAG[6] + 996 zBAG[2,6] <= 996
 MaxsideConnectL[2]ToLAG[6]: L[2] - vLAG[6] - 862 zLAG[2,6] >= -862
 MaxsideConnectR[2]ToRAG[6]: R[2] - vRAG[6] - 862 zRAG[2,6] >= -862
 MaxsideConnectT[2]ToTAG[6]: T[2] - vTAG[6] - 996 zTAG[2,6] >= -996
 MaxsideConnectB[2]ToBAG[6]: B[2] - vBAG[6] - 996 zBAG[2,6] >= -996
 MinsideConnectL[3]ToLAG[6]: L[3] - vLAG[6] + 862 zLAG[3,6] <= 862
 MinsideConnectR[3]ToRAG[6]: R[3] - vRAG[6] + 862 zRAG[3,6] <= 862
 MinsideConnectT[3]ToTAG[6]: T[3] - vTAG[6] + 996 zTAG[3,6] <= 996
 MinsideConnectB[3]ToBAG[6]: B[3] - vBAG[6] + 996 zBAG[3,6] <= 996
 MaxsideConnectL[3]ToLAG[6]: L[3] - vLAG[6] - 862 zLAG[3,6] >= -862
 MaxsideConnectR[3]ToRAG[6]: R[3] - vRAG[6] - 862 zRAG[3,6] >= -862
 MaxsideConnectT[3]ToTAG[6]: T[3] - vTAG[6] - 996 zTAG[3,6] >= -996
 MaxsideConnectB[3]ToBAG[6]: B[3] - vBAG[6] - 996 zBAG[3,6] >= -996
 MinsideConnectL[4]ToLAG[6]: L[4] - vLAG[6] + 862 zLAG[4,6] <= 862
 MinsideConnectR[4]ToRAG[6]: R[4] - vRAG[6] + 862 zRAG[4,6] <= 862
 MinsideConnectT[4]ToTAG[6]: T[4] - vTAG[6] + 996 zTAG[4,6] <= 996
 MinsideConnectB[4]ToBAG[6]: B[4] - vBAG[6] + 996 zBAG[4,6] <= 996
 MaxsideConnectL[4]ToLAG[6]: L[4] - vLAG[6] - 862 zLAG[4,6] >= -862
 MaxsideConnectR[4]ToRAG[6]: R[4] - vRAG[6] - 862 zRAG[4,6] >= -862
 MaxsideConnectT[4]ToTAG[6]: T[4] - vTAG[6] - 996 zTAG[4,6] >= -996
 MaxsideConnectB[4]ToBAG[6]: B[4] - vBAG[6] - 996 zBAG[4,6] >= -996
 MinsideConnectL[5]ToLAG[6]: L[5] - vLAG[6] + 862 zLAG[5,6] <= 862
 MinsideConnectR[5]ToRAG[6]: R[5] - vRAG[6] + 862 zRAG[5,6] <= 862
 MinsideConnectT[5]ToTAG[6]: T[5] - vTAG[6] + 996 zTAG[5,6] <= 996
 MinsideConnectB[5]ToBAG[6]: B[5] - vBAG[6] + 996 zBAG[5,6] <= 996
 MaxsideConnectL[5]ToLAG[6]: L[5] - vLAG[6] - 862 zLAG[5,6] >= -862
 MaxsideConnectR[5]ToRAG[6]: R[5] - vRAG[6] - 862 zRAG[5,6] >= -862
 MaxsideConnectT[5]ToTAG[6]: T[5] - vTAG[6] - 996 zTAG[5,6] >= -996
 MaxsideConnectB[5]ToBAG[6]: B[5] - vBAG[6] - 996 zBAG[5,6] >= -996
 MinsideConnectL[6]ToLAG[6]: L[6] - vLAG[6] + 862 zLAG[6,6] <= 862
 MinsideConnectR[6]ToRAG[6]: R[6] - vRAG[6] + 862 zRAG[6,6] <= 862
 MinsideConnectT[6]ToTAG[6]: T[6] - vTAG[6] + 996 zTAG[6,6] <= 996
 MinsideConnectB[6]ToBAG[6]: B[6] - vBAG[6] + 996 zBAG[6,6] <= 996
 MaxsideConnectL[6]ToLAG[6]: L[6] - vLAG[6] - 862 zLAG[6,6] >= -862
 MaxsideConnectR[6]ToRAG[6]: R[6] - vRAG[6] - 862 zRAG[6,6] >= -862
 MaxsideConnectT[6]ToTAG[6]: T[6] - vTAG[6] - 996 zTAG[6,6] >= -996
 MaxsideConnectB[6]ToBAG[6]: B[6] - vBAG[6] - 996 zBAG[6,6] >= -996
 R911: - R[0] + maxX >= 0
 R912: - B[0] + maxY >= 0
 R913: - R[1] + maxX >= 0
 R914: - B[1] + maxY >= 0
 R915: - R[2] + maxX >= 0
 R916: - B[2] + maxY >= 0
 R917: - R[3] + maxX >= 0
 R918: - B[3] + maxY >= 0
 R919: - R[4] + maxX >= 0
 R920: - B[4] + maxY >= 0
 R921: - R[5] + maxX >= 0
 R922: - B[5] + maxY >= 0
 R923: - R[6] + maxX >= 0
 R924: - B[6] + maxY >= 0
 R925: LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + LAG[5] + LAG[6]
   + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + RAG[6] + TAG[0]
   + TAG[1] + TAG[2] + TAG[3] + TAG[4] + TAG[5] + TAG[6] + BAG[0] + BAG[1]
   + BAG[2] + BAG[3] + BAG[4] + BAG[5] + BAG[6] >= 12
Bounds
 L[0] <= 107.625
 L[1] <= 107.625
 L[2] <= 107.625
 L[3] <= 107.625
 L[4] <= 107.625
 L[5] <= 107.625
 L[6] <= 107.625
 1 <= R[0] <= 107.75
 1 <= R[1] <= 107.75
 1 <= R[2] <= 107.75
 1 <= R[3] <= 107.75
 1 <= R[4] <= 107.75
 1 <= R[5] <= 107.75
 1 <= R[6] <= 107.75
 T[0] <= 124.375
 T[1] <= 124.375
 T[2] <= 124.375
 T[3] <= 124.375
 T[4] <= 124.375
 T[5] <= 124.375
 T[6] <= 124.375
 1 <= B[0] <= 124.5
 1 <= B[1] <= 124.5
 1 <= B[2] <= 124.5
 1 <= B[3] <= 124.5
 1 <= B[4] <= 124.5
 1 <= B[5] <= 124.5
 1 <= B[6] <= 124.5
 1 <= H[0] <= 125
 1 <= H[1] <= 125
 1 <= H[2] <= 125
 1 <= H[3] <= 125
 1 <= H[4] <= 125
 1 <= H[5] <= 125
 1 <= H[6] <= 125
 1 <= W[0] <= 108
 1 <= W[1] <= 108
 1 <= W[2] <= 108
 1 <= W[3] <= 108
 1 <= W[4] <= 108
 1 <= W[5] <= 108
 1 <= W[6] <= 108
 resizeW[0] >= -1000
 resizeW[1] >= -1000
 resizeW[2] >= -1000
 resizeW[3] >= -1000
 resizeW[4] >= -1000
 resizeW[5] >= -1000
 resizeW[6] >= -1000
 resizeH[0] >= -1000
 resizeH[1] >= -1000
 resizeH[2] >= -1000
 resizeH[3] >= -1000
 resizeH[4] >= -1000
 resizeH[5] >= -1000
 resizeH[6] >= -1000
 moveX[0] >= -1000
 moveX[1] >= -1000
 moveX[2] >= -1000
 moveX[3] >= -1000
 moveX[4] >= -1000
 moveX[5] >= -1000
 moveX[6] >= -1000
 moveY[0] >= -1000
 moveY[1] >= -1000
 moveY[2] >= -1000
 moveY[3] >= -1000
 moveY[4] >= -1000
 moveY[5] >= -1000
 moveY[6] >= -1000
 vLAG[0] <= 106.75
 vLAG[1] <= 106.75
 vLAG[2] <= 106.75
 vLAG[3] <= 106.75
 vLAG[4] <= 106.75
 vLAG[5] <= 106.75
 vLAG[6] <= 106.75
 1 <= vRAG[0] <= 107.75
 1 <= vRAG[1] <= 107.75
 1 <= vRAG[2] <= 107.75
 1 <= vRAG[3] <= 107.75
 1 <= vRAG[4] <= 107.75
 1 <= vRAG[5] <= 107.75
 1 <= vRAG[6] <= 107.75
 vTAG[0] <= 123.5
 vTAG[1] <= 123.5
 vTAG[2] <= 123.5
 vTAG[3] <= 123.5
 vTAG[4] <= 123.5
 vTAG[5] <= 123.5
 vTAG[6] <= 123.5
 1 <= vBAG[0] <= 124.5
 1 <= vBAG[1] <= 124.5
 1 <= vBAG[2] <= 124.5
 1 <= vBAG[3] <= 124.5
 1 <= vBAG[4] <= 124.5
 1 <= vBAG[5] <= 124.5
 1 <= vBAG[6] <= 124.5
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
 H[2] H[3] H[4] H[5] H[6] W[0] W[1] W[2] W[3] W[4] W[5] W[6] resizeW[0]
 resizeW[1] resizeW[2] resizeW[3] resizeW[4] resizeW[5] resizeW[6]
 resizeWAbs[0] resizeWAbs[1] resizeWAbs[2] resizeWAbs[3] resizeWAbs[4]
 resizeWAbs[5] resizeWAbs[6] resizeH[0] resizeH[1] resizeH[2] resizeH[3]
 resizeH[4] resizeH[5] resizeH[6] resizeHAbs[0] resizeHAbs[1] resizeHAbs[2]
 resizeHAbs[3] resizeHAbs[4] resizeHAbs[5] resizeHAbs[6] moveX[0] moveX[1]
 moveX[2] moveX[3] moveX[4] moveX[5] moveX[6] moveXAbs[0] moveXAbs[1]
 moveXAbs[2] moveXAbs[3] moveXAbs[4] moveXAbs[5] moveXAbs[6] moveY[0]
 moveY[1] moveY[2] moveY[3] moveY[4] moveY[5] moveY[6] moveYAbs[0]
 moveYAbs[1] moveYAbs[2] moveYAbs[3] moveYAbs[4] moveYAbs[5] moveYAbs[6]
 vLAG[0] vLAG[1] vLAG[2] vLAG[3] vLAG[4] vLAG[5] vLAG[6] vRAG[0] vRAG[1]
 vRAG[2] vRAG[3] vRAG[4] vRAG[5] vRAG[6] vTAG[0] vTAG[1] vTAG[2] vTAG[3]
 vTAG[4] vTAG[5] vTAG[6] vBAG[0] vBAG[1] vBAG[2] vBAG[3] vBAG[4] vBAG[5]
 vBAG[6] maxX maxY
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
 resizeWAbs: resizeWAbs[4] = ABS ( resizeW[4] )
 resizeHAbs: resizeHAbs[4] = ABS ( resizeH[4] )
 moveXAbs: moveXAbs[4] = ABS ( moveX[4] )
 moveYAbs: moveYAbs[4] = ABS ( moveY[4] )
 resizeWAbs: resizeWAbs[5] = ABS ( resizeW[5] )
 resizeHAbs: resizeHAbs[5] = ABS ( resizeH[5] )
 moveXAbs: moveXAbs[5] = ABS ( moveX[5] )
 moveYAbs: moveYAbs[5] = ABS ( moveY[5] )
 resizeWAbs: resizeWAbs[6] = ABS ( resizeW[6] )
 resizeHAbs: resizeHAbs[6] = ABS ( resizeH[6] )
 moveXAbs: moveXAbs[6] = ABS ( moveX[6] )
 moveYAbs: moveYAbs[6] = ABS ( moveY[6] )
End
