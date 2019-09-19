\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 resizeWAbsVar[0] + 0.005 resizeWAbsVar[1] + 0.0025 resizeWAbsVar[2]
   + 0.0025 resizeWAbsVar[3] + 0.0025 resizeWAbsVar[4]
   + 0.0025 resizeWAbsVar[5] + 0.0025 resizeWAbsVar[6]
   + 0.001 resizeWAbsVar[7] + 0.01 resizeHAbsVar[0]
   + 0.00142857142857143 resizeHAbsVar[1]
   + 0.00333333333333333 resizeHAbsVar[2]
   + 0.00333333333333333 resizeHAbsVar[3] + 0.0025 resizeHAbsVar[4]
   + 0.005 resizeHAbsVar[5] + 0.005 resizeHAbsVar[6]
   + 0.00666666666666667 resizeHAbsVar[7] + 0.001 moveXAbs[0]
   + 0.005 moveXAbs[1] + 0.0025 moveXAbs[2] + 0.0025 moveXAbs[3]
   + 0.0025 moveXAbs[4] + 0.0025 moveXAbs[5] + 0.0025 moveXAbs[6]
   + 0.001 moveXAbs[7] + 0.01 moveYAbs[0] + 0.00142857142857143 moveYAbs[1]
   + 0.00333333333333333 moveYAbs[2] + 0.00333333333333333 moveYAbs[3]
   + 0.0025 moveYAbs[4] + 0.005 moveYAbs[5] + 0.005 moveYAbs[6]
   + 0.00666666666666667 moveYAbs[7] + 0 ABOVE[0,0] + 0 ABOVE[1,1]
   + 0 ABOVE[2,2] + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 ABOVE[5,5]
   + 0 ABOVE[6,6] + 0 ABOVE[7,7] + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2]
   + 0 LEFT[3,3] + 0 LEFT[4,4] + 0 LEFT[5,5] + 0 LEFT[6,6] + 0 LEFT[7,7]
   + LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + LAG[5] + LAG[6] + LAG[7]
   + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + RAG[6] + RAG[7]
   + TAG[0] + TAG[1] + TAG[2] + TAG[3] + TAG[4] + TAG[5] + TAG[6] + TAG[7]
   + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4] + BAG[5] + BAG[6] + BAG[7]
Subject To
 resizeW: - W[0] + resizeWVar[0] = -125
 resizeH: - H[0] + resizeHVar[0] = -13
 moveX: - L[0] + moveX[0] = 0
 moveY: - T[0] + moveY[0] = 0
 resizeW: - W[1] + resizeWVar[1] = -25
 resizeH: - H[1] + resizeHVar[1] = -88
 moveX: - L[1] + moveX[1] = -100
 moveY: - T[1] + moveY[1] = -12
 resizeW: - W[2] + resizeWVar[2] = -50
 resizeH: - H[2] + resizeHVar[2] = -38
 moveX: - L[2] + moveX[2] = 0
 moveY: - T[2] + moveY[2] = -62
 resizeW: - W[3] + resizeWVar[3] = -50
 resizeH: - H[3] + resizeHVar[3] = -38
 moveX: - L[3] + moveX[3] = -50
 moveY: - T[3] + moveY[3] = -62
 resizeW: - W[4] + resizeWVar[4] = -50
 resizeH: - H[4] + resizeHVar[4] = -50
 moveX: - L[4] + moveX[4] = -50
 moveY: - T[4] + moveY[4] = -12
 resizeW: - W[5] + resizeWVar[5] = -50
 resizeH: - H[5] + resizeHVar[5] = -25
 moveX: - L[5] + moveX[5] = 0
 moveY: - T[5] + moveY[5] = -38
 resizeW: - W[6] + resizeWVar[6] = -50
 resizeH: - H[6] + resizeHVar[6] = -25
 moveX: - L[6] + moveX[6] = 0
 moveY: - T[6] + moveY[6] = -12
 resizeW: - W[7] + resizeWVar[7] = -125
 resizeH: - H[7] + resizeHVar[7] = -19
 moveX: - L[7] + moveX[7] = 0
 moveY: - T[7] + moveY[7] = -100
 DisableNon-PermittedBasedOnPrespecified: ABOVE[1,0] + ABOVE[2,0]
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
 0(ToLeftOf)1: - L[1] + R[0] + 1000 LEFT[0,1] <= 968
 0(Above)1: - T[1] + B[0] + 950 ABOVE[0,1] <= 918
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 1000 LEFT[0,1] <= 32
 0(ConverseOfAboveOf)1: T[1] - B[0] - 950 ABOVE[0,1] <= 32
 0(ToLeftOf)2: - L[2] + R[0] + 1000 LEFT[0,2] <= 968
 0(Above)2: - T[2] + B[0] + 950 ABOVE[0,2] <= 918
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 1000 LEFT[0,2] <= 32
 0(ConverseOfAboveOf)2: T[2] - B[0] - 950 ABOVE[0,2] <= 32
 0(ToLeftOf)3: - L[3] + R[0] + 1000 LEFT[0,3] <= 968
 0(Above)3: - T[3] + B[0] + 950 ABOVE[0,3] <= 918
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 1000 LEFT[0,3] <= 32
 0(ConverseOfAboveOf)3: T[3] - B[0] - 950 ABOVE[0,3] <= 32
 0(ToLeftOf)4: - L[4] + R[0] + 1000 LEFT[0,4] <= 968
 0(Above)4: - T[4] + B[0] + 950 ABOVE[0,4] <= 918
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 1000 LEFT[0,4] <= 32
 0(ConverseOfAboveOf)4: T[4] - B[0] - 950 ABOVE[0,4] <= 32
 0(ToLeftOf)5: - L[5] + R[0] + 1000 LEFT[0,5] <= 968
 0(Above)5: - T[5] + B[0] + 950 ABOVE[0,5] <= 918
 0(ConverseOfToLeftOf)5: L[5] - R[0] - 1000 LEFT[0,5] <= 32
 0(ConverseOfAboveOf)5: T[5] - B[0] - 950 ABOVE[0,5] <= 32
 0(ToLeftOf)6: - L[6] + R[0] + 1000 LEFT[0,6] <= 968
 0(Above)6: - T[6] + B[0] + 950 ABOVE[0,6] <= 918
 0(ConverseOfToLeftOf)6: L[6] - R[0] - 1000 LEFT[0,6] <= 32
 0(ConverseOfAboveOf)6: T[6] - B[0] - 950 ABOVE[0,6] <= 32
 0(ToLeftOf)7: - L[7] + R[0] + 1000 LEFT[0,7] <= 968
 0(Above)7: - T[7] + B[0] + 950 ABOVE[0,7] <= 918
 0(ConverseOfToLeftOf)7: L[7] - R[0] - 1000 LEFT[0,7] <= 32
 0(ConverseOfAboveOf)7: T[7] - B[0] - 950 ABOVE[0,7] <= 32
 1(ToLeftOf)0: - L[0] + R[1] + 1000 LEFT[1,0] <= 968
 1(Above)0: - T[0] + B[1] + 950 ABOVE[1,0] <= 918
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 1000 LEFT[1,0] <= 32
 1(ConverseOfAboveOf)0: T[0] - B[1] - 950 ABOVE[1,0] <= 32
 1(ToLeftOf)2: - L[2] + R[1] + 1000 LEFT[1,2] <= 968
 1(Above)2: - T[2] + B[1] + 950 ABOVE[1,2] <= 918
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 1000 LEFT[1,2] <= 32
 1(ConverseOfAboveOf)2: T[2] - B[1] - 950 ABOVE[1,2] <= 32
 1(ToLeftOf)3: - L[3] + R[1] + 1000 LEFT[1,3] <= 968
 1(Above)3: - T[3] + B[1] + 950 ABOVE[1,3] <= 918
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 1000 LEFT[1,3] <= 32
 1(ConverseOfAboveOf)3: T[3] - B[1] - 950 ABOVE[1,3] <= 32
 1(ToLeftOf)4: - L[4] + R[1] + 1000 LEFT[1,4] <= 968
 1(Above)4: - T[4] + B[1] + 950 ABOVE[1,4] <= 918
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 1000 LEFT[1,4] <= 32
 1(ConverseOfAboveOf)4: T[4] - B[1] - 950 ABOVE[1,4] <= 32
 1(ToLeftOf)5: - L[5] + R[1] + 1000 LEFT[1,5] <= 968
 1(Above)5: - T[5] + B[1] + 950 ABOVE[1,5] <= 918
 1(ConverseOfToLeftOf)5: L[5] - R[1] - 1000 LEFT[1,5] <= 32
 1(ConverseOfAboveOf)5: T[5] - B[1] - 950 ABOVE[1,5] <= 32
 1(ToLeftOf)6: - L[6] + R[1] + 1000 LEFT[1,6] <= 968
 1(Above)6: - T[6] + B[1] + 950 ABOVE[1,6] <= 918
 1(ConverseOfToLeftOf)6: L[6] - R[1] - 1000 LEFT[1,6] <= 32
 1(ConverseOfAboveOf)6: T[6] - B[1] - 950 ABOVE[1,6] <= 32
 1(ToLeftOf)7: - L[7] + R[1] + 1000 LEFT[1,7] <= 968
 1(Above)7: - T[7] + B[1] + 950 ABOVE[1,7] <= 918
 1(ConverseOfToLeftOf)7: L[7] - R[1] - 1000 LEFT[1,7] <= 32
 1(ConverseOfAboveOf)7: T[7] - B[1] - 950 ABOVE[1,7] <= 32
 2(ToLeftOf)0: - L[0] + R[2] + 1000 LEFT[2,0] <= 968
 2(Above)0: - T[0] + B[2] + 950 ABOVE[2,0] <= 918
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 1000 LEFT[2,0] <= 32
 2(ConverseOfAboveOf)0: T[0] - B[2] - 950 ABOVE[2,0] <= 32
 2(ToLeftOf)1: - L[1] + R[2] + 1000 LEFT[2,1] <= 968
 2(Above)1: - T[1] + B[2] + 950 ABOVE[2,1] <= 918
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 1000 LEFT[2,1] <= 32
 2(ConverseOfAboveOf)1: T[1] - B[2] - 950 ABOVE[2,1] <= 32
 2(ToLeftOf)3: - L[3] + R[2] + 1000 LEFT[2,3] <= 968
 2(Above)3: - T[3] + B[2] + 950 ABOVE[2,3] <= 918
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 1000 LEFT[2,3] <= 32
 2(ConverseOfAboveOf)3: T[3] - B[2] - 950 ABOVE[2,3] <= 32
 2(ToLeftOf)4: - L[4] + R[2] + 1000 LEFT[2,4] <= 968
 2(Above)4: - T[4] + B[2] + 950 ABOVE[2,4] <= 918
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 1000 LEFT[2,4] <= 32
 2(ConverseOfAboveOf)4: T[4] - B[2] - 950 ABOVE[2,4] <= 32
 2(ToLeftOf)5: - L[5] + R[2] + 1000 LEFT[2,5] <= 968
 2(Above)5: - T[5] + B[2] + 950 ABOVE[2,5] <= 918
 2(ConverseOfToLeftOf)5: L[5] - R[2] - 1000 LEFT[2,5] <= 32
 2(ConverseOfAboveOf)5: T[5] - B[2] - 950 ABOVE[2,5] <= 32
 2(ToLeftOf)6: - L[6] + R[2] + 1000 LEFT[2,6] <= 968
 2(Above)6: - T[6] + B[2] + 950 ABOVE[2,6] <= 918
 2(ConverseOfToLeftOf)6: L[6] - R[2] - 1000 LEFT[2,6] <= 32
 2(ConverseOfAboveOf)6: T[6] - B[2] - 950 ABOVE[2,6] <= 32
 2(ToLeftOf)7: - L[7] + R[2] + 1000 LEFT[2,7] <= 968
 2(Above)7: - T[7] + B[2] + 950 ABOVE[2,7] <= 918
 2(ConverseOfToLeftOf)7: L[7] - R[2] - 1000 LEFT[2,7] <= 32
 2(ConverseOfAboveOf)7: T[7] - B[2] - 950 ABOVE[2,7] <= 32
 3(ToLeftOf)0: - L[0] + R[3] + 1000 LEFT[3,0] <= 968
 3(Above)0: - T[0] + B[3] + 950 ABOVE[3,0] <= 918
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 1000 LEFT[3,0] <= 32
 3(ConverseOfAboveOf)0: T[0] - B[3] - 950 ABOVE[3,0] <= 32
 3(ToLeftOf)1: - L[1] + R[3] + 1000 LEFT[3,1] <= 968
 3(Above)1: - T[1] + B[3] + 950 ABOVE[3,1] <= 918
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 1000 LEFT[3,1] <= 32
 3(ConverseOfAboveOf)1: T[1] - B[3] - 950 ABOVE[3,1] <= 32
 3(ToLeftOf)2: - L[2] + R[3] + 1000 LEFT[3,2] <= 968
 3(Above)2: - T[2] + B[3] + 950 ABOVE[3,2] <= 918
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 1000 LEFT[3,2] <= 32
 3(ConverseOfAboveOf)2: T[2] - B[3] - 950 ABOVE[3,2] <= 32
 3(ToLeftOf)4: - L[4] + R[3] + 1000 LEFT[3,4] <= 968
 3(Above)4: - T[4] + B[3] + 950 ABOVE[3,4] <= 918
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 1000 LEFT[3,4] <= 32
 3(ConverseOfAboveOf)4: T[4] - B[3] - 950 ABOVE[3,4] <= 32
 3(ToLeftOf)5: - L[5] + R[3] + 1000 LEFT[3,5] <= 968
 3(Above)5: - T[5] + B[3] + 950 ABOVE[3,5] <= 918
 3(ConverseOfToLeftOf)5: L[5] - R[3] - 1000 LEFT[3,5] <= 32
 3(ConverseOfAboveOf)5: T[5] - B[3] - 950 ABOVE[3,5] <= 32
 3(ToLeftOf)6: - L[6] + R[3] + 1000 LEFT[3,6] <= 968
 3(Above)6: - T[6] + B[3] + 950 ABOVE[3,6] <= 918
 3(ConverseOfToLeftOf)6: L[6] - R[3] - 1000 LEFT[3,6] <= 32
 3(ConverseOfAboveOf)6: T[6] - B[3] - 950 ABOVE[3,6] <= 32
 3(ToLeftOf)7: - L[7] + R[3] + 1000 LEFT[3,7] <= 968
 3(Above)7: - T[7] + B[3] + 950 ABOVE[3,7] <= 918
 3(ConverseOfToLeftOf)7: L[7] - R[3] - 1000 LEFT[3,7] <= 32
 3(ConverseOfAboveOf)7: T[7] - B[3] - 950 ABOVE[3,7] <= 32
 4(ToLeftOf)0: - L[0] + R[4] + 1000 LEFT[4,0] <= 968
 4(Above)0: - T[0] + B[4] + 950 ABOVE[4,0] <= 918
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 1000 LEFT[4,0] <= 32
 4(ConverseOfAboveOf)0: T[0] - B[4] - 950 ABOVE[4,0] <= 32
 4(ToLeftOf)1: - L[1] + R[4] + 1000 LEFT[4,1] <= 968
 4(Above)1: - T[1] + B[4] + 950 ABOVE[4,1] <= 918
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 1000 LEFT[4,1] <= 32
 4(ConverseOfAboveOf)1: T[1] - B[4] - 950 ABOVE[4,1] <= 32
 4(ToLeftOf)2: - L[2] + R[4] + 1000 LEFT[4,2] <= 968
 4(Above)2: - T[2] + B[4] + 950 ABOVE[4,2] <= 918
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 1000 LEFT[4,2] <= 32
 4(ConverseOfAboveOf)2: T[2] - B[4] - 950 ABOVE[4,2] <= 32
 4(ToLeftOf)3: - L[3] + R[4] + 1000 LEFT[4,3] <= 968
 4(Above)3: - T[3] + B[4] + 950 ABOVE[4,3] <= 918
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 1000 LEFT[4,3] <= 32
 4(ConverseOfAboveOf)3: T[3] - B[4] - 950 ABOVE[4,3] <= 32
 4(ToLeftOf)5: - L[5] + R[4] + 1000 LEFT[4,5] <= 968
 4(Above)5: - T[5] + B[4] + 950 ABOVE[4,5] <= 918
 4(ConverseOfToLeftOf)5: L[5] - R[4] - 1000 LEFT[4,5] <= 32
 4(ConverseOfAboveOf)5: T[5] - B[4] - 950 ABOVE[4,5] <= 32
 4(ToLeftOf)6: - L[6] + R[4] + 1000 LEFT[4,6] <= 968
 4(Above)6: - T[6] + B[4] + 950 ABOVE[4,6] <= 918
 4(ConverseOfToLeftOf)6: L[6] - R[4] - 1000 LEFT[4,6] <= 32
 4(ConverseOfAboveOf)6: T[6] - B[4] - 950 ABOVE[4,6] <= 32
 4(ToLeftOf)7: - L[7] + R[4] + 1000 LEFT[4,7] <= 968
 4(Above)7: - T[7] + B[4] + 950 ABOVE[4,7] <= 918
 4(ConverseOfToLeftOf)7: L[7] - R[4] - 1000 LEFT[4,7] <= 32
 4(ConverseOfAboveOf)7: T[7] - B[4] - 950 ABOVE[4,7] <= 32
 5(ToLeftOf)0: - L[0] + R[5] + 1000 LEFT[5,0] <= 968
 5(Above)0: - T[0] + B[5] + 950 ABOVE[5,0] <= 918
 5(ConverseOfToLeftOf)0: L[0] - R[5] - 1000 LEFT[5,0] <= 32
 5(ConverseOfAboveOf)0: T[0] - B[5] - 950 ABOVE[5,0] <= 32
 5(ToLeftOf)1: - L[1] + R[5] + 1000 LEFT[5,1] <= 968
 5(Above)1: - T[1] + B[5] + 950 ABOVE[5,1] <= 918
 5(ConverseOfToLeftOf)1: L[1] - R[5] - 1000 LEFT[5,1] <= 32
 5(ConverseOfAboveOf)1: T[1] - B[5] - 950 ABOVE[5,1] <= 32
 5(ToLeftOf)2: - L[2] + R[5] + 1000 LEFT[5,2] <= 968
 5(Above)2: - T[2] + B[5] + 950 ABOVE[5,2] <= 918
 5(ConverseOfToLeftOf)2: L[2] - R[5] - 1000 LEFT[5,2] <= 32
 5(ConverseOfAboveOf)2: T[2] - B[5] - 950 ABOVE[5,2] <= 32
 5(ToLeftOf)3: - L[3] + R[5] + 1000 LEFT[5,3] <= 968
 5(Above)3: - T[3] + B[5] + 950 ABOVE[5,3] <= 918
 5(ConverseOfToLeftOf)3: L[3] - R[5] - 1000 LEFT[5,3] <= 32
 5(ConverseOfAboveOf)3: T[3] - B[5] - 950 ABOVE[5,3] <= 32
 5(ToLeftOf)4: - L[4] + R[5] + 1000 LEFT[5,4] <= 968
 5(Above)4: - T[4] + B[5] + 950 ABOVE[5,4] <= 918
 5(ConverseOfToLeftOf)4: L[4] - R[5] - 1000 LEFT[5,4] <= 32
 5(ConverseOfAboveOf)4: T[4] - B[5] - 950 ABOVE[5,4] <= 32
 5(ToLeftOf)6: - L[6] + R[5] + 1000 LEFT[5,6] <= 968
 5(Above)6: - T[6] + B[5] + 950 ABOVE[5,6] <= 918
 5(ConverseOfToLeftOf)6: L[6] - R[5] - 1000 LEFT[5,6] <= 32
 5(ConverseOfAboveOf)6: T[6] - B[5] - 950 ABOVE[5,6] <= 32
 5(ToLeftOf)7: - L[7] + R[5] + 1000 LEFT[5,7] <= 968
 5(Above)7: - T[7] + B[5] + 950 ABOVE[5,7] <= 918
 5(ConverseOfToLeftOf)7: L[7] - R[5] - 1000 LEFT[5,7] <= 32
 5(ConverseOfAboveOf)7: T[7] - B[5] - 950 ABOVE[5,7] <= 32
 6(ToLeftOf)0: - L[0] + R[6] + 1000 LEFT[6,0] <= 968
 6(Above)0: - T[0] + B[6] + 950 ABOVE[6,0] <= 918
 6(ConverseOfToLeftOf)0: L[0] - R[6] - 1000 LEFT[6,0] <= 32
 6(ConverseOfAboveOf)0: T[0] - B[6] - 950 ABOVE[6,0] <= 32
 6(ToLeftOf)1: - L[1] + R[6] + 1000 LEFT[6,1] <= 968
 6(Above)1: - T[1] + B[6] + 950 ABOVE[6,1] <= 918
 6(ConverseOfToLeftOf)1: L[1] - R[6] - 1000 LEFT[6,1] <= 32
 6(ConverseOfAboveOf)1: T[1] - B[6] - 950 ABOVE[6,1] <= 32
 6(ToLeftOf)2: - L[2] + R[6] + 1000 LEFT[6,2] <= 968
 6(Above)2: - T[2] + B[6] + 950 ABOVE[6,2] <= 918
 6(ConverseOfToLeftOf)2: L[2] - R[6] - 1000 LEFT[6,2] <= 32
 6(ConverseOfAboveOf)2: T[2] - B[6] - 950 ABOVE[6,2] <= 32
 6(ToLeftOf)3: - L[3] + R[6] + 1000 LEFT[6,3] <= 968
 6(Above)3: - T[3] + B[6] + 950 ABOVE[6,3] <= 918
 6(ConverseOfToLeftOf)3: L[3] - R[6] - 1000 LEFT[6,3] <= 32
 6(ConverseOfAboveOf)3: T[3] - B[6] - 950 ABOVE[6,3] <= 32
 6(ToLeftOf)4: - L[4] + R[6] + 1000 LEFT[6,4] <= 968
 6(Above)4: - T[4] + B[6] + 950 ABOVE[6,4] <= 918
 6(ConverseOfToLeftOf)4: L[4] - R[6] - 1000 LEFT[6,4] <= 32
 6(ConverseOfAboveOf)4: T[4] - B[6] - 950 ABOVE[6,4] <= 32
 6(ToLeftOf)5: - L[5] + R[6] + 1000 LEFT[6,5] <= 968
 6(Above)5: - T[5] + B[6] + 950 ABOVE[6,5] <= 918
 6(ConverseOfToLeftOf)5: L[5] - R[6] - 1000 LEFT[6,5] <= 32
 6(ConverseOfAboveOf)5: T[5] - B[6] - 950 ABOVE[6,5] <= 32
 6(ToLeftOf)7: - L[7] + R[6] + 1000 LEFT[6,7] <= 968
 6(Above)7: - T[7] + B[6] + 950 ABOVE[6,7] <= 918
 6(ConverseOfToLeftOf)7: L[7] - R[6] - 1000 LEFT[6,7] <= 32
 6(ConverseOfAboveOf)7: T[7] - B[6] - 950 ABOVE[6,7] <= 32
 7(ToLeftOf)0: - L[0] + R[7] + 1000 LEFT[7,0] <= 968
 7(Above)0: - T[0] + B[7] + 950 ABOVE[7,0] <= 918
 7(ConverseOfToLeftOf)0: L[0] - R[7] - 1000 LEFT[7,0] <= 32
 7(ConverseOfAboveOf)0: T[0] - B[7] - 950 ABOVE[7,0] <= 32
 7(ToLeftOf)1: - L[1] + R[7] + 1000 LEFT[7,1] <= 968
 7(Above)1: - T[1] + B[7] + 950 ABOVE[7,1] <= 918
 7(ConverseOfToLeftOf)1: L[1] - R[7] - 1000 LEFT[7,1] <= 32
 7(ConverseOfAboveOf)1: T[1] - B[7] - 950 ABOVE[7,1] <= 32
 7(ToLeftOf)2: - L[2] + R[7] + 1000 LEFT[7,2] <= 968
 7(Above)2: - T[2] + B[7] + 950 ABOVE[7,2] <= 918
 7(ConverseOfToLeftOf)2: L[2] - R[7] - 1000 LEFT[7,2] <= 32
 7(ConverseOfAboveOf)2: T[2] - B[7] - 950 ABOVE[7,2] <= 32
 7(ToLeftOf)3: - L[3] + R[7] + 1000 LEFT[7,3] <= 968
 7(Above)3: - T[3] + B[7] + 950 ABOVE[7,3] <= 918
 7(ConverseOfToLeftOf)3: L[3] - R[7] - 1000 LEFT[7,3] <= 32
 7(ConverseOfAboveOf)3: T[3] - B[7] - 950 ABOVE[7,3] <= 32
 7(ToLeftOf)4: - L[4] + R[7] + 1000 LEFT[7,4] <= 968
 7(Above)4: - T[4] + B[7] + 950 ABOVE[7,4] <= 918
 7(ConverseOfToLeftOf)4: L[4] - R[7] - 1000 LEFT[7,4] <= 32
 7(ConverseOfAboveOf)4: T[4] - B[7] - 950 ABOVE[7,4] <= 32
 7(ToLeftOf)5: - L[5] + R[7] + 1000 LEFT[7,5] <= 968
 7(Above)5: - T[5] + B[7] + 950 ABOVE[7,5] <= 918
 7(ConverseOfToLeftOf)5: L[5] - R[7] - 1000 LEFT[7,5] <= 32
 7(ConverseOfAboveOf)5: T[5] - B[7] - 950 ABOVE[7,5] <= 32
 7(ToLeftOf)6: - L[6] + R[7] + 1000 LEFT[7,6] <= 968
 7(Above)6: - T[6] + B[7] + 950 ABOVE[7,6] <= 918
 7(ConverseOfToLeftOf)6: L[6] - R[7] - 1000 LEFT[7,6] <= 32
 7(ConverseOfAboveOf)6: T[6] - B[7] - 950 ABOVE[7,6] <= 32
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
 R417: - LAG[0] + zLAG[0,0] <= 0
 R418: - RAG[0] + zRAG[0,0] <= 0
 R419: - TAG[0] + zTAG[0,0] <= 0
 R420: - BAG[0] + zBAG[0,0] <= 0
 R421: - LAG[0] + zLAG[1,0] <= 0
 R422: - RAG[0] + zRAG[1,0] <= 0
 R423: - TAG[0] + zTAG[1,0] <= 0
 R424: - BAG[0] + zBAG[1,0] <= 0
 R425: - LAG[0] + zLAG[2,0] <= 0
 R426: - RAG[0] + zRAG[2,0] <= 0
 R427: - TAG[0] + zTAG[2,0] <= 0
 R428: - BAG[0] + zBAG[2,0] <= 0
 R429: - LAG[0] + zLAG[3,0] <= 0
 R430: - RAG[0] + zRAG[3,0] <= 0
 R431: - TAG[0] + zTAG[3,0] <= 0
 R432: - BAG[0] + zBAG[3,0] <= 0
 R433: - LAG[0] + zLAG[4,0] <= 0
 R434: - RAG[0] + zRAG[4,0] <= 0
 R435: - TAG[0] + zTAG[4,0] <= 0
 R436: - BAG[0] + zBAG[4,0] <= 0
 R437: - LAG[0] + zLAG[5,0] <= 0
 R438: - RAG[0] + zRAG[5,0] <= 0
 R439: - TAG[0] + zTAG[5,0] <= 0
 R440: - BAG[0] + zBAG[5,0] <= 0
 R441: - LAG[0] + zLAG[6,0] <= 0
 R442: - RAG[0] + zRAG[6,0] <= 0
 R443: - TAG[0] + zTAG[6,0] <= 0
 R444: - BAG[0] + zBAG[6,0] <= 0
 R445: - LAG[0] + zLAG[7,0] <= 0
 R446: - RAG[0] + zRAG[7,0] <= 0
 R447: - TAG[0] + zTAG[7,0] <= 0
 R448: - BAG[0] + zBAG[7,0] <= 0
 R449: - LAG[1] + zLAG[0,1] <= 0
 R450: - RAG[1] + zRAG[0,1] <= 0
 R451: - TAG[1] + zTAG[0,1] <= 0
 R452: - BAG[1] + zBAG[0,1] <= 0
 R453: - LAG[1] + zLAG[1,1] <= 0
 R454: - RAG[1] + zRAG[1,1] <= 0
 R455: - TAG[1] + zTAG[1,1] <= 0
 R456: - BAG[1] + zBAG[1,1] <= 0
 R457: - LAG[1] + zLAG[2,1] <= 0
 R458: - RAG[1] + zRAG[2,1] <= 0
 R459: - TAG[1] + zTAG[2,1] <= 0
 R460: - BAG[1] + zBAG[2,1] <= 0
 R461: - LAG[1] + zLAG[3,1] <= 0
 R462: - RAG[1] + zRAG[3,1] <= 0
 R463: - TAG[1] + zTAG[3,1] <= 0
 R464: - BAG[1] + zBAG[3,1] <= 0
 R465: - LAG[1] + zLAG[4,1] <= 0
 R466: - RAG[1] + zRAG[4,1] <= 0
 R467: - TAG[1] + zTAG[4,1] <= 0
 R468: - BAG[1] + zBAG[4,1] <= 0
 R469: - LAG[1] + zLAG[5,1] <= 0
 R470: - RAG[1] + zRAG[5,1] <= 0
 R471: - TAG[1] + zTAG[5,1] <= 0
 R472: - BAG[1] + zBAG[5,1] <= 0
 R473: - LAG[1] + zLAG[6,1] <= 0
 R474: - RAG[1] + zRAG[6,1] <= 0
 R475: - TAG[1] + zTAG[6,1] <= 0
 R476: - BAG[1] + zBAG[6,1] <= 0
 R477: - LAG[1] + zLAG[7,1] <= 0
 R478: - RAG[1] + zRAG[7,1] <= 0
 R479: - TAG[1] + zTAG[7,1] <= 0
 R480: - BAG[1] + zBAG[7,1] <= 0
 R481: - LAG[2] + zLAG[0,2] <= 0
 R482: - RAG[2] + zRAG[0,2] <= 0
 R483: - TAG[2] + zTAG[0,2] <= 0
 R484: - BAG[2] + zBAG[0,2] <= 0
 R485: - LAG[2] + zLAG[1,2] <= 0
 R486: - RAG[2] + zRAG[1,2] <= 0
 R487: - TAG[2] + zTAG[1,2] <= 0
 R488: - BAG[2] + zBAG[1,2] <= 0
 R489: - LAG[2] + zLAG[2,2] <= 0
 R490: - RAG[2] + zRAG[2,2] <= 0
 R491: - TAG[2] + zTAG[2,2] <= 0
 R492: - BAG[2] + zBAG[2,2] <= 0
 R493: - LAG[2] + zLAG[3,2] <= 0
 R494: - RAG[2] + zRAG[3,2] <= 0
 R495: - TAG[2] + zTAG[3,2] <= 0
 R496: - BAG[2] + zBAG[3,2] <= 0
 R497: - LAG[2] + zLAG[4,2] <= 0
 R498: - RAG[2] + zRAG[4,2] <= 0
 R499: - TAG[2] + zTAG[4,2] <= 0
 R500: - BAG[2] + zBAG[4,2] <= 0
 R501: - LAG[2] + zLAG[5,2] <= 0
 R502: - RAG[2] + zRAG[5,2] <= 0
 R503: - TAG[2] + zTAG[5,2] <= 0
 R504: - BAG[2] + zBAG[5,2] <= 0
 R505: - LAG[2] + zLAG[6,2] <= 0
 R506: - RAG[2] + zRAG[6,2] <= 0
 R507: - TAG[2] + zTAG[6,2] <= 0
 R508: - BAG[2] + zBAG[6,2] <= 0
 R509: - LAG[2] + zLAG[7,2] <= 0
 R510: - RAG[2] + zRAG[7,2] <= 0
 R511: - TAG[2] + zTAG[7,2] <= 0
 R512: - BAG[2] + zBAG[7,2] <= 0
 R513: - LAG[3] + zLAG[0,3] <= 0
 R514: - RAG[3] + zRAG[0,3] <= 0
 R515: - TAG[3] + zTAG[0,3] <= 0
 R516: - BAG[3] + zBAG[0,3] <= 0
 R517: - LAG[3] + zLAG[1,3] <= 0
 R518: - RAG[3] + zRAG[1,3] <= 0
 R519: - TAG[3] + zTAG[1,3] <= 0
 R520: - BAG[3] + zBAG[1,3] <= 0
 R521: - LAG[3] + zLAG[2,3] <= 0
 R522: - RAG[3] + zRAG[2,3] <= 0
 R523: - TAG[3] + zTAG[2,3] <= 0
 R524: - BAG[3] + zBAG[2,3] <= 0
 R525: - LAG[3] + zLAG[3,3] <= 0
 R526: - RAG[3] + zRAG[3,3] <= 0
 R527: - TAG[3] + zTAG[3,3] <= 0
 R528: - BAG[3] + zBAG[3,3] <= 0
 R529: - LAG[3] + zLAG[4,3] <= 0
 R530: - RAG[3] + zRAG[4,3] <= 0
 R531: - TAG[3] + zTAG[4,3] <= 0
 R532: - BAG[3] + zBAG[4,3] <= 0
 R533: - LAG[3] + zLAG[5,3] <= 0
 R534: - RAG[3] + zRAG[5,3] <= 0
 R535: - TAG[3] + zTAG[5,3] <= 0
 R536: - BAG[3] + zBAG[5,3] <= 0
 R537: - LAG[3] + zLAG[6,3] <= 0
 R538: - RAG[3] + zRAG[6,3] <= 0
 R539: - TAG[3] + zTAG[6,3] <= 0
 R540: - BAG[3] + zBAG[6,3] <= 0
 R541: - LAG[3] + zLAG[7,3] <= 0
 R542: - RAG[3] + zRAG[7,3] <= 0
 R543: - TAG[3] + zTAG[7,3] <= 0
 R544: - BAG[3] + zBAG[7,3] <= 0
 R545: - LAG[4] + zLAG[0,4] <= 0
 R546: - RAG[4] + zRAG[0,4] <= 0
 R547: - TAG[4] + zTAG[0,4] <= 0
 R548: - BAG[4] + zBAG[0,4] <= 0
 R549: - LAG[4] + zLAG[1,4] <= 0
 R550: - RAG[4] + zRAG[1,4] <= 0
 R551: - TAG[4] + zTAG[1,4] <= 0
 R552: - BAG[4] + zBAG[1,4] <= 0
 R553: - LAG[4] + zLAG[2,4] <= 0
 R554: - RAG[4] + zRAG[2,4] <= 0
 R555: - TAG[4] + zTAG[2,4] <= 0
 R556: - BAG[4] + zBAG[2,4] <= 0
 R557: - LAG[4] + zLAG[3,4] <= 0
 R558: - RAG[4] + zRAG[3,4] <= 0
 R559: - TAG[4] + zTAG[3,4] <= 0
 R560: - BAG[4] + zBAG[3,4] <= 0
 R561: - LAG[4] + zLAG[4,4] <= 0
 R562: - RAG[4] + zRAG[4,4] <= 0
 R563: - TAG[4] + zTAG[4,4] <= 0
 R564: - BAG[4] + zBAG[4,4] <= 0
 R565: - LAG[4] + zLAG[5,4] <= 0
 R566: - RAG[4] + zRAG[5,4] <= 0
 R567: - TAG[4] + zTAG[5,4] <= 0
 R568: - BAG[4] + zBAG[5,4] <= 0
 R569: - LAG[4] + zLAG[6,4] <= 0
 R570: - RAG[4] + zRAG[6,4] <= 0
 R571: - TAG[4] + zTAG[6,4] <= 0
 R572: - BAG[4] + zBAG[6,4] <= 0
 R573: - LAG[4] + zLAG[7,4] <= 0
 R574: - RAG[4] + zRAG[7,4] <= 0
 R575: - TAG[4] + zTAG[7,4] <= 0
 R576: - BAG[4] + zBAG[7,4] <= 0
 R577: - LAG[5] + zLAG[0,5] <= 0
 R578: - RAG[5] + zRAG[0,5] <= 0
 R579: - TAG[5] + zTAG[0,5] <= 0
 R580: - BAG[5] + zBAG[0,5] <= 0
 R581: - LAG[5] + zLAG[1,5] <= 0
 R582: - RAG[5] + zRAG[1,5] <= 0
 R583: - TAG[5] + zTAG[1,5] <= 0
 R584: - BAG[5] + zBAG[1,5] <= 0
 R585: - LAG[5] + zLAG[2,5] <= 0
 R586: - RAG[5] + zRAG[2,5] <= 0
 R587: - TAG[5] + zTAG[2,5] <= 0
 R588: - BAG[5] + zBAG[2,5] <= 0
 R589: - LAG[5] + zLAG[3,5] <= 0
 R590: - RAG[5] + zRAG[3,5] <= 0
 R591: - TAG[5] + zTAG[3,5] <= 0
 R592: - BAG[5] + zBAG[3,5] <= 0
 R593: - LAG[5] + zLAG[4,5] <= 0
 R594: - RAG[5] + zRAG[4,5] <= 0
 R595: - TAG[5] + zTAG[4,5] <= 0
 R596: - BAG[5] + zBAG[4,5] <= 0
 R597: - LAG[5] + zLAG[5,5] <= 0
 R598: - RAG[5] + zRAG[5,5] <= 0
 R599: - TAG[5] + zTAG[5,5] <= 0
 R600: - BAG[5] + zBAG[5,5] <= 0
 R601: - LAG[5] + zLAG[6,5] <= 0
 R602: - RAG[5] + zRAG[6,5] <= 0
 R603: - TAG[5] + zTAG[6,5] <= 0
 R604: - BAG[5] + zBAG[6,5] <= 0
 R605: - LAG[5] + zLAG[7,5] <= 0
 R606: - RAG[5] + zRAG[7,5] <= 0
 R607: - TAG[5] + zTAG[7,5] <= 0
 R608: - BAG[5] + zBAG[7,5] <= 0
 R609: - LAG[6] + zLAG[0,6] <= 0
 R610: - RAG[6] + zRAG[0,6] <= 0
 R611: - TAG[6] + zTAG[0,6] <= 0
 R612: - BAG[6] + zBAG[0,6] <= 0
 R613: - LAG[6] + zLAG[1,6] <= 0
 R614: - RAG[6] + zRAG[1,6] <= 0
 R615: - TAG[6] + zTAG[1,6] <= 0
 R616: - BAG[6] + zBAG[1,6] <= 0
 R617: - LAG[6] + zLAG[2,6] <= 0
 R618: - RAG[6] + zRAG[2,6] <= 0
 R619: - TAG[6] + zTAG[2,6] <= 0
 R620: - BAG[6] + zBAG[2,6] <= 0
 R621: - LAG[6] + zLAG[3,6] <= 0
 R622: - RAG[6] + zRAG[3,6] <= 0
 R623: - TAG[6] + zTAG[3,6] <= 0
 R624: - BAG[6] + zBAG[3,6] <= 0
 R625: - LAG[6] + zLAG[4,6] <= 0
 R626: - RAG[6] + zRAG[4,6] <= 0
 R627: - TAG[6] + zTAG[4,6] <= 0
 R628: - BAG[6] + zBAG[4,6] <= 0
 R629: - LAG[6] + zLAG[5,6] <= 0
 R630: - RAG[6] + zRAG[5,6] <= 0
 R631: - TAG[6] + zTAG[5,6] <= 0
 R632: - BAG[6] + zBAG[5,6] <= 0
 R633: - LAG[6] + zLAG[6,6] <= 0
 R634: - RAG[6] + zRAG[6,6] <= 0
 R635: - TAG[6] + zTAG[6,6] <= 0
 R636: - BAG[6] + zBAG[6,6] <= 0
 R637: - LAG[6] + zLAG[7,6] <= 0
 R638: - RAG[6] + zRAG[7,6] <= 0
 R639: - TAG[6] + zTAG[7,6] <= 0
 R640: - BAG[6] + zBAG[7,6] <= 0
 R641: - LAG[7] + zLAG[0,7] <= 0
 R642: - RAG[7] + zRAG[0,7] <= 0
 R643: - TAG[7] + zTAG[0,7] <= 0
 R644: - BAG[7] + zBAG[0,7] <= 0
 R645: - LAG[7] + zLAG[1,7] <= 0
 R646: - RAG[7] + zRAG[1,7] <= 0
 R647: - TAG[7] + zTAG[1,7] <= 0
 R648: - BAG[7] + zBAG[1,7] <= 0
 R649: - LAG[7] + zLAG[2,7] <= 0
 R650: - RAG[7] + zRAG[2,7] <= 0
 R651: - TAG[7] + zTAG[2,7] <= 0
 R652: - BAG[7] + zBAG[2,7] <= 0
 R653: - LAG[7] + zLAG[3,7] <= 0
 R654: - RAG[7] + zRAG[3,7] <= 0
 R655: - TAG[7] + zTAG[3,7] <= 0
 R656: - BAG[7] + zBAG[3,7] <= 0
 R657: - LAG[7] + zLAG[4,7] <= 0
 R658: - RAG[7] + zRAG[4,7] <= 0
 R659: - TAG[7] + zTAG[4,7] <= 0
 R660: - BAG[7] + zBAG[4,7] <= 0
 R661: - LAG[7] + zLAG[5,7] <= 0
 R662: - RAG[7] + zRAG[5,7] <= 0
 R663: - TAG[7] + zTAG[5,7] <= 0
 R664: - BAG[7] + zBAG[5,7] <= 0
 R665: - LAG[7] + zLAG[6,7] <= 0
 R666: - RAG[7] + zRAG[6,7] <= 0
 R667: - TAG[7] + zTAG[6,7] <= 0
 R668: - BAG[7] + zBAG[6,7] <= 0
 R669: - LAG[7] + zLAG[7,7] <= 0
 R670: - RAG[7] + zRAG[7,7] <= 0
 R671: - TAG[7] + zTAG[7,7] <= 0
 R672: - BAG[7] + zBAG[7,7] <= 0
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
 R1185: - R[0] + maxX >= 0
 R1186: - B[0] + maxY >= 0
 R1187: - R[1] + maxX >= 0
 R1188: - B[1] + maxY >= 0
 R1189: - R[2] + maxX >= 0
 R1190: - B[2] + maxY >= 0
 R1191: - R[3] + maxX >= 0
 R1192: - B[3] + maxY >= 0
 R1193: - R[4] + maxX >= 0
 R1194: - B[4] + maxY >= 0
 R1195: - R[5] + maxX >= 0
 R1196: - B[5] + maxY >= 0
 R1197: - R[6] + maxX >= 0
 R1198: - B[6] + maxY >= 0
 R1199: - R[7] + maxX >= 0
 R1200: - B[7] + maxY >= 0
 R1201: LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + LAG[5] + LAG[6]
   + LAG[7] + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + RAG[6]
   + RAG[7] + TAG[0] + TAG[1] + TAG[2] + TAG[3] + TAG[4] + TAG[5] + TAG[6]
   + TAG[7] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4] + BAG[5] + BAG[6]
   + BAG[7] >= 12
Bounds
 L[0] <= 124.875
 L[1] <= 124.875
 L[2] <= 124.875
 L[3] <= 124.875
 L[4] <= 124.875
 L[5] <= 124.875
 L[6] <= 124.875
 L[7] <= 124.875
 1 <= R[0] <= 125
 1 <= R[1] <= 125
 1 <= R[2] <= 125
 1 <= R[3] <= 125
 1 <= R[4] <= 125
 1 <= R[5] <= 125
 1 <= R[6] <= 125
 1 <= R[7] <= 125
 T[0] <= 118.625
 T[1] <= 118.625
 T[2] <= 118.625
 T[3] <= 118.625
 T[4] <= 118.625
 T[5] <= 118.625
 T[6] <= 118.625
 T[7] <= 118.625
 1 <= B[0] <= 118.75
 1 <= B[1] <= 118.75
 1 <= B[2] <= 118.75
 1 <= B[3] <= 118.75
 1 <= B[4] <= 118.75
 1 <= B[5] <= 118.75
 1 <= B[6] <= 118.75
 1 <= B[7] <= 118.75
 1 <= H[0] <= 119
 1 <= H[1] <= 119
 1 <= H[2] <= 119
 1 <= H[3] <= 119
 1 <= H[4] <= 119
 1 <= H[5] <= 119
 1 <= H[6] <= 119
 1 <= H[7] <= 119
 1 <= W[0] <= 125
 1 <= W[1] <= 125
 1 <= W[2] <= 125
 1 <= W[3] <= 125
 1 <= W[4] <= 125
 1 <= W[5] <= 125
 1 <= W[6] <= 125
 1 <= W[7] <= 125
 resizeWVar[0] >= -1000
 resizeWVar[1] >= -1000
 resizeWVar[2] >= -1000
 resizeWVar[3] >= -1000
 resizeWVar[4] >= -1000
 resizeWVar[5] >= -1000
 resizeWVar[6] >= -1000
 resizeWVar[7] >= -1000
 resizeHVar[0] >= -1000
 resizeHVar[1] >= -1000
 resizeHVar[2] >= -1000
 resizeHVar[3] >= -1000
 resizeHVar[4] >= -1000
 resizeHVar[5] >= -1000
 resizeHVar[6] >= -1000
 resizeHVar[7] >= -1000
 moveX[0] >= -1000
 moveX[1] >= -1000
 moveX[2] >= -1000
 moveX[3] >= -1000
 moveX[4] >= -1000
 moveX[5] >= -1000
 moveX[6] >= -1000
 moveX[7] >= -1000
 moveY[0] >= -1000
 moveY[1] >= -1000
 moveY[2] >= -1000
 moveY[3] >= -1000
 moveY[4] >= -1000
 moveY[5] >= -1000
 moveY[6] >= -1000
 moveY[7] >= -1000
 vLAG[0] <= 124
 vLAG[1] <= 124
 vLAG[2] <= 124
 vLAG[3] <= 124
 vLAG[4] <= 124
 vLAG[5] <= 124
 vLAG[6] <= 124
 vLAG[7] <= 124
 1 <= vRAG[0] <= 125
 1 <= vRAG[1] <= 125
 1 <= vRAG[2] <= 125
 1 <= vRAG[3] <= 125
 1 <= vRAG[4] <= 125
 1 <= vRAG[5] <= 125
 1 <= vRAG[6] <= 125
 1 <= vRAG[7] <= 125
 vTAG[0] <= 117.75
 vTAG[1] <= 117.75
 vTAG[2] <= 117.75
 vTAG[3] <= 117.75
 vTAG[4] <= 117.75
 vTAG[5] <= 117.75
 vTAG[6] <= 117.75
 vTAG[7] <= 117.75
 1 <= vBAG[0] <= 118.75
 1 <= vBAG[1] <= 118.75
 1 <= vBAG[2] <= 118.75
 1 <= vBAG[3] <= 118.75
 1 <= vBAG[4] <= 118.75
 1 <= vBAG[5] <= 118.75
 1 <= vBAG[6] <= 118.75
 1 <= vBAG[7] <= 118.75
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
 W[5] W[6] W[7] resizeWVar[0] resizeWVar[1] resizeWVar[2] resizeWVar[3]
 resizeWVar[4] resizeWVar[5] resizeWVar[6] resizeWVar[7] resizeWAbsVar[0]
 resizeWAbsVar[1] resizeWAbsVar[2] resizeWAbsVar[3] resizeWAbsVar[4]
 resizeWAbsVar[5] resizeWAbsVar[6] resizeWAbsVar[7] resizeHVar[0]
 resizeHVar[1] resizeHVar[2] resizeHVar[3] resizeHVar[4] resizeHVar[5]
 resizeHVar[6] resizeHVar[7] resizeHAbsVar[0] resizeHAbsVar[1]
 resizeHAbsVar[2] resizeHAbsVar[3] resizeHAbsVar[4] resizeHAbsVar[5]
 resizeHAbsVar[6] resizeHAbsVar[7] moveX[0] moveX[1] moveX[2] moveX[3]
 moveX[4] moveX[5] moveX[6] moveX[7] moveXAbs[0] moveXAbs[1] moveXAbs[2]
 moveXAbs[3] moveXAbs[4] moveXAbs[5] moveXAbs[6] moveXAbs[7] moveY[0]
 moveY[1] moveY[2] moveY[3] moveY[4] moveY[5] moveY[6] moveY[7] moveYAbs[0]
 moveYAbs[1] moveYAbs[2] moveYAbs[3] moveYAbs[4] moveYAbs[5] moveYAbs[6]
 moveYAbs[7] vLAG[0] vLAG[1] vLAG[2] vLAG[3] vLAG[4] vLAG[5] vLAG[6]
 vLAG[7] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vRAG[5] vRAG[6] vRAG[7]
 vTAG[0] vTAG[1] vTAG[2] vTAG[3] vTAG[4] vTAG[5] vTAG[6] vTAG[7] vBAG[0]
 vBAG[1] vBAG[2] vBAG[3] vBAG[4] vBAG[5] vBAG[6] vBAG[7] maxX maxY
General Constraints
 resizeWAbs: resizeWAbsVar[0] = ABS ( resizeWVar[0] )
 resizeHAbs: resizeHAbsVar[0] = ABS ( resizeHVar[0] )
 moveXAbs: moveXAbs[0] = ABS ( moveX[0] )
 moveYAbs: moveYAbs[0] = ABS ( moveY[0] )
 resizeWAbs: resizeWAbsVar[1] = ABS ( resizeWVar[1] )
 resizeHAbs: resizeHAbsVar[1] = ABS ( resizeHVar[1] )
 moveXAbs: moveXAbs[1] = ABS ( moveX[1] )
 moveYAbs: moveYAbs[1] = ABS ( moveY[1] )
 resizeWAbs: resizeWAbsVar[2] = ABS ( resizeWVar[2] )
 resizeHAbs: resizeHAbsVar[2] = ABS ( resizeHVar[2] )
 moveXAbs: moveXAbs[2] = ABS ( moveX[2] )
 moveYAbs: moveYAbs[2] = ABS ( moveY[2] )
 resizeWAbs: resizeWAbsVar[3] = ABS ( resizeWVar[3] )
 resizeHAbs: resizeHAbsVar[3] = ABS ( resizeHVar[3] )
 moveXAbs: moveXAbs[3] = ABS ( moveX[3] )
 moveYAbs: moveYAbs[3] = ABS ( moveY[3] )
 resizeWAbs: resizeWAbsVar[4] = ABS ( resizeWVar[4] )
 resizeHAbs: resizeHAbsVar[4] = ABS ( resizeHVar[4] )
 moveXAbs: moveXAbs[4] = ABS ( moveX[4] )
 moveYAbs: moveYAbs[4] = ABS ( moveY[4] )
 resizeWAbs: resizeWAbsVar[5] = ABS ( resizeWVar[5] )
 resizeHAbs: resizeHAbsVar[5] = ABS ( resizeHVar[5] )
 moveXAbs: moveXAbs[5] = ABS ( moveX[5] )
 moveYAbs: moveYAbs[5] = ABS ( moveY[5] )
 resizeWAbs: resizeWAbsVar[6] = ABS ( resizeWVar[6] )
 resizeHAbs: resizeHAbsVar[6] = ABS ( resizeHVar[6] )
 moveXAbs: moveXAbs[6] = ABS ( moveX[6] )
 moveYAbs: moveYAbs[6] = ABS ( moveY[6] )
 resizeWAbs: resizeWAbsVar[7] = ABS ( resizeWVar[7] )
 resizeHAbs: resizeHAbsVar[7] = ABS ( resizeHVar[7] )
 moveXAbs: moveXAbs[7] = ABS ( moveX[7] )
 moveYAbs: moveYAbs[7] = ABS ( moveY[7] )
End
