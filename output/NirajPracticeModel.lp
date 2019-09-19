\ Model GLayout
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.00490196078431373 resizeWAbsVar[0]
   + 0.00212314225053079 resizeWAbsVar[1]
   + 0.00125944584382872 resizeWAbsVar[2]
   + 0.00255754475703325 resizeWAbsVar[3]
   + 0.00153609831029186 resizeWAbsVar[4] + 0.000926784 resizeWAbsVar[5]
   + 0.00740740740740741 resizeHAbsVar[0]
   + 0.0075187969924812 resizeHAbsVar[1]
   + 0.00495049504950495 resizeHAbsVar[2]
   + 0.00184842883548983 resizeHAbsVar[3]
   + 0.00309597523219814 resizeHAbsVar[4]
   + 0.0053475935828877 resizeHAbsVar[5] + 0.00490196078431373 moveXAbs[0]
   + 0.00212314225053079 moveXAbs[1] + 0.00125944584382872 moveXAbs[2]
   + 0.00255754475703325 moveXAbs[3] + 0.00153609831029186 moveXAbs[4]
   + 0.000926784 moveXAbs[5] + 0.00740740740740741 moveYAbs[0]
   + 0.0075187969924812 moveYAbs[1] + 0.00495049504950495 moveYAbs[2]
   + 0.00184842883548983 moveYAbs[3] + 0.00309597523219814 moveYAbs[4]
   + 0.0053475935828877 moveYAbs[5] + 0 ABOVE[0,0] + 0 ABOVE[1,1]
   + 0 ABOVE[2,2] + 0 ABOVE[3,3] + 0 ABOVE[4,4] + 0 ABOVE[5,5]
   + 0 LEFT[0,0] + 0 LEFT[1,1] + 0 LEFT[2,2] + 0 LEFT[3,3] + 0 LEFT[4,4]
   + 0 LEFT[5,5] + LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + LAG[5]
   + RAG[0] + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + TAG[0] + TAG[1]
   + TAG[2] + TAG[3] + TAG[4] + TAG[5] + BAG[0] + BAG[1] + BAG[2] + BAG[3]
   + BAG[4] + BAG[5]
Subject To
 resizeW: - W[0] + resizeWVar[0] = -26
 resizeH: - H[0] + resizeHVar[0] = -17
 moveX: - L[0] + moveX[0] = -20
 moveY: - T[0] + moveY[0] = -12
 resizeW: - W[1] + resizeWVar[1] = -59
 resizeH: - H[1] + resizeHVar[1] = -17
 moveX: - L[1] + moveX[1] = -64
 moveY: - T[1] + moveY[1] = -8
 resizeW: - W[2] + resizeWVar[2] = -100
 resizeH: - H[2] + resizeHVar[2] = -26
 moveX: - L[2] + moveX[2] = -25
 moveY: - T[2] + moveY[2] = -40
 resizeW: - W[3] + resizeWVar[3] = -49
 resizeH: - H[3] + resizeHVar[3] = -68
 moveX: - L[3] + moveX[3] = 0
 moveY: - T[3] + moveY[3] = 0
 resizeW: - W[4] + resizeWVar[4] = -82
 resizeH: - H[4] + resizeHVar[4] = -41
 moveX: - L[4] + moveX[4] = 0
 moveY: - T[4] + moveY[4] = 0
 resizeW: - W[5] + resizeWVar[5] = -135
 resizeH: - H[5] + resizeHVar[5] = -24
 moveX: - L[5] + moveX[5] = 0
 moveY: - T[5] + moveY[5] = 0
 DisableNon-PermittedBasedOnPrespecified: ABOVE[5,0] + ABOVE[5,1]
   + ABOVE[5,2] + ABOVE[5,3] + ABOVE[5,4] = 0
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
 0(ToLeftOf)1: - L[1] + R[0] + 1080 LEFT[0,1] <= 1048
 0(Above)1: - T[1] + B[0] + 1920 ABOVE[0,1] <= 1888
 0(ConverseOfToLeftOf)1: L[1] - R[0] - 1080 LEFT[0,1] <= 32
 0(ConverseOfAboveOf)1: T[1] - B[0] - 1920 ABOVE[0,1] <= 32
 0(ToLeftOf)2: - L[2] + R[0] + 1080 LEFT[0,2] <= 1048
 0(Above)2: - T[2] + B[0] + 1920 ABOVE[0,2] <= 1888
 0(ConverseOfToLeftOf)2: L[2] - R[0] - 1080 LEFT[0,2] <= 32
 0(ConverseOfAboveOf)2: T[2] - B[0] - 1920 ABOVE[0,2] <= 32
 0(ToLeftOf)3: - L[3] + R[0] + 1080 LEFT[0,3] <= 1048
 0(Above)3: - T[3] + B[0] + 1920 ABOVE[0,3] <= 1888
 0(ConverseOfToLeftOf)3: L[3] - R[0] - 1080 LEFT[0,3] <= 32
 0(ConverseOfAboveOf)3: T[3] - B[0] - 1920 ABOVE[0,3] <= 32
 0(ToLeftOf)4: - L[4] + R[0] + 1080 LEFT[0,4] <= 1048
 0(Above)4: - T[4] + B[0] + 1920 ABOVE[0,4] <= 1888
 0(ConverseOfToLeftOf)4: L[4] - R[0] - 1080 LEFT[0,4] <= 32
 0(ConverseOfAboveOf)4: T[4] - B[0] - 1920 ABOVE[0,4] <= 32
 0(ToLeftOf)5: - L[5] + R[0] + 1080 LEFT[0,5] <= 1048
 0(Above)5: - T[5] + B[0] + 1920 ABOVE[0,5] <= 1888
 0(ConverseOfToLeftOf)5: L[5] - R[0] - 1080 LEFT[0,5] <= 32
 0(ConverseOfAboveOf)5: T[5] - B[0] - 1920 ABOVE[0,5] <= 32
 1(ToLeftOf)0: - L[0] + R[1] + 1080 LEFT[1,0] <= 1048
 1(Above)0: - T[0] + B[1] + 1920 ABOVE[1,0] <= 1888
 1(ConverseOfToLeftOf)0: L[0] - R[1] - 1080 LEFT[1,0] <= 32
 1(ConverseOfAboveOf)0: T[0] - B[1] - 1920 ABOVE[1,0] <= 32
 1(ToLeftOf)2: - L[2] + R[1] + 1080 LEFT[1,2] <= 1048
 1(Above)2: - T[2] + B[1] + 1920 ABOVE[1,2] <= 1888
 1(ConverseOfToLeftOf)2: L[2] - R[1] - 1080 LEFT[1,2] <= 32
 1(ConverseOfAboveOf)2: T[2] - B[1] - 1920 ABOVE[1,2] <= 32
 1(ToLeftOf)3: - L[3] + R[1] + 1080 LEFT[1,3] <= 1048
 1(Above)3: - T[3] + B[1] + 1920 ABOVE[1,3] <= 1888
 1(ConverseOfToLeftOf)3: L[3] - R[1] - 1080 LEFT[1,3] <= 32
 1(ConverseOfAboveOf)3: T[3] - B[1] - 1920 ABOVE[1,3] <= 32
 1(ToLeftOf)4: - L[4] + R[1] + 1080 LEFT[1,4] <= 1048
 1(Above)4: - T[4] + B[1] + 1920 ABOVE[1,4] <= 1888
 1(ConverseOfToLeftOf)4: L[4] - R[1] - 1080 LEFT[1,4] <= 32
 1(ConverseOfAboveOf)4: T[4] - B[1] - 1920 ABOVE[1,4] <= 32
 1(ToLeftOf)5: - L[5] + R[1] + 1080 LEFT[1,5] <= 1048
 1(Above)5: - T[5] + B[1] + 1920 ABOVE[1,5] <= 1888
 1(ConverseOfToLeftOf)5: L[5] - R[1] - 1080 LEFT[1,5] <= 32
 1(ConverseOfAboveOf)5: T[5] - B[1] - 1920 ABOVE[1,5] <= 32
 2(ToLeftOf)0: - L[0] + R[2] + 1080 LEFT[2,0] <= 1048
 2(Above)0: - T[0] + B[2] + 1920 ABOVE[2,0] <= 1888
 2(ConverseOfToLeftOf)0: L[0] - R[2] - 1080 LEFT[2,0] <= 32
 2(ConverseOfAboveOf)0: T[0] - B[2] - 1920 ABOVE[2,0] <= 32
 2(ToLeftOf)1: - L[1] + R[2] + 1080 LEFT[2,1] <= 1048
 2(Above)1: - T[1] + B[2] + 1920 ABOVE[2,1] <= 1888
 2(ConverseOfToLeftOf)1: L[1] - R[2] - 1080 LEFT[2,1] <= 32
 2(ConverseOfAboveOf)1: T[1] - B[2] - 1920 ABOVE[2,1] <= 32
 2(ToLeftOf)3: - L[3] + R[2] + 1080 LEFT[2,3] <= 1048
 2(Above)3: - T[3] + B[2] + 1920 ABOVE[2,3] <= 1888
 2(ConverseOfToLeftOf)3: L[3] - R[2] - 1080 LEFT[2,3] <= 32
 2(ConverseOfAboveOf)3: T[3] - B[2] - 1920 ABOVE[2,3] <= 32
 2(ToLeftOf)4: - L[4] + R[2] + 1080 LEFT[2,4] <= 1048
 2(Above)4: - T[4] + B[2] + 1920 ABOVE[2,4] <= 1888
 2(ConverseOfToLeftOf)4: L[4] - R[2] - 1080 LEFT[2,4] <= 32
 2(ConverseOfAboveOf)4: T[4] - B[2] - 1920 ABOVE[2,4] <= 32
 2(ToLeftOf)5: - L[5] + R[2] + 1080 LEFT[2,5] <= 1048
 2(Above)5: - T[5] + B[2] + 1920 ABOVE[2,5] <= 1888
 2(ConverseOfToLeftOf)5: L[5] - R[2] - 1080 LEFT[2,5] <= 32
 2(ConverseOfAboveOf)5: T[5] - B[2] - 1920 ABOVE[2,5] <= 32
 3(ToLeftOf)0: - L[0] + R[3] + 1080 LEFT[3,0] <= 1048
 3(Above)0: - T[0] + B[3] + 1920 ABOVE[3,0] <= 1888
 3(ConverseOfToLeftOf)0: L[0] - R[3] - 1080 LEFT[3,0] <= 32
 3(ConverseOfAboveOf)0: T[0] - B[3] - 1920 ABOVE[3,0] <= 32
 3(ToLeftOf)1: - L[1] + R[3] + 1080 LEFT[3,1] <= 1048
 3(Above)1: - T[1] + B[3] + 1920 ABOVE[3,1] <= 1888
 3(ConverseOfToLeftOf)1: L[1] - R[3] - 1080 LEFT[3,1] <= 32
 3(ConverseOfAboveOf)1: T[1] - B[3] - 1920 ABOVE[3,1] <= 32
 3(ToLeftOf)2: - L[2] + R[3] + 1080 LEFT[3,2] <= 1048
 3(Above)2: - T[2] + B[3] + 1920 ABOVE[3,2] <= 1888
 3(ConverseOfToLeftOf)2: L[2] - R[3] - 1080 LEFT[3,2] <= 32
 3(ConverseOfAboveOf)2: T[2] - B[3] - 1920 ABOVE[3,2] <= 32
 3(ToLeftOf)4: - L[4] + R[3] + 1080 LEFT[3,4] <= 1048
 3(Above)4: - T[4] + B[3] + 1920 ABOVE[3,4] <= 1888
 3(ConverseOfToLeftOf)4: L[4] - R[3] - 1080 LEFT[3,4] <= 32
 3(ConverseOfAboveOf)4: T[4] - B[3] - 1920 ABOVE[3,4] <= 32
 3(ToLeftOf)5: - L[5] + R[3] + 1080 LEFT[3,5] <= 1048
 3(Above)5: - T[5] + B[3] + 1920 ABOVE[3,5] <= 1888
 3(ConverseOfToLeftOf)5: L[5] - R[3] - 1080 LEFT[3,5] <= 32
 3(ConverseOfAboveOf)5: T[5] - B[3] - 1920 ABOVE[3,5] <= 32
 4(ToLeftOf)0: - L[0] + R[4] + 1080 LEFT[4,0] <= 1048
 4(Above)0: - T[0] + B[4] + 1920 ABOVE[4,0] <= 1888
 4(ConverseOfToLeftOf)0: L[0] - R[4] - 1080 LEFT[4,0] <= 32
 4(ConverseOfAboveOf)0: T[0] - B[4] - 1920 ABOVE[4,0] <= 32
 4(ToLeftOf)1: - L[1] + R[4] + 1080 LEFT[4,1] <= 1048
 4(Above)1: - T[1] + B[4] + 1920 ABOVE[4,1] <= 1888
 4(ConverseOfToLeftOf)1: L[1] - R[4] - 1080 LEFT[4,1] <= 32
 4(ConverseOfAboveOf)1: T[1] - B[4] - 1920 ABOVE[4,1] <= 32
 4(ToLeftOf)2: - L[2] + R[4] + 1080 LEFT[4,2] <= 1048
 4(Above)2: - T[2] + B[4] + 1920 ABOVE[4,2] <= 1888
 4(ConverseOfToLeftOf)2: L[2] - R[4] - 1080 LEFT[4,2] <= 32
 4(ConverseOfAboveOf)2: T[2] - B[4] - 1920 ABOVE[4,2] <= 32
 4(ToLeftOf)3: - L[3] + R[4] + 1080 LEFT[4,3] <= 1048
 4(Above)3: - T[3] + B[4] + 1920 ABOVE[4,3] <= 1888
 4(ConverseOfToLeftOf)3: L[3] - R[4] - 1080 LEFT[4,3] <= 32
 4(ConverseOfAboveOf)3: T[3] - B[4] - 1920 ABOVE[4,3] <= 32
 4(ToLeftOf)5: - L[5] + R[4] + 1080 LEFT[4,5] <= 1048
 4(Above)5: - T[5] + B[4] + 1920 ABOVE[4,5] <= 1888
 4(ConverseOfToLeftOf)5: L[5] - R[4] - 1080 LEFT[4,5] <= 32
 4(ConverseOfAboveOf)5: T[5] - B[4] - 1920 ABOVE[4,5] <= 32
 5(ToLeftOf)0: - L[0] + R[5] + 1080 LEFT[5,0] <= 1048
 5(Above)0: - T[0] + B[5] + 1920 ABOVE[5,0] <= 1888
 5(ConverseOfToLeftOf)0: L[0] - R[5] - 1080 LEFT[5,0] <= 32
 5(ConverseOfAboveOf)0: T[0] - B[5] - 1920 ABOVE[5,0] <= 32
 5(ToLeftOf)1: - L[1] + R[5] + 1080 LEFT[5,1] <= 1048
 5(Above)1: - T[1] + B[5] + 1920 ABOVE[5,1] <= 1888
 5(ConverseOfToLeftOf)1: L[1] - R[5] - 1080 LEFT[5,1] <= 32
 5(ConverseOfAboveOf)1: T[1] - B[5] - 1920 ABOVE[5,1] <= 32
 5(ToLeftOf)2: - L[2] + R[5] + 1080 LEFT[5,2] <= 1048
 5(Above)2: - T[2] + B[5] + 1920 ABOVE[5,2] <= 1888
 5(ConverseOfToLeftOf)2: L[2] - R[5] - 1080 LEFT[5,2] <= 32
 5(ConverseOfAboveOf)2: T[2] - B[5] - 1920 ABOVE[5,2] <= 32
 5(ToLeftOf)3: - L[3] + R[5] + 1080 LEFT[5,3] <= 1048
 5(Above)3: - T[3] + B[5] + 1920 ABOVE[5,3] <= 1888
 5(ConverseOfToLeftOf)3: L[3] - R[5] - 1080 LEFT[5,3] <= 32
 5(ConverseOfAboveOf)3: T[3] - B[5] - 1920 ABOVE[5,3] <= 32
 5(ToLeftOf)4: - L[4] + R[5] + 1080 LEFT[5,4] <= 1048
 5(Above)4: - T[4] + B[5] + 1920 ABOVE[5,4] <= 1888
 5(ConverseOfToLeftOf)4: L[4] - R[5] - 1080 LEFT[5,4] <= 32
 5(ConverseOfAboveOf)4: T[4] - B[5] - 1920 ABOVE[5,4] <= 32
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
 R241: - LAG[0] + zLAG[0,0] <= 0
 R242: - RAG[0] + zRAG[0,0] <= 0
 R243: - TAG[0] + zTAG[0,0] <= 0
 R244: - BAG[0] + zBAG[0,0] <= 0
 R245: - LAG[0] + zLAG[1,0] <= 0
 R246: - RAG[0] + zRAG[1,0] <= 0
 R247: - TAG[0] + zTAG[1,0] <= 0
 R248: - BAG[0] + zBAG[1,0] <= 0
 R249: - LAG[0] + zLAG[2,0] <= 0
 R250: - RAG[0] + zRAG[2,0] <= 0
 R251: - TAG[0] + zTAG[2,0] <= 0
 R252: - BAG[0] + zBAG[2,0] <= 0
 R253: - LAG[0] + zLAG[3,0] <= 0
 R254: - RAG[0] + zRAG[3,0] <= 0
 R255: - TAG[0] + zTAG[3,0] <= 0
 R256: - BAG[0] + zBAG[3,0] <= 0
 R257: - LAG[0] + zLAG[4,0] <= 0
 R258: - RAG[0] + zRAG[4,0] <= 0
 R259: - TAG[0] + zTAG[4,0] <= 0
 R260: - BAG[0] + zBAG[4,0] <= 0
 R261: - LAG[0] + zLAG[5,0] <= 0
 R262: - RAG[0] + zRAG[5,0] <= 0
 R263: - TAG[0] + zTAG[5,0] <= 0
 R264: - BAG[0] + zBAG[5,0] <= 0
 R265: - LAG[1] + zLAG[0,1] <= 0
 R266: - RAG[1] + zRAG[0,1] <= 0
 R267: - TAG[1] + zTAG[0,1] <= 0
 R268: - BAG[1] + zBAG[0,1] <= 0
 R269: - LAG[1] + zLAG[1,1] <= 0
 R270: - RAG[1] + zRAG[1,1] <= 0
 R271: - TAG[1] + zTAG[1,1] <= 0
 R272: - BAG[1] + zBAG[1,1] <= 0
 R273: - LAG[1] + zLAG[2,1] <= 0
 R274: - RAG[1] + zRAG[2,1] <= 0
 R275: - TAG[1] + zTAG[2,1] <= 0
 R276: - BAG[1] + zBAG[2,1] <= 0
 R277: - LAG[1] + zLAG[3,1] <= 0
 R278: - RAG[1] + zRAG[3,1] <= 0
 R279: - TAG[1] + zTAG[3,1] <= 0
 R280: - BAG[1] + zBAG[3,1] <= 0
 R281: - LAG[1] + zLAG[4,1] <= 0
 R282: - RAG[1] + zRAG[4,1] <= 0
 R283: - TAG[1] + zTAG[4,1] <= 0
 R284: - BAG[1] + zBAG[4,1] <= 0
 R285: - LAG[1] + zLAG[5,1] <= 0
 R286: - RAG[1] + zRAG[5,1] <= 0
 R287: - TAG[1] + zTAG[5,1] <= 0
 R288: - BAG[1] + zBAG[5,1] <= 0
 R289: - LAG[2] + zLAG[0,2] <= 0
 R290: - RAG[2] + zRAG[0,2] <= 0
 R291: - TAG[2] + zTAG[0,2] <= 0
 R292: - BAG[2] + zBAG[0,2] <= 0
 R293: - LAG[2] + zLAG[1,2] <= 0
 R294: - RAG[2] + zRAG[1,2] <= 0
 R295: - TAG[2] + zTAG[1,2] <= 0
 R296: - BAG[2] + zBAG[1,2] <= 0
 R297: - LAG[2] + zLAG[2,2] <= 0
 R298: - RAG[2] + zRAG[2,2] <= 0
 R299: - TAG[2] + zTAG[2,2] <= 0
 R300: - BAG[2] + zBAG[2,2] <= 0
 R301: - LAG[2] + zLAG[3,2] <= 0
 R302: - RAG[2] + zRAG[3,2] <= 0
 R303: - TAG[2] + zTAG[3,2] <= 0
 R304: - BAG[2] + zBAG[3,2] <= 0
 R305: - LAG[2] + zLAG[4,2] <= 0
 R306: - RAG[2] + zRAG[4,2] <= 0
 R307: - TAG[2] + zTAG[4,2] <= 0
 R308: - BAG[2] + zBAG[4,2] <= 0
 R309: - LAG[2] + zLAG[5,2] <= 0
 R310: - RAG[2] + zRAG[5,2] <= 0
 R311: - TAG[2] + zTAG[5,2] <= 0
 R312: - BAG[2] + zBAG[5,2] <= 0
 R313: - LAG[3] + zLAG[0,3] <= 0
 R314: - RAG[3] + zRAG[0,3] <= 0
 R315: - TAG[3] + zTAG[0,3] <= 0
 R316: - BAG[3] + zBAG[0,3] <= 0
 R317: - LAG[3] + zLAG[1,3] <= 0
 R318: - RAG[3] + zRAG[1,3] <= 0
 R319: - TAG[3] + zTAG[1,3] <= 0
 R320: - BAG[3] + zBAG[1,3] <= 0
 R321: - LAG[3] + zLAG[2,3] <= 0
 R322: - RAG[3] + zRAG[2,3] <= 0
 R323: - TAG[3] + zTAG[2,3] <= 0
 R324: - BAG[3] + zBAG[2,3] <= 0
 R325: - LAG[3] + zLAG[3,3] <= 0
 R326: - RAG[3] + zRAG[3,3] <= 0
 R327: - TAG[3] + zTAG[3,3] <= 0
 R328: - BAG[3] + zBAG[3,3] <= 0
 R329: - LAG[3] + zLAG[4,3] <= 0
 R330: - RAG[3] + zRAG[4,3] <= 0
 R331: - TAG[3] + zTAG[4,3] <= 0
 R332: - BAG[3] + zBAG[4,3] <= 0
 R333: - LAG[3] + zLAG[5,3] <= 0
 R334: - RAG[3] + zRAG[5,3] <= 0
 R335: - TAG[3] + zTAG[5,3] <= 0
 R336: - BAG[3] + zBAG[5,3] <= 0
 R337: - LAG[4] + zLAG[0,4] <= 0
 R338: - RAG[4] + zRAG[0,4] <= 0
 R339: - TAG[4] + zTAG[0,4] <= 0
 R340: - BAG[4] + zBAG[0,4] <= 0
 R341: - LAG[4] + zLAG[1,4] <= 0
 R342: - RAG[4] + zRAG[1,4] <= 0
 R343: - TAG[4] + zTAG[1,4] <= 0
 R344: - BAG[4] + zBAG[1,4] <= 0
 R345: - LAG[4] + zLAG[2,4] <= 0
 R346: - RAG[4] + zRAG[2,4] <= 0
 R347: - TAG[4] + zTAG[2,4] <= 0
 R348: - BAG[4] + zBAG[2,4] <= 0
 R349: - LAG[4] + zLAG[3,4] <= 0
 R350: - RAG[4] + zRAG[3,4] <= 0
 R351: - TAG[4] + zTAG[3,4] <= 0
 R352: - BAG[4] + zBAG[3,4] <= 0
 R353: - LAG[4] + zLAG[4,4] <= 0
 R354: - RAG[4] + zRAG[4,4] <= 0
 R355: - TAG[4] + zTAG[4,4] <= 0
 R356: - BAG[4] + zBAG[4,4] <= 0
 R357: - LAG[4] + zLAG[5,4] <= 0
 R358: - RAG[4] + zRAG[5,4] <= 0
 R359: - TAG[4] + zTAG[5,4] <= 0
 R360: - BAG[4] + zBAG[5,4] <= 0
 R361: - LAG[5] + zLAG[0,5] <= 0
 R362: - RAG[5] + zRAG[0,5] <= 0
 R363: - TAG[5] + zTAG[0,5] <= 0
 R364: - BAG[5] + zBAG[0,5] <= 0
 R365: - LAG[5] + zLAG[1,5] <= 0
 R366: - RAG[5] + zRAG[1,5] <= 0
 R367: - TAG[5] + zTAG[1,5] <= 0
 R368: - BAG[5] + zBAG[1,5] <= 0
 R369: - LAG[5] + zLAG[2,5] <= 0
 R370: - RAG[5] + zRAG[2,5] <= 0
 R371: - TAG[5] + zTAG[2,5] <= 0
 R372: - BAG[5] + zBAG[2,5] <= 0
 R373: - LAG[5] + zLAG[3,5] <= 0
 R374: - RAG[5] + zRAG[3,5] <= 0
 R375: - TAG[5] + zTAG[3,5] <= 0
 R376: - BAG[5] + zBAG[3,5] <= 0
 R377: - LAG[5] + zLAG[4,5] <= 0
 R378: - RAG[5] + zRAG[4,5] <= 0
 R379: - TAG[5] + zTAG[4,5] <= 0
 R380: - BAG[5] + zBAG[4,5] <= 0
 R381: - LAG[5] + zLAG[5,5] <= 0
 R382: - RAG[5] + zRAG[5,5] <= 0
 R383: - TAG[5] + zTAG[5,5] <= 0
 R384: - BAG[5] + zBAG[5,5] <= 0
 MinsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] + 1080 zLAG[0,0] <= 1080
 MinsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] + 1080 zRAG[0,0] <= 1080
 MinsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] + 1920 zTAG[0,0] <= 1920
 MinsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] + 1920 zBAG[0,0] <= 1920
 MaxsideConnectL[0]ToLAG[0]: L[0] - vLAG[0] - 1080 zLAG[0,0] >= -1080
 MaxsideConnectR[0]ToRAG[0]: R[0] - vRAG[0] - 1080 zRAG[0,0] >= -1080
 MaxsideConnectT[0]ToTAG[0]: T[0] - vTAG[0] - 1920 zTAG[0,0] >= -1920
 MaxsideConnectB[0]ToBAG[0]: B[0] - vBAG[0] - 1920 zBAG[0,0] >= -1920
 MinsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] + 1080 zLAG[1,0] <= 1080
 MinsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] + 1080 zRAG[1,0] <= 1080
 MinsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] + 1920 zTAG[1,0] <= 1920
 MinsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] + 1920 zBAG[1,0] <= 1920
 MaxsideConnectL[1]ToLAG[0]: L[1] - vLAG[0] - 1080 zLAG[1,0] >= -1080
 MaxsideConnectR[1]ToRAG[0]: R[1] - vRAG[0] - 1080 zRAG[1,0] >= -1080
 MaxsideConnectT[1]ToTAG[0]: T[1] - vTAG[0] - 1920 zTAG[1,0] >= -1920
 MaxsideConnectB[1]ToBAG[0]: B[1] - vBAG[0] - 1920 zBAG[1,0] >= -1920
 MinsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] + 1080 zLAG[2,0] <= 1080
 MinsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] + 1080 zRAG[2,0] <= 1080
 MinsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] + 1920 zTAG[2,0] <= 1920
 MinsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] + 1920 zBAG[2,0] <= 1920
 MaxsideConnectL[2]ToLAG[0]: L[2] - vLAG[0] - 1080 zLAG[2,0] >= -1080
 MaxsideConnectR[2]ToRAG[0]: R[2] - vRAG[0] - 1080 zRAG[2,0] >= -1080
 MaxsideConnectT[2]ToTAG[0]: T[2] - vTAG[0] - 1920 zTAG[2,0] >= -1920
 MaxsideConnectB[2]ToBAG[0]: B[2] - vBAG[0] - 1920 zBAG[2,0] >= -1920
 MinsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] + 1080 zLAG[3,0] <= 1080
 MinsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] + 1080 zRAG[3,0] <= 1080
 MinsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] + 1920 zTAG[3,0] <= 1920
 MinsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] + 1920 zBAG[3,0] <= 1920
 MaxsideConnectL[3]ToLAG[0]: L[3] - vLAG[0] - 1080 zLAG[3,0] >= -1080
 MaxsideConnectR[3]ToRAG[0]: R[3] - vRAG[0] - 1080 zRAG[3,0] >= -1080
 MaxsideConnectT[3]ToTAG[0]: T[3] - vTAG[0] - 1920 zTAG[3,0] >= -1920
 MaxsideConnectB[3]ToBAG[0]: B[3] - vBAG[0] - 1920 zBAG[3,0] >= -1920
 MinsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] + 1080 zLAG[4,0] <= 1080
 MinsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] + 1080 zRAG[4,0] <= 1080
 MinsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] + 1920 zTAG[4,0] <= 1920
 MinsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] + 1920 zBAG[4,0] <= 1920
 MaxsideConnectL[4]ToLAG[0]: L[4] - vLAG[0] - 1080 zLAG[4,0] >= -1080
 MaxsideConnectR[4]ToRAG[0]: R[4] - vRAG[0] - 1080 zRAG[4,0] >= -1080
 MaxsideConnectT[4]ToTAG[0]: T[4] - vTAG[0] - 1920 zTAG[4,0] >= -1920
 MaxsideConnectB[4]ToBAG[0]: B[4] - vBAG[0] - 1920 zBAG[4,0] >= -1920
 MinsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] + 1080 zLAG[5,0] <= 1080
 MinsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] + 1080 zRAG[5,0] <= 1080
 MinsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] + 1920 zTAG[5,0] <= 1920
 MinsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] + 1920 zBAG[5,0] <= 1920
 MaxsideConnectL[5]ToLAG[0]: L[5] - vLAG[0] - 1080 zLAG[5,0] >= -1080
 MaxsideConnectR[5]ToRAG[0]: R[5] - vRAG[0] - 1080 zRAG[5,0] >= -1080
 MaxsideConnectT[5]ToTAG[0]: T[5] - vTAG[0] - 1920 zTAG[5,0] >= -1920
 MaxsideConnectB[5]ToBAG[0]: B[5] - vBAG[0] - 1920 zBAG[5,0] >= -1920
 MinsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] + 1080 zLAG[0,1] <= 1080
 MinsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] + 1080 zRAG[0,1] <= 1080
 MinsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] + 1920 zTAG[0,1] <= 1920
 MinsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] + 1920 zBAG[0,1] <= 1920
 MaxsideConnectL[0]ToLAG[1]: L[0] - vLAG[1] - 1080 zLAG[0,1] >= -1080
 MaxsideConnectR[0]ToRAG[1]: R[0] - vRAG[1] - 1080 zRAG[0,1] >= -1080
 MaxsideConnectT[0]ToTAG[1]: T[0] - vTAG[1] - 1920 zTAG[0,1] >= -1920
 MaxsideConnectB[0]ToBAG[1]: B[0] - vBAG[1] - 1920 zBAG[0,1] >= -1920
 MinsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] + 1080 zLAG[1,1] <= 1080
 MinsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] + 1080 zRAG[1,1] <= 1080
 MinsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] + 1920 zTAG[1,1] <= 1920
 MinsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] + 1920 zBAG[1,1] <= 1920
 MaxsideConnectL[1]ToLAG[1]: L[1] - vLAG[1] - 1080 zLAG[1,1] >= -1080
 MaxsideConnectR[1]ToRAG[1]: R[1] - vRAG[1] - 1080 zRAG[1,1] >= -1080
 MaxsideConnectT[1]ToTAG[1]: T[1] - vTAG[1] - 1920 zTAG[1,1] >= -1920
 MaxsideConnectB[1]ToBAG[1]: B[1] - vBAG[1] - 1920 zBAG[1,1] >= -1920
 MinsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] + 1080 zLAG[2,1] <= 1080
 MinsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] + 1080 zRAG[2,1] <= 1080
 MinsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] + 1920 zTAG[2,1] <= 1920
 MinsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] + 1920 zBAG[2,1] <= 1920
 MaxsideConnectL[2]ToLAG[1]: L[2] - vLAG[1] - 1080 zLAG[2,1] >= -1080
 MaxsideConnectR[2]ToRAG[1]: R[2] - vRAG[1] - 1080 zRAG[2,1] >= -1080
 MaxsideConnectT[2]ToTAG[1]: T[2] - vTAG[1] - 1920 zTAG[2,1] >= -1920
 MaxsideConnectB[2]ToBAG[1]: B[2] - vBAG[1] - 1920 zBAG[2,1] >= -1920
 MinsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] + 1080 zLAG[3,1] <= 1080
 MinsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] + 1080 zRAG[3,1] <= 1080
 MinsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] + 1920 zTAG[3,1] <= 1920
 MinsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] + 1920 zBAG[3,1] <= 1920
 MaxsideConnectL[3]ToLAG[1]: L[3] - vLAG[1] - 1080 zLAG[3,1] >= -1080
 MaxsideConnectR[3]ToRAG[1]: R[3] - vRAG[1] - 1080 zRAG[3,1] >= -1080
 MaxsideConnectT[3]ToTAG[1]: T[3] - vTAG[1] - 1920 zTAG[3,1] >= -1920
 MaxsideConnectB[3]ToBAG[1]: B[3] - vBAG[1] - 1920 zBAG[3,1] >= -1920
 MinsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] + 1080 zLAG[4,1] <= 1080
 MinsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] + 1080 zRAG[4,1] <= 1080
 MinsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] + 1920 zTAG[4,1] <= 1920
 MinsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] + 1920 zBAG[4,1] <= 1920
 MaxsideConnectL[4]ToLAG[1]: L[4] - vLAG[1] - 1080 zLAG[4,1] >= -1080
 MaxsideConnectR[4]ToRAG[1]: R[4] - vRAG[1] - 1080 zRAG[4,1] >= -1080
 MaxsideConnectT[4]ToTAG[1]: T[4] - vTAG[1] - 1920 zTAG[4,1] >= -1920
 MaxsideConnectB[4]ToBAG[1]: B[4] - vBAG[1] - 1920 zBAG[4,1] >= -1920
 MinsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] + 1080 zLAG[5,1] <= 1080
 MinsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] + 1080 zRAG[5,1] <= 1080
 MinsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] + 1920 zTAG[5,1] <= 1920
 MinsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] + 1920 zBAG[5,1] <= 1920
 MaxsideConnectL[5]ToLAG[1]: L[5] - vLAG[1] - 1080 zLAG[5,1] >= -1080
 MaxsideConnectR[5]ToRAG[1]: R[5] - vRAG[1] - 1080 zRAG[5,1] >= -1080
 MaxsideConnectT[5]ToTAG[1]: T[5] - vTAG[1] - 1920 zTAG[5,1] >= -1920
 MaxsideConnectB[5]ToBAG[1]: B[5] - vBAG[1] - 1920 zBAG[5,1] >= -1920
 MinsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] + 1080 zLAG[0,2] <= 1080
 MinsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] + 1080 zRAG[0,2] <= 1080
 MinsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] + 1920 zTAG[0,2] <= 1920
 MinsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] + 1920 zBAG[0,2] <= 1920
 MaxsideConnectL[0]ToLAG[2]: L[0] - vLAG[2] - 1080 zLAG[0,2] >= -1080
 MaxsideConnectR[0]ToRAG[2]: R[0] - vRAG[2] - 1080 zRAG[0,2] >= -1080
 MaxsideConnectT[0]ToTAG[2]: T[0] - vTAG[2] - 1920 zTAG[0,2] >= -1920
 MaxsideConnectB[0]ToBAG[2]: B[0] - vBAG[2] - 1920 zBAG[0,2] >= -1920
 MinsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] + 1080 zLAG[1,2] <= 1080
 MinsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] + 1080 zRAG[1,2] <= 1080
 MinsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] + 1920 zTAG[1,2] <= 1920
 MinsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] + 1920 zBAG[1,2] <= 1920
 MaxsideConnectL[1]ToLAG[2]: L[1] - vLAG[2] - 1080 zLAG[1,2] >= -1080
 MaxsideConnectR[1]ToRAG[2]: R[1] - vRAG[2] - 1080 zRAG[1,2] >= -1080
 MaxsideConnectT[1]ToTAG[2]: T[1] - vTAG[2] - 1920 zTAG[1,2] >= -1920
 MaxsideConnectB[1]ToBAG[2]: B[1] - vBAG[2] - 1920 zBAG[1,2] >= -1920
 MinsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] + 1080 zLAG[2,2] <= 1080
 MinsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] + 1080 zRAG[2,2] <= 1080
 MinsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] + 1920 zTAG[2,2] <= 1920
 MinsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] + 1920 zBAG[2,2] <= 1920
 MaxsideConnectL[2]ToLAG[2]: L[2] - vLAG[2] - 1080 zLAG[2,2] >= -1080
 MaxsideConnectR[2]ToRAG[2]: R[2] - vRAG[2] - 1080 zRAG[2,2] >= -1080
 MaxsideConnectT[2]ToTAG[2]: T[2] - vTAG[2] - 1920 zTAG[2,2] >= -1920
 MaxsideConnectB[2]ToBAG[2]: B[2] - vBAG[2] - 1920 zBAG[2,2] >= -1920
 MinsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] + 1080 zLAG[3,2] <= 1080
 MinsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] + 1080 zRAG[3,2] <= 1080
 MinsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] + 1920 zTAG[3,2] <= 1920
 MinsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] + 1920 zBAG[3,2] <= 1920
 MaxsideConnectL[3]ToLAG[2]: L[3] - vLAG[2] - 1080 zLAG[3,2] >= -1080
 MaxsideConnectR[3]ToRAG[2]: R[3] - vRAG[2] - 1080 zRAG[3,2] >= -1080
 MaxsideConnectT[3]ToTAG[2]: T[3] - vTAG[2] - 1920 zTAG[3,2] >= -1920
 MaxsideConnectB[3]ToBAG[2]: B[3] - vBAG[2] - 1920 zBAG[3,2] >= -1920
 MinsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] + 1080 zLAG[4,2] <= 1080
 MinsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] + 1080 zRAG[4,2] <= 1080
 MinsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] + 1920 zTAG[4,2] <= 1920
 MinsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] + 1920 zBAG[4,2] <= 1920
 MaxsideConnectL[4]ToLAG[2]: L[4] - vLAG[2] - 1080 zLAG[4,2] >= -1080
 MaxsideConnectR[4]ToRAG[2]: R[4] - vRAG[2] - 1080 zRAG[4,2] >= -1080
 MaxsideConnectT[4]ToTAG[2]: T[4] - vTAG[2] - 1920 zTAG[4,2] >= -1920
 MaxsideConnectB[4]ToBAG[2]: B[4] - vBAG[2] - 1920 zBAG[4,2] >= -1920
 MinsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] + 1080 zLAG[5,2] <= 1080
 MinsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] + 1080 zRAG[5,2] <= 1080
 MinsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] + 1920 zTAG[5,2] <= 1920
 MinsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] + 1920 zBAG[5,2] <= 1920
 MaxsideConnectL[5]ToLAG[2]: L[5] - vLAG[2] - 1080 zLAG[5,2] >= -1080
 MaxsideConnectR[5]ToRAG[2]: R[5] - vRAG[2] - 1080 zRAG[5,2] >= -1080
 MaxsideConnectT[5]ToTAG[2]: T[5] - vTAG[2] - 1920 zTAG[5,2] >= -1920
 MaxsideConnectB[5]ToBAG[2]: B[5] - vBAG[2] - 1920 zBAG[5,2] >= -1920
 MinsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] + 1080 zLAG[0,3] <= 1080
 MinsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] + 1080 zRAG[0,3] <= 1080
 MinsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] + 1920 zTAG[0,3] <= 1920
 MinsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] + 1920 zBAG[0,3] <= 1920
 MaxsideConnectL[0]ToLAG[3]: L[0] - vLAG[3] - 1080 zLAG[0,3] >= -1080
 MaxsideConnectR[0]ToRAG[3]: R[0] - vRAG[3] - 1080 zRAG[0,3] >= -1080
 MaxsideConnectT[0]ToTAG[3]: T[0] - vTAG[3] - 1920 zTAG[0,3] >= -1920
 MaxsideConnectB[0]ToBAG[3]: B[0] - vBAG[3] - 1920 zBAG[0,3] >= -1920
 MinsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] + 1080 zLAG[1,3] <= 1080
 MinsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] + 1080 zRAG[1,3] <= 1080
 MinsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] + 1920 zTAG[1,3] <= 1920
 MinsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] + 1920 zBAG[1,3] <= 1920
 MaxsideConnectL[1]ToLAG[3]: L[1] - vLAG[3] - 1080 zLAG[1,3] >= -1080
 MaxsideConnectR[1]ToRAG[3]: R[1] - vRAG[3] - 1080 zRAG[1,3] >= -1080
 MaxsideConnectT[1]ToTAG[3]: T[1] - vTAG[3] - 1920 zTAG[1,3] >= -1920
 MaxsideConnectB[1]ToBAG[3]: B[1] - vBAG[3] - 1920 zBAG[1,3] >= -1920
 MinsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] + 1080 zLAG[2,3] <= 1080
 MinsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] + 1080 zRAG[2,3] <= 1080
 MinsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] + 1920 zTAG[2,3] <= 1920
 MinsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] + 1920 zBAG[2,3] <= 1920
 MaxsideConnectL[2]ToLAG[3]: L[2] - vLAG[3] - 1080 zLAG[2,3] >= -1080
 MaxsideConnectR[2]ToRAG[3]: R[2] - vRAG[3] - 1080 zRAG[2,3] >= -1080
 MaxsideConnectT[2]ToTAG[3]: T[2] - vTAG[3] - 1920 zTAG[2,3] >= -1920
 MaxsideConnectB[2]ToBAG[3]: B[2] - vBAG[3] - 1920 zBAG[2,3] >= -1920
 MinsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] + 1080 zLAG[3,3] <= 1080
 MinsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] + 1080 zRAG[3,3] <= 1080
 MinsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] + 1920 zTAG[3,3] <= 1920
 MinsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] + 1920 zBAG[3,3] <= 1920
 MaxsideConnectL[3]ToLAG[3]: L[3] - vLAG[3] - 1080 zLAG[3,3] >= -1080
 MaxsideConnectR[3]ToRAG[3]: R[3] - vRAG[3] - 1080 zRAG[3,3] >= -1080
 MaxsideConnectT[3]ToTAG[3]: T[3] - vTAG[3] - 1920 zTAG[3,3] >= -1920
 MaxsideConnectB[3]ToBAG[3]: B[3] - vBAG[3] - 1920 zBAG[3,3] >= -1920
 MinsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] + 1080 zLAG[4,3] <= 1080
 MinsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] + 1080 zRAG[4,3] <= 1080
 MinsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] + 1920 zTAG[4,3] <= 1920
 MinsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] + 1920 zBAG[4,3] <= 1920
 MaxsideConnectL[4]ToLAG[3]: L[4] - vLAG[3] - 1080 zLAG[4,3] >= -1080
 MaxsideConnectR[4]ToRAG[3]: R[4] - vRAG[3] - 1080 zRAG[4,3] >= -1080
 MaxsideConnectT[4]ToTAG[3]: T[4] - vTAG[3] - 1920 zTAG[4,3] >= -1920
 MaxsideConnectB[4]ToBAG[3]: B[4] - vBAG[3] - 1920 zBAG[4,3] >= -1920
 MinsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] + 1080 zLAG[5,3] <= 1080
 MinsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] + 1080 zRAG[5,3] <= 1080
 MinsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] + 1920 zTAG[5,3] <= 1920
 MinsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] + 1920 zBAG[5,3] <= 1920
 MaxsideConnectL[5]ToLAG[3]: L[5] - vLAG[3] - 1080 zLAG[5,3] >= -1080
 MaxsideConnectR[5]ToRAG[3]: R[5] - vRAG[3] - 1080 zRAG[5,3] >= -1080
 MaxsideConnectT[5]ToTAG[3]: T[5] - vTAG[3] - 1920 zTAG[5,3] >= -1920
 MaxsideConnectB[5]ToBAG[3]: B[5] - vBAG[3] - 1920 zBAG[5,3] >= -1920
 MinsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] + 1080 zLAG[0,4] <= 1080
 MinsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] + 1080 zRAG[0,4] <= 1080
 MinsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] + 1920 zTAG[0,4] <= 1920
 MinsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] + 1920 zBAG[0,4] <= 1920
 MaxsideConnectL[0]ToLAG[4]: L[0] - vLAG[4] - 1080 zLAG[0,4] >= -1080
 MaxsideConnectR[0]ToRAG[4]: R[0] - vRAG[4] - 1080 zRAG[0,4] >= -1080
 MaxsideConnectT[0]ToTAG[4]: T[0] - vTAG[4] - 1920 zTAG[0,4] >= -1920
 MaxsideConnectB[0]ToBAG[4]: B[0] - vBAG[4] - 1920 zBAG[0,4] >= -1920
 MinsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] + 1080 zLAG[1,4] <= 1080
 MinsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] + 1080 zRAG[1,4] <= 1080
 MinsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] + 1920 zTAG[1,4] <= 1920
 MinsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] + 1920 zBAG[1,4] <= 1920
 MaxsideConnectL[1]ToLAG[4]: L[1] - vLAG[4] - 1080 zLAG[1,4] >= -1080
 MaxsideConnectR[1]ToRAG[4]: R[1] - vRAG[4] - 1080 zRAG[1,4] >= -1080
 MaxsideConnectT[1]ToTAG[4]: T[1] - vTAG[4] - 1920 zTAG[1,4] >= -1920
 MaxsideConnectB[1]ToBAG[4]: B[1] - vBAG[4] - 1920 zBAG[1,4] >= -1920
 MinsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] + 1080 zLAG[2,4] <= 1080
 MinsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] + 1080 zRAG[2,4] <= 1080
 MinsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] + 1920 zTAG[2,4] <= 1920
 MinsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] + 1920 zBAG[2,4] <= 1920
 MaxsideConnectL[2]ToLAG[4]: L[2] - vLAG[4] - 1080 zLAG[2,4] >= -1080
 MaxsideConnectR[2]ToRAG[4]: R[2] - vRAG[4] - 1080 zRAG[2,4] >= -1080
 MaxsideConnectT[2]ToTAG[4]: T[2] - vTAG[4] - 1920 zTAG[2,4] >= -1920
 MaxsideConnectB[2]ToBAG[4]: B[2] - vBAG[4] - 1920 zBAG[2,4] >= -1920
 MinsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] + 1080 zLAG[3,4] <= 1080
 MinsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] + 1080 zRAG[3,4] <= 1080
 MinsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] + 1920 zTAG[3,4] <= 1920
 MinsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] + 1920 zBAG[3,4] <= 1920
 MaxsideConnectL[3]ToLAG[4]: L[3] - vLAG[4] - 1080 zLAG[3,4] >= -1080
 MaxsideConnectR[3]ToRAG[4]: R[3] - vRAG[4] - 1080 zRAG[3,4] >= -1080
 MaxsideConnectT[3]ToTAG[4]: T[3] - vTAG[4] - 1920 zTAG[3,4] >= -1920
 MaxsideConnectB[3]ToBAG[4]: B[3] - vBAG[4] - 1920 zBAG[3,4] >= -1920
 MinsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] + 1080 zLAG[4,4] <= 1080
 MinsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] + 1080 zRAG[4,4] <= 1080
 MinsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] + 1920 zTAG[4,4] <= 1920
 MinsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] + 1920 zBAG[4,4] <= 1920
 MaxsideConnectL[4]ToLAG[4]: L[4] - vLAG[4] - 1080 zLAG[4,4] >= -1080
 MaxsideConnectR[4]ToRAG[4]: R[4] - vRAG[4] - 1080 zRAG[4,4] >= -1080
 MaxsideConnectT[4]ToTAG[4]: T[4] - vTAG[4] - 1920 zTAG[4,4] >= -1920
 MaxsideConnectB[4]ToBAG[4]: B[4] - vBAG[4] - 1920 zBAG[4,4] >= -1920
 MinsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] + 1080 zLAG[5,4] <= 1080
 MinsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] + 1080 zRAG[5,4] <= 1080
 MinsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] + 1920 zTAG[5,4] <= 1920
 MinsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] + 1920 zBAG[5,4] <= 1920
 MaxsideConnectL[5]ToLAG[4]: L[5] - vLAG[4] - 1080 zLAG[5,4] >= -1080
 MaxsideConnectR[5]ToRAG[4]: R[5] - vRAG[4] - 1080 zRAG[5,4] >= -1080
 MaxsideConnectT[5]ToTAG[4]: T[5] - vTAG[4] - 1920 zTAG[5,4] >= -1920
 MaxsideConnectB[5]ToBAG[4]: B[5] - vBAG[4] - 1920 zBAG[5,4] >= -1920
 MinsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] + 1080 zLAG[0,5] <= 1080
 MinsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] + 1080 zRAG[0,5] <= 1080
 MinsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] + 1920 zTAG[0,5] <= 1920
 MinsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] + 1920 zBAG[0,5] <= 1920
 MaxsideConnectL[0]ToLAG[5]: L[0] - vLAG[5] - 1080 zLAG[0,5] >= -1080
 MaxsideConnectR[0]ToRAG[5]: R[0] - vRAG[5] - 1080 zRAG[0,5] >= -1080
 MaxsideConnectT[0]ToTAG[5]: T[0] - vTAG[5] - 1920 zTAG[0,5] >= -1920
 MaxsideConnectB[0]ToBAG[5]: B[0] - vBAG[5] - 1920 zBAG[0,5] >= -1920
 MinsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] + 1080 zLAG[1,5] <= 1080
 MinsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] + 1080 zRAG[1,5] <= 1080
 MinsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] + 1920 zTAG[1,5] <= 1920
 MinsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] + 1920 zBAG[1,5] <= 1920
 MaxsideConnectL[1]ToLAG[5]: L[1] - vLAG[5] - 1080 zLAG[1,5] >= -1080
 MaxsideConnectR[1]ToRAG[5]: R[1] - vRAG[5] - 1080 zRAG[1,5] >= -1080
 MaxsideConnectT[1]ToTAG[5]: T[1] - vTAG[5] - 1920 zTAG[1,5] >= -1920
 MaxsideConnectB[1]ToBAG[5]: B[1] - vBAG[5] - 1920 zBAG[1,5] >= -1920
 MinsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] + 1080 zLAG[2,5] <= 1080
 MinsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] + 1080 zRAG[2,5] <= 1080
 MinsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] + 1920 zTAG[2,5] <= 1920
 MinsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] + 1920 zBAG[2,5] <= 1920
 MaxsideConnectL[2]ToLAG[5]: L[2] - vLAG[5] - 1080 zLAG[2,5] >= -1080
 MaxsideConnectR[2]ToRAG[5]: R[2] - vRAG[5] - 1080 zRAG[2,5] >= -1080
 MaxsideConnectT[2]ToTAG[5]: T[2] - vTAG[5] - 1920 zTAG[2,5] >= -1920
 MaxsideConnectB[2]ToBAG[5]: B[2] - vBAG[5] - 1920 zBAG[2,5] >= -1920
 MinsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] + 1080 zLAG[3,5] <= 1080
 MinsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] + 1080 zRAG[3,5] <= 1080
 MinsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] + 1920 zTAG[3,5] <= 1920
 MinsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] + 1920 zBAG[3,5] <= 1920
 MaxsideConnectL[3]ToLAG[5]: L[3] - vLAG[5] - 1080 zLAG[3,5] >= -1080
 MaxsideConnectR[3]ToRAG[5]: R[3] - vRAG[5] - 1080 zRAG[3,5] >= -1080
 MaxsideConnectT[3]ToTAG[5]: T[3] - vTAG[5] - 1920 zTAG[3,5] >= -1920
 MaxsideConnectB[3]ToBAG[5]: B[3] - vBAG[5] - 1920 zBAG[3,5] >= -1920
 MinsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] + 1080 zLAG[4,5] <= 1080
 MinsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] + 1080 zRAG[4,5] <= 1080
 MinsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] + 1920 zTAG[4,5] <= 1920
 MinsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] + 1920 zBAG[4,5] <= 1920
 MaxsideConnectL[4]ToLAG[5]: L[4] - vLAG[5] - 1080 zLAG[4,5] >= -1080
 MaxsideConnectR[4]ToRAG[5]: R[4] - vRAG[5] - 1080 zRAG[4,5] >= -1080
 MaxsideConnectT[4]ToTAG[5]: T[4] - vTAG[5] - 1920 zTAG[4,5] >= -1920
 MaxsideConnectB[4]ToBAG[5]: B[4] - vBAG[5] - 1920 zBAG[4,5] >= -1920
 MinsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] + 1080 zLAG[5,5] <= 1080
 MinsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] + 1080 zRAG[5,5] <= 1080
 MinsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] + 1920 zTAG[5,5] <= 1920
 MinsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] + 1920 zBAG[5,5] <= 1920
 MaxsideConnectL[5]ToLAG[5]: L[5] - vLAG[5] - 1080 zLAG[5,5] >= -1080
 MaxsideConnectR[5]ToRAG[5]: R[5] - vRAG[5] - 1080 zRAG[5,5] >= -1080
 MaxsideConnectT[5]ToTAG[5]: T[5] - vTAG[5] - 1920 zTAG[5,5] >= -1920
 MaxsideConnectB[5]ToBAG[5]: B[5] - vBAG[5] - 1920 zBAG[5,5] >= -1920
 R673: - R[0] + maxX >= 0
 R674: - B[0] + maxY >= 0
 R675: - R[1] + maxX >= 0
 R676: - B[1] + maxY >= 0
 R677: - R[2] + maxX >= 0
 R678: - B[2] + maxY >= 0
 R679: - R[3] + maxX >= 0
 R680: - B[3] + maxY >= 0
 R681: - R[4] + maxX >= 0
 R682: - B[4] + maxY >= 0
 R683: - R[5] + maxX >= 0
 R684: - B[5] + maxY >= 0
 R685: LAG[0] + LAG[1] + LAG[2] + LAG[3] + LAG[4] + LAG[5] + RAG[0]
   + RAG[1] + RAG[2] + RAG[3] + RAG[4] + RAG[5] + TAG[0] + TAG[1] + TAG[2]
   + TAG[3] + TAG[4] + TAG[5] + BAG[0] + BAG[1] + BAG[2] + BAG[3] + BAG[4]
   + BAG[5] >= 10
Bounds
 L[0] <= 134.875
 L[1] <= 134.875
 L[2] <= 134.875
 L[3] <= 134.875
 L[4] <= 134.875
 L[5] <= 134.875
 1 <= R[0] <= 135
 1 <= R[1] <= 135
 1 <= R[2] <= 135
 1 <= R[3] <= 135
 1 <= R[4] <= 135
 1 <= R[5] <= 135
 T[0] <= 239.875
 T[1] <= 239.875
 T[2] <= 239.875
 T[3] <= 239.875
 T[4] <= 239.875
 T[5] <= 239.875
 1 <= B[0] <= 240
 1 <= B[1] <= 240
 1 <= B[2] <= 240
 1 <= B[3] <= 240
 1 <= B[4] <= 240
 1 <= B[5] <= 240
 1 <= H[0] <= 240
 1 <= H[1] <= 240
 1 <= H[2] <= 240
 1 <= H[3] <= 240
 1 <= H[4] <= 240
 1 <= H[5] <= 240
 1 <= W[0] <= 135
 1 <= W[1] <= 135
 1 <= W[2] <= 135
 1 <= W[3] <= 135
 1 <= W[4] <= 135
 1 <= W[5] <= 135
 resizeWVar[0] >= -1000
 resizeWVar[1] >= -1000
 resizeWVar[2] >= -1000
 resizeWVar[3] >= -1000
 resizeWVar[4] >= -1000
 resizeWVar[5] >= -1000
 resizeHVar[0] >= -1000
 resizeHVar[1] >= -1000
 resizeHVar[2] >= -1000
 resizeHVar[3] >= -1000
 resizeHVar[4] >= -1000
 resizeHVar[5] >= -1000
 moveX[0] >= -1000
 moveX[1] >= -1000
 moveX[2] >= -1000
 moveX[3] >= -1000
 moveX[4] >= -1000
 moveX[5] >= -1000
 moveY[0] >= -1000
 moveY[1] >= -1000
 moveY[2] >= -1000
 moveY[3] >= -1000
 moveY[4] >= -1000
 moveY[5] >= -1000
 vLAG[0] <= 134
 vLAG[1] <= 134
 vLAG[2] <= 134
 vLAG[3] <= 134
 vLAG[4] <= 134
 vLAG[5] <= 134
 1 <= vRAG[0] <= 135
 1 <= vRAG[1] <= 135
 1 <= vRAG[2] <= 135
 1 <= vRAG[3] <= 135
 1 <= vRAG[4] <= 135
 1 <= vRAG[5] <= 135
 vTAG[0] <= 239
 vTAG[1] <= 239
 vTAG[2] <= 239
 vTAG[3] <= 239
 vTAG[4] <= 239
 vTAG[5] <= 239
 1 <= vBAG[0] <= 240
 1 <= vBAG[1] <= 240
 1 <= vBAG[2] <= 240
 1 <= vBAG[3] <= 240
 1 <= vBAG[4] <= 240
 1 <= vBAG[5] <= 240
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
 W[0] W[1] W[2] W[3] W[4] W[5] resizeWVar[0] resizeWVar[1] resizeWVar[2]
 resizeWVar[3] resizeWVar[4] resizeWVar[5] resizeWAbsVar[0]
 resizeWAbsVar[1] resizeWAbsVar[2] resizeWAbsVar[3] resizeWAbsVar[4]
 resizeWAbsVar[5] resizeHVar[0] resizeHVar[1] resizeHVar[2] resizeHVar[3]
 resizeHVar[4] resizeHVar[5] resizeHAbsVar[0] resizeHAbsVar[1]
 resizeHAbsVar[2] resizeHAbsVar[3] resizeHAbsVar[4] resizeHAbsVar[5]
 moveX[0] moveX[1] moveX[2] moveX[3] moveX[4] moveX[5] moveXAbs[0]
 moveXAbs[1] moveXAbs[2] moveXAbs[3] moveXAbs[4] moveXAbs[5] moveY[0]
 moveY[1] moveY[2] moveY[3] moveY[4] moveY[5] moveYAbs[0] moveYAbs[1]
 moveYAbs[2] moveYAbs[3] moveYAbs[4] moveYAbs[5] vLAG[0] vLAG[1] vLAG[2]
 vLAG[3] vLAG[4] vLAG[5] vRAG[0] vRAG[1] vRAG[2] vRAG[3] vRAG[4] vRAG[5]
 vTAG[0] vTAG[1] vTAG[2] vTAG[3] vTAG[4] vTAG[5] vBAG[0] vBAG[1] vBAG[2]
 vBAG[3] vBAG[4] vBAG[5] maxX maxY
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
End
