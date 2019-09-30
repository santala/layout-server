\ Model GLayoutQuality
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.001 ResizeWAbs[0] + 0.005 ResizeWAbs[1] + 0.0025 ResizeWAbs[2]
   + 0.0025 ResizeWAbs[3] + 0.0025 ResizeWAbs[4] + 0.0025 ResizeWAbs[5]
   + 0.0025 ResizeWAbs[6] + 0.001 ResizeWAbs[7] + 0.01 ResizeHAbs[0]
   + 0.00142857142857143 ResizeHAbs[1] + 0.00333333333333333 ResizeHAbs[2]
   + 0.00333333333333333 ResizeHAbs[3] + 0.0025 ResizeHAbs[4]
   + 0.005 ResizeHAbs[5] + 0.005 ResizeHAbs[6]
   + 0.00666666666666667 ResizeHAbs[7] + 0.001 MoveXAbs[0]
   + 0.005 MoveXAbs[1] + 0.0025 MoveXAbs[2] + 0.0025 MoveXAbs[3]
   + 0.0025 MoveXAbs[4] + 0.0025 MoveXAbs[5] + 0.0025 MoveXAbs[6]
   + 0.001 MoveXAbs[7] + 0.01 MoveYAbs[0] + 0.00142857142857143 MoveYAbs[1]
   + 0.00333333333333333 MoveYAbs[2] + 0.00333333333333333 MoveYAbs[3]
   + 0.0025 MoveYAbs[4] + 0.005 MoveYAbs[5] + 0.005 MoveYAbs[6]
   + 0.00666666666666667 MoveYAbs[7] + 0 Above[0,0] + 0 Above[1,1]
   + 0 Above[2,2] + 0 Above[3,3] + 0 Above[4,4] + 0 Above[5,5]
   + 0 Above[6,6] + 0 Above[7,7] + 0 OnLeft[0,0] + 0 OnLeft[1,1]
   + 0 OnLeft[2,2] + 0 OnLeft[3,3] + 0 OnLeft[4,4] + 0 OnLeft[5,5]
   + 0 OnLeft[6,6] + 0 OnLeft[7,7] + 0 VerticalOverlap[0,0]
   + 0 VerticalOverlap[1,1] + 0 VerticalOverlap[2,2]
   + 0 VerticalOverlap[3,3] + 0 VerticalOverlap[4,4]
   + 0 VerticalOverlap[5,5] + 0 VerticalOverlap[6,6]
   + 0 VerticalOverlap[7,7] + 0 Overlap[0,0] + 0 Overlap[1,1]
   + 0 Overlap[2,2] + 0 Overlap[3,3] + 0 Overlap[4,4] + 0 Overlap[5,5]
   + 0 Overlap[6,6] + 0 Overlap[7,7] + 0 X0LessThan[0,0]
   + 0 X0LessThan[0,1] + 0 X0LessThan[0,2] + 0 X0LessThan[0,3]
   + 0 X0LessThan[0,4] + 0 X0LessThan[0,5] + 0 X0LessThan[0,6]
   + 0 X0LessThan[0,7] + 0 X0LessThan[1,0] + 0 X0LessThan[1,1]
   + 0 X0LessThan[1,2] + 0 X0LessThan[1,3] + 0 X0LessThan[1,4]
   + 0 X0LessThan[1,5] + 0 X0LessThan[1,6] + 0 X0LessThan[1,7]
   + 0 X0LessThan[2,0] + 0 X0LessThan[2,1] + 0 X0LessThan[2,2]
   + 0 X0LessThan[2,3] + 0 X0LessThan[2,4] + 0 X0LessThan[2,5]
   + 0 X0LessThan[2,6] + 0 X0LessThan[2,7] + 0 X0LessThan[3,0]
   + 0 X0LessThan[3,1] + 0 X0LessThan[3,2] + 0 X0LessThan[3,3]
   + 0 X0LessThan[3,4] + 0 X0LessThan[3,5] + 0 X0LessThan[3,6]
   + 0 X0LessThan[3,7] + 0 X0LessThan[4,0] + 0 X0LessThan[4,1]
   + 0 X0LessThan[4,2] + 0 X0LessThan[4,3] + 0 X0LessThan[4,4]
   + 0 X0LessThan[4,5] + 0 X0LessThan[4,6] + 0 X0LessThan[4,7]
   + 0 X0LessThan[5,0] + 0 X0LessThan[5,1] + 0 X0LessThan[5,2]
   + 0 X0LessThan[5,3] + 0 X0LessThan[5,4] + 0 X0LessThan[5,5]
   + 0 X0LessThan[5,6] + 0 X0LessThan[5,7] + 0 X0LessThan[6,0]
   + 0 X0LessThan[6,1] + 0 X0LessThan[6,2] + 0 X0LessThan[6,3]
   + 0 X0LessThan[6,4] + 0 X0LessThan[6,5] + 0 X0LessThan[6,6]
   + 0 X0LessThan[6,7] + 0 X0LessThan[7,0] + 0 X0LessThan[7,1]
   + 0 X0LessThan[7,2] + 0 X0LessThan[7,3] + 0 X0LessThan[7,4]
   + 0 X0LessThan[7,5] + 0 X0LessThan[7,6] + 0 X0LessThan[7,7]
   + 0 Y0LessThan[0,0] + 0 Y0LessThan[0,1] + 0 Y0LessThan[0,2]
   + 0 Y0LessThan[0,3] + 0 Y0LessThan[0,4] + 0 Y0LessThan[0,5]
   + 0 Y0LessThan[0,6] + 0 Y0LessThan[0,7] + 0 Y0LessThan[1,0]
   + 0 Y0LessThan[1,1] + 0 Y0LessThan[1,2] + 0 Y0LessThan[1,3]
   + 0 Y0LessThan[1,4] + 0 Y0LessThan[1,5] + 0 Y0LessThan[1,6]
   + 0 Y0LessThan[1,7] + 0 Y0LessThan[2,0] + 0 Y0LessThan[2,1]
   + 0 Y0LessThan[2,2] + 0 Y0LessThan[2,3] + 0 Y0LessThan[2,4]
   + 0 Y0LessThan[2,5] + 0 Y0LessThan[2,6] + 0 Y0LessThan[2,7]
   + 0 Y0LessThan[3,0] + 0 Y0LessThan[3,1] + 0 Y0LessThan[3,2]
   + 0 Y0LessThan[3,3] + 0 Y0LessThan[3,4] + 0 Y0LessThan[3,5]
   + 0 Y0LessThan[3,6] + 0 Y0LessThan[3,7] + 0 Y0LessThan[4,0]
   + 0 Y0LessThan[4,1] + 0 Y0LessThan[4,2] + 0 Y0LessThan[4,3]
   + 0 Y0LessThan[4,4] + 0 Y0LessThan[4,5] + 0 Y0LessThan[4,6]
   + 0 Y0LessThan[4,7] + 0 Y0LessThan[5,0] + 0 Y0LessThan[5,1]
   + 0 Y0LessThan[5,2] + 0 Y0LessThan[5,3] + 0 Y0LessThan[5,4]
   + 0 Y0LessThan[5,5] + 0 Y0LessThan[5,6] + 0 Y0LessThan[5,7]
   + 0 Y0LessThan[6,0] + 0 Y0LessThan[6,1] + 0 Y0LessThan[6,2]
   + 0 Y0LessThan[6,3] + 0 Y0LessThan[6,4] + 0 Y0LessThan[6,5]
   + 0 Y0LessThan[6,6] + 0 Y0LessThan[6,7] + 0 Y0LessThan[7,0]
   + 0 Y0LessThan[7,1] + 0 Y0LessThan[7,2] + 0 Y0LessThan[7,3]
   + 0 Y0LessThan[7,4] + 0 Y0LessThan[7,5] + 0 Y0LessThan[7,6]
   + 0 Y0LessThan[7,7] + 0 X1LessThan[0,0] + 0 X1LessThan[0,1]
   + 0 X1LessThan[0,2] + 0 X1LessThan[0,3] + 0 X1LessThan[0,4]
   + 0 X1LessThan[0,5] + 0 X1LessThan[0,6] + 0 X1LessThan[0,7]
   + 0 X1LessThan[1,0] + 0 X1LessThan[1,1] + 0 X1LessThan[1,2]
   + 0 X1LessThan[1,3] + 0 X1LessThan[1,4] + 0 X1LessThan[1,5]
   + 0 X1LessThan[1,6] + 0 X1LessThan[1,7] + 0 X1LessThan[2,0]
   + 0 X1LessThan[2,1] + 0 X1LessThan[2,2] + 0 X1LessThan[2,3]
   + 0 X1LessThan[2,4] + 0 X1LessThan[2,5] + 0 X1LessThan[2,6]
   + 0 X1LessThan[2,7] + 0 X1LessThan[3,0] + 0 X1LessThan[3,1]
   + 0 X1LessThan[3,2] + 0 X1LessThan[3,3] + 0 X1LessThan[3,4]
   + 0 X1LessThan[3,5] + 0 X1LessThan[3,6] + 0 X1LessThan[3,7]
   + 0 X1LessThan[4,0] + 0 X1LessThan[4,1] + 0 X1LessThan[4,2]
   + 0 X1LessThan[4,3] + 0 X1LessThan[4,4] + 0 X1LessThan[4,5]
   + 0 X1LessThan[4,6] + 0 X1LessThan[4,7] + 0 X1LessThan[5,0]
   + 0 X1LessThan[5,1] + 0 X1LessThan[5,2] + 0 X1LessThan[5,3]
   + 0 X1LessThan[5,4] + 0 X1LessThan[5,5] + 0 X1LessThan[5,6]
   + 0 X1LessThan[5,7] + 0 X1LessThan[6,0] + 0 X1LessThan[6,1]
   + 0 X1LessThan[6,2] + 0 X1LessThan[6,3] + 0 X1LessThan[6,4]
   + 0 X1LessThan[6,5] + 0 X1LessThan[6,6] + 0 X1LessThan[6,7]
   + 0 X1LessThan[7,0] + 0 X1LessThan[7,1] + 0 X1LessThan[7,2]
   + 0 X1LessThan[7,3] + 0 X1LessThan[7,4] + 0 X1LessThan[7,5]
   + 0 X1LessThan[7,6] + 0 X1LessThan[7,7] + 0 Y1LessThan[0,0]
   + 0 Y1LessThan[0,1] + 0 Y1LessThan[0,2] + 0 Y1LessThan[0,3]
   + 0 Y1LessThan[0,4] + 0 Y1LessThan[0,5] + 0 Y1LessThan[0,6]
   + 0 Y1LessThan[0,7] + 0 Y1LessThan[1,0] + 0 Y1LessThan[1,1]
   + 0 Y1LessThan[1,2] + 0 Y1LessThan[1,3] + 0 Y1LessThan[1,4]
   + 0 Y1LessThan[1,5] + 0 Y1LessThan[1,6] + 0 Y1LessThan[1,7]
   + 0 Y1LessThan[2,0] + 0 Y1LessThan[2,1] + 0 Y1LessThan[2,2]
   + 0 Y1LessThan[2,3] + 0 Y1LessThan[2,4] + 0 Y1LessThan[2,5]
   + 0 Y1LessThan[2,6] + 0 Y1LessThan[2,7] + 0 Y1LessThan[3,0]
   + 0 Y1LessThan[3,1] + 0 Y1LessThan[3,2] + 0 Y1LessThan[3,3]
   + 0 Y1LessThan[3,4] + 0 Y1LessThan[3,5] + 0 Y1LessThan[3,6]
   + 0 Y1LessThan[3,7] + 0 Y1LessThan[4,0] + 0 Y1LessThan[4,1]
   + 0 Y1LessThan[4,2] + 0 Y1LessThan[4,3] + 0 Y1LessThan[4,4]
   + 0 Y1LessThan[4,5] + 0 Y1LessThan[4,6] + 0 Y1LessThan[4,7]
   + 0 Y1LessThan[5,0] + 0 Y1LessThan[5,1] + 0 Y1LessThan[5,2]
   + 0 Y1LessThan[5,3] + 0 Y1LessThan[5,4] + 0 Y1LessThan[5,5]
   + 0 Y1LessThan[5,6] + 0 Y1LessThan[5,7] + 0 Y1LessThan[6,0]
   + 0 Y1LessThan[6,1] + 0 Y1LessThan[6,2] + 0 Y1LessThan[6,3]
   + 0 Y1LessThan[6,4] + 0 Y1LessThan[6,5] + 0 Y1LessThan[6,6]
   + 0 Y1LessThan[6,7] + 0 Y1LessThan[7,0] + 0 Y1LessThan[7,1]
   + 0 Y1LessThan[7,2] + 0 Y1LessThan[7,3] + 0 Y1LessThan[7,4]
   + 0 Y1LessThan[7,5] + 0 Y1LessThan[7,6] + 0 Y1LessThan[7,7]
   + 0 X0Group[0] + 0 X0Group[1] + 0 X0Group[2] + 0 X0Group[3]
   + 0 X0Group[4] + 0 X0Group[5] + 0 X0Group[6] + 0 X0Group[7]
   + 0 Y0Group[0] + 0 Y0Group[1] + 0 Y0Group[2] + 0 Y0Group[3]
   + 0 Y0Group[4] + 0 Y0Group[5] + 0 Y0Group[6] + 0 Y0Group[7]
   + 0 X1Group[0] + 0 X1Group[1] + 0 X1Group[2] + 0 X1Group[3]
   + 0 X1Group[4] + 0 X1Group[5] + 0 X1Group[6] + 0 X1Group[7]
   + 0 Y1Group[0] + 0 Y1Group[1] + 0 Y1Group[2] + 0 Y1Group[3]
   + 0 Y1Group[4] + 0 Y1Group[5] + 0 Y1Group[6] + 0 Y1Group[7]
   + X0GroupCount + Y0GroupCount + X1GroupCount + Y1GroupCount
   + MarginDiffLooseAbs + 0 C1232
Subject To
 X0X1Sanity[0]: X0[0] - X1[0] <= -1
 X0X1Sanity[1]: X0[1] - X1[1] <= -1
 X0X1Sanity[2]: X0[2] - X1[2] <= -1
 X0X1Sanity[3]: X0[3] - X1[3] <= -1
 X0X1Sanity[4]: X0[4] - X1[4] <= -1
 X0X1Sanity[5]: X0[5] - X1[5] <= -1
 X0X1Sanity[6]: X0[6] - X1[6] <= -1
 X0X1Sanity[7]: X0[7] - X1[7] <= -1
 X0X1Sanity[0]: Y0[0] - Y1[0] <= -1
 X0X1Sanity[1]: Y0[1] - Y1[1] <= -1
 X0X1Sanity[2]: Y0[2] - Y1[2] <= -1
 X0X1Sanity[3]: Y0[3] - Y1[3] <= -1
 X0X1Sanity[4]: Y0[4] - Y1[4] <= -1
 X0X1Sanity[5]: Y0[5] - Y1[5] <= -1
 X0X1Sanity[6]: Y0[6] - Y1[6] <= -1
 X0X1Sanity[7]: Y0[7] - Y1[7] <= -1
 LimitToCanvasW[0]: X1[0] <= 125
 LimitToCanvasW[1]: X1[1] <= 125
 LimitToCanvasW[2]: X1[2] <= 125
 LimitToCanvasW[3]: X1[3] <= 125
 LimitToCanvasW[4]: X1[4] <= 125
 LimitToCanvasW[5]: X1[5] <= 125
 LimitToCanvasW[6]: X1[6] <= 125
 LimitToCanvasW[7]: X1[7] <= 125
 LimitToCanvasH[0]: Y1[0] <= 119
 LimitToCanvasH[1]: Y1[1] <= 119
 LimitToCanvasH[2]: Y1[2] <= 119
 LimitToCanvasH[3]: Y1[3] <= 119
 LimitToCanvasH[4]: Y1[4] <= 119
 LimitToCanvasH[5]: Y1[5] <= 119
 LimitToCanvasH[6]: Y1[6] <= 119
 LimitToCanvasH[7]: Y1[7] <= 119
 X1-X0=W[0]: - X0[0] + X1[0] - W[0] = 0
 X1-X0=W[1]: - X0[1] + X1[1] - W[1] = 0
 X1-X0=W[2]: - X0[2] + X1[2] - W[2] = 0
 X1-X0=W[3]: - X0[3] + X1[3] - W[3] = 0
 X1-X0=W[4]: - X0[4] + X1[4] - W[4] = 0
 X1-X0=W[5]: - X0[5] + X1[5] - W[5] = 0
 X1-X0=W[6]: - X0[6] + X1[6] - W[6] = 0
 X1-X0=W[7]: - X0[7] + X1[7] - W[7] = 0
 Y1-Y0=H[0]: - Y0[0] + Y1[0] - H[0] = 0
 Y1-Y0=H[1]: - Y0[1] + Y1[1] - H[1] = 0
 Y1-Y0=H[2]: - Y0[2] + Y1[2] - H[2] = 0
 Y1-Y0=H[3]: - Y0[3] + Y1[3] - H[3] = 0
 Y1-Y0=H[4]: - Y0[4] + Y1[4] - H[4] = 0
 Y1-Y0=H[5]: - Y0[5] + Y1[5] - H[5] = 0
 Y1-Y0=H[6]: - Y0[6] + Y1[6] - H[6] = 0
 Y1-Y0=H[7]: - Y0[7] + Y1[7] - H[7] = 0
 LinkResizeW[0,<optimizer.classes.Element_object_at_0x103167dd8>]: - W[0]
   + ResizeW[0] = -125
 LinkResizeW[1,<optimizer.classes.Element_object_at_0x103167cc0>]: - W[1]
   + ResizeW[1] = -25
 LinkResizeW[2,<optimizer.classes.Element_object_at_0x103178400>]: - W[2]
   + ResizeW[2] = -50
 LinkResizeW[3,<optimizer.classes.Element_object_at_0x1031783c8>]: - W[3]
   + ResizeW[3] = -50
 LinkResizeW[4,<optimizer.classes.Element_object_at_0x103178390>]: - W[4]
   + ResizeW[4] = -50
 LinkResizeW[5,<optimizer.classes.Element_object_at_0x103178358>]: - W[5]
   + ResizeW[5] = -50
 LinkResizeW[6,<optimizer.classes.Element_object_at_0x103178320>]: - W[6]
   + ResizeW[6] = -50
 LinkResizeW[7,<optimizer.classes.Element_object_at_0x1031782e8>]: - W[7]
   + ResizeW[7] = -125
 LinkResizeH[0,<optimizer.classes.Element_object_at_0x103167dd8>]: - H[0]
   + ResizeH[0] = -12
 LinkResizeH[1,<optimizer.classes.Element_object_at_0x103167cc0>]: - H[1]
   + ResizeH[1] = -88
 LinkResizeH[2,<optimizer.classes.Element_object_at_0x103178400>]: - H[2]
   + ResizeH[2] = -38
 LinkResizeH[3,<optimizer.classes.Element_object_at_0x1031783c8>]: - H[3]
   + ResizeH[3] = -38
 LinkResizeH[4,<optimizer.classes.Element_object_at_0x103178390>]: - H[4]
   + ResizeH[4] = -50
 LinkResizeH[5,<optimizer.classes.Element_object_at_0x103178358>]: - H[5]
   + ResizeH[5] = -25
 LinkResizeH[6,<optimizer.classes.Element_object_at_0x103178320>]: - H[6]
   + ResizeH[6] = -25
 LinkResizeH[7,<optimizer.classes.Element_object_at_0x1031782e8>]: - H[7]
   + ResizeH[7] = -19
 LinkMoveX[0,<optimizer.classes.Element_object_at_0x103167dd8>]: - X0[0]
   + MoveX[0] = 0
 LinkMoveX[1,<optimizer.classes.Element_object_at_0x103167cc0>]: - X0[1]
   + MoveX[1] = 0
 LinkMoveX[2,<optimizer.classes.Element_object_at_0x103178400>]: - X0[2]
   + MoveX[2] = -75
 LinkMoveX[3,<optimizer.classes.Element_object_at_0x1031783c8>]: - X0[3]
   + MoveX[3] = -25
 LinkMoveX[4,<optimizer.classes.Element_object_at_0x103178390>]: - X0[4]
   + MoveX[4] = -25
 LinkMoveX[5,<optimizer.classes.Element_object_at_0x103178358>]: - X0[5]
   + MoveX[5] = -75
 LinkMoveX[6,<optimizer.classes.Element_object_at_0x103178320>]: - X0[6]
   + MoveX[6] = -75
 LinkMoveX[7,<optimizer.classes.Element_object_at_0x1031782e8>]: - X0[7]
   + MoveX[7] = 0
 LinkMoveY[0,<optimizer.classes.Element_object_at_0x103167dd8>]: - Y0[0]
   + MoveY[0] = 0
 LinkMoveY[1,<optimizer.classes.Element_object_at_0x103167cc0>]: - Y0[1]
   + MoveY[1] = -12
 LinkMoveY[2,<optimizer.classes.Element_object_at_0x103178400>]: - Y0[2]
   + MoveY[2] = -62
 LinkMoveY[3,<optimizer.classes.Element_object_at_0x1031783c8>]: - Y0[3]
   + MoveY[3] = -62
 LinkMoveY[4,<optimizer.classes.Element_object_at_0x103178390>]: - Y0[4]
   + MoveY[4] = -12
 LinkMoveY[5,<optimizer.classes.Element_object_at_0x103178358>]: - Y0[5]
   + MoveY[5] = -12
 LinkMoveY[6,<optimizer.classes.Element_object_at_0x103178320>]: - Y0[6]
   + MoveY[6] = -38
 LinkMoveY[7,<optimizer.classes.Element_object_at_0x1031782e8>]: - Y0[7]
   + MoveY[7] = -100
 AboveSanity[0,1]: Above[0,1] + Above[1,0] <= 1
 AboveSanity[0,2]: Above[0,2] + Above[2,0] <= 1
 AboveSanity[0,3]: Above[0,3] + Above[3,0] <= 1
 AboveSanity[0,4]: Above[0,4] + Above[4,0] <= 1
 AboveSanity[0,5]: Above[0,5] + Above[5,0] <= 1
 AboveSanity[0,6]: Above[0,6] + Above[6,0] <= 1
 AboveSanity[0,7]: Above[0,7] + Above[7,0] <= 1
 AboveSanity[1,0]: Above[0,1] + Above[1,0] <= 1
 AboveSanity[1,2]: Above[1,2] + Above[2,1] <= 1
 AboveSanity[1,3]: Above[1,3] + Above[3,1] <= 1
 AboveSanity[1,4]: Above[1,4] + Above[4,1] <= 1
 AboveSanity[1,5]: Above[1,5] + Above[5,1] <= 1
 AboveSanity[1,6]: Above[1,6] + Above[6,1] <= 1
 AboveSanity[1,7]: Above[1,7] + Above[7,1] <= 1
 AboveSanity[2,0]: Above[0,2] + Above[2,0] <= 1
 AboveSanity[2,1]: Above[1,2] + Above[2,1] <= 1
 AboveSanity[2,3]: Above[2,3] + Above[3,2] <= 1
 AboveSanity[2,4]: Above[2,4] + Above[4,2] <= 1
 AboveSanity[2,5]: Above[2,5] + Above[5,2] <= 1
 AboveSanity[2,6]: Above[2,6] + Above[6,2] <= 1
 AboveSanity[2,7]: Above[2,7] + Above[7,2] <= 1
 AboveSanity[3,0]: Above[0,3] + Above[3,0] <= 1
 AboveSanity[3,1]: Above[1,3] + Above[3,1] <= 1
 AboveSanity[3,2]: Above[2,3] + Above[3,2] <= 1
 AboveSanity[3,4]: Above[3,4] + Above[4,3] <= 1
 AboveSanity[3,5]: Above[3,5] + Above[5,3] <= 1
 AboveSanity[3,6]: Above[3,6] + Above[6,3] <= 1
 AboveSanity[3,7]: Above[3,7] + Above[7,3] <= 1
 AboveSanity[4,0]: Above[0,4] + Above[4,0] <= 1
 AboveSanity[4,1]: Above[1,4] + Above[4,1] <= 1
 AboveSanity[4,2]: Above[2,4] + Above[4,2] <= 1
 AboveSanity[4,3]: Above[3,4] + Above[4,3] <= 1
 AboveSanity[4,5]: Above[4,5] + Above[5,4] <= 1
 AboveSanity[4,6]: Above[4,6] + Above[6,4] <= 1
 AboveSanity[4,7]: Above[4,7] + Above[7,4] <= 1
 AboveSanity[5,0]: Above[0,5] + Above[5,0] <= 1
 AboveSanity[5,1]: Above[1,5] + Above[5,1] <= 1
 AboveSanity[5,2]: Above[2,5] + Above[5,2] <= 1
 AboveSanity[5,3]: Above[3,5] + Above[5,3] <= 1
 AboveSanity[5,4]: Above[4,5] + Above[5,4] <= 1
 AboveSanity[5,6]: Above[5,6] + Above[6,5] <= 1
 AboveSanity[5,7]: Above[5,7] + Above[7,5] <= 1
 AboveSanity[6,0]: Above[0,6] + Above[6,0] <= 1
 AboveSanity[6,1]: Above[1,6] + Above[6,1] <= 1
 AboveSanity[6,2]: Above[2,6] + Above[6,2] <= 1
 AboveSanity[6,3]: Above[3,6] + Above[6,3] <= 1
 AboveSanity[6,4]: Above[4,6] + Above[6,4] <= 1
 AboveSanity[6,5]: Above[5,6] + Above[6,5] <= 1
 AboveSanity[6,7]: Above[6,7] + Above[7,6] <= 1
 AboveSanity[7,0]: Above[0,7] + Above[7,0] <= 1
 AboveSanity[7,1]: Above[1,7] + Above[7,1] <= 1
 AboveSanity[7,2]: Above[2,7] + Above[7,2] <= 1
 AboveSanity[7,3]: Above[3,7] + Above[7,3] <= 1
 AboveSanity[7,4]: Above[4,7] + Above[7,4] <= 1
 AboveSanity[7,5]: Above[5,7] + Above[7,5] <= 1
 AboveSanity[7,6]: Above[6,7] + Above[7,6] <= 1
 OnLeftSanity[0,1]: OnLeft[0,1] + OnLeft[1,0] <= 1
 OnLeftSanity[0,2]: OnLeft[0,2] + OnLeft[2,0] <= 1
 OnLeftSanity[0,3]: OnLeft[0,3] + OnLeft[3,0] <= 1
 OnLeftSanity[0,4]: OnLeft[0,4] + OnLeft[4,0] <= 1
 OnLeftSanity[0,5]: OnLeft[0,5] + OnLeft[5,0] <= 1
 OnLeftSanity[0,6]: OnLeft[0,6] + OnLeft[6,0] <= 1
 OnLeftSanity[0,7]: OnLeft[0,7] + OnLeft[7,0] <= 1
 OnLeftSanity[1,0]: OnLeft[0,1] + OnLeft[1,0] <= 1
 OnLeftSanity[1,2]: OnLeft[1,2] + OnLeft[2,1] <= 1
 OnLeftSanity[1,3]: OnLeft[1,3] + OnLeft[3,1] <= 1
 OnLeftSanity[1,4]: OnLeft[1,4] + OnLeft[4,1] <= 1
 OnLeftSanity[1,5]: OnLeft[1,5] + OnLeft[5,1] <= 1
 OnLeftSanity[1,6]: OnLeft[1,6] + OnLeft[6,1] <= 1
 OnLeftSanity[1,7]: OnLeft[1,7] + OnLeft[7,1] <= 1
 OnLeftSanity[2,0]: OnLeft[0,2] + OnLeft[2,0] <= 1
 OnLeftSanity[2,1]: OnLeft[1,2] + OnLeft[2,1] <= 1
 OnLeftSanity[2,3]: OnLeft[2,3] + OnLeft[3,2] <= 1
 OnLeftSanity[2,4]: OnLeft[2,4] + OnLeft[4,2] <= 1
 OnLeftSanity[2,5]: OnLeft[2,5] + OnLeft[5,2] <= 1
 OnLeftSanity[2,6]: OnLeft[2,6] + OnLeft[6,2] <= 1
 OnLeftSanity[2,7]: OnLeft[2,7] + OnLeft[7,2] <= 1
 OnLeftSanity[3,0]: OnLeft[0,3] + OnLeft[3,0] <= 1
 OnLeftSanity[3,1]: OnLeft[1,3] + OnLeft[3,1] <= 1
 OnLeftSanity[3,2]: OnLeft[2,3] + OnLeft[3,2] <= 1
 OnLeftSanity[3,4]: OnLeft[3,4] + OnLeft[4,3] <= 1
 OnLeftSanity[3,5]: OnLeft[3,5] + OnLeft[5,3] <= 1
 OnLeftSanity[3,6]: OnLeft[3,6] + OnLeft[6,3] <= 1
 OnLeftSanity[3,7]: OnLeft[3,7] + OnLeft[7,3] <= 1
 OnLeftSanity[4,0]: OnLeft[0,4] + OnLeft[4,0] <= 1
 OnLeftSanity[4,1]: OnLeft[1,4] + OnLeft[4,1] <= 1
 OnLeftSanity[4,2]: OnLeft[2,4] + OnLeft[4,2] <= 1
 OnLeftSanity[4,3]: OnLeft[3,4] + OnLeft[4,3] <= 1
 OnLeftSanity[4,5]: OnLeft[4,5] + OnLeft[5,4] <= 1
 OnLeftSanity[4,6]: OnLeft[4,6] + OnLeft[6,4] <= 1
 OnLeftSanity[4,7]: OnLeft[4,7] + OnLeft[7,4] <= 1
 OnLeftSanity[5,0]: OnLeft[0,5] + OnLeft[5,0] <= 1
 OnLeftSanity[5,1]: OnLeft[1,5] + OnLeft[5,1] <= 1
 OnLeftSanity[5,2]: OnLeft[2,5] + OnLeft[5,2] <= 1
 OnLeftSanity[5,3]: OnLeft[3,5] + OnLeft[5,3] <= 1
 OnLeftSanity[5,4]: OnLeft[4,5] + OnLeft[5,4] <= 1
 OnLeftSanity[5,6]: OnLeft[5,6] + OnLeft[6,5] <= 1
 OnLeftSanity[5,7]: OnLeft[5,7] + OnLeft[7,5] <= 1
 OnLeftSanity[6,0]: OnLeft[0,6] + OnLeft[6,0] <= 1
 OnLeftSanity[6,1]: OnLeft[1,6] + OnLeft[6,1] <= 1
 OnLeftSanity[6,2]: OnLeft[2,6] + OnLeft[6,2] <= 1
 OnLeftSanity[6,3]: OnLeft[3,6] + OnLeft[6,3] <= 1
 OnLeftSanity[6,4]: OnLeft[4,6] + OnLeft[6,4] <= 1
 OnLeftSanity[6,5]: OnLeft[5,6] + OnLeft[6,5] <= 1
 OnLeftSanity[6,7]: OnLeft[6,7] + OnLeft[7,6] <= 1
 OnLeftSanity[7,0]: OnLeft[0,7] + OnLeft[7,0] <= 1
 OnLeftSanity[7,1]: OnLeft[1,7] + OnLeft[7,1] <= 1
 OnLeftSanity[7,2]: OnLeft[2,7] + OnLeft[7,2] <= 1
 OnLeftSanity[7,3]: OnLeft[3,7] + OnLeft[7,3] <= 1
 OnLeftSanity[7,4]: OnLeft[4,7] + OnLeft[7,4] <= 1
 OnLeftSanity[7,5]: OnLeft[5,7] + OnLeft[7,5] <= 1
 OnLeftSanity[7,6]: OnLeft[6,7] + OnLeft[7,6] <= 1
 LinkHorizontalOverlap[0,1]: OnLeft[0,1] + OnLeft[1,0]
   + HorizontalOverlap[0,1] = 1
 LinkHorizontalOverlap[0,2]: OnLeft[0,2] + OnLeft[2,0]
   + HorizontalOverlap[0,2] = 1
 LinkHorizontalOverlap[0,3]: OnLeft[0,3] + OnLeft[3,0]
   + HorizontalOverlap[0,3] = 1
 LinkHorizontalOverlap[0,4]: OnLeft[0,4] + OnLeft[4,0]
   + HorizontalOverlap[0,4] = 1
 LinkHorizontalOverlap[0,5]: OnLeft[0,5] + OnLeft[5,0]
   + HorizontalOverlap[0,5] = 1
 LinkHorizontalOverlap[0,6]: OnLeft[0,6] + OnLeft[6,0]
   + HorizontalOverlap[0,6] = 1
 LinkHorizontalOverlap[0,7]: OnLeft[0,7] + OnLeft[7,0]
   + HorizontalOverlap[0,7] = 1
 LinkHorizontalOverlap[1,0]: OnLeft[0,1] + OnLeft[1,0]
   + HorizontalOverlap[1,0] = 1
 LinkHorizontalOverlap[1,2]: OnLeft[1,2] + OnLeft[2,1]
   + HorizontalOverlap[1,2] = 1
 LinkHorizontalOverlap[1,3]: OnLeft[1,3] + OnLeft[3,1]
   + HorizontalOverlap[1,3] = 1
 LinkHorizontalOverlap[1,4]: OnLeft[1,4] + OnLeft[4,1]
   + HorizontalOverlap[1,4] = 1
 LinkHorizontalOverlap[1,5]: OnLeft[1,5] + OnLeft[5,1]
   + HorizontalOverlap[1,5] = 1
 LinkHorizontalOverlap[1,6]: OnLeft[1,6] + OnLeft[6,1]
   + HorizontalOverlap[1,6] = 1
 LinkHorizontalOverlap[1,7]: OnLeft[1,7] + OnLeft[7,1]
   + HorizontalOverlap[1,7] = 1
 LinkHorizontalOverlap[2,0]: OnLeft[0,2] + OnLeft[2,0]
   + HorizontalOverlap[2,0] = 1
 LinkHorizontalOverlap[2,1]: OnLeft[1,2] + OnLeft[2,1]
   + HorizontalOverlap[2,1] = 1
 LinkHorizontalOverlap[2,3]: OnLeft[2,3] + OnLeft[3,2]
   + HorizontalOverlap[2,3] = 1
 LinkHorizontalOverlap[2,4]: OnLeft[2,4] + OnLeft[4,2]
   + HorizontalOverlap[2,4] = 1
 LinkHorizontalOverlap[2,5]: OnLeft[2,5] + OnLeft[5,2]
   + HorizontalOverlap[2,5] = 1
 LinkHorizontalOverlap[2,6]: OnLeft[2,6] + OnLeft[6,2]
   + HorizontalOverlap[2,6] = 1
 LinkHorizontalOverlap[2,7]: OnLeft[2,7] + OnLeft[7,2]
   + HorizontalOverlap[2,7] = 1
 LinkHorizontalOverlap[3,0]: OnLeft[0,3] + OnLeft[3,0]
   + HorizontalOverlap[3,0] = 1
 LinkHorizontalOverlap[3,1]: OnLeft[1,3] + OnLeft[3,1]
   + HorizontalOverlap[3,1] = 1
 LinkHorizontalOverlap[3,2]: OnLeft[2,3] + OnLeft[3,2]
   + HorizontalOverlap[3,2] = 1
 LinkHorizontalOverlap[3,4]: OnLeft[3,4] + OnLeft[4,3]
   + HorizontalOverlap[3,4] = 1
 LinkHorizontalOverlap[3,5]: OnLeft[3,5] + OnLeft[5,3]
   + HorizontalOverlap[3,5] = 1
 LinkHorizontalOverlap[3,6]: OnLeft[3,6] + OnLeft[6,3]
   + HorizontalOverlap[3,6] = 1
 LinkHorizontalOverlap[3,7]: OnLeft[3,7] + OnLeft[7,3]
   + HorizontalOverlap[3,7] = 1
 LinkHorizontalOverlap[4,0]: OnLeft[0,4] + OnLeft[4,0]
   + HorizontalOverlap[4,0] = 1
 LinkHorizontalOverlap[4,1]: OnLeft[1,4] + OnLeft[4,1]
   + HorizontalOverlap[4,1] = 1
 LinkHorizontalOverlap[4,2]: OnLeft[2,4] + OnLeft[4,2]
   + HorizontalOverlap[4,2] = 1
 LinkHorizontalOverlap[4,3]: OnLeft[3,4] + OnLeft[4,3]
   + HorizontalOverlap[4,3] = 1
 LinkHorizontalOverlap[4,5]: OnLeft[4,5] + OnLeft[5,4]
   + HorizontalOverlap[4,5] = 1
 LinkHorizontalOverlap[4,6]: OnLeft[4,6] + OnLeft[6,4]
   + HorizontalOverlap[4,6] = 1
 LinkHorizontalOverlap[4,7]: OnLeft[4,7] + OnLeft[7,4]
   + HorizontalOverlap[4,7] = 1
 LinkHorizontalOverlap[5,0]: OnLeft[0,5] + OnLeft[5,0]
   + HorizontalOverlap[5,0] = 1
 LinkHorizontalOverlap[5,1]: OnLeft[1,5] + OnLeft[5,1]
   + HorizontalOverlap[5,1] = 1
 LinkHorizontalOverlap[5,2]: OnLeft[2,5] + OnLeft[5,2]
   + HorizontalOverlap[5,2] = 1
 LinkHorizontalOverlap[5,3]: OnLeft[3,5] + OnLeft[5,3]
   + HorizontalOverlap[5,3] = 1
 LinkHorizontalOverlap[5,4]: OnLeft[4,5] + OnLeft[5,4]
   + HorizontalOverlap[5,4] = 1
 LinkHorizontalOverlap[5,6]: OnLeft[5,6] + OnLeft[6,5]
   + HorizontalOverlap[5,6] = 1
 LinkHorizontalOverlap[5,7]: OnLeft[5,7] + OnLeft[7,5]
   + HorizontalOverlap[5,7] = 1
 LinkHorizontalOverlap[6,0]: OnLeft[0,6] + OnLeft[6,0]
   + HorizontalOverlap[6,0] = 1
 LinkHorizontalOverlap[6,1]: OnLeft[1,6] + OnLeft[6,1]
   + HorizontalOverlap[6,1] = 1
 LinkHorizontalOverlap[6,2]: OnLeft[2,6] + OnLeft[6,2]
   + HorizontalOverlap[6,2] = 1
 LinkHorizontalOverlap[6,3]: OnLeft[3,6] + OnLeft[6,3]
   + HorizontalOverlap[6,3] = 1
 LinkHorizontalOverlap[6,4]: OnLeft[4,6] + OnLeft[6,4]
   + HorizontalOverlap[6,4] = 1
 LinkHorizontalOverlap[6,5]: OnLeft[5,6] + OnLeft[6,5]
   + HorizontalOverlap[6,5] = 1
 LinkHorizontalOverlap[6,7]: OnLeft[6,7] + OnLeft[7,6]
   + HorizontalOverlap[6,7] = 1
 LinkHorizontalOverlap[7,0]: OnLeft[0,7] + OnLeft[7,0]
   + HorizontalOverlap[7,0] = 1
 LinkHorizontalOverlap[7,1]: OnLeft[1,7] + OnLeft[7,1]
   + HorizontalOverlap[7,1] = 1
 LinkHorizontalOverlap[7,2]: OnLeft[2,7] + OnLeft[7,2]
   + HorizontalOverlap[7,2] = 1
 LinkHorizontalOverlap[7,3]: OnLeft[3,7] + OnLeft[7,3]
   + HorizontalOverlap[7,3] = 1
 LinkHorizontalOverlap[7,4]: OnLeft[4,7] + OnLeft[7,4]
   + HorizontalOverlap[7,4] = 1
 LinkHorizontalOverlap[7,5]: OnLeft[5,7] + OnLeft[7,5]
   + HorizontalOverlap[7,5] = 1
 LinkHorizontalOverlap[7,6]: OnLeft[6,7] + OnLeft[7,6]
   + HorizontalOverlap[7,6] = 1
 LinkVerticalOverlap[0,1]: Above[0,1] + Above[1,0] + VerticalOverlap[0,1]
   = 1
 LinkVerticalOverlap[0,2]: Above[0,2] + Above[2,0] + VerticalOverlap[0,2]
   = 1
 LinkVerticalOverlap[0,3]: Above[0,3] + Above[3,0] + VerticalOverlap[0,3]
   = 1
 LinkVerticalOverlap[0,4]: Above[0,4] + Above[4,0] + VerticalOverlap[0,4]
   = 1
 LinkVerticalOverlap[0,5]: Above[0,5] + Above[5,0] + VerticalOverlap[0,5]
   = 1
 LinkVerticalOverlap[0,6]: Above[0,6] + Above[6,0] + VerticalOverlap[0,6]
   = 1
 LinkVerticalOverlap[0,7]: Above[0,7] + Above[7,0] + VerticalOverlap[0,7]
   = 1
 LinkVerticalOverlap[1,0]: Above[0,1] + Above[1,0] + VerticalOverlap[1,0]
   = 1
 LinkVerticalOverlap[1,2]: Above[1,2] + Above[2,1] + VerticalOverlap[1,2]
   = 1
 LinkVerticalOverlap[1,3]: Above[1,3] + Above[3,1] + VerticalOverlap[1,3]
   = 1
 LinkVerticalOverlap[1,4]: Above[1,4] + Above[4,1] + VerticalOverlap[1,4]
   = 1
 LinkVerticalOverlap[1,5]: Above[1,5] + Above[5,1] + VerticalOverlap[1,5]
   = 1
 LinkVerticalOverlap[1,6]: Above[1,6] + Above[6,1] + VerticalOverlap[1,6]
   = 1
 LinkVerticalOverlap[1,7]: Above[1,7] + Above[7,1] + VerticalOverlap[1,7]
   = 1
 LinkVerticalOverlap[2,0]: Above[0,2] + Above[2,0] + VerticalOverlap[2,0]
   = 1
 LinkVerticalOverlap[2,1]: Above[1,2] + Above[2,1] + VerticalOverlap[2,1]
   = 1
 LinkVerticalOverlap[2,3]: Above[2,3] + Above[3,2] + VerticalOverlap[2,3]
   = 1
 LinkVerticalOverlap[2,4]: Above[2,4] + Above[4,2] + VerticalOverlap[2,4]
   = 1
 LinkVerticalOverlap[2,5]: Above[2,5] + Above[5,2] + VerticalOverlap[2,5]
   = 1
 LinkVerticalOverlap[2,6]: Above[2,6] + Above[6,2] + VerticalOverlap[2,6]
   = 1
 LinkVerticalOverlap[2,7]: Above[2,7] + Above[7,2] + VerticalOverlap[2,7]
   = 1
 LinkVerticalOverlap[3,0]: Above[0,3] + Above[3,0] + VerticalOverlap[3,0]
   = 1
 LinkVerticalOverlap[3,1]: Above[1,3] + Above[3,1] + VerticalOverlap[3,1]
   = 1
 LinkVerticalOverlap[3,2]: Above[2,3] + Above[3,2] + VerticalOverlap[3,2]
   = 1
 LinkVerticalOverlap[3,4]: Above[3,4] + Above[4,3] + VerticalOverlap[3,4]
   = 1
 LinkVerticalOverlap[3,5]: Above[3,5] + Above[5,3] + VerticalOverlap[3,5]
   = 1
 LinkVerticalOverlap[3,6]: Above[3,6] + Above[6,3] + VerticalOverlap[3,6]
   = 1
 LinkVerticalOverlap[3,7]: Above[3,7] + Above[7,3] + VerticalOverlap[3,7]
   = 1
 LinkVerticalOverlap[4,0]: Above[0,4] + Above[4,0] + VerticalOverlap[4,0]
   = 1
 LinkVerticalOverlap[4,1]: Above[1,4] + Above[4,1] + VerticalOverlap[4,1]
   = 1
 LinkVerticalOverlap[4,2]: Above[2,4] + Above[4,2] + VerticalOverlap[4,2]
   = 1
 LinkVerticalOverlap[4,3]: Above[3,4] + Above[4,3] + VerticalOverlap[4,3]
   = 1
 LinkVerticalOverlap[4,5]: Above[4,5] + Above[5,4] + VerticalOverlap[4,5]
   = 1
 LinkVerticalOverlap[4,6]: Above[4,6] + Above[6,4] + VerticalOverlap[4,6]
   = 1
 LinkVerticalOverlap[4,7]: Above[4,7] + Above[7,4] + VerticalOverlap[4,7]
   = 1
 LinkVerticalOverlap[5,0]: Above[0,5] + Above[5,0] + VerticalOverlap[5,0]
   = 1
 LinkVerticalOverlap[5,1]: Above[1,5] + Above[5,1] + VerticalOverlap[5,1]
   = 1
 LinkVerticalOverlap[5,2]: Above[2,5] + Above[5,2] + VerticalOverlap[5,2]
   = 1
 LinkVerticalOverlap[5,3]: Above[3,5] + Above[5,3] + VerticalOverlap[5,3]
   = 1
 LinkVerticalOverlap[5,4]: Above[4,5] + Above[5,4] + VerticalOverlap[5,4]
   = 1
 LinkVerticalOverlap[5,6]: Above[5,6] + Above[6,5] + VerticalOverlap[5,6]
   = 1
 LinkVerticalOverlap[5,7]: Above[5,7] + Above[7,5] + VerticalOverlap[5,7]
   = 1
 LinkVerticalOverlap[6,0]: Above[0,6] + Above[6,0] + VerticalOverlap[6,0]
   = 1
 LinkVerticalOverlap[6,1]: Above[1,6] + Above[6,1] + VerticalOverlap[6,1]
   = 1
 LinkVerticalOverlap[6,2]: Above[2,6] + Above[6,2] + VerticalOverlap[6,2]
   = 1
 LinkVerticalOverlap[6,3]: Above[3,6] + Above[6,3] + VerticalOverlap[6,3]
   = 1
 LinkVerticalOverlap[6,4]: Above[4,6] + Above[6,4] + VerticalOverlap[6,4]
   = 1
 LinkVerticalOverlap[6,5]: Above[5,6] + Above[6,5] + VerticalOverlap[6,5]
   = 1
 LinkVerticalOverlap[6,7]: Above[6,7] + Above[7,6] + VerticalOverlap[6,7]
   = 1
 LinkVerticalOverlap[7,0]: Above[0,7] + Above[7,0] + VerticalOverlap[7,0]
   = 1
 LinkVerticalOverlap[7,1]: Above[1,7] + Above[7,1] + VerticalOverlap[7,1]
   = 1
 LinkVerticalOverlap[7,2]: Above[2,7] + Above[7,2] + VerticalOverlap[7,2]
   = 1
 LinkVerticalOverlap[7,3]: Above[3,7] + Above[7,3] + VerticalOverlap[7,3]
   = 1
 LinkVerticalOverlap[7,4]: Above[4,7] + Above[7,4] + VerticalOverlap[7,4]
   = 1
 LinkVerticalOverlap[7,5]: Above[5,7] + Above[7,5] + VerticalOverlap[7,5]
   = 1
 LinkVerticalOverlap[7,6]: Above[6,7] + Above[7,6] + VerticalOverlap[7,6]
   = 1
 Linkx0Diff[0,0]: EdgeDistance[x0,0,0] = 0
 Linkx0Diff[0,1]: - X0[0] + X0[1] + EdgeDistance[x0,0,1] = 0
 Linkx0Diff[0,2]: - X0[0] + X0[2] + EdgeDistance[x0,0,2] = 0
 Linkx0Diff[0,3]: - X0[0] + X0[3] + EdgeDistance[x0,0,3] = 0
 Linkx0Diff[0,4]: - X0[0] + X0[4] + EdgeDistance[x0,0,4] = 0
 Linkx0Diff[0,5]: - X0[0] + X0[5] + EdgeDistance[x0,0,5] = 0
 Linkx0Diff[0,6]: - X0[0] + X0[6] + EdgeDistance[x0,0,6] = 0
 Linkx0Diff[0,7]: - X0[0] + X0[7] + EdgeDistance[x0,0,7] = 0
 Linkx0Diff[1,0]: X0[0] - X0[1] + EdgeDistance[x0,1,0] = 0
 Linkx0Diff[1,1]: EdgeDistance[x0,1,1] = 0
 Linkx0Diff[1,2]: - X0[1] + X0[2] + EdgeDistance[x0,1,2] = 0
 Linkx0Diff[1,3]: - X0[1] + X0[3] + EdgeDistance[x0,1,3] = 0
 Linkx0Diff[1,4]: - X0[1] + X0[4] + EdgeDistance[x0,1,4] = 0
 Linkx0Diff[1,5]: - X0[1] + X0[5] + EdgeDistance[x0,1,5] = 0
 Linkx0Diff[1,6]: - X0[1] + X0[6] + EdgeDistance[x0,1,6] = 0
 Linkx0Diff[1,7]: - X0[1] + X0[7] + EdgeDistance[x0,1,7] = 0
 Linkx0Diff[2,0]: X0[0] - X0[2] + EdgeDistance[x0,2,0] = 0
 Linkx0Diff[2,1]: X0[1] - X0[2] + EdgeDistance[x0,2,1] = 0
 Linkx0Diff[2,2]: EdgeDistance[x0,2,2] = 0
 Linkx0Diff[2,3]: - X0[2] + X0[3] + EdgeDistance[x0,2,3] = 0
 Linkx0Diff[2,4]: - X0[2] + X0[4] + EdgeDistance[x0,2,4] = 0
 Linkx0Diff[2,5]: - X0[2] + X0[5] + EdgeDistance[x0,2,5] = 0
 Linkx0Diff[2,6]: - X0[2] + X0[6] + EdgeDistance[x0,2,6] = 0
 Linkx0Diff[2,7]: - X0[2] + X0[7] + EdgeDistance[x0,2,7] = 0
 Linkx0Diff[3,0]: X0[0] - X0[3] + EdgeDistance[x0,3,0] = 0
 Linkx0Diff[3,1]: X0[1] - X0[3] + EdgeDistance[x0,3,1] = 0
 Linkx0Diff[3,2]: X0[2] - X0[3] + EdgeDistance[x0,3,2] = 0
 Linkx0Diff[3,3]: EdgeDistance[x0,3,3] = 0
 Linkx0Diff[3,4]: - X0[3] + X0[4] + EdgeDistance[x0,3,4] = 0
 Linkx0Diff[3,5]: - X0[3] + X0[5] + EdgeDistance[x0,3,5] = 0
 Linkx0Diff[3,6]: - X0[3] + X0[6] + EdgeDistance[x0,3,6] = 0
 Linkx0Diff[3,7]: - X0[3] + X0[7] + EdgeDistance[x0,3,7] = 0
 Linkx0Diff[4,0]: X0[0] - X0[4] + EdgeDistance[x0,4,0] = 0
 Linkx0Diff[4,1]: X0[1] - X0[4] + EdgeDistance[x0,4,1] = 0
 Linkx0Diff[4,2]: X0[2] - X0[4] + EdgeDistance[x0,4,2] = 0
 Linkx0Diff[4,3]: X0[3] - X0[4] + EdgeDistance[x0,4,3] = 0
 Linkx0Diff[4,4]: EdgeDistance[x0,4,4] = 0
 Linkx0Diff[4,5]: - X0[4] + X0[5] + EdgeDistance[x0,4,5] = 0
 Linkx0Diff[4,6]: - X0[4] + X0[6] + EdgeDistance[x0,4,6] = 0
 Linkx0Diff[4,7]: - X0[4] + X0[7] + EdgeDistance[x0,4,7] = 0
 Linkx0Diff[5,0]: X0[0] - X0[5] + EdgeDistance[x0,5,0] = 0
 Linkx0Diff[5,1]: X0[1] - X0[5] + EdgeDistance[x0,5,1] = 0
 Linkx0Diff[5,2]: X0[2] - X0[5] + EdgeDistance[x0,5,2] = 0
 Linkx0Diff[5,3]: X0[3] - X0[5] + EdgeDistance[x0,5,3] = 0
 Linkx0Diff[5,4]: X0[4] - X0[5] + EdgeDistance[x0,5,4] = 0
 Linkx0Diff[5,5]: EdgeDistance[x0,5,5] = 0
 Linkx0Diff[5,6]: - X0[5] + X0[6] + EdgeDistance[x0,5,6] = 0
 Linkx0Diff[5,7]: - X0[5] + X0[7] + EdgeDistance[x0,5,7] = 0
 Linkx0Diff[6,0]: X0[0] - X0[6] + EdgeDistance[x0,6,0] = 0
 Linkx0Diff[6,1]: X0[1] - X0[6] + EdgeDistance[x0,6,1] = 0
 Linkx0Diff[6,2]: X0[2] - X0[6] + EdgeDistance[x0,6,2] = 0
 Linkx0Diff[6,3]: X0[3] - X0[6] + EdgeDistance[x0,6,3] = 0
 Linkx0Diff[6,4]: X0[4] - X0[6] + EdgeDistance[x0,6,4] = 0
 Linkx0Diff[6,5]: X0[5] - X0[6] + EdgeDistance[x0,6,5] = 0
 Linkx0Diff[6,6]: EdgeDistance[x0,6,6] = 0
 Linkx0Diff[6,7]: - X0[6] + X0[7] + EdgeDistance[x0,6,7] = 0
 Linkx0Diff[7,0]: X0[0] - X0[7] + EdgeDistance[x0,7,0] = 0
 Linkx0Diff[7,1]: X0[1] - X0[7] + EdgeDistance[x0,7,1] = 0
 Linkx0Diff[7,2]: X0[2] - X0[7] + EdgeDistance[x0,7,2] = 0
 Linkx0Diff[7,3]: X0[3] - X0[7] + EdgeDistance[x0,7,3] = 0
 Linkx0Diff[7,4]: X0[4] - X0[7] + EdgeDistance[x0,7,4] = 0
 Linkx0Diff[7,5]: X0[5] - X0[7] + EdgeDistance[x0,7,5] = 0
 Linkx0Diff[7,6]: X0[6] - X0[7] + EdgeDistance[x0,7,6] = 0
 Linkx0Diff[7,7]: EdgeDistance[x0,7,7] = 0
 Linky0Diff[0,0]: EdgeDistance[y0,0,0] = 0
 Linky0Diff[0,1]: - Y0[0] + Y0[1] + EdgeDistance[y0,0,1] = 0
 Linky0Diff[0,2]: - Y0[0] + Y0[2] + EdgeDistance[y0,0,2] = 0
 Linky0Diff[0,3]: - Y0[0] + Y0[3] + EdgeDistance[y0,0,3] = 0
 Linky0Diff[0,4]: - Y0[0] + Y0[4] + EdgeDistance[y0,0,4] = 0
 Linky0Diff[0,5]: - Y0[0] + Y0[5] + EdgeDistance[y0,0,5] = 0
 Linky0Diff[0,6]: - Y0[0] + Y0[6] + EdgeDistance[y0,0,6] = 0
 Linky0Diff[0,7]: - Y0[0] + Y0[7] + EdgeDistance[y0,0,7] = 0
 Linky0Diff[1,0]: Y0[0] - Y0[1] + EdgeDistance[y0,1,0] = 0
 Linky0Diff[1,1]: EdgeDistance[y0,1,1] = 0
 Linky0Diff[1,2]: - Y0[1] + Y0[2] + EdgeDistance[y0,1,2] = 0
 Linky0Diff[1,3]: - Y0[1] + Y0[3] + EdgeDistance[y0,1,3] = 0
 Linky0Diff[1,4]: - Y0[1] + Y0[4] + EdgeDistance[y0,1,4] = 0
 Linky0Diff[1,5]: - Y0[1] + Y0[5] + EdgeDistance[y0,1,5] = 0
 Linky0Diff[1,6]: - Y0[1] + Y0[6] + EdgeDistance[y0,1,6] = 0
 Linky0Diff[1,7]: - Y0[1] + Y0[7] + EdgeDistance[y0,1,7] = 0
 Linky0Diff[2,0]: Y0[0] - Y0[2] + EdgeDistance[y0,2,0] = 0
 Linky0Diff[2,1]: Y0[1] - Y0[2] + EdgeDistance[y0,2,1] = 0
 Linky0Diff[2,2]: EdgeDistance[y0,2,2] = 0
 Linky0Diff[2,3]: - Y0[2] + Y0[3] + EdgeDistance[y0,2,3] = 0
 Linky0Diff[2,4]: - Y0[2] + Y0[4] + EdgeDistance[y0,2,4] = 0
 Linky0Diff[2,5]: - Y0[2] + Y0[5] + EdgeDistance[y0,2,5] = 0
 Linky0Diff[2,6]: - Y0[2] + Y0[6] + EdgeDistance[y0,2,6] = 0
 Linky0Diff[2,7]: - Y0[2] + Y0[7] + EdgeDistance[y0,2,7] = 0
 Linky0Diff[3,0]: Y0[0] - Y0[3] + EdgeDistance[y0,3,0] = 0
 Linky0Diff[3,1]: Y0[1] - Y0[3] + EdgeDistance[y0,3,1] = 0
 Linky0Diff[3,2]: Y0[2] - Y0[3] + EdgeDistance[y0,3,2] = 0
 Linky0Diff[3,3]: EdgeDistance[y0,3,3] = 0
 Linky0Diff[3,4]: - Y0[3] + Y0[4] + EdgeDistance[y0,3,4] = 0
 Linky0Diff[3,5]: - Y0[3] + Y0[5] + EdgeDistance[y0,3,5] = 0
 Linky0Diff[3,6]: - Y0[3] + Y0[6] + EdgeDistance[y0,3,6] = 0
 Linky0Diff[3,7]: - Y0[3] + Y0[7] + EdgeDistance[y0,3,7] = 0
 Linky0Diff[4,0]: Y0[0] - Y0[4] + EdgeDistance[y0,4,0] = 0
 Linky0Diff[4,1]: Y0[1] - Y0[4] + EdgeDistance[y0,4,1] = 0
 Linky0Diff[4,2]: Y0[2] - Y0[4] + EdgeDistance[y0,4,2] = 0
 Linky0Diff[4,3]: Y0[3] - Y0[4] + EdgeDistance[y0,4,3] = 0
 Linky0Diff[4,4]: EdgeDistance[y0,4,4] = 0
 Linky0Diff[4,5]: - Y0[4] + Y0[5] + EdgeDistance[y0,4,5] = 0
 Linky0Diff[4,6]: - Y0[4] + Y0[6] + EdgeDistance[y0,4,6] = 0
 Linky0Diff[4,7]: - Y0[4] + Y0[7] + EdgeDistance[y0,4,7] = 0
 Linky0Diff[5,0]: Y0[0] - Y0[5] + EdgeDistance[y0,5,0] = 0
 Linky0Diff[5,1]: Y0[1] - Y0[5] + EdgeDistance[y0,5,1] = 0
 Linky0Diff[5,2]: Y0[2] - Y0[5] + EdgeDistance[y0,5,2] = 0
 Linky0Diff[5,3]: Y0[3] - Y0[5] + EdgeDistance[y0,5,3] = 0
 Linky0Diff[5,4]: Y0[4] - Y0[5] + EdgeDistance[y0,5,4] = 0
 Linky0Diff[5,5]: EdgeDistance[y0,5,5] = 0
 Linky0Diff[5,6]: - Y0[5] + Y0[6] + EdgeDistance[y0,5,6] = 0
 Linky0Diff[5,7]: - Y0[5] + Y0[7] + EdgeDistance[y0,5,7] = 0
 Linky0Diff[6,0]: Y0[0] - Y0[6] + EdgeDistance[y0,6,0] = 0
 Linky0Diff[6,1]: Y0[1] - Y0[6] + EdgeDistance[y0,6,1] = 0
 Linky0Diff[6,2]: Y0[2] - Y0[6] + EdgeDistance[y0,6,2] = 0
 Linky0Diff[6,3]: Y0[3] - Y0[6] + EdgeDistance[y0,6,3] = 0
 Linky0Diff[6,4]: Y0[4] - Y0[6] + EdgeDistance[y0,6,4] = 0
 Linky0Diff[6,5]: Y0[5] - Y0[6] + EdgeDistance[y0,6,5] = 0
 Linky0Diff[6,6]: EdgeDistance[y0,6,6] = 0
 Linky0Diff[6,7]: - Y0[6] + Y0[7] + EdgeDistance[y0,6,7] = 0
 Linky0Diff[7,0]: Y0[0] - Y0[7] + EdgeDistance[y0,7,0] = 0
 Linky0Diff[7,1]: Y0[1] - Y0[7] + EdgeDistance[y0,7,1] = 0
 Linky0Diff[7,2]: Y0[2] - Y0[7] + EdgeDistance[y0,7,2] = 0
 Linky0Diff[7,3]: Y0[3] - Y0[7] + EdgeDistance[y0,7,3] = 0
 Linky0Diff[7,4]: Y0[4] - Y0[7] + EdgeDistance[y0,7,4] = 0
 Linky0Diff[7,5]: Y0[5] - Y0[7] + EdgeDistance[y0,7,5] = 0
 Linky0Diff[7,6]: Y0[6] - Y0[7] + EdgeDistance[y0,7,6] = 0
 Linky0Diff[7,7]: EdgeDistance[y0,7,7] = 0
 Linkx1Diff[0,0]: EdgeDistance[x1,0,0] = 0
 Linkx1Diff[0,1]: - X1[0] + X1[1] + EdgeDistance[x1,0,1] = 0
 Linkx1Diff[0,2]: - X1[0] + X1[2] + EdgeDistance[x1,0,2] = 0
 Linkx1Diff[0,3]: - X1[0] + X1[3] + EdgeDistance[x1,0,3] = 0
 Linkx1Diff[0,4]: - X1[0] + X1[4] + EdgeDistance[x1,0,4] = 0
 Linkx1Diff[0,5]: - X1[0] + X1[5] + EdgeDistance[x1,0,5] = 0
 Linkx1Diff[0,6]: - X1[0] + X1[6] + EdgeDistance[x1,0,6] = 0
 Linkx1Diff[0,7]: - X1[0] + X1[7] + EdgeDistance[x1,0,7] = 0
 Linkx1Diff[1,0]: X1[0] - X1[1] + EdgeDistance[x1,1,0] = 0
 Linkx1Diff[1,1]: EdgeDistance[x1,1,1] = 0
 Linkx1Diff[1,2]: - X1[1] + X1[2] + EdgeDistance[x1,1,2] = 0
 Linkx1Diff[1,3]: - X1[1] + X1[3] + EdgeDistance[x1,1,3] = 0
 Linkx1Diff[1,4]: - X1[1] + X1[4] + EdgeDistance[x1,1,4] = 0
 Linkx1Diff[1,5]: - X1[1] + X1[5] + EdgeDistance[x1,1,5] = 0
 Linkx1Diff[1,6]: - X1[1] + X1[6] + EdgeDistance[x1,1,6] = 0
 Linkx1Diff[1,7]: - X1[1] + X1[7] + EdgeDistance[x1,1,7] = 0
 Linkx1Diff[2,0]: X1[0] - X1[2] + EdgeDistance[x1,2,0] = 0
 Linkx1Diff[2,1]: X1[1] - X1[2] + EdgeDistance[x1,2,1] = 0
 Linkx1Diff[2,2]: EdgeDistance[x1,2,2] = 0
 Linkx1Diff[2,3]: - X1[2] + X1[3] + EdgeDistance[x1,2,3] = 0
 Linkx1Diff[2,4]: - X1[2] + X1[4] + EdgeDistance[x1,2,4] = 0
 Linkx1Diff[2,5]: - X1[2] + X1[5] + EdgeDistance[x1,2,5] = 0
 Linkx1Diff[2,6]: - X1[2] + X1[6] + EdgeDistance[x1,2,6] = 0
 Linkx1Diff[2,7]: - X1[2] + X1[7] + EdgeDistance[x1,2,7] = 0
 Linkx1Diff[3,0]: X1[0] - X1[3] + EdgeDistance[x1,3,0] = 0
 Linkx1Diff[3,1]: X1[1] - X1[3] + EdgeDistance[x1,3,1] = 0
 Linkx1Diff[3,2]: X1[2] - X1[3] + EdgeDistance[x1,3,2] = 0
 Linkx1Diff[3,3]: EdgeDistance[x1,3,3] = 0
 Linkx1Diff[3,4]: - X1[3] + X1[4] + EdgeDistance[x1,3,4] = 0
 Linkx1Diff[3,5]: - X1[3] + X1[5] + EdgeDistance[x1,3,5] = 0
 Linkx1Diff[3,6]: - X1[3] + X1[6] + EdgeDistance[x1,3,6] = 0
 Linkx1Diff[3,7]: - X1[3] + X1[7] + EdgeDistance[x1,3,7] = 0
 Linkx1Diff[4,0]: X1[0] - X1[4] + EdgeDistance[x1,4,0] = 0
 Linkx1Diff[4,1]: X1[1] - X1[4] + EdgeDistance[x1,4,1] = 0
 Linkx1Diff[4,2]: X1[2] - X1[4] + EdgeDistance[x1,4,2] = 0
 Linkx1Diff[4,3]: X1[3] - X1[4] + EdgeDistance[x1,4,3] = 0
 Linkx1Diff[4,4]: EdgeDistance[x1,4,4] = 0
 Linkx1Diff[4,5]: - X1[4] + X1[5] + EdgeDistance[x1,4,5] = 0
 Linkx1Diff[4,6]: - X1[4] + X1[6] + EdgeDistance[x1,4,6] = 0
 Linkx1Diff[4,7]: - X1[4] + X1[7] + EdgeDistance[x1,4,7] = 0
 Linkx1Diff[5,0]: X1[0] - X1[5] + EdgeDistance[x1,5,0] = 0
 Linkx1Diff[5,1]: X1[1] - X1[5] + EdgeDistance[x1,5,1] = 0
 Linkx1Diff[5,2]: X1[2] - X1[5] + EdgeDistance[x1,5,2] = 0
 Linkx1Diff[5,3]: X1[3] - X1[5] + EdgeDistance[x1,5,3] = 0
 Linkx1Diff[5,4]: X1[4] - X1[5] + EdgeDistance[x1,5,4] = 0
 Linkx1Diff[5,5]: EdgeDistance[x1,5,5] = 0
 Linkx1Diff[5,6]: - X1[5] + X1[6] + EdgeDistance[x1,5,6] = 0
 Linkx1Diff[5,7]: - X1[5] + X1[7] + EdgeDistance[x1,5,7] = 0
 Linkx1Diff[6,0]: X1[0] - X1[6] + EdgeDistance[x1,6,0] = 0
 Linkx1Diff[6,1]: X1[1] - X1[6] + EdgeDistance[x1,6,1] = 0
 Linkx1Diff[6,2]: X1[2] - X1[6] + EdgeDistance[x1,6,2] = 0
 Linkx1Diff[6,3]: X1[3] - X1[6] + EdgeDistance[x1,6,3] = 0
 Linkx1Diff[6,4]: X1[4] - X1[6] + EdgeDistance[x1,6,4] = 0
 Linkx1Diff[6,5]: X1[5] - X1[6] + EdgeDistance[x1,6,5] = 0
 Linkx1Diff[6,6]: EdgeDistance[x1,6,6] = 0
 Linkx1Diff[6,7]: - X1[6] + X1[7] + EdgeDistance[x1,6,7] = 0
 Linkx1Diff[7,0]: X1[0] - X1[7] + EdgeDistance[x1,7,0] = 0
 Linkx1Diff[7,1]: X1[1] - X1[7] + EdgeDistance[x1,7,1] = 0
 Linkx1Diff[7,2]: X1[2] - X1[7] + EdgeDistance[x1,7,2] = 0
 Linkx1Diff[7,3]: X1[3] - X1[7] + EdgeDistance[x1,7,3] = 0
 Linkx1Diff[7,4]: X1[4] - X1[7] + EdgeDistance[x1,7,4] = 0
 Linkx1Diff[7,5]: X1[5] - X1[7] + EdgeDistance[x1,7,5] = 0
 Linkx1Diff[7,6]: X1[6] - X1[7] + EdgeDistance[x1,7,6] = 0
 Linkx1Diff[7,7]: EdgeDistance[x1,7,7] = 0
 Linky1Diff[0,0]: EdgeDistance[y1,0,0] = 0
 Linky1Diff[0,1]: - Y1[0] + Y1[1] + EdgeDistance[y1,0,1] = 0
 Linky1Diff[0,2]: - Y1[0] + Y1[2] + EdgeDistance[y1,0,2] = 0
 Linky1Diff[0,3]: - Y1[0] + Y1[3] + EdgeDistance[y1,0,3] = 0
 Linky1Diff[0,4]: - Y1[0] + Y1[4] + EdgeDistance[y1,0,4] = 0
 Linky1Diff[0,5]: - Y1[0] + Y1[5] + EdgeDistance[y1,0,5] = 0
 Linky1Diff[0,6]: - Y1[0] + Y1[6] + EdgeDistance[y1,0,6] = 0
 Linky1Diff[0,7]: - Y1[0] + Y1[7] + EdgeDistance[y1,0,7] = 0
 Linky1Diff[1,0]: Y1[0] - Y1[1] + EdgeDistance[y1,1,0] = 0
 Linky1Diff[1,1]: EdgeDistance[y1,1,1] = 0
 Linky1Diff[1,2]: - Y1[1] + Y1[2] + EdgeDistance[y1,1,2] = 0
 Linky1Diff[1,3]: - Y1[1] + Y1[3] + EdgeDistance[y1,1,3] = 0
 Linky1Diff[1,4]: - Y1[1] + Y1[4] + EdgeDistance[y1,1,4] = 0
 Linky1Diff[1,5]: - Y1[1] + Y1[5] + EdgeDistance[y1,1,5] = 0
 Linky1Diff[1,6]: - Y1[1] + Y1[6] + EdgeDistance[y1,1,6] = 0
 Linky1Diff[1,7]: - Y1[1] + Y1[7] + EdgeDistance[y1,1,7] = 0
 Linky1Diff[2,0]: Y1[0] - Y1[2] + EdgeDistance[y1,2,0] = 0
 Linky1Diff[2,1]: Y1[1] - Y1[2] + EdgeDistance[y1,2,1] = 0
 Linky1Diff[2,2]: EdgeDistance[y1,2,2] = 0
 Linky1Diff[2,3]: - Y1[2] + Y1[3] + EdgeDistance[y1,2,3] = 0
 Linky1Diff[2,4]: - Y1[2] + Y1[4] + EdgeDistance[y1,2,4] = 0
 Linky1Diff[2,5]: - Y1[2] + Y1[5] + EdgeDistance[y1,2,5] = 0
 Linky1Diff[2,6]: - Y1[2] + Y1[6] + EdgeDistance[y1,2,6] = 0
 Linky1Diff[2,7]: - Y1[2] + Y1[7] + EdgeDistance[y1,2,7] = 0
 Linky1Diff[3,0]: Y1[0] - Y1[3] + EdgeDistance[y1,3,0] = 0
 Linky1Diff[3,1]: Y1[1] - Y1[3] + EdgeDistance[y1,3,1] = 0
 Linky1Diff[3,2]: Y1[2] - Y1[3] + EdgeDistance[y1,3,2] = 0
 Linky1Diff[3,3]: EdgeDistance[y1,3,3] = 0
 Linky1Diff[3,4]: - Y1[3] + Y1[4] + EdgeDistance[y1,3,4] = 0
 Linky1Diff[3,5]: - Y1[3] + Y1[5] + EdgeDistance[y1,3,5] = 0
 Linky1Diff[3,6]: - Y1[3] + Y1[6] + EdgeDistance[y1,3,6] = 0
 Linky1Diff[3,7]: - Y1[3] + Y1[7] + EdgeDistance[y1,3,7] = 0
 Linky1Diff[4,0]: Y1[0] - Y1[4] + EdgeDistance[y1,4,0] = 0
 Linky1Diff[4,1]: Y1[1] - Y1[4] + EdgeDistance[y1,4,1] = 0
 Linky1Diff[4,2]: Y1[2] - Y1[4] + EdgeDistance[y1,4,2] = 0
 Linky1Diff[4,3]: Y1[3] - Y1[4] + EdgeDistance[y1,4,3] = 0
 Linky1Diff[4,4]: EdgeDistance[y1,4,4] = 0
 Linky1Diff[4,5]: - Y1[4] + Y1[5] + EdgeDistance[y1,4,5] = 0
 Linky1Diff[4,6]: - Y1[4] + Y1[6] + EdgeDistance[y1,4,6] = 0
 Linky1Diff[4,7]: - Y1[4] + Y1[7] + EdgeDistance[y1,4,7] = 0
 Linky1Diff[5,0]: Y1[0] - Y1[5] + EdgeDistance[y1,5,0] = 0
 Linky1Diff[5,1]: Y1[1] - Y1[5] + EdgeDistance[y1,5,1] = 0
 Linky1Diff[5,2]: Y1[2] - Y1[5] + EdgeDistance[y1,5,2] = 0
 Linky1Diff[5,3]: Y1[3] - Y1[5] + EdgeDistance[y1,5,3] = 0
 Linky1Diff[5,4]: Y1[4] - Y1[5] + EdgeDistance[y1,5,4] = 0
 Linky1Diff[5,5]: EdgeDistance[y1,5,5] = 0
 Linky1Diff[5,6]: - Y1[5] + Y1[6] + EdgeDistance[y1,5,6] = 0
 Linky1Diff[5,7]: - Y1[5] + Y1[7] + EdgeDistance[y1,5,7] = 0
 Linky1Diff[6,0]: Y1[0] - Y1[6] + EdgeDistance[y1,6,0] = 0
 Linky1Diff[6,1]: Y1[1] - Y1[6] + EdgeDistance[y1,6,1] = 0
 Linky1Diff[6,2]: Y1[2] - Y1[6] + EdgeDistance[y1,6,2] = 0
 Linky1Diff[6,3]: Y1[3] - Y1[6] + EdgeDistance[y1,6,3] = 0
 Linky1Diff[6,4]: Y1[4] - Y1[6] + EdgeDistance[y1,6,4] = 0
 Linky1Diff[6,5]: Y1[5] - Y1[6] + EdgeDistance[y1,6,5] = 0
 Linky1Diff[6,6]: EdgeDistance[y1,6,6] = 0
 Linky1Diff[6,7]: - Y1[6] + Y1[7] + EdgeDistance[y1,6,7] = 0
 Linky1Diff[7,0]: Y1[0] - Y1[7] + EdgeDistance[y1,7,0] = 0
 Linky1Diff[7,1]: Y1[1] - Y1[7] + EdgeDistance[y1,7,1] = 0
 Linky1Diff[7,2]: Y1[2] - Y1[7] + EdgeDistance[y1,7,2] = 0
 Linky1Diff[7,3]: Y1[3] - Y1[7] + EdgeDistance[y1,7,3] = 0
 Linky1Diff[7,4]: Y1[4] - Y1[7] + EdgeDistance[y1,7,4] = 0
 Linky1Diff[7,5]: Y1[5] - Y1[7] + EdgeDistance[y1,7,5] = 0
 Linky1Diff[7,6]: Y1[6] - Y1[7] + EdgeDistance[y1,7,6] = 0
 Linky1Diff[7,7]: EdgeDistance[y1,7,7] = 0
 R560: X0Diff[0,0] = 0
 R561: - X0[0] + X0[1] + X0Diff[0,1] = 0
 R562: - X0[0] + X0[2] + X0Diff[0,2] = 0
 R563: - X0[0] + X0[3] + X0Diff[0,3] = 0
 R564: - X0[0] + X0[4] + X0Diff[0,4] = 0
 R565: - X0[0] + X0[5] + X0Diff[0,5] = 0
 R566: - X0[0] + X0[6] + X0Diff[0,6] = 0
 R567: - X0[0] + X0[7] + X0Diff[0,7] = 0
 R568: X0[0] - X0[1] + X0Diff[1,0] = 0
 R569: X0Diff[1,1] = 0
 R570: - X0[1] + X0[2] + X0Diff[1,2] = 0
 R571: - X0[1] + X0[3] + X0Diff[1,3] = 0
 R572: - X0[1] + X0[4] + X0Diff[1,4] = 0
 R573: - X0[1] + X0[5] + X0Diff[1,5] = 0
 R574: - X0[1] + X0[6] + X0Diff[1,6] = 0
 R575: - X0[1] + X0[7] + X0Diff[1,7] = 0
 R576: X0[0] - X0[2] + X0Diff[2,0] = 0
 R577: X0[1] - X0[2] + X0Diff[2,1] = 0
 R578: X0Diff[2,2] = 0
 R579: - X0[2] + X0[3] + X0Diff[2,3] = 0
 R580: - X0[2] + X0[4] + X0Diff[2,4] = 0
 R581: - X0[2] + X0[5] + X0Diff[2,5] = 0
 R582: - X0[2] + X0[6] + X0Diff[2,6] = 0
 R583: - X0[2] + X0[7] + X0Diff[2,7] = 0
 R584: X0[0] - X0[3] + X0Diff[3,0] = 0
 R585: X0[1] - X0[3] + X0Diff[3,1] = 0
 R586: X0[2] - X0[3] + X0Diff[3,2] = 0
 R587: X0Diff[3,3] = 0
 R588: - X0[3] + X0[4] + X0Diff[3,4] = 0
 R589: - X0[3] + X0[5] + X0Diff[3,5] = 0
 R590: - X0[3] + X0[6] + X0Diff[3,6] = 0
 R591: - X0[3] + X0[7] + X0Diff[3,7] = 0
 R592: X0[0] - X0[4] + X0Diff[4,0] = 0
 R593: X0[1] - X0[4] + X0Diff[4,1] = 0
 R594: X0[2] - X0[4] + X0Diff[4,2] = 0
 R595: X0[3] - X0[4] + X0Diff[4,3] = 0
 R596: X0Diff[4,4] = 0
 R597: - X0[4] + X0[5] + X0Diff[4,5] = 0
 R598: - X0[4] + X0[6] + X0Diff[4,6] = 0
 R599: - X0[4] + X0[7] + X0Diff[4,7] = 0
 R600: X0[0] - X0[5] + X0Diff[5,0] = 0
 R601: X0[1] - X0[5] + X0Diff[5,1] = 0
 R602: X0[2] - X0[5] + X0Diff[5,2] = 0
 R603: X0[3] - X0[5] + X0Diff[5,3] = 0
 R604: X0[4] - X0[5] + X0Diff[5,4] = 0
 R605: X0Diff[5,5] = 0
 R606: - X0[5] + X0[6] + X0Diff[5,6] = 0
 R607: - X0[5] + X0[7] + X0Diff[5,7] = 0
 R608: X0[0] - X0[6] + X0Diff[6,0] = 0
 R609: X0[1] - X0[6] + X0Diff[6,1] = 0
 R610: X0[2] - X0[6] + X0Diff[6,2] = 0
 R611: X0[3] - X0[6] + X0Diff[6,3] = 0
 R612: X0[4] - X0[6] + X0Diff[6,4] = 0
 R613: X0[5] - X0[6] + X0Diff[6,5] = 0
 R614: X0Diff[6,6] = 0
 R615: - X0[6] + X0[7] + X0Diff[6,7] = 0
 R616: X0[0] - X0[7] + X0Diff[7,0] = 0
 R617: X0[1] - X0[7] + X0Diff[7,1] = 0
 R618: X0[2] - X0[7] + X0Diff[7,2] = 0
 R619: X0[3] - X0[7] + X0Diff[7,3] = 0
 R620: X0[4] - X0[7] + X0Diff[7,4] = 0
 R621: X0[5] - X0[7] + X0Diff[7,5] = 0
 R622: X0[6] - X0[7] + X0Diff[7,6] = 0
 R623: X0Diff[7,7] = 0
 R624: Y0Diff[0,0] = 0
 R625: - Y0[0] + Y0[1] + Y0Diff[0,1] = 0
 R626: - Y0[0] + Y0[2] + Y0Diff[0,2] = 0
 R627: - Y0[0] + Y0[3] + Y0Diff[0,3] = 0
 R628: - Y0[0] + Y0[4] + Y0Diff[0,4] = 0
 R629: - Y0[0] + Y0[5] + Y0Diff[0,5] = 0
 R630: - Y0[0] + Y0[6] + Y0Diff[0,6] = 0
 R631: - Y0[0] + Y0[7] + Y0Diff[0,7] = 0
 R632: Y0[0] - Y0[1] + Y0Diff[1,0] = 0
 R633: Y0Diff[1,1] = 0
 R634: - Y0[1] + Y0[2] + Y0Diff[1,2] = 0
 R635: - Y0[1] + Y0[3] + Y0Diff[1,3] = 0
 R636: - Y0[1] + Y0[4] + Y0Diff[1,4] = 0
 R637: - Y0[1] + Y0[5] + Y0Diff[1,5] = 0
 R638: - Y0[1] + Y0[6] + Y0Diff[1,6] = 0
 R639: - Y0[1] + Y0[7] + Y0Diff[1,7] = 0
 R640: Y0[0] - Y0[2] + Y0Diff[2,0] = 0
 R641: Y0[1] - Y0[2] + Y0Diff[2,1] = 0
 R642: Y0Diff[2,2] = 0
 R643: - Y0[2] + Y0[3] + Y0Diff[2,3] = 0
 R644: - Y0[2] + Y0[4] + Y0Diff[2,4] = 0
 R645: - Y0[2] + Y0[5] + Y0Diff[2,5] = 0
 R646: - Y0[2] + Y0[6] + Y0Diff[2,6] = 0
 R647: - Y0[2] + Y0[7] + Y0Diff[2,7] = 0
 R648: Y0[0] - Y0[3] + Y0Diff[3,0] = 0
 R649: Y0[1] - Y0[3] + Y0Diff[3,1] = 0
 R650: Y0[2] - Y0[3] + Y0Diff[3,2] = 0
 R651: Y0Diff[3,3] = 0
 R652: - Y0[3] + Y0[4] + Y0Diff[3,4] = 0
 R653: - Y0[3] + Y0[5] + Y0Diff[3,5] = 0
 R654: - Y0[3] + Y0[6] + Y0Diff[3,6] = 0
 R655: - Y0[3] + Y0[7] + Y0Diff[3,7] = 0
 R656: Y0[0] - Y0[4] + Y0Diff[4,0] = 0
 R657: Y0[1] - Y0[4] + Y0Diff[4,1] = 0
 R658: Y0[2] - Y0[4] + Y0Diff[4,2] = 0
 R659: Y0[3] - Y0[4] + Y0Diff[4,3] = 0
 R660: Y0Diff[4,4] = 0
 R661: - Y0[4] + Y0[5] + Y0Diff[4,5] = 0
 R662: - Y0[4] + Y0[6] + Y0Diff[4,6] = 0
 R663: - Y0[4] + Y0[7] + Y0Diff[4,7] = 0
 R664: Y0[0] - Y0[5] + Y0Diff[5,0] = 0
 R665: Y0[1] - Y0[5] + Y0Diff[5,1] = 0
 R666: Y0[2] - Y0[5] + Y0Diff[5,2] = 0
 R667: Y0[3] - Y0[5] + Y0Diff[5,3] = 0
 R668: Y0[4] - Y0[5] + Y0Diff[5,4] = 0
 R669: Y0Diff[5,5] = 0
 R670: - Y0[5] + Y0[6] + Y0Diff[5,6] = 0
 R671: - Y0[5] + Y0[7] + Y0Diff[5,7] = 0
 R672: Y0[0] - Y0[6] + Y0Diff[6,0] = 0
 R673: Y0[1] - Y0[6] + Y0Diff[6,1] = 0
 R674: Y0[2] - Y0[6] + Y0Diff[6,2] = 0
 R675: Y0[3] - Y0[6] + Y0Diff[6,3] = 0
 R676: Y0[4] - Y0[6] + Y0Diff[6,4] = 0
 R677: Y0[5] - Y0[6] + Y0Diff[6,5] = 0
 R678: Y0Diff[6,6] = 0
 R679: - Y0[6] + Y0[7] + Y0Diff[6,7] = 0
 R680: Y0[0] - Y0[7] + Y0Diff[7,0] = 0
 R681: Y0[1] - Y0[7] + Y0Diff[7,1] = 0
 R682: Y0[2] - Y0[7] + Y0Diff[7,2] = 0
 R683: Y0[3] - Y0[7] + Y0Diff[7,3] = 0
 R684: Y0[4] - Y0[7] + Y0Diff[7,4] = 0
 R685: Y0[5] - Y0[7] + Y0Diff[7,5] = 0
 R686: Y0[6] - Y0[7] + Y0Diff[7,6] = 0
 R687: Y0Diff[7,7] = 0
 R688: X1Diff[0,0] = 0
 R689: - X1[0] + X1[1] + X1Diff[0,1] = 0
 R690: - X1[0] + X1[2] + X1Diff[0,2] = 0
 R691: - X1[0] + X1[3] + X1Diff[0,3] = 0
 R692: - X1[0] + X1[4] + X1Diff[0,4] = 0
 R693: - X1[0] + X1[5] + X1Diff[0,5] = 0
 R694: - X1[0] + X1[6] + X1Diff[0,6] = 0
 R695: - X1[0] + X1[7] + X1Diff[0,7] = 0
 R696: X1[0] - X1[1] + X1Diff[1,0] = 0
 R697: X1Diff[1,1] = 0
 R698: - X1[1] + X1[2] + X1Diff[1,2] = 0
 R699: - X1[1] + X1[3] + X1Diff[1,3] = 0
 R700: - X1[1] + X1[4] + X1Diff[1,4] = 0
 R701: - X1[1] + X1[5] + X1Diff[1,5] = 0
 R702: - X1[1] + X1[6] + X1Diff[1,6] = 0
 R703: - X1[1] + X1[7] + X1Diff[1,7] = 0
 R704: X1[0] - X1[2] + X1Diff[2,0] = 0
 R705: X1[1] - X1[2] + X1Diff[2,1] = 0
 R706: X1Diff[2,2] = 0
 R707: - X1[2] + X1[3] + X1Diff[2,3] = 0
 R708: - X1[2] + X1[4] + X1Diff[2,4] = 0
 R709: - X1[2] + X1[5] + X1Diff[2,5] = 0
 R710: - X1[2] + X1[6] + X1Diff[2,6] = 0
 R711: - X1[2] + X1[7] + X1Diff[2,7] = 0
 R712: X1[0] - X1[3] + X1Diff[3,0] = 0
 R713: X1[1] - X1[3] + X1Diff[3,1] = 0
 R714: X1[2] - X1[3] + X1Diff[3,2] = 0
 R715: X1Diff[3,3] = 0
 R716: - X1[3] + X1[4] + X1Diff[3,4] = 0
 R717: - X1[3] + X1[5] + X1Diff[3,5] = 0
 R718: - X1[3] + X1[6] + X1Diff[3,6] = 0
 R719: - X1[3] + X1[7] + X1Diff[3,7] = 0
 R720: X1[0] - X1[4] + X1Diff[4,0] = 0
 R721: X1[1] - X1[4] + X1Diff[4,1] = 0
 R722: X1[2] - X1[4] + X1Diff[4,2] = 0
 R723: X1[3] - X1[4] + X1Diff[4,3] = 0
 R724: X1Diff[4,4] = 0
 R725: - X1[4] + X1[5] + X1Diff[4,5] = 0
 R726: - X1[4] + X1[6] + X1Diff[4,6] = 0
 R727: - X1[4] + X1[7] + X1Diff[4,7] = 0
 R728: X1[0] - X1[5] + X1Diff[5,0] = 0
 R729: X1[1] - X1[5] + X1Diff[5,1] = 0
 R730: X1[2] - X1[5] + X1Diff[5,2] = 0
 R731: X1[3] - X1[5] + X1Diff[5,3] = 0
 R732: X1[4] - X1[5] + X1Diff[5,4] = 0
 R733: X1Diff[5,5] = 0
 R734: - X1[5] + X1[6] + X1Diff[5,6] = 0
 R735: - X1[5] + X1[7] + X1Diff[5,7] = 0
 R736: X1[0] - X1[6] + X1Diff[6,0] = 0
 R737: X1[1] - X1[6] + X1Diff[6,1] = 0
 R738: X1[2] - X1[6] + X1Diff[6,2] = 0
 R739: X1[3] - X1[6] + X1Diff[6,3] = 0
 R740: X1[4] - X1[6] + X1Diff[6,4] = 0
 R741: X1[5] - X1[6] + X1Diff[6,5] = 0
 R742: X1Diff[6,6] = 0
 R743: - X1[6] + X1[7] + X1Diff[6,7] = 0
 R744: X1[0] - X1[7] + X1Diff[7,0] = 0
 R745: X1[1] - X1[7] + X1Diff[7,1] = 0
 R746: X1[2] - X1[7] + X1Diff[7,2] = 0
 R747: X1[3] - X1[7] + X1Diff[7,3] = 0
 R748: X1[4] - X1[7] + X1Diff[7,4] = 0
 R749: X1[5] - X1[7] + X1Diff[7,5] = 0
 R750: X1[6] - X1[7] + X1Diff[7,6] = 0
 R751: X1Diff[7,7] = 0
 R752: Y1Diff[0,0] = 0
 R753: - Y1[0] + Y1[1] + Y1Diff[0,1] = 0
 R754: - Y1[0] + Y1[2] + Y1Diff[0,2] = 0
 R755: - Y1[0] + Y1[3] + Y1Diff[0,3] = 0
 R756: - Y1[0] + Y1[4] + Y1Diff[0,4] = 0
 R757: - Y1[0] + Y1[5] + Y1Diff[0,5] = 0
 R758: - Y1[0] + Y1[6] + Y1Diff[0,6] = 0
 R759: - Y1[0] + Y1[7] + Y1Diff[0,7] = 0
 R760: Y1[0] - Y1[1] + Y1Diff[1,0] = 0
 R761: Y1Diff[1,1] = 0
 R762: - Y1[1] + Y1[2] + Y1Diff[1,2] = 0
 R763: - Y1[1] + Y1[3] + Y1Diff[1,3] = 0
 R764: - Y1[1] + Y1[4] + Y1Diff[1,4] = 0
 R765: - Y1[1] + Y1[5] + Y1Diff[1,5] = 0
 R766: - Y1[1] + Y1[6] + Y1Diff[1,6] = 0
 R767: - Y1[1] + Y1[7] + Y1Diff[1,7] = 0
 R768: Y1[0] - Y1[2] + Y1Diff[2,0] = 0
 R769: Y1[1] - Y1[2] + Y1Diff[2,1] = 0
 R770: Y1Diff[2,2] = 0
 R771: - Y1[2] + Y1[3] + Y1Diff[2,3] = 0
 R772: - Y1[2] + Y1[4] + Y1Diff[2,4] = 0
 R773: - Y1[2] + Y1[5] + Y1Diff[2,5] = 0
 R774: - Y1[2] + Y1[6] + Y1Diff[2,6] = 0
 R775: - Y1[2] + Y1[7] + Y1Diff[2,7] = 0
 R776: Y1[0] - Y1[3] + Y1Diff[3,0] = 0
 R777: Y1[1] - Y1[3] + Y1Diff[3,1] = 0
 R778: Y1[2] - Y1[3] + Y1Diff[3,2] = 0
 R779: Y1Diff[3,3] = 0
 R780: - Y1[3] + Y1[4] + Y1Diff[3,4] = 0
 R781: - Y1[3] + Y1[5] + Y1Diff[3,5] = 0
 R782: - Y1[3] + Y1[6] + Y1Diff[3,6] = 0
 R783: - Y1[3] + Y1[7] + Y1Diff[3,7] = 0
 R784: Y1[0] - Y1[4] + Y1Diff[4,0] = 0
 R785: Y1[1] - Y1[4] + Y1Diff[4,1] = 0
 R786: Y1[2] - Y1[4] + Y1Diff[4,2] = 0
 R787: Y1[3] - Y1[4] + Y1Diff[4,3] = 0
 R788: Y1Diff[4,4] = 0
 R789: - Y1[4] + Y1[5] + Y1Diff[4,5] = 0
 R790: - Y1[4] + Y1[6] + Y1Diff[4,6] = 0
 R791: - Y1[4] + Y1[7] + Y1Diff[4,7] = 0
 R792: Y1[0] - Y1[5] + Y1Diff[5,0] = 0
 R793: Y1[1] - Y1[5] + Y1Diff[5,1] = 0
 R794: Y1[2] - Y1[5] + Y1Diff[5,2] = 0
 R795: Y1[3] - Y1[5] + Y1Diff[5,3] = 0
 R796: Y1[4] - Y1[5] + Y1Diff[5,4] = 0
 R797: Y1Diff[5,5] = 0
 R798: - Y1[5] + Y1[6] + Y1Diff[5,6] = 0
 R799: - Y1[5] + Y1[7] + Y1Diff[5,7] = 0
 R800: Y1[0] - Y1[6] + Y1Diff[6,0] = 0
 R801: Y1[1] - Y1[6] + Y1Diff[6,1] = 0
 R802: Y1[2] - Y1[6] + Y1Diff[6,2] = 0
 R803: Y1[3] - Y1[6] + Y1Diff[6,3] = 0
 R804: Y1[4] - Y1[6] + Y1Diff[6,4] = 0
 R805: Y1[5] - Y1[6] + Y1Diff[6,5] = 0
 R806: Y1Diff[6,6] = 0
 R807: - Y1[6] + Y1[7] + Y1Diff[6,7] = 0
 R808: Y1[0] - Y1[7] + Y1Diff[7,0] = 0
 R809: Y1[1] - Y1[7] + Y1Diff[7,1] = 0
 R810: Y1[2] - Y1[7] + Y1Diff[7,2] = 0
 R811: Y1[3] - Y1[7] + Y1Diff[7,3] = 0
 R812: Y1[4] - Y1[7] + Y1Diff[7,4] = 0
 R813: Y1[5] - Y1[7] + Y1Diff[7,5] = 0
 R814: Y1[6] - Y1[7] + Y1Diff[7,6] = 0
 R815: Y1Diff[7,7] = 0
 PreventOvertOptimization: X0GroupCount + Y0GroupCount + X1GroupCount
   + Y1GroupCount >= 12
 PreserveOverlap[0,1]: Overlap[0,1] = 0
 PreserveOverlap[0,2]: Overlap[0,2] = 0
 PreserveOverlap[0,3]: Overlap[0,3] = 0
 PreserveOverlap[0,4]: Overlap[0,4] = 0
 PreserveOverlap[0,5]: Overlap[0,5] = 0
 PreserveOverlap[0,6]: Overlap[0,6] = 0
 PreserveOverlap[0,7]: Overlap[0,7] = 0
 PreserveOverlap[1,0]: Overlap[1,0] = 0
 PreserveOverlap[1,2]: Overlap[1,2] = 0
 PreserveOverlap[1,3]: Overlap[1,3] = 0
 PreserveOverlap[1,4]: Overlap[1,4] = 0
 PreserveOverlap[1,5]: Overlap[1,5] = 0
 PreserveOverlap[1,6]: Overlap[1,6] = 0
 PreserveOverlap[1,7]: Overlap[1,7] = 0
 PreserveOverlap[2,0]: Overlap[2,0] = 0
 PreserveOverlap[2,1]: Overlap[2,1] = 0
 PreserveOverlap[2,3]: Overlap[2,3] = 0
 PreserveOverlap[2,4]: Overlap[2,4] = 0
 PreserveOverlap[2,5]: Overlap[2,5] = 0
 PreserveOverlap[2,6]: Overlap[2,6] = 0
 PreserveOverlap[2,7]: Overlap[2,7] = 0
 PreserveOverlap[3,0]: Overlap[3,0] = 0
 PreserveOverlap[3,1]: Overlap[3,1] = 0
 PreserveOverlap[3,2]: Overlap[3,2] = 0
 PreserveOverlap[3,4]: Overlap[3,4] = 0
 PreserveOverlap[3,5]: Overlap[3,5] = 0
 PreserveOverlap[3,6]: Overlap[3,6] = 0
 PreserveOverlap[3,7]: Overlap[3,7] = 0
 PreserveOverlap[4,0]: Overlap[4,0] = 0
 PreserveOverlap[4,1]: Overlap[4,1] = 0
 PreserveOverlap[4,2]: Overlap[4,2] = 0
 PreserveOverlap[4,3]: Overlap[4,3] = 0
 PreserveOverlap[4,5]: Overlap[4,5] = 0
 PreserveOverlap[4,6]: Overlap[4,6] = 0
 PreserveOverlap[4,7]: Overlap[4,7] = 0
 PreserveOverlap[5,0]: Overlap[5,0] = 0
 PreserveOverlap[5,1]: Overlap[5,1] = 0
 PreserveOverlap[5,2]: Overlap[5,2] = 0
 PreserveOverlap[5,3]: Overlap[5,3] = 0
 PreserveOverlap[5,4]: Overlap[5,4] = 0
 PreserveOverlap[5,6]: Overlap[5,6] = 0
 PreserveOverlap[5,7]: Overlap[5,7] = 0
 PreserveOverlap[6,0]: Overlap[6,0] = 0
 PreserveOverlap[6,1]: Overlap[6,1] = 0
 PreserveOverlap[6,2]: Overlap[6,2] = 0
 PreserveOverlap[6,3]: Overlap[6,3] = 0
 PreserveOverlap[6,4]: Overlap[6,4] = 0
 PreserveOverlap[6,5]: Overlap[6,5] = 0
 PreserveOverlap[6,7]: Overlap[6,7] = 0
 PreserveOverlap[7,0]: Overlap[7,0] = 0
 PreserveOverlap[7,1]: Overlap[7,1] = 0
 PreserveOverlap[7,2]: Overlap[7,2] = 0
 PreserveOverlap[7,3]: Overlap[7,3] = 0
 PreserveOverlap[7,4]: Overlap[7,4] = 0
 PreserveOverlap[7,5]: Overlap[7,5] = 0
 PreserveOverlap[7,6]: Overlap[7,6] = 0
 LinkRightMargin: MaxX1 + MaxX1 = 125
 LinkMarginDiffLooseAbs1: - LeftMargin + MaxX1 + MarginDiffLooseAbs >= 0
 LinkMarginDiffLooseAbs2: LeftMargin - MaxX1 + MarginDiffLooseAbs >= 0
 LinkAbove1[0,1]: Above[0,1] = 1 -> Y1[0] - Y0[1] <= -0
 LinkAbove1[0,2]: Above[0,2] = 1 -> Y1[0] - Y0[2] <= -0
 LinkAbove1[0,3]: Above[0,3] = 1 -> Y1[0] - Y0[3] <= -0
 LinkAbove1[0,4]: Above[0,4] = 1 -> Y1[0] - Y0[4] <= -0
 LinkAbove1[0,5]: Above[0,5] = 1 -> Y1[0] - Y0[5] <= -0
 LinkAbove1[0,6]: Above[0,6] = 1 -> Y1[0] - Y0[6] <= -0
 LinkAbove1[0,7]: Above[0,7] = 1 -> Y1[0] - Y0[7] <= -0
 LinkAbove1[1,0]: Above[1,0] = 1 -> Y1[1] - Y0[0] <= -0
 LinkAbove1[1,2]: Above[1,2] = 1 -> Y1[1] - Y0[2] <= -0
 LinkAbove1[1,3]: Above[1,3] = 1 -> Y1[1] - Y0[3] <= -0
 LinkAbove1[1,4]: Above[1,4] = 1 -> Y1[1] - Y0[4] <= -0
 LinkAbove1[1,5]: Above[1,5] = 1 -> Y1[1] - Y0[5] <= -0
 LinkAbove1[1,6]: Above[1,6] = 1 -> Y1[1] - Y0[6] <= -0
 LinkAbove1[1,7]: Above[1,7] = 1 -> Y1[1] - Y0[7] <= -0
 LinkAbove1[2,0]: Above[2,0] = 1 -> Y1[2] - Y0[0] <= -0
 LinkAbove1[2,1]: Above[2,1] = 1 -> Y1[2] - Y0[1] <= -0
 LinkAbove1[2,3]: Above[2,3] = 1 -> Y1[2] - Y0[3] <= -0
 LinkAbove1[2,4]: Above[2,4] = 1 -> Y1[2] - Y0[4] <= -0
 LinkAbove1[2,5]: Above[2,5] = 1 -> Y1[2] - Y0[5] <= -0
 LinkAbove1[2,6]: Above[2,6] = 1 -> Y1[2] - Y0[6] <= -0
 LinkAbove1[2,7]: Above[2,7] = 1 -> Y1[2] - Y0[7] <= -0
 LinkAbove1[3,0]: Above[3,0] = 1 -> Y1[3] - Y0[0] <= -0
 LinkAbove1[3,1]: Above[3,1] = 1 -> Y1[3] - Y0[1] <= -0
 LinkAbove1[3,2]: Above[3,2] = 1 -> Y1[3] - Y0[2] <= -0
 LinkAbove1[3,4]: Above[3,4] = 1 -> Y1[3] - Y0[4] <= -0
 LinkAbove1[3,5]: Above[3,5] = 1 -> Y1[3] - Y0[5] <= -0
 LinkAbove1[3,6]: Above[3,6] = 1 -> Y1[3] - Y0[6] <= -0
 LinkAbove1[3,7]: Above[3,7] = 1 -> Y1[3] - Y0[7] <= -0
 LinkAbove1[4,0]: Above[4,0] = 1 -> Y1[4] - Y0[0] <= -0
 LinkAbove1[4,1]: Above[4,1] = 1 -> Y1[4] - Y0[1] <= -0
 LinkAbove1[4,2]: Above[4,2] = 1 -> Y1[4] - Y0[2] <= -0
 LinkAbove1[4,3]: Above[4,3] = 1 -> Y1[4] - Y0[3] <= -0
 LinkAbove1[4,5]: Above[4,5] = 1 -> Y1[4] - Y0[5] <= -0
 LinkAbove1[4,6]: Above[4,6] = 1 -> Y1[4] - Y0[6] <= -0
 LinkAbove1[4,7]: Above[4,7] = 1 -> Y1[4] - Y0[7] <= -0
 LinkAbove1[5,0]: Above[5,0] = 1 -> Y1[5] - Y0[0] <= -0
 LinkAbove1[5,1]: Above[5,1] = 1 -> Y1[5] - Y0[1] <= -0
 LinkAbove1[5,2]: Above[5,2] = 1 -> Y1[5] - Y0[2] <= -0
 LinkAbove1[5,3]: Above[5,3] = 1 -> Y1[5] - Y0[3] <= -0
 LinkAbove1[5,4]: Above[5,4] = 1 -> Y1[5] - Y0[4] <= -0
 LinkAbove1[5,6]: Above[5,6] = 1 -> Y1[5] - Y0[6] <= -0
 LinkAbove1[5,7]: Above[5,7] = 1 -> Y1[5] - Y0[7] <= -0
 LinkAbove1[6,0]: Above[6,0] = 1 -> Y1[6] - Y0[0] <= -0
 LinkAbove1[6,1]: Above[6,1] = 1 -> Y1[6] - Y0[1] <= -0
 LinkAbove1[6,2]: Above[6,2] = 1 -> Y1[6] - Y0[2] <= -0
 LinkAbove1[6,3]: Above[6,3] = 1 -> Y1[6] - Y0[3] <= -0
 LinkAbove1[6,4]: Above[6,4] = 1 -> Y1[6] - Y0[4] <= -0
 LinkAbove1[6,5]: Above[6,5] = 1 -> Y1[6] - Y0[5] <= -0
 LinkAbove1[6,7]: Above[6,7] = 1 -> Y1[6] - Y0[7] <= -0
 LinkAbove1[7,0]: Above[7,0] = 1 -> Y1[7] - Y0[0] <= -0
 LinkAbove1[7,1]: Above[7,1] = 1 -> Y1[7] - Y0[1] <= -0
 LinkAbove1[7,2]: Above[7,2] = 1 -> Y1[7] - Y0[2] <= -0
 LinkAbove1[7,3]: Above[7,3] = 1 -> Y1[7] - Y0[3] <= -0
 LinkAbove1[7,4]: Above[7,4] = 1 -> Y1[7] - Y0[4] <= -0
 LinkAbove1[7,5]: Above[7,5] = 1 -> Y1[7] - Y0[5] <= -0
 LinkAbove1[7,6]: Above[7,6] = 1 -> Y1[7] - Y0[6] <= -0
 LinkAbove2[0,1]: Above[0,1] = 0 -> Y1[0] - Y0[1] >= 1
 LinkAbove2[0,2]: Above[0,2] = 0 -> Y1[0] - Y0[2] >= 1
 LinkAbove2[0,3]: Above[0,3] = 0 -> Y1[0] - Y0[3] >= 1
 LinkAbove2[0,4]: Above[0,4] = 0 -> Y1[0] - Y0[4] >= 1
 LinkAbove2[0,5]: Above[0,5] = 0 -> Y1[0] - Y0[5] >= 1
 LinkAbove2[0,6]: Above[0,6] = 0 -> Y1[0] - Y0[6] >= 1
 LinkAbove2[0,7]: Above[0,7] = 0 -> Y1[0] - Y0[7] >= 1
 LinkAbove2[1,0]: Above[1,0] = 0 -> Y1[1] - Y0[0] >= 1
 LinkAbove2[1,2]: Above[1,2] = 0 -> Y1[1] - Y0[2] >= 1
 LinkAbove2[1,3]: Above[1,3] = 0 -> Y1[1] - Y0[3] >= 1
 LinkAbove2[1,4]: Above[1,4] = 0 -> Y1[1] - Y0[4] >= 1
 LinkAbove2[1,5]: Above[1,5] = 0 -> Y1[1] - Y0[5] >= 1
 LinkAbove2[1,6]: Above[1,6] = 0 -> Y1[1] - Y0[6] >= 1
 LinkAbove2[1,7]: Above[1,7] = 0 -> Y1[1] - Y0[7] >= 1
 LinkAbove2[2,0]: Above[2,0] = 0 -> Y1[2] - Y0[0] >= 1
 LinkAbove2[2,1]: Above[2,1] = 0 -> Y1[2] - Y0[1] >= 1
 LinkAbove2[2,3]: Above[2,3] = 0 -> Y1[2] - Y0[3] >= 1
 LinkAbove2[2,4]: Above[2,4] = 0 -> Y1[2] - Y0[4] >= 1
 LinkAbove2[2,5]: Above[2,5] = 0 -> Y1[2] - Y0[5] >= 1
 LinkAbove2[2,6]: Above[2,6] = 0 -> Y1[2] - Y0[6] >= 1
 LinkAbove2[2,7]: Above[2,7] = 0 -> Y1[2] - Y0[7] >= 1
 LinkAbove2[3,0]: Above[3,0] = 0 -> Y1[3] - Y0[0] >= 1
 LinkAbove2[3,1]: Above[3,1] = 0 -> Y1[3] - Y0[1] >= 1
 LinkAbove2[3,2]: Above[3,2] = 0 -> Y1[3] - Y0[2] >= 1
 LinkAbove2[3,4]: Above[3,4] = 0 -> Y1[3] - Y0[4] >= 1
 LinkAbove2[3,5]: Above[3,5] = 0 -> Y1[3] - Y0[5] >= 1
 LinkAbove2[3,6]: Above[3,6] = 0 -> Y1[3] - Y0[6] >= 1
 LinkAbove2[3,7]: Above[3,7] = 0 -> Y1[3] - Y0[7] >= 1
 LinkAbove2[4,0]: Above[4,0] = 0 -> Y1[4] - Y0[0] >= 1
 LinkAbove2[4,1]: Above[4,1] = 0 -> Y1[4] - Y0[1] >= 1
 LinkAbove2[4,2]: Above[4,2] = 0 -> Y1[4] - Y0[2] >= 1
 LinkAbove2[4,3]: Above[4,3] = 0 -> Y1[4] - Y0[3] >= 1
 LinkAbove2[4,5]: Above[4,5] = 0 -> Y1[4] - Y0[5] >= 1
 LinkAbove2[4,6]: Above[4,6] = 0 -> Y1[4] - Y0[6] >= 1
 LinkAbove2[4,7]: Above[4,7] = 0 -> Y1[4] - Y0[7] >= 1
 LinkAbove2[5,0]: Above[5,0] = 0 -> Y1[5] - Y0[0] >= 1
 LinkAbove2[5,1]: Above[5,1] = 0 -> Y1[5] - Y0[1] >= 1
 LinkAbove2[5,2]: Above[5,2] = 0 -> Y1[5] - Y0[2] >= 1
 LinkAbove2[5,3]: Above[5,3] = 0 -> Y1[5] - Y0[3] >= 1
 LinkAbove2[5,4]: Above[5,4] = 0 -> Y1[5] - Y0[4] >= 1
 LinkAbove2[5,6]: Above[5,6] = 0 -> Y1[5] - Y0[6] >= 1
 LinkAbove2[5,7]: Above[5,7] = 0 -> Y1[5] - Y0[7] >= 1
 LinkAbove2[6,0]: Above[6,0] = 0 -> Y1[6] - Y0[0] >= 1
 LinkAbove2[6,1]: Above[6,1] = 0 -> Y1[6] - Y0[1] >= 1
 LinkAbove2[6,2]: Above[6,2] = 0 -> Y1[6] - Y0[2] >= 1
 LinkAbove2[6,3]: Above[6,3] = 0 -> Y1[6] - Y0[3] >= 1
 LinkAbove2[6,4]: Above[6,4] = 0 -> Y1[6] - Y0[4] >= 1
 LinkAbove2[6,5]: Above[6,5] = 0 -> Y1[6] - Y0[5] >= 1
 LinkAbove2[6,7]: Above[6,7] = 0 -> Y1[6] - Y0[7] >= 1
 LinkAbove2[7,0]: Above[7,0] = 0 -> Y1[7] - Y0[0] >= 1
 LinkAbove2[7,1]: Above[7,1] = 0 -> Y1[7] - Y0[1] >= 1
 LinkAbove2[7,2]: Above[7,2] = 0 -> Y1[7] - Y0[2] >= 1
 LinkAbove2[7,3]: Above[7,3] = 0 -> Y1[7] - Y0[3] >= 1
 LinkAbove2[7,4]: Above[7,4] = 0 -> Y1[7] - Y0[4] >= 1
 LinkAbove2[7,5]: Above[7,5] = 0 -> Y1[7] - Y0[5] >= 1
 LinkAbove2[7,6]: Above[7,6] = 0 -> Y1[7] - Y0[6] >= 1
 LinkOnLeft1[0,1]: OnLeft[0,1] = 1 -> X1[0] - X0[1] <= -0
 LinkOnLeft1[0,2]: OnLeft[0,2] = 1 -> X1[0] - X0[2] <= -0
 LinkOnLeft1[0,3]: OnLeft[0,3] = 1 -> X1[0] - X0[3] <= -0
 LinkOnLeft1[0,4]: OnLeft[0,4] = 1 -> X1[0] - X0[4] <= -0
 LinkOnLeft1[0,5]: OnLeft[0,5] = 1 -> X1[0] - X0[5] <= -0
 LinkOnLeft1[0,6]: OnLeft[0,6] = 1 -> X1[0] - X0[6] <= -0
 LinkOnLeft1[0,7]: OnLeft[0,7] = 1 -> X1[0] - X0[7] <= -0
 LinkOnLeft1[1,0]: OnLeft[1,0] = 1 -> X1[1] - X0[0] <= -0
 LinkOnLeft1[1,2]: OnLeft[1,2] = 1 -> X1[1] - X0[2] <= -0
 LinkOnLeft1[1,3]: OnLeft[1,3] = 1 -> X1[1] - X0[3] <= -0
 LinkOnLeft1[1,4]: OnLeft[1,4] = 1 -> X1[1] - X0[4] <= -0
 LinkOnLeft1[1,5]: OnLeft[1,5] = 1 -> X1[1] - X0[5] <= -0
 LinkOnLeft1[1,6]: OnLeft[1,6] = 1 -> X1[1] - X0[6] <= -0
 LinkOnLeft1[1,7]: OnLeft[1,7] = 1 -> X1[1] - X0[7] <= -0
 LinkOnLeft1[2,0]: OnLeft[2,0] = 1 -> X1[2] - X0[0] <= -0
 LinkOnLeft1[2,1]: OnLeft[2,1] = 1 -> X1[2] - X0[1] <= -0
 LinkOnLeft1[2,3]: OnLeft[2,3] = 1 -> X1[2] - X0[3] <= -0
 LinkOnLeft1[2,4]: OnLeft[2,4] = 1 -> X1[2] - X0[4] <= -0
 LinkOnLeft1[2,5]: OnLeft[2,5] = 1 -> X1[2] - X0[5] <= -0
 LinkOnLeft1[2,6]: OnLeft[2,6] = 1 -> X1[2] - X0[6] <= -0
 LinkOnLeft1[2,7]: OnLeft[2,7] = 1 -> X1[2] - X0[7] <= -0
 LinkOnLeft1[3,0]: OnLeft[3,0] = 1 -> X1[3] - X0[0] <= -0
 LinkOnLeft1[3,1]: OnLeft[3,1] = 1 -> X1[3] - X0[1] <= -0
 LinkOnLeft1[3,2]: OnLeft[3,2] = 1 -> X1[3] - X0[2] <= -0
 LinkOnLeft1[3,4]: OnLeft[3,4] = 1 -> X1[3] - X0[4] <= -0
 LinkOnLeft1[3,5]: OnLeft[3,5] = 1 -> X1[3] - X0[5] <= -0
 LinkOnLeft1[3,6]: OnLeft[3,6] = 1 -> X1[3] - X0[6] <= -0
 LinkOnLeft1[3,7]: OnLeft[3,7] = 1 -> X1[3] - X0[7] <= -0
 LinkOnLeft1[4,0]: OnLeft[4,0] = 1 -> X1[4] - X0[0] <= -0
 LinkOnLeft1[4,1]: OnLeft[4,1] = 1 -> X1[4] - X0[1] <= -0
 LinkOnLeft1[4,2]: OnLeft[4,2] = 1 -> X1[4] - X0[2] <= -0
 LinkOnLeft1[4,3]: OnLeft[4,3] = 1 -> X1[4] - X0[3] <= -0
 LinkOnLeft1[4,5]: OnLeft[4,5] = 1 -> X1[4] - X0[5] <= -0
 LinkOnLeft1[4,6]: OnLeft[4,6] = 1 -> X1[4] - X0[6] <= -0
 LinkOnLeft1[4,7]: OnLeft[4,7] = 1 -> X1[4] - X0[7] <= -0
 LinkOnLeft1[5,0]: OnLeft[5,0] = 1 -> X1[5] - X0[0] <= -0
 LinkOnLeft1[5,1]: OnLeft[5,1] = 1 -> X1[5] - X0[1] <= -0
 LinkOnLeft1[5,2]: OnLeft[5,2] = 1 -> X1[5] - X0[2] <= -0
 LinkOnLeft1[5,3]: OnLeft[5,3] = 1 -> X1[5] - X0[3] <= -0
 LinkOnLeft1[5,4]: OnLeft[5,4] = 1 -> X1[5] - X0[4] <= -0
 LinkOnLeft1[5,6]: OnLeft[5,6] = 1 -> X1[5] - X0[6] <= -0
 LinkOnLeft1[5,7]: OnLeft[5,7] = 1 -> X1[5] - X0[7] <= -0
 LinkOnLeft1[6,0]: OnLeft[6,0] = 1 -> X1[6] - X0[0] <= -0
 LinkOnLeft1[6,1]: OnLeft[6,1] = 1 -> X1[6] - X0[1] <= -0
 LinkOnLeft1[6,2]: OnLeft[6,2] = 1 -> X1[6] - X0[2] <= -0
 LinkOnLeft1[6,3]: OnLeft[6,3] = 1 -> X1[6] - X0[3] <= -0
 LinkOnLeft1[6,4]: OnLeft[6,4] = 1 -> X1[6] - X0[4] <= -0
 LinkOnLeft1[6,5]: OnLeft[6,5] = 1 -> X1[6] - X0[5] <= -0
 LinkOnLeft1[6,7]: OnLeft[6,7] = 1 -> X1[6] - X0[7] <= -0
 LinkOnLeft1[7,0]: OnLeft[7,0] = 1 -> X1[7] - X0[0] <= -0
 LinkOnLeft1[7,1]: OnLeft[7,1] = 1 -> X1[7] - X0[1] <= -0
 LinkOnLeft1[7,2]: OnLeft[7,2] = 1 -> X1[7] - X0[2] <= -0
 LinkOnLeft1[7,3]: OnLeft[7,3] = 1 -> X1[7] - X0[3] <= -0
 LinkOnLeft1[7,4]: OnLeft[7,4] = 1 -> X1[7] - X0[4] <= -0
 LinkOnLeft1[7,5]: OnLeft[7,5] = 1 -> X1[7] - X0[5] <= -0
 LinkOnLeft1[7,6]: OnLeft[7,6] = 1 -> X1[7] - X0[6] <= -0
 LinkOnLeft2[0,1]: OnLeft[0,1] = 0 -> X1[0] - X0[1] >= 1
 LinkOnLeft2[0,2]: OnLeft[0,2] = 0 -> X1[0] - X0[2] >= 1
 LinkOnLeft2[0,3]: OnLeft[0,3] = 0 -> X1[0] - X0[3] >= 1
 LinkOnLeft2[0,4]: OnLeft[0,4] = 0 -> X1[0] - X0[4] >= 1
 LinkOnLeft2[0,5]: OnLeft[0,5] = 0 -> X1[0] - X0[5] >= 1
 LinkOnLeft2[0,6]: OnLeft[0,6] = 0 -> X1[0] - X0[6] >= 1
 LinkOnLeft2[0,7]: OnLeft[0,7] = 0 -> X1[0] - X0[7] >= 1
 LinkOnLeft2[1,0]: OnLeft[1,0] = 0 -> X1[1] - X0[0] >= 1
 LinkOnLeft2[1,2]: OnLeft[1,2] = 0 -> X1[1] - X0[2] >= 1
 LinkOnLeft2[1,3]: OnLeft[1,3] = 0 -> X1[1] - X0[3] >= 1
 LinkOnLeft2[1,4]: OnLeft[1,4] = 0 -> X1[1] - X0[4] >= 1
 LinkOnLeft2[1,5]: OnLeft[1,5] = 0 -> X1[1] - X0[5] >= 1
 LinkOnLeft2[1,6]: OnLeft[1,6] = 0 -> X1[1] - X0[6] >= 1
 LinkOnLeft2[1,7]: OnLeft[1,7] = 0 -> X1[1] - X0[7] >= 1
 LinkOnLeft2[2,0]: OnLeft[2,0] = 0 -> X1[2] - X0[0] >= 1
 LinkOnLeft2[2,1]: OnLeft[2,1] = 0 -> X1[2] - X0[1] >= 1
 LinkOnLeft2[2,3]: OnLeft[2,3] = 0 -> X1[2] - X0[3] >= 1
 LinkOnLeft2[2,4]: OnLeft[2,4] = 0 -> X1[2] - X0[4] >= 1
 LinkOnLeft2[2,5]: OnLeft[2,5] = 0 -> X1[2] - X0[5] >= 1
 LinkOnLeft2[2,6]: OnLeft[2,6] = 0 -> X1[2] - X0[6] >= 1
 LinkOnLeft2[2,7]: OnLeft[2,7] = 0 -> X1[2] - X0[7] >= 1
 LinkOnLeft2[3,0]: OnLeft[3,0] = 0 -> X1[3] - X0[0] >= 1
 LinkOnLeft2[3,1]: OnLeft[3,1] = 0 -> X1[3] - X0[1] >= 1
 LinkOnLeft2[3,2]: OnLeft[3,2] = 0 -> X1[3] - X0[2] >= 1
 LinkOnLeft2[3,4]: OnLeft[3,4] = 0 -> X1[3] - X0[4] >= 1
 LinkOnLeft2[3,5]: OnLeft[3,5] = 0 -> X1[3] - X0[5] >= 1
 LinkOnLeft2[3,6]: OnLeft[3,6] = 0 -> X1[3] - X0[6] >= 1
 LinkOnLeft2[3,7]: OnLeft[3,7] = 0 -> X1[3] - X0[7] >= 1
 LinkOnLeft2[4,0]: OnLeft[4,0] = 0 -> X1[4] - X0[0] >= 1
 LinkOnLeft2[4,1]: OnLeft[4,1] = 0 -> X1[4] - X0[1] >= 1
 LinkOnLeft2[4,2]: OnLeft[4,2] = 0 -> X1[4] - X0[2] >= 1
 LinkOnLeft2[4,3]: OnLeft[4,3] = 0 -> X1[4] - X0[3] >= 1
 LinkOnLeft2[4,5]: OnLeft[4,5] = 0 -> X1[4] - X0[5] >= 1
 LinkOnLeft2[4,6]: OnLeft[4,6] = 0 -> X1[4] - X0[6] >= 1
 LinkOnLeft2[4,7]: OnLeft[4,7] = 0 -> X1[4] - X0[7] >= 1
 LinkOnLeft2[5,0]: OnLeft[5,0] = 0 -> X1[5] - X0[0] >= 1
 LinkOnLeft2[5,1]: OnLeft[5,1] = 0 -> X1[5] - X0[1] >= 1
 LinkOnLeft2[5,2]: OnLeft[5,2] = 0 -> X1[5] - X0[2] >= 1
 LinkOnLeft2[5,3]: OnLeft[5,3] = 0 -> X1[5] - X0[3] >= 1
 LinkOnLeft2[5,4]: OnLeft[5,4] = 0 -> X1[5] - X0[4] >= 1
 LinkOnLeft2[5,6]: OnLeft[5,6] = 0 -> X1[5] - X0[6] >= 1
 LinkOnLeft2[5,7]: OnLeft[5,7] = 0 -> X1[5] - X0[7] >= 1
 LinkOnLeft2[6,0]: OnLeft[6,0] = 0 -> X1[6] - X0[0] >= 1
 LinkOnLeft2[6,1]: OnLeft[6,1] = 0 -> X1[6] - X0[1] >= 1
 LinkOnLeft2[6,2]: OnLeft[6,2] = 0 -> X1[6] - X0[2] >= 1
 LinkOnLeft2[6,3]: OnLeft[6,3] = 0 -> X1[6] - X0[3] >= 1
 LinkOnLeft2[6,4]: OnLeft[6,4] = 0 -> X1[6] - X0[4] >= 1
 LinkOnLeft2[6,5]: OnLeft[6,5] = 0 -> X1[6] - X0[5] >= 1
 LinkOnLeft2[6,7]: OnLeft[6,7] = 0 -> X1[6] - X0[7] >= 1
 LinkOnLeft2[7,0]: OnLeft[7,0] = 0 -> X1[7] - X0[0] >= 1
 LinkOnLeft2[7,1]: OnLeft[7,1] = 0 -> X1[7] - X0[1] >= 1
 LinkOnLeft2[7,2]: OnLeft[7,2] = 0 -> X1[7] - X0[2] >= 1
 LinkOnLeft2[7,3]: OnLeft[7,3] = 0 -> X1[7] - X0[3] >= 1
 LinkOnLeft2[7,4]: OnLeft[7,4] = 0 -> X1[7] - X0[4] >= 1
 LinkOnLeft2[7,5]: OnLeft[7,5] = 0 -> X1[7] - X0[5] >= 1
 LinkOnLeft2[7,6]: OnLeft[7,6] = 0 -> X1[7] - X0[6] >= 1
 LinkX0LessThan1[0,1]: X0LessThan[0,1] = 1 -> X0Diff[0,1] <= -1
 LinkX0LessThan1[0,2]: X0LessThan[0,2] = 1 -> X0Diff[0,2] <= -1
 LinkX0LessThan1[0,3]: X0LessThan[0,3] = 1 -> X0Diff[0,3] <= -1
 LinkX0LessThan1[0,4]: X0LessThan[0,4] = 1 -> X0Diff[0,4] <= -1
 LinkX0LessThan1[0,5]: X0LessThan[0,5] = 1 -> X0Diff[0,5] <= -1
 LinkX0LessThan1[0,6]: X0LessThan[0,6] = 1 -> X0Diff[0,6] <= -1
 LinkX0LessThan1[0,7]: X0LessThan[0,7] = 1 -> X0Diff[0,7] <= -1
 LinkX0LessThan1[1,0]: X0LessThan[1,0] = 1 -> X0Diff[1,0] <= -1
 LinkX0LessThan1[1,2]: X0LessThan[1,2] = 1 -> X0Diff[1,2] <= -1
 LinkX0LessThan1[1,3]: X0LessThan[1,3] = 1 -> X0Diff[1,3] <= -1
 LinkX0LessThan1[1,4]: X0LessThan[1,4] = 1 -> X0Diff[1,4] <= -1
 LinkX0LessThan1[1,5]: X0LessThan[1,5] = 1 -> X0Diff[1,5] <= -1
 LinkX0LessThan1[1,6]: X0LessThan[1,6] = 1 -> X0Diff[1,6] <= -1
 LinkX0LessThan1[1,7]: X0LessThan[1,7] = 1 -> X0Diff[1,7] <= -1
 LinkX0LessThan1[2,0]: X0LessThan[2,0] = 1 -> X0Diff[2,0] <= -1
 LinkX0LessThan1[2,1]: X0LessThan[2,1] = 1 -> X0Diff[2,1] <= -1
 LinkX0LessThan1[2,3]: X0LessThan[2,3] = 1 -> X0Diff[2,3] <= -1
 LinkX0LessThan1[2,4]: X0LessThan[2,4] = 1 -> X0Diff[2,4] <= -1
 LinkX0LessThan1[2,5]: X0LessThan[2,5] = 1 -> X0Diff[2,5] <= -1
 LinkX0LessThan1[2,6]: X0LessThan[2,6] = 1 -> X0Diff[2,6] <= -1
 LinkX0LessThan1[2,7]: X0LessThan[2,7] = 1 -> X0Diff[2,7] <= -1
 LinkX0LessThan1[3,0]: X0LessThan[3,0] = 1 -> X0Diff[3,0] <= -1
 LinkX0LessThan1[3,1]: X0LessThan[3,1] = 1 -> X0Diff[3,1] <= -1
 LinkX0LessThan1[3,2]: X0LessThan[3,2] = 1 -> X0Diff[3,2] <= -1
 LinkX0LessThan1[3,4]: X0LessThan[3,4] = 1 -> X0Diff[3,4] <= -1
 LinkX0LessThan1[3,5]: X0LessThan[3,5] = 1 -> X0Diff[3,5] <= -1
 LinkX0LessThan1[3,6]: X0LessThan[3,6] = 1 -> X0Diff[3,6] <= -1
 LinkX0LessThan1[3,7]: X0LessThan[3,7] = 1 -> X0Diff[3,7] <= -1
 LinkX0LessThan1[4,0]: X0LessThan[4,0] = 1 -> X0Diff[4,0] <= -1
 LinkX0LessThan1[4,1]: X0LessThan[4,1] = 1 -> X0Diff[4,1] <= -1
 LinkX0LessThan1[4,2]: X0LessThan[4,2] = 1 -> X0Diff[4,2] <= -1
 LinkX0LessThan1[4,3]: X0LessThan[4,3] = 1 -> X0Diff[4,3] <= -1
 LinkX0LessThan1[4,5]: X0LessThan[4,5] = 1 -> X0Diff[4,5] <= -1
 LinkX0LessThan1[4,6]: X0LessThan[4,6] = 1 -> X0Diff[4,6] <= -1
 LinkX0LessThan1[4,7]: X0LessThan[4,7] = 1 -> X0Diff[4,7] <= -1
 LinkX0LessThan1[5,0]: X0LessThan[5,0] = 1 -> X0Diff[5,0] <= -1
 LinkX0LessThan1[5,1]: X0LessThan[5,1] = 1 -> X0Diff[5,1] <= -1
 LinkX0LessThan1[5,2]: X0LessThan[5,2] = 1 -> X0Diff[5,2] <= -1
 LinkX0LessThan1[5,3]: X0LessThan[5,3] = 1 -> X0Diff[5,3] <= -1
 LinkX0LessThan1[5,4]: X0LessThan[5,4] = 1 -> X0Diff[5,4] <= -1
 LinkX0LessThan1[5,6]: X0LessThan[5,6] = 1 -> X0Diff[5,6] <= -1
 LinkX0LessThan1[5,7]: X0LessThan[5,7] = 1 -> X0Diff[5,7] <= -1
 LinkX0LessThan1[6,0]: X0LessThan[6,0] = 1 -> X0Diff[6,0] <= -1
 LinkX0LessThan1[6,1]: X0LessThan[6,1] = 1 -> X0Diff[6,1] <= -1
 LinkX0LessThan1[6,2]: X0LessThan[6,2] = 1 -> X0Diff[6,2] <= -1
 LinkX0LessThan1[6,3]: X0LessThan[6,3] = 1 -> X0Diff[6,3] <= -1
 LinkX0LessThan1[6,4]: X0LessThan[6,4] = 1 -> X0Diff[6,4] <= -1
 LinkX0LessThan1[6,5]: X0LessThan[6,5] = 1 -> X0Diff[6,5] <= -1
 LinkX0LessThan1[6,7]: X0LessThan[6,7] = 1 -> X0Diff[6,7] <= -1
 LinkX0LessThan1[7,0]: X0LessThan[7,0] = 1 -> X0Diff[7,0] <= -1
 LinkX0LessThan1[7,1]: X0LessThan[7,1] = 1 -> X0Diff[7,1] <= -1
 LinkX0LessThan1[7,2]: X0LessThan[7,2] = 1 -> X0Diff[7,2] <= -1
 LinkX0LessThan1[7,3]: X0LessThan[7,3] = 1 -> X0Diff[7,3] <= -1
 LinkX0LessThan1[7,4]: X0LessThan[7,4] = 1 -> X0Diff[7,4] <= -1
 LinkX0LessThan1[7,5]: X0LessThan[7,5] = 1 -> X0Diff[7,5] <= -1
 LinkX0LessThan1[7,6]: X0LessThan[7,6] = 1 -> X0Diff[7,6] <= -1
 LinkX0LessThan2[0,1]: X0LessThan[0,1] = 0 -> X0Diff[0,1] >= -0
 LinkX0LessThan2[0,2]: X0LessThan[0,2] = 0 -> X0Diff[0,2] >= -0
 LinkX0LessThan2[0,3]: X0LessThan[0,3] = 0 -> X0Diff[0,3] >= -0
 LinkX0LessThan2[0,4]: X0LessThan[0,4] = 0 -> X0Diff[0,4] >= -0
 LinkX0LessThan2[0,5]: X0LessThan[0,5] = 0 -> X0Diff[0,5] >= -0
 LinkX0LessThan2[0,6]: X0LessThan[0,6] = 0 -> X0Diff[0,6] >= -0
 LinkX0LessThan2[0,7]: X0LessThan[0,7] = 0 -> X0Diff[0,7] >= -0
 LinkX0LessThan2[1,0]: X0LessThan[1,0] = 0 -> X0Diff[1,0] >= -0
 LinkX0LessThan2[1,2]: X0LessThan[1,2] = 0 -> X0Diff[1,2] >= -0
 LinkX0LessThan2[1,3]: X0LessThan[1,3] = 0 -> X0Diff[1,3] >= -0
 LinkX0LessThan2[1,4]: X0LessThan[1,4] = 0 -> X0Diff[1,4] >= -0
 LinkX0LessThan2[1,5]: X0LessThan[1,5] = 0 -> X0Diff[1,5] >= -0
 LinkX0LessThan2[1,6]: X0LessThan[1,6] = 0 -> X0Diff[1,6] >= -0
 LinkX0LessThan2[1,7]: X0LessThan[1,7] = 0 -> X0Diff[1,7] >= -0
 LinkX0LessThan2[2,0]: X0LessThan[2,0] = 0 -> X0Diff[2,0] >= -0
 LinkX0LessThan2[2,1]: X0LessThan[2,1] = 0 -> X0Diff[2,1] >= -0
 LinkX0LessThan2[2,3]: X0LessThan[2,3] = 0 -> X0Diff[2,3] >= -0
 LinkX0LessThan2[2,4]: X0LessThan[2,4] = 0 -> X0Diff[2,4] >= -0
 LinkX0LessThan2[2,5]: X0LessThan[2,5] = 0 -> X0Diff[2,5] >= -0
 LinkX0LessThan2[2,6]: X0LessThan[2,6] = 0 -> X0Diff[2,6] >= -0
 LinkX0LessThan2[2,7]: X0LessThan[2,7] = 0 -> X0Diff[2,7] >= -0
 LinkX0LessThan2[3,0]: X0LessThan[3,0] = 0 -> X0Diff[3,0] >= -0
 LinkX0LessThan2[3,1]: X0LessThan[3,1] = 0 -> X0Diff[3,1] >= -0
 LinkX0LessThan2[3,2]: X0LessThan[3,2] = 0 -> X0Diff[3,2] >= -0
 LinkX0LessThan2[3,4]: X0LessThan[3,4] = 0 -> X0Diff[3,4] >= -0
 LinkX0LessThan2[3,5]: X0LessThan[3,5] = 0 -> X0Diff[3,5] >= -0
 LinkX0LessThan2[3,6]: X0LessThan[3,6] = 0 -> X0Diff[3,6] >= -0
 LinkX0LessThan2[3,7]: X0LessThan[3,7] = 0 -> X0Diff[3,7] >= -0
 LinkX0LessThan2[4,0]: X0LessThan[4,0] = 0 -> X0Diff[4,0] >= -0
 LinkX0LessThan2[4,1]: X0LessThan[4,1] = 0 -> X0Diff[4,1] >= -0
 LinkX0LessThan2[4,2]: X0LessThan[4,2] = 0 -> X0Diff[4,2] >= -0
 LinkX0LessThan2[4,3]: X0LessThan[4,3] = 0 -> X0Diff[4,3] >= -0
 LinkX0LessThan2[4,5]: X0LessThan[4,5] = 0 -> X0Diff[4,5] >= -0
 LinkX0LessThan2[4,6]: X0LessThan[4,6] = 0 -> X0Diff[4,6] >= -0
 LinkX0LessThan2[4,7]: X0LessThan[4,7] = 0 -> X0Diff[4,7] >= -0
 LinkX0LessThan2[5,0]: X0LessThan[5,0] = 0 -> X0Diff[5,0] >= -0
 LinkX0LessThan2[5,1]: X0LessThan[5,1] = 0 -> X0Diff[5,1] >= -0
 LinkX0LessThan2[5,2]: X0LessThan[5,2] = 0 -> X0Diff[5,2] >= -0
 LinkX0LessThan2[5,3]: X0LessThan[5,3] = 0 -> X0Diff[5,3] >= -0
 LinkX0LessThan2[5,4]: X0LessThan[5,4] = 0 -> X0Diff[5,4] >= -0
 LinkX0LessThan2[5,6]: X0LessThan[5,6] = 0 -> X0Diff[5,6] >= -0
 LinkX0LessThan2[5,7]: X0LessThan[5,7] = 0 -> X0Diff[5,7] >= -0
 LinkX0LessThan2[6,0]: X0LessThan[6,0] = 0 -> X0Diff[6,0] >= -0
 LinkX0LessThan2[6,1]: X0LessThan[6,1] = 0 -> X0Diff[6,1] >= -0
 LinkX0LessThan2[6,2]: X0LessThan[6,2] = 0 -> X0Diff[6,2] >= -0
 LinkX0LessThan2[6,3]: X0LessThan[6,3] = 0 -> X0Diff[6,3] >= -0
 LinkX0LessThan2[6,4]: X0LessThan[6,4] = 0 -> X0Diff[6,4] >= -0
 LinkX0LessThan2[6,5]: X0LessThan[6,5] = 0 -> X0Diff[6,5] >= -0
 LinkX0LessThan2[6,7]: X0LessThan[6,7] = 0 -> X0Diff[6,7] >= -0
 LinkX0LessThan2[7,0]: X0LessThan[7,0] = 0 -> X0Diff[7,0] >= -0
 LinkX0LessThan2[7,1]: X0LessThan[7,1] = 0 -> X0Diff[7,1] >= -0
 LinkX0LessThan2[7,2]: X0LessThan[7,2] = 0 -> X0Diff[7,2] >= -0
 LinkX0LessThan2[7,3]: X0LessThan[7,3] = 0 -> X0Diff[7,3] >= -0
 LinkX0LessThan2[7,4]: X0LessThan[7,4] = 0 -> X0Diff[7,4] >= -0
 LinkX0LessThan2[7,5]: X0LessThan[7,5] = 0 -> X0Diff[7,5] >= -0
 LinkX0LessThan2[7,6]: X0LessThan[7,6] = 0 -> X0Diff[7,6] >= -0
 LinkY0LessThan1[0,1]: Y0LessThan[0,1] = 1 -> Y0Diff[0,1] <= -1
 LinkY0LessThan1[0,2]: Y0LessThan[0,2] = 1 -> Y0Diff[0,2] <= -1
 LinkY0LessThan1[0,3]: Y0LessThan[0,3] = 1 -> Y0Diff[0,3] <= -1
 LinkY0LessThan1[0,4]: Y0LessThan[0,4] = 1 -> Y0Diff[0,4] <= -1
 LinkY0LessThan1[0,5]: Y0LessThan[0,5] = 1 -> Y0Diff[0,5] <= -1
 LinkY0LessThan1[0,6]: Y0LessThan[0,6] = 1 -> Y0Diff[0,6] <= -1
 LinkY0LessThan1[0,7]: Y0LessThan[0,7] = 1 -> Y0Diff[0,7] <= -1
 LinkY0LessThan1[1,0]: Y0LessThan[1,0] = 1 -> Y0Diff[1,0] <= -1
 LinkY0LessThan1[1,2]: Y0LessThan[1,2] = 1 -> Y0Diff[1,2] <= -1
 LinkY0LessThan1[1,3]: Y0LessThan[1,3] = 1 -> Y0Diff[1,3] <= -1
 LinkY0LessThan1[1,4]: Y0LessThan[1,4] = 1 -> Y0Diff[1,4] <= -1
 LinkY0LessThan1[1,5]: Y0LessThan[1,5] = 1 -> Y0Diff[1,5] <= -1
 LinkY0LessThan1[1,6]: Y0LessThan[1,6] = 1 -> Y0Diff[1,6] <= -1
 LinkY0LessThan1[1,7]: Y0LessThan[1,7] = 1 -> Y0Diff[1,7] <= -1
 LinkY0LessThan1[2,0]: Y0LessThan[2,0] = 1 -> Y0Diff[2,0] <= -1
 LinkY0LessThan1[2,1]: Y0LessThan[2,1] = 1 -> Y0Diff[2,1] <= -1
 LinkY0LessThan1[2,3]: Y0LessThan[2,3] = 1 -> Y0Diff[2,3] <= -1
 LinkY0LessThan1[2,4]: Y0LessThan[2,4] = 1 -> Y0Diff[2,4] <= -1
 LinkY0LessThan1[2,5]: Y0LessThan[2,5] = 1 -> Y0Diff[2,5] <= -1
 LinkY0LessThan1[2,6]: Y0LessThan[2,6] = 1 -> Y0Diff[2,6] <= -1
 LinkY0LessThan1[2,7]: Y0LessThan[2,7] = 1 -> Y0Diff[2,7] <= -1
 LinkY0LessThan1[3,0]: Y0LessThan[3,0] = 1 -> Y0Diff[3,0] <= -1
 LinkY0LessThan1[3,1]: Y0LessThan[3,1] = 1 -> Y0Diff[3,1] <= -1
 LinkY0LessThan1[3,2]: Y0LessThan[3,2] = 1 -> Y0Diff[3,2] <= -1
 LinkY0LessThan1[3,4]: Y0LessThan[3,4] = 1 -> Y0Diff[3,4] <= -1
 LinkY0LessThan1[3,5]: Y0LessThan[3,5] = 1 -> Y0Diff[3,5] <= -1
 LinkY0LessThan1[3,6]: Y0LessThan[3,6] = 1 -> Y0Diff[3,6] <= -1
 LinkY0LessThan1[3,7]: Y0LessThan[3,7] = 1 -> Y0Diff[3,7] <= -1
 LinkY0LessThan1[4,0]: Y0LessThan[4,0] = 1 -> Y0Diff[4,0] <= -1
 LinkY0LessThan1[4,1]: Y0LessThan[4,1] = 1 -> Y0Diff[4,1] <= -1
 LinkY0LessThan1[4,2]: Y0LessThan[4,2] = 1 -> Y0Diff[4,2] <= -1
 LinkY0LessThan1[4,3]: Y0LessThan[4,3] = 1 -> Y0Diff[4,3] <= -1
 LinkY0LessThan1[4,5]: Y0LessThan[4,5] = 1 -> Y0Diff[4,5] <= -1
 LinkY0LessThan1[4,6]: Y0LessThan[4,6] = 1 -> Y0Diff[4,6] <= -1
 LinkY0LessThan1[4,7]: Y0LessThan[4,7] = 1 -> Y0Diff[4,7] <= -1
 LinkY0LessThan1[5,0]: Y0LessThan[5,0] = 1 -> Y0Diff[5,0] <= -1
 LinkY0LessThan1[5,1]: Y0LessThan[5,1] = 1 -> Y0Diff[5,1] <= -1
 LinkY0LessThan1[5,2]: Y0LessThan[5,2] = 1 -> Y0Diff[5,2] <= -1
 LinkY0LessThan1[5,3]: Y0LessThan[5,3] = 1 -> Y0Diff[5,3] <= -1
 LinkY0LessThan1[5,4]: Y0LessThan[5,4] = 1 -> Y0Diff[5,4] <= -1
 LinkY0LessThan1[5,6]: Y0LessThan[5,6] = 1 -> Y0Diff[5,6] <= -1
 LinkY0LessThan1[5,7]: Y0LessThan[5,7] = 1 -> Y0Diff[5,7] <= -1
 LinkY0LessThan1[6,0]: Y0LessThan[6,0] = 1 -> Y0Diff[6,0] <= -1
 LinkY0LessThan1[6,1]: Y0LessThan[6,1] = 1 -> Y0Diff[6,1] <= -1
 LinkY0LessThan1[6,2]: Y0LessThan[6,2] = 1 -> Y0Diff[6,2] <= -1
 LinkY0LessThan1[6,3]: Y0LessThan[6,3] = 1 -> Y0Diff[6,3] <= -1
 LinkY0LessThan1[6,4]: Y0LessThan[6,4] = 1 -> Y0Diff[6,4] <= -1
 LinkY0LessThan1[6,5]: Y0LessThan[6,5] = 1 -> Y0Diff[6,5] <= -1
 LinkY0LessThan1[6,7]: Y0LessThan[6,7] = 1 -> Y0Diff[6,7] <= -1
 LinkY0LessThan1[7,0]: Y0LessThan[7,0] = 1 -> Y0Diff[7,0] <= -1
 LinkY0LessThan1[7,1]: Y0LessThan[7,1] = 1 -> Y0Diff[7,1] <= -1
 LinkY0LessThan1[7,2]: Y0LessThan[7,2] = 1 -> Y0Diff[7,2] <= -1
 LinkY0LessThan1[7,3]: Y0LessThan[7,3] = 1 -> Y0Diff[7,3] <= -1
 LinkY0LessThan1[7,4]: Y0LessThan[7,4] = 1 -> Y0Diff[7,4] <= -1
 LinkY0LessThan1[7,5]: Y0LessThan[7,5] = 1 -> Y0Diff[7,5] <= -1
 LinkY0LessThan1[7,6]: Y0LessThan[7,6] = 1 -> Y0Diff[7,6] <= -1
 LinkY0LessThan2[0,1]: Y0LessThan[0,1] = 0 -> Y0Diff[0,1] >= -0
 LinkY0LessThan2[0,2]: Y0LessThan[0,2] = 0 -> Y0Diff[0,2] >= -0
 LinkY0LessThan2[0,3]: Y0LessThan[0,3] = 0 -> Y0Diff[0,3] >= -0
 LinkY0LessThan2[0,4]: Y0LessThan[0,4] = 0 -> Y0Diff[0,4] >= -0
 LinkY0LessThan2[0,5]: Y0LessThan[0,5] = 0 -> Y0Diff[0,5] >= -0
 LinkY0LessThan2[0,6]: Y0LessThan[0,6] = 0 -> Y0Diff[0,6] >= -0
 LinkY0LessThan2[0,7]: Y0LessThan[0,7] = 0 -> Y0Diff[0,7] >= -0
 LinkY0LessThan2[1,0]: Y0LessThan[1,0] = 0 -> Y0Diff[1,0] >= -0
 LinkY0LessThan2[1,2]: Y0LessThan[1,2] = 0 -> Y0Diff[1,2] >= -0
 LinkY0LessThan2[1,3]: Y0LessThan[1,3] = 0 -> Y0Diff[1,3] >= -0
 LinkY0LessThan2[1,4]: Y0LessThan[1,4] = 0 -> Y0Diff[1,4] >= -0
 LinkY0LessThan2[1,5]: Y0LessThan[1,5] = 0 -> Y0Diff[1,5] >= -0
 LinkY0LessThan2[1,6]: Y0LessThan[1,6] = 0 -> Y0Diff[1,6] >= -0
 LinkY0LessThan2[1,7]: Y0LessThan[1,7] = 0 -> Y0Diff[1,7] >= -0
 LinkY0LessThan2[2,0]: Y0LessThan[2,0] = 0 -> Y0Diff[2,0] >= -0
 LinkY0LessThan2[2,1]: Y0LessThan[2,1] = 0 -> Y0Diff[2,1] >= -0
 LinkY0LessThan2[2,3]: Y0LessThan[2,3] = 0 -> Y0Diff[2,3] >= -0
 LinkY0LessThan2[2,4]: Y0LessThan[2,4] = 0 -> Y0Diff[2,4] >= -0
 LinkY0LessThan2[2,5]: Y0LessThan[2,5] = 0 -> Y0Diff[2,5] >= -0
 LinkY0LessThan2[2,6]: Y0LessThan[2,6] = 0 -> Y0Diff[2,6] >= -0
 LinkY0LessThan2[2,7]: Y0LessThan[2,7] = 0 -> Y0Diff[2,7] >= -0
 LinkY0LessThan2[3,0]: Y0LessThan[3,0] = 0 -> Y0Diff[3,0] >= -0
 LinkY0LessThan2[3,1]: Y0LessThan[3,1] = 0 -> Y0Diff[3,1] >= -0
 LinkY0LessThan2[3,2]: Y0LessThan[3,2] = 0 -> Y0Diff[3,2] >= -0
 LinkY0LessThan2[3,4]: Y0LessThan[3,4] = 0 -> Y0Diff[3,4] >= -0
 LinkY0LessThan2[3,5]: Y0LessThan[3,5] = 0 -> Y0Diff[3,5] >= -0
 LinkY0LessThan2[3,6]: Y0LessThan[3,6] = 0 -> Y0Diff[3,6] >= -0
 LinkY0LessThan2[3,7]: Y0LessThan[3,7] = 0 -> Y0Diff[3,7] >= -0
 LinkY0LessThan2[4,0]: Y0LessThan[4,0] = 0 -> Y0Diff[4,0] >= -0
 LinkY0LessThan2[4,1]: Y0LessThan[4,1] = 0 -> Y0Diff[4,1] >= -0
 LinkY0LessThan2[4,2]: Y0LessThan[4,2] = 0 -> Y0Diff[4,2] >= -0
 LinkY0LessThan2[4,3]: Y0LessThan[4,3] = 0 -> Y0Diff[4,3] >= -0
 LinkY0LessThan2[4,5]: Y0LessThan[4,5] = 0 -> Y0Diff[4,5] >= -0
 LinkY0LessThan2[4,6]: Y0LessThan[4,6] = 0 -> Y0Diff[4,6] >= -0
 LinkY0LessThan2[4,7]: Y0LessThan[4,7] = 0 -> Y0Diff[4,7] >= -0
 LinkY0LessThan2[5,0]: Y0LessThan[5,0] = 0 -> Y0Diff[5,0] >= -0
 LinkY0LessThan2[5,1]: Y0LessThan[5,1] = 0 -> Y0Diff[5,1] >= -0
 LinkY0LessThan2[5,2]: Y0LessThan[5,2] = 0 -> Y0Diff[5,2] >= -0
 LinkY0LessThan2[5,3]: Y0LessThan[5,3] = 0 -> Y0Diff[5,3] >= -0
 LinkY0LessThan2[5,4]: Y0LessThan[5,4] = 0 -> Y0Diff[5,4] >= -0
 LinkY0LessThan2[5,6]: Y0LessThan[5,6] = 0 -> Y0Diff[5,6] >= -0
 LinkY0LessThan2[5,7]: Y0LessThan[5,7] = 0 -> Y0Diff[5,7] >= -0
 LinkY0LessThan2[6,0]: Y0LessThan[6,0] = 0 -> Y0Diff[6,0] >= -0
 LinkY0LessThan2[6,1]: Y0LessThan[6,1] = 0 -> Y0Diff[6,1] >= -0
 LinkY0LessThan2[6,2]: Y0LessThan[6,2] = 0 -> Y0Diff[6,2] >= -0
 LinkY0LessThan2[6,3]: Y0LessThan[6,3] = 0 -> Y0Diff[6,3] >= -0
 LinkY0LessThan2[6,4]: Y0LessThan[6,4] = 0 -> Y0Diff[6,4] >= -0
 LinkY0LessThan2[6,5]: Y0LessThan[6,5] = 0 -> Y0Diff[6,5] >= -0
 LinkY0LessThan2[6,7]: Y0LessThan[6,7] = 0 -> Y0Diff[6,7] >= -0
 LinkY0LessThan2[7,0]: Y0LessThan[7,0] = 0 -> Y0Diff[7,0] >= -0
 LinkY0LessThan2[7,1]: Y0LessThan[7,1] = 0 -> Y0Diff[7,1] >= -0
 LinkY0LessThan2[7,2]: Y0LessThan[7,2] = 0 -> Y0Diff[7,2] >= -0
 LinkY0LessThan2[7,3]: Y0LessThan[7,3] = 0 -> Y0Diff[7,3] >= -0
 LinkY0LessThan2[7,4]: Y0LessThan[7,4] = 0 -> Y0Diff[7,4] >= -0
 LinkY0LessThan2[7,5]: Y0LessThan[7,5] = 0 -> Y0Diff[7,5] >= -0
 LinkY0LessThan2[7,6]: Y0LessThan[7,6] = 0 -> Y0Diff[7,6] >= -0
 LinkX1LessThan1[0,1]: X1LessThan[0,1] = 1 -> X1Diff[0,1] <= -1
 LinkX1LessThan1[0,2]: X1LessThan[0,2] = 1 -> X1Diff[0,2] <= -1
 LinkX1LessThan1[0,3]: X1LessThan[0,3] = 1 -> X1Diff[0,3] <= -1
 LinkX1LessThan1[0,4]: X1LessThan[0,4] = 1 -> X1Diff[0,4] <= -1
 LinkX1LessThan1[0,5]: X1LessThan[0,5] = 1 -> X1Diff[0,5] <= -1
 LinkX1LessThan1[0,6]: X1LessThan[0,6] = 1 -> X1Diff[0,6] <= -1
 LinkX1LessThan1[0,7]: X1LessThan[0,7] = 1 -> X1Diff[0,7] <= -1
 LinkX1LessThan1[1,0]: X1LessThan[1,0] = 1 -> X1Diff[1,0] <= -1
 LinkX1LessThan1[1,2]: X1LessThan[1,2] = 1 -> X1Diff[1,2] <= -1
 LinkX1LessThan1[1,3]: X1LessThan[1,3] = 1 -> X1Diff[1,3] <= -1
 LinkX1LessThan1[1,4]: X1LessThan[1,4] = 1 -> X1Diff[1,4] <= -1
 LinkX1LessThan1[1,5]: X1LessThan[1,5] = 1 -> X1Diff[1,5] <= -1
 LinkX1LessThan1[1,6]: X1LessThan[1,6] = 1 -> X1Diff[1,6] <= -1
 LinkX1LessThan1[1,7]: X1LessThan[1,7] = 1 -> X1Diff[1,7] <= -1
 LinkX1LessThan1[2,0]: X1LessThan[2,0] = 1 -> X1Diff[2,0] <= -1
 LinkX1LessThan1[2,1]: X1LessThan[2,1] = 1 -> X1Diff[2,1] <= -1
 LinkX1LessThan1[2,3]: X1LessThan[2,3] = 1 -> X1Diff[2,3] <= -1
 LinkX1LessThan1[2,4]: X1LessThan[2,4] = 1 -> X1Diff[2,4] <= -1
 LinkX1LessThan1[2,5]: X1LessThan[2,5] = 1 -> X1Diff[2,5] <= -1
 LinkX1LessThan1[2,6]: X1LessThan[2,6] = 1 -> X1Diff[2,6] <= -1
 LinkX1LessThan1[2,7]: X1LessThan[2,7] = 1 -> X1Diff[2,7] <= -1
 LinkX1LessThan1[3,0]: X1LessThan[3,0] = 1 -> X1Diff[3,0] <= -1
 LinkX1LessThan1[3,1]: X1LessThan[3,1] = 1 -> X1Diff[3,1] <= -1
 LinkX1LessThan1[3,2]: X1LessThan[3,2] = 1 -> X1Diff[3,2] <= -1
 LinkX1LessThan1[3,4]: X1LessThan[3,4] = 1 -> X1Diff[3,4] <= -1
 LinkX1LessThan1[3,5]: X1LessThan[3,5] = 1 -> X1Diff[3,5] <= -1
 LinkX1LessThan1[3,6]: X1LessThan[3,6] = 1 -> X1Diff[3,6] <= -1
 LinkX1LessThan1[3,7]: X1LessThan[3,7] = 1 -> X1Diff[3,7] <= -1
 LinkX1LessThan1[4,0]: X1LessThan[4,0] = 1 -> X1Diff[4,0] <= -1
 LinkX1LessThan1[4,1]: X1LessThan[4,1] = 1 -> X1Diff[4,1] <= -1
 LinkX1LessThan1[4,2]: X1LessThan[4,2] = 1 -> X1Diff[4,2] <= -1
 LinkX1LessThan1[4,3]: X1LessThan[4,3] = 1 -> X1Diff[4,3] <= -1
 LinkX1LessThan1[4,5]: X1LessThan[4,5] = 1 -> X1Diff[4,5] <= -1
 LinkX1LessThan1[4,6]: X1LessThan[4,6] = 1 -> X1Diff[4,6] <= -1
 LinkX1LessThan1[4,7]: X1LessThan[4,7] = 1 -> X1Diff[4,7] <= -1
 LinkX1LessThan1[5,0]: X1LessThan[5,0] = 1 -> X1Diff[5,0] <= -1
 LinkX1LessThan1[5,1]: X1LessThan[5,1] = 1 -> X1Diff[5,1] <= -1
 LinkX1LessThan1[5,2]: X1LessThan[5,2] = 1 -> X1Diff[5,2] <= -1
 LinkX1LessThan1[5,3]: X1LessThan[5,3] = 1 -> X1Diff[5,3] <= -1
 LinkX1LessThan1[5,4]: X1LessThan[5,4] = 1 -> X1Diff[5,4] <= -1
 LinkX1LessThan1[5,6]: X1LessThan[5,6] = 1 -> X1Diff[5,6] <= -1
 LinkX1LessThan1[5,7]: X1LessThan[5,7] = 1 -> X1Diff[5,7] <= -1
 LinkX1LessThan1[6,0]: X1LessThan[6,0] = 1 -> X1Diff[6,0] <= -1
 LinkX1LessThan1[6,1]: X1LessThan[6,1] = 1 -> X1Diff[6,1] <= -1
 LinkX1LessThan1[6,2]: X1LessThan[6,2] = 1 -> X1Diff[6,2] <= -1
 LinkX1LessThan1[6,3]: X1LessThan[6,3] = 1 -> X1Diff[6,3] <= -1
 LinkX1LessThan1[6,4]: X1LessThan[6,4] = 1 -> X1Diff[6,4] <= -1
 LinkX1LessThan1[6,5]: X1LessThan[6,5] = 1 -> X1Diff[6,5] <= -1
 LinkX1LessThan1[6,7]: X1LessThan[6,7] = 1 -> X1Diff[6,7] <= -1
 LinkX1LessThan1[7,0]: X1LessThan[7,0] = 1 -> X1Diff[7,0] <= -1
 LinkX1LessThan1[7,1]: X1LessThan[7,1] = 1 -> X1Diff[7,1] <= -1
 LinkX1LessThan1[7,2]: X1LessThan[7,2] = 1 -> X1Diff[7,2] <= -1
 LinkX1LessThan1[7,3]: X1LessThan[7,3] = 1 -> X1Diff[7,3] <= -1
 LinkX1LessThan1[7,4]: X1LessThan[7,4] = 1 -> X1Diff[7,4] <= -1
 LinkX1LessThan1[7,5]: X1LessThan[7,5] = 1 -> X1Diff[7,5] <= -1
 LinkX1LessThan1[7,6]: X1LessThan[7,6] = 1 -> X1Diff[7,6] <= -1
 LinkX1LessThan2[0,1]: X1LessThan[0,1] = 0 -> X1Diff[0,1] >= -0
 LinkX1LessThan2[0,2]: X1LessThan[0,2] = 0 -> X1Diff[0,2] >= -0
 LinkX1LessThan2[0,3]: X1LessThan[0,3] = 0 -> X1Diff[0,3] >= -0
 LinkX1LessThan2[0,4]: X1LessThan[0,4] = 0 -> X1Diff[0,4] >= -0
 LinkX1LessThan2[0,5]: X1LessThan[0,5] = 0 -> X1Diff[0,5] >= -0
 LinkX1LessThan2[0,6]: X1LessThan[0,6] = 0 -> X1Diff[0,6] >= -0
 LinkX1LessThan2[0,7]: X1LessThan[0,7] = 0 -> X1Diff[0,7] >= -0
 LinkX1LessThan2[1,0]: X1LessThan[1,0] = 0 -> X1Diff[1,0] >= -0
 LinkX1LessThan2[1,2]: X1LessThan[1,2] = 0 -> X1Diff[1,2] >= -0
 LinkX1LessThan2[1,3]: X1LessThan[1,3] = 0 -> X1Diff[1,3] >= -0
 LinkX1LessThan2[1,4]: X1LessThan[1,4] = 0 -> X1Diff[1,4] >= -0
 LinkX1LessThan2[1,5]: X1LessThan[1,5] = 0 -> X1Diff[1,5] >= -0
 LinkX1LessThan2[1,6]: X1LessThan[1,6] = 0 -> X1Diff[1,6] >= -0
 LinkX1LessThan2[1,7]: X1LessThan[1,7] = 0 -> X1Diff[1,7] >= -0
 LinkX1LessThan2[2,0]: X1LessThan[2,0] = 0 -> X1Diff[2,0] >= -0
 LinkX1LessThan2[2,1]: X1LessThan[2,1] = 0 -> X1Diff[2,1] >= -0
 LinkX1LessThan2[2,3]: X1LessThan[2,3] = 0 -> X1Diff[2,3] >= -0
 LinkX1LessThan2[2,4]: X1LessThan[2,4] = 0 -> X1Diff[2,4] >= -0
 LinkX1LessThan2[2,5]: X1LessThan[2,5] = 0 -> X1Diff[2,5] >= -0
 LinkX1LessThan2[2,6]: X1LessThan[2,6] = 0 -> X1Diff[2,6] >= -0
 LinkX1LessThan2[2,7]: X1LessThan[2,7] = 0 -> X1Diff[2,7] >= -0
 LinkX1LessThan2[3,0]: X1LessThan[3,0] = 0 -> X1Diff[3,0] >= -0
 LinkX1LessThan2[3,1]: X1LessThan[3,1] = 0 -> X1Diff[3,1] >= -0
 LinkX1LessThan2[3,2]: X1LessThan[3,2] = 0 -> X1Diff[3,2] >= -0
 LinkX1LessThan2[3,4]: X1LessThan[3,4] = 0 -> X1Diff[3,4] >= -0
 LinkX1LessThan2[3,5]: X1LessThan[3,5] = 0 -> X1Diff[3,5] >= -0
 LinkX1LessThan2[3,6]: X1LessThan[3,6] = 0 -> X1Diff[3,6] >= -0
 LinkX1LessThan2[3,7]: X1LessThan[3,7] = 0 -> X1Diff[3,7] >= -0
 LinkX1LessThan2[4,0]: X1LessThan[4,0] = 0 -> X1Diff[4,0] >= -0
 LinkX1LessThan2[4,1]: X1LessThan[4,1] = 0 -> X1Diff[4,1] >= -0
 LinkX1LessThan2[4,2]: X1LessThan[4,2] = 0 -> X1Diff[4,2] >= -0
 LinkX1LessThan2[4,3]: X1LessThan[4,3] = 0 -> X1Diff[4,3] >= -0
 LinkX1LessThan2[4,5]: X1LessThan[4,5] = 0 -> X1Diff[4,5] >= -0
 LinkX1LessThan2[4,6]: X1LessThan[4,6] = 0 -> X1Diff[4,6] >= -0
 LinkX1LessThan2[4,7]: X1LessThan[4,7] = 0 -> X1Diff[4,7] >= -0
 LinkX1LessThan2[5,0]: X1LessThan[5,0] = 0 -> X1Diff[5,0] >= -0
 LinkX1LessThan2[5,1]: X1LessThan[5,1] = 0 -> X1Diff[5,1] >= -0
 LinkX1LessThan2[5,2]: X1LessThan[5,2] = 0 -> X1Diff[5,2] >= -0
 LinkX1LessThan2[5,3]: X1LessThan[5,3] = 0 -> X1Diff[5,3] >= -0
 LinkX1LessThan2[5,4]: X1LessThan[5,4] = 0 -> X1Diff[5,4] >= -0
 LinkX1LessThan2[5,6]: X1LessThan[5,6] = 0 -> X1Diff[5,6] >= -0
 LinkX1LessThan2[5,7]: X1LessThan[5,7] = 0 -> X1Diff[5,7] >= -0
 LinkX1LessThan2[6,0]: X1LessThan[6,0] = 0 -> X1Diff[6,0] >= -0
 LinkX1LessThan2[6,1]: X1LessThan[6,1] = 0 -> X1Diff[6,1] >= -0
 LinkX1LessThan2[6,2]: X1LessThan[6,2] = 0 -> X1Diff[6,2] >= -0
 LinkX1LessThan2[6,3]: X1LessThan[6,3] = 0 -> X1Diff[6,3] >= -0
 LinkX1LessThan2[6,4]: X1LessThan[6,4] = 0 -> X1Diff[6,4] >= -0
 LinkX1LessThan2[6,5]: X1LessThan[6,5] = 0 -> X1Diff[6,5] >= -0
 LinkX1LessThan2[6,7]: X1LessThan[6,7] = 0 -> X1Diff[6,7] >= -0
 LinkX1LessThan2[7,0]: X1LessThan[7,0] = 0 -> X1Diff[7,0] >= -0
 LinkX1LessThan2[7,1]: X1LessThan[7,1] = 0 -> X1Diff[7,1] >= -0
 LinkX1LessThan2[7,2]: X1LessThan[7,2] = 0 -> X1Diff[7,2] >= -0
 LinkX1LessThan2[7,3]: X1LessThan[7,3] = 0 -> X1Diff[7,3] >= -0
 LinkX1LessThan2[7,4]: X1LessThan[7,4] = 0 -> X1Diff[7,4] >= -0
 LinkX1LessThan2[7,5]: X1LessThan[7,5] = 0 -> X1Diff[7,5] >= -0
 LinkX1LessThan2[7,6]: X1LessThan[7,6] = 0 -> X1Diff[7,6] >= -0
 LinkY1LessThan1[0,1]: Y1LessThan[0,1] = 1 -> Y1Diff[0,1] <= -1
 LinkY1LessThan1[0,2]: Y1LessThan[0,2] = 1 -> Y1Diff[0,2] <= -1
 LinkY1LessThan1[0,3]: Y1LessThan[0,3] = 1 -> Y1Diff[0,3] <= -1
 LinkY1LessThan1[0,4]: Y1LessThan[0,4] = 1 -> Y1Diff[0,4] <= -1
 LinkY1LessThan1[0,5]: Y1LessThan[0,5] = 1 -> Y1Diff[0,5] <= -1
 LinkY1LessThan1[0,6]: Y1LessThan[0,6] = 1 -> Y1Diff[0,6] <= -1
 LinkY1LessThan1[0,7]: Y1LessThan[0,7] = 1 -> Y1Diff[0,7] <= -1
 LinkY1LessThan1[1,0]: Y1LessThan[1,0] = 1 -> Y1Diff[1,0] <= -1
 LinkY1LessThan1[1,2]: Y1LessThan[1,2] = 1 -> Y1Diff[1,2] <= -1
 LinkY1LessThan1[1,3]: Y1LessThan[1,3] = 1 -> Y1Diff[1,3] <= -1
 LinkY1LessThan1[1,4]: Y1LessThan[1,4] = 1 -> Y1Diff[1,4] <= -1
 LinkY1LessThan1[1,5]: Y1LessThan[1,5] = 1 -> Y1Diff[1,5] <= -1
 LinkY1LessThan1[1,6]: Y1LessThan[1,6] = 1 -> Y1Diff[1,6] <= -1
 LinkY1LessThan1[1,7]: Y1LessThan[1,7] = 1 -> Y1Diff[1,7] <= -1
 LinkY1LessThan1[2,0]: Y1LessThan[2,0] = 1 -> Y1Diff[2,0] <= -1
 LinkY1LessThan1[2,1]: Y1LessThan[2,1] = 1 -> Y1Diff[2,1] <= -1
 LinkY1LessThan1[2,3]: Y1LessThan[2,3] = 1 -> Y1Diff[2,3] <= -1
 LinkY1LessThan1[2,4]: Y1LessThan[2,4] = 1 -> Y1Diff[2,4] <= -1
 LinkY1LessThan1[2,5]: Y1LessThan[2,5] = 1 -> Y1Diff[2,5] <= -1
 LinkY1LessThan1[2,6]: Y1LessThan[2,6] = 1 -> Y1Diff[2,6] <= -1
 LinkY1LessThan1[2,7]: Y1LessThan[2,7] = 1 -> Y1Diff[2,7] <= -1
 LinkY1LessThan1[3,0]: Y1LessThan[3,0] = 1 -> Y1Diff[3,0] <= -1
 LinkY1LessThan1[3,1]: Y1LessThan[3,1] = 1 -> Y1Diff[3,1] <= -1
 LinkY1LessThan1[3,2]: Y1LessThan[3,2] = 1 -> Y1Diff[3,2] <= -1
 LinkY1LessThan1[3,4]: Y1LessThan[3,4] = 1 -> Y1Diff[3,4] <= -1
 LinkY1LessThan1[3,5]: Y1LessThan[3,5] = 1 -> Y1Diff[3,5] <= -1
 LinkY1LessThan1[3,6]: Y1LessThan[3,6] = 1 -> Y1Diff[3,6] <= -1
 LinkY1LessThan1[3,7]: Y1LessThan[3,7] = 1 -> Y1Diff[3,7] <= -1
 LinkY1LessThan1[4,0]: Y1LessThan[4,0] = 1 -> Y1Diff[4,0] <= -1
 LinkY1LessThan1[4,1]: Y1LessThan[4,1] = 1 -> Y1Diff[4,1] <= -1
 LinkY1LessThan1[4,2]: Y1LessThan[4,2] = 1 -> Y1Diff[4,2] <= -1
 LinkY1LessThan1[4,3]: Y1LessThan[4,3] = 1 -> Y1Diff[4,3] <= -1
 LinkY1LessThan1[4,5]: Y1LessThan[4,5] = 1 -> Y1Diff[4,5] <= -1
 LinkY1LessThan1[4,6]: Y1LessThan[4,6] = 1 -> Y1Diff[4,6] <= -1
 LinkY1LessThan1[4,7]: Y1LessThan[4,7] = 1 -> Y1Diff[4,7] <= -1
 LinkY1LessThan1[5,0]: Y1LessThan[5,0] = 1 -> Y1Diff[5,0] <= -1
 LinkY1LessThan1[5,1]: Y1LessThan[5,1] = 1 -> Y1Diff[5,1] <= -1
 LinkY1LessThan1[5,2]: Y1LessThan[5,2] = 1 -> Y1Diff[5,2] <= -1
 LinkY1LessThan1[5,3]: Y1LessThan[5,3] = 1 -> Y1Diff[5,3] <= -1
 LinkY1LessThan1[5,4]: Y1LessThan[5,4] = 1 -> Y1Diff[5,4] <= -1
 LinkY1LessThan1[5,6]: Y1LessThan[5,6] = 1 -> Y1Diff[5,6] <= -1
 LinkY1LessThan1[5,7]: Y1LessThan[5,7] = 1 -> Y1Diff[5,7] <= -1
 LinkY1LessThan1[6,0]: Y1LessThan[6,0] = 1 -> Y1Diff[6,0] <= -1
 LinkY1LessThan1[6,1]: Y1LessThan[6,1] = 1 -> Y1Diff[6,1] <= -1
 LinkY1LessThan1[6,2]: Y1LessThan[6,2] = 1 -> Y1Diff[6,2] <= -1
 LinkY1LessThan1[6,3]: Y1LessThan[6,3] = 1 -> Y1Diff[6,3] <= -1
 LinkY1LessThan1[6,4]: Y1LessThan[6,4] = 1 -> Y1Diff[6,4] <= -1
 LinkY1LessThan1[6,5]: Y1LessThan[6,5] = 1 -> Y1Diff[6,5] <= -1
 LinkY1LessThan1[6,7]: Y1LessThan[6,7] = 1 -> Y1Diff[6,7] <= -1
 LinkY1LessThan1[7,0]: Y1LessThan[7,0] = 1 -> Y1Diff[7,0] <= -1
 LinkY1LessThan1[7,1]: Y1LessThan[7,1] = 1 -> Y1Diff[7,1] <= -1
 LinkY1LessThan1[7,2]: Y1LessThan[7,2] = 1 -> Y1Diff[7,2] <= -1
 LinkY1LessThan1[7,3]: Y1LessThan[7,3] = 1 -> Y1Diff[7,3] <= -1
 LinkY1LessThan1[7,4]: Y1LessThan[7,4] = 1 -> Y1Diff[7,4] <= -1
 LinkY1LessThan1[7,5]: Y1LessThan[7,5] = 1 -> Y1Diff[7,5] <= -1
 LinkY1LessThan1[7,6]: Y1LessThan[7,6] = 1 -> Y1Diff[7,6] <= -1
 LinkY1LessThan2[0,1]: Y1LessThan[0,1] = 0 -> Y1Diff[0,1] >= -0
 LinkY1LessThan2[0,2]: Y1LessThan[0,2] = 0 -> Y1Diff[0,2] >= -0
 LinkY1LessThan2[0,3]: Y1LessThan[0,3] = 0 -> Y1Diff[0,3] >= -0
 LinkY1LessThan2[0,4]: Y1LessThan[0,4] = 0 -> Y1Diff[0,4] >= -0
 LinkY1LessThan2[0,5]: Y1LessThan[0,5] = 0 -> Y1Diff[0,5] >= -0
 LinkY1LessThan2[0,6]: Y1LessThan[0,6] = 0 -> Y1Diff[0,6] >= -0
 LinkY1LessThan2[0,7]: Y1LessThan[0,7] = 0 -> Y1Diff[0,7] >= -0
 LinkY1LessThan2[1,0]: Y1LessThan[1,0] = 0 -> Y1Diff[1,0] >= -0
 LinkY1LessThan2[1,2]: Y1LessThan[1,2] = 0 -> Y1Diff[1,2] >= -0
 LinkY1LessThan2[1,3]: Y1LessThan[1,3] = 0 -> Y1Diff[1,3] >= -0
 LinkY1LessThan2[1,4]: Y1LessThan[1,4] = 0 -> Y1Diff[1,4] >= -0
 LinkY1LessThan2[1,5]: Y1LessThan[1,5] = 0 -> Y1Diff[1,5] >= -0
 LinkY1LessThan2[1,6]: Y1LessThan[1,6] = 0 -> Y1Diff[1,6] >= -0
 LinkY1LessThan2[1,7]: Y1LessThan[1,7] = 0 -> Y1Diff[1,7] >= -0
 LinkY1LessThan2[2,0]: Y1LessThan[2,0] = 0 -> Y1Diff[2,0] >= -0
 LinkY1LessThan2[2,1]: Y1LessThan[2,1] = 0 -> Y1Diff[2,1] >= -0
 LinkY1LessThan2[2,3]: Y1LessThan[2,3] = 0 -> Y1Diff[2,3] >= -0
 LinkY1LessThan2[2,4]: Y1LessThan[2,4] = 0 -> Y1Diff[2,4] >= -0
 LinkY1LessThan2[2,5]: Y1LessThan[2,5] = 0 -> Y1Diff[2,5] >= -0
 LinkY1LessThan2[2,6]: Y1LessThan[2,6] = 0 -> Y1Diff[2,6] >= -0
 LinkY1LessThan2[2,7]: Y1LessThan[2,7] = 0 -> Y1Diff[2,7] >= -0
 LinkY1LessThan2[3,0]: Y1LessThan[3,0] = 0 -> Y1Diff[3,0] >= -0
 LinkY1LessThan2[3,1]: Y1LessThan[3,1] = 0 -> Y1Diff[3,1] >= -0
 LinkY1LessThan2[3,2]: Y1LessThan[3,2] = 0 -> Y1Diff[3,2] >= -0
 LinkY1LessThan2[3,4]: Y1LessThan[3,4] = 0 -> Y1Diff[3,4] >= -0
 LinkY1LessThan2[3,5]: Y1LessThan[3,5] = 0 -> Y1Diff[3,5] >= -0
 LinkY1LessThan2[3,6]: Y1LessThan[3,6] = 0 -> Y1Diff[3,6] >= -0
 LinkY1LessThan2[3,7]: Y1LessThan[3,7] = 0 -> Y1Diff[3,7] >= -0
 LinkY1LessThan2[4,0]: Y1LessThan[4,0] = 0 -> Y1Diff[4,0] >= -0
 LinkY1LessThan2[4,1]: Y1LessThan[4,1] = 0 -> Y1Diff[4,1] >= -0
 LinkY1LessThan2[4,2]: Y1LessThan[4,2] = 0 -> Y1Diff[4,2] >= -0
 LinkY1LessThan2[4,3]: Y1LessThan[4,3] = 0 -> Y1Diff[4,3] >= -0
 LinkY1LessThan2[4,5]: Y1LessThan[4,5] = 0 -> Y1Diff[4,5] >= -0
 LinkY1LessThan2[4,6]: Y1LessThan[4,6] = 0 -> Y1Diff[4,6] >= -0
 LinkY1LessThan2[4,7]: Y1LessThan[4,7] = 0 -> Y1Diff[4,7] >= -0
 LinkY1LessThan2[5,0]: Y1LessThan[5,0] = 0 -> Y1Diff[5,0] >= -0
 LinkY1LessThan2[5,1]: Y1LessThan[5,1] = 0 -> Y1Diff[5,1] >= -0
 LinkY1LessThan2[5,2]: Y1LessThan[5,2] = 0 -> Y1Diff[5,2] >= -0
 LinkY1LessThan2[5,3]: Y1LessThan[5,3] = 0 -> Y1Diff[5,3] >= -0
 LinkY1LessThan2[5,4]: Y1LessThan[5,4] = 0 -> Y1Diff[5,4] >= -0
 LinkY1LessThan2[5,6]: Y1LessThan[5,6] = 0 -> Y1Diff[5,6] >= -0
 LinkY1LessThan2[5,7]: Y1LessThan[5,7] = 0 -> Y1Diff[5,7] >= -0
 LinkY1LessThan2[6,0]: Y1LessThan[6,0] = 0 -> Y1Diff[6,0] >= -0
 LinkY1LessThan2[6,1]: Y1LessThan[6,1] = 0 -> Y1Diff[6,1] >= -0
 LinkY1LessThan2[6,2]: Y1LessThan[6,2] = 0 -> Y1Diff[6,2] >= -0
 LinkY1LessThan2[6,3]: Y1LessThan[6,3] = 0 -> Y1Diff[6,3] >= -0
 LinkY1LessThan2[6,4]: Y1LessThan[6,4] = 0 -> Y1Diff[6,4] >= -0
 LinkY1LessThan2[6,5]: Y1LessThan[6,5] = 0 -> Y1Diff[6,5] >= -0
 LinkY1LessThan2[6,7]: Y1LessThan[6,7] = 0 -> Y1Diff[6,7] >= -0
 LinkY1LessThan2[7,0]: Y1LessThan[7,0] = 0 -> Y1Diff[7,0] >= -0
 LinkY1LessThan2[7,1]: Y1LessThan[7,1] = 0 -> Y1Diff[7,1] >= -0
 LinkY1LessThan2[7,2]: Y1LessThan[7,2] = 0 -> Y1Diff[7,2] >= -0
 LinkY1LessThan2[7,3]: Y1LessThan[7,3] = 0 -> Y1Diff[7,3] >= -0
 LinkY1LessThan2[7,4]: Y1LessThan[7,4] = 0 -> Y1Diff[7,4] >= -0
 LinkY1LessThan2[7,5]: Y1LessThan[7,5] = 0 -> Y1Diff[7,5] >= -0
 LinkY1LessThan2[7,6]: Y1LessThan[7,6] = 0 -> Y1Diff[7,6] >= -0
 LinkX0Group1[0,1]: X0LessThan[0,1] = 1 -> X0Group[0] - X0Group[1] <= -1
 LinkX0Group1[0,2]: X0LessThan[0,2] = 1 -> X0Group[0] - X0Group[2] <= -1
 LinkX0Group1[0,3]: X0LessThan[0,3] = 1 -> X0Group[0] - X0Group[3] <= -1
 LinkX0Group1[0,4]: X0LessThan[0,4] = 1 -> X0Group[0] - X0Group[4] <= -1
 LinkX0Group1[0,5]: X0LessThan[0,5] = 1 -> X0Group[0] - X0Group[5] <= -1
 LinkX0Group1[0,6]: X0LessThan[0,6] = 1 -> X0Group[0] - X0Group[6] <= -1
 LinkX0Group1[0,7]: X0LessThan[0,7] = 1 -> X0Group[0] - X0Group[7] <= -1
 LinkX0Group1[1,0]: X0LessThan[1,0] = 1 -> X0Group[1] - X0Group[0] <= -1
 LinkX0Group1[1,2]: X0LessThan[1,2] = 1 -> X0Group[1] - X0Group[2] <= -1
 LinkX0Group1[1,3]: X0LessThan[1,3] = 1 -> X0Group[1] - X0Group[3] <= -1
 LinkX0Group1[1,4]: X0LessThan[1,4] = 1 -> X0Group[1] - X0Group[4] <= -1
 LinkX0Group1[1,5]: X0LessThan[1,5] = 1 -> X0Group[1] - X0Group[5] <= -1
 LinkX0Group1[1,6]: X0LessThan[1,6] = 1 -> X0Group[1] - X0Group[6] <= -1
 LinkX0Group1[1,7]: X0LessThan[1,7] = 1 -> X0Group[1] - X0Group[7] <= -1
 LinkX0Group1[2,0]: X0LessThan[2,0] = 1 -> X0Group[2] - X0Group[0] <= -1
 LinkX0Group1[2,1]: X0LessThan[2,1] = 1 -> X0Group[2] - X0Group[1] <= -1
 LinkX0Group1[2,3]: X0LessThan[2,3] = 1 -> X0Group[2] - X0Group[3] <= -1
 LinkX0Group1[2,4]: X0LessThan[2,4] = 1 -> X0Group[2] - X0Group[4] <= -1
 LinkX0Group1[2,5]: X0LessThan[2,5] = 1 -> X0Group[2] - X0Group[5] <= -1
 LinkX0Group1[2,6]: X0LessThan[2,6] = 1 -> X0Group[2] - X0Group[6] <= -1
 LinkX0Group1[2,7]: X0LessThan[2,7] = 1 -> X0Group[2] - X0Group[7] <= -1
 LinkX0Group1[3,0]: X0LessThan[3,0] = 1 -> X0Group[3] - X0Group[0] <= -1
 LinkX0Group1[3,1]: X0LessThan[3,1] = 1 -> X0Group[3] - X0Group[1] <= -1
 LinkX0Group1[3,2]: X0LessThan[3,2] = 1 -> X0Group[3] - X0Group[2] <= -1
 LinkX0Group1[3,4]: X0LessThan[3,4] = 1 -> X0Group[3] - X0Group[4] <= -1
 LinkX0Group1[3,5]: X0LessThan[3,5] = 1 -> X0Group[3] - X0Group[5] <= -1
 LinkX0Group1[3,6]: X0LessThan[3,6] = 1 -> X0Group[3] - X0Group[6] <= -1
 LinkX0Group1[3,7]: X0LessThan[3,7] = 1 -> X0Group[3] - X0Group[7] <= -1
 LinkX0Group1[4,0]: X0LessThan[4,0] = 1 -> X0Group[4] - X0Group[0] <= -1
 LinkX0Group1[4,1]: X0LessThan[4,1] = 1 -> X0Group[4] - X0Group[1] <= -1
 LinkX0Group1[4,2]: X0LessThan[4,2] = 1 -> X0Group[4] - X0Group[2] <= -1
 LinkX0Group1[4,3]: X0LessThan[4,3] = 1 -> X0Group[4] - X0Group[3] <= -1
 LinkX0Group1[4,5]: X0LessThan[4,5] = 1 -> X0Group[4] - X0Group[5] <= -1
 LinkX0Group1[4,6]: X0LessThan[4,6] = 1 -> X0Group[4] - X0Group[6] <= -1
 LinkX0Group1[4,7]: X0LessThan[4,7] = 1 -> X0Group[4] - X0Group[7] <= -1
 LinkX0Group1[5,0]: X0LessThan[5,0] = 1 -> X0Group[5] - X0Group[0] <= -1
 LinkX0Group1[5,1]: X0LessThan[5,1] = 1 -> X0Group[5] - X0Group[1] <= -1
 LinkX0Group1[5,2]: X0LessThan[5,2] = 1 -> X0Group[5] - X0Group[2] <= -1
 LinkX0Group1[5,3]: X0LessThan[5,3] = 1 -> X0Group[5] - X0Group[3] <= -1
 LinkX0Group1[5,4]: X0LessThan[5,4] = 1 -> X0Group[5] - X0Group[4] <= -1
 LinkX0Group1[5,6]: X0LessThan[5,6] = 1 -> X0Group[5] - X0Group[6] <= -1
 LinkX0Group1[5,7]: X0LessThan[5,7] = 1 -> X0Group[5] - X0Group[7] <= -1
 LinkX0Group1[6,0]: X0LessThan[6,0] = 1 -> X0Group[6] - X0Group[0] <= -1
 LinkX0Group1[6,1]: X0LessThan[6,1] = 1 -> X0Group[6] - X0Group[1] <= -1
 LinkX0Group1[6,2]: X0LessThan[6,2] = 1 -> X0Group[6] - X0Group[2] <= -1
 LinkX0Group1[6,3]: X0LessThan[6,3] = 1 -> X0Group[6] - X0Group[3] <= -1
 LinkX0Group1[6,4]: X0LessThan[6,4] = 1 -> X0Group[6] - X0Group[4] <= -1
 LinkX0Group1[6,5]: X0LessThan[6,5] = 1 -> X0Group[6] - X0Group[5] <= -1
 LinkX0Group1[6,7]: X0LessThan[6,7] = 1 -> X0Group[6] - X0Group[7] <= -1
 LinkX0Group1[7,0]: X0LessThan[7,0] = 1 -> X0Group[7] - X0Group[0] <= -1
 LinkX0Group1[7,1]: X0LessThan[7,1] = 1 -> X0Group[7] - X0Group[1] <= -1
 LinkX0Group1[7,2]: X0LessThan[7,2] = 1 -> X0Group[7] - X0Group[2] <= -1
 LinkX0Group1[7,3]: X0LessThan[7,3] = 1 -> X0Group[7] - X0Group[3] <= -1
 LinkX0Group1[7,4]: X0LessThan[7,4] = 1 -> X0Group[7] - X0Group[4] <= -1
 LinkX0Group1[7,5]: X0LessThan[7,5] = 1 -> X0Group[7] - X0Group[5] <= -1
 LinkX0Group1[7,6]: X0LessThan[7,6] = 1 -> X0Group[7] - X0Group[6] <= -1
 LinkX0Group2[0,1]: X0LessThan[0,1] = 0 -> X0Group[0] - X0Group[1] >= -0
 LinkX0Group2[0,2]: X0LessThan[0,2] = 0 -> X0Group[0] - X0Group[2] >= -0
 LinkX0Group2[0,3]: X0LessThan[0,3] = 0 -> X0Group[0] - X0Group[3] >= -0
 LinkX0Group2[0,4]: X0LessThan[0,4] = 0 -> X0Group[0] - X0Group[4] >= -0
 LinkX0Group2[0,5]: X0LessThan[0,5] = 0 -> X0Group[0] - X0Group[5] >= -0
 LinkX0Group2[0,6]: X0LessThan[0,6] = 0 -> X0Group[0] - X0Group[6] >= -0
 LinkX0Group2[0,7]: X0LessThan[0,7] = 0 -> X0Group[0] - X0Group[7] >= -0
 LinkX0Group2[1,0]: X0LessThan[1,0] = 0 -> X0Group[1] - X0Group[0] >= -0
 LinkX0Group2[1,2]: X0LessThan[1,2] = 0 -> X0Group[1] - X0Group[2] >= -0
 LinkX0Group2[1,3]: X0LessThan[1,3] = 0 -> X0Group[1] - X0Group[3] >= -0
 LinkX0Group2[1,4]: X0LessThan[1,4] = 0 -> X0Group[1] - X0Group[4] >= -0
 LinkX0Group2[1,5]: X0LessThan[1,5] = 0 -> X0Group[1] - X0Group[5] >= -0
 LinkX0Group2[1,6]: X0LessThan[1,6] = 0 -> X0Group[1] - X0Group[6] >= -0
 LinkX0Group2[1,7]: X0LessThan[1,7] = 0 -> X0Group[1] - X0Group[7] >= -0
 LinkX0Group2[2,0]: X0LessThan[2,0] = 0 -> X0Group[2] - X0Group[0] >= -0
 LinkX0Group2[2,1]: X0LessThan[2,1] = 0 -> X0Group[2] - X0Group[1] >= -0
 LinkX0Group2[2,3]: X0LessThan[2,3] = 0 -> X0Group[2] - X0Group[3] >= -0
 LinkX0Group2[2,4]: X0LessThan[2,4] = 0 -> X0Group[2] - X0Group[4] >= -0
 LinkX0Group2[2,5]: X0LessThan[2,5] = 0 -> X0Group[2] - X0Group[5] >= -0
 LinkX0Group2[2,6]: X0LessThan[2,6] = 0 -> X0Group[2] - X0Group[6] >= -0
 LinkX0Group2[2,7]: X0LessThan[2,7] = 0 -> X0Group[2] - X0Group[7] >= -0
 LinkX0Group2[3,0]: X0LessThan[3,0] = 0 -> X0Group[3] - X0Group[0] >= -0
 LinkX0Group2[3,1]: X0LessThan[3,1] = 0 -> X0Group[3] - X0Group[1] >= -0
 LinkX0Group2[3,2]: X0LessThan[3,2] = 0 -> X0Group[3] - X0Group[2] >= -0
 LinkX0Group2[3,4]: X0LessThan[3,4] = 0 -> X0Group[3] - X0Group[4] >= -0
 LinkX0Group2[3,5]: X0LessThan[3,5] = 0 -> X0Group[3] - X0Group[5] >= -0
 LinkX0Group2[3,6]: X0LessThan[3,6] = 0 -> X0Group[3] - X0Group[6] >= -0
 LinkX0Group2[3,7]: X0LessThan[3,7] = 0 -> X0Group[3] - X0Group[7] >= -0
 LinkX0Group2[4,0]: X0LessThan[4,0] = 0 -> X0Group[4] - X0Group[0] >= -0
 LinkX0Group2[4,1]: X0LessThan[4,1] = 0 -> X0Group[4] - X0Group[1] >= -0
 LinkX0Group2[4,2]: X0LessThan[4,2] = 0 -> X0Group[4] - X0Group[2] >= -0
 LinkX0Group2[4,3]: X0LessThan[4,3] = 0 -> X0Group[4] - X0Group[3] >= -0
 LinkX0Group2[4,5]: X0LessThan[4,5] = 0 -> X0Group[4] - X0Group[5] >= -0
 LinkX0Group2[4,6]: X0LessThan[4,6] = 0 -> X0Group[4] - X0Group[6] >= -0
 LinkX0Group2[4,7]: X0LessThan[4,7] = 0 -> X0Group[4] - X0Group[7] >= -0
 LinkX0Group2[5,0]: X0LessThan[5,0] = 0 -> X0Group[5] - X0Group[0] >= -0
 LinkX0Group2[5,1]: X0LessThan[5,1] = 0 -> X0Group[5] - X0Group[1] >= -0
 LinkX0Group2[5,2]: X0LessThan[5,2] = 0 -> X0Group[5] - X0Group[2] >= -0
 LinkX0Group2[5,3]: X0LessThan[5,3] = 0 -> X0Group[5] - X0Group[3] >= -0
 LinkX0Group2[5,4]: X0LessThan[5,4] = 0 -> X0Group[5] - X0Group[4] >= -0
 LinkX0Group2[5,6]: X0LessThan[5,6] = 0 -> X0Group[5] - X0Group[6] >= -0
 LinkX0Group2[5,7]: X0LessThan[5,7] = 0 -> X0Group[5] - X0Group[7] >= -0
 LinkX0Group2[6,0]: X0LessThan[6,0] = 0 -> X0Group[6] - X0Group[0] >= -0
 LinkX0Group2[6,1]: X0LessThan[6,1] = 0 -> X0Group[6] - X0Group[1] >= -0
 LinkX0Group2[6,2]: X0LessThan[6,2] = 0 -> X0Group[6] - X0Group[2] >= -0
 LinkX0Group2[6,3]: X0LessThan[6,3] = 0 -> X0Group[6] - X0Group[3] >= -0
 LinkX0Group2[6,4]: X0LessThan[6,4] = 0 -> X0Group[6] - X0Group[4] >= -0
 LinkX0Group2[6,5]: X0LessThan[6,5] = 0 -> X0Group[6] - X0Group[5] >= -0
 LinkX0Group2[6,7]: X0LessThan[6,7] = 0 -> X0Group[6] - X0Group[7] >= -0
 LinkX0Group2[7,0]: X0LessThan[7,0] = 0 -> X0Group[7] - X0Group[0] >= -0
 LinkX0Group2[7,1]: X0LessThan[7,1] = 0 -> X0Group[7] - X0Group[1] >= -0
 LinkX0Group2[7,2]: X0LessThan[7,2] = 0 -> X0Group[7] - X0Group[2] >= -0
 LinkX0Group2[7,3]: X0LessThan[7,3] = 0 -> X0Group[7] - X0Group[3] >= -0
 LinkX0Group2[7,4]: X0LessThan[7,4] = 0 -> X0Group[7] - X0Group[4] >= -0
 LinkX0Group2[7,5]: X0LessThan[7,5] = 0 -> X0Group[7] - X0Group[5] >= -0
 LinkX0Group2[7,6]: X0LessThan[7,6] = 0 -> X0Group[7] - X0Group[6] >= -0
 LinkY0Group1[0,1]: Y0LessThan[0,1] = 1 -> Y0Group[0] - Y0Group[1] <= -1
 LinkY0Group1[0,2]: Y0LessThan[0,2] = 1 -> Y0Group[0] - Y0Group[2] <= -1
 LinkY0Group1[0,3]: Y0LessThan[0,3] = 1 -> Y0Group[0] - Y0Group[3] <= -1
 LinkY0Group1[0,4]: Y0LessThan[0,4] = 1 -> Y0Group[0] - Y0Group[4] <= -1
 LinkY0Group1[0,5]: Y0LessThan[0,5] = 1 -> Y0Group[0] - Y0Group[5] <= -1
 LinkY0Group1[0,6]: Y0LessThan[0,6] = 1 -> Y0Group[0] - Y0Group[6] <= -1
 LinkY0Group1[0,7]: Y0LessThan[0,7] = 1 -> Y0Group[0] - Y0Group[7] <= -1
 LinkY0Group1[1,0]: Y0LessThan[1,0] = 1 -> Y0Group[1] - Y0Group[0] <= -1
 LinkY0Group1[1,2]: Y0LessThan[1,2] = 1 -> Y0Group[1] - Y0Group[2] <= -1
 LinkY0Group1[1,3]: Y0LessThan[1,3] = 1 -> Y0Group[1] - Y0Group[3] <= -1
 LinkY0Group1[1,4]: Y0LessThan[1,4] = 1 -> Y0Group[1] - Y0Group[4] <= -1
 LinkY0Group1[1,5]: Y0LessThan[1,5] = 1 -> Y0Group[1] - Y0Group[5] <= -1
 LinkY0Group1[1,6]: Y0LessThan[1,6] = 1 -> Y0Group[1] - Y0Group[6] <= -1
 LinkY0Group1[1,7]: Y0LessThan[1,7] = 1 -> Y0Group[1] - Y0Group[7] <= -1
 LinkY0Group1[2,0]: Y0LessThan[2,0] = 1 -> Y0Group[2] - Y0Group[0] <= -1
 LinkY0Group1[2,1]: Y0LessThan[2,1] = 1 -> Y0Group[2] - Y0Group[1] <= -1
 LinkY0Group1[2,3]: Y0LessThan[2,3] = 1 -> Y0Group[2] - Y0Group[3] <= -1
 LinkY0Group1[2,4]: Y0LessThan[2,4] = 1 -> Y0Group[2] - Y0Group[4] <= -1
 LinkY0Group1[2,5]: Y0LessThan[2,5] = 1 -> Y0Group[2] - Y0Group[5] <= -1
 LinkY0Group1[2,6]: Y0LessThan[2,6] = 1 -> Y0Group[2] - Y0Group[6] <= -1
 LinkY0Group1[2,7]: Y0LessThan[2,7] = 1 -> Y0Group[2] - Y0Group[7] <= -1
 LinkY0Group1[3,0]: Y0LessThan[3,0] = 1 -> Y0Group[3] - Y0Group[0] <= -1
 LinkY0Group1[3,1]: Y0LessThan[3,1] = 1 -> Y0Group[3] - Y0Group[1] <= -1
 LinkY0Group1[3,2]: Y0LessThan[3,2] = 1 -> Y0Group[3] - Y0Group[2] <= -1
 LinkY0Group1[3,4]: Y0LessThan[3,4] = 1 -> Y0Group[3] - Y0Group[4] <= -1
 LinkY0Group1[3,5]: Y0LessThan[3,5] = 1 -> Y0Group[3] - Y0Group[5] <= -1
 LinkY0Group1[3,6]: Y0LessThan[3,6] = 1 -> Y0Group[3] - Y0Group[6] <= -1
 LinkY0Group1[3,7]: Y0LessThan[3,7] = 1 -> Y0Group[3] - Y0Group[7] <= -1
 LinkY0Group1[4,0]: Y0LessThan[4,0] = 1 -> Y0Group[4] - Y0Group[0] <= -1
 LinkY0Group1[4,1]: Y0LessThan[4,1] = 1 -> Y0Group[4] - Y0Group[1] <= -1
 LinkY0Group1[4,2]: Y0LessThan[4,2] = 1 -> Y0Group[4] - Y0Group[2] <= -1
 LinkY0Group1[4,3]: Y0LessThan[4,3] = 1 -> Y0Group[4] - Y0Group[3] <= -1
 LinkY0Group1[4,5]: Y0LessThan[4,5] = 1 -> Y0Group[4] - Y0Group[5] <= -1
 LinkY0Group1[4,6]: Y0LessThan[4,6] = 1 -> Y0Group[4] - Y0Group[6] <= -1
 LinkY0Group1[4,7]: Y0LessThan[4,7] = 1 -> Y0Group[4] - Y0Group[7] <= -1
 LinkY0Group1[5,0]: Y0LessThan[5,0] = 1 -> Y0Group[5] - Y0Group[0] <= -1
 LinkY0Group1[5,1]: Y0LessThan[5,1] = 1 -> Y0Group[5] - Y0Group[1] <= -1
 LinkY0Group1[5,2]: Y0LessThan[5,2] = 1 -> Y0Group[5] - Y0Group[2] <= -1
 LinkY0Group1[5,3]: Y0LessThan[5,3] = 1 -> Y0Group[5] - Y0Group[3] <= -1
 LinkY0Group1[5,4]: Y0LessThan[5,4] = 1 -> Y0Group[5] - Y0Group[4] <= -1
 LinkY0Group1[5,6]: Y0LessThan[5,6] = 1 -> Y0Group[5] - Y0Group[6] <= -1
 LinkY0Group1[5,7]: Y0LessThan[5,7] = 1 -> Y0Group[5] - Y0Group[7] <= -1
 LinkY0Group1[6,0]: Y0LessThan[6,0] = 1 -> Y0Group[6] - Y0Group[0] <= -1
 LinkY0Group1[6,1]: Y0LessThan[6,1] = 1 -> Y0Group[6] - Y0Group[1] <= -1
 LinkY0Group1[6,2]: Y0LessThan[6,2] = 1 -> Y0Group[6] - Y0Group[2] <= -1
 LinkY0Group1[6,3]: Y0LessThan[6,3] = 1 -> Y0Group[6] - Y0Group[3] <= -1
 LinkY0Group1[6,4]: Y0LessThan[6,4] = 1 -> Y0Group[6] - Y0Group[4] <= -1
 LinkY0Group1[6,5]: Y0LessThan[6,5] = 1 -> Y0Group[6] - Y0Group[5] <= -1
 LinkY0Group1[6,7]: Y0LessThan[6,7] = 1 -> Y0Group[6] - Y0Group[7] <= -1
 LinkY0Group1[7,0]: Y0LessThan[7,0] = 1 -> Y0Group[7] - Y0Group[0] <= -1
 LinkY0Group1[7,1]: Y0LessThan[7,1] = 1 -> Y0Group[7] - Y0Group[1] <= -1
 LinkY0Group1[7,2]: Y0LessThan[7,2] = 1 -> Y0Group[7] - Y0Group[2] <= -1
 LinkY0Group1[7,3]: Y0LessThan[7,3] = 1 -> Y0Group[7] - Y0Group[3] <= -1
 LinkY0Group1[7,4]: Y0LessThan[7,4] = 1 -> Y0Group[7] - Y0Group[4] <= -1
 LinkY0Group1[7,5]: Y0LessThan[7,5] = 1 -> Y0Group[7] - Y0Group[5] <= -1
 LinkY0Group1[7,6]: Y0LessThan[7,6] = 1 -> Y0Group[7] - Y0Group[6] <= -1
 LinkY0Group2[0,1]: Y0LessThan[0,1] = 0 -> Y0Group[0] - Y0Group[1] >= -0
 LinkY0Group2[0,2]: Y0LessThan[0,2] = 0 -> Y0Group[0] - Y0Group[2] >= -0
 LinkY0Group2[0,3]: Y0LessThan[0,3] = 0 -> Y0Group[0] - Y0Group[3] >= -0
 LinkY0Group2[0,4]: Y0LessThan[0,4] = 0 -> Y0Group[0] - Y0Group[4] >= -0
 LinkY0Group2[0,5]: Y0LessThan[0,5] = 0 -> Y0Group[0] - Y0Group[5] >= -0
 LinkY0Group2[0,6]: Y0LessThan[0,6] = 0 -> Y0Group[0] - Y0Group[6] >= -0
 LinkY0Group2[0,7]: Y0LessThan[0,7] = 0 -> Y0Group[0] - Y0Group[7] >= -0
 LinkY0Group2[1,0]: Y0LessThan[1,0] = 0 -> Y0Group[1] - Y0Group[0] >= -0
 LinkY0Group2[1,2]: Y0LessThan[1,2] = 0 -> Y0Group[1] - Y0Group[2] >= -0
 LinkY0Group2[1,3]: Y0LessThan[1,3] = 0 -> Y0Group[1] - Y0Group[3] >= -0
 LinkY0Group2[1,4]: Y0LessThan[1,4] = 0 -> Y0Group[1] - Y0Group[4] >= -0
 LinkY0Group2[1,5]: Y0LessThan[1,5] = 0 -> Y0Group[1] - Y0Group[5] >= -0
 LinkY0Group2[1,6]: Y0LessThan[1,6] = 0 -> Y0Group[1] - Y0Group[6] >= -0
 LinkY0Group2[1,7]: Y0LessThan[1,7] = 0 -> Y0Group[1] - Y0Group[7] >= -0
 LinkY0Group2[2,0]: Y0LessThan[2,0] = 0 -> Y0Group[2] - Y0Group[0] >= -0
 LinkY0Group2[2,1]: Y0LessThan[2,1] = 0 -> Y0Group[2] - Y0Group[1] >= -0
 LinkY0Group2[2,3]: Y0LessThan[2,3] = 0 -> Y0Group[2] - Y0Group[3] >= -0
 LinkY0Group2[2,4]: Y0LessThan[2,4] = 0 -> Y0Group[2] - Y0Group[4] >= -0
 LinkY0Group2[2,5]: Y0LessThan[2,5] = 0 -> Y0Group[2] - Y0Group[5] >= -0
 LinkY0Group2[2,6]: Y0LessThan[2,6] = 0 -> Y0Group[2] - Y0Group[6] >= -0
 LinkY0Group2[2,7]: Y0LessThan[2,7] = 0 -> Y0Group[2] - Y0Group[7] >= -0
 LinkY0Group2[3,0]: Y0LessThan[3,0] = 0 -> Y0Group[3] - Y0Group[0] >= -0
 LinkY0Group2[3,1]: Y0LessThan[3,1] = 0 -> Y0Group[3] - Y0Group[1] >= -0
 LinkY0Group2[3,2]: Y0LessThan[3,2] = 0 -> Y0Group[3] - Y0Group[2] >= -0
 LinkY0Group2[3,4]: Y0LessThan[3,4] = 0 -> Y0Group[3] - Y0Group[4] >= -0
 LinkY0Group2[3,5]: Y0LessThan[3,5] = 0 -> Y0Group[3] - Y0Group[5] >= -0
 LinkY0Group2[3,6]: Y0LessThan[3,6] = 0 -> Y0Group[3] - Y0Group[6] >= -0
 LinkY0Group2[3,7]: Y0LessThan[3,7] = 0 -> Y0Group[3] - Y0Group[7] >= -0
 LinkY0Group2[4,0]: Y0LessThan[4,0] = 0 -> Y0Group[4] - Y0Group[0] >= -0
 LinkY0Group2[4,1]: Y0LessThan[4,1] = 0 -> Y0Group[4] - Y0Group[1] >= -0
 LinkY0Group2[4,2]: Y0LessThan[4,2] = 0 -> Y0Group[4] - Y0Group[2] >= -0
 LinkY0Group2[4,3]: Y0LessThan[4,3] = 0 -> Y0Group[4] - Y0Group[3] >= -0
 LinkY0Group2[4,5]: Y0LessThan[4,5] = 0 -> Y0Group[4] - Y0Group[5] >= -0
 LinkY0Group2[4,6]: Y0LessThan[4,6] = 0 -> Y0Group[4] - Y0Group[6] >= -0
 LinkY0Group2[4,7]: Y0LessThan[4,7] = 0 -> Y0Group[4] - Y0Group[7] >= -0
 LinkY0Group2[5,0]: Y0LessThan[5,0] = 0 -> Y0Group[5] - Y0Group[0] >= -0
 LinkY0Group2[5,1]: Y0LessThan[5,1] = 0 -> Y0Group[5] - Y0Group[1] >= -0
 LinkY0Group2[5,2]: Y0LessThan[5,2] = 0 -> Y0Group[5] - Y0Group[2] >= -0
 LinkY0Group2[5,3]: Y0LessThan[5,3] = 0 -> Y0Group[5] - Y0Group[3] >= -0
 LinkY0Group2[5,4]: Y0LessThan[5,4] = 0 -> Y0Group[5] - Y0Group[4] >= -0
 LinkY0Group2[5,6]: Y0LessThan[5,6] = 0 -> Y0Group[5] - Y0Group[6] >= -0
 LinkY0Group2[5,7]: Y0LessThan[5,7] = 0 -> Y0Group[5] - Y0Group[7] >= -0
 LinkY0Group2[6,0]: Y0LessThan[6,0] = 0 -> Y0Group[6] - Y0Group[0] >= -0
 LinkY0Group2[6,1]: Y0LessThan[6,1] = 0 -> Y0Group[6] - Y0Group[1] >= -0
 LinkY0Group2[6,2]: Y0LessThan[6,2] = 0 -> Y0Group[6] - Y0Group[2] >= -0
 LinkY0Group2[6,3]: Y0LessThan[6,3] = 0 -> Y0Group[6] - Y0Group[3] >= -0
 LinkY0Group2[6,4]: Y0LessThan[6,4] = 0 -> Y0Group[6] - Y0Group[4] >= -0
 LinkY0Group2[6,5]: Y0LessThan[6,5] = 0 -> Y0Group[6] - Y0Group[5] >= -0
 LinkY0Group2[6,7]: Y0LessThan[6,7] = 0 -> Y0Group[6] - Y0Group[7] >= -0
 LinkY0Group2[7,0]: Y0LessThan[7,0] = 0 -> Y0Group[7] - Y0Group[0] >= -0
 LinkY0Group2[7,1]: Y0LessThan[7,1] = 0 -> Y0Group[7] - Y0Group[1] >= -0
 LinkY0Group2[7,2]: Y0LessThan[7,2] = 0 -> Y0Group[7] - Y0Group[2] >= -0
 LinkY0Group2[7,3]: Y0LessThan[7,3] = 0 -> Y0Group[7] - Y0Group[3] >= -0
 LinkY0Group2[7,4]: Y0LessThan[7,4] = 0 -> Y0Group[7] - Y0Group[4] >= -0
 LinkY0Group2[7,5]: Y0LessThan[7,5] = 0 -> Y0Group[7] - Y0Group[5] >= -0
 LinkY0Group2[7,6]: Y0LessThan[7,6] = 0 -> Y0Group[7] - Y0Group[6] >= -0
 LinkX1Group1[0,1]: X1LessThan[0,1] = 1 -> X1Group[0] - X1Group[1] <= -1
 LinkX1Group1[0,2]: X1LessThan[0,2] = 1 -> X1Group[0] - X1Group[2] <= -1
 LinkX1Group1[0,3]: X1LessThan[0,3] = 1 -> X1Group[0] - X1Group[3] <= -1
 LinkX1Group1[0,4]: X1LessThan[0,4] = 1 -> X1Group[0] - X1Group[4] <= -1
 LinkX1Group1[0,5]: X1LessThan[0,5] = 1 -> X1Group[0] - X1Group[5] <= -1
 LinkX1Group1[0,6]: X1LessThan[0,6] = 1 -> X1Group[0] - X1Group[6] <= -1
 LinkX1Group1[0,7]: X1LessThan[0,7] = 1 -> X1Group[0] - X1Group[7] <= -1
 LinkX1Group1[1,0]: X1LessThan[1,0] = 1 -> X1Group[1] - X1Group[0] <= -1
 LinkX1Group1[1,2]: X1LessThan[1,2] = 1 -> X1Group[1] - X1Group[2] <= -1
 LinkX1Group1[1,3]: X1LessThan[1,3] = 1 -> X1Group[1] - X1Group[3] <= -1
 LinkX1Group1[1,4]: X1LessThan[1,4] = 1 -> X1Group[1] - X1Group[4] <= -1
 LinkX1Group1[1,5]: X1LessThan[1,5] = 1 -> X1Group[1] - X1Group[5] <= -1
 LinkX1Group1[1,6]: X1LessThan[1,6] = 1 -> X1Group[1] - X1Group[6] <= -1
 LinkX1Group1[1,7]: X1LessThan[1,7] = 1 -> X1Group[1] - X1Group[7] <= -1
 LinkX1Group1[2,0]: X1LessThan[2,0] = 1 -> X1Group[2] - X1Group[0] <= -1
 LinkX1Group1[2,1]: X1LessThan[2,1] = 1 -> X1Group[2] - X1Group[1] <= -1
 LinkX1Group1[2,3]: X1LessThan[2,3] = 1 -> X1Group[2] - X1Group[3] <= -1
 LinkX1Group1[2,4]: X1LessThan[2,4] = 1 -> X1Group[2] - X1Group[4] <= -1
 LinkX1Group1[2,5]: X1LessThan[2,5] = 1 -> X1Group[2] - X1Group[5] <= -1
 LinkX1Group1[2,6]: X1LessThan[2,6] = 1 -> X1Group[2] - X1Group[6] <= -1
 LinkX1Group1[2,7]: X1LessThan[2,7] = 1 -> X1Group[2] - X1Group[7] <= -1
 LinkX1Group1[3,0]: X1LessThan[3,0] = 1 -> X1Group[3] - X1Group[0] <= -1
 LinkX1Group1[3,1]: X1LessThan[3,1] = 1 -> X1Group[3] - X1Group[1] <= -1
 LinkX1Group1[3,2]: X1LessThan[3,2] = 1 -> X1Group[3] - X1Group[2] <= -1
 LinkX1Group1[3,4]: X1LessThan[3,4] = 1 -> X1Group[3] - X1Group[4] <= -1
 LinkX1Group1[3,5]: X1LessThan[3,5] = 1 -> X1Group[3] - X1Group[5] <= -1
 LinkX1Group1[3,6]: X1LessThan[3,6] = 1 -> X1Group[3] - X1Group[6] <= -1
 LinkX1Group1[3,7]: X1LessThan[3,7] = 1 -> X1Group[3] - X1Group[7] <= -1
 LinkX1Group1[4,0]: X1LessThan[4,0] = 1 -> X1Group[4] - X1Group[0] <= -1
 LinkX1Group1[4,1]: X1LessThan[4,1] = 1 -> X1Group[4] - X1Group[1] <= -1
 LinkX1Group1[4,2]: X1LessThan[4,2] = 1 -> X1Group[4] - X1Group[2] <= -1
 LinkX1Group1[4,3]: X1LessThan[4,3] = 1 -> X1Group[4] - X1Group[3] <= -1
 LinkX1Group1[4,5]: X1LessThan[4,5] = 1 -> X1Group[4] - X1Group[5] <= -1
 LinkX1Group1[4,6]: X1LessThan[4,6] = 1 -> X1Group[4] - X1Group[6] <= -1
 LinkX1Group1[4,7]: X1LessThan[4,7] = 1 -> X1Group[4] - X1Group[7] <= -1
 LinkX1Group1[5,0]: X1LessThan[5,0] = 1 -> X1Group[5] - X1Group[0] <= -1
 LinkX1Group1[5,1]: X1LessThan[5,1] = 1 -> X1Group[5] - X1Group[1] <= -1
 LinkX1Group1[5,2]: X1LessThan[5,2] = 1 -> X1Group[5] - X1Group[2] <= -1
 LinkX1Group1[5,3]: X1LessThan[5,3] = 1 -> X1Group[5] - X1Group[3] <= -1
 LinkX1Group1[5,4]: X1LessThan[5,4] = 1 -> X1Group[5] - X1Group[4] <= -1
 LinkX1Group1[5,6]: X1LessThan[5,6] = 1 -> X1Group[5] - X1Group[6] <= -1
 LinkX1Group1[5,7]: X1LessThan[5,7] = 1 -> X1Group[5] - X1Group[7] <= -1
 LinkX1Group1[6,0]: X1LessThan[6,0] = 1 -> X1Group[6] - X1Group[0] <= -1
 LinkX1Group1[6,1]: X1LessThan[6,1] = 1 -> X1Group[6] - X1Group[1] <= -1
 LinkX1Group1[6,2]: X1LessThan[6,2] = 1 -> X1Group[6] - X1Group[2] <= -1
 LinkX1Group1[6,3]: X1LessThan[6,3] = 1 -> X1Group[6] - X1Group[3] <= -1
 LinkX1Group1[6,4]: X1LessThan[6,4] = 1 -> X1Group[6] - X1Group[4] <= -1
 LinkX1Group1[6,5]: X1LessThan[6,5] = 1 -> X1Group[6] - X1Group[5] <= -1
 LinkX1Group1[6,7]: X1LessThan[6,7] = 1 -> X1Group[6] - X1Group[7] <= -1
 LinkX1Group1[7,0]: X1LessThan[7,0] = 1 -> X1Group[7] - X1Group[0] <= -1
 LinkX1Group1[7,1]: X1LessThan[7,1] = 1 -> X1Group[7] - X1Group[1] <= -1
 LinkX1Group1[7,2]: X1LessThan[7,2] = 1 -> X1Group[7] - X1Group[2] <= -1
 LinkX1Group1[7,3]: X1LessThan[7,3] = 1 -> X1Group[7] - X1Group[3] <= -1
 LinkX1Group1[7,4]: X1LessThan[7,4] = 1 -> X1Group[7] - X1Group[4] <= -1
 LinkX1Group1[7,5]: X1LessThan[7,5] = 1 -> X1Group[7] - X1Group[5] <= -1
 LinkX1Group1[7,6]: X1LessThan[7,6] = 1 -> X1Group[7] - X1Group[6] <= -1
 LinkX1Group2[0,1]: X1LessThan[0,1] = 0 -> X1Group[0] - X1Group[1] >= -0
 LinkX1Group2[0,2]: X1LessThan[0,2] = 0 -> X1Group[0] - X1Group[2] >= -0
 LinkX1Group2[0,3]: X1LessThan[0,3] = 0 -> X1Group[0] - X1Group[3] >= -0
 LinkX1Group2[0,4]: X1LessThan[0,4] = 0 -> X1Group[0] - X1Group[4] >= -0
 LinkX1Group2[0,5]: X1LessThan[0,5] = 0 -> X1Group[0] - X1Group[5] >= -0
 LinkX1Group2[0,6]: X1LessThan[0,6] = 0 -> X1Group[0] - X1Group[6] >= -0
 LinkX1Group2[0,7]: X1LessThan[0,7] = 0 -> X1Group[0] - X1Group[7] >= -0
 LinkX1Group2[1,0]: X1LessThan[1,0] = 0 -> X1Group[1] - X1Group[0] >= -0
 LinkX1Group2[1,2]: X1LessThan[1,2] = 0 -> X1Group[1] - X1Group[2] >= -0
 LinkX1Group2[1,3]: X1LessThan[1,3] = 0 -> X1Group[1] - X1Group[3] >= -0
 LinkX1Group2[1,4]: X1LessThan[1,4] = 0 -> X1Group[1] - X1Group[4] >= -0
 LinkX1Group2[1,5]: X1LessThan[1,5] = 0 -> X1Group[1] - X1Group[5] >= -0
 LinkX1Group2[1,6]: X1LessThan[1,6] = 0 -> X1Group[1] - X1Group[6] >= -0
 LinkX1Group2[1,7]: X1LessThan[1,7] = 0 -> X1Group[1] - X1Group[7] >= -0
 LinkX1Group2[2,0]: X1LessThan[2,0] = 0 -> X1Group[2] - X1Group[0] >= -0
 LinkX1Group2[2,1]: X1LessThan[2,1] = 0 -> X1Group[2] - X1Group[1] >= -0
 LinkX1Group2[2,3]: X1LessThan[2,3] = 0 -> X1Group[2] - X1Group[3] >= -0
 LinkX1Group2[2,4]: X1LessThan[2,4] = 0 -> X1Group[2] - X1Group[4] >= -0
 LinkX1Group2[2,5]: X1LessThan[2,5] = 0 -> X1Group[2] - X1Group[5] >= -0
 LinkX1Group2[2,6]: X1LessThan[2,6] = 0 -> X1Group[2] - X1Group[6] >= -0
 LinkX1Group2[2,7]: X1LessThan[2,7] = 0 -> X1Group[2] - X1Group[7] >= -0
 LinkX1Group2[3,0]: X1LessThan[3,0] = 0 -> X1Group[3] - X1Group[0] >= -0
 LinkX1Group2[3,1]: X1LessThan[3,1] = 0 -> X1Group[3] - X1Group[1] >= -0
 LinkX1Group2[3,2]: X1LessThan[3,2] = 0 -> X1Group[3] - X1Group[2] >= -0
 LinkX1Group2[3,4]: X1LessThan[3,4] = 0 -> X1Group[3] - X1Group[4] >= -0
 LinkX1Group2[3,5]: X1LessThan[3,5] = 0 -> X1Group[3] - X1Group[5] >= -0
 LinkX1Group2[3,6]: X1LessThan[3,6] = 0 -> X1Group[3] - X1Group[6] >= -0
 LinkX1Group2[3,7]: X1LessThan[3,7] = 0 -> X1Group[3] - X1Group[7] >= -0
 LinkX1Group2[4,0]: X1LessThan[4,0] = 0 -> X1Group[4] - X1Group[0] >= -0
 LinkX1Group2[4,1]: X1LessThan[4,1] = 0 -> X1Group[4] - X1Group[1] >= -0
 LinkX1Group2[4,2]: X1LessThan[4,2] = 0 -> X1Group[4] - X1Group[2] >= -0
 LinkX1Group2[4,3]: X1LessThan[4,3] = 0 -> X1Group[4] - X1Group[3] >= -0
 LinkX1Group2[4,5]: X1LessThan[4,5] = 0 -> X1Group[4] - X1Group[5] >= -0
 LinkX1Group2[4,6]: X1LessThan[4,6] = 0 -> X1Group[4] - X1Group[6] >= -0
 LinkX1Group2[4,7]: X1LessThan[4,7] = 0 -> X1Group[4] - X1Group[7] >= -0
 LinkX1Group2[5,0]: X1LessThan[5,0] = 0 -> X1Group[5] - X1Group[0] >= -0
 LinkX1Group2[5,1]: X1LessThan[5,1] = 0 -> X1Group[5] - X1Group[1] >= -0
 LinkX1Group2[5,2]: X1LessThan[5,2] = 0 -> X1Group[5] - X1Group[2] >= -0
 LinkX1Group2[5,3]: X1LessThan[5,3] = 0 -> X1Group[5] - X1Group[3] >= -0
 LinkX1Group2[5,4]: X1LessThan[5,4] = 0 -> X1Group[5] - X1Group[4] >= -0
 LinkX1Group2[5,6]: X1LessThan[5,6] = 0 -> X1Group[5] - X1Group[6] >= -0
 LinkX1Group2[5,7]: X1LessThan[5,7] = 0 -> X1Group[5] - X1Group[7] >= -0
 LinkX1Group2[6,0]: X1LessThan[6,0] = 0 -> X1Group[6] - X1Group[0] >= -0
 LinkX1Group2[6,1]: X1LessThan[6,1] = 0 -> X1Group[6] - X1Group[1] >= -0
 LinkX1Group2[6,2]: X1LessThan[6,2] = 0 -> X1Group[6] - X1Group[2] >= -0
 LinkX1Group2[6,3]: X1LessThan[6,3] = 0 -> X1Group[6] - X1Group[3] >= -0
 LinkX1Group2[6,4]: X1LessThan[6,4] = 0 -> X1Group[6] - X1Group[4] >= -0
 LinkX1Group2[6,5]: X1LessThan[6,5] = 0 -> X1Group[6] - X1Group[5] >= -0
 LinkX1Group2[6,7]: X1LessThan[6,7] = 0 -> X1Group[6] - X1Group[7] >= -0
 LinkX1Group2[7,0]: X1LessThan[7,0] = 0 -> X1Group[7] - X1Group[0] >= -0
 LinkX1Group2[7,1]: X1LessThan[7,1] = 0 -> X1Group[7] - X1Group[1] >= -0
 LinkX1Group2[7,2]: X1LessThan[7,2] = 0 -> X1Group[7] - X1Group[2] >= -0
 LinkX1Group2[7,3]: X1LessThan[7,3] = 0 -> X1Group[7] - X1Group[3] >= -0
 LinkX1Group2[7,4]: X1LessThan[7,4] = 0 -> X1Group[7] - X1Group[4] >= -0
 LinkX1Group2[7,5]: X1LessThan[7,5] = 0 -> X1Group[7] - X1Group[5] >= -0
 LinkX1Group2[7,6]: X1LessThan[7,6] = 0 -> X1Group[7] - X1Group[6] >= -0
 LinkY1Group1[0,1]: Y1LessThan[0,1] = 1 -> Y1Group[0] - Y1Group[1] <= -1
 LinkY1Group1[0,2]: Y1LessThan[0,2] = 1 -> Y1Group[0] - Y1Group[2] <= -1
 LinkY1Group1[0,3]: Y1LessThan[0,3] = 1 -> Y1Group[0] - Y1Group[3] <= -1
 LinkY1Group1[0,4]: Y1LessThan[0,4] = 1 -> Y1Group[0] - Y1Group[4] <= -1
 LinkY1Group1[0,5]: Y1LessThan[0,5] = 1 -> Y1Group[0] - Y1Group[5] <= -1
 LinkY1Group1[0,6]: Y1LessThan[0,6] = 1 -> Y1Group[0] - Y1Group[6] <= -1
 LinkY1Group1[0,7]: Y1LessThan[0,7] = 1 -> Y1Group[0] - Y1Group[7] <= -1
 LinkY1Group1[1,0]: Y1LessThan[1,0] = 1 -> Y1Group[1] - Y1Group[0] <= -1
 LinkY1Group1[1,2]: Y1LessThan[1,2] = 1 -> Y1Group[1] - Y1Group[2] <= -1
 LinkY1Group1[1,3]: Y1LessThan[1,3] = 1 -> Y1Group[1] - Y1Group[3] <= -1
 LinkY1Group1[1,4]: Y1LessThan[1,4] = 1 -> Y1Group[1] - Y1Group[4] <= -1
 LinkY1Group1[1,5]: Y1LessThan[1,5] = 1 -> Y1Group[1] - Y1Group[5] <= -1
 LinkY1Group1[1,6]: Y1LessThan[1,6] = 1 -> Y1Group[1] - Y1Group[6] <= -1
 LinkY1Group1[1,7]: Y1LessThan[1,7] = 1 -> Y1Group[1] - Y1Group[7] <= -1
 LinkY1Group1[2,0]: Y1LessThan[2,0] = 1 -> Y1Group[2] - Y1Group[0] <= -1
 LinkY1Group1[2,1]: Y1LessThan[2,1] = 1 -> Y1Group[2] - Y1Group[1] <= -1
 LinkY1Group1[2,3]: Y1LessThan[2,3] = 1 -> Y1Group[2] - Y1Group[3] <= -1
 LinkY1Group1[2,4]: Y1LessThan[2,4] = 1 -> Y1Group[2] - Y1Group[4] <= -1
 LinkY1Group1[2,5]: Y1LessThan[2,5] = 1 -> Y1Group[2] - Y1Group[5] <= -1
 LinkY1Group1[2,6]: Y1LessThan[2,6] = 1 -> Y1Group[2] - Y1Group[6] <= -1
 LinkY1Group1[2,7]: Y1LessThan[2,7] = 1 -> Y1Group[2] - Y1Group[7] <= -1
 LinkY1Group1[3,0]: Y1LessThan[3,0] = 1 -> Y1Group[3] - Y1Group[0] <= -1
 LinkY1Group1[3,1]: Y1LessThan[3,1] = 1 -> Y1Group[3] - Y1Group[1] <= -1
 LinkY1Group1[3,2]: Y1LessThan[3,2] = 1 -> Y1Group[3] - Y1Group[2] <= -1
 LinkY1Group1[3,4]: Y1LessThan[3,4] = 1 -> Y1Group[3] - Y1Group[4] <= -1
 LinkY1Group1[3,5]: Y1LessThan[3,5] = 1 -> Y1Group[3] - Y1Group[5] <= -1
 LinkY1Group1[3,6]: Y1LessThan[3,6] = 1 -> Y1Group[3] - Y1Group[6] <= -1
 LinkY1Group1[3,7]: Y1LessThan[3,7] = 1 -> Y1Group[3] - Y1Group[7] <= -1
 LinkY1Group1[4,0]: Y1LessThan[4,0] = 1 -> Y1Group[4] - Y1Group[0] <= -1
 LinkY1Group1[4,1]: Y1LessThan[4,1] = 1 -> Y1Group[4] - Y1Group[1] <= -1
 LinkY1Group1[4,2]: Y1LessThan[4,2] = 1 -> Y1Group[4] - Y1Group[2] <= -1
 LinkY1Group1[4,3]: Y1LessThan[4,3] = 1 -> Y1Group[4] - Y1Group[3] <= -1
 LinkY1Group1[4,5]: Y1LessThan[4,5] = 1 -> Y1Group[4] - Y1Group[5] <= -1
 LinkY1Group1[4,6]: Y1LessThan[4,6] = 1 -> Y1Group[4] - Y1Group[6] <= -1
 LinkY1Group1[4,7]: Y1LessThan[4,7] = 1 -> Y1Group[4] - Y1Group[7] <= -1
 LinkY1Group1[5,0]: Y1LessThan[5,0] = 1 -> Y1Group[5] - Y1Group[0] <= -1
 LinkY1Group1[5,1]: Y1LessThan[5,1] = 1 -> Y1Group[5] - Y1Group[1] <= -1
 LinkY1Group1[5,2]: Y1LessThan[5,2] = 1 -> Y1Group[5] - Y1Group[2] <= -1
 LinkY1Group1[5,3]: Y1LessThan[5,3] = 1 -> Y1Group[5] - Y1Group[3] <= -1
 LinkY1Group1[5,4]: Y1LessThan[5,4] = 1 -> Y1Group[5] - Y1Group[4] <= -1
 LinkY1Group1[5,6]: Y1LessThan[5,6] = 1 -> Y1Group[5] - Y1Group[6] <= -1
 LinkY1Group1[5,7]: Y1LessThan[5,7] = 1 -> Y1Group[5] - Y1Group[7] <= -1
 LinkY1Group1[6,0]: Y1LessThan[6,0] = 1 -> Y1Group[6] - Y1Group[0] <= -1
 LinkY1Group1[6,1]: Y1LessThan[6,1] = 1 -> Y1Group[6] - Y1Group[1] <= -1
 LinkY1Group1[6,2]: Y1LessThan[6,2] = 1 -> Y1Group[6] - Y1Group[2] <= -1
 LinkY1Group1[6,3]: Y1LessThan[6,3] = 1 -> Y1Group[6] - Y1Group[3] <= -1
 LinkY1Group1[6,4]: Y1LessThan[6,4] = 1 -> Y1Group[6] - Y1Group[4] <= -1
 LinkY1Group1[6,5]: Y1LessThan[6,5] = 1 -> Y1Group[6] - Y1Group[5] <= -1
 LinkY1Group1[6,7]: Y1LessThan[6,7] = 1 -> Y1Group[6] - Y1Group[7] <= -1
 LinkY1Group1[7,0]: Y1LessThan[7,0] = 1 -> Y1Group[7] - Y1Group[0] <= -1
 LinkY1Group1[7,1]: Y1LessThan[7,1] = 1 -> Y1Group[7] - Y1Group[1] <= -1
 LinkY1Group1[7,2]: Y1LessThan[7,2] = 1 -> Y1Group[7] - Y1Group[2] <= -1
 LinkY1Group1[7,3]: Y1LessThan[7,3] = 1 -> Y1Group[7] - Y1Group[3] <= -1
 LinkY1Group1[7,4]: Y1LessThan[7,4] = 1 -> Y1Group[7] - Y1Group[4] <= -1
 LinkY1Group1[7,5]: Y1LessThan[7,5] = 1 -> Y1Group[7] - Y1Group[5] <= -1
 LinkY1Group1[7,6]: Y1LessThan[7,6] = 1 -> Y1Group[7] - Y1Group[6] <= -1
 LinkY1Group2[0,1]: Y1LessThan[0,1] = 0 -> Y1Group[0] - Y1Group[1] >= -0
 LinkY1Group2[0,2]: Y1LessThan[0,2] = 0 -> Y1Group[0] - Y1Group[2] >= -0
 LinkY1Group2[0,3]: Y1LessThan[0,3] = 0 -> Y1Group[0] - Y1Group[3] >= -0
 LinkY1Group2[0,4]: Y1LessThan[0,4] = 0 -> Y1Group[0] - Y1Group[4] >= -0
 LinkY1Group2[0,5]: Y1LessThan[0,5] = 0 -> Y1Group[0] - Y1Group[5] >= -0
 LinkY1Group2[0,6]: Y1LessThan[0,6] = 0 -> Y1Group[0] - Y1Group[6] >= -0
 LinkY1Group2[0,7]: Y1LessThan[0,7] = 0 -> Y1Group[0] - Y1Group[7] >= -0
 LinkY1Group2[1,0]: Y1LessThan[1,0] = 0 -> Y1Group[1] - Y1Group[0] >= -0
 LinkY1Group2[1,2]: Y1LessThan[1,2] = 0 -> Y1Group[1] - Y1Group[2] >= -0
 LinkY1Group2[1,3]: Y1LessThan[1,3] = 0 -> Y1Group[1] - Y1Group[3] >= -0
 LinkY1Group2[1,4]: Y1LessThan[1,4] = 0 -> Y1Group[1] - Y1Group[4] >= -0
 LinkY1Group2[1,5]: Y1LessThan[1,5] = 0 -> Y1Group[1] - Y1Group[5] >= -0
 LinkY1Group2[1,6]: Y1LessThan[1,6] = 0 -> Y1Group[1] - Y1Group[6] >= -0
 LinkY1Group2[1,7]: Y1LessThan[1,7] = 0 -> Y1Group[1] - Y1Group[7] >= -0
 LinkY1Group2[2,0]: Y1LessThan[2,0] = 0 -> Y1Group[2] - Y1Group[0] >= -0
 LinkY1Group2[2,1]: Y1LessThan[2,1] = 0 -> Y1Group[2] - Y1Group[1] >= -0
 LinkY1Group2[2,3]: Y1LessThan[2,3] = 0 -> Y1Group[2] - Y1Group[3] >= -0
 LinkY1Group2[2,4]: Y1LessThan[2,4] = 0 -> Y1Group[2] - Y1Group[4] >= -0
 LinkY1Group2[2,5]: Y1LessThan[2,5] = 0 -> Y1Group[2] - Y1Group[5] >= -0
 LinkY1Group2[2,6]: Y1LessThan[2,6] = 0 -> Y1Group[2] - Y1Group[6] >= -0
 LinkY1Group2[2,7]: Y1LessThan[2,7] = 0 -> Y1Group[2] - Y1Group[7] >= -0
 LinkY1Group2[3,0]: Y1LessThan[3,0] = 0 -> Y1Group[3] - Y1Group[0] >= -0
 LinkY1Group2[3,1]: Y1LessThan[3,1] = 0 -> Y1Group[3] - Y1Group[1] >= -0
 LinkY1Group2[3,2]: Y1LessThan[3,2] = 0 -> Y1Group[3] - Y1Group[2] >= -0
 LinkY1Group2[3,4]: Y1LessThan[3,4] = 0 -> Y1Group[3] - Y1Group[4] >= -0
 LinkY1Group2[3,5]: Y1LessThan[3,5] = 0 -> Y1Group[3] - Y1Group[5] >= -0
 LinkY1Group2[3,6]: Y1LessThan[3,6] = 0 -> Y1Group[3] - Y1Group[6] >= -0
 LinkY1Group2[3,7]: Y1LessThan[3,7] = 0 -> Y1Group[3] - Y1Group[7] >= -0
 LinkY1Group2[4,0]: Y1LessThan[4,0] = 0 -> Y1Group[4] - Y1Group[0] >= -0
 LinkY1Group2[4,1]: Y1LessThan[4,1] = 0 -> Y1Group[4] - Y1Group[1] >= -0
 LinkY1Group2[4,2]: Y1LessThan[4,2] = 0 -> Y1Group[4] - Y1Group[2] >= -0
 LinkY1Group2[4,3]: Y1LessThan[4,3] = 0 -> Y1Group[4] - Y1Group[3] >= -0
 LinkY1Group2[4,5]: Y1LessThan[4,5] = 0 -> Y1Group[4] - Y1Group[5] >= -0
 LinkY1Group2[4,6]: Y1LessThan[4,6] = 0 -> Y1Group[4] - Y1Group[6] >= -0
 LinkY1Group2[4,7]: Y1LessThan[4,7] = 0 -> Y1Group[4] - Y1Group[7] >= -0
 LinkY1Group2[5,0]: Y1LessThan[5,0] = 0 -> Y1Group[5] - Y1Group[0] >= -0
 LinkY1Group2[5,1]: Y1LessThan[5,1] = 0 -> Y1Group[5] - Y1Group[1] >= -0
 LinkY1Group2[5,2]: Y1LessThan[5,2] = 0 -> Y1Group[5] - Y1Group[2] >= -0
 LinkY1Group2[5,3]: Y1LessThan[5,3] = 0 -> Y1Group[5] - Y1Group[3] >= -0
 LinkY1Group2[5,4]: Y1LessThan[5,4] = 0 -> Y1Group[5] - Y1Group[4] >= -0
 LinkY1Group2[5,6]: Y1LessThan[5,6] = 0 -> Y1Group[5] - Y1Group[6] >= -0
 LinkY1Group2[5,7]: Y1LessThan[5,7] = 0 -> Y1Group[5] - Y1Group[7] >= -0
 LinkY1Group2[6,0]: Y1LessThan[6,0] = 0 -> Y1Group[6] - Y1Group[0] >= -0
 LinkY1Group2[6,1]: Y1LessThan[6,1] = 0 -> Y1Group[6] - Y1Group[1] >= -0
 LinkY1Group2[6,2]: Y1LessThan[6,2] = 0 -> Y1Group[6] - Y1Group[2] >= -0
 LinkY1Group2[6,3]: Y1LessThan[6,3] = 0 -> Y1Group[6] - Y1Group[3] >= -0
 LinkY1Group2[6,4]: Y1LessThan[6,4] = 0 -> Y1Group[6] - Y1Group[4] >= -0
 LinkY1Group2[6,5]: Y1LessThan[6,5] = 0 -> Y1Group[6] - Y1Group[5] >= -0
 LinkY1Group2[6,7]: Y1LessThan[6,7] = 0 -> Y1Group[6] - Y1Group[7] >= -0
 LinkY1Group2[7,0]: Y1LessThan[7,0] = 0 -> Y1Group[7] - Y1Group[0] >= -0
 LinkY1Group2[7,1]: Y1LessThan[7,1] = 0 -> Y1Group[7] - Y1Group[1] >= -0
 LinkY1Group2[7,2]: Y1LessThan[7,2] = 0 -> Y1Group[7] - Y1Group[2] >= -0
 LinkY1Group2[7,3]: Y1LessThan[7,3] = 0 -> Y1Group[7] - Y1Group[3] >= -0
 LinkY1Group2[7,4]: Y1LessThan[7,4] = 0 -> Y1Group[7] - Y1Group[4] >= -0
 LinkY1Group2[7,5]: Y1LessThan[7,5] = 0 -> Y1Group[7] - Y1Group[5] >= -0
 LinkY1Group2[7,6]: Y1LessThan[7,6] = 0 -> Y1Group[7] - Y1Group[6] >= -0
Bounds
 W[0] >= 1
 W[1] >= 1
 W[2] >= 1
 W[3] >= 1
 W[4] >= 1
 W[5] >= 1
 W[6] >= 1
 W[7] >= 1
 H[0] >= 1
 H[1] >= 1
 H[2] >= 1
 H[3] >= 1
 H[4] >= 1
 H[5] >= 1
 H[6] >= 1
 H[7] >= 1
 ResizeW[0] free
 ResizeW[1] free
 ResizeW[2] free
 ResizeW[3] free
 ResizeW[4] free
 ResizeW[5] free
 ResizeW[6] free
 ResizeW[7] free
 ResizeH[0] free
 ResizeH[1] free
 ResizeH[2] free
 ResizeH[3] free
 ResizeH[4] free
 ResizeH[5] free
 ResizeH[6] free
 ResizeH[7] free
 MoveX[0] free
 MoveX[1] free
 MoveX[2] free
 MoveX[3] free
 MoveX[4] free
 MoveX[5] free
 MoveX[6] free
 MoveX[7] free
 MoveY[0] free
 MoveY[1] free
 MoveY[2] free
 MoveY[3] free
 MoveY[4] free
 MoveY[5] free
 MoveY[6] free
 MoveY[7] free
 EdgeDistance[x0,0,0] free
 EdgeDistance[x0,0,1] free
 EdgeDistance[x0,0,2] free
 EdgeDistance[x0,0,3] free
 EdgeDistance[x0,0,4] free
 EdgeDistance[x0,0,5] free
 EdgeDistance[x0,0,6] free
 EdgeDistance[x0,0,7] free
 EdgeDistance[x0,1,0] free
 EdgeDistance[x0,1,1] free
 EdgeDistance[x0,1,2] free
 EdgeDistance[x0,1,3] free
 EdgeDistance[x0,1,4] free
 EdgeDistance[x0,1,5] free
 EdgeDistance[x0,1,6] free
 EdgeDistance[x0,1,7] free
 EdgeDistance[x0,2,0] free
 EdgeDistance[x0,2,1] free
 EdgeDistance[x0,2,2] free
 EdgeDistance[x0,2,3] free
 EdgeDistance[x0,2,4] free
 EdgeDistance[x0,2,5] free
 EdgeDistance[x0,2,6] free
 EdgeDistance[x0,2,7] free
 EdgeDistance[x0,3,0] free
 EdgeDistance[x0,3,1] free
 EdgeDistance[x0,3,2] free
 EdgeDistance[x0,3,3] free
 EdgeDistance[x0,3,4] free
 EdgeDistance[x0,3,5] free
 EdgeDistance[x0,3,6] free
 EdgeDistance[x0,3,7] free
 EdgeDistance[x0,4,0] free
 EdgeDistance[x0,4,1] free
 EdgeDistance[x0,4,2] free
 EdgeDistance[x0,4,3] free
 EdgeDistance[x0,4,4] free
 EdgeDistance[x0,4,5] free
 EdgeDistance[x0,4,6] free
 EdgeDistance[x0,4,7] free
 EdgeDistance[x0,5,0] free
 EdgeDistance[x0,5,1] free
 EdgeDistance[x0,5,2] free
 EdgeDistance[x0,5,3] free
 EdgeDistance[x0,5,4] free
 EdgeDistance[x0,5,5] free
 EdgeDistance[x0,5,6] free
 EdgeDistance[x0,5,7] free
 EdgeDistance[x0,6,0] free
 EdgeDistance[x0,6,1] free
 EdgeDistance[x0,6,2] free
 EdgeDistance[x0,6,3] free
 EdgeDistance[x0,6,4] free
 EdgeDistance[x0,6,5] free
 EdgeDistance[x0,6,6] free
 EdgeDistance[x0,6,7] free
 EdgeDistance[x0,7,0] free
 EdgeDistance[x0,7,1] free
 EdgeDistance[x0,7,2] free
 EdgeDistance[x0,7,3] free
 EdgeDistance[x0,7,4] free
 EdgeDistance[x0,7,5] free
 EdgeDistance[x0,7,6] free
 EdgeDistance[x0,7,7] free
 EdgeDistance[y0,0,0] free
 EdgeDistance[y0,0,1] free
 EdgeDistance[y0,0,2] free
 EdgeDistance[y0,0,3] free
 EdgeDistance[y0,0,4] free
 EdgeDistance[y0,0,5] free
 EdgeDistance[y0,0,6] free
 EdgeDistance[y0,0,7] free
 EdgeDistance[y0,1,0] free
 EdgeDistance[y0,1,1] free
 EdgeDistance[y0,1,2] free
 EdgeDistance[y0,1,3] free
 EdgeDistance[y0,1,4] free
 EdgeDistance[y0,1,5] free
 EdgeDistance[y0,1,6] free
 EdgeDistance[y0,1,7] free
 EdgeDistance[y0,2,0] free
 EdgeDistance[y0,2,1] free
 EdgeDistance[y0,2,2] free
 EdgeDistance[y0,2,3] free
 EdgeDistance[y0,2,4] free
 EdgeDistance[y0,2,5] free
 EdgeDistance[y0,2,6] free
 EdgeDistance[y0,2,7] free
 EdgeDistance[y0,3,0] free
 EdgeDistance[y0,3,1] free
 EdgeDistance[y0,3,2] free
 EdgeDistance[y0,3,3] free
 EdgeDistance[y0,3,4] free
 EdgeDistance[y0,3,5] free
 EdgeDistance[y0,3,6] free
 EdgeDistance[y0,3,7] free
 EdgeDistance[y0,4,0] free
 EdgeDistance[y0,4,1] free
 EdgeDistance[y0,4,2] free
 EdgeDistance[y0,4,3] free
 EdgeDistance[y0,4,4] free
 EdgeDistance[y0,4,5] free
 EdgeDistance[y0,4,6] free
 EdgeDistance[y0,4,7] free
 EdgeDistance[y0,5,0] free
 EdgeDistance[y0,5,1] free
 EdgeDistance[y0,5,2] free
 EdgeDistance[y0,5,3] free
 EdgeDistance[y0,5,4] free
 EdgeDistance[y0,5,5] free
 EdgeDistance[y0,5,6] free
 EdgeDistance[y0,5,7] free
 EdgeDistance[y0,6,0] free
 EdgeDistance[y0,6,1] free
 EdgeDistance[y0,6,2] free
 EdgeDistance[y0,6,3] free
 EdgeDistance[y0,6,4] free
 EdgeDistance[y0,6,5] free
 EdgeDistance[y0,6,6] free
 EdgeDistance[y0,6,7] free
 EdgeDistance[y0,7,0] free
 EdgeDistance[y0,7,1] free
 EdgeDistance[y0,7,2] free
 EdgeDistance[y0,7,3] free
 EdgeDistance[y0,7,4] free
 EdgeDistance[y0,7,5] free
 EdgeDistance[y0,7,6] free
 EdgeDistance[y0,7,7] free
 EdgeDistance[x1,0,0] free
 EdgeDistance[x1,0,1] free
 EdgeDistance[x1,0,2] free
 EdgeDistance[x1,0,3] free
 EdgeDistance[x1,0,4] free
 EdgeDistance[x1,0,5] free
 EdgeDistance[x1,0,6] free
 EdgeDistance[x1,0,7] free
 EdgeDistance[x1,1,0] free
 EdgeDistance[x1,1,1] free
 EdgeDistance[x1,1,2] free
 EdgeDistance[x1,1,3] free
 EdgeDistance[x1,1,4] free
 EdgeDistance[x1,1,5] free
 EdgeDistance[x1,1,6] free
 EdgeDistance[x1,1,7] free
 EdgeDistance[x1,2,0] free
 EdgeDistance[x1,2,1] free
 EdgeDistance[x1,2,2] free
 EdgeDistance[x1,2,3] free
 EdgeDistance[x1,2,4] free
 EdgeDistance[x1,2,5] free
 EdgeDistance[x1,2,6] free
 EdgeDistance[x1,2,7] free
 EdgeDistance[x1,3,0] free
 EdgeDistance[x1,3,1] free
 EdgeDistance[x1,3,2] free
 EdgeDistance[x1,3,3] free
 EdgeDistance[x1,3,4] free
 EdgeDistance[x1,3,5] free
 EdgeDistance[x1,3,6] free
 EdgeDistance[x1,3,7] free
 EdgeDistance[x1,4,0] free
 EdgeDistance[x1,4,1] free
 EdgeDistance[x1,4,2] free
 EdgeDistance[x1,4,3] free
 EdgeDistance[x1,4,4] free
 EdgeDistance[x1,4,5] free
 EdgeDistance[x1,4,6] free
 EdgeDistance[x1,4,7] free
 EdgeDistance[x1,5,0] free
 EdgeDistance[x1,5,1] free
 EdgeDistance[x1,5,2] free
 EdgeDistance[x1,5,3] free
 EdgeDistance[x1,5,4] free
 EdgeDistance[x1,5,5] free
 EdgeDistance[x1,5,6] free
 EdgeDistance[x1,5,7] free
 EdgeDistance[x1,6,0] free
 EdgeDistance[x1,6,1] free
 EdgeDistance[x1,6,2] free
 EdgeDistance[x1,6,3] free
 EdgeDistance[x1,6,4] free
 EdgeDistance[x1,6,5] free
 EdgeDistance[x1,6,6] free
 EdgeDistance[x1,6,7] free
 EdgeDistance[x1,7,0] free
 EdgeDistance[x1,7,1] free
 EdgeDistance[x1,7,2] free
 EdgeDistance[x1,7,3] free
 EdgeDistance[x1,7,4] free
 EdgeDistance[x1,7,5] free
 EdgeDistance[x1,7,6] free
 EdgeDistance[x1,7,7] free
 EdgeDistance[y1,0,0] free
 EdgeDistance[y1,0,1] free
 EdgeDistance[y1,0,2] free
 EdgeDistance[y1,0,3] free
 EdgeDistance[y1,0,4] free
 EdgeDistance[y1,0,5] free
 EdgeDistance[y1,0,6] free
 EdgeDistance[y1,0,7] free
 EdgeDistance[y1,1,0] free
 EdgeDistance[y1,1,1] free
 EdgeDistance[y1,1,2] free
 EdgeDistance[y1,1,3] free
 EdgeDistance[y1,1,4] free
 EdgeDistance[y1,1,5] free
 EdgeDistance[y1,1,6] free
 EdgeDistance[y1,1,7] free
 EdgeDistance[y1,2,0] free
 EdgeDistance[y1,2,1] free
 EdgeDistance[y1,2,2] free
 EdgeDistance[y1,2,3] free
 EdgeDistance[y1,2,4] free
 EdgeDistance[y1,2,5] free
 EdgeDistance[y1,2,6] free
 EdgeDistance[y1,2,7] free
 EdgeDistance[y1,3,0] free
 EdgeDistance[y1,3,1] free
 EdgeDistance[y1,3,2] free
 EdgeDistance[y1,3,3] free
 EdgeDistance[y1,3,4] free
 EdgeDistance[y1,3,5] free
 EdgeDistance[y1,3,6] free
 EdgeDistance[y1,3,7] free
 EdgeDistance[y1,4,0] free
 EdgeDistance[y1,4,1] free
 EdgeDistance[y1,4,2] free
 EdgeDistance[y1,4,3] free
 EdgeDistance[y1,4,4] free
 EdgeDistance[y1,4,5] free
 EdgeDistance[y1,4,6] free
 EdgeDistance[y1,4,7] free
 EdgeDistance[y1,5,0] free
 EdgeDistance[y1,5,1] free
 EdgeDistance[y1,5,2] free
 EdgeDistance[y1,5,3] free
 EdgeDistance[y1,5,4] free
 EdgeDistance[y1,5,5] free
 EdgeDistance[y1,5,6] free
 EdgeDistance[y1,5,7] free
 EdgeDistance[y1,6,0] free
 EdgeDistance[y1,6,1] free
 EdgeDistance[y1,6,2] free
 EdgeDistance[y1,6,3] free
 EdgeDistance[y1,6,4] free
 EdgeDistance[y1,6,5] free
 EdgeDistance[y1,6,6] free
 EdgeDistance[y1,6,7] free
 EdgeDistance[y1,7,0] free
 EdgeDistance[y1,7,1] free
 EdgeDistance[y1,7,2] free
 EdgeDistance[y1,7,3] free
 EdgeDistance[y1,7,4] free
 EdgeDistance[y1,7,5] free
 EdgeDistance[y1,7,6] free
 EdgeDistance[y1,7,7] free
 X0Diff[0,0] free
 X0Diff[0,1] free
 X0Diff[0,2] free
 X0Diff[0,3] free
 X0Diff[0,4] free
 X0Diff[0,5] free
 X0Diff[0,6] free
 X0Diff[0,7] free
 X0Diff[1,0] free
 X0Diff[1,1] free
 X0Diff[1,2] free
 X0Diff[1,3] free
 X0Diff[1,4] free
 X0Diff[1,5] free
 X0Diff[1,6] free
 X0Diff[1,7] free
 X0Diff[2,0] free
 X0Diff[2,1] free
 X0Diff[2,2] free
 X0Diff[2,3] free
 X0Diff[2,4] free
 X0Diff[2,5] free
 X0Diff[2,6] free
 X0Diff[2,7] free
 X0Diff[3,0] free
 X0Diff[3,1] free
 X0Diff[3,2] free
 X0Diff[3,3] free
 X0Diff[3,4] free
 X0Diff[3,5] free
 X0Diff[3,6] free
 X0Diff[3,7] free
 X0Diff[4,0] free
 X0Diff[4,1] free
 X0Diff[4,2] free
 X0Diff[4,3] free
 X0Diff[4,4] free
 X0Diff[4,5] free
 X0Diff[4,6] free
 X0Diff[4,7] free
 X0Diff[5,0] free
 X0Diff[5,1] free
 X0Diff[5,2] free
 X0Diff[5,3] free
 X0Diff[5,4] free
 X0Diff[5,5] free
 X0Diff[5,6] free
 X0Diff[5,7] free
 X0Diff[6,0] free
 X0Diff[6,1] free
 X0Diff[6,2] free
 X0Diff[6,3] free
 X0Diff[6,4] free
 X0Diff[6,5] free
 X0Diff[6,6] free
 X0Diff[6,7] free
 X0Diff[7,0] free
 X0Diff[7,1] free
 X0Diff[7,2] free
 X0Diff[7,3] free
 X0Diff[7,4] free
 X0Diff[7,5] free
 X0Diff[7,6] free
 X0Diff[7,7] free
 Y0Diff[0,0] free
 Y0Diff[0,1] free
 Y0Diff[0,2] free
 Y0Diff[0,3] free
 Y0Diff[0,4] free
 Y0Diff[0,5] free
 Y0Diff[0,6] free
 Y0Diff[0,7] free
 Y0Diff[1,0] free
 Y0Diff[1,1] free
 Y0Diff[1,2] free
 Y0Diff[1,3] free
 Y0Diff[1,4] free
 Y0Diff[1,5] free
 Y0Diff[1,6] free
 Y0Diff[1,7] free
 Y0Diff[2,0] free
 Y0Diff[2,1] free
 Y0Diff[2,2] free
 Y0Diff[2,3] free
 Y0Diff[2,4] free
 Y0Diff[2,5] free
 Y0Diff[2,6] free
 Y0Diff[2,7] free
 Y0Diff[3,0] free
 Y0Diff[3,1] free
 Y0Diff[3,2] free
 Y0Diff[3,3] free
 Y0Diff[3,4] free
 Y0Diff[3,5] free
 Y0Diff[3,6] free
 Y0Diff[3,7] free
 Y0Diff[4,0] free
 Y0Diff[4,1] free
 Y0Diff[4,2] free
 Y0Diff[4,3] free
 Y0Diff[4,4] free
 Y0Diff[4,5] free
 Y0Diff[4,6] free
 Y0Diff[4,7] free
 Y0Diff[5,0] free
 Y0Diff[5,1] free
 Y0Diff[5,2] free
 Y0Diff[5,3] free
 Y0Diff[5,4] free
 Y0Diff[5,5] free
 Y0Diff[5,6] free
 Y0Diff[5,7] free
 Y0Diff[6,0] free
 Y0Diff[6,1] free
 Y0Diff[6,2] free
 Y0Diff[6,3] free
 Y0Diff[6,4] free
 Y0Diff[6,5] free
 Y0Diff[6,6] free
 Y0Diff[6,7] free
 Y0Diff[7,0] free
 Y0Diff[7,1] free
 Y0Diff[7,2] free
 Y0Diff[7,3] free
 Y0Diff[7,4] free
 Y0Diff[7,5] free
 Y0Diff[7,6] free
 Y0Diff[7,7] free
 X1Diff[0,0] free
 X1Diff[0,1] free
 X1Diff[0,2] free
 X1Diff[0,3] free
 X1Diff[0,4] free
 X1Diff[0,5] free
 X1Diff[0,6] free
 X1Diff[0,7] free
 X1Diff[1,0] free
 X1Diff[1,1] free
 X1Diff[1,2] free
 X1Diff[1,3] free
 X1Diff[1,4] free
 X1Diff[1,5] free
 X1Diff[1,6] free
 X1Diff[1,7] free
 X1Diff[2,0] free
 X1Diff[2,1] free
 X1Diff[2,2] free
 X1Diff[2,3] free
 X1Diff[2,4] free
 X1Diff[2,5] free
 X1Diff[2,6] free
 X1Diff[2,7] free
 X1Diff[3,0] free
 X1Diff[3,1] free
 X1Diff[3,2] free
 X1Diff[3,3] free
 X1Diff[3,4] free
 X1Diff[3,5] free
 X1Diff[3,6] free
 X1Diff[3,7] free
 X1Diff[4,0] free
 X1Diff[4,1] free
 X1Diff[4,2] free
 X1Diff[4,3] free
 X1Diff[4,4] free
 X1Diff[4,5] free
 X1Diff[4,6] free
 X1Diff[4,7] free
 X1Diff[5,0] free
 X1Diff[5,1] free
 X1Diff[5,2] free
 X1Diff[5,3] free
 X1Diff[5,4] free
 X1Diff[5,5] free
 X1Diff[5,6] free
 X1Diff[5,7] free
 X1Diff[6,0] free
 X1Diff[6,1] free
 X1Diff[6,2] free
 X1Diff[6,3] free
 X1Diff[6,4] free
 X1Diff[6,5] free
 X1Diff[6,6] free
 X1Diff[6,7] free
 X1Diff[7,0] free
 X1Diff[7,1] free
 X1Diff[7,2] free
 X1Diff[7,3] free
 X1Diff[7,4] free
 X1Diff[7,5] free
 X1Diff[7,6] free
 X1Diff[7,7] free
 Y1Diff[0,0] free
 Y1Diff[0,1] free
 Y1Diff[0,2] free
 Y1Diff[0,3] free
 Y1Diff[0,4] free
 Y1Diff[0,5] free
 Y1Diff[0,6] free
 Y1Diff[0,7] free
 Y1Diff[1,0] free
 Y1Diff[1,1] free
 Y1Diff[1,2] free
 Y1Diff[1,3] free
 Y1Diff[1,4] free
 Y1Diff[1,5] free
 Y1Diff[1,6] free
 Y1Diff[1,7] free
 Y1Diff[2,0] free
 Y1Diff[2,1] free
 Y1Diff[2,2] free
 Y1Diff[2,3] free
 Y1Diff[2,4] free
 Y1Diff[2,5] free
 Y1Diff[2,6] free
 Y1Diff[2,7] free
 Y1Diff[3,0] free
 Y1Diff[3,1] free
 Y1Diff[3,2] free
 Y1Diff[3,3] free
 Y1Diff[3,4] free
 Y1Diff[3,5] free
 Y1Diff[3,6] free
 Y1Diff[3,7] free
 Y1Diff[4,0] free
 Y1Diff[4,1] free
 Y1Diff[4,2] free
 Y1Diff[4,3] free
 Y1Diff[4,4] free
 Y1Diff[4,5] free
 Y1Diff[4,6] free
 Y1Diff[4,7] free
 Y1Diff[5,0] free
 Y1Diff[5,1] free
 Y1Diff[5,2] free
 Y1Diff[5,3] free
 Y1Diff[5,4] free
 Y1Diff[5,5] free
 Y1Diff[5,6] free
 Y1Diff[5,7] free
 Y1Diff[6,0] free
 Y1Diff[6,1] free
 Y1Diff[6,2] free
 Y1Diff[6,3] free
 Y1Diff[6,4] free
 Y1Diff[6,5] free
 Y1Diff[6,6] free
 Y1Diff[6,7] free
 Y1Diff[7,0] free
 Y1Diff[7,1] free
 Y1Diff[7,2] free
 Y1Diff[7,3] free
 Y1Diff[7,4] free
 Y1Diff[7,5] free
 Y1Diff[7,6] free
 Y1Diff[7,7] free
 1 <= X0Group[0] <= 8
 1 <= X0Group[1] <= 8
 1 <= X0Group[2] <= 8
 1 <= X0Group[3] <= 8
 1 <= X0Group[4] <= 8
 1 <= X0Group[5] <= 8
 1 <= X0Group[6] <= 8
 1 <= X0Group[7] <= 8
 1 <= Y0Group[0] <= 8
 1 <= Y0Group[1] <= 8
 1 <= Y0Group[2] <= 8
 1 <= Y0Group[3] <= 8
 1 <= Y0Group[4] <= 8
 1 <= Y0Group[5] <= 8
 1 <= Y0Group[6] <= 8
 1 <= Y0Group[7] <= 8
 1 <= X1Group[0] <= 8
 1 <= X1Group[1] <= 8
 1 <= X1Group[2] <= 8
 1 <= X1Group[3] <= 8
 1 <= X1Group[4] <= 8
 1 <= X1Group[5] <= 8
 1 <= X1Group[6] <= 8
 1 <= X1Group[7] <= 8
 1 <= Y1Group[0] <= 8
 1 <= Y1Group[1] <= 8
 1 <= Y1Group[2] <= 8
 1 <= Y1Group[3] <= 8
 1 <= Y1Group[4] <= 8
 1 <= Y1Group[5] <= 8
 1 <= Y1Group[6] <= 8
 1 <= Y1Group[7] <= 8
 1 <= X0GroupCount <= 8
 1 <= Y0GroupCount <= 8
 1 <= X1GroupCount <= 8
 1 <= Y1GroupCount <= 8
 C1232 >= 1
Binaries
 Above[0,0] Above[0,1] Above[0,2] Above[0,3] Above[0,4] Above[0,5]
 Above[0,6] Above[0,7] Above[1,0] Above[1,1] Above[1,2] Above[1,3]
 Above[1,4] Above[1,5] Above[1,6] Above[1,7] Above[2,0] Above[2,1]
 Above[2,2] Above[2,3] Above[2,4] Above[2,5] Above[2,6] Above[2,7]
 Above[3,0] Above[3,1] Above[3,2] Above[3,3] Above[3,4] Above[3,5]
 Above[3,6] Above[3,7] Above[4,0] Above[4,1] Above[4,2] Above[4,3]
 Above[4,4] Above[4,5] Above[4,6] Above[4,7] Above[5,0] Above[5,1]
 Above[5,2] Above[5,3] Above[5,4] Above[5,5] Above[5,6] Above[5,7]
 Above[6,0] Above[6,1] Above[6,2] Above[6,3] Above[6,4] Above[6,5]
 Above[6,6] Above[6,7] Above[7,0] Above[7,1] Above[7,2] Above[7,3]
 Above[7,4] Above[7,5] Above[7,6] Above[7,7] OnLeft[0,0] OnLeft[0,1]
 OnLeft[0,2] OnLeft[0,3] OnLeft[0,4] OnLeft[0,5] OnLeft[0,6] OnLeft[0,7]
 OnLeft[1,0] OnLeft[1,1] OnLeft[1,2] OnLeft[1,3] OnLeft[1,4] OnLeft[1,5]
 OnLeft[1,6] OnLeft[1,7] OnLeft[2,0] OnLeft[2,1] OnLeft[2,2] OnLeft[2,3]
 OnLeft[2,4] OnLeft[2,5] OnLeft[2,6] OnLeft[2,7] OnLeft[3,0] OnLeft[3,1]
 OnLeft[3,2] OnLeft[3,3] OnLeft[3,4] OnLeft[3,5] OnLeft[3,6] OnLeft[3,7]
 OnLeft[4,0] OnLeft[4,1] OnLeft[4,2] OnLeft[4,3] OnLeft[4,4] OnLeft[4,5]
 OnLeft[4,6] OnLeft[4,7] OnLeft[5,0] OnLeft[5,1] OnLeft[5,2] OnLeft[5,3]
 OnLeft[5,4] OnLeft[5,5] OnLeft[5,6] OnLeft[5,7] OnLeft[6,0] OnLeft[6,1]
 OnLeft[6,2] OnLeft[6,3] OnLeft[6,4] OnLeft[6,5] OnLeft[6,6] OnLeft[6,7]
 OnLeft[7,0] OnLeft[7,1] OnLeft[7,2] OnLeft[7,3] OnLeft[7,4] OnLeft[7,5]
 OnLeft[7,6] OnLeft[7,7] HorizontalOverlap[0,1] HorizontalOverlap[0,2]
 HorizontalOverlap[0,3] HorizontalOverlap[0,4] HorizontalOverlap[0,5]
 HorizontalOverlap[0,6] HorizontalOverlap[0,7] HorizontalOverlap[1,0]
 HorizontalOverlap[1,2] HorizontalOverlap[1,3] HorizontalOverlap[1,4]
 HorizontalOverlap[1,5] HorizontalOverlap[1,6] HorizontalOverlap[1,7]
 HorizontalOverlap[2,0] HorizontalOverlap[2,1] HorizontalOverlap[2,3]
 HorizontalOverlap[2,4] HorizontalOverlap[2,5] HorizontalOverlap[2,6]
 HorizontalOverlap[2,7] HorizontalOverlap[3,0] HorizontalOverlap[3,1]
 HorizontalOverlap[3,2] HorizontalOverlap[3,4] HorizontalOverlap[3,5]
 HorizontalOverlap[3,6] HorizontalOverlap[3,7] HorizontalOverlap[4,0]
 HorizontalOverlap[4,1] HorizontalOverlap[4,2] HorizontalOverlap[4,3]
 HorizontalOverlap[4,5] HorizontalOverlap[4,6] HorizontalOverlap[4,7]
 HorizontalOverlap[5,0] HorizontalOverlap[5,1] HorizontalOverlap[5,2]
 HorizontalOverlap[5,3] HorizontalOverlap[5,4] HorizontalOverlap[5,6]
 HorizontalOverlap[5,7] HorizontalOverlap[6,0] HorizontalOverlap[6,1]
 HorizontalOverlap[6,2] HorizontalOverlap[6,3] HorizontalOverlap[6,4]
 HorizontalOverlap[6,5] HorizontalOverlap[6,7] HorizontalOverlap[7,0]
 HorizontalOverlap[7,1] HorizontalOverlap[7,2] HorizontalOverlap[7,3]
 HorizontalOverlap[7,4] HorizontalOverlap[7,5] HorizontalOverlap[7,6]
 VerticalOverlap[0,0] VerticalOverlap[0,1] VerticalOverlap[0,2]
 VerticalOverlap[0,3] VerticalOverlap[0,4] VerticalOverlap[0,5]
 VerticalOverlap[0,6] VerticalOverlap[0,7] VerticalOverlap[1,0]
 VerticalOverlap[1,1] VerticalOverlap[1,2] VerticalOverlap[1,3]
 VerticalOverlap[1,4] VerticalOverlap[1,5] VerticalOverlap[1,6]
 VerticalOverlap[1,7] VerticalOverlap[2,0] VerticalOverlap[2,1]
 VerticalOverlap[2,2] VerticalOverlap[2,3] VerticalOverlap[2,4]
 VerticalOverlap[2,5] VerticalOverlap[2,6] VerticalOverlap[2,7]
 VerticalOverlap[3,0] VerticalOverlap[3,1] VerticalOverlap[3,2]
 VerticalOverlap[3,3] VerticalOverlap[3,4] VerticalOverlap[3,5]
 VerticalOverlap[3,6] VerticalOverlap[3,7] VerticalOverlap[4,0]
 VerticalOverlap[4,1] VerticalOverlap[4,2] VerticalOverlap[4,3]
 VerticalOverlap[4,4] VerticalOverlap[4,5] VerticalOverlap[4,6]
 VerticalOverlap[4,7] VerticalOverlap[5,0] VerticalOverlap[5,1]
 VerticalOverlap[5,2] VerticalOverlap[5,3] VerticalOverlap[5,4]
 VerticalOverlap[5,5] VerticalOverlap[5,6] VerticalOverlap[5,7]
 VerticalOverlap[6,0] VerticalOverlap[6,1] VerticalOverlap[6,2]
 VerticalOverlap[6,3] VerticalOverlap[6,4] VerticalOverlap[6,5]
 VerticalOverlap[6,6] VerticalOverlap[6,7] VerticalOverlap[7,0]
 VerticalOverlap[7,1] VerticalOverlap[7,2] VerticalOverlap[7,3]
 VerticalOverlap[7,4] VerticalOverlap[7,5] VerticalOverlap[7,6]
 VerticalOverlap[7,7] Overlap[0,0] Overlap[0,1] Overlap[0,2] Overlap[0,3]
 Overlap[0,4] Overlap[0,5] Overlap[0,6] Overlap[0,7] Overlap[1,0]
 Overlap[1,1] Overlap[1,2] Overlap[1,3] Overlap[1,4] Overlap[1,5]
 Overlap[1,6] Overlap[1,7] Overlap[2,0] Overlap[2,1] Overlap[2,2]
 Overlap[2,3] Overlap[2,4] Overlap[2,5] Overlap[2,6] Overlap[2,7]
 Overlap[3,0] Overlap[3,1] Overlap[3,2] Overlap[3,3] Overlap[3,4]
 Overlap[3,5] Overlap[3,6] Overlap[3,7] Overlap[4,0] Overlap[4,1]
 Overlap[4,2] Overlap[4,3] Overlap[4,4] Overlap[4,5] Overlap[4,6]
 Overlap[4,7] Overlap[5,0] Overlap[5,1] Overlap[5,2] Overlap[5,3]
 Overlap[5,4] Overlap[5,5] Overlap[5,6] Overlap[5,7] Overlap[6,0]
 Overlap[6,1] Overlap[6,2] Overlap[6,3] Overlap[6,4] Overlap[6,5]
 Overlap[6,6] Overlap[6,7] Overlap[7,0] Overlap[7,1] Overlap[7,2]
 Overlap[7,3] Overlap[7,4] Overlap[7,5] Overlap[7,6] Overlap[7,7]
 X0LessThan[0,0] X0LessThan[0,1] X0LessThan[0,2] X0LessThan[0,3]
 X0LessThan[0,4] X0LessThan[0,5] X0LessThan[0,6] X0LessThan[0,7]
 X0LessThan[1,0] X0LessThan[1,1] X0LessThan[1,2] X0LessThan[1,3]
 X0LessThan[1,4] X0LessThan[1,5] X0LessThan[1,6] X0LessThan[1,7]
 X0LessThan[2,0] X0LessThan[2,1] X0LessThan[2,2] X0LessThan[2,3]
 X0LessThan[2,4] X0LessThan[2,5] X0LessThan[2,6] X0LessThan[2,7]
 X0LessThan[3,0] X0LessThan[3,1] X0LessThan[3,2] X0LessThan[3,3]
 X0LessThan[3,4] X0LessThan[3,5] X0LessThan[3,6] X0LessThan[3,7]
 X0LessThan[4,0] X0LessThan[4,1] X0LessThan[4,2] X0LessThan[4,3]
 X0LessThan[4,4] X0LessThan[4,5] X0LessThan[4,6] X0LessThan[4,7]
 X0LessThan[5,0] X0LessThan[5,1] X0LessThan[5,2] X0LessThan[5,3]
 X0LessThan[5,4] X0LessThan[5,5] X0LessThan[5,6] X0LessThan[5,7]
 X0LessThan[6,0] X0LessThan[6,1] X0LessThan[6,2] X0LessThan[6,3]
 X0LessThan[6,4] X0LessThan[6,5] X0LessThan[6,6] X0LessThan[6,7]
 X0LessThan[7,0] X0LessThan[7,1] X0LessThan[7,2] X0LessThan[7,3]
 X0LessThan[7,4] X0LessThan[7,5] X0LessThan[7,6] X0LessThan[7,7]
 Y0LessThan[0,0] Y0LessThan[0,1] Y0LessThan[0,2] Y0LessThan[0,3]
 Y0LessThan[0,4] Y0LessThan[0,5] Y0LessThan[0,6] Y0LessThan[0,7]
 Y0LessThan[1,0] Y0LessThan[1,1] Y0LessThan[1,2] Y0LessThan[1,3]
 Y0LessThan[1,4] Y0LessThan[1,5] Y0LessThan[1,6] Y0LessThan[1,7]
 Y0LessThan[2,0] Y0LessThan[2,1] Y0LessThan[2,2] Y0LessThan[2,3]
 Y0LessThan[2,4] Y0LessThan[2,5] Y0LessThan[2,6] Y0LessThan[2,7]
 Y0LessThan[3,0] Y0LessThan[3,1] Y0LessThan[3,2] Y0LessThan[3,3]
 Y0LessThan[3,4] Y0LessThan[3,5] Y0LessThan[3,6] Y0LessThan[3,7]
 Y0LessThan[4,0] Y0LessThan[4,1] Y0LessThan[4,2] Y0LessThan[4,3]
 Y0LessThan[4,4] Y0LessThan[4,5] Y0LessThan[4,6] Y0LessThan[4,7]
 Y0LessThan[5,0] Y0LessThan[5,1] Y0LessThan[5,2] Y0LessThan[5,3]
 Y0LessThan[5,4] Y0LessThan[5,5] Y0LessThan[5,6] Y0LessThan[5,7]
 Y0LessThan[6,0] Y0LessThan[6,1] Y0LessThan[6,2] Y0LessThan[6,3]
 Y0LessThan[6,4] Y0LessThan[6,5] Y0LessThan[6,6] Y0LessThan[6,7]
 Y0LessThan[7,0] Y0LessThan[7,1] Y0LessThan[7,2] Y0LessThan[7,3]
 Y0LessThan[7,4] Y0LessThan[7,5] Y0LessThan[7,6] Y0LessThan[7,7]
 X1LessThan[0,0] X1LessThan[0,1] X1LessThan[0,2] X1LessThan[0,3]
 X1LessThan[0,4] X1LessThan[0,5] X1LessThan[0,6] X1LessThan[0,7]
 X1LessThan[1,0] X1LessThan[1,1] X1LessThan[1,2] X1LessThan[1,3]
 X1LessThan[1,4] X1LessThan[1,5] X1LessThan[1,6] X1LessThan[1,7]
 X1LessThan[2,0] X1LessThan[2,1] X1LessThan[2,2] X1LessThan[2,3]
 X1LessThan[2,4] X1LessThan[2,5] X1LessThan[2,6] X1LessThan[2,7]
 X1LessThan[3,0] X1LessThan[3,1] X1LessThan[3,2] X1LessThan[3,3]
 X1LessThan[3,4] X1LessThan[3,5] X1LessThan[3,6] X1LessThan[3,7]
 X1LessThan[4,0] X1LessThan[4,1] X1LessThan[4,2] X1LessThan[4,3]
 X1LessThan[4,4] X1LessThan[4,5] X1LessThan[4,6] X1LessThan[4,7]
 X1LessThan[5,0] X1LessThan[5,1] X1LessThan[5,2] X1LessThan[5,3]
 X1LessThan[5,4] X1LessThan[5,5] X1LessThan[5,6] X1LessThan[5,7]
 X1LessThan[6,0] X1LessThan[6,1] X1LessThan[6,2] X1LessThan[6,3]
 X1LessThan[6,4] X1LessThan[6,5] X1LessThan[6,6] X1LessThan[6,7]
 X1LessThan[7,0] X1LessThan[7,1] X1LessThan[7,2] X1LessThan[7,3]
 X1LessThan[7,4] X1LessThan[7,5] X1LessThan[7,6] X1LessThan[7,7]
 Y1LessThan[0,0] Y1LessThan[0,1] Y1LessThan[0,2] Y1LessThan[0,3]
 Y1LessThan[0,4] Y1LessThan[0,5] Y1LessThan[0,6] Y1LessThan[0,7]
 Y1LessThan[1,0] Y1LessThan[1,1] Y1LessThan[1,2] Y1LessThan[1,3]
 Y1LessThan[1,4] Y1LessThan[1,5] Y1LessThan[1,6] Y1LessThan[1,7]
 Y1LessThan[2,0] Y1LessThan[2,1] Y1LessThan[2,2] Y1LessThan[2,3]
 Y1LessThan[2,4] Y1LessThan[2,5] Y1LessThan[2,6] Y1LessThan[2,7]
 Y1LessThan[3,0] Y1LessThan[3,1] Y1LessThan[3,2] Y1LessThan[3,3]
 Y1LessThan[3,4] Y1LessThan[3,5] Y1LessThan[3,6] Y1LessThan[3,7]
 Y1LessThan[4,0] Y1LessThan[4,1] Y1LessThan[4,2] Y1LessThan[4,3]
 Y1LessThan[4,4] Y1LessThan[4,5] Y1LessThan[4,6] Y1LessThan[4,7]
 Y1LessThan[5,0] Y1LessThan[5,1] Y1LessThan[5,2] Y1LessThan[5,3]
 Y1LessThan[5,4] Y1LessThan[5,5] Y1LessThan[5,6] Y1LessThan[5,7]
 Y1LessThan[6,0] Y1LessThan[6,1] Y1LessThan[6,2] Y1LessThan[6,3]
 Y1LessThan[6,4] Y1LessThan[6,5] Y1LessThan[6,6] Y1LessThan[6,7]
 Y1LessThan[7,0] Y1LessThan[7,1] Y1LessThan[7,2] Y1LessThan[7,3]
 Y1LessThan[7,4] Y1LessThan[7,5] Y1LessThan[7,6] Y1LessThan[7,7]
Generals
 X0[0] X0[1] X0[2] X0[3] X0[4] X0[5] X0[6] X0[7] Y0[0] Y0[1] Y0[2] Y0[3]
 Y0[4] Y0[5] Y0[6] Y0[7] X1[0] X1[1] X1[2] X1[3] X1[4] X1[5] X1[6] X1[7]
 Y1[0] Y1[1] Y1[2] Y1[3] Y1[4] Y1[5] Y1[6] Y1[7] W[0] W[1] W[2] W[3] W[4]
 W[5] W[6] W[7] H[0] H[1] H[2] H[3] H[4] H[5] H[6] H[7] ResizeW[0]
 ResizeW[1] ResizeW[2] ResizeW[3] ResizeW[4] ResizeW[5] ResizeW[6]
 ResizeW[7] ResizeWAbs[0] ResizeWAbs[1] ResizeWAbs[2] ResizeWAbs[3]
 ResizeWAbs[4] ResizeWAbs[5] ResizeWAbs[6] ResizeWAbs[7] ResizeH[0]
 ResizeH[1] ResizeH[2] ResizeH[3] ResizeH[4] ResizeH[5] ResizeH[6]
 ResizeH[7] ResizeHAbs[0] ResizeHAbs[1] ResizeHAbs[2] ResizeHAbs[3]
 ResizeHAbs[4] ResizeHAbs[5] ResizeHAbs[6] ResizeHAbs[7] MoveX[0] MoveX[1]
 MoveX[2] MoveX[3] MoveX[4] MoveX[5] MoveX[6] MoveX[7] MoveXAbs[0]
 MoveXAbs[1] MoveXAbs[2] MoveXAbs[3] MoveXAbs[4] MoveXAbs[5] MoveXAbs[6]
 MoveXAbs[7] MoveY[0] MoveY[1] MoveY[2] MoveY[3] MoveY[4] MoveY[5] MoveY[6]
 MoveY[7] MoveYAbs[0] MoveYAbs[1] MoveYAbs[2] MoveYAbs[3] MoveYAbs[4]
 MoveYAbs[5] MoveYAbs[6] MoveYAbs[7] EdgeDistance[x0,0,0]
 EdgeDistance[x0,0,1] EdgeDistance[x0,0,2] EdgeDistance[x0,0,3]
 EdgeDistance[x0,0,4] EdgeDistance[x0,0,5] EdgeDistance[x0,0,6]
 EdgeDistance[x0,0,7] EdgeDistance[x0,1,0] EdgeDistance[x0,1,1]
 EdgeDistance[x0,1,2] EdgeDistance[x0,1,3] EdgeDistance[x0,1,4]
 EdgeDistance[x0,1,5] EdgeDistance[x0,1,6] EdgeDistance[x0,1,7]
 EdgeDistance[x0,2,0] EdgeDistance[x0,2,1] EdgeDistance[x0,2,2]
 EdgeDistance[x0,2,3] EdgeDistance[x0,2,4] EdgeDistance[x0,2,5]
 EdgeDistance[x0,2,6] EdgeDistance[x0,2,7] EdgeDistance[x0,3,0]
 EdgeDistance[x0,3,1] EdgeDistance[x0,3,2] EdgeDistance[x0,3,3]
 EdgeDistance[x0,3,4] EdgeDistance[x0,3,5] EdgeDistance[x0,3,6]
 EdgeDistance[x0,3,7] EdgeDistance[x0,4,0] EdgeDistance[x0,4,1]
 EdgeDistance[x0,4,2] EdgeDistance[x0,4,3] EdgeDistance[x0,4,4]
 EdgeDistance[x0,4,5] EdgeDistance[x0,4,6] EdgeDistance[x0,4,7]
 EdgeDistance[x0,5,0] EdgeDistance[x0,5,1] EdgeDistance[x0,5,2]
 EdgeDistance[x0,5,3] EdgeDistance[x0,5,4] EdgeDistance[x0,5,5]
 EdgeDistance[x0,5,6] EdgeDistance[x0,5,7] EdgeDistance[x0,6,0]
 EdgeDistance[x0,6,1] EdgeDistance[x0,6,2] EdgeDistance[x0,6,3]
 EdgeDistance[x0,6,4] EdgeDistance[x0,6,5] EdgeDistance[x0,6,6]
 EdgeDistance[x0,6,7] EdgeDistance[x0,7,0] EdgeDistance[x0,7,1]
 EdgeDistance[x0,7,2] EdgeDistance[x0,7,3] EdgeDistance[x0,7,4]
 EdgeDistance[x0,7,5] EdgeDistance[x0,7,6] EdgeDistance[x0,7,7]
 EdgeDistance[y0,0,0] EdgeDistance[y0,0,1] EdgeDistance[y0,0,2]
 EdgeDistance[y0,0,3] EdgeDistance[y0,0,4] EdgeDistance[y0,0,5]
 EdgeDistance[y0,0,6] EdgeDistance[y0,0,7] EdgeDistance[y0,1,0]
 EdgeDistance[y0,1,1] EdgeDistance[y0,1,2] EdgeDistance[y0,1,3]
 EdgeDistance[y0,1,4] EdgeDistance[y0,1,5] EdgeDistance[y0,1,6]
 EdgeDistance[y0,1,7] EdgeDistance[y0,2,0] EdgeDistance[y0,2,1]
 EdgeDistance[y0,2,2] EdgeDistance[y0,2,3] EdgeDistance[y0,2,4]
 EdgeDistance[y0,2,5] EdgeDistance[y0,2,6] EdgeDistance[y0,2,7]
 EdgeDistance[y0,3,0] EdgeDistance[y0,3,1] EdgeDistance[y0,3,2]
 EdgeDistance[y0,3,3] EdgeDistance[y0,3,4] EdgeDistance[y0,3,5]
 EdgeDistance[y0,3,6] EdgeDistance[y0,3,7] EdgeDistance[y0,4,0]
 EdgeDistance[y0,4,1] EdgeDistance[y0,4,2] EdgeDistance[y0,4,3]
 EdgeDistance[y0,4,4] EdgeDistance[y0,4,5] EdgeDistance[y0,4,6]
 EdgeDistance[y0,4,7] EdgeDistance[y0,5,0] EdgeDistance[y0,5,1]
 EdgeDistance[y0,5,2] EdgeDistance[y0,5,3] EdgeDistance[y0,5,4]
 EdgeDistance[y0,5,5] EdgeDistance[y0,5,6] EdgeDistance[y0,5,7]
 EdgeDistance[y0,6,0] EdgeDistance[y0,6,1] EdgeDistance[y0,6,2]
 EdgeDistance[y0,6,3] EdgeDistance[y0,6,4] EdgeDistance[y0,6,5]
 EdgeDistance[y0,6,6] EdgeDistance[y0,6,7] EdgeDistance[y0,7,0]
 EdgeDistance[y0,7,1] EdgeDistance[y0,7,2] EdgeDistance[y0,7,3]
 EdgeDistance[y0,7,4] EdgeDistance[y0,7,5] EdgeDistance[y0,7,6]
 EdgeDistance[y0,7,7] EdgeDistance[x1,0,0] EdgeDistance[x1,0,1]
 EdgeDistance[x1,0,2] EdgeDistance[x1,0,3] EdgeDistance[x1,0,4]
 EdgeDistance[x1,0,5] EdgeDistance[x1,0,6] EdgeDistance[x1,0,7]
 EdgeDistance[x1,1,0] EdgeDistance[x1,1,1] EdgeDistance[x1,1,2]
 EdgeDistance[x1,1,3] EdgeDistance[x1,1,4] EdgeDistance[x1,1,5]
 EdgeDistance[x1,1,6] EdgeDistance[x1,1,7] EdgeDistance[x1,2,0]
 EdgeDistance[x1,2,1] EdgeDistance[x1,2,2] EdgeDistance[x1,2,3]
 EdgeDistance[x1,2,4] EdgeDistance[x1,2,5] EdgeDistance[x1,2,6]
 EdgeDistance[x1,2,7] EdgeDistance[x1,3,0] EdgeDistance[x1,3,1]
 EdgeDistance[x1,3,2] EdgeDistance[x1,3,3] EdgeDistance[x1,3,4]
 EdgeDistance[x1,3,5] EdgeDistance[x1,3,6] EdgeDistance[x1,3,7]
 EdgeDistance[x1,4,0] EdgeDistance[x1,4,1] EdgeDistance[x1,4,2]
 EdgeDistance[x1,4,3] EdgeDistance[x1,4,4] EdgeDistance[x1,4,5]
 EdgeDistance[x1,4,6] EdgeDistance[x1,4,7] EdgeDistance[x1,5,0]
 EdgeDistance[x1,5,1] EdgeDistance[x1,5,2] EdgeDistance[x1,5,3]
 EdgeDistance[x1,5,4] EdgeDistance[x1,5,5] EdgeDistance[x1,5,6]
 EdgeDistance[x1,5,7] EdgeDistance[x1,6,0] EdgeDistance[x1,6,1]
 EdgeDistance[x1,6,2] EdgeDistance[x1,6,3] EdgeDistance[x1,6,4]
 EdgeDistance[x1,6,5] EdgeDistance[x1,6,6] EdgeDistance[x1,6,7]
 EdgeDistance[x1,7,0] EdgeDistance[x1,7,1] EdgeDistance[x1,7,2]
 EdgeDistance[x1,7,3] EdgeDistance[x1,7,4] EdgeDistance[x1,7,5]
 EdgeDistance[x1,7,6] EdgeDistance[x1,7,7] EdgeDistance[y1,0,0]
 EdgeDistance[y1,0,1] EdgeDistance[y1,0,2] EdgeDistance[y1,0,3]
 EdgeDistance[y1,0,4] EdgeDistance[y1,0,5] EdgeDistance[y1,0,6]
 EdgeDistance[y1,0,7] EdgeDistance[y1,1,0] EdgeDistance[y1,1,1]
 EdgeDistance[y1,1,2] EdgeDistance[y1,1,3] EdgeDistance[y1,1,4]
 EdgeDistance[y1,1,5] EdgeDistance[y1,1,6] EdgeDistance[y1,1,7]
 EdgeDistance[y1,2,0] EdgeDistance[y1,2,1] EdgeDistance[y1,2,2]
 EdgeDistance[y1,2,3] EdgeDistance[y1,2,4] EdgeDistance[y1,2,5]
 EdgeDistance[y1,2,6] EdgeDistance[y1,2,7] EdgeDistance[y1,3,0]
 EdgeDistance[y1,3,1] EdgeDistance[y1,3,2] EdgeDistance[y1,3,3]
 EdgeDistance[y1,3,4] EdgeDistance[y1,3,5] EdgeDistance[y1,3,6]
 EdgeDistance[y1,3,7] EdgeDistance[y1,4,0] EdgeDistance[y1,4,1]
 EdgeDistance[y1,4,2] EdgeDistance[y1,4,3] EdgeDistance[y1,4,4]
 EdgeDistance[y1,4,5] EdgeDistance[y1,4,6] EdgeDistance[y1,4,7]
 EdgeDistance[y1,5,0] EdgeDistance[y1,5,1] EdgeDistance[y1,5,2]
 EdgeDistance[y1,5,3] EdgeDistance[y1,5,4] EdgeDistance[y1,5,5]
 EdgeDistance[y1,5,6] EdgeDistance[y1,5,7] EdgeDistance[y1,6,0]
 EdgeDistance[y1,6,1] EdgeDistance[y1,6,2] EdgeDistance[y1,6,3]
 EdgeDistance[y1,6,4] EdgeDistance[y1,6,5] EdgeDistance[y1,6,6]
 EdgeDistance[y1,6,7] EdgeDistance[y1,7,0] EdgeDistance[y1,7,1]
 EdgeDistance[y1,7,2] EdgeDistance[y1,7,3] EdgeDistance[y1,7,4]
 EdgeDistance[y1,7,5] EdgeDistance[y1,7,6] EdgeDistance[y1,7,7] X0Diff[0,0]
 X0Diff[0,1] X0Diff[0,2] X0Diff[0,3] X0Diff[0,4] X0Diff[0,5] X0Diff[0,6]
 X0Diff[0,7] X0Diff[1,0] X0Diff[1,1] X0Diff[1,2] X0Diff[1,3] X0Diff[1,4]
 X0Diff[1,5] X0Diff[1,6] X0Diff[1,7] X0Diff[2,0] X0Diff[2,1] X0Diff[2,2]
 X0Diff[2,3] X0Diff[2,4] X0Diff[2,5] X0Diff[2,6] X0Diff[2,7] X0Diff[3,0]
 X0Diff[3,1] X0Diff[3,2] X0Diff[3,3] X0Diff[3,4] X0Diff[3,5] X0Diff[3,6]
 X0Diff[3,7] X0Diff[4,0] X0Diff[4,1] X0Diff[4,2] X0Diff[4,3] X0Diff[4,4]
 X0Diff[4,5] X0Diff[4,6] X0Diff[4,7] X0Diff[5,0] X0Diff[5,1] X0Diff[5,2]
 X0Diff[5,3] X0Diff[5,4] X0Diff[5,5] X0Diff[5,6] X0Diff[5,7] X0Diff[6,0]
 X0Diff[6,1] X0Diff[6,2] X0Diff[6,3] X0Diff[6,4] X0Diff[6,5] X0Diff[6,6]
 X0Diff[6,7] X0Diff[7,0] X0Diff[7,1] X0Diff[7,2] X0Diff[7,3] X0Diff[7,4]
 X0Diff[7,5] X0Diff[7,6] X0Diff[7,7] Y0Diff[0,0] Y0Diff[0,1] Y0Diff[0,2]
 Y0Diff[0,3] Y0Diff[0,4] Y0Diff[0,5] Y0Diff[0,6] Y0Diff[0,7] Y0Diff[1,0]
 Y0Diff[1,1] Y0Diff[1,2] Y0Diff[1,3] Y0Diff[1,4] Y0Diff[1,5] Y0Diff[1,6]
 Y0Diff[1,7] Y0Diff[2,0] Y0Diff[2,1] Y0Diff[2,2] Y0Diff[2,3] Y0Diff[2,4]
 Y0Diff[2,5] Y0Diff[2,6] Y0Diff[2,7] Y0Diff[3,0] Y0Diff[3,1] Y0Diff[3,2]
 Y0Diff[3,3] Y0Diff[3,4] Y0Diff[3,5] Y0Diff[3,6] Y0Diff[3,7] Y0Diff[4,0]
 Y0Diff[4,1] Y0Diff[4,2] Y0Diff[4,3] Y0Diff[4,4] Y0Diff[4,5] Y0Diff[4,6]
 Y0Diff[4,7] Y0Diff[5,0] Y0Diff[5,1] Y0Diff[5,2] Y0Diff[5,3] Y0Diff[5,4]
 Y0Diff[5,5] Y0Diff[5,6] Y0Diff[5,7] Y0Diff[6,0] Y0Diff[6,1] Y0Diff[6,2]
 Y0Diff[6,3] Y0Diff[6,4] Y0Diff[6,5] Y0Diff[6,6] Y0Diff[6,7] Y0Diff[7,0]
 Y0Diff[7,1] Y0Diff[7,2] Y0Diff[7,3] Y0Diff[7,4] Y0Diff[7,5] Y0Diff[7,6]
 Y0Diff[7,7] X1Diff[0,0] X1Diff[0,1] X1Diff[0,2] X1Diff[0,3] X1Diff[0,4]
 X1Diff[0,5] X1Diff[0,6] X1Diff[0,7] X1Diff[1,0] X1Diff[1,1] X1Diff[1,2]
 X1Diff[1,3] X1Diff[1,4] X1Diff[1,5] X1Diff[1,6] X1Diff[1,7] X1Diff[2,0]
 X1Diff[2,1] X1Diff[2,2] X1Diff[2,3] X1Diff[2,4] X1Diff[2,5] X1Diff[2,6]
 X1Diff[2,7] X1Diff[3,0] X1Diff[3,1] X1Diff[3,2] X1Diff[3,3] X1Diff[3,4]
 X1Diff[3,5] X1Diff[3,6] X1Diff[3,7] X1Diff[4,0] X1Diff[4,1] X1Diff[4,2]
 X1Diff[4,3] X1Diff[4,4] X1Diff[4,5] X1Diff[4,6] X1Diff[4,7] X1Diff[5,0]
 X1Diff[5,1] X1Diff[5,2] X1Diff[5,3] X1Diff[5,4] X1Diff[5,5] X1Diff[5,6]
 X1Diff[5,7] X1Diff[6,0] X1Diff[6,1] X1Diff[6,2] X1Diff[6,3] X1Diff[6,4]
 X1Diff[6,5] X1Diff[6,6] X1Diff[6,7] X1Diff[7,0] X1Diff[7,1] X1Diff[7,2]
 X1Diff[7,3] X1Diff[7,4] X1Diff[7,5] X1Diff[7,6] X1Diff[7,7] Y1Diff[0,0]
 Y1Diff[0,1] Y1Diff[0,2] Y1Diff[0,3] Y1Diff[0,4] Y1Diff[0,5] Y1Diff[0,6]
 Y1Diff[0,7] Y1Diff[1,0] Y1Diff[1,1] Y1Diff[1,2] Y1Diff[1,3] Y1Diff[1,4]
 Y1Diff[1,5] Y1Diff[1,6] Y1Diff[1,7] Y1Diff[2,0] Y1Diff[2,1] Y1Diff[2,2]
 Y1Diff[2,3] Y1Diff[2,4] Y1Diff[2,5] Y1Diff[2,6] Y1Diff[2,7] Y1Diff[3,0]
 Y1Diff[3,1] Y1Diff[3,2] Y1Diff[3,3] Y1Diff[3,4] Y1Diff[3,5] Y1Diff[3,6]
 Y1Diff[3,7] Y1Diff[4,0] Y1Diff[4,1] Y1Diff[4,2] Y1Diff[4,3] Y1Diff[4,4]
 Y1Diff[4,5] Y1Diff[4,6] Y1Diff[4,7] Y1Diff[5,0] Y1Diff[5,1] Y1Diff[5,2]
 Y1Diff[5,3] Y1Diff[5,4] Y1Diff[5,5] Y1Diff[5,6] Y1Diff[5,7] Y1Diff[6,0]
 Y1Diff[6,1] Y1Diff[6,2] Y1Diff[6,3] Y1Diff[6,4] Y1Diff[6,5] Y1Diff[6,6]
 Y1Diff[6,7] Y1Diff[7,0] Y1Diff[7,1] Y1Diff[7,2] Y1Diff[7,3] Y1Diff[7,4]
 Y1Diff[7,5] Y1Diff[7,6] Y1Diff[7,7] X0Group[0] X0Group[1] X0Group[2]
 X0Group[3] X0Group[4] X0Group[5] X0Group[6] X0Group[7] Y0Group[0]
 Y0Group[1] Y0Group[2] Y0Group[3] Y0Group[4] Y0Group[5] Y0Group[6]
 Y0Group[7] X1Group[0] X1Group[1] X1Group[2] X1Group[3] X1Group[4]
 X1Group[5] X1Group[6] X1Group[7] Y1Group[0] Y1Group[1] Y1Group[2]
 Y1Group[3] Y1Group[4] Y1Group[5] Y1Group[6] Y1Group[7] X0GroupCount
 Y0GroupCount X1GroupCount Y1GroupCount LeftMargin MaxX1 MaxX1
 MarginDiffLooseAbs C1232
General Constraints
 LinkResizeWAbs[0]: ResizeWAbs[0] = ABS ( ResizeW[0] )
 LinkResizeWAbs[1]: ResizeWAbs[1] = ABS ( ResizeW[1] )
 LinkResizeWAbs[2]: ResizeWAbs[2] = ABS ( ResizeW[2] )
 LinkResizeWAbs[3]: ResizeWAbs[3] = ABS ( ResizeW[3] )
 LinkResizeWAbs[4]: ResizeWAbs[4] = ABS ( ResizeW[4] )
 LinkResizeWAbs[5]: ResizeWAbs[5] = ABS ( ResizeW[5] )
 LinkResizeWAbs[6]: ResizeWAbs[6] = ABS ( ResizeW[6] )
 LinkResizeWAbs[7]: ResizeWAbs[7] = ABS ( ResizeW[7] )
 LinkResizeHAbs[0]: ResizeHAbs[0] = ABS ( ResizeH[0] )
 LinkResizeHAbs[1]: ResizeHAbs[1] = ABS ( ResizeH[1] )
 LinkResizeHAbs[2]: ResizeHAbs[2] = ABS ( ResizeH[2] )
 LinkResizeHAbs[3]: ResizeHAbs[3] = ABS ( ResizeH[3] )
 LinkResizeHAbs[4]: ResizeHAbs[4] = ABS ( ResizeH[4] )
 LinkResizeHAbs[5]: ResizeHAbs[5] = ABS ( ResizeH[5] )
 LinkResizeHAbs[6]: ResizeHAbs[6] = ABS ( ResizeH[6] )
 LinkResizeHAbs[7]: ResizeHAbs[7] = ABS ( ResizeH[7] )
 LinkMoveXAbs[0]: MoveXAbs[0] = ABS ( MoveX[0] )
 LinkMoveXAbs[1]: MoveXAbs[1] = ABS ( MoveX[1] )
 LinkMoveXAbs[2]: MoveXAbs[2] = ABS ( MoveX[2] )
 LinkMoveXAbs[3]: MoveXAbs[3] = ABS ( MoveX[3] )
 LinkMoveXAbs[4]: MoveXAbs[4] = ABS ( MoveX[4] )
 LinkMoveXAbs[5]: MoveXAbs[5] = ABS ( MoveX[5] )
 LinkMoveXAbs[6]: MoveXAbs[6] = ABS ( MoveX[6] )
 LinkMoveXAbs[7]: MoveXAbs[7] = ABS ( MoveX[7] )
 LinkMoveYAbs[0]: MoveYAbs[0] = ABS ( MoveY[0] )
 LinkMoveYAbs[1]: MoveYAbs[1] = ABS ( MoveY[1] )
 LinkMoveYAbs[2]: MoveYAbs[2] = ABS ( MoveY[2] )
 LinkMoveYAbs[3]: MoveYAbs[3] = ABS ( MoveY[3] )
 LinkMoveYAbs[4]: MoveYAbs[4] = ABS ( MoveY[4] )
 LinkMoveYAbs[5]: MoveYAbs[5] = ABS ( MoveY[5] )
 LinkMoveYAbs[6]: MoveYAbs[6] = ABS ( MoveY[6] )
 LinkMoveYAbs[7]: MoveYAbs[7] = ABS ( MoveY[7] )
 LinkOverlap[0,1]: Overlap[0,1] = AND ( HorizontalOverlap[0,1] ,
   VerticalOverlap[0,1] )
 LinkOverlap[0,2]: Overlap[0,2] = AND ( HorizontalOverlap[0,2] ,
   VerticalOverlap[0,2] )
 LinkOverlap[0,3]: Overlap[0,3] = AND ( HorizontalOverlap[0,3] ,
   VerticalOverlap[0,3] )
 LinkOverlap[0,4]: Overlap[0,4] = AND ( HorizontalOverlap[0,4] ,
   VerticalOverlap[0,4] )
 LinkOverlap[0,5]: Overlap[0,5] = AND ( HorizontalOverlap[0,5] ,
   VerticalOverlap[0,5] )
 LinkOverlap[0,6]: Overlap[0,6] = AND ( HorizontalOverlap[0,6] ,
   VerticalOverlap[0,6] )
 LinkOverlap[0,7]: Overlap[0,7] = AND ( HorizontalOverlap[0,7] ,
   VerticalOverlap[0,7] )
 LinkOverlap[1,0]: Overlap[1,0] = AND ( HorizontalOverlap[1,0] ,
   VerticalOverlap[1,0] )
 LinkOverlap[1,2]: Overlap[1,2] = AND ( HorizontalOverlap[1,2] ,
   VerticalOverlap[1,2] )
 LinkOverlap[1,3]: Overlap[1,3] = AND ( HorizontalOverlap[1,3] ,
   VerticalOverlap[1,3] )
 LinkOverlap[1,4]: Overlap[1,4] = AND ( HorizontalOverlap[1,4] ,
   VerticalOverlap[1,4] )
 LinkOverlap[1,5]: Overlap[1,5] = AND ( HorizontalOverlap[1,5] ,
   VerticalOverlap[1,5] )
 LinkOverlap[1,6]: Overlap[1,6] = AND ( HorizontalOverlap[1,6] ,
   VerticalOverlap[1,6] )
 LinkOverlap[1,7]: Overlap[1,7] = AND ( HorizontalOverlap[1,7] ,
   VerticalOverlap[1,7] )
 LinkOverlap[2,0]: Overlap[2,0] = AND ( HorizontalOverlap[2,0] ,
   VerticalOverlap[2,0] )
 LinkOverlap[2,1]: Overlap[2,1] = AND ( HorizontalOverlap[2,1] ,
   VerticalOverlap[2,1] )
 LinkOverlap[2,3]: Overlap[2,3] = AND ( HorizontalOverlap[2,3] ,
   VerticalOverlap[2,3] )
 LinkOverlap[2,4]: Overlap[2,4] = AND ( HorizontalOverlap[2,4] ,
   VerticalOverlap[2,4] )
 LinkOverlap[2,5]: Overlap[2,5] = AND ( HorizontalOverlap[2,5] ,
   VerticalOverlap[2,5] )
 LinkOverlap[2,6]: Overlap[2,6] = AND ( HorizontalOverlap[2,6] ,
   VerticalOverlap[2,6] )
 LinkOverlap[2,7]: Overlap[2,7] = AND ( HorizontalOverlap[2,7] ,
   VerticalOverlap[2,7] )
 LinkOverlap[3,0]: Overlap[3,0] = AND ( HorizontalOverlap[3,0] ,
   VerticalOverlap[3,0] )
 LinkOverlap[3,1]: Overlap[3,1] = AND ( HorizontalOverlap[3,1] ,
   VerticalOverlap[3,1] )
 LinkOverlap[3,2]: Overlap[3,2] = AND ( HorizontalOverlap[3,2] ,
   VerticalOverlap[3,2] )
 LinkOverlap[3,4]: Overlap[3,4] = AND ( HorizontalOverlap[3,4] ,
   VerticalOverlap[3,4] )
 LinkOverlap[3,5]: Overlap[3,5] = AND ( HorizontalOverlap[3,5] ,
   VerticalOverlap[3,5] )
 LinkOverlap[3,6]: Overlap[3,6] = AND ( HorizontalOverlap[3,6] ,
   VerticalOverlap[3,6] )
 LinkOverlap[3,7]: Overlap[3,7] = AND ( HorizontalOverlap[3,7] ,
   VerticalOverlap[3,7] )
 LinkOverlap[4,0]: Overlap[4,0] = AND ( HorizontalOverlap[4,0] ,
   VerticalOverlap[4,0] )
 LinkOverlap[4,1]: Overlap[4,1] = AND ( HorizontalOverlap[4,1] ,
   VerticalOverlap[4,1] )
 LinkOverlap[4,2]: Overlap[4,2] = AND ( HorizontalOverlap[4,2] ,
   VerticalOverlap[4,2] )
 LinkOverlap[4,3]: Overlap[4,3] = AND ( HorizontalOverlap[4,3] ,
   VerticalOverlap[4,3] )
 LinkOverlap[4,5]: Overlap[4,5] = AND ( HorizontalOverlap[4,5] ,
   VerticalOverlap[4,5] )
 LinkOverlap[4,6]: Overlap[4,6] = AND ( HorizontalOverlap[4,6] ,
   VerticalOverlap[4,6] )
 LinkOverlap[4,7]: Overlap[4,7] = AND ( HorizontalOverlap[4,7] ,
   VerticalOverlap[4,7] )
 LinkOverlap[5,0]: Overlap[5,0] = AND ( HorizontalOverlap[5,0] ,
   VerticalOverlap[5,0] )
 LinkOverlap[5,1]: Overlap[5,1] = AND ( HorizontalOverlap[5,1] ,
   VerticalOverlap[5,1] )
 LinkOverlap[5,2]: Overlap[5,2] = AND ( HorizontalOverlap[5,2] ,
   VerticalOverlap[5,2] )
 LinkOverlap[5,3]: Overlap[5,3] = AND ( HorizontalOverlap[5,3] ,
   VerticalOverlap[5,3] )
 LinkOverlap[5,4]: Overlap[5,4] = AND ( HorizontalOverlap[5,4] ,
   VerticalOverlap[5,4] )
 LinkOverlap[5,6]: Overlap[5,6] = AND ( HorizontalOverlap[5,6] ,
   VerticalOverlap[5,6] )
 LinkOverlap[5,7]: Overlap[5,7] = AND ( HorizontalOverlap[5,7] ,
   VerticalOverlap[5,7] )
 LinkOverlap[6,0]: Overlap[6,0] = AND ( HorizontalOverlap[6,0] ,
   VerticalOverlap[6,0] )
 LinkOverlap[6,1]: Overlap[6,1] = AND ( HorizontalOverlap[6,1] ,
   VerticalOverlap[6,1] )
 LinkOverlap[6,2]: Overlap[6,2] = AND ( HorizontalOverlap[6,2] ,
   VerticalOverlap[6,2] )
 LinkOverlap[6,3]: Overlap[6,3] = AND ( HorizontalOverlap[6,3] ,
   VerticalOverlap[6,3] )
 LinkOverlap[6,4]: Overlap[6,4] = AND ( HorizontalOverlap[6,4] ,
   VerticalOverlap[6,4] )
 LinkOverlap[6,5]: Overlap[6,5] = AND ( HorizontalOverlap[6,5] ,
   VerticalOverlap[6,5] )
 LinkOverlap[6,7]: Overlap[6,7] = AND ( HorizontalOverlap[6,7] ,
   VerticalOverlap[6,7] )
 LinkOverlap[7,0]: Overlap[7,0] = AND ( HorizontalOverlap[7,0] ,
   VerticalOverlap[7,0] )
 LinkOverlap[7,1]: Overlap[7,1] = AND ( HorizontalOverlap[7,1] ,
   VerticalOverlap[7,1] )
 LinkOverlap[7,2]: Overlap[7,2] = AND ( HorizontalOverlap[7,2] ,
   VerticalOverlap[7,2] )
 LinkOverlap[7,3]: Overlap[7,3] = AND ( HorizontalOverlap[7,3] ,
   VerticalOverlap[7,3] )
 LinkOverlap[7,4]: Overlap[7,4] = AND ( HorizontalOverlap[7,4] ,
   VerticalOverlap[7,4] )
 LinkOverlap[7,5]: Overlap[7,5] = AND ( HorizontalOverlap[7,5] ,
   VerticalOverlap[7,5] )
 LinkOverlap[7,6]: Overlap[7,6] = AND ( HorizontalOverlap[7,6] ,
   VerticalOverlap[7,6] )
 GC1208: X0GroupCount = MAX ( X0Group[0] , X0Group[1] , X0Group[2] ,
   X0Group[3] , X0Group[4] , X0Group[5] , X0Group[6] , X0Group[7] )
 GC1209: Y0GroupCount = MAX ( Y0Group[0] , Y0Group[1] , Y0Group[2] ,
   Y0Group[3] , Y0Group[4] , Y0Group[5] , Y0Group[6] , Y0Group[7] )
 GC1210: X1GroupCount = MAX ( X1Group[0] , X1Group[1] , X1Group[2] ,
   X1Group[3] , X1Group[4] , X1Group[5] , X1Group[6] , X1Group[7] )
 GC1211: Y1GroupCount = MAX ( Y1Group[0] , Y1Group[1] , Y1Group[2] ,
   Y1Group[3] , Y1Group[4] , Y1Group[5] , Y1Group[6] , Y1Group[7] )
 LinkLeftMargin: LeftMargin = MIN ( X0[0] , X0[1] , X0[2] , X0[3] , X0[4] ,
   X0[5] , X0[6] , X0[7] )
 LinkMaxX1: MaxX1 = MAX ( X1[0] , X1[1] , X1[2] , X1[3] , X1[4] , X1[5] ,
   X1[6] , X1[7] )
End
