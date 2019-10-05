\ Model GLayoutQuality
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.078125 ResizeWAbs[0] + 0.078125 ResizeWAbs[1] + 0.078125 ResizeWAbs[2]
   + 0.078125 ResizeWAbs[3] + 0.0178571428571429 ResizeWAbs[4]
   + 0.078125 ResizeWAbs[5] + 0.078125 ResizeWAbs[6]
   + 0.078125 ResizeWAbs[7] + 0.0367647058823529 ResizeWAbs[8]
   + 0.0367647058823529 ResizeWAbs[9] + 0.0178571428571429 ResizeWAbs[10]
   + 0.0735294117647059 ResizeHAbs[0] + 0.0735294117647059 ResizeHAbs[1]
   + 0.0595238095238095 ResizeHAbs[2] + 0.0735294117647059 ResizeHAbs[3]
   + 0.0304878048780488 ResizeHAbs[4] + 0.0735294117647059 ResizeHAbs[5]
   + 0.0595238095238095 ResizeHAbs[6] + 0.0735294117647059 ResizeHAbs[7]
   + 0.0595238095238095 ResizeHAbs[8] + 0.0595238095238095 ResizeHAbs[9]
   + 0.0347222222222222 ResizeHAbs[10] + 0.0078125 MoveXAbs[0]
   + 0.0078125 MoveXAbs[1] + 0.0078125 MoveXAbs[2] + 0.0078125 MoveXAbs[3]
   + 0.00178571428571429 MoveXAbs[4] + 0.0078125 MoveXAbs[5]
   + 0.0078125 MoveXAbs[6] + 0.0078125 MoveXAbs[7]
   + 0.00367647058823529 MoveXAbs[8] + 0.00367647058823529 MoveXAbs[9]
   + 0.00178571428571429 MoveXAbs[10] + 0.00735294117647059 MoveYAbs[0]
   + 0.00735294117647059 MoveYAbs[1] + 0.00595238095238095 MoveYAbs[2]
   + 0.00735294117647059 MoveYAbs[3] + 0.00304878048780488 MoveYAbs[4]
   + 0.00735294117647059 MoveYAbs[5] + 0.00595238095238095 MoveYAbs[6]
   + 0.00735294117647059 MoveYAbs[7] + 0.00595238095238095 MoveYAbs[8]
   + 0.00595238095238095 MoveYAbs[9] + 0.00347222222222222 MoveYAbs[10]
   + 0 Above[0,0] + 0 Above[1,1] + 0 Above[2,2] + 0 Above[3,3]
   + 0 Above[4,4] + 0 Above[5,5] + 0 Above[6,6] + 0 Above[7,7]
   + 0 Above[8,8] + 0 Above[9,9] + 0 Above[10,10] + 0 OnLeft[0,0]
   + 0 OnLeft[1,1] + 0 OnLeft[2,2] + 0 OnLeft[3,3] + 0 OnLeft[4,4]
   + 0 OnLeft[5,5] + 0 OnLeft[6,6] + 0 OnLeft[7,7] + 0 OnLeft[8,8]
   + 0 OnLeft[9,9] + 0 OnLeft[10,10] + 0 VerticalOverlap[0,0]
   + 0 VerticalOverlap[1,1] + 0 VerticalOverlap[2,2]
   + 0 VerticalOverlap[3,3] + 0 VerticalOverlap[4,4]
   + 0 VerticalOverlap[5,5] + 0 VerticalOverlap[6,6]
   + 0 VerticalOverlap[7,7] + 0 VerticalOverlap[8,8]
   + 0 VerticalOverlap[9,9] + 0 VerticalOverlap[10,10] + 0 Overlap[0,0]
   + 0 Overlap[1,1] + 0 Overlap[2,2] + 0 Overlap[3,3] + 0 Overlap[4,4]
   + 0 Overlap[5,5] + 0 Overlap[6,6] + 0 Overlap[7,7] + 0 Overlap[8,8]
   + 0 Overlap[9,9] + 0 Overlap[10,10] + 0 X0LessThan[0,0]
   + 0 X0LessThan[1,1] + 0 X0LessThan[2,2] + 0 X0LessThan[3,3]
   + 0 X0LessThan[4,4] + 0 X0LessThan[5,5] + 0 X0LessThan[6,6]
   + 0 X0LessThan[7,7] + 0 X0LessThan[8,8] + 0 X0LessThan[9,9]
   + 0 X0LessThan[10,10] + 0 Y0LessThan[0,0] + 0 Y0LessThan[0,1]
   + 0 Y0LessThan[0,2] + 0 Y0LessThan[0,3] + 0 Y0LessThan[0,4]
   + 0 Y0LessThan[0,5] + 0 Y0LessThan[0,6] + 0 Y0LessThan[0,7]
   + 0 Y0LessThan[0,8] + 0 Y0LessThan[0,9] + 0 Y0LessThan[0,10]
   + 0 Y0LessThan[1,0] + 0 Y0LessThan[1,1] + 0 Y0LessThan[1,2]
   + 0 Y0LessThan[1,3] + 0 Y0LessThan[1,4] + 0 Y0LessThan[1,5]
   + 0 Y0LessThan[1,6] + 0 Y0LessThan[1,7] + 0 Y0LessThan[1,8]
   + 0 Y0LessThan[1,9] + 0 Y0LessThan[1,10] + 0 Y0LessThan[2,0]
   + 0 Y0LessThan[2,1] + 0 Y0LessThan[2,2] + 0 Y0LessThan[2,3]
   + 0 Y0LessThan[2,4] + 0 Y0LessThan[2,5] + 0 Y0LessThan[2,6]
   + 0 Y0LessThan[2,7] + 0 Y0LessThan[2,8] + 0 Y0LessThan[2,9]
   + 0 Y0LessThan[2,10] + 0 Y0LessThan[3,0] + 0 Y0LessThan[3,1]
   + 0 Y0LessThan[3,2] + 0 Y0LessThan[3,3] + 0 Y0LessThan[3,4]
   + 0 Y0LessThan[3,5] + 0 Y0LessThan[3,6] + 0 Y0LessThan[3,7]
   + 0 Y0LessThan[3,8] + 0 Y0LessThan[3,9] + 0 Y0LessThan[3,10]
   + 0 Y0LessThan[4,0] + 0 Y0LessThan[4,1] + 0 Y0LessThan[4,2]
   + 0 Y0LessThan[4,3] + 0 Y0LessThan[4,4] + 0 Y0LessThan[4,5]
   + 0 Y0LessThan[4,6] + 0 Y0LessThan[4,7] + 0 Y0LessThan[4,8]
   + 0 Y0LessThan[4,9] + 0 Y0LessThan[4,10] + 0 Y0LessThan[5,0]
   + 0 Y0LessThan[5,1] + 0 Y0LessThan[5,2] + 0 Y0LessThan[5,3]
   + 0 Y0LessThan[5,4] + 0 Y0LessThan[5,5] + 0 Y0LessThan[5,6]
   + 0 Y0LessThan[5,7] + 0 Y0LessThan[5,8] + 0 Y0LessThan[5,9]
   + 0 Y0LessThan[5,10] + 0 Y0LessThan[6,0] + 0 Y0LessThan[6,1]
   + 0 Y0LessThan[6,2] + 0 Y0LessThan[6,3] + 0 Y0LessThan[6,4]
   + 0 Y0LessThan[6,5] + 0 Y0LessThan[6,6] + 0 Y0LessThan[6,7]
   + 0 Y0LessThan[6,8] + 0 Y0LessThan[6,9] + 0 Y0LessThan[6,10]
   + 0 Y0LessThan[7,0] + 0 Y0LessThan[7,1] + 0 Y0LessThan[7,2]
   + 0 Y0LessThan[7,3] + 0 Y0LessThan[7,4] + 0 Y0LessThan[7,5]
   + 0 Y0LessThan[7,6] + 0 Y0LessThan[7,7] + 0 Y0LessThan[7,8]
   + 0 Y0LessThan[7,9] + 0 Y0LessThan[7,10] + 0 Y0LessThan[8,0]
   + 0 Y0LessThan[8,1] + 0 Y0LessThan[8,2] + 0 Y0LessThan[8,3]
   + 0 Y0LessThan[8,4] + 0 Y0LessThan[8,5] + 0 Y0LessThan[8,6]
   + 0 Y0LessThan[8,7] + 0 Y0LessThan[8,8] + 0 Y0LessThan[8,9]
   + 0 Y0LessThan[8,10] + 0 Y0LessThan[9,0] + 0 Y0LessThan[9,1]
   + 0 Y0LessThan[9,2] + 0 Y0LessThan[9,3] + 0 Y0LessThan[9,4]
   + 0 Y0LessThan[9,5] + 0 Y0LessThan[9,6] + 0 Y0LessThan[9,7]
   + 0 Y0LessThan[9,8] + 0 Y0LessThan[9,9] + 0 Y0LessThan[9,10]
   + 0 Y0LessThan[10,0] + 0 Y0LessThan[10,1] + 0 Y0LessThan[10,2]
   + 0 Y0LessThan[10,3] + 0 Y0LessThan[10,4] + 0 Y0LessThan[10,5]
   + 0 Y0LessThan[10,6] + 0 Y0LessThan[10,7] + 0 Y0LessThan[10,8]
   + 0 Y0LessThan[10,9] + 0 Y0LessThan[10,10] + 0 X1LessThan[0,0]
   + 0 X1LessThan[1,1] + 0 X1LessThan[2,2] + 0 X1LessThan[3,3]
   + 0 X1LessThan[4,4] + 0 X1LessThan[5,5] + 0 X1LessThan[6,6]
   + 0 X1LessThan[7,7] + 0 X1LessThan[8,8] + 0 X1LessThan[9,9]
   + 0 X1LessThan[10,10] + 0 Y1LessThan[0,0] + 0 Y1LessThan[0,1]
   + 0 Y1LessThan[0,2] + 0 Y1LessThan[0,3] + 0 Y1LessThan[0,4]
   + 0 Y1LessThan[0,5] + 0 Y1LessThan[0,6] + 0 Y1LessThan[0,7]
   + 0 Y1LessThan[0,8] + 0 Y1LessThan[0,9] + 0 Y1LessThan[0,10]
   + 0 Y1LessThan[1,0] + 0 Y1LessThan[1,1] + 0 Y1LessThan[1,2]
   + 0 Y1LessThan[1,3] + 0 Y1LessThan[1,4] + 0 Y1LessThan[1,5]
   + 0 Y1LessThan[1,6] + 0 Y1LessThan[1,7] + 0 Y1LessThan[1,8]
   + 0 Y1LessThan[1,9] + 0 Y1LessThan[1,10] + 0 Y1LessThan[2,0]
   + 0 Y1LessThan[2,1] + 0 Y1LessThan[2,2] + 0 Y1LessThan[2,3]
   + 0 Y1LessThan[2,4] + 0 Y1LessThan[2,5] + 0 Y1LessThan[2,6]
   + 0 Y1LessThan[2,7] + 0 Y1LessThan[2,8] + 0 Y1LessThan[2,9]
   + 0 Y1LessThan[2,10] + 0 Y1LessThan[3,0] + 0 Y1LessThan[3,1]
   + 0 Y1LessThan[3,2] + 0 Y1LessThan[3,3] + 0 Y1LessThan[3,4]
   + 0 Y1LessThan[3,5] + 0 Y1LessThan[3,6] + 0 Y1LessThan[3,7]
   + 0 Y1LessThan[3,8] + 0 Y1LessThan[3,9] + 0 Y1LessThan[3,10]
   + 0 Y1LessThan[4,0] + 0 Y1LessThan[4,1] + 0 Y1LessThan[4,2]
   + 0 Y1LessThan[4,3] + 0 Y1LessThan[4,4] + 0 Y1LessThan[4,5]
   + 0 Y1LessThan[4,6] + 0 Y1LessThan[4,7] + 0 Y1LessThan[4,8]
   + 0 Y1LessThan[4,9] + 0 Y1LessThan[4,10] + 0 Y1LessThan[5,0]
   + 0 Y1LessThan[5,1] + 0 Y1LessThan[5,2] + 0 Y1LessThan[5,3]
   + 0 Y1LessThan[5,4] + 0 Y1LessThan[5,5] + 0 Y1LessThan[5,6]
   + 0 Y1LessThan[5,7] + 0 Y1LessThan[5,8] + 0 Y1LessThan[5,9]
   + 0 Y1LessThan[5,10] + 0 Y1LessThan[6,0] + 0 Y1LessThan[6,1]
   + 0 Y1LessThan[6,2] + 0 Y1LessThan[6,3] + 0 Y1LessThan[6,4]
   + 0 Y1LessThan[6,5] + 0 Y1LessThan[6,6] + 0 Y1LessThan[6,7]
   + 0 Y1LessThan[6,8] + 0 Y1LessThan[6,9] + 0 Y1LessThan[6,10]
   + 0 Y1LessThan[7,0] + 0 Y1LessThan[7,1] + 0 Y1LessThan[7,2]
   + 0 Y1LessThan[7,3] + 0 Y1LessThan[7,4] + 0 Y1LessThan[7,5]
   + 0 Y1LessThan[7,6] + 0 Y1LessThan[7,7] + 0 Y1LessThan[7,8]
   + 0 Y1LessThan[7,9] + 0 Y1LessThan[7,10] + 0 Y1LessThan[8,0]
   + 0 Y1LessThan[8,1] + 0 Y1LessThan[8,2] + 0 Y1LessThan[8,3]
   + 0 Y1LessThan[8,4] + 0 Y1LessThan[8,5] + 0 Y1LessThan[8,6]
   + 0 Y1LessThan[8,7] + 0 Y1LessThan[8,8] + 0 Y1LessThan[8,9]
   + 0 Y1LessThan[8,10] + 0 Y1LessThan[9,0] + 0 Y1LessThan[9,1]
   + 0 Y1LessThan[9,2] + 0 Y1LessThan[9,3] + 0 Y1LessThan[9,4]
   + 0 Y1LessThan[9,5] + 0 Y1LessThan[9,6] + 0 Y1LessThan[9,7]
   + 0 Y1LessThan[9,8] + 0 Y1LessThan[9,9] + 0 Y1LessThan[9,10]
   + 0 Y1LessThan[10,0] + 0 Y1LessThan[10,1] + 0 Y1LessThan[10,2]
   + 0 Y1LessThan[10,3] + 0 Y1LessThan[10,4] + 0 Y1LessThan[10,5]
   + 0 Y1LessThan[10,6] + 0 Y1LessThan[10,7] + 0 Y1LessThan[10,8]
   + 0 Y1LessThan[10,9] + 0 Y1LessThan[10,10] + 0 X0Group[0] + 0 X0Group[1]
   + 0 X0Group[2] + 0 X0Group[3] + 0 X0Group[4] + 0 X0Group[5]
   + 0 X0Group[6] + 0 X0Group[7] + 0 X0Group[8] + 0 X0Group[9]
   + 0 X0Group[10] + 0 Y0Group[0] + 0 Y0Group[1] + 0 Y0Group[2]
   + 0 Y0Group[3] + 0 Y0Group[4] + 0 Y0Group[5] + 0 Y0Group[6]
   + 0 Y0Group[7] + 0 Y0Group[8] + 0 Y0Group[9] + 0 Y0Group[10]
   + 0 X1Group[0] + 0 X1Group[1] + 0 X1Group[2] + 0 X1Group[3]
   + 0 X1Group[4] + 0 X1Group[5] + 0 X1Group[6] + 0 X1Group[7]
   + 0 X1Group[8] + 0 X1Group[9] + 0 X1Group[10] + 0 Y1Group[0]
   + 0 Y1Group[1] + 0 Y1Group[2] + 0 Y1Group[3] + 0 Y1Group[4]
   + 0 Y1Group[5] + 0 Y1Group[6] + 0 Y1Group[7] + 0 Y1Group[8]
   + 0 Y1Group[9] + 0 Y1Group[10] + MarginDiffLooseAbs + 0 C2252
   + 0 OtherElementStartsInNextColumn[0,0]
   + 0 OtherElementStartsInNextColumn[1,1]
   + 0 OtherElementStartsInNextColumn[2,2]
   + 0 OtherElementStartsInNextColumn[3,3]
   + 0 OtherElementStartsInNextColumn[4,4]
   + 0 OtherElementStartsInNextColumn[5,5]
   + 0 OtherElementStartsInNextColumn[6,6]
   + 0 OtherElementStartsInNextColumn[7,7]
   + 0 OtherElementStartsInNextColumn[8,8]
   + 0 OtherElementStartsInNextColumn[9,9]
   + 0 OtherElementStartsInNextColumn[10,10]
   + 0 OtherElementEndsInNextColumn[0,0]
   + 0 OtherElementEndsInNextColumn[1,1]
   + 0 OtherElementEndsInNextColumn[2,2]
   + 0 OtherElementEndsInNextColumn[3,3]
   + 0 OtherElementEndsInNextColumn[4,4]
   + 0 OtherElementEndsInNextColumn[5,5]
   + 0 OtherElementEndsInNextColumn[6,6]
   + 0 OtherElementEndsInNextColumn[7,7]
   + 0 OtherElementEndsInNextColumn[8,8]
   + 0 OtherElementEndsInNextColumn[9,9]
   + 0 OtherElementEndsInNextColumn[10,10] + ColumnCount + 0 FirstColumn
Subject To
 X0X1Sanity[0]: X0[0] - X1[0] <= -1
 X0X1Sanity[1]: X0[1] - X1[1] <= -1
 X0X1Sanity[2]: X0[2] - X1[2] <= -1
 X0X1Sanity[3]: X0[3] - X1[3] <= -1
 X0X1Sanity[4]: X0[4] - X1[4] <= -1
 X0X1Sanity[5]: X0[5] - X1[5] <= -1
 X0X1Sanity[6]: X0[6] - X1[6] <= -1
 X0X1Sanity[7]: X0[7] - X1[7] <= -1
 X0X1Sanity[8]: X0[8] - X1[8] <= -1
 X0X1Sanity[9]: X0[9] - X1[9] <= -1
 X0X1Sanity[10]: X0[10] - X1[10] <= -1
 X0X1Sanity[0]: Y0[0] - Y1[0] <= -1
 X0X1Sanity[1]: Y0[1] - Y1[1] <= -1
 X0X1Sanity[2]: Y0[2] - Y1[2] <= -1
 X0X1Sanity[3]: Y0[3] - Y1[3] <= -1
 X0X1Sanity[4]: Y0[4] - Y1[4] <= -1
 X0X1Sanity[5]: Y0[5] - Y1[5] <= -1
 X0X1Sanity[6]: Y0[6] - Y1[6] <= -1
 X0X1Sanity[7]: Y0[7] - Y1[7] <= -1
 X0X1Sanity[8]: Y0[8] - Y1[8] <= -1
 X0X1Sanity[9]: Y0[9] - Y1[9] <= -1
 X0X1Sanity[10]: Y0[10] - Y1[10] <= -1
 LimitToCanvasW[0]: X1[0] <= 100
 LimitToCanvasW[1]: X1[1] <= 100
 LimitToCanvasW[2]: X1[2] <= 100
 LimitToCanvasW[3]: X1[3] <= 100
 LimitToCanvasW[4]: X1[4] <= 100
 LimitToCanvasW[5]: X1[5] <= 100
 LimitToCanvasW[6]: X1[6] <= 100
 LimitToCanvasW[7]: X1[7] <= 100
 LimitToCanvasW[8]: X1[8] <= 100
 LimitToCanvasW[9]: X1[9] <= 100
 LimitToCanvasW[10]: X1[10] <= 100
 LimitToCanvasH[0]: Y1[0] <= 124
 LimitToCanvasH[1]: Y1[1] <= 124
 LimitToCanvasH[2]: Y1[2] <= 124
 LimitToCanvasH[3]: Y1[3] <= 124
 LimitToCanvasH[4]: Y1[4] <= 124
 LimitToCanvasH[5]: Y1[5] <= 124
 LimitToCanvasH[6]: Y1[6] <= 124
 LimitToCanvasH[7]: Y1[7] <= 124
 LimitToCanvasH[8]: Y1[8] <= 124
 LimitToCanvasH[9]: Y1[9] <= 124
 LimitToCanvasH[10]: Y1[10] <= 124
 X1-X0=W[0]: - X0[0] + X1[0] - W[0] = 0
 X1-X0=W[1]: - X0[1] + X1[1] - W[1] = 0
 X1-X0=W[2]: - X0[2] + X1[2] - W[2] = 0
 X1-X0=W[3]: - X0[3] + X1[3] - W[3] = 0
 X1-X0=W[4]: - X0[4] + X1[4] - W[4] = 0
 X1-X0=W[5]: - X0[5] + X1[5] - W[5] = 0
 X1-X0=W[6]: - X0[6] + X1[6] - W[6] = 0
 X1-X0=W[7]: - X0[7] + X1[7] - W[7] = 0
 X1-X0=W[8]: - X0[8] + X1[8] - W[8] = 0
 X1-X0=W[9]: - X0[9] + X1[9] - W[9] = 0
 X1-X0=W[10]: - X0[10] + X1[10] - W[10] = 0
 Y1-Y0=H[0]: - Y0[0] + Y1[0] - H[0] = 0
 Y1-Y0=H[1]: - Y0[1] + Y1[1] - H[1] = 0
 Y1-Y0=H[2]: - Y0[2] + Y1[2] - H[2] = 0
 Y1-Y0=H[3]: - Y0[3] + Y1[3] - H[3] = 0
 Y1-Y0=H[4]: - Y0[4] + Y1[4] - H[4] = 0
 Y1-Y0=H[5]: - Y0[5] + Y1[5] - H[5] = 0
 Y1-Y0=H[6]: - Y0[6] + Y1[6] - H[6] = 0
 Y1-Y0=H[7]: - Y0[7] + Y1[7] - H[7] = 0
 Y1-Y0=H[8]: - Y0[8] + Y1[8] - H[8] = 0
 Y1-Y0=H[9]: - Y0[9] + Y1[9] - H[9] = 0
 Y1-Y0=H[10]: - Y0[10] + Y1[10] - H[10] = 0
 LinkResizeW[0,<optimizer.classes.Element_object_at_0x105273128>]: - W[0]
   + ResizeW[0] = -16
 LinkResizeW[1,<optimizer.classes.Element_object_at_0x1052730f0>]: - W[1]
   + ResizeW[1] = -16
 LinkResizeW[2,<optimizer.classes.Element_object_at_0x1052733c8>]: - W[2]
   + ResizeW[2] = -16
 LinkResizeW[3,<optimizer.classes.Element_object_at_0x1052730b8>]: - W[3]
   + ResizeW[3] = -16
 LinkResizeW[4,<optimizer.classes.Element_object_at_0x105273438>]: - W[4]
   + ResizeW[4] = -70
 LinkResizeW[5,<optimizer.classes.Element_object_at_0x105273320>]: - W[5]
   + ResizeW[5] = -16
 LinkResizeW[6,<optimizer.classes.Element_object_at_0x1052734a8>]: - W[6]
   + ResizeW[6] = -16
 LinkResizeW[7,<optimizer.classes.Element_object_at_0x105273390>]: - W[7]
   + ResizeW[7] = -16
 LinkResizeW[8,<optimizer.classes.Element_object_at_0x105273518>]: - W[8]
   + ResizeW[8] = -34
 LinkResizeW[9,<optimizer.classes.Element_object_at_0x105273400>]: - W[9]
   + ResizeW[9] = -34
 LinkResizeW[10,<optimizer.classes.Element_object_at_0x105273588>]: - W[10]
   + ResizeW[10] = -70
 LinkResizeH[0,<optimizer.classes.Element_object_at_0x105273128>]: - H[0]
   + ResizeH[0] = -17
 LinkResizeH[1,<optimizer.classes.Element_object_at_0x1052730f0>]: - H[1]
   + ResizeH[1] = -17
 LinkResizeH[2,<optimizer.classes.Element_object_at_0x1052733c8>]: - H[2]
   + ResizeH[2] = -21
 LinkResizeH[3,<optimizer.classes.Element_object_at_0x1052730b8>]: - H[3]
   + ResizeH[3] = -17
 LinkResizeH[4,<optimizer.classes.Element_object_at_0x105273438>]: - H[4]
   + ResizeH[4] = -41
 LinkResizeH[5,<optimizer.classes.Element_object_at_0x105273320>]: - H[5]
   + ResizeH[5] = -17
 LinkResizeH[6,<optimizer.classes.Element_object_at_0x1052734a8>]: - H[6]
   + ResizeH[6] = -21
 LinkResizeH[7,<optimizer.classes.Element_object_at_0x105273390>]: - H[7]
   + ResizeH[7] = -17
 LinkResizeH[8,<optimizer.classes.Element_object_at_0x105273518>]: - H[8]
   + ResizeH[8] = -21
 LinkResizeH[9,<optimizer.classes.Element_object_at_0x105273400>]: - H[9]
   + ResizeH[9] = -21
 LinkResizeH[10,<optimizer.classes.Element_object_at_0x105273588>]: - H[10]
   + ResizeH[10] = -36
 LinkMoveX[0,<optimizer.classes.Element_object_at_0x105273128>]: - X0[0]
   + MoveX[0] = -33
 LinkMoveX[1,<optimizer.classes.Element_object_at_0x1052730f0>]: - X0[1]
   + MoveX[1] = -33
 LinkMoveX[2,<optimizer.classes.Element_object_at_0x1052733c8>]: - X0[2]
   + MoveX[2] = -51
 LinkMoveX[3,<optimizer.classes.Element_object_at_0x1052730b8>]: - X0[3]
   + MoveX[3] = -51
 LinkMoveX[4,<optimizer.classes.Element_object_at_0x105273438>]: - X0[4]
   + MoveX[4] = -15
 LinkMoveX[5,<optimizer.classes.Element_object_at_0x105273320>]: - X0[5]
   + MoveX[5] = -33
 LinkMoveX[6,<optimizer.classes.Element_object_at_0x1052734a8>]: - X0[6]
   + MoveX[6] = -51
 LinkMoveX[7,<optimizer.classes.Element_object_at_0x105273390>]: - X0[7]
   + MoveX[7] = -33
 LinkMoveX[8,<optimizer.classes.Element_object_at_0x105273518>]: - X0[8]
   + MoveX[8] = -15
 LinkMoveX[9,<optimizer.classes.Element_object_at_0x105273400>]: - X0[9]
   + MoveX[9] = -51
 LinkMoveX[10,<optimizer.classes.Element_object_at_0x105273588>]: - X0[10]
   + MoveX[10] = -15
 LinkMoveY[0,<optimizer.classes.Element_object_at_0x105273128>]: - Y0[0]
   + MoveY[0] = -13
 LinkMoveY[1,<optimizer.classes.Element_object_at_0x1052730f0>]: - Y0[1]
   + MoveY[1] = -30
 LinkMoveY[2,<optimizer.classes.Element_object_at_0x1052733c8>]: - Y0[2]
   + MoveY[2] = -30
 LinkMoveY[3,<optimizer.classes.Element_object_at_0x1052730b8>]: - Y0[3]
   + MoveY[3] = -12
 LinkMoveY[4,<optimizer.classes.Element_object_at_0x105273438>]: - Y0[4]
   + MoveY[4] = -11
 LinkMoveY[5,<optimizer.classes.Element_object_at_0x105273320>]: - Y0[5]
   + MoveY[5] = -70
 LinkMoveY[6,<optimizer.classes.Element_object_at_0x1052734a8>]: - Y0[6]
   + MoveY[6] = -59
 LinkMoveY[7,<optimizer.classes.Element_object_at_0x105273390>]: - Y0[7]
   + MoveY[7] = -53
 LinkMoveY[8,<optimizer.classes.Element_object_at_0x105273518>]: - Y0[8]
   + MoveY[8] = -94
 LinkMoveY[9,<optimizer.classes.Element_object_at_0x105273400>]: - Y0[9]
   + MoveY[9] = -88
 LinkMoveY[10,<optimizer.classes.Element_object_at_0x105273588>]: - Y0[10]
   + MoveY[10] = -52
 AboveSanity[0,1]: Above[0,1] + Above[1,0] <= 1
 AboveSanity[0,2]: Above[0,2] + Above[2,0] <= 1
 AboveSanity[0,3]: Above[0,3] + Above[3,0] <= 1
 AboveSanity[0,4]: Above[0,4] + Above[4,0] <= 1
 AboveSanity[0,5]: Above[0,5] + Above[5,0] <= 1
 AboveSanity[0,6]: Above[0,6] + Above[6,0] <= 1
 AboveSanity[0,7]: Above[0,7] + Above[7,0] <= 1
 AboveSanity[0,8]: Above[0,8] + Above[8,0] <= 1
 AboveSanity[0,9]: Above[0,9] + Above[9,0] <= 1
 AboveSanity[0,10]: Above[0,10] + Above[10,0] <= 1
 AboveSanity[1,0]: Above[0,1] + Above[1,0] <= 1
 AboveSanity[1,2]: Above[1,2] + Above[2,1] <= 1
 AboveSanity[1,3]: Above[1,3] + Above[3,1] <= 1
 AboveSanity[1,4]: Above[1,4] + Above[4,1] <= 1
 AboveSanity[1,5]: Above[1,5] + Above[5,1] <= 1
 AboveSanity[1,6]: Above[1,6] + Above[6,1] <= 1
 AboveSanity[1,7]: Above[1,7] + Above[7,1] <= 1
 AboveSanity[1,8]: Above[1,8] + Above[8,1] <= 1
 AboveSanity[1,9]: Above[1,9] + Above[9,1] <= 1
 AboveSanity[1,10]: Above[1,10] + Above[10,1] <= 1
 AboveSanity[2,0]: Above[0,2] + Above[2,0] <= 1
 AboveSanity[2,1]: Above[1,2] + Above[2,1] <= 1
 AboveSanity[2,3]: Above[2,3] + Above[3,2] <= 1
 AboveSanity[2,4]: Above[2,4] + Above[4,2] <= 1
 AboveSanity[2,5]: Above[2,5] + Above[5,2] <= 1
 AboveSanity[2,6]: Above[2,6] + Above[6,2] <= 1
 AboveSanity[2,7]: Above[2,7] + Above[7,2] <= 1
 AboveSanity[2,8]: Above[2,8] + Above[8,2] <= 1
 AboveSanity[2,9]: Above[2,9] + Above[9,2] <= 1
 AboveSanity[2,10]: Above[2,10] + Above[10,2] <= 1
 AboveSanity[3,0]: Above[0,3] + Above[3,0] <= 1
 AboveSanity[3,1]: Above[1,3] + Above[3,1] <= 1
 AboveSanity[3,2]: Above[2,3] + Above[3,2] <= 1
 AboveSanity[3,4]: Above[3,4] + Above[4,3] <= 1
 AboveSanity[3,5]: Above[3,5] + Above[5,3] <= 1
 AboveSanity[3,6]: Above[3,6] + Above[6,3] <= 1
 AboveSanity[3,7]: Above[3,7] + Above[7,3] <= 1
 AboveSanity[3,8]: Above[3,8] + Above[8,3] <= 1
 AboveSanity[3,9]: Above[3,9] + Above[9,3] <= 1
 AboveSanity[3,10]: Above[3,10] + Above[10,3] <= 1
 AboveSanity[4,0]: Above[0,4] + Above[4,0] <= 1
 AboveSanity[4,1]: Above[1,4] + Above[4,1] <= 1
 AboveSanity[4,2]: Above[2,4] + Above[4,2] <= 1
 AboveSanity[4,3]: Above[3,4] + Above[4,3] <= 1
 AboveSanity[4,5]: Above[4,5] + Above[5,4] <= 1
 AboveSanity[4,6]: Above[4,6] + Above[6,4] <= 1
 AboveSanity[4,7]: Above[4,7] + Above[7,4] <= 1
 AboveSanity[4,8]: Above[4,8] + Above[8,4] <= 1
 AboveSanity[4,9]: Above[4,9] + Above[9,4] <= 1
 AboveSanity[4,10]: Above[4,10] + Above[10,4] <= 1
 AboveSanity[5,0]: Above[0,5] + Above[5,0] <= 1
 AboveSanity[5,1]: Above[1,5] + Above[5,1] <= 1
 AboveSanity[5,2]: Above[2,5] + Above[5,2] <= 1
 AboveSanity[5,3]: Above[3,5] + Above[5,3] <= 1
 AboveSanity[5,4]: Above[4,5] + Above[5,4] <= 1
 AboveSanity[5,6]: Above[5,6] + Above[6,5] <= 1
 AboveSanity[5,7]: Above[5,7] + Above[7,5] <= 1
 AboveSanity[5,8]: Above[5,8] + Above[8,5] <= 1
 AboveSanity[5,9]: Above[5,9] + Above[9,5] <= 1
 AboveSanity[5,10]: Above[5,10] + Above[10,5] <= 1
 AboveSanity[6,0]: Above[0,6] + Above[6,0] <= 1
 AboveSanity[6,1]: Above[1,6] + Above[6,1] <= 1
 AboveSanity[6,2]: Above[2,6] + Above[6,2] <= 1
 AboveSanity[6,3]: Above[3,6] + Above[6,3] <= 1
 AboveSanity[6,4]: Above[4,6] + Above[6,4] <= 1
 AboveSanity[6,5]: Above[5,6] + Above[6,5] <= 1
 AboveSanity[6,7]: Above[6,7] + Above[7,6] <= 1
 AboveSanity[6,8]: Above[6,8] + Above[8,6] <= 1
 AboveSanity[6,9]: Above[6,9] + Above[9,6] <= 1
 AboveSanity[6,10]: Above[6,10] + Above[10,6] <= 1
 AboveSanity[7,0]: Above[0,7] + Above[7,0] <= 1
 AboveSanity[7,1]: Above[1,7] + Above[7,1] <= 1
 AboveSanity[7,2]: Above[2,7] + Above[7,2] <= 1
 AboveSanity[7,3]: Above[3,7] + Above[7,3] <= 1
 AboveSanity[7,4]: Above[4,7] + Above[7,4] <= 1
 AboveSanity[7,5]: Above[5,7] + Above[7,5] <= 1
 AboveSanity[7,6]: Above[6,7] + Above[7,6] <= 1
 AboveSanity[7,8]: Above[7,8] + Above[8,7] <= 1
 AboveSanity[7,9]: Above[7,9] + Above[9,7] <= 1
 AboveSanity[7,10]: Above[7,10] + Above[10,7] <= 1
 AboveSanity[8,0]: Above[0,8] + Above[8,0] <= 1
 AboveSanity[8,1]: Above[1,8] + Above[8,1] <= 1
 AboveSanity[8,2]: Above[2,8] + Above[8,2] <= 1
 AboveSanity[8,3]: Above[3,8] + Above[8,3] <= 1
 AboveSanity[8,4]: Above[4,8] + Above[8,4] <= 1
 AboveSanity[8,5]: Above[5,8] + Above[8,5] <= 1
 AboveSanity[8,6]: Above[6,8] + Above[8,6] <= 1
 AboveSanity[8,7]: Above[7,8] + Above[8,7] <= 1
 AboveSanity[8,9]: Above[8,9] + Above[9,8] <= 1
 AboveSanity[8,10]: Above[8,10] + Above[10,8] <= 1
 AboveSanity[9,0]: Above[0,9] + Above[9,0] <= 1
 AboveSanity[9,1]: Above[1,9] + Above[9,1] <= 1
 AboveSanity[9,2]: Above[2,9] + Above[9,2] <= 1
 AboveSanity[9,3]: Above[3,9] + Above[9,3] <= 1
 AboveSanity[9,4]: Above[4,9] + Above[9,4] <= 1
 AboveSanity[9,5]: Above[5,9] + Above[9,5] <= 1
 AboveSanity[9,6]: Above[6,9] + Above[9,6] <= 1
 AboveSanity[9,7]: Above[7,9] + Above[9,7] <= 1
 AboveSanity[9,8]: Above[8,9] + Above[9,8] <= 1
 AboveSanity[9,10]: Above[9,10] + Above[10,9] <= 1
 AboveSanity[10,0]: Above[0,10] + Above[10,0] <= 1
 AboveSanity[10,1]: Above[1,10] + Above[10,1] <= 1
 AboveSanity[10,2]: Above[2,10] + Above[10,2] <= 1
 AboveSanity[10,3]: Above[3,10] + Above[10,3] <= 1
 AboveSanity[10,4]: Above[4,10] + Above[10,4] <= 1
 AboveSanity[10,5]: Above[5,10] + Above[10,5] <= 1
 AboveSanity[10,6]: Above[6,10] + Above[10,6] <= 1
 AboveSanity[10,7]: Above[7,10] + Above[10,7] <= 1
 AboveSanity[10,8]: Above[8,10] + Above[10,8] <= 1
 AboveSanity[10,9]: Above[9,10] + Above[10,9] <= 1
 OnLeftSanity[0,1]: OnLeft[0,1] + OnLeft[1,0] <= 1
 OnLeftSanity[0,2]: OnLeft[0,2] + OnLeft[2,0] <= 1
 OnLeftSanity[0,3]: OnLeft[0,3] + OnLeft[3,0] <= 1
 OnLeftSanity[0,4]: OnLeft[0,4] + OnLeft[4,0] <= 1
 OnLeftSanity[0,5]: OnLeft[0,5] + OnLeft[5,0] <= 1
 OnLeftSanity[0,6]: OnLeft[0,6] + OnLeft[6,0] <= 1
 OnLeftSanity[0,7]: OnLeft[0,7] + OnLeft[7,0] <= 1
 OnLeftSanity[0,8]: OnLeft[0,8] + OnLeft[8,0] <= 1
 OnLeftSanity[0,9]: OnLeft[0,9] + OnLeft[9,0] <= 1
 OnLeftSanity[0,10]: OnLeft[0,10] + OnLeft[10,0] <= 1
 OnLeftSanity[1,0]: OnLeft[0,1] + OnLeft[1,0] <= 1
 OnLeftSanity[1,2]: OnLeft[1,2] + OnLeft[2,1] <= 1
 OnLeftSanity[1,3]: OnLeft[1,3] + OnLeft[3,1] <= 1
 OnLeftSanity[1,4]: OnLeft[1,4] + OnLeft[4,1] <= 1
 OnLeftSanity[1,5]: OnLeft[1,5] + OnLeft[5,1] <= 1
 OnLeftSanity[1,6]: OnLeft[1,6] + OnLeft[6,1] <= 1
 OnLeftSanity[1,7]: OnLeft[1,7] + OnLeft[7,1] <= 1
 OnLeftSanity[1,8]: OnLeft[1,8] + OnLeft[8,1] <= 1
 OnLeftSanity[1,9]: OnLeft[1,9] + OnLeft[9,1] <= 1
 OnLeftSanity[1,10]: OnLeft[1,10] + OnLeft[10,1] <= 1
 OnLeftSanity[2,0]: OnLeft[0,2] + OnLeft[2,0] <= 1
 OnLeftSanity[2,1]: OnLeft[1,2] + OnLeft[2,1] <= 1
 OnLeftSanity[2,3]: OnLeft[2,3] + OnLeft[3,2] <= 1
 OnLeftSanity[2,4]: OnLeft[2,4] + OnLeft[4,2] <= 1
 OnLeftSanity[2,5]: OnLeft[2,5] + OnLeft[5,2] <= 1
 OnLeftSanity[2,6]: OnLeft[2,6] + OnLeft[6,2] <= 1
 OnLeftSanity[2,7]: OnLeft[2,7] + OnLeft[7,2] <= 1
 OnLeftSanity[2,8]: OnLeft[2,8] + OnLeft[8,2] <= 1
 OnLeftSanity[2,9]: OnLeft[2,9] + OnLeft[9,2] <= 1
 OnLeftSanity[2,10]: OnLeft[2,10] + OnLeft[10,2] <= 1
 OnLeftSanity[3,0]: OnLeft[0,3] + OnLeft[3,0] <= 1
 OnLeftSanity[3,1]: OnLeft[1,3] + OnLeft[3,1] <= 1
 OnLeftSanity[3,2]: OnLeft[2,3] + OnLeft[3,2] <= 1
 OnLeftSanity[3,4]: OnLeft[3,4] + OnLeft[4,3] <= 1
 OnLeftSanity[3,5]: OnLeft[3,5] + OnLeft[5,3] <= 1
 OnLeftSanity[3,6]: OnLeft[3,6] + OnLeft[6,3] <= 1
 OnLeftSanity[3,7]: OnLeft[3,7] + OnLeft[7,3] <= 1
 OnLeftSanity[3,8]: OnLeft[3,8] + OnLeft[8,3] <= 1
 OnLeftSanity[3,9]: OnLeft[3,9] + OnLeft[9,3] <= 1
 OnLeftSanity[3,10]: OnLeft[3,10] + OnLeft[10,3] <= 1
 OnLeftSanity[4,0]: OnLeft[0,4] + OnLeft[4,0] <= 1
 OnLeftSanity[4,1]: OnLeft[1,4] + OnLeft[4,1] <= 1
 OnLeftSanity[4,2]: OnLeft[2,4] + OnLeft[4,2] <= 1
 OnLeftSanity[4,3]: OnLeft[3,4] + OnLeft[4,3] <= 1
 OnLeftSanity[4,5]: OnLeft[4,5] + OnLeft[5,4] <= 1
 OnLeftSanity[4,6]: OnLeft[4,6] + OnLeft[6,4] <= 1
 OnLeftSanity[4,7]: OnLeft[4,7] + OnLeft[7,4] <= 1
 OnLeftSanity[4,8]: OnLeft[4,8] + OnLeft[8,4] <= 1
 OnLeftSanity[4,9]: OnLeft[4,9] + OnLeft[9,4] <= 1
 OnLeftSanity[4,10]: OnLeft[4,10] + OnLeft[10,4] <= 1
 OnLeftSanity[5,0]: OnLeft[0,5] + OnLeft[5,0] <= 1
 OnLeftSanity[5,1]: OnLeft[1,5] + OnLeft[5,1] <= 1
 OnLeftSanity[5,2]: OnLeft[2,5] + OnLeft[5,2] <= 1
 OnLeftSanity[5,3]: OnLeft[3,5] + OnLeft[5,3] <= 1
 OnLeftSanity[5,4]: OnLeft[4,5] + OnLeft[5,4] <= 1
 OnLeftSanity[5,6]: OnLeft[5,6] + OnLeft[6,5] <= 1
 OnLeftSanity[5,7]: OnLeft[5,7] + OnLeft[7,5] <= 1
 OnLeftSanity[5,8]: OnLeft[5,8] + OnLeft[8,5] <= 1
 OnLeftSanity[5,9]: OnLeft[5,9] + OnLeft[9,5] <= 1
 OnLeftSanity[5,10]: OnLeft[5,10] + OnLeft[10,5] <= 1
 OnLeftSanity[6,0]: OnLeft[0,6] + OnLeft[6,0] <= 1
 OnLeftSanity[6,1]: OnLeft[1,6] + OnLeft[6,1] <= 1
 OnLeftSanity[6,2]: OnLeft[2,6] + OnLeft[6,2] <= 1
 OnLeftSanity[6,3]: OnLeft[3,6] + OnLeft[6,3] <= 1
 OnLeftSanity[6,4]: OnLeft[4,6] + OnLeft[6,4] <= 1
 OnLeftSanity[6,5]: OnLeft[5,6] + OnLeft[6,5] <= 1
 OnLeftSanity[6,7]: OnLeft[6,7] + OnLeft[7,6] <= 1
 OnLeftSanity[6,8]: OnLeft[6,8] + OnLeft[8,6] <= 1
 OnLeftSanity[6,9]: OnLeft[6,9] + OnLeft[9,6] <= 1
 OnLeftSanity[6,10]: OnLeft[6,10] + OnLeft[10,6] <= 1
 OnLeftSanity[7,0]: OnLeft[0,7] + OnLeft[7,0] <= 1
 OnLeftSanity[7,1]: OnLeft[1,7] + OnLeft[7,1] <= 1
 OnLeftSanity[7,2]: OnLeft[2,7] + OnLeft[7,2] <= 1
 OnLeftSanity[7,3]: OnLeft[3,7] + OnLeft[7,3] <= 1
 OnLeftSanity[7,4]: OnLeft[4,7] + OnLeft[7,4] <= 1
 OnLeftSanity[7,5]: OnLeft[5,7] + OnLeft[7,5] <= 1
 OnLeftSanity[7,6]: OnLeft[6,7] + OnLeft[7,6] <= 1
 OnLeftSanity[7,8]: OnLeft[7,8] + OnLeft[8,7] <= 1
 OnLeftSanity[7,9]: OnLeft[7,9] + OnLeft[9,7] <= 1
 OnLeftSanity[7,10]: OnLeft[7,10] + OnLeft[10,7] <= 1
 OnLeftSanity[8,0]: OnLeft[0,8] + OnLeft[8,0] <= 1
 OnLeftSanity[8,1]: OnLeft[1,8] + OnLeft[8,1] <= 1
 OnLeftSanity[8,2]: OnLeft[2,8] + OnLeft[8,2] <= 1
 OnLeftSanity[8,3]: OnLeft[3,8] + OnLeft[8,3] <= 1
 OnLeftSanity[8,4]: OnLeft[4,8] + OnLeft[8,4] <= 1
 OnLeftSanity[8,5]: OnLeft[5,8] + OnLeft[8,5] <= 1
 OnLeftSanity[8,6]: OnLeft[6,8] + OnLeft[8,6] <= 1
 OnLeftSanity[8,7]: OnLeft[7,8] + OnLeft[8,7] <= 1
 OnLeftSanity[8,9]: OnLeft[8,9] + OnLeft[9,8] <= 1
 OnLeftSanity[8,10]: OnLeft[8,10] + OnLeft[10,8] <= 1
 OnLeftSanity[9,0]: OnLeft[0,9] + OnLeft[9,0] <= 1
 OnLeftSanity[9,1]: OnLeft[1,9] + OnLeft[9,1] <= 1
 OnLeftSanity[9,2]: OnLeft[2,9] + OnLeft[9,2] <= 1
 OnLeftSanity[9,3]: OnLeft[3,9] + OnLeft[9,3] <= 1
 OnLeftSanity[9,4]: OnLeft[4,9] + OnLeft[9,4] <= 1
 OnLeftSanity[9,5]: OnLeft[5,9] + OnLeft[9,5] <= 1
 OnLeftSanity[9,6]: OnLeft[6,9] + OnLeft[9,6] <= 1
 OnLeftSanity[9,7]: OnLeft[7,9] + OnLeft[9,7] <= 1
 OnLeftSanity[9,8]: OnLeft[8,9] + OnLeft[9,8] <= 1
 OnLeftSanity[9,10]: OnLeft[9,10] + OnLeft[10,9] <= 1
 OnLeftSanity[10,0]: OnLeft[0,10] + OnLeft[10,0] <= 1
 OnLeftSanity[10,1]: OnLeft[1,10] + OnLeft[10,1] <= 1
 OnLeftSanity[10,2]: OnLeft[2,10] + OnLeft[10,2] <= 1
 OnLeftSanity[10,3]: OnLeft[3,10] + OnLeft[10,3] <= 1
 OnLeftSanity[10,4]: OnLeft[4,10] + OnLeft[10,4] <= 1
 OnLeftSanity[10,5]: OnLeft[5,10] + OnLeft[10,5] <= 1
 OnLeftSanity[10,6]: OnLeft[6,10] + OnLeft[10,6] <= 1
 OnLeftSanity[10,7]: OnLeft[7,10] + OnLeft[10,7] <= 1
 OnLeftSanity[10,8]: OnLeft[8,10] + OnLeft[10,8] <= 1
 OnLeftSanity[10,9]: OnLeft[9,10] + OnLeft[10,9] <= 1
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
 LinkHorizontalOverlap[0,8]: OnLeft[0,8] + OnLeft[8,0]
   + HorizontalOverlap[0,8] = 1
 LinkHorizontalOverlap[0,9]: OnLeft[0,9] + OnLeft[9,0]
   + HorizontalOverlap[0,9] = 1
 LinkHorizontalOverlap[0,10]: OnLeft[0,10] + OnLeft[10,0]
   + HorizontalOverlap[0,10] = 1
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
 LinkHorizontalOverlap[1,8]: OnLeft[1,8] + OnLeft[8,1]
   + HorizontalOverlap[1,8] = 1
 LinkHorizontalOverlap[1,9]: OnLeft[1,9] + OnLeft[9,1]
   + HorizontalOverlap[1,9] = 1
 LinkHorizontalOverlap[1,10]: OnLeft[1,10] + OnLeft[10,1]
   + HorizontalOverlap[1,10] = 1
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
 LinkHorizontalOverlap[2,8]: OnLeft[2,8] + OnLeft[8,2]
   + HorizontalOverlap[2,8] = 1
 LinkHorizontalOverlap[2,9]: OnLeft[2,9] + OnLeft[9,2]
   + HorizontalOverlap[2,9] = 1
 LinkHorizontalOverlap[2,10]: OnLeft[2,10] + OnLeft[10,2]
   + HorizontalOverlap[2,10] = 1
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
 LinkHorizontalOverlap[3,8]: OnLeft[3,8] + OnLeft[8,3]
   + HorizontalOverlap[3,8] = 1
 LinkHorizontalOverlap[3,9]: OnLeft[3,9] + OnLeft[9,3]
   + HorizontalOverlap[3,9] = 1
 LinkHorizontalOverlap[3,10]: OnLeft[3,10] + OnLeft[10,3]
   + HorizontalOverlap[3,10] = 1
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
 LinkHorizontalOverlap[4,8]: OnLeft[4,8] + OnLeft[8,4]
   + HorizontalOverlap[4,8] = 1
 LinkHorizontalOverlap[4,9]: OnLeft[4,9] + OnLeft[9,4]
   + HorizontalOverlap[4,9] = 1
 LinkHorizontalOverlap[4,10]: OnLeft[4,10] + OnLeft[10,4]
   + HorizontalOverlap[4,10] = 1
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
 LinkHorizontalOverlap[5,8]: OnLeft[5,8] + OnLeft[8,5]
   + HorizontalOverlap[5,8] = 1
 LinkHorizontalOverlap[5,9]: OnLeft[5,9] + OnLeft[9,5]
   + HorizontalOverlap[5,9] = 1
 LinkHorizontalOverlap[5,10]: OnLeft[5,10] + OnLeft[10,5]
   + HorizontalOverlap[5,10] = 1
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
 LinkHorizontalOverlap[6,8]: OnLeft[6,8] + OnLeft[8,6]
   + HorizontalOverlap[6,8] = 1
 LinkHorizontalOverlap[6,9]: OnLeft[6,9] + OnLeft[9,6]
   + HorizontalOverlap[6,9] = 1
 LinkHorizontalOverlap[6,10]: OnLeft[6,10] + OnLeft[10,6]
   + HorizontalOverlap[6,10] = 1
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
 LinkHorizontalOverlap[7,8]: OnLeft[7,8] + OnLeft[8,7]
   + HorizontalOverlap[7,8] = 1
 LinkHorizontalOverlap[7,9]: OnLeft[7,9] + OnLeft[9,7]
   + HorizontalOverlap[7,9] = 1
 LinkHorizontalOverlap[7,10]: OnLeft[7,10] + OnLeft[10,7]
   + HorizontalOverlap[7,10] = 1
 LinkHorizontalOverlap[8,0]: OnLeft[0,8] + OnLeft[8,0]
   + HorizontalOverlap[8,0] = 1
 LinkHorizontalOverlap[8,1]: OnLeft[1,8] + OnLeft[8,1]
   + HorizontalOverlap[8,1] = 1
 LinkHorizontalOverlap[8,2]: OnLeft[2,8] + OnLeft[8,2]
   + HorizontalOverlap[8,2] = 1
 LinkHorizontalOverlap[8,3]: OnLeft[3,8] + OnLeft[8,3]
   + HorizontalOverlap[8,3] = 1
 LinkHorizontalOverlap[8,4]: OnLeft[4,8] + OnLeft[8,4]
   + HorizontalOverlap[8,4] = 1
 LinkHorizontalOverlap[8,5]: OnLeft[5,8] + OnLeft[8,5]
   + HorizontalOverlap[8,5] = 1
 LinkHorizontalOverlap[8,6]: OnLeft[6,8] + OnLeft[8,6]
   + HorizontalOverlap[8,6] = 1
 LinkHorizontalOverlap[8,7]: OnLeft[7,8] + OnLeft[8,7]
   + HorizontalOverlap[8,7] = 1
 LinkHorizontalOverlap[8,9]: OnLeft[8,9] + OnLeft[9,8]
   + HorizontalOverlap[8,9] = 1
 LinkHorizontalOverlap[8,10]: OnLeft[8,10] + OnLeft[10,8]
   + HorizontalOverlap[8,10] = 1
 LinkHorizontalOverlap[9,0]: OnLeft[0,9] + OnLeft[9,0]
   + HorizontalOverlap[9,0] = 1
 LinkHorizontalOverlap[9,1]: OnLeft[1,9] + OnLeft[9,1]
   + HorizontalOverlap[9,1] = 1
 LinkHorizontalOverlap[9,2]: OnLeft[2,9] + OnLeft[9,2]
   + HorizontalOverlap[9,2] = 1
 LinkHorizontalOverlap[9,3]: OnLeft[3,9] + OnLeft[9,3]
   + HorizontalOverlap[9,3] = 1
 LinkHorizontalOverlap[9,4]: OnLeft[4,9] + OnLeft[9,4]
   + HorizontalOverlap[9,4] = 1
 LinkHorizontalOverlap[9,5]: OnLeft[5,9] + OnLeft[9,5]
   + HorizontalOverlap[9,5] = 1
 LinkHorizontalOverlap[9,6]: OnLeft[6,9] + OnLeft[9,6]
   + HorizontalOverlap[9,6] = 1
 LinkHorizontalOverlap[9,7]: OnLeft[7,9] + OnLeft[9,7]
   + HorizontalOverlap[9,7] = 1
 LinkHorizontalOverlap[9,8]: OnLeft[8,9] + OnLeft[9,8]
   + HorizontalOverlap[9,8] = 1
 LinkHorizontalOverlap[9,10]: OnLeft[9,10] + OnLeft[10,9]
   + HorizontalOverlap[9,10] = 1
 LinkHorizontalOverlap[10,0]: OnLeft[0,10] + OnLeft[10,0]
   + HorizontalOverlap[10,0] = 1
 LinkHorizontalOverlap[10,1]: OnLeft[1,10] + OnLeft[10,1]
   + HorizontalOverlap[10,1] = 1
 LinkHorizontalOverlap[10,2]: OnLeft[2,10] + OnLeft[10,2]
   + HorizontalOverlap[10,2] = 1
 LinkHorizontalOverlap[10,3]: OnLeft[3,10] + OnLeft[10,3]
   + HorizontalOverlap[10,3] = 1
 LinkHorizontalOverlap[10,4]: OnLeft[4,10] + OnLeft[10,4]
   + HorizontalOverlap[10,4] = 1
 LinkHorizontalOverlap[10,5]: OnLeft[5,10] + OnLeft[10,5]
   + HorizontalOverlap[10,5] = 1
 LinkHorizontalOverlap[10,6]: OnLeft[6,10] + OnLeft[10,6]
   + HorizontalOverlap[10,6] = 1
 LinkHorizontalOverlap[10,7]: OnLeft[7,10] + OnLeft[10,7]
   + HorizontalOverlap[10,7] = 1
 LinkHorizontalOverlap[10,8]: OnLeft[8,10] + OnLeft[10,8]
   + HorizontalOverlap[10,8] = 1
 LinkHorizontalOverlap[10,9]: OnLeft[9,10] + OnLeft[10,9]
   + HorizontalOverlap[10,9] = 1
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
 LinkVerticalOverlap[0,8]: Above[0,8] + Above[8,0] + VerticalOverlap[0,8]
   = 1
 LinkVerticalOverlap[0,9]: Above[0,9] + Above[9,0] + VerticalOverlap[0,9]
   = 1
 LinkVerticalOverlap[0,10]: Above[0,10] + Above[10,0]
   + VerticalOverlap[0,10] = 1
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
 LinkVerticalOverlap[1,8]: Above[1,8] + Above[8,1] + VerticalOverlap[1,8]
   = 1
 LinkVerticalOverlap[1,9]: Above[1,9] + Above[9,1] + VerticalOverlap[1,9]
   = 1
 LinkVerticalOverlap[1,10]: Above[1,10] + Above[10,1]
   + VerticalOverlap[1,10] = 1
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
 LinkVerticalOverlap[2,8]: Above[2,8] + Above[8,2] + VerticalOverlap[2,8]
   = 1
 LinkVerticalOverlap[2,9]: Above[2,9] + Above[9,2] + VerticalOverlap[2,9]
   = 1
 LinkVerticalOverlap[2,10]: Above[2,10] + Above[10,2]
   + VerticalOverlap[2,10] = 1
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
 LinkVerticalOverlap[3,8]: Above[3,8] + Above[8,3] + VerticalOverlap[3,8]
   = 1
 LinkVerticalOverlap[3,9]: Above[3,9] + Above[9,3] + VerticalOverlap[3,9]
   = 1
 LinkVerticalOverlap[3,10]: Above[3,10] + Above[10,3]
   + VerticalOverlap[3,10] = 1
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
 LinkVerticalOverlap[4,8]: Above[4,8] + Above[8,4] + VerticalOverlap[4,8]
   = 1
 LinkVerticalOverlap[4,9]: Above[4,9] + Above[9,4] + VerticalOverlap[4,9]
   = 1
 LinkVerticalOverlap[4,10]: Above[4,10] + Above[10,4]
   + VerticalOverlap[4,10] = 1
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
 LinkVerticalOverlap[5,8]: Above[5,8] + Above[8,5] + VerticalOverlap[5,8]
   = 1
 LinkVerticalOverlap[5,9]: Above[5,9] + Above[9,5] + VerticalOverlap[5,9]
   = 1
 LinkVerticalOverlap[5,10]: Above[5,10] + Above[10,5]
   + VerticalOverlap[5,10] = 1
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
 LinkVerticalOverlap[6,8]: Above[6,8] + Above[8,6] + VerticalOverlap[6,8]
   = 1
 LinkVerticalOverlap[6,9]: Above[6,9] + Above[9,6] + VerticalOverlap[6,9]
   = 1
 LinkVerticalOverlap[6,10]: Above[6,10] + Above[10,6]
   + VerticalOverlap[6,10] = 1
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
 LinkVerticalOverlap[7,8]: Above[7,8] + Above[8,7] + VerticalOverlap[7,8]
   = 1
 LinkVerticalOverlap[7,9]: Above[7,9] + Above[9,7] + VerticalOverlap[7,9]
   = 1
 LinkVerticalOverlap[7,10]: Above[7,10] + Above[10,7]
   + VerticalOverlap[7,10] = 1
 LinkVerticalOverlap[8,0]: Above[0,8] + Above[8,0] + VerticalOverlap[8,0]
   = 1
 LinkVerticalOverlap[8,1]: Above[1,8] + Above[8,1] + VerticalOverlap[8,1]
   = 1
 LinkVerticalOverlap[8,2]: Above[2,8] + Above[8,2] + VerticalOverlap[8,2]
   = 1
 LinkVerticalOverlap[8,3]: Above[3,8] + Above[8,3] + VerticalOverlap[8,3]
   = 1
 LinkVerticalOverlap[8,4]: Above[4,8] + Above[8,4] + VerticalOverlap[8,4]
   = 1
 LinkVerticalOverlap[8,5]: Above[5,8] + Above[8,5] + VerticalOverlap[8,5]
   = 1
 LinkVerticalOverlap[8,6]: Above[6,8] + Above[8,6] + VerticalOverlap[8,6]
   = 1
 LinkVerticalOverlap[8,7]: Above[7,8] + Above[8,7] + VerticalOverlap[8,7]
   = 1
 LinkVerticalOverlap[8,9]: Above[8,9] + Above[9,8] + VerticalOverlap[8,9]
   = 1
 LinkVerticalOverlap[8,10]: Above[8,10] + Above[10,8]
   + VerticalOverlap[8,10] = 1
 LinkVerticalOverlap[9,0]: Above[0,9] + Above[9,0] + VerticalOverlap[9,0]
   = 1
 LinkVerticalOverlap[9,1]: Above[1,9] + Above[9,1] + VerticalOverlap[9,1]
   = 1
 LinkVerticalOverlap[9,2]: Above[2,9] + Above[9,2] + VerticalOverlap[9,2]
   = 1
 LinkVerticalOverlap[9,3]: Above[3,9] + Above[9,3] + VerticalOverlap[9,3]
   = 1
 LinkVerticalOverlap[9,4]: Above[4,9] + Above[9,4] + VerticalOverlap[9,4]
   = 1
 LinkVerticalOverlap[9,5]: Above[5,9] + Above[9,5] + VerticalOverlap[9,5]
   = 1
 LinkVerticalOverlap[9,6]: Above[6,9] + Above[9,6] + VerticalOverlap[9,6]
   = 1
 LinkVerticalOverlap[9,7]: Above[7,9] + Above[9,7] + VerticalOverlap[9,7]
   = 1
 LinkVerticalOverlap[9,8]: Above[8,9] + Above[9,8] + VerticalOverlap[9,8]
   = 1
 LinkVerticalOverlap[9,10]: Above[9,10] + Above[10,9]
   + VerticalOverlap[9,10] = 1
 LinkVerticalOverlap[10,0]: Above[0,10] + Above[10,0]
   + VerticalOverlap[10,0] = 1
 LinkVerticalOverlap[10,1]: Above[1,10] + Above[10,1]
   + VerticalOverlap[10,1] = 1
 LinkVerticalOverlap[10,2]: Above[2,10] + Above[10,2]
   + VerticalOverlap[10,2] = 1
 LinkVerticalOverlap[10,3]: Above[3,10] + Above[10,3]
   + VerticalOverlap[10,3] = 1
 LinkVerticalOverlap[10,4]: Above[4,10] + Above[10,4]
   + VerticalOverlap[10,4] = 1
 LinkVerticalOverlap[10,5]: Above[5,10] + Above[10,5]
   + VerticalOverlap[10,5] = 1
 LinkVerticalOverlap[10,6]: Above[6,10] + Above[10,6]
   + VerticalOverlap[10,6] = 1
 LinkVerticalOverlap[10,7]: Above[7,10] + Above[10,7]
   + VerticalOverlap[10,7] = 1
 LinkVerticalOverlap[10,8]: Above[8,10] + Above[10,8]
   + VerticalOverlap[10,8] = 1
 LinkVerticalOverlap[10,9]: Above[9,10] + Above[10,9]
   + VerticalOverlap[10,9] = 1
 Linkx0Diff[0,0]: EdgeDistance[x0,0,0] = 0
 Linkx0Diff[0,1]: - X0[0] + X0[1] + EdgeDistance[x0,0,1] = 0
 Linkx0Diff[0,2]: - X0[0] + X0[2] + EdgeDistance[x0,0,2] = 0
 Linkx0Diff[0,3]: - X0[0] + X0[3] + EdgeDistance[x0,0,3] = 0
 Linkx0Diff[0,4]: - X0[0] + X0[4] + EdgeDistance[x0,0,4] = 0
 Linkx0Diff[0,5]: - X0[0] + X0[5] + EdgeDistance[x0,0,5] = 0
 Linkx0Diff[0,6]: - X0[0] + X0[6] + EdgeDistance[x0,0,6] = 0
 Linkx0Diff[0,7]: - X0[0] + X0[7] + EdgeDistance[x0,0,7] = 0
 Linkx0Diff[0,8]: - X0[0] + X0[8] + EdgeDistance[x0,0,8] = 0
 Linkx0Diff[0,9]: - X0[0] + X0[9] + EdgeDistance[x0,0,9] = 0
 Linkx0Diff[0,10]: - X0[0] + X0[10] + EdgeDistance[x0,0,10] = 0
 Linkx0Diff[1,0]: X0[0] - X0[1] + EdgeDistance[x0,1,0] = 0
 Linkx0Diff[1,1]: EdgeDistance[x0,1,1] = 0
 Linkx0Diff[1,2]: - X0[1] + X0[2] + EdgeDistance[x0,1,2] = 0
 Linkx0Diff[1,3]: - X0[1] + X0[3] + EdgeDistance[x0,1,3] = 0
 Linkx0Diff[1,4]: - X0[1] + X0[4] + EdgeDistance[x0,1,4] = 0
 Linkx0Diff[1,5]: - X0[1] + X0[5] + EdgeDistance[x0,1,5] = 0
 Linkx0Diff[1,6]: - X0[1] + X0[6] + EdgeDistance[x0,1,6] = 0
 Linkx0Diff[1,7]: - X0[1] + X0[7] + EdgeDistance[x0,1,7] = 0
 Linkx0Diff[1,8]: - X0[1] + X0[8] + EdgeDistance[x0,1,8] = 0
 Linkx0Diff[1,9]: - X0[1] + X0[9] + EdgeDistance[x0,1,9] = 0
 Linkx0Diff[1,10]: - X0[1] + X0[10] + EdgeDistance[x0,1,10] = 0
 Linkx0Diff[2,0]: X0[0] - X0[2] + EdgeDistance[x0,2,0] = 0
 Linkx0Diff[2,1]: X0[1] - X0[2] + EdgeDistance[x0,2,1] = 0
 Linkx0Diff[2,2]: EdgeDistance[x0,2,2] = 0
 Linkx0Diff[2,3]: - X0[2] + X0[3] + EdgeDistance[x0,2,3] = 0
 Linkx0Diff[2,4]: - X0[2] + X0[4] + EdgeDistance[x0,2,4] = 0
 Linkx0Diff[2,5]: - X0[2] + X0[5] + EdgeDistance[x0,2,5] = 0
 Linkx0Diff[2,6]: - X0[2] + X0[6] + EdgeDistance[x0,2,6] = 0
 Linkx0Diff[2,7]: - X0[2] + X0[7] + EdgeDistance[x0,2,7] = 0
 Linkx0Diff[2,8]: - X0[2] + X0[8] + EdgeDistance[x0,2,8] = 0
 Linkx0Diff[2,9]: - X0[2] + X0[9] + EdgeDistance[x0,2,9] = 0
 Linkx0Diff[2,10]: - X0[2] + X0[10] + EdgeDistance[x0,2,10] = 0
 Linkx0Diff[3,0]: X0[0] - X0[3] + EdgeDistance[x0,3,0] = 0
 Linkx0Diff[3,1]: X0[1] - X0[3] + EdgeDistance[x0,3,1] = 0
 Linkx0Diff[3,2]: X0[2] - X0[3] + EdgeDistance[x0,3,2] = 0
 Linkx0Diff[3,3]: EdgeDistance[x0,3,3] = 0
 Linkx0Diff[3,4]: - X0[3] + X0[4] + EdgeDistance[x0,3,4] = 0
 Linkx0Diff[3,5]: - X0[3] + X0[5] + EdgeDistance[x0,3,5] = 0
 Linkx0Diff[3,6]: - X0[3] + X0[6] + EdgeDistance[x0,3,6] = 0
 Linkx0Diff[3,7]: - X0[3] + X0[7] + EdgeDistance[x0,3,7] = 0
 Linkx0Diff[3,8]: - X0[3] + X0[8] + EdgeDistance[x0,3,8] = 0
 Linkx0Diff[3,9]: - X0[3] + X0[9] + EdgeDistance[x0,3,9] = 0
 Linkx0Diff[3,10]: - X0[3] + X0[10] + EdgeDistance[x0,3,10] = 0
 Linkx0Diff[4,0]: X0[0] - X0[4] + EdgeDistance[x0,4,0] = 0
 Linkx0Diff[4,1]: X0[1] - X0[4] + EdgeDistance[x0,4,1] = 0
 Linkx0Diff[4,2]: X0[2] - X0[4] + EdgeDistance[x0,4,2] = 0
 Linkx0Diff[4,3]: X0[3] - X0[4] + EdgeDistance[x0,4,3] = 0
 Linkx0Diff[4,4]: EdgeDistance[x0,4,4] = 0
 Linkx0Diff[4,5]: - X0[4] + X0[5] + EdgeDistance[x0,4,5] = 0
 Linkx0Diff[4,6]: - X0[4] + X0[6] + EdgeDistance[x0,4,6] = 0
 Linkx0Diff[4,7]: - X0[4] + X0[7] + EdgeDistance[x0,4,7] = 0
 Linkx0Diff[4,8]: - X0[4] + X0[8] + EdgeDistance[x0,4,8] = 0
 Linkx0Diff[4,9]: - X0[4] + X0[9] + EdgeDistance[x0,4,9] = 0
 Linkx0Diff[4,10]: - X0[4] + X0[10] + EdgeDistance[x0,4,10] = 0
 Linkx0Diff[5,0]: X0[0] - X0[5] + EdgeDistance[x0,5,0] = 0
 Linkx0Diff[5,1]: X0[1] - X0[5] + EdgeDistance[x0,5,1] = 0
 Linkx0Diff[5,2]: X0[2] - X0[5] + EdgeDistance[x0,5,2] = 0
 Linkx0Diff[5,3]: X0[3] - X0[5] + EdgeDistance[x0,5,3] = 0
 Linkx0Diff[5,4]: X0[4] - X0[5] + EdgeDistance[x0,5,4] = 0
 Linkx0Diff[5,5]: EdgeDistance[x0,5,5] = 0
 Linkx0Diff[5,6]: - X0[5] + X0[6] + EdgeDistance[x0,5,6] = 0
 Linkx0Diff[5,7]: - X0[5] + X0[7] + EdgeDistance[x0,5,7] = 0
 Linkx0Diff[5,8]: - X0[5] + X0[8] + EdgeDistance[x0,5,8] = 0
 Linkx0Diff[5,9]: - X0[5] + X0[9] + EdgeDistance[x0,5,9] = 0
 Linkx0Diff[5,10]: - X0[5] + X0[10] + EdgeDistance[x0,5,10] = 0
 Linkx0Diff[6,0]: X0[0] - X0[6] + EdgeDistance[x0,6,0] = 0
 Linkx0Diff[6,1]: X0[1] - X0[6] + EdgeDistance[x0,6,1] = 0
 Linkx0Diff[6,2]: X0[2] - X0[6] + EdgeDistance[x0,6,2] = 0
 Linkx0Diff[6,3]: X0[3] - X0[6] + EdgeDistance[x0,6,3] = 0
 Linkx0Diff[6,4]: X0[4] - X0[6] + EdgeDistance[x0,6,4] = 0
 Linkx0Diff[6,5]: X0[5] - X0[6] + EdgeDistance[x0,6,5] = 0
 Linkx0Diff[6,6]: EdgeDistance[x0,6,6] = 0
 Linkx0Diff[6,7]: - X0[6] + X0[7] + EdgeDistance[x0,6,7] = 0
 Linkx0Diff[6,8]: - X0[6] + X0[8] + EdgeDistance[x0,6,8] = 0
 Linkx0Diff[6,9]: - X0[6] + X0[9] + EdgeDistance[x0,6,9] = 0
 Linkx0Diff[6,10]: - X0[6] + X0[10] + EdgeDistance[x0,6,10] = 0
 Linkx0Diff[7,0]: X0[0] - X0[7] + EdgeDistance[x0,7,0] = 0
 Linkx0Diff[7,1]: X0[1] - X0[7] + EdgeDistance[x0,7,1] = 0
 Linkx0Diff[7,2]: X0[2] - X0[7] + EdgeDistance[x0,7,2] = 0
 Linkx0Diff[7,3]: X0[3] - X0[7] + EdgeDistance[x0,7,3] = 0
 Linkx0Diff[7,4]: X0[4] - X0[7] + EdgeDistance[x0,7,4] = 0
 Linkx0Diff[7,5]: X0[5] - X0[7] + EdgeDistance[x0,7,5] = 0
 Linkx0Diff[7,6]: X0[6] - X0[7] + EdgeDistance[x0,7,6] = 0
 Linkx0Diff[7,7]: EdgeDistance[x0,7,7] = 0
 Linkx0Diff[7,8]: - X0[7] + X0[8] + EdgeDistance[x0,7,8] = 0
 Linkx0Diff[7,9]: - X0[7] + X0[9] + EdgeDistance[x0,7,9] = 0
 Linkx0Diff[7,10]: - X0[7] + X0[10] + EdgeDistance[x0,7,10] = 0
 Linkx0Diff[8,0]: X0[0] - X0[8] + EdgeDistance[x0,8,0] = 0
 Linkx0Diff[8,1]: X0[1] - X0[8] + EdgeDistance[x0,8,1] = 0
 Linkx0Diff[8,2]: X0[2] - X0[8] + EdgeDistance[x0,8,2] = 0
 Linkx0Diff[8,3]: X0[3] - X0[8] + EdgeDistance[x0,8,3] = 0
 Linkx0Diff[8,4]: X0[4] - X0[8] + EdgeDistance[x0,8,4] = 0
 Linkx0Diff[8,5]: X0[5] - X0[8] + EdgeDistance[x0,8,5] = 0
 Linkx0Diff[8,6]: X0[6] - X0[8] + EdgeDistance[x0,8,6] = 0
 Linkx0Diff[8,7]: X0[7] - X0[8] + EdgeDistance[x0,8,7] = 0
 Linkx0Diff[8,8]: EdgeDistance[x0,8,8] = 0
 Linkx0Diff[8,9]: - X0[8] + X0[9] + EdgeDistance[x0,8,9] = 0
 Linkx0Diff[8,10]: - X0[8] + X0[10] + EdgeDistance[x0,8,10] = 0
 Linkx0Diff[9,0]: X0[0] - X0[9] + EdgeDistance[x0,9,0] = 0
 Linkx0Diff[9,1]: X0[1] - X0[9] + EdgeDistance[x0,9,1] = 0
 Linkx0Diff[9,2]: X0[2] - X0[9] + EdgeDistance[x0,9,2] = 0
 Linkx0Diff[9,3]: X0[3] - X0[9] + EdgeDistance[x0,9,3] = 0
 Linkx0Diff[9,4]: X0[4] - X0[9] + EdgeDistance[x0,9,4] = 0
 Linkx0Diff[9,5]: X0[5] - X0[9] + EdgeDistance[x0,9,5] = 0
 Linkx0Diff[9,6]: X0[6] - X0[9] + EdgeDistance[x0,9,6] = 0
 Linkx0Diff[9,7]: X0[7] - X0[9] + EdgeDistance[x0,9,7] = 0
 Linkx0Diff[9,8]: X0[8] - X0[9] + EdgeDistance[x0,9,8] = 0
 Linkx0Diff[9,9]: EdgeDistance[x0,9,9] = 0
 Linkx0Diff[9,10]: - X0[9] + X0[10] + EdgeDistance[x0,9,10] = 0
 Linkx0Diff[10,0]: X0[0] - X0[10] + EdgeDistance[x0,10,0] = 0
 Linkx0Diff[10,1]: X0[1] - X0[10] + EdgeDistance[x0,10,1] = 0
 Linkx0Diff[10,2]: X0[2] - X0[10] + EdgeDistance[x0,10,2] = 0
 Linkx0Diff[10,3]: X0[3] - X0[10] + EdgeDistance[x0,10,3] = 0
 Linkx0Diff[10,4]: X0[4] - X0[10] + EdgeDistance[x0,10,4] = 0
 Linkx0Diff[10,5]: X0[5] - X0[10] + EdgeDistance[x0,10,5] = 0
 Linkx0Diff[10,6]: X0[6] - X0[10] + EdgeDistance[x0,10,6] = 0
 Linkx0Diff[10,7]: X0[7] - X0[10] + EdgeDistance[x0,10,7] = 0
 Linkx0Diff[10,8]: X0[8] - X0[10] + EdgeDistance[x0,10,8] = 0
 Linkx0Diff[10,9]: X0[9] - X0[10] + EdgeDistance[x0,10,9] = 0
 Linkx0Diff[10,10]: EdgeDistance[x0,10,10] = 0
 Linky0Diff[0,0]: EdgeDistance[y0,0,0] = 0
 Linky0Diff[0,1]: - Y0[0] + Y0[1] + EdgeDistance[y0,0,1] = 0
 Linky0Diff[0,2]: - Y0[0] + Y0[2] + EdgeDistance[y0,0,2] = 0
 Linky0Diff[0,3]: - Y0[0] + Y0[3] + EdgeDistance[y0,0,3] = 0
 Linky0Diff[0,4]: - Y0[0] + Y0[4] + EdgeDistance[y0,0,4] = 0
 Linky0Diff[0,5]: - Y0[0] + Y0[5] + EdgeDistance[y0,0,5] = 0
 Linky0Diff[0,6]: - Y0[0] + Y0[6] + EdgeDistance[y0,0,6] = 0
 Linky0Diff[0,7]: - Y0[0] + Y0[7] + EdgeDistance[y0,0,7] = 0
 Linky0Diff[0,8]: - Y0[0] + Y0[8] + EdgeDistance[y0,0,8] = 0
 Linky0Diff[0,9]: - Y0[0] + Y0[9] + EdgeDistance[y0,0,9] = 0
 Linky0Diff[0,10]: - Y0[0] + Y0[10] + EdgeDistance[y0,0,10] = 0
 Linky0Diff[1,0]: Y0[0] - Y0[1] + EdgeDistance[y0,1,0] = 0
 Linky0Diff[1,1]: EdgeDistance[y0,1,1] = 0
 Linky0Diff[1,2]: - Y0[1] + Y0[2] + EdgeDistance[y0,1,2] = 0
 Linky0Diff[1,3]: - Y0[1] + Y0[3] + EdgeDistance[y0,1,3] = 0
 Linky0Diff[1,4]: - Y0[1] + Y0[4] + EdgeDistance[y0,1,4] = 0
 Linky0Diff[1,5]: - Y0[1] + Y0[5] + EdgeDistance[y0,1,5] = 0
 Linky0Diff[1,6]: - Y0[1] + Y0[6] + EdgeDistance[y0,1,6] = 0
 Linky0Diff[1,7]: - Y0[1] + Y0[7] + EdgeDistance[y0,1,7] = 0
 Linky0Diff[1,8]: - Y0[1] + Y0[8] + EdgeDistance[y0,1,8] = 0
 Linky0Diff[1,9]: - Y0[1] + Y0[9] + EdgeDistance[y0,1,9] = 0
 Linky0Diff[1,10]: - Y0[1] + Y0[10] + EdgeDistance[y0,1,10] = 0
 Linky0Diff[2,0]: Y0[0] - Y0[2] + EdgeDistance[y0,2,0] = 0
 Linky0Diff[2,1]: Y0[1] - Y0[2] + EdgeDistance[y0,2,1] = 0
 Linky0Diff[2,2]: EdgeDistance[y0,2,2] = 0
 Linky0Diff[2,3]: - Y0[2] + Y0[3] + EdgeDistance[y0,2,3] = 0
 Linky0Diff[2,4]: - Y0[2] + Y0[4] + EdgeDistance[y0,2,4] = 0
 Linky0Diff[2,5]: - Y0[2] + Y0[5] + EdgeDistance[y0,2,5] = 0
 Linky0Diff[2,6]: - Y0[2] + Y0[6] + EdgeDistance[y0,2,6] = 0
 Linky0Diff[2,7]: - Y0[2] + Y0[7] + EdgeDistance[y0,2,7] = 0
 Linky0Diff[2,8]: - Y0[2] + Y0[8] + EdgeDistance[y0,2,8] = 0
 Linky0Diff[2,9]: - Y0[2] + Y0[9] + EdgeDistance[y0,2,9] = 0
 Linky0Diff[2,10]: - Y0[2] + Y0[10] + EdgeDistance[y0,2,10] = 0
 Linky0Diff[3,0]: Y0[0] - Y0[3] + EdgeDistance[y0,3,0] = 0
 Linky0Diff[3,1]: Y0[1] - Y0[3] + EdgeDistance[y0,3,1] = 0
 Linky0Diff[3,2]: Y0[2] - Y0[3] + EdgeDistance[y0,3,2] = 0
 Linky0Diff[3,3]: EdgeDistance[y0,3,3] = 0
 Linky0Diff[3,4]: - Y0[3] + Y0[4] + EdgeDistance[y0,3,4] = 0
 Linky0Diff[3,5]: - Y0[3] + Y0[5] + EdgeDistance[y0,3,5] = 0
 Linky0Diff[3,6]: - Y0[3] + Y0[6] + EdgeDistance[y0,3,6] = 0
 Linky0Diff[3,7]: - Y0[3] + Y0[7] + EdgeDistance[y0,3,7] = 0
 Linky0Diff[3,8]: - Y0[3] + Y0[8] + EdgeDistance[y0,3,8] = 0
 Linky0Diff[3,9]: - Y0[3] + Y0[9] + EdgeDistance[y0,3,9] = 0
 Linky0Diff[3,10]: - Y0[3] + Y0[10] + EdgeDistance[y0,3,10] = 0
 Linky0Diff[4,0]: Y0[0] - Y0[4] + EdgeDistance[y0,4,0] = 0
 Linky0Diff[4,1]: Y0[1] - Y0[4] + EdgeDistance[y0,4,1] = 0
 Linky0Diff[4,2]: Y0[2] - Y0[4] + EdgeDistance[y0,4,2] = 0
 Linky0Diff[4,3]: Y0[3] - Y0[4] + EdgeDistance[y0,4,3] = 0
 Linky0Diff[4,4]: EdgeDistance[y0,4,4] = 0
 Linky0Diff[4,5]: - Y0[4] + Y0[5] + EdgeDistance[y0,4,5] = 0
 Linky0Diff[4,6]: - Y0[4] + Y0[6] + EdgeDistance[y0,4,6] = 0
 Linky0Diff[4,7]: - Y0[4] + Y0[7] + EdgeDistance[y0,4,7] = 0
 Linky0Diff[4,8]: - Y0[4] + Y0[8] + EdgeDistance[y0,4,8] = 0
 Linky0Diff[4,9]: - Y0[4] + Y0[9] + EdgeDistance[y0,4,9] = 0
 Linky0Diff[4,10]: - Y0[4] + Y0[10] + EdgeDistance[y0,4,10] = 0
 Linky0Diff[5,0]: Y0[0] - Y0[5] + EdgeDistance[y0,5,0] = 0
 Linky0Diff[5,1]: Y0[1] - Y0[5] + EdgeDistance[y0,5,1] = 0
 Linky0Diff[5,2]: Y0[2] - Y0[5] + EdgeDistance[y0,5,2] = 0
 Linky0Diff[5,3]: Y0[3] - Y0[5] + EdgeDistance[y0,5,3] = 0
 Linky0Diff[5,4]: Y0[4] - Y0[5] + EdgeDistance[y0,5,4] = 0
 Linky0Diff[5,5]: EdgeDistance[y0,5,5] = 0
 Linky0Diff[5,6]: - Y0[5] + Y0[6] + EdgeDistance[y0,5,6] = 0
 Linky0Diff[5,7]: - Y0[5] + Y0[7] + EdgeDistance[y0,5,7] = 0
 Linky0Diff[5,8]: - Y0[5] + Y0[8] + EdgeDistance[y0,5,8] = 0
 Linky0Diff[5,9]: - Y0[5] + Y0[9] + EdgeDistance[y0,5,9] = 0
 Linky0Diff[5,10]: - Y0[5] + Y0[10] + EdgeDistance[y0,5,10] = 0
 Linky0Diff[6,0]: Y0[0] - Y0[6] + EdgeDistance[y0,6,0] = 0
 Linky0Diff[6,1]: Y0[1] - Y0[6] + EdgeDistance[y0,6,1] = 0
 Linky0Diff[6,2]: Y0[2] - Y0[6] + EdgeDistance[y0,6,2] = 0
 Linky0Diff[6,3]: Y0[3] - Y0[6] + EdgeDistance[y0,6,3] = 0
 Linky0Diff[6,4]: Y0[4] - Y0[6] + EdgeDistance[y0,6,4] = 0
 Linky0Diff[6,5]: Y0[5] - Y0[6] + EdgeDistance[y0,6,5] = 0
 Linky0Diff[6,6]: EdgeDistance[y0,6,6] = 0
 Linky0Diff[6,7]: - Y0[6] + Y0[7] + EdgeDistance[y0,6,7] = 0
 Linky0Diff[6,8]: - Y0[6] + Y0[8] + EdgeDistance[y0,6,8] = 0
 Linky0Diff[6,9]: - Y0[6] + Y0[9] + EdgeDistance[y0,6,9] = 0
 Linky0Diff[6,10]: - Y0[6] + Y0[10] + EdgeDistance[y0,6,10] = 0
 Linky0Diff[7,0]: Y0[0] - Y0[7] + EdgeDistance[y0,7,0] = 0
 Linky0Diff[7,1]: Y0[1] - Y0[7] + EdgeDistance[y0,7,1] = 0
 Linky0Diff[7,2]: Y0[2] - Y0[7] + EdgeDistance[y0,7,2] = 0
 Linky0Diff[7,3]: Y0[3] - Y0[7] + EdgeDistance[y0,7,3] = 0
 Linky0Diff[7,4]: Y0[4] - Y0[7] + EdgeDistance[y0,7,4] = 0
 Linky0Diff[7,5]: Y0[5] - Y0[7] + EdgeDistance[y0,7,5] = 0
 Linky0Diff[7,6]: Y0[6] - Y0[7] + EdgeDistance[y0,7,6] = 0
 Linky0Diff[7,7]: EdgeDistance[y0,7,7] = 0
 Linky0Diff[7,8]: - Y0[7] + Y0[8] + EdgeDistance[y0,7,8] = 0
 Linky0Diff[7,9]: - Y0[7] + Y0[9] + EdgeDistance[y0,7,9] = 0
 Linky0Diff[7,10]: - Y0[7] + Y0[10] + EdgeDistance[y0,7,10] = 0
 Linky0Diff[8,0]: Y0[0] - Y0[8] + EdgeDistance[y0,8,0] = 0
 Linky0Diff[8,1]: Y0[1] - Y0[8] + EdgeDistance[y0,8,1] = 0
 Linky0Diff[8,2]: Y0[2] - Y0[8] + EdgeDistance[y0,8,2] = 0
 Linky0Diff[8,3]: Y0[3] - Y0[8] + EdgeDistance[y0,8,3] = 0
 Linky0Diff[8,4]: Y0[4] - Y0[8] + EdgeDistance[y0,8,4] = 0
 Linky0Diff[8,5]: Y0[5] - Y0[8] + EdgeDistance[y0,8,5] = 0
 Linky0Diff[8,6]: Y0[6] - Y0[8] + EdgeDistance[y0,8,6] = 0
 Linky0Diff[8,7]: Y0[7] - Y0[8] + EdgeDistance[y0,8,7] = 0
 Linky0Diff[8,8]: EdgeDistance[y0,8,8] = 0
 Linky0Diff[8,9]: - Y0[8] + Y0[9] + EdgeDistance[y0,8,9] = 0
 Linky0Diff[8,10]: - Y0[8] + Y0[10] + EdgeDistance[y0,8,10] = 0
 Linky0Diff[9,0]: Y0[0] - Y0[9] + EdgeDistance[y0,9,0] = 0
 Linky0Diff[9,1]: Y0[1] - Y0[9] + EdgeDistance[y0,9,1] = 0
 Linky0Diff[9,2]: Y0[2] - Y0[9] + EdgeDistance[y0,9,2] = 0
 Linky0Diff[9,3]: Y0[3] - Y0[9] + EdgeDistance[y0,9,3] = 0
 Linky0Diff[9,4]: Y0[4] - Y0[9] + EdgeDistance[y0,9,4] = 0
 Linky0Diff[9,5]: Y0[5] - Y0[9] + EdgeDistance[y0,9,5] = 0
 Linky0Diff[9,6]: Y0[6] - Y0[9] + EdgeDistance[y0,9,6] = 0
 Linky0Diff[9,7]: Y0[7] - Y0[9] + EdgeDistance[y0,9,7] = 0
 Linky0Diff[9,8]: Y0[8] - Y0[9] + EdgeDistance[y0,9,8] = 0
 Linky0Diff[9,9]: EdgeDistance[y0,9,9] = 0
 Linky0Diff[9,10]: - Y0[9] + Y0[10] + EdgeDistance[y0,9,10] = 0
 Linky0Diff[10,0]: Y0[0] - Y0[10] + EdgeDistance[y0,10,0] = 0
 Linky0Diff[10,1]: Y0[1] - Y0[10] + EdgeDistance[y0,10,1] = 0
 Linky0Diff[10,2]: Y0[2] - Y0[10] + EdgeDistance[y0,10,2] = 0
 Linky0Diff[10,3]: Y0[3] - Y0[10] + EdgeDistance[y0,10,3] = 0
 Linky0Diff[10,4]: Y0[4] - Y0[10] + EdgeDistance[y0,10,4] = 0
 Linky0Diff[10,5]: Y0[5] - Y0[10] + EdgeDistance[y0,10,5] = 0
 Linky0Diff[10,6]: Y0[6] - Y0[10] + EdgeDistance[y0,10,6] = 0
 Linky0Diff[10,7]: Y0[7] - Y0[10] + EdgeDistance[y0,10,7] = 0
 Linky0Diff[10,8]: Y0[8] - Y0[10] + EdgeDistance[y0,10,8] = 0
 Linky0Diff[10,9]: Y0[9] - Y0[10] + EdgeDistance[y0,10,9] = 0
 Linky0Diff[10,10]: EdgeDistance[y0,10,10] = 0
 Linkx1Diff[0,0]: EdgeDistance[x1,0,0] = 0
 Linkx1Diff[0,1]: - X1[0] + X1[1] + EdgeDistance[x1,0,1] = 0
 Linkx1Diff[0,2]: - X1[0] + X1[2] + EdgeDistance[x1,0,2] = 0
 Linkx1Diff[0,3]: - X1[0] + X1[3] + EdgeDistance[x1,0,3] = 0
 Linkx1Diff[0,4]: - X1[0] + X1[4] + EdgeDistance[x1,0,4] = 0
 Linkx1Diff[0,5]: - X1[0] + X1[5] + EdgeDistance[x1,0,5] = 0
 Linkx1Diff[0,6]: - X1[0] + X1[6] + EdgeDistance[x1,0,6] = 0
 Linkx1Diff[0,7]: - X1[0] + X1[7] + EdgeDistance[x1,0,7] = 0
 Linkx1Diff[0,8]: - X1[0] + X1[8] + EdgeDistance[x1,0,8] = 0
 Linkx1Diff[0,9]: - X1[0] + X1[9] + EdgeDistance[x1,0,9] = 0
 Linkx1Diff[0,10]: - X1[0] + X1[10] + EdgeDistance[x1,0,10] = 0
 Linkx1Diff[1,0]: X1[0] - X1[1] + EdgeDistance[x1,1,0] = 0
 Linkx1Diff[1,1]: EdgeDistance[x1,1,1] = 0
 Linkx1Diff[1,2]: - X1[1] + X1[2] + EdgeDistance[x1,1,2] = 0
 Linkx1Diff[1,3]: - X1[1] + X1[3] + EdgeDistance[x1,1,3] = 0
 Linkx1Diff[1,4]: - X1[1] + X1[4] + EdgeDistance[x1,1,4] = 0
 Linkx1Diff[1,5]: - X1[1] + X1[5] + EdgeDistance[x1,1,5] = 0
 Linkx1Diff[1,6]: - X1[1] + X1[6] + EdgeDistance[x1,1,6] = 0
 Linkx1Diff[1,7]: - X1[1] + X1[7] + EdgeDistance[x1,1,7] = 0
 Linkx1Diff[1,8]: - X1[1] + X1[8] + EdgeDistance[x1,1,8] = 0
 Linkx1Diff[1,9]: - X1[1] + X1[9] + EdgeDistance[x1,1,9] = 0
 Linkx1Diff[1,10]: - X1[1] + X1[10] + EdgeDistance[x1,1,10] = 0
 Linkx1Diff[2,0]: X1[0] - X1[2] + EdgeDistance[x1,2,0] = 0
 Linkx1Diff[2,1]: X1[1] - X1[2] + EdgeDistance[x1,2,1] = 0
 Linkx1Diff[2,2]: EdgeDistance[x1,2,2] = 0
 Linkx1Diff[2,3]: - X1[2] + X1[3] + EdgeDistance[x1,2,3] = 0
 Linkx1Diff[2,4]: - X1[2] + X1[4] + EdgeDistance[x1,2,4] = 0
 Linkx1Diff[2,5]: - X1[2] + X1[5] + EdgeDistance[x1,2,5] = 0
 Linkx1Diff[2,6]: - X1[2] + X1[6] + EdgeDistance[x1,2,6] = 0
 Linkx1Diff[2,7]: - X1[2] + X1[7] + EdgeDistance[x1,2,7] = 0
 Linkx1Diff[2,8]: - X1[2] + X1[8] + EdgeDistance[x1,2,8] = 0
 Linkx1Diff[2,9]: - X1[2] + X1[9] + EdgeDistance[x1,2,9] = 0
 Linkx1Diff[2,10]: - X1[2] + X1[10] + EdgeDistance[x1,2,10] = 0
 Linkx1Diff[3,0]: X1[0] - X1[3] + EdgeDistance[x1,3,0] = 0
 Linkx1Diff[3,1]: X1[1] - X1[3] + EdgeDistance[x1,3,1] = 0
 Linkx1Diff[3,2]: X1[2] - X1[3] + EdgeDistance[x1,3,2] = 0
 Linkx1Diff[3,3]: EdgeDistance[x1,3,3] = 0
 Linkx1Diff[3,4]: - X1[3] + X1[4] + EdgeDistance[x1,3,4] = 0
 Linkx1Diff[3,5]: - X1[3] + X1[5] + EdgeDistance[x1,3,5] = 0
 Linkx1Diff[3,6]: - X1[3] + X1[6] + EdgeDistance[x1,3,6] = 0
 Linkx1Diff[3,7]: - X1[3] + X1[7] + EdgeDistance[x1,3,7] = 0
 Linkx1Diff[3,8]: - X1[3] + X1[8] + EdgeDistance[x1,3,8] = 0
 Linkx1Diff[3,9]: - X1[3] + X1[9] + EdgeDistance[x1,3,9] = 0
 Linkx1Diff[3,10]: - X1[3] + X1[10] + EdgeDistance[x1,3,10] = 0
 Linkx1Diff[4,0]: X1[0] - X1[4] + EdgeDistance[x1,4,0] = 0
 Linkx1Diff[4,1]: X1[1] - X1[4] + EdgeDistance[x1,4,1] = 0
 Linkx1Diff[4,2]: X1[2] - X1[4] + EdgeDistance[x1,4,2] = 0
 Linkx1Diff[4,3]: X1[3] - X1[4] + EdgeDistance[x1,4,3] = 0
 Linkx1Diff[4,4]: EdgeDistance[x1,4,4] = 0
 Linkx1Diff[4,5]: - X1[4] + X1[5] + EdgeDistance[x1,4,5] = 0
 Linkx1Diff[4,6]: - X1[4] + X1[6] + EdgeDistance[x1,4,6] = 0
 Linkx1Diff[4,7]: - X1[4] + X1[7] + EdgeDistance[x1,4,7] = 0
 Linkx1Diff[4,8]: - X1[4] + X1[8] + EdgeDistance[x1,4,8] = 0
 Linkx1Diff[4,9]: - X1[4] + X1[9] + EdgeDistance[x1,4,9] = 0
 Linkx1Diff[4,10]: - X1[4] + X1[10] + EdgeDistance[x1,4,10] = 0
 Linkx1Diff[5,0]: X1[0] - X1[5] + EdgeDistance[x1,5,0] = 0
 Linkx1Diff[5,1]: X1[1] - X1[5] + EdgeDistance[x1,5,1] = 0
 Linkx1Diff[5,2]: X1[2] - X1[5] + EdgeDistance[x1,5,2] = 0
 Linkx1Diff[5,3]: X1[3] - X1[5] + EdgeDistance[x1,5,3] = 0
 Linkx1Diff[5,4]: X1[4] - X1[5] + EdgeDistance[x1,5,4] = 0
 Linkx1Diff[5,5]: EdgeDistance[x1,5,5] = 0
 Linkx1Diff[5,6]: - X1[5] + X1[6] + EdgeDistance[x1,5,6] = 0
 Linkx1Diff[5,7]: - X1[5] + X1[7] + EdgeDistance[x1,5,7] = 0
 Linkx1Diff[5,8]: - X1[5] + X1[8] + EdgeDistance[x1,5,8] = 0
 Linkx1Diff[5,9]: - X1[5] + X1[9] + EdgeDistance[x1,5,9] = 0
 Linkx1Diff[5,10]: - X1[5] + X1[10] + EdgeDistance[x1,5,10] = 0
 Linkx1Diff[6,0]: X1[0] - X1[6] + EdgeDistance[x1,6,0] = 0
 Linkx1Diff[6,1]: X1[1] - X1[6] + EdgeDistance[x1,6,1] = 0
 Linkx1Diff[6,2]: X1[2] - X1[6] + EdgeDistance[x1,6,2] = 0
 Linkx1Diff[6,3]: X1[3] - X1[6] + EdgeDistance[x1,6,3] = 0
 Linkx1Diff[6,4]: X1[4] - X1[6] + EdgeDistance[x1,6,4] = 0
 Linkx1Diff[6,5]: X1[5] - X1[6] + EdgeDistance[x1,6,5] = 0
 Linkx1Diff[6,6]: EdgeDistance[x1,6,6] = 0
 Linkx1Diff[6,7]: - X1[6] + X1[7] + EdgeDistance[x1,6,7] = 0
 Linkx1Diff[6,8]: - X1[6] + X1[8] + EdgeDistance[x1,6,8] = 0
 Linkx1Diff[6,9]: - X1[6] + X1[9] + EdgeDistance[x1,6,9] = 0
 Linkx1Diff[6,10]: - X1[6] + X1[10] + EdgeDistance[x1,6,10] = 0
 Linkx1Diff[7,0]: X1[0] - X1[7] + EdgeDistance[x1,7,0] = 0
 Linkx1Diff[7,1]: X1[1] - X1[7] + EdgeDistance[x1,7,1] = 0
 Linkx1Diff[7,2]: X1[2] - X1[7] + EdgeDistance[x1,7,2] = 0
 Linkx1Diff[7,3]: X1[3] - X1[7] + EdgeDistance[x1,7,3] = 0
 Linkx1Diff[7,4]: X1[4] - X1[7] + EdgeDistance[x1,7,4] = 0
 Linkx1Diff[7,5]: X1[5] - X1[7] + EdgeDistance[x1,7,5] = 0
 Linkx1Diff[7,6]: X1[6] - X1[7] + EdgeDistance[x1,7,6] = 0
 Linkx1Diff[7,7]: EdgeDistance[x1,7,7] = 0
 Linkx1Diff[7,8]: - X1[7] + X1[8] + EdgeDistance[x1,7,8] = 0
 Linkx1Diff[7,9]: - X1[7] + X1[9] + EdgeDistance[x1,7,9] = 0
 Linkx1Diff[7,10]: - X1[7] + X1[10] + EdgeDistance[x1,7,10] = 0
 Linkx1Diff[8,0]: X1[0] - X1[8] + EdgeDistance[x1,8,0] = 0
 Linkx1Diff[8,1]: X1[1] - X1[8] + EdgeDistance[x1,8,1] = 0
 Linkx1Diff[8,2]: X1[2] - X1[8] + EdgeDistance[x1,8,2] = 0
 Linkx1Diff[8,3]: X1[3] - X1[8] + EdgeDistance[x1,8,3] = 0
 Linkx1Diff[8,4]: X1[4] - X1[8] + EdgeDistance[x1,8,4] = 0
 Linkx1Diff[8,5]: X1[5] - X1[8] + EdgeDistance[x1,8,5] = 0
 Linkx1Diff[8,6]: X1[6] - X1[8] + EdgeDistance[x1,8,6] = 0
 Linkx1Diff[8,7]: X1[7] - X1[8] + EdgeDistance[x1,8,7] = 0
 Linkx1Diff[8,8]: EdgeDistance[x1,8,8] = 0
 Linkx1Diff[8,9]: - X1[8] + X1[9] + EdgeDistance[x1,8,9] = 0
 Linkx1Diff[8,10]: - X1[8] + X1[10] + EdgeDistance[x1,8,10] = 0
 Linkx1Diff[9,0]: X1[0] - X1[9] + EdgeDistance[x1,9,0] = 0
 Linkx1Diff[9,1]: X1[1] - X1[9] + EdgeDistance[x1,9,1] = 0
 Linkx1Diff[9,2]: X1[2] - X1[9] + EdgeDistance[x1,9,2] = 0
 Linkx1Diff[9,3]: X1[3] - X1[9] + EdgeDistance[x1,9,3] = 0
 Linkx1Diff[9,4]: X1[4] - X1[9] + EdgeDistance[x1,9,4] = 0
 Linkx1Diff[9,5]: X1[5] - X1[9] + EdgeDistance[x1,9,5] = 0
 Linkx1Diff[9,6]: X1[6] - X1[9] + EdgeDistance[x1,9,6] = 0
 Linkx1Diff[9,7]: X1[7] - X1[9] + EdgeDistance[x1,9,7] = 0
 Linkx1Diff[9,8]: X1[8] - X1[9] + EdgeDistance[x1,9,8] = 0
 Linkx1Diff[9,9]: EdgeDistance[x1,9,9] = 0
 Linkx1Diff[9,10]: - X1[9] + X1[10] + EdgeDistance[x1,9,10] = 0
 Linkx1Diff[10,0]: X1[0] - X1[10] + EdgeDistance[x1,10,0] = 0
 Linkx1Diff[10,1]: X1[1] - X1[10] + EdgeDistance[x1,10,1] = 0
 Linkx1Diff[10,2]: X1[2] - X1[10] + EdgeDistance[x1,10,2] = 0
 Linkx1Diff[10,3]: X1[3] - X1[10] + EdgeDistance[x1,10,3] = 0
 Linkx1Diff[10,4]: X1[4] - X1[10] + EdgeDistance[x1,10,4] = 0
 Linkx1Diff[10,5]: X1[5] - X1[10] + EdgeDistance[x1,10,5] = 0
 Linkx1Diff[10,6]: X1[6] - X1[10] + EdgeDistance[x1,10,6] = 0
 Linkx1Diff[10,7]: X1[7] - X1[10] + EdgeDistance[x1,10,7] = 0
 Linkx1Diff[10,8]: X1[8] - X1[10] + EdgeDistance[x1,10,8] = 0
 Linkx1Diff[10,9]: X1[9] - X1[10] + EdgeDistance[x1,10,9] = 0
 Linkx1Diff[10,10]: EdgeDistance[x1,10,10] = 0
 Linky1Diff[0,0]: EdgeDistance[y1,0,0] = 0
 Linky1Diff[0,1]: - Y1[0] + Y1[1] + EdgeDistance[y1,0,1] = 0
 Linky1Diff[0,2]: - Y1[0] + Y1[2] + EdgeDistance[y1,0,2] = 0
 Linky1Diff[0,3]: - Y1[0] + Y1[3] + EdgeDistance[y1,0,3] = 0
 Linky1Diff[0,4]: - Y1[0] + Y1[4] + EdgeDistance[y1,0,4] = 0
 Linky1Diff[0,5]: - Y1[0] + Y1[5] + EdgeDistance[y1,0,5] = 0
 Linky1Diff[0,6]: - Y1[0] + Y1[6] + EdgeDistance[y1,0,6] = 0
 Linky1Diff[0,7]: - Y1[0] + Y1[7] + EdgeDistance[y1,0,7] = 0
 Linky1Diff[0,8]: - Y1[0] + Y1[8] + EdgeDistance[y1,0,8] = 0
 Linky1Diff[0,9]: - Y1[0] + Y1[9] + EdgeDistance[y1,0,9] = 0
 Linky1Diff[0,10]: - Y1[0] + Y1[10] + EdgeDistance[y1,0,10] = 0
 Linky1Diff[1,0]: Y1[0] - Y1[1] + EdgeDistance[y1,1,0] = 0
 Linky1Diff[1,1]: EdgeDistance[y1,1,1] = 0
 Linky1Diff[1,2]: - Y1[1] + Y1[2] + EdgeDistance[y1,1,2] = 0
 Linky1Diff[1,3]: - Y1[1] + Y1[3] + EdgeDistance[y1,1,3] = 0
 Linky1Diff[1,4]: - Y1[1] + Y1[4] + EdgeDistance[y1,1,4] = 0
 Linky1Diff[1,5]: - Y1[1] + Y1[5] + EdgeDistance[y1,1,5] = 0
 Linky1Diff[1,6]: - Y1[1] + Y1[6] + EdgeDistance[y1,1,6] = 0
 Linky1Diff[1,7]: - Y1[1] + Y1[7] + EdgeDistance[y1,1,7] = 0
 Linky1Diff[1,8]: - Y1[1] + Y1[8] + EdgeDistance[y1,1,8] = 0
 Linky1Diff[1,9]: - Y1[1] + Y1[9] + EdgeDistance[y1,1,9] = 0
 Linky1Diff[1,10]: - Y1[1] + Y1[10] + EdgeDistance[y1,1,10] = 0
 Linky1Diff[2,0]: Y1[0] - Y1[2] + EdgeDistance[y1,2,0] = 0
 Linky1Diff[2,1]: Y1[1] - Y1[2] + EdgeDistance[y1,2,1] = 0
 Linky1Diff[2,2]: EdgeDistance[y1,2,2] = 0
 Linky1Diff[2,3]: - Y1[2] + Y1[3] + EdgeDistance[y1,2,3] = 0
 Linky1Diff[2,4]: - Y1[2] + Y1[4] + EdgeDistance[y1,2,4] = 0
 Linky1Diff[2,5]: - Y1[2] + Y1[5] + EdgeDistance[y1,2,5] = 0
 Linky1Diff[2,6]: - Y1[2] + Y1[6] + EdgeDistance[y1,2,6] = 0
 Linky1Diff[2,7]: - Y1[2] + Y1[7] + EdgeDistance[y1,2,7] = 0
 Linky1Diff[2,8]: - Y1[2] + Y1[8] + EdgeDistance[y1,2,8] = 0
 Linky1Diff[2,9]: - Y1[2] + Y1[9] + EdgeDistance[y1,2,9] = 0
 Linky1Diff[2,10]: - Y1[2] + Y1[10] + EdgeDistance[y1,2,10] = 0
 Linky1Diff[3,0]: Y1[0] - Y1[3] + EdgeDistance[y1,3,0] = 0
 Linky1Diff[3,1]: Y1[1] - Y1[3] + EdgeDistance[y1,3,1] = 0
 Linky1Diff[3,2]: Y1[2] - Y1[3] + EdgeDistance[y1,3,2] = 0
 Linky1Diff[3,3]: EdgeDistance[y1,3,3] = 0
 Linky1Diff[3,4]: - Y1[3] + Y1[4] + EdgeDistance[y1,3,4] = 0
 Linky1Diff[3,5]: - Y1[3] + Y1[5] + EdgeDistance[y1,3,5] = 0
 Linky1Diff[3,6]: - Y1[3] + Y1[6] + EdgeDistance[y1,3,6] = 0
 Linky1Diff[3,7]: - Y1[3] + Y1[7] + EdgeDistance[y1,3,7] = 0
 Linky1Diff[3,8]: - Y1[3] + Y1[8] + EdgeDistance[y1,3,8] = 0
 Linky1Diff[3,9]: - Y1[3] + Y1[9] + EdgeDistance[y1,3,9] = 0
 Linky1Diff[3,10]: - Y1[3] + Y1[10] + EdgeDistance[y1,3,10] = 0
 Linky1Diff[4,0]: Y1[0] - Y1[4] + EdgeDistance[y1,4,0] = 0
 Linky1Diff[4,1]: Y1[1] - Y1[4] + EdgeDistance[y1,4,1] = 0
 Linky1Diff[4,2]: Y1[2] - Y1[4] + EdgeDistance[y1,4,2] = 0
 Linky1Diff[4,3]: Y1[3] - Y1[4] + EdgeDistance[y1,4,3] = 0
 Linky1Diff[4,4]: EdgeDistance[y1,4,4] = 0
 Linky1Diff[4,5]: - Y1[4] + Y1[5] + EdgeDistance[y1,4,5] = 0
 Linky1Diff[4,6]: - Y1[4] + Y1[6] + EdgeDistance[y1,4,6] = 0
 Linky1Diff[4,7]: - Y1[4] + Y1[7] + EdgeDistance[y1,4,7] = 0
 Linky1Diff[4,8]: - Y1[4] + Y1[8] + EdgeDistance[y1,4,8] = 0
 Linky1Diff[4,9]: - Y1[4] + Y1[9] + EdgeDistance[y1,4,9] = 0
 Linky1Diff[4,10]: - Y1[4] + Y1[10] + EdgeDistance[y1,4,10] = 0
 Linky1Diff[5,0]: Y1[0] - Y1[5] + EdgeDistance[y1,5,0] = 0
 Linky1Diff[5,1]: Y1[1] - Y1[5] + EdgeDistance[y1,5,1] = 0
 Linky1Diff[5,2]: Y1[2] - Y1[5] + EdgeDistance[y1,5,2] = 0
 Linky1Diff[5,3]: Y1[3] - Y1[5] + EdgeDistance[y1,5,3] = 0
 Linky1Diff[5,4]: Y1[4] - Y1[5] + EdgeDistance[y1,5,4] = 0
 Linky1Diff[5,5]: EdgeDistance[y1,5,5] = 0
 Linky1Diff[5,6]: - Y1[5] + Y1[6] + EdgeDistance[y1,5,6] = 0
 Linky1Diff[5,7]: - Y1[5] + Y1[7] + EdgeDistance[y1,5,7] = 0
 Linky1Diff[5,8]: - Y1[5] + Y1[8] + EdgeDistance[y1,5,8] = 0
 Linky1Diff[5,9]: - Y1[5] + Y1[9] + EdgeDistance[y1,5,9] = 0
 Linky1Diff[5,10]: - Y1[5] + Y1[10] + EdgeDistance[y1,5,10] = 0
 Linky1Diff[6,0]: Y1[0] - Y1[6] + EdgeDistance[y1,6,0] = 0
 Linky1Diff[6,1]: Y1[1] - Y1[6] + EdgeDistance[y1,6,1] = 0
 Linky1Diff[6,2]: Y1[2] - Y1[6] + EdgeDistance[y1,6,2] = 0
 Linky1Diff[6,3]: Y1[3] - Y1[6] + EdgeDistance[y1,6,3] = 0
 Linky1Diff[6,4]: Y1[4] - Y1[6] + EdgeDistance[y1,6,4] = 0
 Linky1Diff[6,5]: Y1[5] - Y1[6] + EdgeDistance[y1,6,5] = 0
 Linky1Diff[6,6]: EdgeDistance[y1,6,6] = 0
 Linky1Diff[6,7]: - Y1[6] + Y1[7] + EdgeDistance[y1,6,7] = 0
 Linky1Diff[6,8]: - Y1[6] + Y1[8] + EdgeDistance[y1,6,8] = 0
 Linky1Diff[6,9]: - Y1[6] + Y1[9] + EdgeDistance[y1,6,9] = 0
 Linky1Diff[6,10]: - Y1[6] + Y1[10] + EdgeDistance[y1,6,10] = 0
 Linky1Diff[7,0]: Y1[0] - Y1[7] + EdgeDistance[y1,7,0] = 0
 Linky1Diff[7,1]: Y1[1] - Y1[7] + EdgeDistance[y1,7,1] = 0
 Linky1Diff[7,2]: Y1[2] - Y1[7] + EdgeDistance[y1,7,2] = 0
 Linky1Diff[7,3]: Y1[3] - Y1[7] + EdgeDistance[y1,7,3] = 0
 Linky1Diff[7,4]: Y1[4] - Y1[7] + EdgeDistance[y1,7,4] = 0
 Linky1Diff[7,5]: Y1[5] - Y1[7] + EdgeDistance[y1,7,5] = 0
 Linky1Diff[7,6]: Y1[6] - Y1[7] + EdgeDistance[y1,7,6] = 0
 Linky1Diff[7,7]: EdgeDistance[y1,7,7] = 0
 Linky1Diff[7,8]: - Y1[7] + Y1[8] + EdgeDistance[y1,7,8] = 0
 Linky1Diff[7,9]: - Y1[7] + Y1[9] + EdgeDistance[y1,7,9] = 0
 Linky1Diff[7,10]: - Y1[7] + Y1[10] + EdgeDistance[y1,7,10] = 0
 Linky1Diff[8,0]: Y1[0] - Y1[8] + EdgeDistance[y1,8,0] = 0
 Linky1Diff[8,1]: Y1[1] - Y1[8] + EdgeDistance[y1,8,1] = 0
 Linky1Diff[8,2]: Y1[2] - Y1[8] + EdgeDistance[y1,8,2] = 0
 Linky1Diff[8,3]: Y1[3] - Y1[8] + EdgeDistance[y1,8,3] = 0
 Linky1Diff[8,4]: Y1[4] - Y1[8] + EdgeDistance[y1,8,4] = 0
 Linky1Diff[8,5]: Y1[5] - Y1[8] + EdgeDistance[y1,8,5] = 0
 Linky1Diff[8,6]: Y1[6] - Y1[8] + EdgeDistance[y1,8,6] = 0
 Linky1Diff[8,7]: Y1[7] - Y1[8] + EdgeDistance[y1,8,7] = 0
 Linky1Diff[8,8]: EdgeDistance[y1,8,8] = 0
 Linky1Diff[8,9]: - Y1[8] + Y1[9] + EdgeDistance[y1,8,9] = 0
 Linky1Diff[8,10]: - Y1[8] + Y1[10] + EdgeDistance[y1,8,10] = 0
 Linky1Diff[9,0]: Y1[0] - Y1[9] + EdgeDistance[y1,9,0] = 0
 Linky1Diff[9,1]: Y1[1] - Y1[9] + EdgeDistance[y1,9,1] = 0
 Linky1Diff[9,2]: Y1[2] - Y1[9] + EdgeDistance[y1,9,2] = 0
 Linky1Diff[9,3]: Y1[3] - Y1[9] + EdgeDistance[y1,9,3] = 0
 Linky1Diff[9,4]: Y1[4] - Y1[9] + EdgeDistance[y1,9,4] = 0
 Linky1Diff[9,5]: Y1[5] - Y1[9] + EdgeDistance[y1,9,5] = 0
 Linky1Diff[9,6]: Y1[6] - Y1[9] + EdgeDistance[y1,9,6] = 0
 Linky1Diff[9,7]: Y1[7] - Y1[9] + EdgeDistance[y1,9,7] = 0
 Linky1Diff[9,8]: Y1[8] - Y1[9] + EdgeDistance[y1,9,8] = 0
 Linky1Diff[9,9]: EdgeDistance[y1,9,9] = 0
 Linky1Diff[9,10]: - Y1[9] + Y1[10] + EdgeDistance[y1,9,10] = 0
 Linky1Diff[10,0]: Y1[0] - Y1[10] + EdgeDistance[y1,10,0] = 0
 Linky1Diff[10,1]: Y1[1] - Y1[10] + EdgeDistance[y1,10,1] = 0
 Linky1Diff[10,2]: Y1[2] - Y1[10] + EdgeDistance[y1,10,2] = 0
 Linky1Diff[10,3]: Y1[3] - Y1[10] + EdgeDistance[y1,10,3] = 0
 Linky1Diff[10,4]: Y1[4] - Y1[10] + EdgeDistance[y1,10,4] = 0
 Linky1Diff[10,5]: Y1[5] - Y1[10] + EdgeDistance[y1,10,5] = 0
 Linky1Diff[10,6]: Y1[6] - Y1[10] + EdgeDistance[y1,10,6] = 0
 Linky1Diff[10,7]: Y1[7] - Y1[10] + EdgeDistance[y1,10,7] = 0
 Linky1Diff[10,8]: Y1[8] - Y1[10] + EdgeDistance[y1,10,8] = 0
 Linky1Diff[10,9]: Y1[9] - Y1[10] + EdgeDistance[y1,10,9] = 0
 Linky1Diff[10,10]: EdgeDistance[y1,10,10] = 0
 R1034: X0Diff[0,0] = 0
 R1035: - X0[0] + X0[1] + X0Diff[0,1] = 0
 R1036: - X0[0] + X0[2] + X0Diff[0,2] = 0
 R1037: - X0[0] + X0[3] + X0Diff[0,3] = 0
 R1038: - X0[0] + X0[4] + X0Diff[0,4] = 0
 R1039: - X0[0] + X0[5] + X0Diff[0,5] = 0
 R1040: - X0[0] + X0[6] + X0Diff[0,6] = 0
 R1041: - X0[0] + X0[7] + X0Diff[0,7] = 0
 R1042: - X0[0] + X0[8] + X0Diff[0,8] = 0
 R1043: - X0[0] + X0[9] + X0Diff[0,9] = 0
 R1044: - X0[0] + X0[10] + X0Diff[0,10] = 0
 R1045: X0[0] - X0[1] + X0Diff[1,0] = 0
 R1046: X0Diff[1,1] = 0
 R1047: - X0[1] + X0[2] + X0Diff[1,2] = 0
 R1048: - X0[1] + X0[3] + X0Diff[1,3] = 0
 R1049: - X0[1] + X0[4] + X0Diff[1,4] = 0
 R1050: - X0[1] + X0[5] + X0Diff[1,5] = 0
 R1051: - X0[1] + X0[6] + X0Diff[1,6] = 0
 R1052: - X0[1] + X0[7] + X0Diff[1,7] = 0
 R1053: - X0[1] + X0[8] + X0Diff[1,8] = 0
 R1054: - X0[1] + X0[9] + X0Diff[1,9] = 0
 R1055: - X0[1] + X0[10] + X0Diff[1,10] = 0
 R1056: X0[0] - X0[2] + X0Diff[2,0] = 0
 R1057: X0[1] - X0[2] + X0Diff[2,1] = 0
 R1058: X0Diff[2,2] = 0
 R1059: - X0[2] + X0[3] + X0Diff[2,3] = 0
 R1060: - X0[2] + X0[4] + X0Diff[2,4] = 0
 R1061: - X0[2] + X0[5] + X0Diff[2,5] = 0
 R1062: - X0[2] + X0[6] + X0Diff[2,6] = 0
 R1063: - X0[2] + X0[7] + X0Diff[2,7] = 0
 R1064: - X0[2] + X0[8] + X0Diff[2,8] = 0
 R1065: - X0[2] + X0[9] + X0Diff[2,9] = 0
 R1066: - X0[2] + X0[10] + X0Diff[2,10] = 0
 R1067: X0[0] - X0[3] + X0Diff[3,0] = 0
 R1068: X0[1] - X0[3] + X0Diff[3,1] = 0
 R1069: X0[2] - X0[3] + X0Diff[3,2] = 0
 R1070: X0Diff[3,3] = 0
 R1071: - X0[3] + X0[4] + X0Diff[3,4] = 0
 R1072: - X0[3] + X0[5] + X0Diff[3,5] = 0
 R1073: - X0[3] + X0[6] + X0Diff[3,6] = 0
 R1074: - X0[3] + X0[7] + X0Diff[3,7] = 0
 R1075: - X0[3] + X0[8] + X0Diff[3,8] = 0
 R1076: - X0[3] + X0[9] + X0Diff[3,9] = 0
 R1077: - X0[3] + X0[10] + X0Diff[3,10] = 0
 R1078: X0[0] - X0[4] + X0Diff[4,0] = 0
 R1079: X0[1] - X0[4] + X0Diff[4,1] = 0
 R1080: X0[2] - X0[4] + X0Diff[4,2] = 0
 R1081: X0[3] - X0[4] + X0Diff[4,3] = 0
 R1082: X0Diff[4,4] = 0
 R1083: - X0[4] + X0[5] + X0Diff[4,5] = 0
 R1084: - X0[4] + X0[6] + X0Diff[4,6] = 0
 R1085: - X0[4] + X0[7] + X0Diff[4,7] = 0
 R1086: - X0[4] + X0[8] + X0Diff[4,8] = 0
 R1087: - X0[4] + X0[9] + X0Diff[4,9] = 0
 R1088: - X0[4] + X0[10] + X0Diff[4,10] = 0
 R1089: X0[0] - X0[5] + X0Diff[5,0] = 0
 R1090: X0[1] - X0[5] + X0Diff[5,1] = 0
 R1091: X0[2] - X0[5] + X0Diff[5,2] = 0
 R1092: X0[3] - X0[5] + X0Diff[5,3] = 0
 R1093: X0[4] - X0[5] + X0Diff[5,4] = 0
 R1094: X0Diff[5,5] = 0
 R1095: - X0[5] + X0[6] + X0Diff[5,6] = 0
 R1096: - X0[5] + X0[7] + X0Diff[5,7] = 0
 R1097: - X0[5] + X0[8] + X0Diff[5,8] = 0
 R1098: - X0[5] + X0[9] + X0Diff[5,9] = 0
 R1099: - X0[5] + X0[10] + X0Diff[5,10] = 0
 R1100: X0[0] - X0[6] + X0Diff[6,0] = 0
 R1101: X0[1] - X0[6] + X0Diff[6,1] = 0
 R1102: X0[2] - X0[6] + X0Diff[6,2] = 0
 R1103: X0[3] - X0[6] + X0Diff[6,3] = 0
 R1104: X0[4] - X0[6] + X0Diff[6,4] = 0
 R1105: X0[5] - X0[6] + X0Diff[6,5] = 0
 R1106: X0Diff[6,6] = 0
 R1107: - X0[6] + X0[7] + X0Diff[6,7] = 0
 R1108: - X0[6] + X0[8] + X0Diff[6,8] = 0
 R1109: - X0[6] + X0[9] + X0Diff[6,9] = 0
 R1110: - X0[6] + X0[10] + X0Diff[6,10] = 0
 R1111: X0[0] - X0[7] + X0Diff[7,0] = 0
 R1112: X0[1] - X0[7] + X0Diff[7,1] = 0
 R1113: X0[2] - X0[7] + X0Diff[7,2] = 0
 R1114: X0[3] - X0[7] + X0Diff[7,3] = 0
 R1115: X0[4] - X0[7] + X0Diff[7,4] = 0
 R1116: X0[5] - X0[7] + X0Diff[7,5] = 0
 R1117: X0[6] - X0[7] + X0Diff[7,6] = 0
 R1118: X0Diff[7,7] = 0
 R1119: - X0[7] + X0[8] + X0Diff[7,8] = 0
 R1120: - X0[7] + X0[9] + X0Diff[7,9] = 0
 R1121: - X0[7] + X0[10] + X0Diff[7,10] = 0
 R1122: X0[0] - X0[8] + X0Diff[8,0] = 0
 R1123: X0[1] - X0[8] + X0Diff[8,1] = 0
 R1124: X0[2] - X0[8] + X0Diff[8,2] = 0
 R1125: X0[3] - X0[8] + X0Diff[8,3] = 0
 R1126: X0[4] - X0[8] + X0Diff[8,4] = 0
 R1127: X0[5] - X0[8] + X0Diff[8,5] = 0
 R1128: X0[6] - X0[8] + X0Diff[8,6] = 0
 R1129: X0[7] - X0[8] + X0Diff[8,7] = 0
 R1130: X0Diff[8,8] = 0
 R1131: - X0[8] + X0[9] + X0Diff[8,9] = 0
 R1132: - X0[8] + X0[10] + X0Diff[8,10] = 0
 R1133: X0[0] - X0[9] + X0Diff[9,0] = 0
 R1134: X0[1] - X0[9] + X0Diff[9,1] = 0
 R1135: X0[2] - X0[9] + X0Diff[9,2] = 0
 R1136: X0[3] - X0[9] + X0Diff[9,3] = 0
 R1137: X0[4] - X0[9] + X0Diff[9,4] = 0
 R1138: X0[5] - X0[9] + X0Diff[9,5] = 0
 R1139: X0[6] - X0[9] + X0Diff[9,6] = 0
 R1140: X0[7] - X0[9] + X0Diff[9,7] = 0
 R1141: X0[8] - X0[9] + X0Diff[9,8] = 0
 R1142: X0Diff[9,9] = 0
 R1143: - X0[9] + X0[10] + X0Diff[9,10] = 0
 R1144: X0[0] - X0[10] + X0Diff[10,0] = 0
 R1145: X0[1] - X0[10] + X0Diff[10,1] = 0
 R1146: X0[2] - X0[10] + X0Diff[10,2] = 0
 R1147: X0[3] - X0[10] + X0Diff[10,3] = 0
 R1148: X0[4] - X0[10] + X0Diff[10,4] = 0
 R1149: X0[5] - X0[10] + X0Diff[10,5] = 0
 R1150: X0[6] - X0[10] + X0Diff[10,6] = 0
 R1151: X0[7] - X0[10] + X0Diff[10,7] = 0
 R1152: X0[8] - X0[10] + X0Diff[10,8] = 0
 R1153: X0[9] - X0[10] + X0Diff[10,9] = 0
 R1154: X0Diff[10,10] = 0
 R1155: Y0Diff[0,0] = 0
 R1156: - Y0[0] + Y0[1] + Y0Diff[0,1] = 0
 R1157: - Y0[0] + Y0[2] + Y0Diff[0,2] = 0
 R1158: - Y0[0] + Y0[3] + Y0Diff[0,3] = 0
 R1159: - Y0[0] + Y0[4] + Y0Diff[0,4] = 0
 R1160: - Y0[0] + Y0[5] + Y0Diff[0,5] = 0
 R1161: - Y0[0] + Y0[6] + Y0Diff[0,6] = 0
 R1162: - Y0[0] + Y0[7] + Y0Diff[0,7] = 0
 R1163: - Y0[0] + Y0[8] + Y0Diff[0,8] = 0
 R1164: - Y0[0] + Y0[9] + Y0Diff[0,9] = 0
 R1165: - Y0[0] + Y0[10] + Y0Diff[0,10] = 0
 R1166: Y0[0] - Y0[1] + Y0Diff[1,0] = 0
 R1167: Y0Diff[1,1] = 0
 R1168: - Y0[1] + Y0[2] + Y0Diff[1,2] = 0
 R1169: - Y0[1] + Y0[3] + Y0Diff[1,3] = 0
 R1170: - Y0[1] + Y0[4] + Y0Diff[1,4] = 0
 R1171: - Y0[1] + Y0[5] + Y0Diff[1,5] = 0
 R1172: - Y0[1] + Y0[6] + Y0Diff[1,6] = 0
 R1173: - Y0[1] + Y0[7] + Y0Diff[1,7] = 0
 R1174: - Y0[1] + Y0[8] + Y0Diff[1,8] = 0
 R1175: - Y0[1] + Y0[9] + Y0Diff[1,9] = 0
 R1176: - Y0[1] + Y0[10] + Y0Diff[1,10] = 0
 R1177: Y0[0] - Y0[2] + Y0Diff[2,0] = 0
 R1178: Y0[1] - Y0[2] + Y0Diff[2,1] = 0
 R1179: Y0Diff[2,2] = 0
 R1180: - Y0[2] + Y0[3] + Y0Diff[2,3] = 0
 R1181: - Y0[2] + Y0[4] + Y0Diff[2,4] = 0
 R1182: - Y0[2] + Y0[5] + Y0Diff[2,5] = 0
 R1183: - Y0[2] + Y0[6] + Y0Diff[2,6] = 0
 R1184: - Y0[2] + Y0[7] + Y0Diff[2,7] = 0
 R1185: - Y0[2] + Y0[8] + Y0Diff[2,8] = 0
 R1186: - Y0[2] + Y0[9] + Y0Diff[2,9] = 0
 R1187: - Y0[2] + Y0[10] + Y0Diff[2,10] = 0
 R1188: Y0[0] - Y0[3] + Y0Diff[3,0] = 0
 R1189: Y0[1] - Y0[3] + Y0Diff[3,1] = 0
 R1190: Y0[2] - Y0[3] + Y0Diff[3,2] = 0
 R1191: Y0Diff[3,3] = 0
 R1192: - Y0[3] + Y0[4] + Y0Diff[3,4] = 0
 R1193: - Y0[3] + Y0[5] + Y0Diff[3,5] = 0
 R1194: - Y0[3] + Y0[6] + Y0Diff[3,6] = 0
 R1195: - Y0[3] + Y0[7] + Y0Diff[3,7] = 0
 R1196: - Y0[3] + Y0[8] + Y0Diff[3,8] = 0
 R1197: - Y0[3] + Y0[9] + Y0Diff[3,9] = 0
 R1198: - Y0[3] + Y0[10] + Y0Diff[3,10] = 0
 R1199: Y0[0] - Y0[4] + Y0Diff[4,0] = 0
 R1200: Y0[1] - Y0[4] + Y0Diff[4,1] = 0
 R1201: Y0[2] - Y0[4] + Y0Diff[4,2] = 0
 R1202: Y0[3] - Y0[4] + Y0Diff[4,3] = 0
 R1203: Y0Diff[4,4] = 0
 R1204: - Y0[4] + Y0[5] + Y0Diff[4,5] = 0
 R1205: - Y0[4] + Y0[6] + Y0Diff[4,6] = 0
 R1206: - Y0[4] + Y0[7] + Y0Diff[4,7] = 0
 R1207: - Y0[4] + Y0[8] + Y0Diff[4,8] = 0
 R1208: - Y0[4] + Y0[9] + Y0Diff[4,9] = 0
 R1209: - Y0[4] + Y0[10] + Y0Diff[4,10] = 0
 R1210: Y0[0] - Y0[5] + Y0Diff[5,0] = 0
 R1211: Y0[1] - Y0[5] + Y0Diff[5,1] = 0
 R1212: Y0[2] - Y0[5] + Y0Diff[5,2] = 0
 R1213: Y0[3] - Y0[5] + Y0Diff[5,3] = 0
 R1214: Y0[4] - Y0[5] + Y0Diff[5,4] = 0
 R1215: Y0Diff[5,5] = 0
 R1216: - Y0[5] + Y0[6] + Y0Diff[5,6] = 0
 R1217: - Y0[5] + Y0[7] + Y0Diff[5,7] = 0
 R1218: - Y0[5] + Y0[8] + Y0Diff[5,8] = 0
 R1219: - Y0[5] + Y0[9] + Y0Diff[5,9] = 0
 R1220: - Y0[5] + Y0[10] + Y0Diff[5,10] = 0
 R1221: Y0[0] - Y0[6] + Y0Diff[6,0] = 0
 R1222: Y0[1] - Y0[6] + Y0Diff[6,1] = 0
 R1223: Y0[2] - Y0[6] + Y0Diff[6,2] = 0
 R1224: Y0[3] - Y0[6] + Y0Diff[6,3] = 0
 R1225: Y0[4] - Y0[6] + Y0Diff[6,4] = 0
 R1226: Y0[5] - Y0[6] + Y0Diff[6,5] = 0
 R1227: Y0Diff[6,6] = 0
 R1228: - Y0[6] + Y0[7] + Y0Diff[6,7] = 0
 R1229: - Y0[6] + Y0[8] + Y0Diff[6,8] = 0
 R1230: - Y0[6] + Y0[9] + Y0Diff[6,9] = 0
 R1231: - Y0[6] + Y0[10] + Y0Diff[6,10] = 0
 R1232: Y0[0] - Y0[7] + Y0Diff[7,0] = 0
 R1233: Y0[1] - Y0[7] + Y0Diff[7,1] = 0
 R1234: Y0[2] - Y0[7] + Y0Diff[7,2] = 0
 R1235: Y0[3] - Y0[7] + Y0Diff[7,3] = 0
 R1236: Y0[4] - Y0[7] + Y0Diff[7,4] = 0
 R1237: Y0[5] - Y0[7] + Y0Diff[7,5] = 0
 R1238: Y0[6] - Y0[7] + Y0Diff[7,6] = 0
 R1239: Y0Diff[7,7] = 0
 R1240: - Y0[7] + Y0[8] + Y0Diff[7,8] = 0
 R1241: - Y0[7] + Y0[9] + Y0Diff[7,9] = 0
 R1242: - Y0[7] + Y0[10] + Y0Diff[7,10] = 0
 R1243: Y0[0] - Y0[8] + Y0Diff[8,0] = 0
 R1244: Y0[1] - Y0[8] + Y0Diff[8,1] = 0
 R1245: Y0[2] - Y0[8] + Y0Diff[8,2] = 0
 R1246: Y0[3] - Y0[8] + Y0Diff[8,3] = 0
 R1247: Y0[4] - Y0[8] + Y0Diff[8,4] = 0
 R1248: Y0[5] - Y0[8] + Y0Diff[8,5] = 0
 R1249: Y0[6] - Y0[8] + Y0Diff[8,6] = 0
 R1250: Y0[7] - Y0[8] + Y0Diff[8,7] = 0
 R1251: Y0Diff[8,8] = 0
 R1252: - Y0[8] + Y0[9] + Y0Diff[8,9] = 0
 R1253: - Y0[8] + Y0[10] + Y0Diff[8,10] = 0
 R1254: Y0[0] - Y0[9] + Y0Diff[9,0] = 0
 R1255: Y0[1] - Y0[9] + Y0Diff[9,1] = 0
 R1256: Y0[2] - Y0[9] + Y0Diff[9,2] = 0
 R1257: Y0[3] - Y0[9] + Y0Diff[9,3] = 0
 R1258: Y0[4] - Y0[9] + Y0Diff[9,4] = 0
 R1259: Y0[5] - Y0[9] + Y0Diff[9,5] = 0
 R1260: Y0[6] - Y0[9] + Y0Diff[9,6] = 0
 R1261: Y0[7] - Y0[9] + Y0Diff[9,7] = 0
 R1262: Y0[8] - Y0[9] + Y0Diff[9,8] = 0
 R1263: Y0Diff[9,9] = 0
 R1264: - Y0[9] + Y0[10] + Y0Diff[9,10] = 0
 R1265: Y0[0] - Y0[10] + Y0Diff[10,0] = 0
 R1266: Y0[1] - Y0[10] + Y0Diff[10,1] = 0
 R1267: Y0[2] - Y0[10] + Y0Diff[10,2] = 0
 R1268: Y0[3] - Y0[10] + Y0Diff[10,3] = 0
 R1269: Y0[4] - Y0[10] + Y0Diff[10,4] = 0
 R1270: Y0[5] - Y0[10] + Y0Diff[10,5] = 0
 R1271: Y0[6] - Y0[10] + Y0Diff[10,6] = 0
 R1272: Y0[7] - Y0[10] + Y0Diff[10,7] = 0
 R1273: Y0[8] - Y0[10] + Y0Diff[10,8] = 0
 R1274: Y0[9] - Y0[10] + Y0Diff[10,9] = 0
 R1275: Y0Diff[10,10] = 0
 R1276: X1Diff[0,0] = 0
 R1277: - X1[0] + X1[1] + X1Diff[0,1] = 0
 R1278: - X1[0] + X1[2] + X1Diff[0,2] = 0
 R1279: - X1[0] + X1[3] + X1Diff[0,3] = 0
 R1280: - X1[0] + X1[4] + X1Diff[0,4] = 0
 R1281: - X1[0] + X1[5] + X1Diff[0,5] = 0
 R1282: - X1[0] + X1[6] + X1Diff[0,6] = 0
 R1283: - X1[0] + X1[7] + X1Diff[0,7] = 0
 R1284: - X1[0] + X1[8] + X1Diff[0,8] = 0
 R1285: - X1[0] + X1[9] + X1Diff[0,9] = 0
 R1286: - X1[0] + X1[10] + X1Diff[0,10] = 0
 R1287: X1[0] - X1[1] + X1Diff[1,0] = 0
 R1288: X1Diff[1,1] = 0
 R1289: - X1[1] + X1[2] + X1Diff[1,2] = 0
 R1290: - X1[1] + X1[3] + X1Diff[1,3] = 0
 R1291: - X1[1] + X1[4] + X1Diff[1,4] = 0
 R1292: - X1[1] + X1[5] + X1Diff[1,5] = 0
 R1293: - X1[1] + X1[6] + X1Diff[1,6] = 0
 R1294: - X1[1] + X1[7] + X1Diff[1,7] = 0
 R1295: - X1[1] + X1[8] + X1Diff[1,8] = 0
 R1296: - X1[1] + X1[9] + X1Diff[1,9] = 0
 R1297: - X1[1] + X1[10] + X1Diff[1,10] = 0
 R1298: X1[0] - X1[2] + X1Diff[2,0] = 0
 R1299: X1[1] - X1[2] + X1Diff[2,1] = 0
 R1300: X1Diff[2,2] = 0
 R1301: - X1[2] + X1[3] + X1Diff[2,3] = 0
 R1302: - X1[2] + X1[4] + X1Diff[2,4] = 0
 R1303: - X1[2] + X1[5] + X1Diff[2,5] = 0
 R1304: - X1[2] + X1[6] + X1Diff[2,6] = 0
 R1305: - X1[2] + X1[7] + X1Diff[2,7] = 0
 R1306: - X1[2] + X1[8] + X1Diff[2,8] = 0
 R1307: - X1[2] + X1[9] + X1Diff[2,9] = 0
 R1308: - X1[2] + X1[10] + X1Diff[2,10] = 0
 R1309: X1[0] - X1[3] + X1Diff[3,0] = 0
 R1310: X1[1] - X1[3] + X1Diff[3,1] = 0
 R1311: X1[2] - X1[3] + X1Diff[3,2] = 0
 R1312: X1Diff[3,3] = 0
 R1313: - X1[3] + X1[4] + X1Diff[3,4] = 0
 R1314: - X1[3] + X1[5] + X1Diff[3,5] = 0
 R1315: - X1[3] + X1[6] + X1Diff[3,6] = 0
 R1316: - X1[3] + X1[7] + X1Diff[3,7] = 0
 R1317: - X1[3] + X1[8] + X1Diff[3,8] = 0
 R1318: - X1[3] + X1[9] + X1Diff[3,9] = 0
 R1319: - X1[3] + X1[10] + X1Diff[3,10] = 0
 R1320: X1[0] - X1[4] + X1Diff[4,0] = 0
 R1321: X1[1] - X1[4] + X1Diff[4,1] = 0
 R1322: X1[2] - X1[4] + X1Diff[4,2] = 0
 R1323: X1[3] - X1[4] + X1Diff[4,3] = 0
 R1324: X1Diff[4,4] = 0
 R1325: - X1[4] + X1[5] + X1Diff[4,5] = 0
 R1326: - X1[4] + X1[6] + X1Diff[4,6] = 0
 R1327: - X1[4] + X1[7] + X1Diff[4,7] = 0
 R1328: - X1[4] + X1[8] + X1Diff[4,8] = 0
 R1329: - X1[4] + X1[9] + X1Diff[4,9] = 0
 R1330: - X1[4] + X1[10] + X1Diff[4,10] = 0
 R1331: X1[0] - X1[5] + X1Diff[5,0] = 0
 R1332: X1[1] - X1[5] + X1Diff[5,1] = 0
 R1333: X1[2] - X1[5] + X1Diff[5,2] = 0
 R1334: X1[3] - X1[5] + X1Diff[5,3] = 0
 R1335: X1[4] - X1[5] + X1Diff[5,4] = 0
 R1336: X1Diff[5,5] = 0
 R1337: - X1[5] + X1[6] + X1Diff[5,6] = 0
 R1338: - X1[5] + X1[7] + X1Diff[5,7] = 0
 R1339: - X1[5] + X1[8] + X1Diff[5,8] = 0
 R1340: - X1[5] + X1[9] + X1Diff[5,9] = 0
 R1341: - X1[5] + X1[10] + X1Diff[5,10] = 0
 R1342: X1[0] - X1[6] + X1Diff[6,0] = 0
 R1343: X1[1] - X1[6] + X1Diff[6,1] = 0
 R1344: X1[2] - X1[6] + X1Diff[6,2] = 0
 R1345: X1[3] - X1[6] + X1Diff[6,3] = 0
 R1346: X1[4] - X1[6] + X1Diff[6,4] = 0
 R1347: X1[5] - X1[6] + X1Diff[6,5] = 0
 R1348: X1Diff[6,6] = 0
 R1349: - X1[6] + X1[7] + X1Diff[6,7] = 0
 R1350: - X1[6] + X1[8] + X1Diff[6,8] = 0
 R1351: - X1[6] + X1[9] + X1Diff[6,9] = 0
 R1352: - X1[6] + X1[10] + X1Diff[6,10] = 0
 R1353: X1[0] - X1[7] + X1Diff[7,0] = 0
 R1354: X1[1] - X1[7] + X1Diff[7,1] = 0
 R1355: X1[2] - X1[7] + X1Diff[7,2] = 0
 R1356: X1[3] - X1[7] + X1Diff[7,3] = 0
 R1357: X1[4] - X1[7] + X1Diff[7,4] = 0
 R1358: X1[5] - X1[7] + X1Diff[7,5] = 0
 R1359: X1[6] - X1[7] + X1Diff[7,6] = 0
 R1360: X1Diff[7,7] = 0
 R1361: - X1[7] + X1[8] + X1Diff[7,8] = 0
 R1362: - X1[7] + X1[9] + X1Diff[7,9] = 0
 R1363: - X1[7] + X1[10] + X1Diff[7,10] = 0
 R1364: X1[0] - X1[8] + X1Diff[8,0] = 0
 R1365: X1[1] - X1[8] + X1Diff[8,1] = 0
 R1366: X1[2] - X1[8] + X1Diff[8,2] = 0
 R1367: X1[3] - X1[8] + X1Diff[8,3] = 0
 R1368: X1[4] - X1[8] + X1Diff[8,4] = 0
 R1369: X1[5] - X1[8] + X1Diff[8,5] = 0
 R1370: X1[6] - X1[8] + X1Diff[8,6] = 0
 R1371: X1[7] - X1[8] + X1Diff[8,7] = 0
 R1372: X1Diff[8,8] = 0
 R1373: - X1[8] + X1[9] + X1Diff[8,9] = 0
 R1374: - X1[8] + X1[10] + X1Diff[8,10] = 0
 R1375: X1[0] - X1[9] + X1Diff[9,0] = 0
 R1376: X1[1] - X1[9] + X1Diff[9,1] = 0
 R1377: X1[2] - X1[9] + X1Diff[9,2] = 0
 R1378: X1[3] - X1[9] + X1Diff[9,3] = 0
 R1379: X1[4] - X1[9] + X1Diff[9,4] = 0
 R1380: X1[5] - X1[9] + X1Diff[9,5] = 0
 R1381: X1[6] - X1[9] + X1Diff[9,6] = 0
 R1382: X1[7] - X1[9] + X1Diff[9,7] = 0
 R1383: X1[8] - X1[9] + X1Diff[9,8] = 0
 R1384: X1Diff[9,9] = 0
 R1385: - X1[9] + X1[10] + X1Diff[9,10] = 0
 R1386: X1[0] - X1[10] + X1Diff[10,0] = 0
 R1387: X1[1] - X1[10] + X1Diff[10,1] = 0
 R1388: X1[2] - X1[10] + X1Diff[10,2] = 0
 R1389: X1[3] - X1[10] + X1Diff[10,3] = 0
 R1390: X1[4] - X1[10] + X1Diff[10,4] = 0
 R1391: X1[5] - X1[10] + X1Diff[10,5] = 0
 R1392: X1[6] - X1[10] + X1Diff[10,6] = 0
 R1393: X1[7] - X1[10] + X1Diff[10,7] = 0
 R1394: X1[8] - X1[10] + X1Diff[10,8] = 0
 R1395: X1[9] - X1[10] + X1Diff[10,9] = 0
 R1396: X1Diff[10,10] = 0
 R1397: Y1Diff[0,0] = 0
 R1398: - Y1[0] + Y1[1] + Y1Diff[0,1] = 0
 R1399: - Y1[0] + Y1[2] + Y1Diff[0,2] = 0
 R1400: - Y1[0] + Y1[3] + Y1Diff[0,3] = 0
 R1401: - Y1[0] + Y1[4] + Y1Diff[0,4] = 0
 R1402: - Y1[0] + Y1[5] + Y1Diff[0,5] = 0
 R1403: - Y1[0] + Y1[6] + Y1Diff[0,6] = 0
 R1404: - Y1[0] + Y1[7] + Y1Diff[0,7] = 0
 R1405: - Y1[0] + Y1[8] + Y1Diff[0,8] = 0
 R1406: - Y1[0] + Y1[9] + Y1Diff[0,9] = 0
 R1407: - Y1[0] + Y1[10] + Y1Diff[0,10] = 0
 R1408: Y1[0] - Y1[1] + Y1Diff[1,0] = 0
 R1409: Y1Diff[1,1] = 0
 R1410: - Y1[1] + Y1[2] + Y1Diff[1,2] = 0
 R1411: - Y1[1] + Y1[3] + Y1Diff[1,3] = 0
 R1412: - Y1[1] + Y1[4] + Y1Diff[1,4] = 0
 R1413: - Y1[1] + Y1[5] + Y1Diff[1,5] = 0
 R1414: - Y1[1] + Y1[6] + Y1Diff[1,6] = 0
 R1415: - Y1[1] + Y1[7] + Y1Diff[1,7] = 0
 R1416: - Y1[1] + Y1[8] + Y1Diff[1,8] = 0
 R1417: - Y1[1] + Y1[9] + Y1Diff[1,9] = 0
 R1418: - Y1[1] + Y1[10] + Y1Diff[1,10] = 0
 R1419: Y1[0] - Y1[2] + Y1Diff[2,0] = 0
 R1420: Y1[1] - Y1[2] + Y1Diff[2,1] = 0
 R1421: Y1Diff[2,2] = 0
 R1422: - Y1[2] + Y1[3] + Y1Diff[2,3] = 0
 R1423: - Y1[2] + Y1[4] + Y1Diff[2,4] = 0
 R1424: - Y1[2] + Y1[5] + Y1Diff[2,5] = 0
 R1425: - Y1[2] + Y1[6] + Y1Diff[2,6] = 0
 R1426: - Y1[2] + Y1[7] + Y1Diff[2,7] = 0
 R1427: - Y1[2] + Y1[8] + Y1Diff[2,8] = 0
 R1428: - Y1[2] + Y1[9] + Y1Diff[2,9] = 0
 R1429: - Y1[2] + Y1[10] + Y1Diff[2,10] = 0
 R1430: Y1[0] - Y1[3] + Y1Diff[3,0] = 0
 R1431: Y1[1] - Y1[3] + Y1Diff[3,1] = 0
 R1432: Y1[2] - Y1[3] + Y1Diff[3,2] = 0
 R1433: Y1Diff[3,3] = 0
 R1434: - Y1[3] + Y1[4] + Y1Diff[3,4] = 0
 R1435: - Y1[3] + Y1[5] + Y1Diff[3,5] = 0
 R1436: - Y1[3] + Y1[6] + Y1Diff[3,6] = 0
 R1437: - Y1[3] + Y1[7] + Y1Diff[3,7] = 0
 R1438: - Y1[3] + Y1[8] + Y1Diff[3,8] = 0
 R1439: - Y1[3] + Y1[9] + Y1Diff[3,9] = 0
 R1440: - Y1[3] + Y1[10] + Y1Diff[3,10] = 0
 R1441: Y1[0] - Y1[4] + Y1Diff[4,0] = 0
 R1442: Y1[1] - Y1[4] + Y1Diff[4,1] = 0
 R1443: Y1[2] - Y1[4] + Y1Diff[4,2] = 0
 R1444: Y1[3] - Y1[4] + Y1Diff[4,3] = 0
 R1445: Y1Diff[4,4] = 0
 R1446: - Y1[4] + Y1[5] + Y1Diff[4,5] = 0
 R1447: - Y1[4] + Y1[6] + Y1Diff[4,6] = 0
 R1448: - Y1[4] + Y1[7] + Y1Diff[4,7] = 0
 R1449: - Y1[4] + Y1[8] + Y1Diff[4,8] = 0
 R1450: - Y1[4] + Y1[9] + Y1Diff[4,9] = 0
 R1451: - Y1[4] + Y1[10] + Y1Diff[4,10] = 0
 R1452: Y1[0] - Y1[5] + Y1Diff[5,0] = 0
 R1453: Y1[1] - Y1[5] + Y1Diff[5,1] = 0
 R1454: Y1[2] - Y1[5] + Y1Diff[5,2] = 0
 R1455: Y1[3] - Y1[5] + Y1Diff[5,3] = 0
 R1456: Y1[4] - Y1[5] + Y1Diff[5,4] = 0
 R1457: Y1Diff[5,5] = 0
 R1458: - Y1[5] + Y1[6] + Y1Diff[5,6] = 0
 R1459: - Y1[5] + Y1[7] + Y1Diff[5,7] = 0
 R1460: - Y1[5] + Y1[8] + Y1Diff[5,8] = 0
 R1461: - Y1[5] + Y1[9] + Y1Diff[5,9] = 0
 R1462: - Y1[5] + Y1[10] + Y1Diff[5,10] = 0
 R1463: Y1[0] - Y1[6] + Y1Diff[6,0] = 0
 R1464: Y1[1] - Y1[6] + Y1Diff[6,1] = 0
 R1465: Y1[2] - Y1[6] + Y1Diff[6,2] = 0
 R1466: Y1[3] - Y1[6] + Y1Diff[6,3] = 0
 R1467: Y1[4] - Y1[6] + Y1Diff[6,4] = 0
 R1468: Y1[5] - Y1[6] + Y1Diff[6,5] = 0
 R1469: Y1Diff[6,6] = 0
 R1470: - Y1[6] + Y1[7] + Y1Diff[6,7] = 0
 R1471: - Y1[6] + Y1[8] + Y1Diff[6,8] = 0
 R1472: - Y1[6] + Y1[9] + Y1Diff[6,9] = 0
 R1473: - Y1[6] + Y1[10] + Y1Diff[6,10] = 0
 R1474: Y1[0] - Y1[7] + Y1Diff[7,0] = 0
 R1475: Y1[1] - Y1[7] + Y1Diff[7,1] = 0
 R1476: Y1[2] - Y1[7] + Y1Diff[7,2] = 0
 R1477: Y1[3] - Y1[7] + Y1Diff[7,3] = 0
 R1478: Y1[4] - Y1[7] + Y1Diff[7,4] = 0
 R1479: Y1[5] - Y1[7] + Y1Diff[7,5] = 0
 R1480: Y1[6] - Y1[7] + Y1Diff[7,6] = 0
 R1481: Y1Diff[7,7] = 0
 R1482: - Y1[7] + Y1[8] + Y1Diff[7,8] = 0
 R1483: - Y1[7] + Y1[9] + Y1Diff[7,9] = 0
 R1484: - Y1[7] + Y1[10] + Y1Diff[7,10] = 0
 R1485: Y1[0] - Y1[8] + Y1Diff[8,0] = 0
 R1486: Y1[1] - Y1[8] + Y1Diff[8,1] = 0
 R1487: Y1[2] - Y1[8] + Y1Diff[8,2] = 0
 R1488: Y1[3] - Y1[8] + Y1Diff[8,3] = 0
 R1489: Y1[4] - Y1[8] + Y1Diff[8,4] = 0
 R1490: Y1[5] - Y1[8] + Y1Diff[8,5] = 0
 R1491: Y1[6] - Y1[8] + Y1Diff[8,6] = 0
 R1492: Y1[7] - Y1[8] + Y1Diff[8,7] = 0
 R1493: Y1Diff[8,8] = 0
 R1494: - Y1[8] + Y1[9] + Y1Diff[8,9] = 0
 R1495: - Y1[8] + Y1[10] + Y1Diff[8,10] = 0
 R1496: Y1[0] - Y1[9] + Y1Diff[9,0] = 0
 R1497: Y1[1] - Y1[9] + Y1Diff[9,1] = 0
 R1498: Y1[2] - Y1[9] + Y1Diff[9,2] = 0
 R1499: Y1[3] - Y1[9] + Y1Diff[9,3] = 0
 R1500: Y1[4] - Y1[9] + Y1Diff[9,4] = 0
 R1501: Y1[5] - Y1[9] + Y1Diff[9,5] = 0
 R1502: Y1[6] - Y1[9] + Y1Diff[9,6] = 0
 R1503: Y1[7] - Y1[9] + Y1Diff[9,7] = 0
 R1504: Y1[8] - Y1[9] + Y1Diff[9,8] = 0
 R1505: Y1Diff[9,9] = 0
 R1506: - Y1[9] + Y1[10] + Y1Diff[9,10] = 0
 R1507: Y1[0] - Y1[10] + Y1Diff[10,0] = 0
 R1508: Y1[1] - Y1[10] + Y1Diff[10,1] = 0
 R1509: Y1[2] - Y1[10] + Y1Diff[10,2] = 0
 R1510: Y1[3] - Y1[10] + Y1Diff[10,3] = 0
 R1511: Y1[4] - Y1[10] + Y1Diff[10,4] = 0
 R1512: Y1[5] - Y1[10] + Y1Diff[10,5] = 0
 R1513: Y1[6] - Y1[10] + Y1Diff[10,6] = 0
 R1514: Y1[7] - Y1[10] + Y1Diff[10,7] = 0
 R1515: Y1[8] - Y1[10] + Y1Diff[10,8] = 0
 R1516: Y1[9] - Y1[10] + Y1Diff[10,9] = 0
 R1517: Y1Diff[10,10] = 0
 PreventOvertOptimization: X0GroupCount + Y0GroupCount + X1GroupCount
   + Y1GroupCount >= 14
 PreserveOverlap[0,1]: Overlap[0,1] = 0
 PreserveOverlap[0,2]: Overlap[0,2] = 0
 PreserveOverlap[0,3]: Overlap[0,3] = 0
 PreserveOverlap[0,4]: Overlap[0,4] = 1
 PreserveOverlap[0,5]: Overlap[0,5] = 0
 PreserveOverlap[0,6]: Overlap[0,6] = 0
 PreserveOverlap[0,7]: Overlap[0,7] = 0
 PreserveOverlap[0,8]: Overlap[0,8] = 0
 PreserveOverlap[0,9]: Overlap[0,9] = 0
 PreserveOverlap[0,10]: Overlap[0,10] = 0
 PreserveOverlap[1,0]: Overlap[1,0] = 0
 PreserveOverlap[1,2]: Overlap[1,2] = 0
 PreserveOverlap[1,3]: Overlap[1,3] = 0
 PreserveOverlap[1,4]: Overlap[1,4] = 1
 PreserveOverlap[1,5]: Overlap[1,5] = 0
 PreserveOverlap[1,6]: Overlap[1,6] = 0
 PreserveOverlap[1,7]: Overlap[1,7] = 0
 PreserveOverlap[1,8]: Overlap[1,8] = 0
 PreserveOverlap[1,9]: Overlap[1,9] = 0
 PreserveOverlap[1,10]: Overlap[1,10] = 0
 PreserveOverlap[2,0]: Overlap[2,0] = 0
 PreserveOverlap[2,1]: Overlap[2,1] = 0
 PreserveOverlap[2,3]: Overlap[2,3] = 0
 PreserveOverlap[2,4]: Overlap[2,4] = 1
 PreserveOverlap[2,5]: Overlap[2,5] = 0
 PreserveOverlap[2,6]: Overlap[2,6] = 0
 PreserveOverlap[2,7]: Overlap[2,7] = 0
 PreserveOverlap[2,8]: Overlap[2,8] = 0
 PreserveOverlap[2,9]: Overlap[2,9] = 0
 PreserveOverlap[2,10]: Overlap[2,10] = 0
 PreserveOverlap[3,0]: Overlap[3,0] = 0
 PreserveOverlap[3,1]: Overlap[3,1] = 0
 PreserveOverlap[3,2]: Overlap[3,2] = 0
 PreserveOverlap[3,4]: Overlap[3,4] = 1
 PreserveOverlap[3,5]: Overlap[3,5] = 0
 PreserveOverlap[3,6]: Overlap[3,6] = 0
 PreserveOverlap[3,7]: Overlap[3,7] = 0
 PreserveOverlap[3,8]: Overlap[3,8] = 0
 PreserveOverlap[3,9]: Overlap[3,9] = 0
 PreserveOverlap[3,10]: Overlap[3,10] = 0
 PreserveOverlap[4,0]: Overlap[4,0] = 1
 PreserveOverlap[4,1]: Overlap[4,1] = 1
 PreserveOverlap[4,2]: Overlap[4,2] = 1
 PreserveOverlap[4,3]: Overlap[4,3] = 1
 PreserveOverlap[4,5]: Overlap[4,5] = 0
 PreserveOverlap[4,6]: Overlap[4,6] = 0
 PreserveOverlap[4,7]: Overlap[4,7] = 0
 PreserveOverlap[4,8]: Overlap[4,8] = 0
 PreserveOverlap[4,9]: Overlap[4,9] = 0
 PreserveOverlap[4,10]: Overlap[4,10] = 0
 PreserveOverlap[5,0]: Overlap[5,0] = 0
 PreserveOverlap[5,1]: Overlap[5,1] = 0
 PreserveOverlap[5,2]: Overlap[5,2] = 0
 PreserveOverlap[5,3]: Overlap[5,3] = 0
 PreserveOverlap[5,4]: Overlap[5,4] = 0
 PreserveOverlap[5,6]: Overlap[5,6] = 0
 PreserveOverlap[5,7]: Overlap[5,7] = 0
 PreserveOverlap[5,8]: Overlap[5,8] = 0
 PreserveOverlap[5,9]: Overlap[5,9] = 0
 PreserveOverlap[5,10]: Overlap[5,10] = 1
 PreserveOverlap[6,0]: Overlap[6,0] = 0
 PreserveOverlap[6,1]: Overlap[6,1] = 0
 PreserveOverlap[6,2]: Overlap[6,2] = 0
 PreserveOverlap[6,3]: Overlap[6,3] = 0
 PreserveOverlap[6,4]: Overlap[6,4] = 0
 PreserveOverlap[6,5]: Overlap[6,5] = 0
 PreserveOverlap[6,7]: Overlap[6,7] = 0
 PreserveOverlap[6,8]: Overlap[6,8] = 0
 PreserveOverlap[6,9]: Overlap[6,9] = 0
 PreserveOverlap[6,10]: Overlap[6,10] = 1
 PreserveOverlap[7,0]: Overlap[7,0] = 0
 PreserveOverlap[7,1]: Overlap[7,1] = 0
 PreserveOverlap[7,2]: Overlap[7,2] = 0
 PreserveOverlap[7,3]: Overlap[7,3] = 0
 PreserveOverlap[7,4]: Overlap[7,4] = 0
 PreserveOverlap[7,5]: Overlap[7,5] = 0
 PreserveOverlap[7,6]: Overlap[7,6] = 0
 PreserveOverlap[7,8]: Overlap[7,8] = 0
 PreserveOverlap[7,9]: Overlap[7,9] = 0
 PreserveOverlap[7,10]: Overlap[7,10] = 1
 PreserveOverlap[8,0]: Overlap[8,0] = 0
 PreserveOverlap[8,1]: Overlap[8,1] = 0
 PreserveOverlap[8,2]: Overlap[8,2] = 0
 PreserveOverlap[8,3]: Overlap[8,3] = 0
 PreserveOverlap[8,4]: Overlap[8,4] = 0
 PreserveOverlap[8,5]: Overlap[8,5] = 0
 PreserveOverlap[8,6]: Overlap[8,6] = 0
 PreserveOverlap[8,7]: Overlap[8,7] = 0
 PreserveOverlap[8,9]: Overlap[8,9] = 0
 PreserveOverlap[8,10]: Overlap[8,10] = 0
 PreserveOverlap[9,0]: Overlap[9,0] = 0
 PreserveOverlap[9,1]: Overlap[9,1] = 0
 PreserveOverlap[9,2]: Overlap[9,2] = 0
 PreserveOverlap[9,3]: Overlap[9,3] = 0
 PreserveOverlap[9,4]: Overlap[9,4] = 0
 PreserveOverlap[9,5]: Overlap[9,5] = 0
 PreserveOverlap[9,6]: Overlap[9,6] = 0
 PreserveOverlap[9,7]: Overlap[9,7] = 0
 PreserveOverlap[9,8]: Overlap[9,8] = 0
 PreserveOverlap[9,10]: Overlap[9,10] = 0
 PreserveOverlap[10,0]: Overlap[10,0] = 0
 PreserveOverlap[10,1]: Overlap[10,1] = 0
 PreserveOverlap[10,2]: Overlap[10,2] = 0
 PreserveOverlap[10,3]: Overlap[10,3] = 0
 PreserveOverlap[10,4]: Overlap[10,4] = 0
 PreserveOverlap[10,5]: Overlap[10,5] = 1
 PreserveOverlap[10,6]: Overlap[10,6] = 1
 PreserveOverlap[10,7]: Overlap[10,7] = 1
 PreserveOverlap[10,8]: Overlap[10,8] = 0
 PreserveOverlap[10,9]: Overlap[10,9] = 0
 DontAlignOverlappingX00,4: X0[0] - X0[4] >= 1
 DontAlignOverlappingX10,4: X1[0] - X1[4] <= -1
 DontAlignOverlappingY0,4: Y0[0] - Y0[4] >= 1
 DontAlignOverlappingY10,4: Y1[0] - Y1[4] <= -1
 DontAlignOverlappingX01,4: X0[1] - X0[4] >= 1
 DontAlignOverlappingX11,4: X1[1] - X1[4] <= -1
 DontAlignOverlappingY1,4: Y0[1] - Y0[4] >= 1
 DontAlignOverlappingY11,4: Y1[1] - Y1[4] <= -1
 DontAlignOverlappingX02,4: X0[2] - X0[4] >= 1
 DontAlignOverlappingX12,4: X1[2] - X1[4] <= -1
 DontAlignOverlappingY2,4: Y0[2] - Y0[4] >= 1
 DontAlignOverlappingY12,4: Y1[2] - Y1[4] <= -1
 DontAlignOverlappingX03,4: X0[3] - X0[4] >= 1
 DontAlignOverlappingX13,4: X1[3] - X1[4] <= -1
 DontAlignOverlappingY3,4: Y0[3] - Y0[4] >= 1
 DontAlignOverlappingY13,4: Y1[3] - Y1[4] <= -1
 DontAlignOverlappingX04,0: - X0[0] + X0[4] <= -1
 DontAlignOverlappingX14,0: - X1[0] + X1[4] >= 1
 DontAlignOverlappingY4,0: - Y0[0] + Y0[4] <= -1
 DontAlignOverlappingY14,0: - Y1[0] + Y1[4] >= 1
 DontAlignOverlappingX04,1: - X0[1] + X0[4] <= -1
 DontAlignOverlappingX14,1: - X1[1] + X1[4] >= 1
 DontAlignOverlappingY4,1: - Y0[1] + Y0[4] <= -1
 DontAlignOverlappingY14,1: - Y1[1] + Y1[4] >= 1
 DontAlignOverlappingX04,2: - X0[2] + X0[4] <= -1
 DontAlignOverlappingX14,2: - X1[2] + X1[4] >= 1
 DontAlignOverlappingY4,2: - Y0[2] + Y0[4] <= -1
 DontAlignOverlappingY14,2: - Y1[2] + Y1[4] >= 1
 DontAlignOverlappingX04,3: - X0[3] + X0[4] <= -1
 DontAlignOverlappingX14,3: - X1[3] + X1[4] >= 1
 DontAlignOverlappingY4,3: - Y0[3] + Y0[4] <= -1
 DontAlignOverlappingY14,3: - Y1[3] + Y1[4] >= 1
 DontAlignOverlappingX05,10: X0[5] - X0[10] >= 1
 DontAlignOverlappingX15,10: X1[5] - X1[10] <= -1
 DontAlignOverlappingY5,10: Y0[5] - Y0[10] >= 1
 DontAlignOverlappingY15,10: Y1[5] - Y1[10] <= -1
 DontAlignOverlappingX06,10: X0[6] - X0[10] >= 1
 DontAlignOverlappingX16,10: X1[6] - X1[10] <= -1
 DontAlignOverlappingY6,10: Y0[6] - Y0[10] >= 1
 DontAlignOverlappingY16,10: Y1[6] - Y1[10] <= -1
 DontAlignOverlappingX07,10: X0[7] - X0[10] >= 1
 DontAlignOverlappingX17,10: X1[7] - X1[10] <= -1
 DontAlignOverlappingY7,10: Y0[7] - Y0[10] >= 1
 DontAlignOverlappingY17,10: Y1[7] - Y1[10] <= -1
 DontAlignOverlappingX010,5: - X0[5] + X0[10] <= -1
 DontAlignOverlappingX110,5: - X1[5] + X1[10] >= 1
 DontAlignOverlappingY10,5: - Y0[5] + Y0[10] <= -1
 DontAlignOverlappingY110,5: - Y1[5] + Y1[10] >= 1
 DontAlignOverlappingX010,6: - X0[6] + X0[10] <= -1
 DontAlignOverlappingX110,6: - X1[6] + X1[10] >= 1
 DontAlignOverlappingY10,6: - Y0[6] + Y0[10] <= -1
 DontAlignOverlappingY110,6: - Y1[6] + Y1[10] >= 1
 DontAlignOverlappingX010,7: - X0[7] + X0[10] <= -1
 DontAlignOverlappingX110,7: - X1[7] + X1[10] >= 1
 DontAlignOverlappingY10,7: - Y0[7] + Y0[10] <= -1
 DontAlignOverlappingY110,7: - Y1[7] + Y1[10] >= 1
 LinkRightMargin: MaxX1 + MaxX1 = 100
 LinkMarginDiffLooseAbs1: - LeftMargin + MaxX1 + MarginDiffLooseAbs >= 0
 LinkMarginDiffLooseAbs2: LeftMargin - MaxX1 + MarginDiffLooseAbs >= 0
 ColumnStartEndSanity[0]: - StartColumn[0] + EndColumn[0] >= 0
 ColumnStartEndSanity[1]: - StartColumn[1] + EndColumn[1] >= 0
 ColumnStartEndSanity[2]: - StartColumn[2] + EndColumn[2] >= 0
 ColumnStartEndSanity[3]: - StartColumn[3] + EndColumn[3] >= 0
 ColumnStartEndSanity[4]: - StartColumn[4] + EndColumn[4] >= 0
 ColumnStartEndSanity[5]: - StartColumn[5] + EndColumn[5] >= 0
 ColumnStartEndSanity[6]: - StartColumn[6] + EndColumn[6] >= 0
 ColumnStartEndSanity[7]: - StartColumn[7] + EndColumn[7] >= 0
 ColumnStartEndSanity[8]: - StartColumn[8] + EndColumn[8] >= 0
 ColumnStartEndSanity[9]: - StartColumn[9] + EndColumn[9] >= 0
 ColumnStartEndSanity[10]: - StartColumn[10] + EndColumn[10] >= 0
 LinkOtherElementStartsInNextColumn1[0,1]: - X0LessThan[0,1]
   + OtherElementStartsInNextColumn[0,1] <= 0
 LinkOtherElementStartsInNextColumn1[0,2]: - X0LessThan[0,2]
   + OtherElementStartsInNextColumn[0,2] <= 0
 LinkOtherElementStartsInNextColumn1[0,3]: - X0LessThan[0,3]
   + OtherElementStartsInNextColumn[0,3] <= 0
 LinkOtherElementStartsInNextColumn1[0,4]: - X0LessThan[0,4]
   + OtherElementStartsInNextColumn[0,4] <= 0
 LinkOtherElementStartsInNextColumn1[0,5]: - X0LessThan[0,5]
   + OtherElementStartsInNextColumn[0,5] <= 0
 LinkOtherElementStartsInNextColumn1[0,6]: - X0LessThan[0,6]
   + OtherElementStartsInNextColumn[0,6] <= 0
 LinkOtherElementStartsInNextColumn1[0,7]: - X0LessThan[0,7]
   + OtherElementStartsInNextColumn[0,7] <= 0
 LinkOtherElementStartsInNextColumn1[0,8]: - X0LessThan[0,8]
   + OtherElementStartsInNextColumn[0,8] <= 0
 LinkOtherElementStartsInNextColumn1[0,9]: - X0LessThan[0,9]
   + OtherElementStartsInNextColumn[0,9] <= 0
 LinkOtherElementStartsInNextColumn1[0,10]: - X0LessThan[0,10]
   + OtherElementStartsInNextColumn[0,10] <= 0
 LinkOtherElementStartsInNextColumn1[1,0]: - X0LessThan[1,0]
   + OtherElementStartsInNextColumn[1,0] <= 0
 LinkOtherElementStartsInNextColumn1[1,2]: - X0LessThan[1,2]
   + OtherElementStartsInNextColumn[1,2] <= 0
 LinkOtherElementStartsInNextColumn1[1,3]: - X0LessThan[1,3]
   + OtherElementStartsInNextColumn[1,3] <= 0
 LinkOtherElementStartsInNextColumn1[1,4]: - X0LessThan[1,4]
   + OtherElementStartsInNextColumn[1,4] <= 0
 LinkOtherElementStartsInNextColumn1[1,5]: - X0LessThan[1,5]
   + OtherElementStartsInNextColumn[1,5] <= 0
 LinkOtherElementStartsInNextColumn1[1,6]: - X0LessThan[1,6]
   + OtherElementStartsInNextColumn[1,6] <= 0
 LinkOtherElementStartsInNextColumn1[1,7]: - X0LessThan[1,7]
   + OtherElementStartsInNextColumn[1,7] <= 0
 LinkOtherElementStartsInNextColumn1[1,8]: - X0LessThan[1,8]
   + OtherElementStartsInNextColumn[1,8] <= 0
 LinkOtherElementStartsInNextColumn1[1,9]: - X0LessThan[1,9]
   + OtherElementStartsInNextColumn[1,9] <= 0
 LinkOtherElementStartsInNextColumn1[1,10]: - X0LessThan[1,10]
   + OtherElementStartsInNextColumn[1,10] <= 0
 LinkOtherElementStartsInNextColumn1[2,0]: - X0LessThan[2,0]
   + OtherElementStartsInNextColumn[2,0] <= 0
 LinkOtherElementStartsInNextColumn1[2,1]: - X0LessThan[2,1]
   + OtherElementStartsInNextColumn[2,1] <= 0
 LinkOtherElementStartsInNextColumn1[2,3]: - X0LessThan[2,3]
   + OtherElementStartsInNextColumn[2,3] <= 0
 LinkOtherElementStartsInNextColumn1[2,4]: - X0LessThan[2,4]
   + OtherElementStartsInNextColumn[2,4] <= 0
 LinkOtherElementStartsInNextColumn1[2,5]: - X0LessThan[2,5]
   + OtherElementStartsInNextColumn[2,5] <= 0
 LinkOtherElementStartsInNextColumn1[2,6]: - X0LessThan[2,6]
   + OtherElementStartsInNextColumn[2,6] <= 0
 LinkOtherElementStartsInNextColumn1[2,7]: - X0LessThan[2,7]
   + OtherElementStartsInNextColumn[2,7] <= 0
 LinkOtherElementStartsInNextColumn1[2,8]: - X0LessThan[2,8]
   + OtherElementStartsInNextColumn[2,8] <= 0
 LinkOtherElementStartsInNextColumn1[2,9]: - X0LessThan[2,9]
   + OtherElementStartsInNextColumn[2,9] <= 0
 LinkOtherElementStartsInNextColumn1[2,10]: - X0LessThan[2,10]
   + OtherElementStartsInNextColumn[2,10] <= 0
 LinkOtherElementStartsInNextColumn1[3,0]: - X0LessThan[3,0]
   + OtherElementStartsInNextColumn[3,0] <= 0
 LinkOtherElementStartsInNextColumn1[3,1]: - X0LessThan[3,1]
   + OtherElementStartsInNextColumn[3,1] <= 0
 LinkOtherElementStartsInNextColumn1[3,2]: - X0LessThan[3,2]
   + OtherElementStartsInNextColumn[3,2] <= 0
 LinkOtherElementStartsInNextColumn1[3,4]: - X0LessThan[3,4]
   + OtherElementStartsInNextColumn[3,4] <= 0
 LinkOtherElementStartsInNextColumn1[3,5]: - X0LessThan[3,5]
   + OtherElementStartsInNextColumn[3,5] <= 0
 LinkOtherElementStartsInNextColumn1[3,6]: - X0LessThan[3,6]
   + OtherElementStartsInNextColumn[3,6] <= 0
 LinkOtherElementStartsInNextColumn1[3,7]: - X0LessThan[3,7]
   + OtherElementStartsInNextColumn[3,7] <= 0
 LinkOtherElementStartsInNextColumn1[3,8]: - X0LessThan[3,8]
   + OtherElementStartsInNextColumn[3,8] <= 0
 LinkOtherElementStartsInNextColumn1[3,9]: - X0LessThan[3,9]
   + OtherElementStartsInNextColumn[3,9] <= 0
 LinkOtherElementStartsInNextColumn1[3,10]: - X0LessThan[3,10]
   + OtherElementStartsInNextColumn[3,10] <= 0
 LinkOtherElementStartsInNextColumn1[4,0]: - X0LessThan[4,0]
   + OtherElementStartsInNextColumn[4,0] <= 0
 LinkOtherElementStartsInNextColumn1[4,1]: - X0LessThan[4,1]
   + OtherElementStartsInNextColumn[4,1] <= 0
 LinkOtherElementStartsInNextColumn1[4,2]: - X0LessThan[4,2]
   + OtherElementStartsInNextColumn[4,2] <= 0
 LinkOtherElementStartsInNextColumn1[4,3]: - X0LessThan[4,3]
   + OtherElementStartsInNextColumn[4,3] <= 0
 LinkOtherElementStartsInNextColumn1[4,5]: - X0LessThan[4,5]
   + OtherElementStartsInNextColumn[4,5] <= 0
 LinkOtherElementStartsInNextColumn1[4,6]: - X0LessThan[4,6]
   + OtherElementStartsInNextColumn[4,6] <= 0
 LinkOtherElementStartsInNextColumn1[4,7]: - X0LessThan[4,7]
   + OtherElementStartsInNextColumn[4,7] <= 0
 LinkOtherElementStartsInNextColumn1[4,8]: - X0LessThan[4,8]
   + OtherElementStartsInNextColumn[4,8] <= 0
 LinkOtherElementStartsInNextColumn1[4,9]: - X0LessThan[4,9]
   + OtherElementStartsInNextColumn[4,9] <= 0
 LinkOtherElementStartsInNextColumn1[4,10]: - X0LessThan[4,10]
   + OtherElementStartsInNextColumn[4,10] <= 0
 LinkOtherElementStartsInNextColumn1[5,0]: - X0LessThan[5,0]
   + OtherElementStartsInNextColumn[5,0] <= 0
 LinkOtherElementStartsInNextColumn1[5,1]: - X0LessThan[5,1]
   + OtherElementStartsInNextColumn[5,1] <= 0
 LinkOtherElementStartsInNextColumn1[5,2]: - X0LessThan[5,2]
   + OtherElementStartsInNextColumn[5,2] <= 0
 LinkOtherElementStartsInNextColumn1[5,3]: - X0LessThan[5,3]
   + OtherElementStartsInNextColumn[5,3] <= 0
 LinkOtherElementStartsInNextColumn1[5,4]: - X0LessThan[5,4]
   + OtherElementStartsInNextColumn[5,4] <= 0
 LinkOtherElementStartsInNextColumn1[5,6]: - X0LessThan[5,6]
   + OtherElementStartsInNextColumn[5,6] <= 0
 LinkOtherElementStartsInNextColumn1[5,7]: - X0LessThan[5,7]
   + OtherElementStartsInNextColumn[5,7] <= 0
 LinkOtherElementStartsInNextColumn1[5,8]: - X0LessThan[5,8]
   + OtherElementStartsInNextColumn[5,8] <= 0
 LinkOtherElementStartsInNextColumn1[5,9]: - X0LessThan[5,9]
   + OtherElementStartsInNextColumn[5,9] <= 0
 LinkOtherElementStartsInNextColumn1[5,10]: - X0LessThan[5,10]
   + OtherElementStartsInNextColumn[5,10] <= 0
 LinkOtherElementStartsInNextColumn1[6,0]: - X0LessThan[6,0]
   + OtherElementStartsInNextColumn[6,0] <= 0
 LinkOtherElementStartsInNextColumn1[6,1]: - X0LessThan[6,1]
   + OtherElementStartsInNextColumn[6,1] <= 0
 LinkOtherElementStartsInNextColumn1[6,2]: - X0LessThan[6,2]
   + OtherElementStartsInNextColumn[6,2] <= 0
 LinkOtherElementStartsInNextColumn1[6,3]: - X0LessThan[6,3]
   + OtherElementStartsInNextColumn[6,3] <= 0
 LinkOtherElementStartsInNextColumn1[6,4]: - X0LessThan[6,4]
   + OtherElementStartsInNextColumn[6,4] <= 0
 LinkOtherElementStartsInNextColumn1[6,5]: - X0LessThan[6,5]
   + OtherElementStartsInNextColumn[6,5] <= 0
 LinkOtherElementStartsInNextColumn1[6,7]: - X0LessThan[6,7]
   + OtherElementStartsInNextColumn[6,7] <= 0
 LinkOtherElementStartsInNextColumn1[6,8]: - X0LessThan[6,8]
   + OtherElementStartsInNextColumn[6,8] <= 0
 LinkOtherElementStartsInNextColumn1[6,9]: - X0LessThan[6,9]
   + OtherElementStartsInNextColumn[6,9] <= 0
 LinkOtherElementStartsInNextColumn1[6,10]: - X0LessThan[6,10]
   + OtherElementStartsInNextColumn[6,10] <= 0
 LinkOtherElementStartsInNextColumn1[7,0]: - X0LessThan[7,0]
   + OtherElementStartsInNextColumn[7,0] <= 0
 LinkOtherElementStartsInNextColumn1[7,1]: - X0LessThan[7,1]
   + OtherElementStartsInNextColumn[7,1] <= 0
 LinkOtherElementStartsInNextColumn1[7,2]: - X0LessThan[7,2]
   + OtherElementStartsInNextColumn[7,2] <= 0
 LinkOtherElementStartsInNextColumn1[7,3]: - X0LessThan[7,3]
   + OtherElementStartsInNextColumn[7,3] <= 0
 LinkOtherElementStartsInNextColumn1[7,4]: - X0LessThan[7,4]
   + OtherElementStartsInNextColumn[7,4] <= 0
 LinkOtherElementStartsInNextColumn1[7,5]: - X0LessThan[7,5]
   + OtherElementStartsInNextColumn[7,5] <= 0
 LinkOtherElementStartsInNextColumn1[7,6]: - X0LessThan[7,6]
   + OtherElementStartsInNextColumn[7,6] <= 0
 LinkOtherElementStartsInNextColumn1[7,8]: - X0LessThan[7,8]
   + OtherElementStartsInNextColumn[7,8] <= 0
 LinkOtherElementStartsInNextColumn1[7,9]: - X0LessThan[7,9]
   + OtherElementStartsInNextColumn[7,9] <= 0
 LinkOtherElementStartsInNextColumn1[7,10]: - X0LessThan[7,10]
   + OtherElementStartsInNextColumn[7,10] <= 0
 LinkOtherElementStartsInNextColumn1[8,0]: - X0LessThan[8,0]
   + OtherElementStartsInNextColumn[8,0] <= 0
 LinkOtherElementStartsInNextColumn1[8,1]: - X0LessThan[8,1]
   + OtherElementStartsInNextColumn[8,1] <= 0
 LinkOtherElementStartsInNextColumn1[8,2]: - X0LessThan[8,2]
   + OtherElementStartsInNextColumn[8,2] <= 0
 LinkOtherElementStartsInNextColumn1[8,3]: - X0LessThan[8,3]
   + OtherElementStartsInNextColumn[8,3] <= 0
 LinkOtherElementStartsInNextColumn1[8,4]: - X0LessThan[8,4]
   + OtherElementStartsInNextColumn[8,4] <= 0
 LinkOtherElementStartsInNextColumn1[8,5]: - X0LessThan[8,5]
   + OtherElementStartsInNextColumn[8,5] <= 0
 LinkOtherElementStartsInNextColumn1[8,6]: - X0LessThan[8,6]
   + OtherElementStartsInNextColumn[8,6] <= 0
 LinkOtherElementStartsInNextColumn1[8,7]: - X0LessThan[8,7]
   + OtherElementStartsInNextColumn[8,7] <= 0
 LinkOtherElementStartsInNextColumn1[8,9]: - X0LessThan[8,9]
   + OtherElementStartsInNextColumn[8,9] <= 0
 LinkOtherElementStartsInNextColumn1[8,10]: - X0LessThan[8,10]
   + OtherElementStartsInNextColumn[8,10] <= 0
 LinkOtherElementStartsInNextColumn1[9,0]: - X0LessThan[9,0]
   + OtherElementStartsInNextColumn[9,0] <= 0
 LinkOtherElementStartsInNextColumn1[9,1]: - X0LessThan[9,1]
   + OtherElementStartsInNextColumn[9,1] <= 0
 LinkOtherElementStartsInNextColumn1[9,2]: - X0LessThan[9,2]
   + OtherElementStartsInNextColumn[9,2] <= 0
 LinkOtherElementStartsInNextColumn1[9,3]: - X0LessThan[9,3]
   + OtherElementStartsInNextColumn[9,3] <= 0
 LinkOtherElementStartsInNextColumn1[9,4]: - X0LessThan[9,4]
   + OtherElementStartsInNextColumn[9,4] <= 0
 LinkOtherElementStartsInNextColumn1[9,5]: - X0LessThan[9,5]
   + OtherElementStartsInNextColumn[9,5] <= 0
 LinkOtherElementStartsInNextColumn1[9,6]: - X0LessThan[9,6]
   + OtherElementStartsInNextColumn[9,6] <= 0
 LinkOtherElementStartsInNextColumn1[9,7]: - X0LessThan[9,7]
   + OtherElementStartsInNextColumn[9,7] <= 0
 LinkOtherElementStartsInNextColumn1[9,8]: - X0LessThan[9,8]
   + OtherElementStartsInNextColumn[9,8] <= 0
 LinkOtherElementStartsInNextColumn1[9,10]: - X0LessThan[9,10]
   + OtherElementStartsInNextColumn[9,10] <= 0
 LinkOtherElementStartsInNextColumn1[10,0]: - X0LessThan[10,0]
   + OtherElementStartsInNextColumn[10,0] <= 0
 LinkOtherElementStartsInNextColumn1[10,1]: - X0LessThan[10,1]
   + OtherElementStartsInNextColumn[10,1] <= 0
 LinkOtherElementStartsInNextColumn1[10,2]: - X0LessThan[10,2]
   + OtherElementStartsInNextColumn[10,2] <= 0
 LinkOtherElementStartsInNextColumn1[10,3]: - X0LessThan[10,3]
   + OtherElementStartsInNextColumn[10,3] <= 0
 LinkOtherElementStartsInNextColumn1[10,4]: - X0LessThan[10,4]
   + OtherElementStartsInNextColumn[10,4] <= 0
 LinkOtherElementStartsInNextColumn1[10,5]: - X0LessThan[10,5]
   + OtherElementStartsInNextColumn[10,5] <= 0
 LinkOtherElementStartsInNextColumn1[10,6]: - X0LessThan[10,6]
   + OtherElementStartsInNextColumn[10,6] <= 0
 LinkOtherElementStartsInNextColumn1[10,7]: - X0LessThan[10,7]
   + OtherElementStartsInNextColumn[10,7] <= 0
 LinkOtherElementStartsInNextColumn1[10,8]: - X0LessThan[10,8]
   + OtherElementStartsInNextColumn[10,8] <= 0
 LinkOtherElementStartsInNextColumn1[10,9]: - X0LessThan[10,9]
   + OtherElementStartsInNextColumn[10,9] <= 0
 LinkOtherElementEndsInNextColumn1[0,1]: - X1LessThan[0,1]
   + OtherElementEndsInNextColumn[0,1] <= 0
 LinkOtherElementEndsInNextColumn1[0,2]: - X1LessThan[0,2]
   + OtherElementEndsInNextColumn[0,2] <= 0
 LinkOtherElementEndsInNextColumn1[0,3]: - X1LessThan[0,3]
   + OtherElementEndsInNextColumn[0,3] <= 0
 LinkOtherElementEndsInNextColumn1[0,4]: - X1LessThan[0,4]
   + OtherElementEndsInNextColumn[0,4] <= 0
 LinkOtherElementEndsInNextColumn1[0,5]: - X1LessThan[0,5]
   + OtherElementEndsInNextColumn[0,5] <= 0
 LinkOtherElementEndsInNextColumn1[0,6]: - X1LessThan[0,6]
   + OtherElementEndsInNextColumn[0,6] <= 0
 LinkOtherElementEndsInNextColumn1[0,7]: - X1LessThan[0,7]
   + OtherElementEndsInNextColumn[0,7] <= 0
 LinkOtherElementEndsInNextColumn1[0,8]: - X1LessThan[0,8]
   + OtherElementEndsInNextColumn[0,8] <= 0
 LinkOtherElementEndsInNextColumn1[0,9]: - X1LessThan[0,9]
   + OtherElementEndsInNextColumn[0,9] <= 0
 LinkOtherElementEndsInNextColumn1[0,10]: - X1LessThan[0,10]
   + OtherElementEndsInNextColumn[0,10] <= 0
 LinkOtherElementEndsInNextColumn1[1,0]: - X1LessThan[1,0]
   + OtherElementEndsInNextColumn[1,0] <= 0
 LinkOtherElementEndsInNextColumn1[1,2]: - X1LessThan[1,2]
   + OtherElementEndsInNextColumn[1,2] <= 0
 LinkOtherElementEndsInNextColumn1[1,3]: - X1LessThan[1,3]
   + OtherElementEndsInNextColumn[1,3] <= 0
 LinkOtherElementEndsInNextColumn1[1,4]: - X1LessThan[1,4]
   + OtherElementEndsInNextColumn[1,4] <= 0
 LinkOtherElementEndsInNextColumn1[1,5]: - X1LessThan[1,5]
   + OtherElementEndsInNextColumn[1,5] <= 0
 LinkOtherElementEndsInNextColumn1[1,6]: - X1LessThan[1,6]
   + OtherElementEndsInNextColumn[1,6] <= 0
 LinkOtherElementEndsInNextColumn1[1,7]: - X1LessThan[1,7]
   + OtherElementEndsInNextColumn[1,7] <= 0
 LinkOtherElementEndsInNextColumn1[1,8]: - X1LessThan[1,8]
   + OtherElementEndsInNextColumn[1,8] <= 0
 LinkOtherElementEndsInNextColumn1[1,9]: - X1LessThan[1,9]
   + OtherElementEndsInNextColumn[1,9] <= 0
 LinkOtherElementEndsInNextColumn1[1,10]: - X1LessThan[1,10]
   + OtherElementEndsInNextColumn[1,10] <= 0
 LinkOtherElementEndsInNextColumn1[2,0]: - X1LessThan[2,0]
   + OtherElementEndsInNextColumn[2,0] <= 0
 LinkOtherElementEndsInNextColumn1[2,1]: - X1LessThan[2,1]
   + OtherElementEndsInNextColumn[2,1] <= 0
 LinkOtherElementEndsInNextColumn1[2,3]: - X1LessThan[2,3]
   + OtherElementEndsInNextColumn[2,3] <= 0
 LinkOtherElementEndsInNextColumn1[2,4]: - X1LessThan[2,4]
   + OtherElementEndsInNextColumn[2,4] <= 0
 LinkOtherElementEndsInNextColumn1[2,5]: - X1LessThan[2,5]
   + OtherElementEndsInNextColumn[2,5] <= 0
 LinkOtherElementEndsInNextColumn1[2,6]: - X1LessThan[2,6]
   + OtherElementEndsInNextColumn[2,6] <= 0
 LinkOtherElementEndsInNextColumn1[2,7]: - X1LessThan[2,7]
   + OtherElementEndsInNextColumn[2,7] <= 0
 LinkOtherElementEndsInNextColumn1[2,8]: - X1LessThan[2,8]
   + OtherElementEndsInNextColumn[2,8] <= 0
 LinkOtherElementEndsInNextColumn1[2,9]: - X1LessThan[2,9]
   + OtherElementEndsInNextColumn[2,9] <= 0
 LinkOtherElementEndsInNextColumn1[2,10]: - X1LessThan[2,10]
   + OtherElementEndsInNextColumn[2,10] <= 0
 LinkOtherElementEndsInNextColumn1[3,0]: - X1LessThan[3,0]
   + OtherElementEndsInNextColumn[3,0] <= 0
 LinkOtherElementEndsInNextColumn1[3,1]: - X1LessThan[3,1]
   + OtherElementEndsInNextColumn[3,1] <= 0
 LinkOtherElementEndsInNextColumn1[3,2]: - X1LessThan[3,2]
   + OtherElementEndsInNextColumn[3,2] <= 0
 LinkOtherElementEndsInNextColumn1[3,4]: - X1LessThan[3,4]
   + OtherElementEndsInNextColumn[3,4] <= 0
 LinkOtherElementEndsInNextColumn1[3,5]: - X1LessThan[3,5]
   + OtherElementEndsInNextColumn[3,5] <= 0
 LinkOtherElementEndsInNextColumn1[3,6]: - X1LessThan[3,6]
   + OtherElementEndsInNextColumn[3,6] <= 0
 LinkOtherElementEndsInNextColumn1[3,7]: - X1LessThan[3,7]
   + OtherElementEndsInNextColumn[3,7] <= 0
 LinkOtherElementEndsInNextColumn1[3,8]: - X1LessThan[3,8]
   + OtherElementEndsInNextColumn[3,8] <= 0
 LinkOtherElementEndsInNextColumn1[3,9]: - X1LessThan[3,9]
   + OtherElementEndsInNextColumn[3,9] <= 0
 LinkOtherElementEndsInNextColumn1[3,10]: - X1LessThan[3,10]
   + OtherElementEndsInNextColumn[3,10] <= 0
 LinkOtherElementEndsInNextColumn1[4,0]: - X1LessThan[4,0]
   + OtherElementEndsInNextColumn[4,0] <= 0
 LinkOtherElementEndsInNextColumn1[4,1]: - X1LessThan[4,1]
   + OtherElementEndsInNextColumn[4,1] <= 0
 LinkOtherElementEndsInNextColumn1[4,2]: - X1LessThan[4,2]
   + OtherElementEndsInNextColumn[4,2] <= 0
 LinkOtherElementEndsInNextColumn1[4,3]: - X1LessThan[4,3]
   + OtherElementEndsInNextColumn[4,3] <= 0
 LinkOtherElementEndsInNextColumn1[4,5]: - X1LessThan[4,5]
   + OtherElementEndsInNextColumn[4,5] <= 0
 LinkOtherElementEndsInNextColumn1[4,6]: - X1LessThan[4,6]
   + OtherElementEndsInNextColumn[4,6] <= 0
 LinkOtherElementEndsInNextColumn1[4,7]: - X1LessThan[4,7]
   + OtherElementEndsInNextColumn[4,7] <= 0
 LinkOtherElementEndsInNextColumn1[4,8]: - X1LessThan[4,8]
   + OtherElementEndsInNextColumn[4,8] <= 0
 LinkOtherElementEndsInNextColumn1[4,9]: - X1LessThan[4,9]
   + OtherElementEndsInNextColumn[4,9] <= 0
 LinkOtherElementEndsInNextColumn1[4,10]: - X1LessThan[4,10]
   + OtherElementEndsInNextColumn[4,10] <= 0
 LinkOtherElementEndsInNextColumn1[5,0]: - X1LessThan[5,0]
   + OtherElementEndsInNextColumn[5,0] <= 0
 LinkOtherElementEndsInNextColumn1[5,1]: - X1LessThan[5,1]
   + OtherElementEndsInNextColumn[5,1] <= 0
 LinkOtherElementEndsInNextColumn1[5,2]: - X1LessThan[5,2]
   + OtherElementEndsInNextColumn[5,2] <= 0
 LinkOtherElementEndsInNextColumn1[5,3]: - X1LessThan[5,3]
   + OtherElementEndsInNextColumn[5,3] <= 0
 LinkOtherElementEndsInNextColumn1[5,4]: - X1LessThan[5,4]
   + OtherElementEndsInNextColumn[5,4] <= 0
 LinkOtherElementEndsInNextColumn1[5,6]: - X1LessThan[5,6]
   + OtherElementEndsInNextColumn[5,6] <= 0
 LinkOtherElementEndsInNextColumn1[5,7]: - X1LessThan[5,7]
   + OtherElementEndsInNextColumn[5,7] <= 0
 LinkOtherElementEndsInNextColumn1[5,8]: - X1LessThan[5,8]
   + OtherElementEndsInNextColumn[5,8] <= 0
 LinkOtherElementEndsInNextColumn1[5,9]: - X1LessThan[5,9]
   + OtherElementEndsInNextColumn[5,9] <= 0
 LinkOtherElementEndsInNextColumn1[5,10]: - X1LessThan[5,10]
   + OtherElementEndsInNextColumn[5,10] <= 0
 LinkOtherElementEndsInNextColumn1[6,0]: - X1LessThan[6,0]
   + OtherElementEndsInNextColumn[6,0] <= 0
 LinkOtherElementEndsInNextColumn1[6,1]: - X1LessThan[6,1]
   + OtherElementEndsInNextColumn[6,1] <= 0
 LinkOtherElementEndsInNextColumn1[6,2]: - X1LessThan[6,2]
   + OtherElementEndsInNextColumn[6,2] <= 0
 LinkOtherElementEndsInNextColumn1[6,3]: - X1LessThan[6,3]
   + OtherElementEndsInNextColumn[6,3] <= 0
 LinkOtherElementEndsInNextColumn1[6,4]: - X1LessThan[6,4]
   + OtherElementEndsInNextColumn[6,4] <= 0
 LinkOtherElementEndsInNextColumn1[6,5]: - X1LessThan[6,5]
   + OtherElementEndsInNextColumn[6,5] <= 0
 LinkOtherElementEndsInNextColumn1[6,7]: - X1LessThan[6,7]
   + OtherElementEndsInNextColumn[6,7] <= 0
 LinkOtherElementEndsInNextColumn1[6,8]: - X1LessThan[6,8]
   + OtherElementEndsInNextColumn[6,8] <= 0
 LinkOtherElementEndsInNextColumn1[6,9]: - X1LessThan[6,9]
   + OtherElementEndsInNextColumn[6,9] <= 0
 LinkOtherElementEndsInNextColumn1[6,10]: - X1LessThan[6,10]
   + OtherElementEndsInNextColumn[6,10] <= 0
 LinkOtherElementEndsInNextColumn1[7,0]: - X1LessThan[7,0]
   + OtherElementEndsInNextColumn[7,0] <= 0
 LinkOtherElementEndsInNextColumn1[7,1]: - X1LessThan[7,1]
   + OtherElementEndsInNextColumn[7,1] <= 0
 LinkOtherElementEndsInNextColumn1[7,2]: - X1LessThan[7,2]
   + OtherElementEndsInNextColumn[7,2] <= 0
 LinkOtherElementEndsInNextColumn1[7,3]: - X1LessThan[7,3]
   + OtherElementEndsInNextColumn[7,3] <= 0
 LinkOtherElementEndsInNextColumn1[7,4]: - X1LessThan[7,4]
   + OtherElementEndsInNextColumn[7,4] <= 0
 LinkOtherElementEndsInNextColumn1[7,5]: - X1LessThan[7,5]
   + OtherElementEndsInNextColumn[7,5] <= 0
 LinkOtherElementEndsInNextColumn1[7,6]: - X1LessThan[7,6]
   + OtherElementEndsInNextColumn[7,6] <= 0
 LinkOtherElementEndsInNextColumn1[7,8]: - X1LessThan[7,8]
   + OtherElementEndsInNextColumn[7,8] <= 0
 LinkOtherElementEndsInNextColumn1[7,9]: - X1LessThan[7,9]
   + OtherElementEndsInNextColumn[7,9] <= 0
 LinkOtherElementEndsInNextColumn1[7,10]: - X1LessThan[7,10]
   + OtherElementEndsInNextColumn[7,10] <= 0
 LinkOtherElementEndsInNextColumn1[8,0]: - X1LessThan[8,0]
   + OtherElementEndsInNextColumn[8,0] <= 0
 LinkOtherElementEndsInNextColumn1[8,1]: - X1LessThan[8,1]
   + OtherElementEndsInNextColumn[8,1] <= 0
 LinkOtherElementEndsInNextColumn1[8,2]: - X1LessThan[8,2]
   + OtherElementEndsInNextColumn[8,2] <= 0
 LinkOtherElementEndsInNextColumn1[8,3]: - X1LessThan[8,3]
   + OtherElementEndsInNextColumn[8,3] <= 0
 LinkOtherElementEndsInNextColumn1[8,4]: - X1LessThan[8,4]
   + OtherElementEndsInNextColumn[8,4] <= 0
 LinkOtherElementEndsInNextColumn1[8,5]: - X1LessThan[8,5]
   + OtherElementEndsInNextColumn[8,5] <= 0
 LinkOtherElementEndsInNextColumn1[8,6]: - X1LessThan[8,6]
   + OtherElementEndsInNextColumn[8,6] <= 0
 LinkOtherElementEndsInNextColumn1[8,7]: - X1LessThan[8,7]
   + OtherElementEndsInNextColumn[8,7] <= 0
 LinkOtherElementEndsInNextColumn1[8,9]: - X1LessThan[8,9]
   + OtherElementEndsInNextColumn[8,9] <= 0
 LinkOtherElementEndsInNextColumn1[8,10]: - X1LessThan[8,10]
   + OtherElementEndsInNextColumn[8,10] <= 0
 LinkOtherElementEndsInNextColumn1[9,0]: - X1LessThan[9,0]
   + OtherElementEndsInNextColumn[9,0] <= 0
 LinkOtherElementEndsInNextColumn1[9,1]: - X1LessThan[9,1]
   + OtherElementEndsInNextColumn[9,1] <= 0
 LinkOtherElementEndsInNextColumn1[9,2]: - X1LessThan[9,2]
   + OtherElementEndsInNextColumn[9,2] <= 0
 LinkOtherElementEndsInNextColumn1[9,3]: - X1LessThan[9,3]
   + OtherElementEndsInNextColumn[9,3] <= 0
 LinkOtherElementEndsInNextColumn1[9,4]: - X1LessThan[9,4]
   + OtherElementEndsInNextColumn[9,4] <= 0
 LinkOtherElementEndsInNextColumn1[9,5]: - X1LessThan[9,5]
   + OtherElementEndsInNextColumn[9,5] <= 0
 LinkOtherElementEndsInNextColumn1[9,6]: - X1LessThan[9,6]
   + OtherElementEndsInNextColumn[9,6] <= 0
 LinkOtherElementEndsInNextColumn1[9,7]: - X1LessThan[9,7]
   + OtherElementEndsInNextColumn[9,7] <= 0
 LinkOtherElementEndsInNextColumn1[9,8]: - X1LessThan[9,8]
   + OtherElementEndsInNextColumn[9,8] <= 0
 LinkOtherElementEndsInNextColumn1[9,10]: - X1LessThan[9,10]
   + OtherElementEndsInNextColumn[9,10] <= 0
 LinkOtherElementEndsInNextColumn1[10,0]: - X1LessThan[10,0]
   + OtherElementEndsInNextColumn[10,0] <= 0
 LinkOtherElementEndsInNextColumn1[10,1]: - X1LessThan[10,1]
   + OtherElementEndsInNextColumn[10,1] <= 0
 LinkOtherElementEndsInNextColumn1[10,2]: - X1LessThan[10,2]
   + OtherElementEndsInNextColumn[10,2] <= 0
 LinkOtherElementEndsInNextColumn1[10,3]: - X1LessThan[10,3]
   + OtherElementEndsInNextColumn[10,3] <= 0
 LinkOtherElementEndsInNextColumn1[10,4]: - X1LessThan[10,4]
   + OtherElementEndsInNextColumn[10,4] <= 0
 LinkOtherElementEndsInNextColumn1[10,5]: - X1LessThan[10,5]
   + OtherElementEndsInNextColumn[10,5] <= 0
 LinkOtherElementEndsInNextColumn1[10,6]: - X1LessThan[10,6]
   + OtherElementEndsInNextColumn[10,6] <= 0
 LinkOtherElementEndsInNextColumn1[10,7]: - X1LessThan[10,7]
   + OtherElementEndsInNextColumn[10,7] <= 0
 LinkOtherElementEndsInNextColumn1[10,8]: - X1LessThan[10,8]
   + OtherElementEndsInNextColumn[10,8] <= 0
 LinkOtherElementEndsInNextColumn1[10,9]: - X1LessThan[10,9]
   + OtherElementEndsInNextColumn[10,9] <= 0
 R1919: ColumnCount >= 3
 LinkAbove1[0,1]: Above[0,1] = 1 -> Y1[0] - Y0[1] <= -0
 LinkAbove1[0,2]: Above[0,2] = 1 -> Y1[0] - Y0[2] <= -0
 LinkAbove1[0,3]: Above[0,3] = 1 -> Y1[0] - Y0[3] <= -0
 LinkAbove1[0,4]: Above[0,4] = 1 -> Y1[0] - Y0[4] <= -0
 LinkAbove1[0,5]: Above[0,5] = 1 -> Y1[0] - Y0[5] <= -0
 LinkAbove1[0,6]: Above[0,6] = 1 -> Y1[0] - Y0[6] <= -0
 LinkAbove1[0,7]: Above[0,7] = 1 -> Y1[0] - Y0[7] <= -0
 LinkAbove1[0,8]: Above[0,8] = 1 -> Y1[0] - Y0[8] <= -0
 LinkAbove1[0,9]: Above[0,9] = 1 -> Y1[0] - Y0[9] <= -0
 LinkAbove1[0,10]: Above[0,10] = 1 -> Y1[0] - Y0[10] <= -0
 LinkAbove1[1,0]: Above[1,0] = 1 -> Y1[1] - Y0[0] <= -0
 LinkAbove1[1,2]: Above[1,2] = 1 -> Y1[1] - Y0[2] <= -0
 LinkAbove1[1,3]: Above[1,3] = 1 -> Y1[1] - Y0[3] <= -0
 LinkAbove1[1,4]: Above[1,4] = 1 -> Y1[1] - Y0[4] <= -0
 LinkAbove1[1,5]: Above[1,5] = 1 -> Y1[1] - Y0[5] <= -0
 LinkAbove1[1,6]: Above[1,6] = 1 -> Y1[1] - Y0[6] <= -0
 LinkAbove1[1,7]: Above[1,7] = 1 -> Y1[1] - Y0[7] <= -0
 LinkAbove1[1,8]: Above[1,8] = 1 -> Y1[1] - Y0[8] <= -0
 LinkAbove1[1,9]: Above[1,9] = 1 -> Y1[1] - Y0[9] <= -0
 LinkAbove1[1,10]: Above[1,10] = 1 -> Y1[1] - Y0[10] <= -0
 LinkAbove1[2,0]: Above[2,0] = 1 -> Y1[2] - Y0[0] <= -0
 LinkAbove1[2,1]: Above[2,1] = 1 -> Y1[2] - Y0[1] <= -0
 LinkAbove1[2,3]: Above[2,3] = 1 -> Y1[2] - Y0[3] <= -0
 LinkAbove1[2,4]: Above[2,4] = 1 -> Y1[2] - Y0[4] <= -0
 LinkAbove1[2,5]: Above[2,5] = 1 -> Y1[2] - Y0[5] <= -0
 LinkAbove1[2,6]: Above[2,6] = 1 -> Y1[2] - Y0[6] <= -0
 LinkAbove1[2,7]: Above[2,7] = 1 -> Y1[2] - Y0[7] <= -0
 LinkAbove1[2,8]: Above[2,8] = 1 -> Y1[2] - Y0[8] <= -0
 LinkAbove1[2,9]: Above[2,9] = 1 -> Y1[2] - Y0[9] <= -0
 LinkAbove1[2,10]: Above[2,10] = 1 -> Y1[2] - Y0[10] <= -0
 LinkAbove1[3,0]: Above[3,0] = 1 -> Y1[3] - Y0[0] <= -0
 LinkAbove1[3,1]: Above[3,1] = 1 -> Y1[3] - Y0[1] <= -0
 LinkAbove1[3,2]: Above[3,2] = 1 -> Y1[3] - Y0[2] <= -0
 LinkAbove1[3,4]: Above[3,4] = 1 -> Y1[3] - Y0[4] <= -0
 LinkAbove1[3,5]: Above[3,5] = 1 -> Y1[3] - Y0[5] <= -0
 LinkAbove1[3,6]: Above[3,6] = 1 -> Y1[3] - Y0[6] <= -0
 LinkAbove1[3,7]: Above[3,7] = 1 -> Y1[3] - Y0[7] <= -0
 LinkAbove1[3,8]: Above[3,8] = 1 -> Y1[3] - Y0[8] <= -0
 LinkAbove1[3,9]: Above[3,9] = 1 -> Y1[3] - Y0[9] <= -0
 LinkAbove1[3,10]: Above[3,10] = 1 -> Y1[3] - Y0[10] <= -0
 LinkAbove1[4,0]: Above[4,0] = 1 -> Y1[4] - Y0[0] <= -0
 LinkAbove1[4,1]: Above[4,1] = 1 -> Y1[4] - Y0[1] <= -0
 LinkAbove1[4,2]: Above[4,2] = 1 -> Y1[4] - Y0[2] <= -0
 LinkAbove1[4,3]: Above[4,3] = 1 -> Y1[4] - Y0[3] <= -0
 LinkAbove1[4,5]: Above[4,5] = 1 -> Y1[4] - Y0[5] <= -0
 LinkAbove1[4,6]: Above[4,6] = 1 -> Y1[4] - Y0[6] <= -0
 LinkAbove1[4,7]: Above[4,7] = 1 -> Y1[4] - Y0[7] <= -0
 LinkAbove1[4,8]: Above[4,8] = 1 -> Y1[4] - Y0[8] <= -0
 LinkAbove1[4,9]: Above[4,9] = 1 -> Y1[4] - Y0[9] <= -0
 LinkAbove1[4,10]: Above[4,10] = 1 -> Y1[4] - Y0[10] <= -0
 LinkAbove1[5,0]: Above[5,0] = 1 -> Y1[5] - Y0[0] <= -0
 LinkAbove1[5,1]: Above[5,1] = 1 -> Y1[5] - Y0[1] <= -0
 LinkAbove1[5,2]: Above[5,2] = 1 -> Y1[5] - Y0[2] <= -0
 LinkAbove1[5,3]: Above[5,3] = 1 -> Y1[5] - Y0[3] <= -0
 LinkAbove1[5,4]: Above[5,4] = 1 -> Y1[5] - Y0[4] <= -0
 LinkAbove1[5,6]: Above[5,6] = 1 -> Y1[5] - Y0[6] <= -0
 LinkAbove1[5,7]: Above[5,7] = 1 -> Y1[5] - Y0[7] <= -0
 LinkAbove1[5,8]: Above[5,8] = 1 -> Y1[5] - Y0[8] <= -0
 LinkAbove1[5,9]: Above[5,9] = 1 -> Y1[5] - Y0[9] <= -0
 LinkAbove1[5,10]: Above[5,10] = 1 -> Y1[5] - Y0[10] <= -0
 LinkAbove1[6,0]: Above[6,0] = 1 -> Y1[6] - Y0[0] <= -0
 LinkAbove1[6,1]: Above[6,1] = 1 -> Y1[6] - Y0[1] <= -0
 LinkAbove1[6,2]: Above[6,2] = 1 -> Y1[6] - Y0[2] <= -0
 LinkAbove1[6,3]: Above[6,3] = 1 -> Y1[6] - Y0[3] <= -0
 LinkAbove1[6,4]: Above[6,4] = 1 -> Y1[6] - Y0[4] <= -0
 LinkAbove1[6,5]: Above[6,5] = 1 -> Y1[6] - Y0[5] <= -0
 LinkAbove1[6,7]: Above[6,7] = 1 -> Y1[6] - Y0[7] <= -0
 LinkAbove1[6,8]: Above[6,8] = 1 -> Y1[6] - Y0[8] <= -0
 LinkAbove1[6,9]: Above[6,9] = 1 -> Y1[6] - Y0[9] <= -0
 LinkAbove1[6,10]: Above[6,10] = 1 -> Y1[6] - Y0[10] <= -0
 LinkAbove1[7,0]: Above[7,0] = 1 -> Y1[7] - Y0[0] <= -0
 LinkAbove1[7,1]: Above[7,1] = 1 -> Y1[7] - Y0[1] <= -0
 LinkAbove1[7,2]: Above[7,2] = 1 -> Y1[7] - Y0[2] <= -0
 LinkAbove1[7,3]: Above[7,3] = 1 -> Y1[7] - Y0[3] <= -0
 LinkAbove1[7,4]: Above[7,4] = 1 -> Y1[7] - Y0[4] <= -0
 LinkAbove1[7,5]: Above[7,5] = 1 -> Y1[7] - Y0[5] <= -0
 LinkAbove1[7,6]: Above[7,6] = 1 -> Y1[7] - Y0[6] <= -0
 LinkAbove1[7,8]: Above[7,8] = 1 -> Y1[7] - Y0[8] <= -0
 LinkAbove1[7,9]: Above[7,9] = 1 -> Y1[7] - Y0[9] <= -0
 LinkAbove1[7,10]: Above[7,10] = 1 -> Y1[7] - Y0[10] <= -0
 LinkAbove1[8,0]: Above[8,0] = 1 -> Y1[8] - Y0[0] <= -0
 LinkAbove1[8,1]: Above[8,1] = 1 -> Y1[8] - Y0[1] <= -0
 LinkAbove1[8,2]: Above[8,2] = 1 -> Y1[8] - Y0[2] <= -0
 LinkAbove1[8,3]: Above[8,3] = 1 -> Y1[8] - Y0[3] <= -0
 LinkAbove1[8,4]: Above[8,4] = 1 -> Y1[8] - Y0[4] <= -0
 LinkAbove1[8,5]: Above[8,5] = 1 -> Y1[8] - Y0[5] <= -0
 LinkAbove1[8,6]: Above[8,6] = 1 -> Y1[8] - Y0[6] <= -0
 LinkAbove1[8,7]: Above[8,7] = 1 -> Y1[8] - Y0[7] <= -0
 LinkAbove1[8,9]: Above[8,9] = 1 -> Y1[8] - Y0[9] <= -0
 LinkAbove1[8,10]: Above[8,10] = 1 -> Y1[8] - Y0[10] <= -0
 LinkAbove1[9,0]: Above[9,0] = 1 -> Y1[9] - Y0[0] <= -0
 LinkAbove1[9,1]: Above[9,1] = 1 -> Y1[9] - Y0[1] <= -0
 LinkAbove1[9,2]: Above[9,2] = 1 -> Y1[9] - Y0[2] <= -0
 LinkAbove1[9,3]: Above[9,3] = 1 -> Y1[9] - Y0[3] <= -0
 LinkAbove1[9,4]: Above[9,4] = 1 -> Y1[9] - Y0[4] <= -0
 LinkAbove1[9,5]: Above[9,5] = 1 -> Y1[9] - Y0[5] <= -0
 LinkAbove1[9,6]: Above[9,6] = 1 -> Y1[9] - Y0[6] <= -0
 LinkAbove1[9,7]: Above[9,7] = 1 -> Y1[9] - Y0[7] <= -0
 LinkAbove1[9,8]: Above[9,8] = 1 -> Y1[9] - Y0[8] <= -0
 LinkAbove1[9,10]: Above[9,10] = 1 -> Y1[9] - Y0[10] <= -0
 LinkAbove1[10,0]: Above[10,0] = 1 -> Y1[10] - Y0[0] <= -0
 LinkAbove1[10,1]: Above[10,1] = 1 -> Y1[10] - Y0[1] <= -0
 LinkAbove1[10,2]: Above[10,2] = 1 -> Y1[10] - Y0[2] <= -0
 LinkAbove1[10,3]: Above[10,3] = 1 -> Y1[10] - Y0[3] <= -0
 LinkAbove1[10,4]: Above[10,4] = 1 -> Y1[10] - Y0[4] <= -0
 LinkAbove1[10,5]: Above[10,5] = 1 -> Y1[10] - Y0[5] <= -0
 LinkAbove1[10,6]: Above[10,6] = 1 -> Y1[10] - Y0[6] <= -0
 LinkAbove1[10,7]: Above[10,7] = 1 -> Y1[10] - Y0[7] <= -0
 LinkAbove1[10,8]: Above[10,8] = 1 -> Y1[10] - Y0[8] <= -0
 LinkAbove1[10,9]: Above[10,9] = 1 -> Y1[10] - Y0[9] <= -0
 LinkAbove2[0,1]: Above[0,1] = 0 -> Y1[0] - Y0[1] >= 1
 LinkAbove2[0,2]: Above[0,2] = 0 -> Y1[0] - Y0[2] >= 1
 LinkAbove2[0,3]: Above[0,3] = 0 -> Y1[0] - Y0[3] >= 1
 LinkAbove2[0,4]: Above[0,4] = 0 -> Y1[0] - Y0[4] >= 1
 LinkAbove2[0,5]: Above[0,5] = 0 -> Y1[0] - Y0[5] >= 1
 LinkAbove2[0,6]: Above[0,6] = 0 -> Y1[0] - Y0[6] >= 1
 LinkAbove2[0,7]: Above[0,7] = 0 -> Y1[0] - Y0[7] >= 1
 LinkAbove2[0,8]: Above[0,8] = 0 -> Y1[0] - Y0[8] >= 1
 LinkAbove2[0,9]: Above[0,9] = 0 -> Y1[0] - Y0[9] >= 1
 LinkAbove2[0,10]: Above[0,10] = 0 -> Y1[0] - Y0[10] >= 1
 LinkAbove2[1,0]: Above[1,0] = 0 -> Y1[1] - Y0[0] >= 1
 LinkAbove2[1,2]: Above[1,2] = 0 -> Y1[1] - Y0[2] >= 1
 LinkAbove2[1,3]: Above[1,3] = 0 -> Y1[1] - Y0[3] >= 1
 LinkAbove2[1,4]: Above[1,4] = 0 -> Y1[1] - Y0[4] >= 1
 LinkAbove2[1,5]: Above[1,5] = 0 -> Y1[1] - Y0[5] >= 1
 LinkAbove2[1,6]: Above[1,6] = 0 -> Y1[1] - Y0[6] >= 1
 LinkAbove2[1,7]: Above[1,7] = 0 -> Y1[1] - Y0[7] >= 1
 LinkAbove2[1,8]: Above[1,8] = 0 -> Y1[1] - Y0[8] >= 1
 LinkAbove2[1,9]: Above[1,9] = 0 -> Y1[1] - Y0[9] >= 1
 LinkAbove2[1,10]: Above[1,10] = 0 -> Y1[1] - Y0[10] >= 1
 LinkAbove2[2,0]: Above[2,0] = 0 -> Y1[2] - Y0[0] >= 1
 LinkAbove2[2,1]: Above[2,1] = 0 -> Y1[2] - Y0[1] >= 1
 LinkAbove2[2,3]: Above[2,3] = 0 -> Y1[2] - Y0[3] >= 1
 LinkAbove2[2,4]: Above[2,4] = 0 -> Y1[2] - Y0[4] >= 1
 LinkAbove2[2,5]: Above[2,5] = 0 -> Y1[2] - Y0[5] >= 1
 LinkAbove2[2,6]: Above[2,6] = 0 -> Y1[2] - Y0[6] >= 1
 LinkAbove2[2,7]: Above[2,7] = 0 -> Y1[2] - Y0[7] >= 1
 LinkAbove2[2,8]: Above[2,8] = 0 -> Y1[2] - Y0[8] >= 1
 LinkAbove2[2,9]: Above[2,9] = 0 -> Y1[2] - Y0[9] >= 1
 LinkAbove2[2,10]: Above[2,10] = 0 -> Y1[2] - Y0[10] >= 1
 LinkAbove2[3,0]: Above[3,0] = 0 -> Y1[3] - Y0[0] >= 1
 LinkAbove2[3,1]: Above[3,1] = 0 -> Y1[3] - Y0[1] >= 1
 LinkAbove2[3,2]: Above[3,2] = 0 -> Y1[3] - Y0[2] >= 1
 LinkAbove2[3,4]: Above[3,4] = 0 -> Y1[3] - Y0[4] >= 1
 LinkAbove2[3,5]: Above[3,5] = 0 -> Y1[3] - Y0[5] >= 1
 LinkAbove2[3,6]: Above[3,6] = 0 -> Y1[3] - Y0[6] >= 1
 LinkAbove2[3,7]: Above[3,7] = 0 -> Y1[3] - Y0[7] >= 1
 LinkAbove2[3,8]: Above[3,8] = 0 -> Y1[3] - Y0[8] >= 1
 LinkAbove2[3,9]: Above[3,9] = 0 -> Y1[3] - Y0[9] >= 1
 LinkAbove2[3,10]: Above[3,10] = 0 -> Y1[3] - Y0[10] >= 1
 LinkAbove2[4,0]: Above[4,0] = 0 -> Y1[4] - Y0[0] >= 1
 LinkAbove2[4,1]: Above[4,1] = 0 -> Y1[4] - Y0[1] >= 1
 LinkAbove2[4,2]: Above[4,2] = 0 -> Y1[4] - Y0[2] >= 1
 LinkAbove2[4,3]: Above[4,3] = 0 -> Y1[4] - Y0[3] >= 1
 LinkAbove2[4,5]: Above[4,5] = 0 -> Y1[4] - Y0[5] >= 1
 LinkAbove2[4,6]: Above[4,6] = 0 -> Y1[4] - Y0[6] >= 1
 LinkAbove2[4,7]: Above[4,7] = 0 -> Y1[4] - Y0[7] >= 1
 LinkAbove2[4,8]: Above[4,8] = 0 -> Y1[4] - Y0[8] >= 1
 LinkAbove2[4,9]: Above[4,9] = 0 -> Y1[4] - Y0[9] >= 1
 LinkAbove2[4,10]: Above[4,10] = 0 -> Y1[4] - Y0[10] >= 1
 LinkAbove2[5,0]: Above[5,0] = 0 -> Y1[5] - Y0[0] >= 1
 LinkAbove2[5,1]: Above[5,1] = 0 -> Y1[5] - Y0[1] >= 1
 LinkAbove2[5,2]: Above[5,2] = 0 -> Y1[5] - Y0[2] >= 1
 LinkAbove2[5,3]: Above[5,3] = 0 -> Y1[5] - Y0[3] >= 1
 LinkAbove2[5,4]: Above[5,4] = 0 -> Y1[5] - Y0[4] >= 1
 LinkAbove2[5,6]: Above[5,6] = 0 -> Y1[5] - Y0[6] >= 1
 LinkAbove2[5,7]: Above[5,7] = 0 -> Y1[5] - Y0[7] >= 1
 LinkAbove2[5,8]: Above[5,8] = 0 -> Y1[5] - Y0[8] >= 1
 LinkAbove2[5,9]: Above[5,9] = 0 -> Y1[5] - Y0[9] >= 1
 LinkAbove2[5,10]: Above[5,10] = 0 -> Y1[5] - Y0[10] >= 1
 LinkAbove2[6,0]: Above[6,0] = 0 -> Y1[6] - Y0[0] >= 1
 LinkAbove2[6,1]: Above[6,1] = 0 -> Y1[6] - Y0[1] >= 1
 LinkAbove2[6,2]: Above[6,2] = 0 -> Y1[6] - Y0[2] >= 1
 LinkAbove2[6,3]: Above[6,3] = 0 -> Y1[6] - Y0[3] >= 1
 LinkAbove2[6,4]: Above[6,4] = 0 -> Y1[6] - Y0[4] >= 1
 LinkAbove2[6,5]: Above[6,5] = 0 -> Y1[6] - Y0[5] >= 1
 LinkAbove2[6,7]: Above[6,7] = 0 -> Y1[6] - Y0[7] >= 1
 LinkAbove2[6,8]: Above[6,8] = 0 -> Y1[6] - Y0[8] >= 1
 LinkAbove2[6,9]: Above[6,9] = 0 -> Y1[6] - Y0[9] >= 1
 LinkAbove2[6,10]: Above[6,10] = 0 -> Y1[6] - Y0[10] >= 1
 LinkAbove2[7,0]: Above[7,0] = 0 -> Y1[7] - Y0[0] >= 1
 LinkAbove2[7,1]: Above[7,1] = 0 -> Y1[7] - Y0[1] >= 1
 LinkAbove2[7,2]: Above[7,2] = 0 -> Y1[7] - Y0[2] >= 1
 LinkAbove2[7,3]: Above[7,3] = 0 -> Y1[7] - Y0[3] >= 1
 LinkAbove2[7,4]: Above[7,4] = 0 -> Y1[7] - Y0[4] >= 1
 LinkAbove2[7,5]: Above[7,5] = 0 -> Y1[7] - Y0[5] >= 1
 LinkAbove2[7,6]: Above[7,6] = 0 -> Y1[7] - Y0[6] >= 1
 LinkAbove2[7,8]: Above[7,8] = 0 -> Y1[7] - Y0[8] >= 1
 LinkAbove2[7,9]: Above[7,9] = 0 -> Y1[7] - Y0[9] >= 1
 LinkAbove2[7,10]: Above[7,10] = 0 -> Y1[7] - Y0[10] >= 1
 LinkAbove2[8,0]: Above[8,0] = 0 -> Y1[8] - Y0[0] >= 1
 LinkAbove2[8,1]: Above[8,1] = 0 -> Y1[8] - Y0[1] >= 1
 LinkAbove2[8,2]: Above[8,2] = 0 -> Y1[8] - Y0[2] >= 1
 LinkAbove2[8,3]: Above[8,3] = 0 -> Y1[8] - Y0[3] >= 1
 LinkAbove2[8,4]: Above[8,4] = 0 -> Y1[8] - Y0[4] >= 1
 LinkAbove2[8,5]: Above[8,5] = 0 -> Y1[8] - Y0[5] >= 1
 LinkAbove2[8,6]: Above[8,6] = 0 -> Y1[8] - Y0[6] >= 1
 LinkAbove2[8,7]: Above[8,7] = 0 -> Y1[8] - Y0[7] >= 1
 LinkAbove2[8,9]: Above[8,9] = 0 -> Y1[8] - Y0[9] >= 1
 LinkAbove2[8,10]: Above[8,10] = 0 -> Y1[8] - Y0[10] >= 1
 LinkAbove2[9,0]: Above[9,0] = 0 -> Y1[9] - Y0[0] >= 1
 LinkAbove2[9,1]: Above[9,1] = 0 -> Y1[9] - Y0[1] >= 1
 LinkAbove2[9,2]: Above[9,2] = 0 -> Y1[9] - Y0[2] >= 1
 LinkAbove2[9,3]: Above[9,3] = 0 -> Y1[9] - Y0[3] >= 1
 LinkAbove2[9,4]: Above[9,4] = 0 -> Y1[9] - Y0[4] >= 1
 LinkAbove2[9,5]: Above[9,5] = 0 -> Y1[9] - Y0[5] >= 1
 LinkAbove2[9,6]: Above[9,6] = 0 -> Y1[9] - Y0[6] >= 1
 LinkAbove2[9,7]: Above[9,7] = 0 -> Y1[9] - Y0[7] >= 1
 LinkAbove2[9,8]: Above[9,8] = 0 -> Y1[9] - Y0[8] >= 1
 LinkAbove2[9,10]: Above[9,10] = 0 -> Y1[9] - Y0[10] >= 1
 LinkAbove2[10,0]: Above[10,0] = 0 -> Y1[10] - Y0[0] >= 1
 LinkAbove2[10,1]: Above[10,1] = 0 -> Y1[10] - Y0[1] >= 1
 LinkAbove2[10,2]: Above[10,2] = 0 -> Y1[10] - Y0[2] >= 1
 LinkAbove2[10,3]: Above[10,3] = 0 -> Y1[10] - Y0[3] >= 1
 LinkAbove2[10,4]: Above[10,4] = 0 -> Y1[10] - Y0[4] >= 1
 LinkAbove2[10,5]: Above[10,5] = 0 -> Y1[10] - Y0[5] >= 1
 LinkAbove2[10,6]: Above[10,6] = 0 -> Y1[10] - Y0[6] >= 1
 LinkAbove2[10,7]: Above[10,7] = 0 -> Y1[10] - Y0[7] >= 1
 LinkAbove2[10,8]: Above[10,8] = 0 -> Y1[10] - Y0[8] >= 1
 LinkAbove2[10,9]: Above[10,9] = 0 -> Y1[10] - Y0[9] >= 1
 LinkOnLeft1[0,1]: OnLeft[0,1] = 1 -> X1[0] - X0[1] <= -0
 LinkOnLeft1[0,2]: OnLeft[0,2] = 1 -> X1[0] - X0[2] <= -0
 LinkOnLeft1[0,3]: OnLeft[0,3] = 1 -> X1[0] - X0[3] <= -0
 LinkOnLeft1[0,4]: OnLeft[0,4] = 1 -> X1[0] - X0[4] <= -0
 LinkOnLeft1[0,5]: OnLeft[0,5] = 1 -> X1[0] - X0[5] <= -0
 LinkOnLeft1[0,6]: OnLeft[0,6] = 1 -> X1[0] - X0[6] <= -0
 LinkOnLeft1[0,7]: OnLeft[0,7] = 1 -> X1[0] - X0[7] <= -0
 LinkOnLeft1[0,8]: OnLeft[0,8] = 1 -> X1[0] - X0[8] <= -0
 LinkOnLeft1[0,9]: OnLeft[0,9] = 1 -> X1[0] - X0[9] <= -0
 LinkOnLeft1[0,10]: OnLeft[0,10] = 1 -> X1[0] - X0[10] <= -0
 LinkOnLeft1[1,0]: OnLeft[1,0] = 1 -> X1[1] - X0[0] <= -0
 LinkOnLeft1[1,2]: OnLeft[1,2] = 1 -> X1[1] - X0[2] <= -0
 LinkOnLeft1[1,3]: OnLeft[1,3] = 1 -> X1[1] - X0[3] <= -0
 LinkOnLeft1[1,4]: OnLeft[1,4] = 1 -> X1[1] - X0[4] <= -0
 LinkOnLeft1[1,5]: OnLeft[1,5] = 1 -> X1[1] - X0[5] <= -0
 LinkOnLeft1[1,6]: OnLeft[1,6] = 1 -> X1[1] - X0[6] <= -0
 LinkOnLeft1[1,7]: OnLeft[1,7] = 1 -> X1[1] - X0[7] <= -0
 LinkOnLeft1[1,8]: OnLeft[1,8] = 1 -> X1[1] - X0[8] <= -0
 LinkOnLeft1[1,9]: OnLeft[1,9] = 1 -> X1[1] - X0[9] <= -0
 LinkOnLeft1[1,10]: OnLeft[1,10] = 1 -> X1[1] - X0[10] <= -0
 LinkOnLeft1[2,0]: OnLeft[2,0] = 1 -> X1[2] - X0[0] <= -0
 LinkOnLeft1[2,1]: OnLeft[2,1] = 1 -> X1[2] - X0[1] <= -0
 LinkOnLeft1[2,3]: OnLeft[2,3] = 1 -> X1[2] - X0[3] <= -0
 LinkOnLeft1[2,4]: OnLeft[2,4] = 1 -> X1[2] - X0[4] <= -0
 LinkOnLeft1[2,5]: OnLeft[2,5] = 1 -> X1[2] - X0[5] <= -0
 LinkOnLeft1[2,6]: OnLeft[2,6] = 1 -> X1[2] - X0[6] <= -0
 LinkOnLeft1[2,7]: OnLeft[2,7] = 1 -> X1[2] - X0[7] <= -0
 LinkOnLeft1[2,8]: OnLeft[2,8] = 1 -> X1[2] - X0[8] <= -0
 LinkOnLeft1[2,9]: OnLeft[2,9] = 1 -> X1[2] - X0[9] <= -0
 LinkOnLeft1[2,10]: OnLeft[2,10] = 1 -> X1[2] - X0[10] <= -0
 LinkOnLeft1[3,0]: OnLeft[3,0] = 1 -> X1[3] - X0[0] <= -0
 LinkOnLeft1[3,1]: OnLeft[3,1] = 1 -> X1[3] - X0[1] <= -0
 LinkOnLeft1[3,2]: OnLeft[3,2] = 1 -> X1[3] - X0[2] <= -0
 LinkOnLeft1[3,4]: OnLeft[3,4] = 1 -> X1[3] - X0[4] <= -0
 LinkOnLeft1[3,5]: OnLeft[3,5] = 1 -> X1[3] - X0[5] <= -0
 LinkOnLeft1[3,6]: OnLeft[3,6] = 1 -> X1[3] - X0[6] <= -0
 LinkOnLeft1[3,7]: OnLeft[3,7] = 1 -> X1[3] - X0[7] <= -0
 LinkOnLeft1[3,8]: OnLeft[3,8] = 1 -> X1[3] - X0[8] <= -0
 LinkOnLeft1[3,9]: OnLeft[3,9] = 1 -> X1[3] - X0[9] <= -0
 LinkOnLeft1[3,10]: OnLeft[3,10] = 1 -> X1[3] - X0[10] <= -0
 LinkOnLeft1[4,0]: OnLeft[4,0] = 1 -> X1[4] - X0[0] <= -0
 LinkOnLeft1[4,1]: OnLeft[4,1] = 1 -> X1[4] - X0[1] <= -0
 LinkOnLeft1[4,2]: OnLeft[4,2] = 1 -> X1[4] - X0[2] <= -0
 LinkOnLeft1[4,3]: OnLeft[4,3] = 1 -> X1[4] - X0[3] <= -0
 LinkOnLeft1[4,5]: OnLeft[4,5] = 1 -> X1[4] - X0[5] <= -0
 LinkOnLeft1[4,6]: OnLeft[4,6] = 1 -> X1[4] - X0[6] <= -0
 LinkOnLeft1[4,7]: OnLeft[4,7] = 1 -> X1[4] - X0[7] <= -0
 LinkOnLeft1[4,8]: OnLeft[4,8] = 1 -> X1[4] - X0[8] <= -0
 LinkOnLeft1[4,9]: OnLeft[4,9] = 1 -> X1[4] - X0[9] <= -0
 LinkOnLeft1[4,10]: OnLeft[4,10] = 1 -> X1[4] - X0[10] <= -0
 LinkOnLeft1[5,0]: OnLeft[5,0] = 1 -> X1[5] - X0[0] <= -0
 LinkOnLeft1[5,1]: OnLeft[5,1] = 1 -> X1[5] - X0[1] <= -0
 LinkOnLeft1[5,2]: OnLeft[5,2] = 1 -> X1[5] - X0[2] <= -0
 LinkOnLeft1[5,3]: OnLeft[5,3] = 1 -> X1[5] - X0[3] <= -0
 LinkOnLeft1[5,4]: OnLeft[5,4] = 1 -> X1[5] - X0[4] <= -0
 LinkOnLeft1[5,6]: OnLeft[5,6] = 1 -> X1[5] - X0[6] <= -0
 LinkOnLeft1[5,7]: OnLeft[5,7] = 1 -> X1[5] - X0[7] <= -0
 LinkOnLeft1[5,8]: OnLeft[5,8] = 1 -> X1[5] - X0[8] <= -0
 LinkOnLeft1[5,9]: OnLeft[5,9] = 1 -> X1[5] - X0[9] <= -0
 LinkOnLeft1[5,10]: OnLeft[5,10] = 1 -> X1[5] - X0[10] <= -0
 LinkOnLeft1[6,0]: OnLeft[6,0] = 1 -> X1[6] - X0[0] <= -0
 LinkOnLeft1[6,1]: OnLeft[6,1] = 1 -> X1[6] - X0[1] <= -0
 LinkOnLeft1[6,2]: OnLeft[6,2] = 1 -> X1[6] - X0[2] <= -0
 LinkOnLeft1[6,3]: OnLeft[6,3] = 1 -> X1[6] - X0[3] <= -0
 LinkOnLeft1[6,4]: OnLeft[6,4] = 1 -> X1[6] - X0[4] <= -0
 LinkOnLeft1[6,5]: OnLeft[6,5] = 1 -> X1[6] - X0[5] <= -0
 LinkOnLeft1[6,7]: OnLeft[6,7] = 1 -> X1[6] - X0[7] <= -0
 LinkOnLeft1[6,8]: OnLeft[6,8] = 1 -> X1[6] - X0[8] <= -0
 LinkOnLeft1[6,9]: OnLeft[6,9] = 1 -> X1[6] - X0[9] <= -0
 LinkOnLeft1[6,10]: OnLeft[6,10] = 1 -> X1[6] - X0[10] <= -0
 LinkOnLeft1[7,0]: OnLeft[7,0] = 1 -> X1[7] - X0[0] <= -0
 LinkOnLeft1[7,1]: OnLeft[7,1] = 1 -> X1[7] - X0[1] <= -0
 LinkOnLeft1[7,2]: OnLeft[7,2] = 1 -> X1[7] - X0[2] <= -0
 LinkOnLeft1[7,3]: OnLeft[7,3] = 1 -> X1[7] - X0[3] <= -0
 LinkOnLeft1[7,4]: OnLeft[7,4] = 1 -> X1[7] - X0[4] <= -0
 LinkOnLeft1[7,5]: OnLeft[7,5] = 1 -> X1[7] - X0[5] <= -0
 LinkOnLeft1[7,6]: OnLeft[7,6] = 1 -> X1[7] - X0[6] <= -0
 LinkOnLeft1[7,8]: OnLeft[7,8] = 1 -> X1[7] - X0[8] <= -0
 LinkOnLeft1[7,9]: OnLeft[7,9] = 1 -> X1[7] - X0[9] <= -0
 LinkOnLeft1[7,10]: OnLeft[7,10] = 1 -> X1[7] - X0[10] <= -0
 LinkOnLeft1[8,0]: OnLeft[8,0] = 1 -> X1[8] - X0[0] <= -0
 LinkOnLeft1[8,1]: OnLeft[8,1] = 1 -> X1[8] - X0[1] <= -0
 LinkOnLeft1[8,2]: OnLeft[8,2] = 1 -> X1[8] - X0[2] <= -0
 LinkOnLeft1[8,3]: OnLeft[8,3] = 1 -> X1[8] - X0[3] <= -0
 LinkOnLeft1[8,4]: OnLeft[8,4] = 1 -> X1[8] - X0[4] <= -0
 LinkOnLeft1[8,5]: OnLeft[8,5] = 1 -> X1[8] - X0[5] <= -0
 LinkOnLeft1[8,6]: OnLeft[8,6] = 1 -> X1[8] - X0[6] <= -0
 LinkOnLeft1[8,7]: OnLeft[8,7] = 1 -> X1[8] - X0[7] <= -0
 LinkOnLeft1[8,9]: OnLeft[8,9] = 1 -> X1[8] - X0[9] <= -0
 LinkOnLeft1[8,10]: OnLeft[8,10] = 1 -> X1[8] - X0[10] <= -0
 LinkOnLeft1[9,0]: OnLeft[9,0] = 1 -> X1[9] - X0[0] <= -0
 LinkOnLeft1[9,1]: OnLeft[9,1] = 1 -> X1[9] - X0[1] <= -0
 LinkOnLeft1[9,2]: OnLeft[9,2] = 1 -> X1[9] - X0[2] <= -0
 LinkOnLeft1[9,3]: OnLeft[9,3] = 1 -> X1[9] - X0[3] <= -0
 LinkOnLeft1[9,4]: OnLeft[9,4] = 1 -> X1[9] - X0[4] <= -0
 LinkOnLeft1[9,5]: OnLeft[9,5] = 1 -> X1[9] - X0[5] <= -0
 LinkOnLeft1[9,6]: OnLeft[9,6] = 1 -> X1[9] - X0[6] <= -0
 LinkOnLeft1[9,7]: OnLeft[9,7] = 1 -> X1[9] - X0[7] <= -0
 LinkOnLeft1[9,8]: OnLeft[9,8] = 1 -> X1[9] - X0[8] <= -0
 LinkOnLeft1[9,10]: OnLeft[9,10] = 1 -> X1[9] - X0[10] <= -0
 LinkOnLeft1[10,0]: OnLeft[10,0] = 1 -> X1[10] - X0[0] <= -0
 LinkOnLeft1[10,1]: OnLeft[10,1] = 1 -> X1[10] - X0[1] <= -0
 LinkOnLeft1[10,2]: OnLeft[10,2] = 1 -> X1[10] - X0[2] <= -0
 LinkOnLeft1[10,3]: OnLeft[10,3] = 1 -> X1[10] - X0[3] <= -0
 LinkOnLeft1[10,4]: OnLeft[10,4] = 1 -> X1[10] - X0[4] <= -0
 LinkOnLeft1[10,5]: OnLeft[10,5] = 1 -> X1[10] - X0[5] <= -0
 LinkOnLeft1[10,6]: OnLeft[10,6] = 1 -> X1[10] - X0[6] <= -0
 LinkOnLeft1[10,7]: OnLeft[10,7] = 1 -> X1[10] - X0[7] <= -0
 LinkOnLeft1[10,8]: OnLeft[10,8] = 1 -> X1[10] - X0[8] <= -0
 LinkOnLeft1[10,9]: OnLeft[10,9] = 1 -> X1[10] - X0[9] <= -0
 LinkOnLeft2[0,1]: OnLeft[0,1] = 0 -> X1[0] - X0[1] >= 1
 LinkOnLeft2[0,2]: OnLeft[0,2] = 0 -> X1[0] - X0[2] >= 1
 LinkOnLeft2[0,3]: OnLeft[0,3] = 0 -> X1[0] - X0[3] >= 1
 LinkOnLeft2[0,4]: OnLeft[0,4] = 0 -> X1[0] - X0[4] >= 1
 LinkOnLeft2[0,5]: OnLeft[0,5] = 0 -> X1[0] - X0[5] >= 1
 LinkOnLeft2[0,6]: OnLeft[0,6] = 0 -> X1[0] - X0[6] >= 1
 LinkOnLeft2[0,7]: OnLeft[0,7] = 0 -> X1[0] - X0[7] >= 1
 LinkOnLeft2[0,8]: OnLeft[0,8] = 0 -> X1[0] - X0[8] >= 1
 LinkOnLeft2[0,9]: OnLeft[0,9] = 0 -> X1[0] - X0[9] >= 1
 LinkOnLeft2[0,10]: OnLeft[0,10] = 0 -> X1[0] - X0[10] >= 1
 LinkOnLeft2[1,0]: OnLeft[1,0] = 0 -> X1[1] - X0[0] >= 1
 LinkOnLeft2[1,2]: OnLeft[1,2] = 0 -> X1[1] - X0[2] >= 1
 LinkOnLeft2[1,3]: OnLeft[1,3] = 0 -> X1[1] - X0[3] >= 1
 LinkOnLeft2[1,4]: OnLeft[1,4] = 0 -> X1[1] - X0[4] >= 1
 LinkOnLeft2[1,5]: OnLeft[1,5] = 0 -> X1[1] - X0[5] >= 1
 LinkOnLeft2[1,6]: OnLeft[1,6] = 0 -> X1[1] - X0[6] >= 1
 LinkOnLeft2[1,7]: OnLeft[1,7] = 0 -> X1[1] - X0[7] >= 1
 LinkOnLeft2[1,8]: OnLeft[1,8] = 0 -> X1[1] - X0[8] >= 1
 LinkOnLeft2[1,9]: OnLeft[1,9] = 0 -> X1[1] - X0[9] >= 1
 LinkOnLeft2[1,10]: OnLeft[1,10] = 0 -> X1[1] - X0[10] >= 1
 LinkOnLeft2[2,0]: OnLeft[2,0] = 0 -> X1[2] - X0[0] >= 1
 LinkOnLeft2[2,1]: OnLeft[2,1] = 0 -> X1[2] - X0[1] >= 1
 LinkOnLeft2[2,3]: OnLeft[2,3] = 0 -> X1[2] - X0[3] >= 1
 LinkOnLeft2[2,4]: OnLeft[2,4] = 0 -> X1[2] - X0[4] >= 1
 LinkOnLeft2[2,5]: OnLeft[2,5] = 0 -> X1[2] - X0[5] >= 1
 LinkOnLeft2[2,6]: OnLeft[2,6] = 0 -> X1[2] - X0[6] >= 1
 LinkOnLeft2[2,7]: OnLeft[2,7] = 0 -> X1[2] - X0[7] >= 1
 LinkOnLeft2[2,8]: OnLeft[2,8] = 0 -> X1[2] - X0[8] >= 1
 LinkOnLeft2[2,9]: OnLeft[2,9] = 0 -> X1[2] - X0[9] >= 1
 LinkOnLeft2[2,10]: OnLeft[2,10] = 0 -> X1[2] - X0[10] >= 1
 LinkOnLeft2[3,0]: OnLeft[3,0] = 0 -> X1[3] - X0[0] >= 1
 LinkOnLeft2[3,1]: OnLeft[3,1] = 0 -> X1[3] - X0[1] >= 1
 LinkOnLeft2[3,2]: OnLeft[3,2] = 0 -> X1[3] - X0[2] >= 1
 LinkOnLeft2[3,4]: OnLeft[3,4] = 0 -> X1[3] - X0[4] >= 1
 LinkOnLeft2[3,5]: OnLeft[3,5] = 0 -> X1[3] - X0[5] >= 1
 LinkOnLeft2[3,6]: OnLeft[3,6] = 0 -> X1[3] - X0[6] >= 1
 LinkOnLeft2[3,7]: OnLeft[3,7] = 0 -> X1[3] - X0[7] >= 1
 LinkOnLeft2[3,8]: OnLeft[3,8] = 0 -> X1[3] - X0[8] >= 1
 LinkOnLeft2[3,9]: OnLeft[3,9] = 0 -> X1[3] - X0[9] >= 1
 LinkOnLeft2[3,10]: OnLeft[3,10] = 0 -> X1[3] - X0[10] >= 1
 LinkOnLeft2[4,0]: OnLeft[4,0] = 0 -> X1[4] - X0[0] >= 1
 LinkOnLeft2[4,1]: OnLeft[4,1] = 0 -> X1[4] - X0[1] >= 1
 LinkOnLeft2[4,2]: OnLeft[4,2] = 0 -> X1[4] - X0[2] >= 1
 LinkOnLeft2[4,3]: OnLeft[4,3] = 0 -> X1[4] - X0[3] >= 1
 LinkOnLeft2[4,5]: OnLeft[4,5] = 0 -> X1[4] - X0[5] >= 1
 LinkOnLeft2[4,6]: OnLeft[4,6] = 0 -> X1[4] - X0[6] >= 1
 LinkOnLeft2[4,7]: OnLeft[4,7] = 0 -> X1[4] - X0[7] >= 1
 LinkOnLeft2[4,8]: OnLeft[4,8] = 0 -> X1[4] - X0[8] >= 1
 LinkOnLeft2[4,9]: OnLeft[4,9] = 0 -> X1[4] - X0[9] >= 1
 LinkOnLeft2[4,10]: OnLeft[4,10] = 0 -> X1[4] - X0[10] >= 1
 LinkOnLeft2[5,0]: OnLeft[5,0] = 0 -> X1[5] - X0[0] >= 1
 LinkOnLeft2[5,1]: OnLeft[5,1] = 0 -> X1[5] - X0[1] >= 1
 LinkOnLeft2[5,2]: OnLeft[5,2] = 0 -> X1[5] - X0[2] >= 1
 LinkOnLeft2[5,3]: OnLeft[5,3] = 0 -> X1[5] - X0[3] >= 1
 LinkOnLeft2[5,4]: OnLeft[5,4] = 0 -> X1[5] - X0[4] >= 1
 LinkOnLeft2[5,6]: OnLeft[5,6] = 0 -> X1[5] - X0[6] >= 1
 LinkOnLeft2[5,7]: OnLeft[5,7] = 0 -> X1[5] - X0[7] >= 1
 LinkOnLeft2[5,8]: OnLeft[5,8] = 0 -> X1[5] - X0[8] >= 1
 LinkOnLeft2[5,9]: OnLeft[5,9] = 0 -> X1[5] - X0[9] >= 1
 LinkOnLeft2[5,10]: OnLeft[5,10] = 0 -> X1[5] - X0[10] >= 1
 LinkOnLeft2[6,0]: OnLeft[6,0] = 0 -> X1[6] - X0[0] >= 1
 LinkOnLeft2[6,1]: OnLeft[6,1] = 0 -> X1[6] - X0[1] >= 1
 LinkOnLeft2[6,2]: OnLeft[6,2] = 0 -> X1[6] - X0[2] >= 1
 LinkOnLeft2[6,3]: OnLeft[6,3] = 0 -> X1[6] - X0[3] >= 1
 LinkOnLeft2[6,4]: OnLeft[6,4] = 0 -> X1[6] - X0[4] >= 1
 LinkOnLeft2[6,5]: OnLeft[6,5] = 0 -> X1[6] - X0[5] >= 1
 LinkOnLeft2[6,7]: OnLeft[6,7] = 0 -> X1[6] - X0[7] >= 1
 LinkOnLeft2[6,8]: OnLeft[6,8] = 0 -> X1[6] - X0[8] >= 1
 LinkOnLeft2[6,9]: OnLeft[6,9] = 0 -> X1[6] - X0[9] >= 1
 LinkOnLeft2[6,10]: OnLeft[6,10] = 0 -> X1[6] - X0[10] >= 1
 LinkOnLeft2[7,0]: OnLeft[7,0] = 0 -> X1[7] - X0[0] >= 1
 LinkOnLeft2[7,1]: OnLeft[7,1] = 0 -> X1[7] - X0[1] >= 1
 LinkOnLeft2[7,2]: OnLeft[7,2] = 0 -> X1[7] - X0[2] >= 1
 LinkOnLeft2[7,3]: OnLeft[7,3] = 0 -> X1[7] - X0[3] >= 1
 LinkOnLeft2[7,4]: OnLeft[7,4] = 0 -> X1[7] - X0[4] >= 1
 LinkOnLeft2[7,5]: OnLeft[7,5] = 0 -> X1[7] - X0[5] >= 1
 LinkOnLeft2[7,6]: OnLeft[7,6] = 0 -> X1[7] - X0[6] >= 1
 LinkOnLeft2[7,8]: OnLeft[7,8] = 0 -> X1[7] - X0[8] >= 1
 LinkOnLeft2[7,9]: OnLeft[7,9] = 0 -> X1[7] - X0[9] >= 1
 LinkOnLeft2[7,10]: OnLeft[7,10] = 0 -> X1[7] - X0[10] >= 1
 LinkOnLeft2[8,0]: OnLeft[8,0] = 0 -> X1[8] - X0[0] >= 1
 LinkOnLeft2[8,1]: OnLeft[8,1] = 0 -> X1[8] - X0[1] >= 1
 LinkOnLeft2[8,2]: OnLeft[8,2] = 0 -> X1[8] - X0[2] >= 1
 LinkOnLeft2[8,3]: OnLeft[8,3] = 0 -> X1[8] - X0[3] >= 1
 LinkOnLeft2[8,4]: OnLeft[8,4] = 0 -> X1[8] - X0[4] >= 1
 LinkOnLeft2[8,5]: OnLeft[8,5] = 0 -> X1[8] - X0[5] >= 1
 LinkOnLeft2[8,6]: OnLeft[8,6] = 0 -> X1[8] - X0[6] >= 1
 LinkOnLeft2[8,7]: OnLeft[8,7] = 0 -> X1[8] - X0[7] >= 1
 LinkOnLeft2[8,9]: OnLeft[8,9] = 0 -> X1[8] - X0[9] >= 1
 LinkOnLeft2[8,10]: OnLeft[8,10] = 0 -> X1[8] - X0[10] >= 1
 LinkOnLeft2[9,0]: OnLeft[9,0] = 0 -> X1[9] - X0[0] >= 1
 LinkOnLeft2[9,1]: OnLeft[9,1] = 0 -> X1[9] - X0[1] >= 1
 LinkOnLeft2[9,2]: OnLeft[9,2] = 0 -> X1[9] - X0[2] >= 1
 LinkOnLeft2[9,3]: OnLeft[9,3] = 0 -> X1[9] - X0[3] >= 1
 LinkOnLeft2[9,4]: OnLeft[9,4] = 0 -> X1[9] - X0[4] >= 1
 LinkOnLeft2[9,5]: OnLeft[9,5] = 0 -> X1[9] - X0[5] >= 1
 LinkOnLeft2[9,6]: OnLeft[9,6] = 0 -> X1[9] - X0[6] >= 1
 LinkOnLeft2[9,7]: OnLeft[9,7] = 0 -> X1[9] - X0[7] >= 1
 LinkOnLeft2[9,8]: OnLeft[9,8] = 0 -> X1[9] - X0[8] >= 1
 LinkOnLeft2[9,10]: OnLeft[9,10] = 0 -> X1[9] - X0[10] >= 1
 LinkOnLeft2[10,0]: OnLeft[10,0] = 0 -> X1[10] - X0[0] >= 1
 LinkOnLeft2[10,1]: OnLeft[10,1] = 0 -> X1[10] - X0[1] >= 1
 LinkOnLeft2[10,2]: OnLeft[10,2] = 0 -> X1[10] - X0[2] >= 1
 LinkOnLeft2[10,3]: OnLeft[10,3] = 0 -> X1[10] - X0[3] >= 1
 LinkOnLeft2[10,4]: OnLeft[10,4] = 0 -> X1[10] - X0[4] >= 1
 LinkOnLeft2[10,5]: OnLeft[10,5] = 0 -> X1[10] - X0[5] >= 1
 LinkOnLeft2[10,6]: OnLeft[10,6] = 0 -> X1[10] - X0[6] >= 1
 LinkOnLeft2[10,7]: OnLeft[10,7] = 0 -> X1[10] - X0[7] >= 1
 LinkOnLeft2[10,8]: OnLeft[10,8] = 0 -> X1[10] - X0[8] >= 1
 LinkOnLeft2[10,9]: OnLeft[10,9] = 0 -> X1[10] - X0[9] >= 1
 LinkX0LessThan1[0,0]: X0LessThan[0,0] = 1 -> X0Diff[0,0] <= -1
 LinkX0LessThan1[0,1]: X0LessThan[0,1] = 1 -> X0Diff[0,1] <= -1
 LinkX0LessThan1[0,2]: X0LessThan[0,2] = 1 -> X0Diff[0,2] <= -1
 LinkX0LessThan1[0,3]: X0LessThan[0,3] = 1 -> X0Diff[0,3] <= -1
 LinkX0LessThan1[0,4]: X0LessThan[0,4] = 1 -> X0Diff[0,4] <= -1
 LinkX0LessThan1[0,5]: X0LessThan[0,5] = 1 -> X0Diff[0,5] <= -1
 LinkX0LessThan1[0,6]: X0LessThan[0,6] = 1 -> X0Diff[0,6] <= -1
 LinkX0LessThan1[0,7]: X0LessThan[0,7] = 1 -> X0Diff[0,7] <= -1
 LinkX0LessThan1[0,8]: X0LessThan[0,8] = 1 -> X0Diff[0,8] <= -1
 LinkX0LessThan1[0,9]: X0LessThan[0,9] = 1 -> X0Diff[0,9] <= -1
 LinkX0LessThan1[0,10]: X0LessThan[0,10] = 1 -> X0Diff[0,10] <= -1
 LinkX0LessThan1[1,0]: X0LessThan[1,0] = 1 -> X0Diff[1,0] <= -1
 LinkX0LessThan1[1,1]: X0LessThan[1,1] = 1 -> X0Diff[1,1] <= -1
 LinkX0LessThan1[1,2]: X0LessThan[1,2] = 1 -> X0Diff[1,2] <= -1
 LinkX0LessThan1[1,3]: X0LessThan[1,3] = 1 -> X0Diff[1,3] <= -1
 LinkX0LessThan1[1,4]: X0LessThan[1,4] = 1 -> X0Diff[1,4] <= -1
 LinkX0LessThan1[1,5]: X0LessThan[1,5] = 1 -> X0Diff[1,5] <= -1
 LinkX0LessThan1[1,6]: X0LessThan[1,6] = 1 -> X0Diff[1,6] <= -1
 LinkX0LessThan1[1,7]: X0LessThan[1,7] = 1 -> X0Diff[1,7] <= -1
 LinkX0LessThan1[1,8]: X0LessThan[1,8] = 1 -> X0Diff[1,8] <= -1
 LinkX0LessThan1[1,9]: X0LessThan[1,9] = 1 -> X0Diff[1,9] <= -1
 LinkX0LessThan1[1,10]: X0LessThan[1,10] = 1 -> X0Diff[1,10] <= -1
 LinkX0LessThan1[2,0]: X0LessThan[2,0] = 1 -> X0Diff[2,0] <= -1
 LinkX0LessThan1[2,1]: X0LessThan[2,1] = 1 -> X0Diff[2,1] <= -1
 LinkX0LessThan1[2,2]: X0LessThan[2,2] = 1 -> X0Diff[2,2] <= -1
 LinkX0LessThan1[2,3]: X0LessThan[2,3] = 1 -> X0Diff[2,3] <= -1
 LinkX0LessThan1[2,4]: X0LessThan[2,4] = 1 -> X0Diff[2,4] <= -1
 LinkX0LessThan1[2,5]: X0LessThan[2,5] = 1 -> X0Diff[2,5] <= -1
 LinkX0LessThan1[2,6]: X0LessThan[2,6] = 1 -> X0Diff[2,6] <= -1
 LinkX0LessThan1[2,7]: X0LessThan[2,7] = 1 -> X0Diff[2,7] <= -1
 LinkX0LessThan1[2,8]: X0LessThan[2,8] = 1 -> X0Diff[2,8] <= -1
 LinkX0LessThan1[2,9]: X0LessThan[2,9] = 1 -> X0Diff[2,9] <= -1
 LinkX0LessThan1[2,10]: X0LessThan[2,10] = 1 -> X0Diff[2,10] <= -1
 LinkX0LessThan1[3,0]: X0LessThan[3,0] = 1 -> X0Diff[3,0] <= -1
 LinkX0LessThan1[3,1]: X0LessThan[3,1] = 1 -> X0Diff[3,1] <= -1
 LinkX0LessThan1[3,2]: X0LessThan[3,2] = 1 -> X0Diff[3,2] <= -1
 LinkX0LessThan1[3,3]: X0LessThan[3,3] = 1 -> X0Diff[3,3] <= -1
 LinkX0LessThan1[3,4]: X0LessThan[3,4] = 1 -> X0Diff[3,4] <= -1
 LinkX0LessThan1[3,5]: X0LessThan[3,5] = 1 -> X0Diff[3,5] <= -1
 LinkX0LessThan1[3,6]: X0LessThan[3,6] = 1 -> X0Diff[3,6] <= -1
 LinkX0LessThan1[3,7]: X0LessThan[3,7] = 1 -> X0Diff[3,7] <= -1
 LinkX0LessThan1[3,8]: X0LessThan[3,8] = 1 -> X0Diff[3,8] <= -1
 LinkX0LessThan1[3,9]: X0LessThan[3,9] = 1 -> X0Diff[3,9] <= -1
 LinkX0LessThan1[3,10]: X0LessThan[3,10] = 1 -> X0Diff[3,10] <= -1
 LinkX0LessThan1[4,0]: X0LessThan[4,0] = 1 -> X0Diff[4,0] <= -1
 LinkX0LessThan1[4,1]: X0LessThan[4,1] = 1 -> X0Diff[4,1] <= -1
 LinkX0LessThan1[4,2]: X0LessThan[4,2] = 1 -> X0Diff[4,2] <= -1
 LinkX0LessThan1[4,3]: X0LessThan[4,3] = 1 -> X0Diff[4,3] <= -1
 LinkX0LessThan1[4,4]: X0LessThan[4,4] = 1 -> X0Diff[4,4] <= -1
 LinkX0LessThan1[4,5]: X0LessThan[4,5] = 1 -> X0Diff[4,5] <= -1
 LinkX0LessThan1[4,6]: X0LessThan[4,6] = 1 -> X0Diff[4,6] <= -1
 LinkX0LessThan1[4,7]: X0LessThan[4,7] = 1 -> X0Diff[4,7] <= -1
 LinkX0LessThan1[4,8]: X0LessThan[4,8] = 1 -> X0Diff[4,8] <= -1
 LinkX0LessThan1[4,9]: X0LessThan[4,9] = 1 -> X0Diff[4,9] <= -1
 LinkX0LessThan1[4,10]: X0LessThan[4,10] = 1 -> X0Diff[4,10] <= -1
 LinkX0LessThan1[5,0]: X0LessThan[5,0] = 1 -> X0Diff[5,0] <= -1
 LinkX0LessThan1[5,1]: X0LessThan[5,1] = 1 -> X0Diff[5,1] <= -1
 LinkX0LessThan1[5,2]: X0LessThan[5,2] = 1 -> X0Diff[5,2] <= -1
 LinkX0LessThan1[5,3]: X0LessThan[5,3] = 1 -> X0Diff[5,3] <= -1
 LinkX0LessThan1[5,4]: X0LessThan[5,4] = 1 -> X0Diff[5,4] <= -1
 LinkX0LessThan1[5,5]: X0LessThan[5,5] = 1 -> X0Diff[5,5] <= -1
 LinkX0LessThan1[5,6]: X0LessThan[5,6] = 1 -> X0Diff[5,6] <= -1
 LinkX0LessThan1[5,7]: X0LessThan[5,7] = 1 -> X0Diff[5,7] <= -1
 LinkX0LessThan1[5,8]: X0LessThan[5,8] = 1 -> X0Diff[5,8] <= -1
 LinkX0LessThan1[5,9]: X0LessThan[5,9] = 1 -> X0Diff[5,9] <= -1
 LinkX0LessThan1[5,10]: X0LessThan[5,10] = 1 -> X0Diff[5,10] <= -1
 LinkX0LessThan1[6,0]: X0LessThan[6,0] = 1 -> X0Diff[6,0] <= -1
 LinkX0LessThan1[6,1]: X0LessThan[6,1] = 1 -> X0Diff[6,1] <= -1
 LinkX0LessThan1[6,2]: X0LessThan[6,2] = 1 -> X0Diff[6,2] <= -1
 LinkX0LessThan1[6,3]: X0LessThan[6,3] = 1 -> X0Diff[6,3] <= -1
 LinkX0LessThan1[6,4]: X0LessThan[6,4] = 1 -> X0Diff[6,4] <= -1
 LinkX0LessThan1[6,5]: X0LessThan[6,5] = 1 -> X0Diff[6,5] <= -1
 LinkX0LessThan1[6,6]: X0LessThan[6,6] = 1 -> X0Diff[6,6] <= -1
 LinkX0LessThan1[6,7]: X0LessThan[6,7] = 1 -> X0Diff[6,7] <= -1
 LinkX0LessThan1[6,8]: X0LessThan[6,8] = 1 -> X0Diff[6,8] <= -1
 LinkX0LessThan1[6,9]: X0LessThan[6,9] = 1 -> X0Diff[6,9] <= -1
 LinkX0LessThan1[6,10]: X0LessThan[6,10] = 1 -> X0Diff[6,10] <= -1
 LinkX0LessThan1[7,0]: X0LessThan[7,0] = 1 -> X0Diff[7,0] <= -1
 LinkX0LessThan1[7,1]: X0LessThan[7,1] = 1 -> X0Diff[7,1] <= -1
 LinkX0LessThan1[7,2]: X0LessThan[7,2] = 1 -> X0Diff[7,2] <= -1
 LinkX0LessThan1[7,3]: X0LessThan[7,3] = 1 -> X0Diff[7,3] <= -1
 LinkX0LessThan1[7,4]: X0LessThan[7,4] = 1 -> X0Diff[7,4] <= -1
 LinkX0LessThan1[7,5]: X0LessThan[7,5] = 1 -> X0Diff[7,5] <= -1
 LinkX0LessThan1[7,6]: X0LessThan[7,6] = 1 -> X0Diff[7,6] <= -1
 LinkX0LessThan1[7,7]: X0LessThan[7,7] = 1 -> X0Diff[7,7] <= -1
 LinkX0LessThan1[7,8]: X0LessThan[7,8] = 1 -> X0Diff[7,8] <= -1
 LinkX0LessThan1[7,9]: X0LessThan[7,9] = 1 -> X0Diff[7,9] <= -1
 LinkX0LessThan1[7,10]: X0LessThan[7,10] = 1 -> X0Diff[7,10] <= -1
 LinkX0LessThan1[8,0]: X0LessThan[8,0] = 1 -> X0Diff[8,0] <= -1
 LinkX0LessThan1[8,1]: X0LessThan[8,1] = 1 -> X0Diff[8,1] <= -1
 LinkX0LessThan1[8,2]: X0LessThan[8,2] = 1 -> X0Diff[8,2] <= -1
 LinkX0LessThan1[8,3]: X0LessThan[8,3] = 1 -> X0Diff[8,3] <= -1
 LinkX0LessThan1[8,4]: X0LessThan[8,4] = 1 -> X0Diff[8,4] <= -1
 LinkX0LessThan1[8,5]: X0LessThan[8,5] = 1 -> X0Diff[8,5] <= -1
 LinkX0LessThan1[8,6]: X0LessThan[8,6] = 1 -> X0Diff[8,6] <= -1
 LinkX0LessThan1[8,7]: X0LessThan[8,7] = 1 -> X0Diff[8,7] <= -1
 LinkX0LessThan1[8,8]: X0LessThan[8,8] = 1 -> X0Diff[8,8] <= -1
 LinkX0LessThan1[8,9]: X0LessThan[8,9] = 1 -> X0Diff[8,9] <= -1
 LinkX0LessThan1[8,10]: X0LessThan[8,10] = 1 -> X0Diff[8,10] <= -1
 LinkX0LessThan1[9,0]: X0LessThan[9,0] = 1 -> X0Diff[9,0] <= -1
 LinkX0LessThan1[9,1]: X0LessThan[9,1] = 1 -> X0Diff[9,1] <= -1
 LinkX0LessThan1[9,2]: X0LessThan[9,2] = 1 -> X0Diff[9,2] <= -1
 LinkX0LessThan1[9,3]: X0LessThan[9,3] = 1 -> X0Diff[9,3] <= -1
 LinkX0LessThan1[9,4]: X0LessThan[9,4] = 1 -> X0Diff[9,4] <= -1
 LinkX0LessThan1[9,5]: X0LessThan[9,5] = 1 -> X0Diff[9,5] <= -1
 LinkX0LessThan1[9,6]: X0LessThan[9,6] = 1 -> X0Diff[9,6] <= -1
 LinkX0LessThan1[9,7]: X0LessThan[9,7] = 1 -> X0Diff[9,7] <= -1
 LinkX0LessThan1[9,8]: X0LessThan[9,8] = 1 -> X0Diff[9,8] <= -1
 LinkX0LessThan1[9,9]: X0LessThan[9,9] = 1 -> X0Diff[9,9] <= -1
 LinkX0LessThan1[9,10]: X0LessThan[9,10] = 1 -> X0Diff[9,10] <= -1
 LinkX0LessThan1[10,0]: X0LessThan[10,0] = 1 -> X0Diff[10,0] <= -1
 LinkX0LessThan1[10,1]: X0LessThan[10,1] = 1 -> X0Diff[10,1] <= -1
 LinkX0LessThan1[10,2]: X0LessThan[10,2] = 1 -> X0Diff[10,2] <= -1
 LinkX0LessThan1[10,3]: X0LessThan[10,3] = 1 -> X0Diff[10,3] <= -1
 LinkX0LessThan1[10,4]: X0LessThan[10,4] = 1 -> X0Diff[10,4] <= -1
 LinkX0LessThan1[10,5]: X0LessThan[10,5] = 1 -> X0Diff[10,5] <= -1
 LinkX0LessThan1[10,6]: X0LessThan[10,6] = 1 -> X0Diff[10,6] <= -1
 LinkX0LessThan1[10,7]: X0LessThan[10,7] = 1 -> X0Diff[10,7] <= -1
 LinkX0LessThan1[10,8]: X0LessThan[10,8] = 1 -> X0Diff[10,8] <= -1
 LinkX0LessThan1[10,9]: X0LessThan[10,9] = 1 -> X0Diff[10,9] <= -1
 LinkX0LessThan1[10,10]: X0LessThan[10,10] = 1 -> X0Diff[10,10] <= -1
 LinkX0LessThan2[0,0]: X0LessThan[0,0] = 0 -> X0Diff[0,0] >= -0
 LinkX0LessThan2[0,1]: X0LessThan[0,1] = 0 -> X0Diff[0,1] >= -0
 LinkX0LessThan2[0,2]: X0LessThan[0,2] = 0 -> X0Diff[0,2] >= -0
 LinkX0LessThan2[0,3]: X0LessThan[0,3] = 0 -> X0Diff[0,3] >= -0
 LinkX0LessThan2[0,4]: X0LessThan[0,4] = 0 -> X0Diff[0,4] >= -0
 LinkX0LessThan2[0,5]: X0LessThan[0,5] = 0 -> X0Diff[0,5] >= -0
 LinkX0LessThan2[0,6]: X0LessThan[0,6] = 0 -> X0Diff[0,6] >= -0
 LinkX0LessThan2[0,7]: X0LessThan[0,7] = 0 -> X0Diff[0,7] >= -0
 LinkX0LessThan2[0,8]: X0LessThan[0,8] = 0 -> X0Diff[0,8] >= -0
 LinkX0LessThan2[0,9]: X0LessThan[0,9] = 0 -> X0Diff[0,9] >= -0
 LinkX0LessThan2[0,10]: X0LessThan[0,10] = 0 -> X0Diff[0,10] >= -0
 LinkX0LessThan2[1,0]: X0LessThan[1,0] = 0 -> X0Diff[1,0] >= -0
 LinkX0LessThan2[1,1]: X0LessThan[1,1] = 0 -> X0Diff[1,1] >= -0
 LinkX0LessThan2[1,2]: X0LessThan[1,2] = 0 -> X0Diff[1,2] >= -0
 LinkX0LessThan2[1,3]: X0LessThan[1,3] = 0 -> X0Diff[1,3] >= -0
 LinkX0LessThan2[1,4]: X0LessThan[1,4] = 0 -> X0Diff[1,4] >= -0
 LinkX0LessThan2[1,5]: X0LessThan[1,5] = 0 -> X0Diff[1,5] >= -0
 LinkX0LessThan2[1,6]: X0LessThan[1,6] = 0 -> X0Diff[1,6] >= -0
 LinkX0LessThan2[1,7]: X0LessThan[1,7] = 0 -> X0Diff[1,7] >= -0
 LinkX0LessThan2[1,8]: X0LessThan[1,8] = 0 -> X0Diff[1,8] >= -0
 LinkX0LessThan2[1,9]: X0LessThan[1,9] = 0 -> X0Diff[1,9] >= -0
 LinkX0LessThan2[1,10]: X0LessThan[1,10] = 0 -> X0Diff[1,10] >= -0
 LinkX0LessThan2[2,0]: X0LessThan[2,0] = 0 -> X0Diff[2,0] >= -0
 LinkX0LessThan2[2,1]: X0LessThan[2,1] = 0 -> X0Diff[2,1] >= -0
 LinkX0LessThan2[2,2]: X0LessThan[2,2] = 0 -> X0Diff[2,2] >= -0
 LinkX0LessThan2[2,3]: X0LessThan[2,3] = 0 -> X0Diff[2,3] >= -0
 LinkX0LessThan2[2,4]: X0LessThan[2,4] = 0 -> X0Diff[2,4] >= -0
 LinkX0LessThan2[2,5]: X0LessThan[2,5] = 0 -> X0Diff[2,5] >= -0
 LinkX0LessThan2[2,6]: X0LessThan[2,6] = 0 -> X0Diff[2,6] >= -0
 LinkX0LessThan2[2,7]: X0LessThan[2,7] = 0 -> X0Diff[2,7] >= -0
 LinkX0LessThan2[2,8]: X0LessThan[2,8] = 0 -> X0Diff[2,8] >= -0
 LinkX0LessThan2[2,9]: X0LessThan[2,9] = 0 -> X0Diff[2,9] >= -0
 LinkX0LessThan2[2,10]: X0LessThan[2,10] = 0 -> X0Diff[2,10] >= -0
 LinkX0LessThan2[3,0]: X0LessThan[3,0] = 0 -> X0Diff[3,0] >= -0
 LinkX0LessThan2[3,1]: X0LessThan[3,1] = 0 -> X0Diff[3,1] >= -0
 LinkX0LessThan2[3,2]: X0LessThan[3,2] = 0 -> X0Diff[3,2] >= -0
 LinkX0LessThan2[3,3]: X0LessThan[3,3] = 0 -> X0Diff[3,3] >= -0
 LinkX0LessThan2[3,4]: X0LessThan[3,4] = 0 -> X0Diff[3,4] >= -0
 LinkX0LessThan2[3,5]: X0LessThan[3,5] = 0 -> X0Diff[3,5] >= -0
 LinkX0LessThan2[3,6]: X0LessThan[3,6] = 0 -> X0Diff[3,6] >= -0
 LinkX0LessThan2[3,7]: X0LessThan[3,7] = 0 -> X0Diff[3,7] >= -0
 LinkX0LessThan2[3,8]: X0LessThan[3,8] = 0 -> X0Diff[3,8] >= -0
 LinkX0LessThan2[3,9]: X0LessThan[3,9] = 0 -> X0Diff[3,9] >= -0
 LinkX0LessThan2[3,10]: X0LessThan[3,10] = 0 -> X0Diff[3,10] >= -0
 LinkX0LessThan2[4,0]: X0LessThan[4,0] = 0 -> X0Diff[4,0] >= -0
 LinkX0LessThan2[4,1]: X0LessThan[4,1] = 0 -> X0Diff[4,1] >= -0
 LinkX0LessThan2[4,2]: X0LessThan[4,2] = 0 -> X0Diff[4,2] >= -0
 LinkX0LessThan2[4,3]: X0LessThan[4,3] = 0 -> X0Diff[4,3] >= -0
 LinkX0LessThan2[4,4]: X0LessThan[4,4] = 0 -> X0Diff[4,4] >= -0
 LinkX0LessThan2[4,5]: X0LessThan[4,5] = 0 -> X0Diff[4,5] >= -0
 LinkX0LessThan2[4,6]: X0LessThan[4,6] = 0 -> X0Diff[4,6] >= -0
 LinkX0LessThan2[4,7]: X0LessThan[4,7] = 0 -> X0Diff[4,7] >= -0
 LinkX0LessThan2[4,8]: X0LessThan[4,8] = 0 -> X0Diff[4,8] >= -0
 LinkX0LessThan2[4,9]: X0LessThan[4,9] = 0 -> X0Diff[4,9] >= -0
 LinkX0LessThan2[4,10]: X0LessThan[4,10] = 0 -> X0Diff[4,10] >= -0
 LinkX0LessThan2[5,0]: X0LessThan[5,0] = 0 -> X0Diff[5,0] >= -0
 LinkX0LessThan2[5,1]: X0LessThan[5,1] = 0 -> X0Diff[5,1] >= -0
 LinkX0LessThan2[5,2]: X0LessThan[5,2] = 0 -> X0Diff[5,2] >= -0
 LinkX0LessThan2[5,3]: X0LessThan[5,3] = 0 -> X0Diff[5,3] >= -0
 LinkX0LessThan2[5,4]: X0LessThan[5,4] = 0 -> X0Diff[5,4] >= -0
 LinkX0LessThan2[5,5]: X0LessThan[5,5] = 0 -> X0Diff[5,5] >= -0
 LinkX0LessThan2[5,6]: X0LessThan[5,6] = 0 -> X0Diff[5,6] >= -0
 LinkX0LessThan2[5,7]: X0LessThan[5,7] = 0 -> X0Diff[5,7] >= -0
 LinkX0LessThan2[5,8]: X0LessThan[5,8] = 0 -> X0Diff[5,8] >= -0
 LinkX0LessThan2[5,9]: X0LessThan[5,9] = 0 -> X0Diff[5,9] >= -0
 LinkX0LessThan2[5,10]: X0LessThan[5,10] = 0 -> X0Diff[5,10] >= -0
 LinkX0LessThan2[6,0]: X0LessThan[6,0] = 0 -> X0Diff[6,0] >= -0
 LinkX0LessThan2[6,1]: X0LessThan[6,1] = 0 -> X0Diff[6,1] >= -0
 LinkX0LessThan2[6,2]: X0LessThan[6,2] = 0 -> X0Diff[6,2] >= -0
 LinkX0LessThan2[6,3]: X0LessThan[6,3] = 0 -> X0Diff[6,3] >= -0
 LinkX0LessThan2[6,4]: X0LessThan[6,4] = 0 -> X0Diff[6,4] >= -0
 LinkX0LessThan2[6,5]: X0LessThan[6,5] = 0 -> X0Diff[6,5] >= -0
 LinkX0LessThan2[6,6]: X0LessThan[6,6] = 0 -> X0Diff[6,6] >= -0
 LinkX0LessThan2[6,7]: X0LessThan[6,7] = 0 -> X0Diff[6,7] >= -0
 LinkX0LessThan2[6,8]: X0LessThan[6,8] = 0 -> X0Diff[6,8] >= -0
 LinkX0LessThan2[6,9]: X0LessThan[6,9] = 0 -> X0Diff[6,9] >= -0
 LinkX0LessThan2[6,10]: X0LessThan[6,10] = 0 -> X0Diff[6,10] >= -0
 LinkX0LessThan2[7,0]: X0LessThan[7,0] = 0 -> X0Diff[7,0] >= -0
 LinkX0LessThan2[7,1]: X0LessThan[7,1] = 0 -> X0Diff[7,1] >= -0
 LinkX0LessThan2[7,2]: X0LessThan[7,2] = 0 -> X0Diff[7,2] >= -0
 LinkX0LessThan2[7,3]: X0LessThan[7,3] = 0 -> X0Diff[7,3] >= -0
 LinkX0LessThan2[7,4]: X0LessThan[7,4] = 0 -> X0Diff[7,4] >= -0
 LinkX0LessThan2[7,5]: X0LessThan[7,5] = 0 -> X0Diff[7,5] >= -0
 LinkX0LessThan2[7,6]: X0LessThan[7,6] = 0 -> X0Diff[7,6] >= -0
 LinkX0LessThan2[7,7]: X0LessThan[7,7] = 0 -> X0Diff[7,7] >= -0
 LinkX0LessThan2[7,8]: X0LessThan[7,8] = 0 -> X0Diff[7,8] >= -0
 LinkX0LessThan2[7,9]: X0LessThan[7,9] = 0 -> X0Diff[7,9] >= -0
 LinkX0LessThan2[7,10]: X0LessThan[7,10] = 0 -> X0Diff[7,10] >= -0
 LinkX0LessThan2[8,0]: X0LessThan[8,0] = 0 -> X0Diff[8,0] >= -0
 LinkX0LessThan2[8,1]: X0LessThan[8,1] = 0 -> X0Diff[8,1] >= -0
 LinkX0LessThan2[8,2]: X0LessThan[8,2] = 0 -> X0Diff[8,2] >= -0
 LinkX0LessThan2[8,3]: X0LessThan[8,3] = 0 -> X0Diff[8,3] >= -0
 LinkX0LessThan2[8,4]: X0LessThan[8,4] = 0 -> X0Diff[8,4] >= -0
 LinkX0LessThan2[8,5]: X0LessThan[8,5] = 0 -> X0Diff[8,5] >= -0
 LinkX0LessThan2[8,6]: X0LessThan[8,6] = 0 -> X0Diff[8,6] >= -0
 LinkX0LessThan2[8,7]: X0LessThan[8,7] = 0 -> X0Diff[8,7] >= -0
 LinkX0LessThan2[8,8]: X0LessThan[8,8] = 0 -> X0Diff[8,8] >= -0
 LinkX0LessThan2[8,9]: X0LessThan[8,9] = 0 -> X0Diff[8,9] >= -0
 LinkX0LessThan2[8,10]: X0LessThan[8,10] = 0 -> X0Diff[8,10] >= -0
 LinkX0LessThan2[9,0]: X0LessThan[9,0] = 0 -> X0Diff[9,0] >= -0
 LinkX0LessThan2[9,1]: X0LessThan[9,1] = 0 -> X0Diff[9,1] >= -0
 LinkX0LessThan2[9,2]: X0LessThan[9,2] = 0 -> X0Diff[9,2] >= -0
 LinkX0LessThan2[9,3]: X0LessThan[9,3] = 0 -> X0Diff[9,3] >= -0
 LinkX0LessThan2[9,4]: X0LessThan[9,4] = 0 -> X0Diff[9,4] >= -0
 LinkX0LessThan2[9,5]: X0LessThan[9,5] = 0 -> X0Diff[9,5] >= -0
 LinkX0LessThan2[9,6]: X0LessThan[9,6] = 0 -> X0Diff[9,6] >= -0
 LinkX0LessThan2[9,7]: X0LessThan[9,7] = 0 -> X0Diff[9,7] >= -0
 LinkX0LessThan2[9,8]: X0LessThan[9,8] = 0 -> X0Diff[9,8] >= -0
 LinkX0LessThan2[9,9]: X0LessThan[9,9] = 0 -> X0Diff[9,9] >= -0
 LinkX0LessThan2[9,10]: X0LessThan[9,10] = 0 -> X0Diff[9,10] >= -0
 LinkX0LessThan2[10,0]: X0LessThan[10,0] = 0 -> X0Diff[10,0] >= -0
 LinkX0LessThan2[10,1]: X0LessThan[10,1] = 0 -> X0Diff[10,1] >= -0
 LinkX0LessThan2[10,2]: X0LessThan[10,2] = 0 -> X0Diff[10,2] >= -0
 LinkX0LessThan2[10,3]: X0LessThan[10,3] = 0 -> X0Diff[10,3] >= -0
 LinkX0LessThan2[10,4]: X0LessThan[10,4] = 0 -> X0Diff[10,4] >= -0
 LinkX0LessThan2[10,5]: X0LessThan[10,5] = 0 -> X0Diff[10,5] >= -0
 LinkX0LessThan2[10,6]: X0LessThan[10,6] = 0 -> X0Diff[10,6] >= -0
 LinkX0LessThan2[10,7]: X0LessThan[10,7] = 0 -> X0Diff[10,7] >= -0
 LinkX0LessThan2[10,8]: X0LessThan[10,8] = 0 -> X0Diff[10,8] >= -0
 LinkX0LessThan2[10,9]: X0LessThan[10,9] = 0 -> X0Diff[10,9] >= -0
 LinkX0LessThan2[10,10]: X0LessThan[10,10] = 0 -> X0Diff[10,10] >= -0
 LinkY0LessThan1[0,0]: Y0LessThan[0,0] = 1 -> Y0Diff[0,0] <= -1
 LinkY0LessThan1[0,1]: Y0LessThan[0,1] = 1 -> Y0Diff[0,1] <= -1
 LinkY0LessThan1[0,2]: Y0LessThan[0,2] = 1 -> Y0Diff[0,2] <= -1
 LinkY0LessThan1[0,3]: Y0LessThan[0,3] = 1 -> Y0Diff[0,3] <= -1
 LinkY0LessThan1[0,4]: Y0LessThan[0,4] = 1 -> Y0Diff[0,4] <= -1
 LinkY0LessThan1[0,5]: Y0LessThan[0,5] = 1 -> Y0Diff[0,5] <= -1
 LinkY0LessThan1[0,6]: Y0LessThan[0,6] = 1 -> Y0Diff[0,6] <= -1
 LinkY0LessThan1[0,7]: Y0LessThan[0,7] = 1 -> Y0Diff[0,7] <= -1
 LinkY0LessThan1[0,8]: Y0LessThan[0,8] = 1 -> Y0Diff[0,8] <= -1
 LinkY0LessThan1[0,9]: Y0LessThan[0,9] = 1 -> Y0Diff[0,9] <= -1
 LinkY0LessThan1[0,10]: Y0LessThan[0,10] = 1 -> Y0Diff[0,10] <= -1
 LinkY0LessThan1[1,0]: Y0LessThan[1,0] = 1 -> Y0Diff[1,0] <= -1
 LinkY0LessThan1[1,1]: Y0LessThan[1,1] = 1 -> Y0Diff[1,1] <= -1
 LinkY0LessThan1[1,2]: Y0LessThan[1,2] = 1 -> Y0Diff[1,2] <= -1
 LinkY0LessThan1[1,3]: Y0LessThan[1,3] = 1 -> Y0Diff[1,3] <= -1
 LinkY0LessThan1[1,4]: Y0LessThan[1,4] = 1 -> Y0Diff[1,4] <= -1
 LinkY0LessThan1[1,5]: Y0LessThan[1,5] = 1 -> Y0Diff[1,5] <= -1
 LinkY0LessThan1[1,6]: Y0LessThan[1,6] = 1 -> Y0Diff[1,6] <= -1
 LinkY0LessThan1[1,7]: Y0LessThan[1,7] = 1 -> Y0Diff[1,7] <= -1
 LinkY0LessThan1[1,8]: Y0LessThan[1,8] = 1 -> Y0Diff[1,8] <= -1
 LinkY0LessThan1[1,9]: Y0LessThan[1,9] = 1 -> Y0Diff[1,9] <= -1
 LinkY0LessThan1[1,10]: Y0LessThan[1,10] = 1 -> Y0Diff[1,10] <= -1
 LinkY0LessThan1[2,0]: Y0LessThan[2,0] = 1 -> Y0Diff[2,0] <= -1
 LinkY0LessThan1[2,1]: Y0LessThan[2,1] = 1 -> Y0Diff[2,1] <= -1
 LinkY0LessThan1[2,2]: Y0LessThan[2,2] = 1 -> Y0Diff[2,2] <= -1
 LinkY0LessThan1[2,3]: Y0LessThan[2,3] = 1 -> Y0Diff[2,3] <= -1
 LinkY0LessThan1[2,4]: Y0LessThan[2,4] = 1 -> Y0Diff[2,4] <= -1
 LinkY0LessThan1[2,5]: Y0LessThan[2,5] = 1 -> Y0Diff[2,5] <= -1
 LinkY0LessThan1[2,6]: Y0LessThan[2,6] = 1 -> Y0Diff[2,6] <= -1
 LinkY0LessThan1[2,7]: Y0LessThan[2,7] = 1 -> Y0Diff[2,7] <= -1
 LinkY0LessThan1[2,8]: Y0LessThan[2,8] = 1 -> Y0Diff[2,8] <= -1
 LinkY0LessThan1[2,9]: Y0LessThan[2,9] = 1 -> Y0Diff[2,9] <= -1
 LinkY0LessThan1[2,10]: Y0LessThan[2,10] = 1 -> Y0Diff[2,10] <= -1
 LinkY0LessThan1[3,0]: Y0LessThan[3,0] = 1 -> Y0Diff[3,0] <= -1
 LinkY0LessThan1[3,1]: Y0LessThan[3,1] = 1 -> Y0Diff[3,1] <= -1
 LinkY0LessThan1[3,2]: Y0LessThan[3,2] = 1 -> Y0Diff[3,2] <= -1
 LinkY0LessThan1[3,3]: Y0LessThan[3,3] = 1 -> Y0Diff[3,3] <= -1
 LinkY0LessThan1[3,4]: Y0LessThan[3,4] = 1 -> Y0Diff[3,4] <= -1
 LinkY0LessThan1[3,5]: Y0LessThan[3,5] = 1 -> Y0Diff[3,5] <= -1
 LinkY0LessThan1[3,6]: Y0LessThan[3,6] = 1 -> Y0Diff[3,6] <= -1
 LinkY0LessThan1[3,7]: Y0LessThan[3,7] = 1 -> Y0Diff[3,7] <= -1
 LinkY0LessThan1[3,8]: Y0LessThan[3,8] = 1 -> Y0Diff[3,8] <= -1
 LinkY0LessThan1[3,9]: Y0LessThan[3,9] = 1 -> Y0Diff[3,9] <= -1
 LinkY0LessThan1[3,10]: Y0LessThan[3,10] = 1 -> Y0Diff[3,10] <= -1
 LinkY0LessThan1[4,0]: Y0LessThan[4,0] = 1 -> Y0Diff[4,0] <= -1
 LinkY0LessThan1[4,1]: Y0LessThan[4,1] = 1 -> Y0Diff[4,1] <= -1
 LinkY0LessThan1[4,2]: Y0LessThan[4,2] = 1 -> Y0Diff[4,2] <= -1
 LinkY0LessThan1[4,3]: Y0LessThan[4,3] = 1 -> Y0Diff[4,3] <= -1
 LinkY0LessThan1[4,4]: Y0LessThan[4,4] = 1 -> Y0Diff[4,4] <= -1
 LinkY0LessThan1[4,5]: Y0LessThan[4,5] = 1 -> Y0Diff[4,5] <= -1
 LinkY0LessThan1[4,6]: Y0LessThan[4,6] = 1 -> Y0Diff[4,6] <= -1
 LinkY0LessThan1[4,7]: Y0LessThan[4,7] = 1 -> Y0Diff[4,7] <= -1
 LinkY0LessThan1[4,8]: Y0LessThan[4,8] = 1 -> Y0Diff[4,8] <= -1
 LinkY0LessThan1[4,9]: Y0LessThan[4,9] = 1 -> Y0Diff[4,9] <= -1
 LinkY0LessThan1[4,10]: Y0LessThan[4,10] = 1 -> Y0Diff[4,10] <= -1
 LinkY0LessThan1[5,0]: Y0LessThan[5,0] = 1 -> Y0Diff[5,0] <= -1
 LinkY0LessThan1[5,1]: Y0LessThan[5,1] = 1 -> Y0Diff[5,1] <= -1
 LinkY0LessThan1[5,2]: Y0LessThan[5,2] = 1 -> Y0Diff[5,2] <= -1
 LinkY0LessThan1[5,3]: Y0LessThan[5,3] = 1 -> Y0Diff[5,3] <= -1
 LinkY0LessThan1[5,4]: Y0LessThan[5,4] = 1 -> Y0Diff[5,4] <= -1
 LinkY0LessThan1[5,5]: Y0LessThan[5,5] = 1 -> Y0Diff[5,5] <= -1
 LinkY0LessThan1[5,6]: Y0LessThan[5,6] = 1 -> Y0Diff[5,6] <= -1
 LinkY0LessThan1[5,7]: Y0LessThan[5,7] = 1 -> Y0Diff[5,7] <= -1
 LinkY0LessThan1[5,8]: Y0LessThan[5,8] = 1 -> Y0Diff[5,8] <= -1
 LinkY0LessThan1[5,9]: Y0LessThan[5,9] = 1 -> Y0Diff[5,9] <= -1
 LinkY0LessThan1[5,10]: Y0LessThan[5,10] = 1 -> Y0Diff[5,10] <= -1
 LinkY0LessThan1[6,0]: Y0LessThan[6,0] = 1 -> Y0Diff[6,0] <= -1
 LinkY0LessThan1[6,1]: Y0LessThan[6,1] = 1 -> Y0Diff[6,1] <= -1
 LinkY0LessThan1[6,2]: Y0LessThan[6,2] = 1 -> Y0Diff[6,2] <= -1
 LinkY0LessThan1[6,3]: Y0LessThan[6,3] = 1 -> Y0Diff[6,3] <= -1
 LinkY0LessThan1[6,4]: Y0LessThan[6,4] = 1 -> Y0Diff[6,4] <= -1
 LinkY0LessThan1[6,5]: Y0LessThan[6,5] = 1 -> Y0Diff[6,5] <= -1
 LinkY0LessThan1[6,6]: Y0LessThan[6,6] = 1 -> Y0Diff[6,6] <= -1
 LinkY0LessThan1[6,7]: Y0LessThan[6,7] = 1 -> Y0Diff[6,7] <= -1
 LinkY0LessThan1[6,8]: Y0LessThan[6,8] = 1 -> Y0Diff[6,8] <= -1
 LinkY0LessThan1[6,9]: Y0LessThan[6,9] = 1 -> Y0Diff[6,9] <= -1
 LinkY0LessThan1[6,10]: Y0LessThan[6,10] = 1 -> Y0Diff[6,10] <= -1
 LinkY0LessThan1[7,0]: Y0LessThan[7,0] = 1 -> Y0Diff[7,0] <= -1
 LinkY0LessThan1[7,1]: Y0LessThan[7,1] = 1 -> Y0Diff[7,1] <= -1
 LinkY0LessThan1[7,2]: Y0LessThan[7,2] = 1 -> Y0Diff[7,2] <= -1
 LinkY0LessThan1[7,3]: Y0LessThan[7,3] = 1 -> Y0Diff[7,3] <= -1
 LinkY0LessThan1[7,4]: Y0LessThan[7,4] = 1 -> Y0Diff[7,4] <= -1
 LinkY0LessThan1[7,5]: Y0LessThan[7,5] = 1 -> Y0Diff[7,5] <= -1
 LinkY0LessThan1[7,6]: Y0LessThan[7,6] = 1 -> Y0Diff[7,6] <= -1
 LinkY0LessThan1[7,7]: Y0LessThan[7,7] = 1 -> Y0Diff[7,7] <= -1
 LinkY0LessThan1[7,8]: Y0LessThan[7,8] = 1 -> Y0Diff[7,8] <= -1
 LinkY0LessThan1[7,9]: Y0LessThan[7,9] = 1 -> Y0Diff[7,9] <= -1
 LinkY0LessThan1[7,10]: Y0LessThan[7,10] = 1 -> Y0Diff[7,10] <= -1
 LinkY0LessThan1[8,0]: Y0LessThan[8,0] = 1 -> Y0Diff[8,0] <= -1
 LinkY0LessThan1[8,1]: Y0LessThan[8,1] = 1 -> Y0Diff[8,1] <= -1
 LinkY0LessThan1[8,2]: Y0LessThan[8,2] = 1 -> Y0Diff[8,2] <= -1
 LinkY0LessThan1[8,3]: Y0LessThan[8,3] = 1 -> Y0Diff[8,3] <= -1
 LinkY0LessThan1[8,4]: Y0LessThan[8,4] = 1 -> Y0Diff[8,4] <= -1
 LinkY0LessThan1[8,5]: Y0LessThan[8,5] = 1 -> Y0Diff[8,5] <= -1
 LinkY0LessThan1[8,6]: Y0LessThan[8,6] = 1 -> Y0Diff[8,6] <= -1
 LinkY0LessThan1[8,7]: Y0LessThan[8,7] = 1 -> Y0Diff[8,7] <= -1
 LinkY0LessThan1[8,8]: Y0LessThan[8,8] = 1 -> Y0Diff[8,8] <= -1
 LinkY0LessThan1[8,9]: Y0LessThan[8,9] = 1 -> Y0Diff[8,9] <= -1
 LinkY0LessThan1[8,10]: Y0LessThan[8,10] = 1 -> Y0Diff[8,10] <= -1
 LinkY0LessThan1[9,0]: Y0LessThan[9,0] = 1 -> Y0Diff[9,0] <= -1
 LinkY0LessThan1[9,1]: Y0LessThan[9,1] = 1 -> Y0Diff[9,1] <= -1
 LinkY0LessThan1[9,2]: Y0LessThan[9,2] = 1 -> Y0Diff[9,2] <= -1
 LinkY0LessThan1[9,3]: Y0LessThan[9,3] = 1 -> Y0Diff[9,3] <= -1
 LinkY0LessThan1[9,4]: Y0LessThan[9,4] = 1 -> Y0Diff[9,4] <= -1
 LinkY0LessThan1[9,5]: Y0LessThan[9,5] = 1 -> Y0Diff[9,5] <= -1
 LinkY0LessThan1[9,6]: Y0LessThan[9,6] = 1 -> Y0Diff[9,6] <= -1
 LinkY0LessThan1[9,7]: Y0LessThan[9,7] = 1 -> Y0Diff[9,7] <= -1
 LinkY0LessThan1[9,8]: Y0LessThan[9,8] = 1 -> Y0Diff[9,8] <= -1
 LinkY0LessThan1[9,9]: Y0LessThan[9,9] = 1 -> Y0Diff[9,9] <= -1
 LinkY0LessThan1[9,10]: Y0LessThan[9,10] = 1 -> Y0Diff[9,10] <= -1
 LinkY0LessThan1[10,0]: Y0LessThan[10,0] = 1 -> Y0Diff[10,0] <= -1
 LinkY0LessThan1[10,1]: Y0LessThan[10,1] = 1 -> Y0Diff[10,1] <= -1
 LinkY0LessThan1[10,2]: Y0LessThan[10,2] = 1 -> Y0Diff[10,2] <= -1
 LinkY0LessThan1[10,3]: Y0LessThan[10,3] = 1 -> Y0Diff[10,3] <= -1
 LinkY0LessThan1[10,4]: Y0LessThan[10,4] = 1 -> Y0Diff[10,4] <= -1
 LinkY0LessThan1[10,5]: Y0LessThan[10,5] = 1 -> Y0Diff[10,5] <= -1
 LinkY0LessThan1[10,6]: Y0LessThan[10,6] = 1 -> Y0Diff[10,6] <= -1
 LinkY0LessThan1[10,7]: Y0LessThan[10,7] = 1 -> Y0Diff[10,7] <= -1
 LinkY0LessThan1[10,8]: Y0LessThan[10,8] = 1 -> Y0Diff[10,8] <= -1
 LinkY0LessThan1[10,9]: Y0LessThan[10,9] = 1 -> Y0Diff[10,9] <= -1
 LinkY0LessThan1[10,10]: Y0LessThan[10,10] = 1 -> Y0Diff[10,10] <= -1
 LinkY0LessThan2[0,0]: Y0LessThan[0,0] = 0 -> Y0Diff[0,0] >= -0
 LinkY0LessThan2[0,1]: Y0LessThan[0,1] = 0 -> Y0Diff[0,1] >= -0
 LinkY0LessThan2[0,2]: Y0LessThan[0,2] = 0 -> Y0Diff[0,2] >= -0
 LinkY0LessThan2[0,3]: Y0LessThan[0,3] = 0 -> Y0Diff[0,3] >= -0
 LinkY0LessThan2[0,4]: Y0LessThan[0,4] = 0 -> Y0Diff[0,4] >= -0
 LinkY0LessThan2[0,5]: Y0LessThan[0,5] = 0 -> Y0Diff[0,5] >= -0
 LinkY0LessThan2[0,6]: Y0LessThan[0,6] = 0 -> Y0Diff[0,6] >= -0
 LinkY0LessThan2[0,7]: Y0LessThan[0,7] = 0 -> Y0Diff[0,7] >= -0
 LinkY0LessThan2[0,8]: Y0LessThan[0,8] = 0 -> Y0Diff[0,8] >= -0
 LinkY0LessThan2[0,9]: Y0LessThan[0,9] = 0 -> Y0Diff[0,9] >= -0
 LinkY0LessThan2[0,10]: Y0LessThan[0,10] = 0 -> Y0Diff[0,10] >= -0
 LinkY0LessThan2[1,0]: Y0LessThan[1,0] = 0 -> Y0Diff[1,0] >= -0
 LinkY0LessThan2[1,1]: Y0LessThan[1,1] = 0 -> Y0Diff[1,1] >= -0
 LinkY0LessThan2[1,2]: Y0LessThan[1,2] = 0 -> Y0Diff[1,2] >= -0
 LinkY0LessThan2[1,3]: Y0LessThan[1,3] = 0 -> Y0Diff[1,3] >= -0
 LinkY0LessThan2[1,4]: Y0LessThan[1,4] = 0 -> Y0Diff[1,4] >= -0
 LinkY0LessThan2[1,5]: Y0LessThan[1,5] = 0 -> Y0Diff[1,5] >= -0
 LinkY0LessThan2[1,6]: Y0LessThan[1,6] = 0 -> Y0Diff[1,6] >= -0
 LinkY0LessThan2[1,7]: Y0LessThan[1,7] = 0 -> Y0Diff[1,7] >= -0
 LinkY0LessThan2[1,8]: Y0LessThan[1,8] = 0 -> Y0Diff[1,8] >= -0
 LinkY0LessThan2[1,9]: Y0LessThan[1,9] = 0 -> Y0Diff[1,9] >= -0
 LinkY0LessThan2[1,10]: Y0LessThan[1,10] = 0 -> Y0Diff[1,10] >= -0
 LinkY0LessThan2[2,0]: Y0LessThan[2,0] = 0 -> Y0Diff[2,0] >= -0
 LinkY0LessThan2[2,1]: Y0LessThan[2,1] = 0 -> Y0Diff[2,1] >= -0
 LinkY0LessThan2[2,2]: Y0LessThan[2,2] = 0 -> Y0Diff[2,2] >= -0
 LinkY0LessThan2[2,3]: Y0LessThan[2,3] = 0 -> Y0Diff[2,3] >= -0
 LinkY0LessThan2[2,4]: Y0LessThan[2,4] = 0 -> Y0Diff[2,4] >= -0
 LinkY0LessThan2[2,5]: Y0LessThan[2,5] = 0 -> Y0Diff[2,5] >= -0
 LinkY0LessThan2[2,6]: Y0LessThan[2,6] = 0 -> Y0Diff[2,6] >= -0
 LinkY0LessThan2[2,7]: Y0LessThan[2,7] = 0 -> Y0Diff[2,7] >= -0
 LinkY0LessThan2[2,8]: Y0LessThan[2,8] = 0 -> Y0Diff[2,8] >= -0
 LinkY0LessThan2[2,9]: Y0LessThan[2,9] = 0 -> Y0Diff[2,9] >= -0
 LinkY0LessThan2[2,10]: Y0LessThan[2,10] = 0 -> Y0Diff[2,10] >= -0
 LinkY0LessThan2[3,0]: Y0LessThan[3,0] = 0 -> Y0Diff[3,0] >= -0
 LinkY0LessThan2[3,1]: Y0LessThan[3,1] = 0 -> Y0Diff[3,1] >= -0
 LinkY0LessThan2[3,2]: Y0LessThan[3,2] = 0 -> Y0Diff[3,2] >= -0
 LinkY0LessThan2[3,3]: Y0LessThan[3,3] = 0 -> Y0Diff[3,3] >= -0
 LinkY0LessThan2[3,4]: Y0LessThan[3,4] = 0 -> Y0Diff[3,4] >= -0
 LinkY0LessThan2[3,5]: Y0LessThan[3,5] = 0 -> Y0Diff[3,5] >= -0
 LinkY0LessThan2[3,6]: Y0LessThan[3,6] = 0 -> Y0Diff[3,6] >= -0
 LinkY0LessThan2[3,7]: Y0LessThan[3,7] = 0 -> Y0Diff[3,7] >= -0
 LinkY0LessThan2[3,8]: Y0LessThan[3,8] = 0 -> Y0Diff[3,8] >= -0
 LinkY0LessThan2[3,9]: Y0LessThan[3,9] = 0 -> Y0Diff[3,9] >= -0
 LinkY0LessThan2[3,10]: Y0LessThan[3,10] = 0 -> Y0Diff[3,10] >= -0
 LinkY0LessThan2[4,0]: Y0LessThan[4,0] = 0 -> Y0Diff[4,0] >= -0
 LinkY0LessThan2[4,1]: Y0LessThan[4,1] = 0 -> Y0Diff[4,1] >= -0
 LinkY0LessThan2[4,2]: Y0LessThan[4,2] = 0 -> Y0Diff[4,2] >= -0
 LinkY0LessThan2[4,3]: Y0LessThan[4,3] = 0 -> Y0Diff[4,3] >= -0
 LinkY0LessThan2[4,4]: Y0LessThan[4,4] = 0 -> Y0Diff[4,4] >= -0
 LinkY0LessThan2[4,5]: Y0LessThan[4,5] = 0 -> Y0Diff[4,5] >= -0
 LinkY0LessThan2[4,6]: Y0LessThan[4,6] = 0 -> Y0Diff[4,6] >= -0
 LinkY0LessThan2[4,7]: Y0LessThan[4,7] = 0 -> Y0Diff[4,7] >= -0
 LinkY0LessThan2[4,8]: Y0LessThan[4,8] = 0 -> Y0Diff[4,8] >= -0
 LinkY0LessThan2[4,9]: Y0LessThan[4,9] = 0 -> Y0Diff[4,9] >= -0
 LinkY0LessThan2[4,10]: Y0LessThan[4,10] = 0 -> Y0Diff[4,10] >= -0
 LinkY0LessThan2[5,0]: Y0LessThan[5,0] = 0 -> Y0Diff[5,0] >= -0
 LinkY0LessThan2[5,1]: Y0LessThan[5,1] = 0 -> Y0Diff[5,1] >= -0
 LinkY0LessThan2[5,2]: Y0LessThan[5,2] = 0 -> Y0Diff[5,2] >= -0
 LinkY0LessThan2[5,3]: Y0LessThan[5,3] = 0 -> Y0Diff[5,3] >= -0
 LinkY0LessThan2[5,4]: Y0LessThan[5,4] = 0 -> Y0Diff[5,4] >= -0
 LinkY0LessThan2[5,5]: Y0LessThan[5,5] = 0 -> Y0Diff[5,5] >= -0
 LinkY0LessThan2[5,6]: Y0LessThan[5,6] = 0 -> Y0Diff[5,6] >= -0
 LinkY0LessThan2[5,7]: Y0LessThan[5,7] = 0 -> Y0Diff[5,7] >= -0
 LinkY0LessThan2[5,8]: Y0LessThan[5,8] = 0 -> Y0Diff[5,8] >= -0
 LinkY0LessThan2[5,9]: Y0LessThan[5,9] = 0 -> Y0Diff[5,9] >= -0
 LinkY0LessThan2[5,10]: Y0LessThan[5,10] = 0 -> Y0Diff[5,10] >= -0
 LinkY0LessThan2[6,0]: Y0LessThan[6,0] = 0 -> Y0Diff[6,0] >= -0
 LinkY0LessThan2[6,1]: Y0LessThan[6,1] = 0 -> Y0Diff[6,1] >= -0
 LinkY0LessThan2[6,2]: Y0LessThan[6,2] = 0 -> Y0Diff[6,2] >= -0
 LinkY0LessThan2[6,3]: Y0LessThan[6,3] = 0 -> Y0Diff[6,3] >= -0
 LinkY0LessThan2[6,4]: Y0LessThan[6,4] = 0 -> Y0Diff[6,4] >= -0
 LinkY0LessThan2[6,5]: Y0LessThan[6,5] = 0 -> Y0Diff[6,5] >= -0
 LinkY0LessThan2[6,6]: Y0LessThan[6,6] = 0 -> Y0Diff[6,6] >= -0
 LinkY0LessThan2[6,7]: Y0LessThan[6,7] = 0 -> Y0Diff[6,7] >= -0
 LinkY0LessThan2[6,8]: Y0LessThan[6,8] = 0 -> Y0Diff[6,8] >= -0
 LinkY0LessThan2[6,9]: Y0LessThan[6,9] = 0 -> Y0Diff[6,9] >= -0
 LinkY0LessThan2[6,10]: Y0LessThan[6,10] = 0 -> Y0Diff[6,10] >= -0
 LinkY0LessThan2[7,0]: Y0LessThan[7,0] = 0 -> Y0Diff[7,0] >= -0
 LinkY0LessThan2[7,1]: Y0LessThan[7,1] = 0 -> Y0Diff[7,1] >= -0
 LinkY0LessThan2[7,2]: Y0LessThan[7,2] = 0 -> Y0Diff[7,2] >= -0
 LinkY0LessThan2[7,3]: Y0LessThan[7,3] = 0 -> Y0Diff[7,3] >= -0
 LinkY0LessThan2[7,4]: Y0LessThan[7,4] = 0 -> Y0Diff[7,4] >= -0
 LinkY0LessThan2[7,5]: Y0LessThan[7,5] = 0 -> Y0Diff[7,5] >= -0
 LinkY0LessThan2[7,6]: Y0LessThan[7,6] = 0 -> Y0Diff[7,6] >= -0
 LinkY0LessThan2[7,7]: Y0LessThan[7,7] = 0 -> Y0Diff[7,7] >= -0
 LinkY0LessThan2[7,8]: Y0LessThan[7,8] = 0 -> Y0Diff[7,8] >= -0
 LinkY0LessThan2[7,9]: Y0LessThan[7,9] = 0 -> Y0Diff[7,9] >= -0
 LinkY0LessThan2[7,10]: Y0LessThan[7,10] = 0 -> Y0Diff[7,10] >= -0
 LinkY0LessThan2[8,0]: Y0LessThan[8,0] = 0 -> Y0Diff[8,0] >= -0
 LinkY0LessThan2[8,1]: Y0LessThan[8,1] = 0 -> Y0Diff[8,1] >= -0
 LinkY0LessThan2[8,2]: Y0LessThan[8,2] = 0 -> Y0Diff[8,2] >= -0
 LinkY0LessThan2[8,3]: Y0LessThan[8,3] = 0 -> Y0Diff[8,3] >= -0
 LinkY0LessThan2[8,4]: Y0LessThan[8,4] = 0 -> Y0Diff[8,4] >= -0
 LinkY0LessThan2[8,5]: Y0LessThan[8,5] = 0 -> Y0Diff[8,5] >= -0
 LinkY0LessThan2[8,6]: Y0LessThan[8,6] = 0 -> Y0Diff[8,6] >= -0
 LinkY0LessThan2[8,7]: Y0LessThan[8,7] = 0 -> Y0Diff[8,7] >= -0
 LinkY0LessThan2[8,8]: Y0LessThan[8,8] = 0 -> Y0Diff[8,8] >= -0
 LinkY0LessThan2[8,9]: Y0LessThan[8,9] = 0 -> Y0Diff[8,9] >= -0
 LinkY0LessThan2[8,10]: Y0LessThan[8,10] = 0 -> Y0Diff[8,10] >= -0
 LinkY0LessThan2[9,0]: Y0LessThan[9,0] = 0 -> Y0Diff[9,0] >= -0
 LinkY0LessThan2[9,1]: Y0LessThan[9,1] = 0 -> Y0Diff[9,1] >= -0
 LinkY0LessThan2[9,2]: Y0LessThan[9,2] = 0 -> Y0Diff[9,2] >= -0
 LinkY0LessThan2[9,3]: Y0LessThan[9,3] = 0 -> Y0Diff[9,3] >= -0
 LinkY0LessThan2[9,4]: Y0LessThan[9,4] = 0 -> Y0Diff[9,4] >= -0
 LinkY0LessThan2[9,5]: Y0LessThan[9,5] = 0 -> Y0Diff[9,5] >= -0
 LinkY0LessThan2[9,6]: Y0LessThan[9,6] = 0 -> Y0Diff[9,6] >= -0
 LinkY0LessThan2[9,7]: Y0LessThan[9,7] = 0 -> Y0Diff[9,7] >= -0
 LinkY0LessThan2[9,8]: Y0LessThan[9,8] = 0 -> Y0Diff[9,8] >= -0
 LinkY0LessThan2[9,9]: Y0LessThan[9,9] = 0 -> Y0Diff[9,9] >= -0
 LinkY0LessThan2[9,10]: Y0LessThan[9,10] = 0 -> Y0Diff[9,10] >= -0
 LinkY0LessThan2[10,0]: Y0LessThan[10,0] = 0 -> Y0Diff[10,0] >= -0
 LinkY0LessThan2[10,1]: Y0LessThan[10,1] = 0 -> Y0Diff[10,1] >= -0
 LinkY0LessThan2[10,2]: Y0LessThan[10,2] = 0 -> Y0Diff[10,2] >= -0
 LinkY0LessThan2[10,3]: Y0LessThan[10,3] = 0 -> Y0Diff[10,3] >= -0
 LinkY0LessThan2[10,4]: Y0LessThan[10,4] = 0 -> Y0Diff[10,4] >= -0
 LinkY0LessThan2[10,5]: Y0LessThan[10,5] = 0 -> Y0Diff[10,5] >= -0
 LinkY0LessThan2[10,6]: Y0LessThan[10,6] = 0 -> Y0Diff[10,6] >= -0
 LinkY0LessThan2[10,7]: Y0LessThan[10,7] = 0 -> Y0Diff[10,7] >= -0
 LinkY0LessThan2[10,8]: Y0LessThan[10,8] = 0 -> Y0Diff[10,8] >= -0
 LinkY0LessThan2[10,9]: Y0LessThan[10,9] = 0 -> Y0Diff[10,9] >= -0
 LinkY0LessThan2[10,10]: Y0LessThan[10,10] = 0 -> Y0Diff[10,10] >= -0
 LinkX1LessThan1[0,0]: X1LessThan[0,0] = 1 -> X1Diff[0,0] <= -1
 LinkX1LessThan1[0,1]: X1LessThan[0,1] = 1 -> X1Diff[0,1] <= -1
 LinkX1LessThan1[0,2]: X1LessThan[0,2] = 1 -> X1Diff[0,2] <= -1
 LinkX1LessThan1[0,3]: X1LessThan[0,3] = 1 -> X1Diff[0,3] <= -1
 LinkX1LessThan1[0,4]: X1LessThan[0,4] = 1 -> X1Diff[0,4] <= -1
 LinkX1LessThan1[0,5]: X1LessThan[0,5] = 1 -> X1Diff[0,5] <= -1
 LinkX1LessThan1[0,6]: X1LessThan[0,6] = 1 -> X1Diff[0,6] <= -1
 LinkX1LessThan1[0,7]: X1LessThan[0,7] = 1 -> X1Diff[0,7] <= -1
 LinkX1LessThan1[0,8]: X1LessThan[0,8] = 1 -> X1Diff[0,8] <= -1
 LinkX1LessThan1[0,9]: X1LessThan[0,9] = 1 -> X1Diff[0,9] <= -1
 LinkX1LessThan1[0,10]: X1LessThan[0,10] = 1 -> X1Diff[0,10] <= -1
 LinkX1LessThan1[1,0]: X1LessThan[1,0] = 1 -> X1Diff[1,0] <= -1
 LinkX1LessThan1[1,1]: X1LessThan[1,1] = 1 -> X1Diff[1,1] <= -1
 LinkX1LessThan1[1,2]: X1LessThan[1,2] = 1 -> X1Diff[1,2] <= -1
 LinkX1LessThan1[1,3]: X1LessThan[1,3] = 1 -> X1Diff[1,3] <= -1
 LinkX1LessThan1[1,4]: X1LessThan[1,4] = 1 -> X1Diff[1,4] <= -1
 LinkX1LessThan1[1,5]: X1LessThan[1,5] = 1 -> X1Diff[1,5] <= -1
 LinkX1LessThan1[1,6]: X1LessThan[1,6] = 1 -> X1Diff[1,6] <= -1
 LinkX1LessThan1[1,7]: X1LessThan[1,7] = 1 -> X1Diff[1,7] <= -1
 LinkX1LessThan1[1,8]: X1LessThan[1,8] = 1 -> X1Diff[1,8] <= -1
 LinkX1LessThan1[1,9]: X1LessThan[1,9] = 1 -> X1Diff[1,9] <= -1
 LinkX1LessThan1[1,10]: X1LessThan[1,10] = 1 -> X1Diff[1,10] <= -1
 LinkX1LessThan1[2,0]: X1LessThan[2,0] = 1 -> X1Diff[2,0] <= -1
 LinkX1LessThan1[2,1]: X1LessThan[2,1] = 1 -> X1Diff[2,1] <= -1
 LinkX1LessThan1[2,2]: X1LessThan[2,2] = 1 -> X1Diff[2,2] <= -1
 LinkX1LessThan1[2,3]: X1LessThan[2,3] = 1 -> X1Diff[2,3] <= -1
 LinkX1LessThan1[2,4]: X1LessThan[2,4] = 1 -> X1Diff[2,4] <= -1
 LinkX1LessThan1[2,5]: X1LessThan[2,5] = 1 -> X1Diff[2,5] <= -1
 LinkX1LessThan1[2,6]: X1LessThan[2,6] = 1 -> X1Diff[2,6] <= -1
 LinkX1LessThan1[2,7]: X1LessThan[2,7] = 1 -> X1Diff[2,7] <= -1
 LinkX1LessThan1[2,8]: X1LessThan[2,8] = 1 -> X1Diff[2,8] <= -1
 LinkX1LessThan1[2,9]: X1LessThan[2,9] = 1 -> X1Diff[2,9] <= -1
 LinkX1LessThan1[2,10]: X1LessThan[2,10] = 1 -> X1Diff[2,10] <= -1
 LinkX1LessThan1[3,0]: X1LessThan[3,0] = 1 -> X1Diff[3,0] <= -1
 LinkX1LessThan1[3,1]: X1LessThan[3,1] = 1 -> X1Diff[3,1] <= -1
 LinkX1LessThan1[3,2]: X1LessThan[3,2] = 1 -> X1Diff[3,2] <= -1
 LinkX1LessThan1[3,3]: X1LessThan[3,3] = 1 -> X1Diff[3,3] <= -1
 LinkX1LessThan1[3,4]: X1LessThan[3,4] = 1 -> X1Diff[3,4] <= -1
 LinkX1LessThan1[3,5]: X1LessThan[3,5] = 1 -> X1Diff[3,5] <= -1
 LinkX1LessThan1[3,6]: X1LessThan[3,6] = 1 -> X1Diff[3,6] <= -1
 LinkX1LessThan1[3,7]: X1LessThan[3,7] = 1 -> X1Diff[3,7] <= -1
 LinkX1LessThan1[3,8]: X1LessThan[3,8] = 1 -> X1Diff[3,8] <= -1
 LinkX1LessThan1[3,9]: X1LessThan[3,9] = 1 -> X1Diff[3,9] <= -1
 LinkX1LessThan1[3,10]: X1LessThan[3,10] = 1 -> X1Diff[3,10] <= -1
 LinkX1LessThan1[4,0]: X1LessThan[4,0] = 1 -> X1Diff[4,0] <= -1
 LinkX1LessThan1[4,1]: X1LessThan[4,1] = 1 -> X1Diff[4,1] <= -1
 LinkX1LessThan1[4,2]: X1LessThan[4,2] = 1 -> X1Diff[4,2] <= -1
 LinkX1LessThan1[4,3]: X1LessThan[4,3] = 1 -> X1Diff[4,3] <= -1
 LinkX1LessThan1[4,4]: X1LessThan[4,4] = 1 -> X1Diff[4,4] <= -1
 LinkX1LessThan1[4,5]: X1LessThan[4,5] = 1 -> X1Diff[4,5] <= -1
 LinkX1LessThan1[4,6]: X1LessThan[4,6] = 1 -> X1Diff[4,6] <= -1
 LinkX1LessThan1[4,7]: X1LessThan[4,7] = 1 -> X1Diff[4,7] <= -1
 LinkX1LessThan1[4,8]: X1LessThan[4,8] = 1 -> X1Diff[4,8] <= -1
 LinkX1LessThan1[4,9]: X1LessThan[4,9] = 1 -> X1Diff[4,9] <= -1
 LinkX1LessThan1[4,10]: X1LessThan[4,10] = 1 -> X1Diff[4,10] <= -1
 LinkX1LessThan1[5,0]: X1LessThan[5,0] = 1 -> X1Diff[5,0] <= -1
 LinkX1LessThan1[5,1]: X1LessThan[5,1] = 1 -> X1Diff[5,1] <= -1
 LinkX1LessThan1[5,2]: X1LessThan[5,2] = 1 -> X1Diff[5,2] <= -1
 LinkX1LessThan1[5,3]: X1LessThan[5,3] = 1 -> X1Diff[5,3] <= -1
 LinkX1LessThan1[5,4]: X1LessThan[5,4] = 1 -> X1Diff[5,4] <= -1
 LinkX1LessThan1[5,5]: X1LessThan[5,5] = 1 -> X1Diff[5,5] <= -1
 LinkX1LessThan1[5,6]: X1LessThan[5,6] = 1 -> X1Diff[5,6] <= -1
 LinkX1LessThan1[5,7]: X1LessThan[5,7] = 1 -> X1Diff[5,7] <= -1
 LinkX1LessThan1[5,8]: X1LessThan[5,8] = 1 -> X1Diff[5,8] <= -1
 LinkX1LessThan1[5,9]: X1LessThan[5,9] = 1 -> X1Diff[5,9] <= -1
 LinkX1LessThan1[5,10]: X1LessThan[5,10] = 1 -> X1Diff[5,10] <= -1
 LinkX1LessThan1[6,0]: X1LessThan[6,0] = 1 -> X1Diff[6,0] <= -1
 LinkX1LessThan1[6,1]: X1LessThan[6,1] = 1 -> X1Diff[6,1] <= -1
 LinkX1LessThan1[6,2]: X1LessThan[6,2] = 1 -> X1Diff[6,2] <= -1
 LinkX1LessThan1[6,3]: X1LessThan[6,3] = 1 -> X1Diff[6,3] <= -1
 LinkX1LessThan1[6,4]: X1LessThan[6,4] = 1 -> X1Diff[6,4] <= -1
 LinkX1LessThan1[6,5]: X1LessThan[6,5] = 1 -> X1Diff[6,5] <= -1
 LinkX1LessThan1[6,6]: X1LessThan[6,6] = 1 -> X1Diff[6,6] <= -1
 LinkX1LessThan1[6,7]: X1LessThan[6,7] = 1 -> X1Diff[6,7] <= -1
 LinkX1LessThan1[6,8]: X1LessThan[6,8] = 1 -> X1Diff[6,8] <= -1
 LinkX1LessThan1[6,9]: X1LessThan[6,9] = 1 -> X1Diff[6,9] <= -1
 LinkX1LessThan1[6,10]: X1LessThan[6,10] = 1 -> X1Diff[6,10] <= -1
 LinkX1LessThan1[7,0]: X1LessThan[7,0] = 1 -> X1Diff[7,0] <= -1
 LinkX1LessThan1[7,1]: X1LessThan[7,1] = 1 -> X1Diff[7,1] <= -1
 LinkX1LessThan1[7,2]: X1LessThan[7,2] = 1 -> X1Diff[7,2] <= -1
 LinkX1LessThan1[7,3]: X1LessThan[7,3] = 1 -> X1Diff[7,3] <= -1
 LinkX1LessThan1[7,4]: X1LessThan[7,4] = 1 -> X1Diff[7,4] <= -1
 LinkX1LessThan1[7,5]: X1LessThan[7,5] = 1 -> X1Diff[7,5] <= -1
 LinkX1LessThan1[7,6]: X1LessThan[7,6] = 1 -> X1Diff[7,6] <= -1
 LinkX1LessThan1[7,7]: X1LessThan[7,7] = 1 -> X1Diff[7,7] <= -1
 LinkX1LessThan1[7,8]: X1LessThan[7,8] = 1 -> X1Diff[7,8] <= -1
 LinkX1LessThan1[7,9]: X1LessThan[7,9] = 1 -> X1Diff[7,9] <= -1
 LinkX1LessThan1[7,10]: X1LessThan[7,10] = 1 -> X1Diff[7,10] <= -1
 LinkX1LessThan1[8,0]: X1LessThan[8,0] = 1 -> X1Diff[8,0] <= -1
 LinkX1LessThan1[8,1]: X1LessThan[8,1] = 1 -> X1Diff[8,1] <= -1
 LinkX1LessThan1[8,2]: X1LessThan[8,2] = 1 -> X1Diff[8,2] <= -1
 LinkX1LessThan1[8,3]: X1LessThan[8,3] = 1 -> X1Diff[8,3] <= -1
 LinkX1LessThan1[8,4]: X1LessThan[8,4] = 1 -> X1Diff[8,4] <= -1
 LinkX1LessThan1[8,5]: X1LessThan[8,5] = 1 -> X1Diff[8,5] <= -1
 LinkX1LessThan1[8,6]: X1LessThan[8,6] = 1 -> X1Diff[8,6] <= -1
 LinkX1LessThan1[8,7]: X1LessThan[8,7] = 1 -> X1Diff[8,7] <= -1
 LinkX1LessThan1[8,8]: X1LessThan[8,8] = 1 -> X1Diff[8,8] <= -1
 LinkX1LessThan1[8,9]: X1LessThan[8,9] = 1 -> X1Diff[8,9] <= -1
 LinkX1LessThan1[8,10]: X1LessThan[8,10] = 1 -> X1Diff[8,10] <= -1
 LinkX1LessThan1[9,0]: X1LessThan[9,0] = 1 -> X1Diff[9,0] <= -1
 LinkX1LessThan1[9,1]: X1LessThan[9,1] = 1 -> X1Diff[9,1] <= -1
 LinkX1LessThan1[9,2]: X1LessThan[9,2] = 1 -> X1Diff[9,2] <= -1
 LinkX1LessThan1[9,3]: X1LessThan[9,3] = 1 -> X1Diff[9,3] <= -1
 LinkX1LessThan1[9,4]: X1LessThan[9,4] = 1 -> X1Diff[9,4] <= -1
 LinkX1LessThan1[9,5]: X1LessThan[9,5] = 1 -> X1Diff[9,5] <= -1
 LinkX1LessThan1[9,6]: X1LessThan[9,6] = 1 -> X1Diff[9,6] <= -1
 LinkX1LessThan1[9,7]: X1LessThan[9,7] = 1 -> X1Diff[9,7] <= -1
 LinkX1LessThan1[9,8]: X1LessThan[9,8] = 1 -> X1Diff[9,8] <= -1
 LinkX1LessThan1[9,9]: X1LessThan[9,9] = 1 -> X1Diff[9,9] <= -1
 LinkX1LessThan1[9,10]: X1LessThan[9,10] = 1 -> X1Diff[9,10] <= -1
 LinkX1LessThan1[10,0]: X1LessThan[10,0] = 1 -> X1Diff[10,0] <= -1
 LinkX1LessThan1[10,1]: X1LessThan[10,1] = 1 -> X1Diff[10,1] <= -1
 LinkX1LessThan1[10,2]: X1LessThan[10,2] = 1 -> X1Diff[10,2] <= -1
 LinkX1LessThan1[10,3]: X1LessThan[10,3] = 1 -> X1Diff[10,3] <= -1
 LinkX1LessThan1[10,4]: X1LessThan[10,4] = 1 -> X1Diff[10,4] <= -1
 LinkX1LessThan1[10,5]: X1LessThan[10,5] = 1 -> X1Diff[10,5] <= -1
 LinkX1LessThan1[10,6]: X1LessThan[10,6] = 1 -> X1Diff[10,6] <= -1
 LinkX1LessThan1[10,7]: X1LessThan[10,7] = 1 -> X1Diff[10,7] <= -1
 LinkX1LessThan1[10,8]: X1LessThan[10,8] = 1 -> X1Diff[10,8] <= -1
 LinkX1LessThan1[10,9]: X1LessThan[10,9] = 1 -> X1Diff[10,9] <= -1
 LinkX1LessThan1[10,10]: X1LessThan[10,10] = 1 -> X1Diff[10,10] <= -1
 LinkX1LessThan2[0,0]: X1LessThan[0,0] = 0 -> X1Diff[0,0] >= -0
 LinkX1LessThan2[0,1]: X1LessThan[0,1] = 0 -> X1Diff[0,1] >= -0
 LinkX1LessThan2[0,2]: X1LessThan[0,2] = 0 -> X1Diff[0,2] >= -0
 LinkX1LessThan2[0,3]: X1LessThan[0,3] = 0 -> X1Diff[0,3] >= -0
 LinkX1LessThan2[0,4]: X1LessThan[0,4] = 0 -> X1Diff[0,4] >= -0
 LinkX1LessThan2[0,5]: X1LessThan[0,5] = 0 -> X1Diff[0,5] >= -0
 LinkX1LessThan2[0,6]: X1LessThan[0,6] = 0 -> X1Diff[0,6] >= -0
 LinkX1LessThan2[0,7]: X1LessThan[0,7] = 0 -> X1Diff[0,7] >= -0
 LinkX1LessThan2[0,8]: X1LessThan[0,8] = 0 -> X1Diff[0,8] >= -0
 LinkX1LessThan2[0,9]: X1LessThan[0,9] = 0 -> X1Diff[0,9] >= -0
 LinkX1LessThan2[0,10]: X1LessThan[0,10] = 0 -> X1Diff[0,10] >= -0
 LinkX1LessThan2[1,0]: X1LessThan[1,0] = 0 -> X1Diff[1,0] >= -0
 LinkX1LessThan2[1,1]: X1LessThan[1,1] = 0 -> X1Diff[1,1] >= -0
 LinkX1LessThan2[1,2]: X1LessThan[1,2] = 0 -> X1Diff[1,2] >= -0
 LinkX1LessThan2[1,3]: X1LessThan[1,3] = 0 -> X1Diff[1,3] >= -0
 LinkX1LessThan2[1,4]: X1LessThan[1,4] = 0 -> X1Diff[1,4] >= -0
 LinkX1LessThan2[1,5]: X1LessThan[1,5] = 0 -> X1Diff[1,5] >= -0
 LinkX1LessThan2[1,6]: X1LessThan[1,6] = 0 -> X1Diff[1,6] >= -0
 LinkX1LessThan2[1,7]: X1LessThan[1,7] = 0 -> X1Diff[1,7] >= -0
 LinkX1LessThan2[1,8]: X1LessThan[1,8] = 0 -> X1Diff[1,8] >= -0
 LinkX1LessThan2[1,9]: X1LessThan[1,9] = 0 -> X1Diff[1,9] >= -0
 LinkX1LessThan2[1,10]: X1LessThan[1,10] = 0 -> X1Diff[1,10] >= -0
 LinkX1LessThan2[2,0]: X1LessThan[2,0] = 0 -> X1Diff[2,0] >= -0
 LinkX1LessThan2[2,1]: X1LessThan[2,1] = 0 -> X1Diff[2,1] >= -0
 LinkX1LessThan2[2,2]: X1LessThan[2,2] = 0 -> X1Diff[2,2] >= -0
 LinkX1LessThan2[2,3]: X1LessThan[2,3] = 0 -> X1Diff[2,3] >= -0
 LinkX1LessThan2[2,4]: X1LessThan[2,4] = 0 -> X1Diff[2,4] >= -0
 LinkX1LessThan2[2,5]: X1LessThan[2,5] = 0 -> X1Diff[2,5] >= -0
 LinkX1LessThan2[2,6]: X1LessThan[2,6] = 0 -> X1Diff[2,6] >= -0
 LinkX1LessThan2[2,7]: X1LessThan[2,7] = 0 -> X1Diff[2,7] >= -0
 LinkX1LessThan2[2,8]: X1LessThan[2,8] = 0 -> X1Diff[2,8] >= -0
 LinkX1LessThan2[2,9]: X1LessThan[2,9] = 0 -> X1Diff[2,9] >= -0
 LinkX1LessThan2[2,10]: X1LessThan[2,10] = 0 -> X1Diff[2,10] >= -0
 LinkX1LessThan2[3,0]: X1LessThan[3,0] = 0 -> X1Diff[3,0] >= -0
 LinkX1LessThan2[3,1]: X1LessThan[3,1] = 0 -> X1Diff[3,1] >= -0
 LinkX1LessThan2[3,2]: X1LessThan[3,2] = 0 -> X1Diff[3,2] >= -0
 LinkX1LessThan2[3,3]: X1LessThan[3,3] = 0 -> X1Diff[3,3] >= -0
 LinkX1LessThan2[3,4]: X1LessThan[3,4] = 0 -> X1Diff[3,4] >= -0
 LinkX1LessThan2[3,5]: X1LessThan[3,5] = 0 -> X1Diff[3,5] >= -0
 LinkX1LessThan2[3,6]: X1LessThan[3,6] = 0 -> X1Diff[3,6] >= -0
 LinkX1LessThan2[3,7]: X1LessThan[3,7] = 0 -> X1Diff[3,7] >= -0
 LinkX1LessThan2[3,8]: X1LessThan[3,8] = 0 -> X1Diff[3,8] >= -0
 LinkX1LessThan2[3,9]: X1LessThan[3,9] = 0 -> X1Diff[3,9] >= -0
 LinkX1LessThan2[3,10]: X1LessThan[3,10] = 0 -> X1Diff[3,10] >= -0
 LinkX1LessThan2[4,0]: X1LessThan[4,0] = 0 -> X1Diff[4,0] >= -0
 LinkX1LessThan2[4,1]: X1LessThan[4,1] = 0 -> X1Diff[4,1] >= -0
 LinkX1LessThan2[4,2]: X1LessThan[4,2] = 0 -> X1Diff[4,2] >= -0
 LinkX1LessThan2[4,3]: X1LessThan[4,3] = 0 -> X1Diff[4,3] >= -0
 LinkX1LessThan2[4,4]: X1LessThan[4,4] = 0 -> X1Diff[4,4] >= -0
 LinkX1LessThan2[4,5]: X1LessThan[4,5] = 0 -> X1Diff[4,5] >= -0
 LinkX1LessThan2[4,6]: X1LessThan[4,6] = 0 -> X1Diff[4,6] >= -0
 LinkX1LessThan2[4,7]: X1LessThan[4,7] = 0 -> X1Diff[4,7] >= -0
 LinkX1LessThan2[4,8]: X1LessThan[4,8] = 0 -> X1Diff[4,8] >= -0
 LinkX1LessThan2[4,9]: X1LessThan[4,9] = 0 -> X1Diff[4,9] >= -0
 LinkX1LessThan2[4,10]: X1LessThan[4,10] = 0 -> X1Diff[4,10] >= -0
 LinkX1LessThan2[5,0]: X1LessThan[5,0] = 0 -> X1Diff[5,0] >= -0
 LinkX1LessThan2[5,1]: X1LessThan[5,1] = 0 -> X1Diff[5,1] >= -0
 LinkX1LessThan2[5,2]: X1LessThan[5,2] = 0 -> X1Diff[5,2] >= -0
 LinkX1LessThan2[5,3]: X1LessThan[5,3] = 0 -> X1Diff[5,3] >= -0
 LinkX1LessThan2[5,4]: X1LessThan[5,4] = 0 -> X1Diff[5,4] >= -0
 LinkX1LessThan2[5,5]: X1LessThan[5,5] = 0 -> X1Diff[5,5] >= -0
 LinkX1LessThan2[5,6]: X1LessThan[5,6] = 0 -> X1Diff[5,6] >= -0
 LinkX1LessThan2[5,7]: X1LessThan[5,7] = 0 -> X1Diff[5,7] >= -0
 LinkX1LessThan2[5,8]: X1LessThan[5,8] = 0 -> X1Diff[5,8] >= -0
 LinkX1LessThan2[5,9]: X1LessThan[5,9] = 0 -> X1Diff[5,9] >= -0
 LinkX1LessThan2[5,10]: X1LessThan[5,10] = 0 -> X1Diff[5,10] >= -0
 LinkX1LessThan2[6,0]: X1LessThan[6,0] = 0 -> X1Diff[6,0] >= -0
 LinkX1LessThan2[6,1]: X1LessThan[6,1] = 0 -> X1Diff[6,1] >= -0
 LinkX1LessThan2[6,2]: X1LessThan[6,2] = 0 -> X1Diff[6,2] >= -0
 LinkX1LessThan2[6,3]: X1LessThan[6,3] = 0 -> X1Diff[6,3] >= -0
 LinkX1LessThan2[6,4]: X1LessThan[6,4] = 0 -> X1Diff[6,4] >= -0
 LinkX1LessThan2[6,5]: X1LessThan[6,5] = 0 -> X1Diff[6,5] >= -0
 LinkX1LessThan2[6,6]: X1LessThan[6,6] = 0 -> X1Diff[6,6] >= -0
 LinkX1LessThan2[6,7]: X1LessThan[6,7] = 0 -> X1Diff[6,7] >= -0
 LinkX1LessThan2[6,8]: X1LessThan[6,8] = 0 -> X1Diff[6,8] >= -0
 LinkX1LessThan2[6,9]: X1LessThan[6,9] = 0 -> X1Diff[6,9] >= -0
 LinkX1LessThan2[6,10]: X1LessThan[6,10] = 0 -> X1Diff[6,10] >= -0
 LinkX1LessThan2[7,0]: X1LessThan[7,0] = 0 -> X1Diff[7,0] >= -0
 LinkX1LessThan2[7,1]: X1LessThan[7,1] = 0 -> X1Diff[7,1] >= -0
 LinkX1LessThan2[7,2]: X1LessThan[7,2] = 0 -> X1Diff[7,2] >= -0
 LinkX1LessThan2[7,3]: X1LessThan[7,3] = 0 -> X1Diff[7,3] >= -0
 LinkX1LessThan2[7,4]: X1LessThan[7,4] = 0 -> X1Diff[7,4] >= -0
 LinkX1LessThan2[7,5]: X1LessThan[7,5] = 0 -> X1Diff[7,5] >= -0
 LinkX1LessThan2[7,6]: X1LessThan[7,6] = 0 -> X1Diff[7,6] >= -0
 LinkX1LessThan2[7,7]: X1LessThan[7,7] = 0 -> X1Diff[7,7] >= -0
 LinkX1LessThan2[7,8]: X1LessThan[7,8] = 0 -> X1Diff[7,8] >= -0
 LinkX1LessThan2[7,9]: X1LessThan[7,9] = 0 -> X1Diff[7,9] >= -0
 LinkX1LessThan2[7,10]: X1LessThan[7,10] = 0 -> X1Diff[7,10] >= -0
 LinkX1LessThan2[8,0]: X1LessThan[8,0] = 0 -> X1Diff[8,0] >= -0
 LinkX1LessThan2[8,1]: X1LessThan[8,1] = 0 -> X1Diff[8,1] >= -0
 LinkX1LessThan2[8,2]: X1LessThan[8,2] = 0 -> X1Diff[8,2] >= -0
 LinkX1LessThan2[8,3]: X1LessThan[8,3] = 0 -> X1Diff[8,3] >= -0
 LinkX1LessThan2[8,4]: X1LessThan[8,4] = 0 -> X1Diff[8,4] >= -0
 LinkX1LessThan2[8,5]: X1LessThan[8,5] = 0 -> X1Diff[8,5] >= -0
 LinkX1LessThan2[8,6]: X1LessThan[8,6] = 0 -> X1Diff[8,6] >= -0
 LinkX1LessThan2[8,7]: X1LessThan[8,7] = 0 -> X1Diff[8,7] >= -0
 LinkX1LessThan2[8,8]: X1LessThan[8,8] = 0 -> X1Diff[8,8] >= -0
 LinkX1LessThan2[8,9]: X1LessThan[8,9] = 0 -> X1Diff[8,9] >= -0
 LinkX1LessThan2[8,10]: X1LessThan[8,10] = 0 -> X1Diff[8,10] >= -0
 LinkX1LessThan2[9,0]: X1LessThan[9,0] = 0 -> X1Diff[9,0] >= -0
 LinkX1LessThan2[9,1]: X1LessThan[9,1] = 0 -> X1Diff[9,1] >= -0
 LinkX1LessThan2[9,2]: X1LessThan[9,2] = 0 -> X1Diff[9,2] >= -0
 LinkX1LessThan2[9,3]: X1LessThan[9,3] = 0 -> X1Diff[9,3] >= -0
 LinkX1LessThan2[9,4]: X1LessThan[9,4] = 0 -> X1Diff[9,4] >= -0
 LinkX1LessThan2[9,5]: X1LessThan[9,5] = 0 -> X1Diff[9,5] >= -0
 LinkX1LessThan2[9,6]: X1LessThan[9,6] = 0 -> X1Diff[9,6] >= -0
 LinkX1LessThan2[9,7]: X1LessThan[9,7] = 0 -> X1Diff[9,7] >= -0
 LinkX1LessThan2[9,8]: X1LessThan[9,8] = 0 -> X1Diff[9,8] >= -0
 LinkX1LessThan2[9,9]: X1LessThan[9,9] = 0 -> X1Diff[9,9] >= -0
 LinkX1LessThan2[9,10]: X1LessThan[9,10] = 0 -> X1Diff[9,10] >= -0
 LinkX1LessThan2[10,0]: X1LessThan[10,0] = 0 -> X1Diff[10,0] >= -0
 LinkX1LessThan2[10,1]: X1LessThan[10,1] = 0 -> X1Diff[10,1] >= -0
 LinkX1LessThan2[10,2]: X1LessThan[10,2] = 0 -> X1Diff[10,2] >= -0
 LinkX1LessThan2[10,3]: X1LessThan[10,3] = 0 -> X1Diff[10,3] >= -0
 LinkX1LessThan2[10,4]: X1LessThan[10,4] = 0 -> X1Diff[10,4] >= -0
 LinkX1LessThan2[10,5]: X1LessThan[10,5] = 0 -> X1Diff[10,5] >= -0
 LinkX1LessThan2[10,6]: X1LessThan[10,6] = 0 -> X1Diff[10,6] >= -0
 LinkX1LessThan2[10,7]: X1LessThan[10,7] = 0 -> X1Diff[10,7] >= -0
 LinkX1LessThan2[10,8]: X1LessThan[10,8] = 0 -> X1Diff[10,8] >= -0
 LinkX1LessThan2[10,9]: X1LessThan[10,9] = 0 -> X1Diff[10,9] >= -0
 LinkX1LessThan2[10,10]: X1LessThan[10,10] = 0 -> X1Diff[10,10] >= -0
 LinkY1LessThan1[0,0]: Y1LessThan[0,0] = 1 -> Y1Diff[0,0] <= -1
 LinkY1LessThan1[0,1]: Y1LessThan[0,1] = 1 -> Y1Diff[0,1] <= -1
 LinkY1LessThan1[0,2]: Y1LessThan[0,2] = 1 -> Y1Diff[0,2] <= -1
 LinkY1LessThan1[0,3]: Y1LessThan[0,3] = 1 -> Y1Diff[0,3] <= -1
 LinkY1LessThan1[0,4]: Y1LessThan[0,4] = 1 -> Y1Diff[0,4] <= -1
 LinkY1LessThan1[0,5]: Y1LessThan[0,5] = 1 -> Y1Diff[0,5] <= -1
 LinkY1LessThan1[0,6]: Y1LessThan[0,6] = 1 -> Y1Diff[0,6] <= -1
 LinkY1LessThan1[0,7]: Y1LessThan[0,7] = 1 -> Y1Diff[0,7] <= -1
 LinkY1LessThan1[0,8]: Y1LessThan[0,8] = 1 -> Y1Diff[0,8] <= -1
 LinkY1LessThan1[0,9]: Y1LessThan[0,9] = 1 -> Y1Diff[0,9] <= -1
 LinkY1LessThan1[0,10]: Y1LessThan[0,10] = 1 -> Y1Diff[0,10] <= -1
 LinkY1LessThan1[1,0]: Y1LessThan[1,0] = 1 -> Y1Diff[1,0] <= -1
 LinkY1LessThan1[1,1]: Y1LessThan[1,1] = 1 -> Y1Diff[1,1] <= -1
 LinkY1LessThan1[1,2]: Y1LessThan[1,2] = 1 -> Y1Diff[1,2] <= -1
 LinkY1LessThan1[1,3]: Y1LessThan[1,3] = 1 -> Y1Diff[1,3] <= -1
 LinkY1LessThan1[1,4]: Y1LessThan[1,4] = 1 -> Y1Diff[1,4] <= -1
 LinkY1LessThan1[1,5]: Y1LessThan[1,5] = 1 -> Y1Diff[1,5] <= -1
 LinkY1LessThan1[1,6]: Y1LessThan[1,6] = 1 -> Y1Diff[1,6] <= -1
 LinkY1LessThan1[1,7]: Y1LessThan[1,7] = 1 -> Y1Diff[1,7] <= -1
 LinkY1LessThan1[1,8]: Y1LessThan[1,8] = 1 -> Y1Diff[1,8] <= -1
 LinkY1LessThan1[1,9]: Y1LessThan[1,9] = 1 -> Y1Diff[1,9] <= -1
 LinkY1LessThan1[1,10]: Y1LessThan[1,10] = 1 -> Y1Diff[1,10] <= -1
 LinkY1LessThan1[2,0]: Y1LessThan[2,0] = 1 -> Y1Diff[2,0] <= -1
 LinkY1LessThan1[2,1]: Y1LessThan[2,1] = 1 -> Y1Diff[2,1] <= -1
 LinkY1LessThan1[2,2]: Y1LessThan[2,2] = 1 -> Y1Diff[2,2] <= -1
 LinkY1LessThan1[2,3]: Y1LessThan[2,3] = 1 -> Y1Diff[2,3] <= -1
 LinkY1LessThan1[2,4]: Y1LessThan[2,4] = 1 -> Y1Diff[2,4] <= -1
 LinkY1LessThan1[2,5]: Y1LessThan[2,5] = 1 -> Y1Diff[2,5] <= -1
 LinkY1LessThan1[2,6]: Y1LessThan[2,6] = 1 -> Y1Diff[2,6] <= -1
 LinkY1LessThan1[2,7]: Y1LessThan[2,7] = 1 -> Y1Diff[2,7] <= -1
 LinkY1LessThan1[2,8]: Y1LessThan[2,8] = 1 -> Y1Diff[2,8] <= -1
 LinkY1LessThan1[2,9]: Y1LessThan[2,9] = 1 -> Y1Diff[2,9] <= -1
 LinkY1LessThan1[2,10]: Y1LessThan[2,10] = 1 -> Y1Diff[2,10] <= -1
 LinkY1LessThan1[3,0]: Y1LessThan[3,0] = 1 -> Y1Diff[3,0] <= -1
 LinkY1LessThan1[3,1]: Y1LessThan[3,1] = 1 -> Y1Diff[3,1] <= -1
 LinkY1LessThan1[3,2]: Y1LessThan[3,2] = 1 -> Y1Diff[3,2] <= -1
 LinkY1LessThan1[3,3]: Y1LessThan[3,3] = 1 -> Y1Diff[3,3] <= -1
 LinkY1LessThan1[3,4]: Y1LessThan[3,4] = 1 -> Y1Diff[3,4] <= -1
 LinkY1LessThan1[3,5]: Y1LessThan[3,5] = 1 -> Y1Diff[3,5] <= -1
 LinkY1LessThan1[3,6]: Y1LessThan[3,6] = 1 -> Y1Diff[3,6] <= -1
 LinkY1LessThan1[3,7]: Y1LessThan[3,7] = 1 -> Y1Diff[3,7] <= -1
 LinkY1LessThan1[3,8]: Y1LessThan[3,8] = 1 -> Y1Diff[3,8] <= -1
 LinkY1LessThan1[3,9]: Y1LessThan[3,9] = 1 -> Y1Diff[3,9] <= -1
 LinkY1LessThan1[3,10]: Y1LessThan[3,10] = 1 -> Y1Diff[3,10] <= -1
 LinkY1LessThan1[4,0]: Y1LessThan[4,0] = 1 -> Y1Diff[4,0] <= -1
 LinkY1LessThan1[4,1]: Y1LessThan[4,1] = 1 -> Y1Diff[4,1] <= -1
 LinkY1LessThan1[4,2]: Y1LessThan[4,2] = 1 -> Y1Diff[4,2] <= -1
 LinkY1LessThan1[4,3]: Y1LessThan[4,3] = 1 -> Y1Diff[4,3] <= -1
 LinkY1LessThan1[4,4]: Y1LessThan[4,4] = 1 -> Y1Diff[4,4] <= -1
 LinkY1LessThan1[4,5]: Y1LessThan[4,5] = 1 -> Y1Diff[4,5] <= -1
 LinkY1LessThan1[4,6]: Y1LessThan[4,6] = 1 -> Y1Diff[4,6] <= -1
 LinkY1LessThan1[4,7]: Y1LessThan[4,7] = 1 -> Y1Diff[4,7] <= -1
 LinkY1LessThan1[4,8]: Y1LessThan[4,8] = 1 -> Y1Diff[4,8] <= -1
 LinkY1LessThan1[4,9]: Y1LessThan[4,9] = 1 -> Y1Diff[4,9] <= -1
 LinkY1LessThan1[4,10]: Y1LessThan[4,10] = 1 -> Y1Diff[4,10] <= -1
 LinkY1LessThan1[5,0]: Y1LessThan[5,0] = 1 -> Y1Diff[5,0] <= -1
 LinkY1LessThan1[5,1]: Y1LessThan[5,1] = 1 -> Y1Diff[5,1] <= -1
 LinkY1LessThan1[5,2]: Y1LessThan[5,2] = 1 -> Y1Diff[5,2] <= -1
 LinkY1LessThan1[5,3]: Y1LessThan[5,3] = 1 -> Y1Diff[5,3] <= -1
 LinkY1LessThan1[5,4]: Y1LessThan[5,4] = 1 -> Y1Diff[5,4] <= -1
 LinkY1LessThan1[5,5]: Y1LessThan[5,5] = 1 -> Y1Diff[5,5] <= -1
 LinkY1LessThan1[5,6]: Y1LessThan[5,6] = 1 -> Y1Diff[5,6] <= -1
 LinkY1LessThan1[5,7]: Y1LessThan[5,7] = 1 -> Y1Diff[5,7] <= -1
 LinkY1LessThan1[5,8]: Y1LessThan[5,8] = 1 -> Y1Diff[5,8] <= -1
 LinkY1LessThan1[5,9]: Y1LessThan[5,9] = 1 -> Y1Diff[5,9] <= -1
 LinkY1LessThan1[5,10]: Y1LessThan[5,10] = 1 -> Y1Diff[5,10] <= -1
 LinkY1LessThan1[6,0]: Y1LessThan[6,0] = 1 -> Y1Diff[6,0] <= -1
 LinkY1LessThan1[6,1]: Y1LessThan[6,1] = 1 -> Y1Diff[6,1] <= -1
 LinkY1LessThan1[6,2]: Y1LessThan[6,2] = 1 -> Y1Diff[6,2] <= -1
 LinkY1LessThan1[6,3]: Y1LessThan[6,3] = 1 -> Y1Diff[6,3] <= -1
 LinkY1LessThan1[6,4]: Y1LessThan[6,4] = 1 -> Y1Diff[6,4] <= -1
 LinkY1LessThan1[6,5]: Y1LessThan[6,5] = 1 -> Y1Diff[6,5] <= -1
 LinkY1LessThan1[6,6]: Y1LessThan[6,6] = 1 -> Y1Diff[6,6] <= -1
 LinkY1LessThan1[6,7]: Y1LessThan[6,7] = 1 -> Y1Diff[6,7] <= -1
 LinkY1LessThan1[6,8]: Y1LessThan[6,8] = 1 -> Y1Diff[6,8] <= -1
 LinkY1LessThan1[6,9]: Y1LessThan[6,9] = 1 -> Y1Diff[6,9] <= -1
 LinkY1LessThan1[6,10]: Y1LessThan[6,10] = 1 -> Y1Diff[6,10] <= -1
 LinkY1LessThan1[7,0]: Y1LessThan[7,0] = 1 -> Y1Diff[7,0] <= -1
 LinkY1LessThan1[7,1]: Y1LessThan[7,1] = 1 -> Y1Diff[7,1] <= -1
 LinkY1LessThan1[7,2]: Y1LessThan[7,2] = 1 -> Y1Diff[7,2] <= -1
 LinkY1LessThan1[7,3]: Y1LessThan[7,3] = 1 -> Y1Diff[7,3] <= -1
 LinkY1LessThan1[7,4]: Y1LessThan[7,4] = 1 -> Y1Diff[7,4] <= -1
 LinkY1LessThan1[7,5]: Y1LessThan[7,5] = 1 -> Y1Diff[7,5] <= -1
 LinkY1LessThan1[7,6]: Y1LessThan[7,6] = 1 -> Y1Diff[7,6] <= -1
 LinkY1LessThan1[7,7]: Y1LessThan[7,7] = 1 -> Y1Diff[7,7] <= -1
 LinkY1LessThan1[7,8]: Y1LessThan[7,8] = 1 -> Y1Diff[7,8] <= -1
 LinkY1LessThan1[7,9]: Y1LessThan[7,9] = 1 -> Y1Diff[7,9] <= -1
 LinkY1LessThan1[7,10]: Y1LessThan[7,10] = 1 -> Y1Diff[7,10] <= -1
 LinkY1LessThan1[8,0]: Y1LessThan[8,0] = 1 -> Y1Diff[8,0] <= -1
 LinkY1LessThan1[8,1]: Y1LessThan[8,1] = 1 -> Y1Diff[8,1] <= -1
 LinkY1LessThan1[8,2]: Y1LessThan[8,2] = 1 -> Y1Diff[8,2] <= -1
 LinkY1LessThan1[8,3]: Y1LessThan[8,3] = 1 -> Y1Diff[8,3] <= -1
 LinkY1LessThan1[8,4]: Y1LessThan[8,4] = 1 -> Y1Diff[8,4] <= -1
 LinkY1LessThan1[8,5]: Y1LessThan[8,5] = 1 -> Y1Diff[8,5] <= -1
 LinkY1LessThan1[8,6]: Y1LessThan[8,6] = 1 -> Y1Diff[8,6] <= -1
 LinkY1LessThan1[8,7]: Y1LessThan[8,7] = 1 -> Y1Diff[8,7] <= -1
 LinkY1LessThan1[8,8]: Y1LessThan[8,8] = 1 -> Y1Diff[8,8] <= -1
 LinkY1LessThan1[8,9]: Y1LessThan[8,9] = 1 -> Y1Diff[8,9] <= -1
 LinkY1LessThan1[8,10]: Y1LessThan[8,10] = 1 -> Y1Diff[8,10] <= -1
 LinkY1LessThan1[9,0]: Y1LessThan[9,0] = 1 -> Y1Diff[9,0] <= -1
 LinkY1LessThan1[9,1]: Y1LessThan[9,1] = 1 -> Y1Diff[9,1] <= -1
 LinkY1LessThan1[9,2]: Y1LessThan[9,2] = 1 -> Y1Diff[9,2] <= -1
 LinkY1LessThan1[9,3]: Y1LessThan[9,3] = 1 -> Y1Diff[9,3] <= -1
 LinkY1LessThan1[9,4]: Y1LessThan[9,4] = 1 -> Y1Diff[9,4] <= -1
 LinkY1LessThan1[9,5]: Y1LessThan[9,5] = 1 -> Y1Diff[9,5] <= -1
 LinkY1LessThan1[9,6]: Y1LessThan[9,6] = 1 -> Y1Diff[9,6] <= -1
 LinkY1LessThan1[9,7]: Y1LessThan[9,7] = 1 -> Y1Diff[9,7] <= -1
 LinkY1LessThan1[9,8]: Y1LessThan[9,8] = 1 -> Y1Diff[9,8] <= -1
 LinkY1LessThan1[9,9]: Y1LessThan[9,9] = 1 -> Y1Diff[9,9] <= -1
 LinkY1LessThan1[9,10]: Y1LessThan[9,10] = 1 -> Y1Diff[9,10] <= -1
 LinkY1LessThan1[10,0]: Y1LessThan[10,0] = 1 -> Y1Diff[10,0] <= -1
 LinkY1LessThan1[10,1]: Y1LessThan[10,1] = 1 -> Y1Diff[10,1] <= -1
 LinkY1LessThan1[10,2]: Y1LessThan[10,2] = 1 -> Y1Diff[10,2] <= -1
 LinkY1LessThan1[10,3]: Y1LessThan[10,3] = 1 -> Y1Diff[10,3] <= -1
 LinkY1LessThan1[10,4]: Y1LessThan[10,4] = 1 -> Y1Diff[10,4] <= -1
 LinkY1LessThan1[10,5]: Y1LessThan[10,5] = 1 -> Y1Diff[10,5] <= -1
 LinkY1LessThan1[10,6]: Y1LessThan[10,6] = 1 -> Y1Diff[10,6] <= -1
 LinkY1LessThan1[10,7]: Y1LessThan[10,7] = 1 -> Y1Diff[10,7] <= -1
 LinkY1LessThan1[10,8]: Y1LessThan[10,8] = 1 -> Y1Diff[10,8] <= -1
 LinkY1LessThan1[10,9]: Y1LessThan[10,9] = 1 -> Y1Diff[10,9] <= -1
 LinkY1LessThan1[10,10]: Y1LessThan[10,10] = 1 -> Y1Diff[10,10] <= -1
 LinkY1LessThan2[0,0]: Y1LessThan[0,0] = 0 -> Y1Diff[0,0] >= -0
 LinkY1LessThan2[0,1]: Y1LessThan[0,1] = 0 -> Y1Diff[0,1] >= -0
 LinkY1LessThan2[0,2]: Y1LessThan[0,2] = 0 -> Y1Diff[0,2] >= -0
 LinkY1LessThan2[0,3]: Y1LessThan[0,3] = 0 -> Y1Diff[0,3] >= -0
 LinkY1LessThan2[0,4]: Y1LessThan[0,4] = 0 -> Y1Diff[0,4] >= -0
 LinkY1LessThan2[0,5]: Y1LessThan[0,5] = 0 -> Y1Diff[0,5] >= -0
 LinkY1LessThan2[0,6]: Y1LessThan[0,6] = 0 -> Y1Diff[0,6] >= -0
 LinkY1LessThan2[0,7]: Y1LessThan[0,7] = 0 -> Y1Diff[0,7] >= -0
 LinkY1LessThan2[0,8]: Y1LessThan[0,8] = 0 -> Y1Diff[0,8] >= -0
 LinkY1LessThan2[0,9]: Y1LessThan[0,9] = 0 -> Y1Diff[0,9] >= -0
 LinkY1LessThan2[0,10]: Y1LessThan[0,10] = 0 -> Y1Diff[0,10] >= -0
 LinkY1LessThan2[1,0]: Y1LessThan[1,0] = 0 -> Y1Diff[1,0] >= -0
 LinkY1LessThan2[1,1]: Y1LessThan[1,1] = 0 -> Y1Diff[1,1] >= -0
 LinkY1LessThan2[1,2]: Y1LessThan[1,2] = 0 -> Y1Diff[1,2] >= -0
 LinkY1LessThan2[1,3]: Y1LessThan[1,3] = 0 -> Y1Diff[1,3] >= -0
 LinkY1LessThan2[1,4]: Y1LessThan[1,4] = 0 -> Y1Diff[1,4] >= -0
 LinkY1LessThan2[1,5]: Y1LessThan[1,5] = 0 -> Y1Diff[1,5] >= -0
 LinkY1LessThan2[1,6]: Y1LessThan[1,6] = 0 -> Y1Diff[1,6] >= -0
 LinkY1LessThan2[1,7]: Y1LessThan[1,7] = 0 -> Y1Diff[1,7] >= -0
 LinkY1LessThan2[1,8]: Y1LessThan[1,8] = 0 -> Y1Diff[1,8] >= -0
 LinkY1LessThan2[1,9]: Y1LessThan[1,9] = 0 -> Y1Diff[1,9] >= -0
 LinkY1LessThan2[1,10]: Y1LessThan[1,10] = 0 -> Y1Diff[1,10] >= -0
 LinkY1LessThan2[2,0]: Y1LessThan[2,0] = 0 -> Y1Diff[2,0] >= -0
 LinkY1LessThan2[2,1]: Y1LessThan[2,1] = 0 -> Y1Diff[2,1] >= -0
 LinkY1LessThan2[2,2]: Y1LessThan[2,2] = 0 -> Y1Diff[2,2] >= -0
 LinkY1LessThan2[2,3]: Y1LessThan[2,3] = 0 -> Y1Diff[2,3] >= -0
 LinkY1LessThan2[2,4]: Y1LessThan[2,4] = 0 -> Y1Diff[2,4] >= -0
 LinkY1LessThan2[2,5]: Y1LessThan[2,5] = 0 -> Y1Diff[2,5] >= -0
 LinkY1LessThan2[2,6]: Y1LessThan[2,6] = 0 -> Y1Diff[2,6] >= -0
 LinkY1LessThan2[2,7]: Y1LessThan[2,7] = 0 -> Y1Diff[2,7] >= -0
 LinkY1LessThan2[2,8]: Y1LessThan[2,8] = 0 -> Y1Diff[2,8] >= -0
 LinkY1LessThan2[2,9]: Y1LessThan[2,9] = 0 -> Y1Diff[2,9] >= -0
 LinkY1LessThan2[2,10]: Y1LessThan[2,10] = 0 -> Y1Diff[2,10] >= -0
 LinkY1LessThan2[3,0]: Y1LessThan[3,0] = 0 -> Y1Diff[3,0] >= -0
 LinkY1LessThan2[3,1]: Y1LessThan[3,1] = 0 -> Y1Diff[3,1] >= -0
 LinkY1LessThan2[3,2]: Y1LessThan[3,2] = 0 -> Y1Diff[3,2] >= -0
 LinkY1LessThan2[3,3]: Y1LessThan[3,3] = 0 -> Y1Diff[3,3] >= -0
 LinkY1LessThan2[3,4]: Y1LessThan[3,4] = 0 -> Y1Diff[3,4] >= -0
 LinkY1LessThan2[3,5]: Y1LessThan[3,5] = 0 -> Y1Diff[3,5] >= -0
 LinkY1LessThan2[3,6]: Y1LessThan[3,6] = 0 -> Y1Diff[3,6] >= -0
 LinkY1LessThan2[3,7]: Y1LessThan[3,7] = 0 -> Y1Diff[3,7] >= -0
 LinkY1LessThan2[3,8]: Y1LessThan[3,8] = 0 -> Y1Diff[3,8] >= -0
 LinkY1LessThan2[3,9]: Y1LessThan[3,9] = 0 -> Y1Diff[3,9] >= -0
 LinkY1LessThan2[3,10]: Y1LessThan[3,10] = 0 -> Y1Diff[3,10] >= -0
 LinkY1LessThan2[4,0]: Y1LessThan[4,0] = 0 -> Y1Diff[4,0] >= -0
 LinkY1LessThan2[4,1]: Y1LessThan[4,1] = 0 -> Y1Diff[4,1] >= -0
 LinkY1LessThan2[4,2]: Y1LessThan[4,2] = 0 -> Y1Diff[4,2] >= -0
 LinkY1LessThan2[4,3]: Y1LessThan[4,3] = 0 -> Y1Diff[4,3] >= -0
 LinkY1LessThan2[4,4]: Y1LessThan[4,4] = 0 -> Y1Diff[4,4] >= -0
 LinkY1LessThan2[4,5]: Y1LessThan[4,5] = 0 -> Y1Diff[4,5] >= -0
 LinkY1LessThan2[4,6]: Y1LessThan[4,6] = 0 -> Y1Diff[4,6] >= -0
 LinkY1LessThan2[4,7]: Y1LessThan[4,7] = 0 -> Y1Diff[4,7] >= -0
 LinkY1LessThan2[4,8]: Y1LessThan[4,8] = 0 -> Y1Diff[4,8] >= -0
 LinkY1LessThan2[4,9]: Y1LessThan[4,9] = 0 -> Y1Diff[4,9] >= -0
 LinkY1LessThan2[4,10]: Y1LessThan[4,10] = 0 -> Y1Diff[4,10] >= -0
 LinkY1LessThan2[5,0]: Y1LessThan[5,0] = 0 -> Y1Diff[5,0] >= -0
 LinkY1LessThan2[5,1]: Y1LessThan[5,1] = 0 -> Y1Diff[5,1] >= -0
 LinkY1LessThan2[5,2]: Y1LessThan[5,2] = 0 -> Y1Diff[5,2] >= -0
 LinkY1LessThan2[5,3]: Y1LessThan[5,3] = 0 -> Y1Diff[5,3] >= -0
 LinkY1LessThan2[5,4]: Y1LessThan[5,4] = 0 -> Y1Diff[5,4] >= -0
 LinkY1LessThan2[5,5]: Y1LessThan[5,5] = 0 -> Y1Diff[5,5] >= -0
 LinkY1LessThan2[5,6]: Y1LessThan[5,6] = 0 -> Y1Diff[5,6] >= -0
 LinkY1LessThan2[5,7]: Y1LessThan[5,7] = 0 -> Y1Diff[5,7] >= -0
 LinkY1LessThan2[5,8]: Y1LessThan[5,8] = 0 -> Y1Diff[5,8] >= -0
 LinkY1LessThan2[5,9]: Y1LessThan[5,9] = 0 -> Y1Diff[5,9] >= -0
 LinkY1LessThan2[5,10]: Y1LessThan[5,10] = 0 -> Y1Diff[5,10] >= -0
 LinkY1LessThan2[6,0]: Y1LessThan[6,0] = 0 -> Y1Diff[6,0] >= -0
 LinkY1LessThan2[6,1]: Y1LessThan[6,1] = 0 -> Y1Diff[6,1] >= -0
 LinkY1LessThan2[6,2]: Y1LessThan[6,2] = 0 -> Y1Diff[6,2] >= -0
 LinkY1LessThan2[6,3]: Y1LessThan[6,3] = 0 -> Y1Diff[6,3] >= -0
 LinkY1LessThan2[6,4]: Y1LessThan[6,4] = 0 -> Y1Diff[6,4] >= -0
 LinkY1LessThan2[6,5]: Y1LessThan[6,5] = 0 -> Y1Diff[6,5] >= -0
 LinkY1LessThan2[6,6]: Y1LessThan[6,6] = 0 -> Y1Diff[6,6] >= -0
 LinkY1LessThan2[6,7]: Y1LessThan[6,7] = 0 -> Y1Diff[6,7] >= -0
 LinkY1LessThan2[6,8]: Y1LessThan[6,8] = 0 -> Y1Diff[6,8] >= -0
 LinkY1LessThan2[6,9]: Y1LessThan[6,9] = 0 -> Y1Diff[6,9] >= -0
 LinkY1LessThan2[6,10]: Y1LessThan[6,10] = 0 -> Y1Diff[6,10] >= -0
 LinkY1LessThan2[7,0]: Y1LessThan[7,0] = 0 -> Y1Diff[7,0] >= -0
 LinkY1LessThan2[7,1]: Y1LessThan[7,1] = 0 -> Y1Diff[7,1] >= -0
 LinkY1LessThan2[7,2]: Y1LessThan[7,2] = 0 -> Y1Diff[7,2] >= -0
 LinkY1LessThan2[7,3]: Y1LessThan[7,3] = 0 -> Y1Diff[7,3] >= -0
 LinkY1LessThan2[7,4]: Y1LessThan[7,4] = 0 -> Y1Diff[7,4] >= -0
 LinkY1LessThan2[7,5]: Y1LessThan[7,5] = 0 -> Y1Diff[7,5] >= -0
 LinkY1LessThan2[7,6]: Y1LessThan[7,6] = 0 -> Y1Diff[7,6] >= -0
 LinkY1LessThan2[7,7]: Y1LessThan[7,7] = 0 -> Y1Diff[7,7] >= -0
 LinkY1LessThan2[7,8]: Y1LessThan[7,8] = 0 -> Y1Diff[7,8] >= -0
 LinkY1LessThan2[7,9]: Y1LessThan[7,9] = 0 -> Y1Diff[7,9] >= -0
 LinkY1LessThan2[7,10]: Y1LessThan[7,10] = 0 -> Y1Diff[7,10] >= -0
 LinkY1LessThan2[8,0]: Y1LessThan[8,0] = 0 -> Y1Diff[8,0] >= -0
 LinkY1LessThan2[8,1]: Y1LessThan[8,1] = 0 -> Y1Diff[8,1] >= -0
 LinkY1LessThan2[8,2]: Y1LessThan[8,2] = 0 -> Y1Diff[8,2] >= -0
 LinkY1LessThan2[8,3]: Y1LessThan[8,3] = 0 -> Y1Diff[8,3] >= -0
 LinkY1LessThan2[8,4]: Y1LessThan[8,4] = 0 -> Y1Diff[8,4] >= -0
 LinkY1LessThan2[8,5]: Y1LessThan[8,5] = 0 -> Y1Diff[8,5] >= -0
 LinkY1LessThan2[8,6]: Y1LessThan[8,6] = 0 -> Y1Diff[8,6] >= -0
 LinkY1LessThan2[8,7]: Y1LessThan[8,7] = 0 -> Y1Diff[8,7] >= -0
 LinkY1LessThan2[8,8]: Y1LessThan[8,8] = 0 -> Y1Diff[8,8] >= -0
 LinkY1LessThan2[8,9]: Y1LessThan[8,9] = 0 -> Y1Diff[8,9] >= -0
 LinkY1LessThan2[8,10]: Y1LessThan[8,10] = 0 -> Y1Diff[8,10] >= -0
 LinkY1LessThan2[9,0]: Y1LessThan[9,0] = 0 -> Y1Diff[9,0] >= -0
 LinkY1LessThan2[9,1]: Y1LessThan[9,1] = 0 -> Y1Diff[9,1] >= -0
 LinkY1LessThan2[9,2]: Y1LessThan[9,2] = 0 -> Y1Diff[9,2] >= -0
 LinkY1LessThan2[9,3]: Y1LessThan[9,3] = 0 -> Y1Diff[9,3] >= -0
 LinkY1LessThan2[9,4]: Y1LessThan[9,4] = 0 -> Y1Diff[9,4] >= -0
 LinkY1LessThan2[9,5]: Y1LessThan[9,5] = 0 -> Y1Diff[9,5] >= -0
 LinkY1LessThan2[9,6]: Y1LessThan[9,6] = 0 -> Y1Diff[9,6] >= -0
 LinkY1LessThan2[9,7]: Y1LessThan[9,7] = 0 -> Y1Diff[9,7] >= -0
 LinkY1LessThan2[9,8]: Y1LessThan[9,8] = 0 -> Y1Diff[9,8] >= -0
 LinkY1LessThan2[9,9]: Y1LessThan[9,9] = 0 -> Y1Diff[9,9] >= -0
 LinkY1LessThan2[9,10]: Y1LessThan[9,10] = 0 -> Y1Diff[9,10] >= -0
 LinkY1LessThan2[10,0]: Y1LessThan[10,0] = 0 -> Y1Diff[10,0] >= -0
 LinkY1LessThan2[10,1]: Y1LessThan[10,1] = 0 -> Y1Diff[10,1] >= -0
 LinkY1LessThan2[10,2]: Y1LessThan[10,2] = 0 -> Y1Diff[10,2] >= -0
 LinkY1LessThan2[10,3]: Y1LessThan[10,3] = 0 -> Y1Diff[10,3] >= -0
 LinkY1LessThan2[10,4]: Y1LessThan[10,4] = 0 -> Y1Diff[10,4] >= -0
 LinkY1LessThan2[10,5]: Y1LessThan[10,5] = 0 -> Y1Diff[10,5] >= -0
 LinkY1LessThan2[10,6]: Y1LessThan[10,6] = 0 -> Y1Diff[10,6] >= -0
 LinkY1LessThan2[10,7]: Y1LessThan[10,7] = 0 -> Y1Diff[10,7] >= -0
 LinkY1LessThan2[10,8]: Y1LessThan[10,8] = 0 -> Y1Diff[10,8] >= -0
 LinkY1LessThan2[10,9]: Y1LessThan[10,9] = 0 -> Y1Diff[10,9] >= -0
 LinkY1LessThan2[10,10]: Y1LessThan[10,10] = 0 -> Y1Diff[10,10] >= -0
 LinkX0Group1[0,1]: X0LessThan[0,1] = 1 -> X0Group[0] - X0Group[1] <= -1
 LinkX0Group1[0,2]: X0LessThan[0,2] = 1 -> X0Group[0] - X0Group[2] <= -1
 LinkX0Group1[0,3]: X0LessThan[0,3] = 1 -> X0Group[0] - X0Group[3] <= -1
 LinkX0Group1[0,4]: X0LessThan[0,4] = 1 -> X0Group[0] - X0Group[4] <= -1
 LinkX0Group1[0,5]: X0LessThan[0,5] = 1 -> X0Group[0] - X0Group[5] <= -1
 LinkX0Group1[0,6]: X0LessThan[0,6] = 1 -> X0Group[0] - X0Group[6] <= -1
 LinkX0Group1[0,7]: X0LessThan[0,7] = 1 -> X0Group[0] - X0Group[7] <= -1
 LinkX0Group1[0,8]: X0LessThan[0,8] = 1 -> X0Group[0] - X0Group[8] <= -1
 LinkX0Group1[0,9]: X0LessThan[0,9] = 1 -> X0Group[0] - X0Group[9] <= -1
 LinkX0Group1[0,10]: X0LessThan[0,10] = 1 -> X0Group[0] - X0Group[10]
   <= -1
 LinkX0Group1[1,0]: X0LessThan[1,0] = 1 -> X0Group[1] - X0Group[0] <= -1
 LinkX0Group1[1,2]: X0LessThan[1,2] = 1 -> X0Group[1] - X0Group[2] <= -1
 LinkX0Group1[1,3]: X0LessThan[1,3] = 1 -> X0Group[1] - X0Group[3] <= -1
 LinkX0Group1[1,4]: X0LessThan[1,4] = 1 -> X0Group[1] - X0Group[4] <= -1
 LinkX0Group1[1,5]: X0LessThan[1,5] = 1 -> X0Group[1] - X0Group[5] <= -1
 LinkX0Group1[1,6]: X0LessThan[1,6] = 1 -> X0Group[1] - X0Group[6] <= -1
 LinkX0Group1[1,7]: X0LessThan[1,7] = 1 -> X0Group[1] - X0Group[7] <= -1
 LinkX0Group1[1,8]: X0LessThan[1,8] = 1 -> X0Group[1] - X0Group[8] <= -1
 LinkX0Group1[1,9]: X0LessThan[1,9] = 1 -> X0Group[1] - X0Group[9] <= -1
 LinkX0Group1[1,10]: X0LessThan[1,10] = 1 -> X0Group[1] - X0Group[10]
   <= -1
 LinkX0Group1[2,0]: X0LessThan[2,0] = 1 -> X0Group[2] - X0Group[0] <= -1
 LinkX0Group1[2,1]: X0LessThan[2,1] = 1 -> X0Group[2] - X0Group[1] <= -1
 LinkX0Group1[2,3]: X0LessThan[2,3] = 1 -> X0Group[2] - X0Group[3] <= -1
 LinkX0Group1[2,4]: X0LessThan[2,4] = 1 -> X0Group[2] - X0Group[4] <= -1
 LinkX0Group1[2,5]: X0LessThan[2,5] = 1 -> X0Group[2] - X0Group[5] <= -1
 LinkX0Group1[2,6]: X0LessThan[2,6] = 1 -> X0Group[2] - X0Group[6] <= -1
 LinkX0Group1[2,7]: X0LessThan[2,7] = 1 -> X0Group[2] - X0Group[7] <= -1
 LinkX0Group1[2,8]: X0LessThan[2,8] = 1 -> X0Group[2] - X0Group[8] <= -1
 LinkX0Group1[2,9]: X0LessThan[2,9] = 1 -> X0Group[2] - X0Group[9] <= -1
 LinkX0Group1[2,10]: X0LessThan[2,10] = 1 -> X0Group[2] - X0Group[10]
   <= -1
 LinkX0Group1[3,0]: X0LessThan[3,0] = 1 -> X0Group[3] - X0Group[0] <= -1
 LinkX0Group1[3,1]: X0LessThan[3,1] = 1 -> X0Group[3] - X0Group[1] <= -1
 LinkX0Group1[3,2]: X0LessThan[3,2] = 1 -> X0Group[3] - X0Group[2] <= -1
 LinkX0Group1[3,4]: X0LessThan[3,4] = 1 -> X0Group[3] - X0Group[4] <= -1
 LinkX0Group1[3,5]: X0LessThan[3,5] = 1 -> X0Group[3] - X0Group[5] <= -1
 LinkX0Group1[3,6]: X0LessThan[3,6] = 1 -> X0Group[3] - X0Group[6] <= -1
 LinkX0Group1[3,7]: X0LessThan[3,7] = 1 -> X0Group[3] - X0Group[7] <= -1
 LinkX0Group1[3,8]: X0LessThan[3,8] = 1 -> X0Group[3] - X0Group[8] <= -1
 LinkX0Group1[3,9]: X0LessThan[3,9] = 1 -> X0Group[3] - X0Group[9] <= -1
 LinkX0Group1[3,10]: X0LessThan[3,10] = 1 -> X0Group[3] - X0Group[10]
   <= -1
 LinkX0Group1[4,0]: X0LessThan[4,0] = 1 -> X0Group[4] - X0Group[0] <= -1
 LinkX0Group1[4,1]: X0LessThan[4,1] = 1 -> X0Group[4] - X0Group[1] <= -1
 LinkX0Group1[4,2]: X0LessThan[4,2] = 1 -> X0Group[4] - X0Group[2] <= -1
 LinkX0Group1[4,3]: X0LessThan[4,3] = 1 -> X0Group[4] - X0Group[3] <= -1
 LinkX0Group1[4,5]: X0LessThan[4,5] = 1 -> X0Group[4] - X0Group[5] <= -1
 LinkX0Group1[4,6]: X0LessThan[4,6] = 1 -> X0Group[4] - X0Group[6] <= -1
 LinkX0Group1[4,7]: X0LessThan[4,7] = 1 -> X0Group[4] - X0Group[7] <= -1
 LinkX0Group1[4,8]: X0LessThan[4,8] = 1 -> X0Group[4] - X0Group[8] <= -1
 LinkX0Group1[4,9]: X0LessThan[4,9] = 1 -> X0Group[4] - X0Group[9] <= -1
 LinkX0Group1[4,10]: X0LessThan[4,10] = 1 -> X0Group[4] - X0Group[10]
   <= -1
 LinkX0Group1[5,0]: X0LessThan[5,0] = 1 -> X0Group[5] - X0Group[0] <= -1
 LinkX0Group1[5,1]: X0LessThan[5,1] = 1 -> X0Group[5] - X0Group[1] <= -1
 LinkX0Group1[5,2]: X0LessThan[5,2] = 1 -> X0Group[5] - X0Group[2] <= -1
 LinkX0Group1[5,3]: X0LessThan[5,3] = 1 -> X0Group[5] - X0Group[3] <= -1
 LinkX0Group1[5,4]: X0LessThan[5,4] = 1 -> X0Group[5] - X0Group[4] <= -1
 LinkX0Group1[5,6]: X0LessThan[5,6] = 1 -> X0Group[5] - X0Group[6] <= -1
 LinkX0Group1[5,7]: X0LessThan[5,7] = 1 -> X0Group[5] - X0Group[7] <= -1
 LinkX0Group1[5,8]: X0LessThan[5,8] = 1 -> X0Group[5] - X0Group[8] <= -1
 LinkX0Group1[5,9]: X0LessThan[5,9] = 1 -> X0Group[5] - X0Group[9] <= -1
 LinkX0Group1[5,10]: X0LessThan[5,10] = 1 -> X0Group[5] - X0Group[10]
   <= -1
 LinkX0Group1[6,0]: X0LessThan[6,0] = 1 -> X0Group[6] - X0Group[0] <= -1
 LinkX0Group1[6,1]: X0LessThan[6,1] = 1 -> X0Group[6] - X0Group[1] <= -1
 LinkX0Group1[6,2]: X0LessThan[6,2] = 1 -> X0Group[6] - X0Group[2] <= -1
 LinkX0Group1[6,3]: X0LessThan[6,3] = 1 -> X0Group[6] - X0Group[3] <= -1
 LinkX0Group1[6,4]: X0LessThan[6,4] = 1 -> X0Group[6] - X0Group[4] <= -1
 LinkX0Group1[6,5]: X0LessThan[6,5] = 1 -> X0Group[6] - X0Group[5] <= -1
 LinkX0Group1[6,7]: X0LessThan[6,7] = 1 -> X0Group[6] - X0Group[7] <= -1
 LinkX0Group1[6,8]: X0LessThan[6,8] = 1 -> X0Group[6] - X0Group[8] <= -1
 LinkX0Group1[6,9]: X0LessThan[6,9] = 1 -> X0Group[6] - X0Group[9] <= -1
 LinkX0Group1[6,10]: X0LessThan[6,10] = 1 -> X0Group[6] - X0Group[10]
   <= -1
 LinkX0Group1[7,0]: X0LessThan[7,0] = 1 -> X0Group[7] - X0Group[0] <= -1
 LinkX0Group1[7,1]: X0LessThan[7,1] = 1 -> X0Group[7] - X0Group[1] <= -1
 LinkX0Group1[7,2]: X0LessThan[7,2] = 1 -> X0Group[7] - X0Group[2] <= -1
 LinkX0Group1[7,3]: X0LessThan[7,3] = 1 -> X0Group[7] - X0Group[3] <= -1
 LinkX0Group1[7,4]: X0LessThan[7,4] = 1 -> X0Group[7] - X0Group[4] <= -1
 LinkX0Group1[7,5]: X0LessThan[7,5] = 1 -> X0Group[7] - X0Group[5] <= -1
 LinkX0Group1[7,6]: X0LessThan[7,6] = 1 -> X0Group[7] - X0Group[6] <= -1
 LinkX0Group1[7,8]: X0LessThan[7,8] = 1 -> X0Group[7] - X0Group[8] <= -1
 LinkX0Group1[7,9]: X0LessThan[7,9] = 1 -> X0Group[7] - X0Group[9] <= -1
 LinkX0Group1[7,10]: X0LessThan[7,10] = 1 -> X0Group[7] - X0Group[10]
   <= -1
 LinkX0Group1[8,0]: X0LessThan[8,0] = 1 -> X0Group[8] - X0Group[0] <= -1
 LinkX0Group1[8,1]: X0LessThan[8,1] = 1 -> X0Group[8] - X0Group[1] <= -1
 LinkX0Group1[8,2]: X0LessThan[8,2] = 1 -> X0Group[8] - X0Group[2] <= -1
 LinkX0Group1[8,3]: X0LessThan[8,3] = 1 -> X0Group[8] - X0Group[3] <= -1
 LinkX0Group1[8,4]: X0LessThan[8,4] = 1 -> X0Group[8] - X0Group[4] <= -1
 LinkX0Group1[8,5]: X0LessThan[8,5] = 1 -> X0Group[8] - X0Group[5] <= -1
 LinkX0Group1[8,6]: X0LessThan[8,6] = 1 -> X0Group[8] - X0Group[6] <= -1
 LinkX0Group1[8,7]: X0LessThan[8,7] = 1 -> X0Group[8] - X0Group[7] <= -1
 LinkX0Group1[8,9]: X0LessThan[8,9] = 1 -> X0Group[8] - X0Group[9] <= -1
 LinkX0Group1[8,10]: X0LessThan[8,10] = 1 -> X0Group[8] - X0Group[10]
   <= -1
 LinkX0Group1[9,0]: X0LessThan[9,0] = 1 -> X0Group[9] - X0Group[0] <= -1
 LinkX0Group1[9,1]: X0LessThan[9,1] = 1 -> X0Group[9] - X0Group[1] <= -1
 LinkX0Group1[9,2]: X0LessThan[9,2] = 1 -> X0Group[9] - X0Group[2] <= -1
 LinkX0Group1[9,3]: X0LessThan[9,3] = 1 -> X0Group[9] - X0Group[3] <= -1
 LinkX0Group1[9,4]: X0LessThan[9,4] = 1 -> X0Group[9] - X0Group[4] <= -1
 LinkX0Group1[9,5]: X0LessThan[9,5] = 1 -> X0Group[9] - X0Group[5] <= -1
 LinkX0Group1[9,6]: X0LessThan[9,6] = 1 -> X0Group[9] - X0Group[6] <= -1
 LinkX0Group1[9,7]: X0LessThan[9,7] = 1 -> X0Group[9] - X0Group[7] <= -1
 LinkX0Group1[9,8]: X0LessThan[9,8] = 1 -> X0Group[9] - X0Group[8] <= -1
 LinkX0Group1[9,10]: X0LessThan[9,10] = 1 -> X0Group[9] - X0Group[10]
   <= -1
 LinkX0Group1[10,0]: X0LessThan[10,0] = 1 -> X0Group[10] - X0Group[0]
   <= -1
 LinkX0Group1[10,1]: X0LessThan[10,1] = 1 -> X0Group[10] - X0Group[1]
   <= -1
 LinkX0Group1[10,2]: X0LessThan[10,2] = 1 -> X0Group[10] - X0Group[2]
   <= -1
 LinkX0Group1[10,3]: X0LessThan[10,3] = 1 -> X0Group[10] - X0Group[3]
   <= -1
 LinkX0Group1[10,4]: X0LessThan[10,4] = 1 -> X0Group[10] - X0Group[4]
   <= -1
 LinkX0Group1[10,5]: X0LessThan[10,5] = 1 -> X0Group[10] - X0Group[5]
   <= -1
 LinkX0Group1[10,6]: X0LessThan[10,6] = 1 -> X0Group[10] - X0Group[6]
   <= -1
 LinkX0Group1[10,7]: X0LessThan[10,7] = 1 -> X0Group[10] - X0Group[7]
   <= -1
 LinkX0Group1[10,8]: X0LessThan[10,8] = 1 -> X0Group[10] - X0Group[8]
   <= -1
 LinkX0Group1[10,9]: X0LessThan[10,9] = 1 -> X0Group[10] - X0Group[9]
   <= -1
 LinkX0Group2[0,1]: X0LessThan[0,1] = 0 -> X0Group[0] - X0Group[1] >= -0
 LinkX0Group2[0,2]: X0LessThan[0,2] = 0 -> X0Group[0] - X0Group[2] >= -0
 LinkX0Group2[0,3]: X0LessThan[0,3] = 0 -> X0Group[0] - X0Group[3] >= -0
 LinkX0Group2[0,4]: X0LessThan[0,4] = 0 -> X0Group[0] - X0Group[4] >= -0
 LinkX0Group2[0,5]: X0LessThan[0,5] = 0 -> X0Group[0] - X0Group[5] >= -0
 LinkX0Group2[0,6]: X0LessThan[0,6] = 0 -> X0Group[0] - X0Group[6] >= -0
 LinkX0Group2[0,7]: X0LessThan[0,7] = 0 -> X0Group[0] - X0Group[7] >= -0
 LinkX0Group2[0,8]: X0LessThan[0,8] = 0 -> X0Group[0] - X0Group[8] >= -0
 LinkX0Group2[0,9]: X0LessThan[0,9] = 0 -> X0Group[0] - X0Group[9] >= -0
 LinkX0Group2[0,10]: X0LessThan[0,10] = 0 -> X0Group[0] - X0Group[10]
   >= -0
 LinkX0Group2[1,0]: X0LessThan[1,0] = 0 -> X0Group[1] - X0Group[0] >= -0
 LinkX0Group2[1,2]: X0LessThan[1,2] = 0 -> X0Group[1] - X0Group[2] >= -0
 LinkX0Group2[1,3]: X0LessThan[1,3] = 0 -> X0Group[1] - X0Group[3] >= -0
 LinkX0Group2[1,4]: X0LessThan[1,4] = 0 -> X0Group[1] - X0Group[4] >= -0
 LinkX0Group2[1,5]: X0LessThan[1,5] = 0 -> X0Group[1] - X0Group[5] >= -0
 LinkX0Group2[1,6]: X0LessThan[1,6] = 0 -> X0Group[1] - X0Group[6] >= -0
 LinkX0Group2[1,7]: X0LessThan[1,7] = 0 -> X0Group[1] - X0Group[7] >= -0
 LinkX0Group2[1,8]: X0LessThan[1,8] = 0 -> X0Group[1] - X0Group[8] >= -0
 LinkX0Group2[1,9]: X0LessThan[1,9] = 0 -> X0Group[1] - X0Group[9] >= -0
 LinkX0Group2[1,10]: X0LessThan[1,10] = 0 -> X0Group[1] - X0Group[10]
   >= -0
 LinkX0Group2[2,0]: X0LessThan[2,0] = 0 -> X0Group[2] - X0Group[0] >= -0
 LinkX0Group2[2,1]: X0LessThan[2,1] = 0 -> X0Group[2] - X0Group[1] >= -0
 LinkX0Group2[2,3]: X0LessThan[2,3] = 0 -> X0Group[2] - X0Group[3] >= -0
 LinkX0Group2[2,4]: X0LessThan[2,4] = 0 -> X0Group[2] - X0Group[4] >= -0
 LinkX0Group2[2,5]: X0LessThan[2,5] = 0 -> X0Group[2] - X0Group[5] >= -0
 LinkX0Group2[2,6]: X0LessThan[2,6] = 0 -> X0Group[2] - X0Group[6] >= -0
 LinkX0Group2[2,7]: X0LessThan[2,7] = 0 -> X0Group[2] - X0Group[7] >= -0
 LinkX0Group2[2,8]: X0LessThan[2,8] = 0 -> X0Group[2] - X0Group[8] >= -0
 LinkX0Group2[2,9]: X0LessThan[2,9] = 0 -> X0Group[2] - X0Group[9] >= -0
 LinkX0Group2[2,10]: X0LessThan[2,10] = 0 -> X0Group[2] - X0Group[10]
   >= -0
 LinkX0Group2[3,0]: X0LessThan[3,0] = 0 -> X0Group[3] - X0Group[0] >= -0
 LinkX0Group2[3,1]: X0LessThan[3,1] = 0 -> X0Group[3] - X0Group[1] >= -0
 LinkX0Group2[3,2]: X0LessThan[3,2] = 0 -> X0Group[3] - X0Group[2] >= -0
 LinkX0Group2[3,4]: X0LessThan[3,4] = 0 -> X0Group[3] - X0Group[4] >= -0
 LinkX0Group2[3,5]: X0LessThan[3,5] = 0 -> X0Group[3] - X0Group[5] >= -0
 LinkX0Group2[3,6]: X0LessThan[3,6] = 0 -> X0Group[3] - X0Group[6] >= -0
 LinkX0Group2[3,7]: X0LessThan[3,7] = 0 -> X0Group[3] - X0Group[7] >= -0
 LinkX0Group2[3,8]: X0LessThan[3,8] = 0 -> X0Group[3] - X0Group[8] >= -0
 LinkX0Group2[3,9]: X0LessThan[3,9] = 0 -> X0Group[3] - X0Group[9] >= -0
 LinkX0Group2[3,10]: X0LessThan[3,10] = 0 -> X0Group[3] - X0Group[10]
   >= -0
 LinkX0Group2[4,0]: X0LessThan[4,0] = 0 -> X0Group[4] - X0Group[0] >= -0
 LinkX0Group2[4,1]: X0LessThan[4,1] = 0 -> X0Group[4] - X0Group[1] >= -0
 LinkX0Group2[4,2]: X0LessThan[4,2] = 0 -> X0Group[4] - X0Group[2] >= -0
 LinkX0Group2[4,3]: X0LessThan[4,3] = 0 -> X0Group[4] - X0Group[3] >= -0
 LinkX0Group2[4,5]: X0LessThan[4,5] = 0 -> X0Group[4] - X0Group[5] >= -0
 LinkX0Group2[4,6]: X0LessThan[4,6] = 0 -> X0Group[4] - X0Group[6] >= -0
 LinkX0Group2[4,7]: X0LessThan[4,7] = 0 -> X0Group[4] - X0Group[7] >= -0
 LinkX0Group2[4,8]: X0LessThan[4,8] = 0 -> X0Group[4] - X0Group[8] >= -0
 LinkX0Group2[4,9]: X0LessThan[4,9] = 0 -> X0Group[4] - X0Group[9] >= -0
 LinkX0Group2[4,10]: X0LessThan[4,10] = 0 -> X0Group[4] - X0Group[10]
   >= -0
 LinkX0Group2[5,0]: X0LessThan[5,0] = 0 -> X0Group[5] - X0Group[0] >= -0
 LinkX0Group2[5,1]: X0LessThan[5,1] = 0 -> X0Group[5] - X0Group[1] >= -0
 LinkX0Group2[5,2]: X0LessThan[5,2] = 0 -> X0Group[5] - X0Group[2] >= -0
 LinkX0Group2[5,3]: X0LessThan[5,3] = 0 -> X0Group[5] - X0Group[3] >= -0
 LinkX0Group2[5,4]: X0LessThan[5,4] = 0 -> X0Group[5] - X0Group[4] >= -0
 LinkX0Group2[5,6]: X0LessThan[5,6] = 0 -> X0Group[5] - X0Group[6] >= -0
 LinkX0Group2[5,7]: X0LessThan[5,7] = 0 -> X0Group[5] - X0Group[7] >= -0
 LinkX0Group2[5,8]: X0LessThan[5,8] = 0 -> X0Group[5] - X0Group[8] >= -0
 LinkX0Group2[5,9]: X0LessThan[5,9] = 0 -> X0Group[5] - X0Group[9] >= -0
 LinkX0Group2[5,10]: X0LessThan[5,10] = 0 -> X0Group[5] - X0Group[10]
   >= -0
 LinkX0Group2[6,0]: X0LessThan[6,0] = 0 -> X0Group[6] - X0Group[0] >= -0
 LinkX0Group2[6,1]: X0LessThan[6,1] = 0 -> X0Group[6] - X0Group[1] >= -0
 LinkX0Group2[6,2]: X0LessThan[6,2] = 0 -> X0Group[6] - X0Group[2] >= -0
 LinkX0Group2[6,3]: X0LessThan[6,3] = 0 -> X0Group[6] - X0Group[3] >= -0
 LinkX0Group2[6,4]: X0LessThan[6,4] = 0 -> X0Group[6] - X0Group[4] >= -0
 LinkX0Group2[6,5]: X0LessThan[6,5] = 0 -> X0Group[6] - X0Group[5] >= -0
 LinkX0Group2[6,7]: X0LessThan[6,7] = 0 -> X0Group[6] - X0Group[7] >= -0
 LinkX0Group2[6,8]: X0LessThan[6,8] = 0 -> X0Group[6] - X0Group[8] >= -0
 LinkX0Group2[6,9]: X0LessThan[6,9] = 0 -> X0Group[6] - X0Group[9] >= -0
 LinkX0Group2[6,10]: X0LessThan[6,10] = 0 -> X0Group[6] - X0Group[10]
   >= -0
 LinkX0Group2[7,0]: X0LessThan[7,0] = 0 -> X0Group[7] - X0Group[0] >= -0
 LinkX0Group2[7,1]: X0LessThan[7,1] = 0 -> X0Group[7] - X0Group[1] >= -0
 LinkX0Group2[7,2]: X0LessThan[7,2] = 0 -> X0Group[7] - X0Group[2] >= -0
 LinkX0Group2[7,3]: X0LessThan[7,3] = 0 -> X0Group[7] - X0Group[3] >= -0
 LinkX0Group2[7,4]: X0LessThan[7,4] = 0 -> X0Group[7] - X0Group[4] >= -0
 LinkX0Group2[7,5]: X0LessThan[7,5] = 0 -> X0Group[7] - X0Group[5] >= -0
 LinkX0Group2[7,6]: X0LessThan[7,6] = 0 -> X0Group[7] - X0Group[6] >= -0
 LinkX0Group2[7,8]: X0LessThan[7,8] = 0 -> X0Group[7] - X0Group[8] >= -0
 LinkX0Group2[7,9]: X0LessThan[7,9] = 0 -> X0Group[7] - X0Group[9] >= -0
 LinkX0Group2[7,10]: X0LessThan[7,10] = 0 -> X0Group[7] - X0Group[10]
   >= -0
 LinkX0Group2[8,0]: X0LessThan[8,0] = 0 -> X0Group[8] - X0Group[0] >= -0
 LinkX0Group2[8,1]: X0LessThan[8,1] = 0 -> X0Group[8] - X0Group[1] >= -0
 LinkX0Group2[8,2]: X0LessThan[8,2] = 0 -> X0Group[8] - X0Group[2] >= -0
 LinkX0Group2[8,3]: X0LessThan[8,3] = 0 -> X0Group[8] - X0Group[3] >= -0
 LinkX0Group2[8,4]: X0LessThan[8,4] = 0 -> X0Group[8] - X0Group[4] >= -0
 LinkX0Group2[8,5]: X0LessThan[8,5] = 0 -> X0Group[8] - X0Group[5] >= -0
 LinkX0Group2[8,6]: X0LessThan[8,6] = 0 -> X0Group[8] - X0Group[6] >= -0
 LinkX0Group2[8,7]: X0LessThan[8,7] = 0 -> X0Group[8] - X0Group[7] >= -0
 LinkX0Group2[8,9]: X0LessThan[8,9] = 0 -> X0Group[8] - X0Group[9] >= -0
 LinkX0Group2[8,10]: X0LessThan[8,10] = 0 -> X0Group[8] - X0Group[10]
   >= -0
 LinkX0Group2[9,0]: X0LessThan[9,0] = 0 -> X0Group[9] - X0Group[0] >= -0
 LinkX0Group2[9,1]: X0LessThan[9,1] = 0 -> X0Group[9] - X0Group[1] >= -0
 LinkX0Group2[9,2]: X0LessThan[9,2] = 0 -> X0Group[9] - X0Group[2] >= -0
 LinkX0Group2[9,3]: X0LessThan[9,3] = 0 -> X0Group[9] - X0Group[3] >= -0
 LinkX0Group2[9,4]: X0LessThan[9,4] = 0 -> X0Group[9] - X0Group[4] >= -0
 LinkX0Group2[9,5]: X0LessThan[9,5] = 0 -> X0Group[9] - X0Group[5] >= -0
 LinkX0Group2[9,6]: X0LessThan[9,6] = 0 -> X0Group[9] - X0Group[6] >= -0
 LinkX0Group2[9,7]: X0LessThan[9,7] = 0 -> X0Group[9] - X0Group[7] >= -0
 LinkX0Group2[9,8]: X0LessThan[9,8] = 0 -> X0Group[9] - X0Group[8] >= -0
 LinkX0Group2[9,10]: X0LessThan[9,10] = 0 -> X0Group[9] - X0Group[10]
   >= -0
 LinkX0Group2[10,0]: X0LessThan[10,0] = 0 -> X0Group[10] - X0Group[0]
   >= -0
 LinkX0Group2[10,1]: X0LessThan[10,1] = 0 -> X0Group[10] - X0Group[1]
   >= -0
 LinkX0Group2[10,2]: X0LessThan[10,2] = 0 -> X0Group[10] - X0Group[2]
   >= -0
 LinkX0Group2[10,3]: X0LessThan[10,3] = 0 -> X0Group[10] - X0Group[3]
   >= -0
 LinkX0Group2[10,4]: X0LessThan[10,4] = 0 -> X0Group[10] - X0Group[4]
   >= -0
 LinkX0Group2[10,5]: X0LessThan[10,5] = 0 -> X0Group[10] - X0Group[5]
   >= -0
 LinkX0Group2[10,6]: X0LessThan[10,6] = 0 -> X0Group[10] - X0Group[6]
   >= -0
 LinkX0Group2[10,7]: X0LessThan[10,7] = 0 -> X0Group[10] - X0Group[7]
   >= -0
 LinkX0Group2[10,8]: X0LessThan[10,8] = 0 -> X0Group[10] - X0Group[8]
   >= -0
 LinkX0Group2[10,9]: X0LessThan[10,9] = 0 -> X0Group[10] - X0Group[9]
   >= -0
 LinkY0Group1[0,1]: Y0LessThan[0,1] = 1 -> Y0Group[0] - Y0Group[1] <= -1
 LinkY0Group1[0,2]: Y0LessThan[0,2] = 1 -> Y0Group[0] - Y0Group[2] <= -1
 LinkY0Group1[0,3]: Y0LessThan[0,3] = 1 -> Y0Group[0] - Y0Group[3] <= -1
 LinkY0Group1[0,4]: Y0LessThan[0,4] = 1 -> Y0Group[0] - Y0Group[4] <= -1
 LinkY0Group1[0,5]: Y0LessThan[0,5] = 1 -> Y0Group[0] - Y0Group[5] <= -1
 LinkY0Group1[0,6]: Y0LessThan[0,6] = 1 -> Y0Group[0] - Y0Group[6] <= -1
 LinkY0Group1[0,7]: Y0LessThan[0,7] = 1 -> Y0Group[0] - Y0Group[7] <= -1
 LinkY0Group1[0,8]: Y0LessThan[0,8] = 1 -> Y0Group[0] - Y0Group[8] <= -1
 LinkY0Group1[0,9]: Y0LessThan[0,9] = 1 -> Y0Group[0] - Y0Group[9] <= -1
 LinkY0Group1[0,10]: Y0LessThan[0,10] = 1 -> Y0Group[0] - Y0Group[10]
   <= -1
 LinkY0Group1[1,0]: Y0LessThan[1,0] = 1 -> Y0Group[1] - Y0Group[0] <= -1
 LinkY0Group1[1,2]: Y0LessThan[1,2] = 1 -> Y0Group[1] - Y0Group[2] <= -1
 LinkY0Group1[1,3]: Y0LessThan[1,3] = 1 -> Y0Group[1] - Y0Group[3] <= -1
 LinkY0Group1[1,4]: Y0LessThan[1,4] = 1 -> Y0Group[1] - Y0Group[4] <= -1
 LinkY0Group1[1,5]: Y0LessThan[1,5] = 1 -> Y0Group[1] - Y0Group[5] <= -1
 LinkY0Group1[1,6]: Y0LessThan[1,6] = 1 -> Y0Group[1] - Y0Group[6] <= -1
 LinkY0Group1[1,7]: Y0LessThan[1,7] = 1 -> Y0Group[1] - Y0Group[7] <= -1
 LinkY0Group1[1,8]: Y0LessThan[1,8] = 1 -> Y0Group[1] - Y0Group[8] <= -1
 LinkY0Group1[1,9]: Y0LessThan[1,9] = 1 -> Y0Group[1] - Y0Group[9] <= -1
 LinkY0Group1[1,10]: Y0LessThan[1,10] = 1 -> Y0Group[1] - Y0Group[10]
   <= -1
 LinkY0Group1[2,0]: Y0LessThan[2,0] = 1 -> Y0Group[2] - Y0Group[0] <= -1
 LinkY0Group1[2,1]: Y0LessThan[2,1] = 1 -> Y0Group[2] - Y0Group[1] <= -1
 LinkY0Group1[2,3]: Y0LessThan[2,3] = 1 -> Y0Group[2] - Y0Group[3] <= -1
 LinkY0Group1[2,4]: Y0LessThan[2,4] = 1 -> Y0Group[2] - Y0Group[4] <= -1
 LinkY0Group1[2,5]: Y0LessThan[2,5] = 1 -> Y0Group[2] - Y0Group[5] <= -1
 LinkY0Group1[2,6]: Y0LessThan[2,6] = 1 -> Y0Group[2] - Y0Group[6] <= -1
 LinkY0Group1[2,7]: Y0LessThan[2,7] = 1 -> Y0Group[2] - Y0Group[7] <= -1
 LinkY0Group1[2,8]: Y0LessThan[2,8] = 1 -> Y0Group[2] - Y0Group[8] <= -1
 LinkY0Group1[2,9]: Y0LessThan[2,9] = 1 -> Y0Group[2] - Y0Group[9] <= -1
 LinkY0Group1[2,10]: Y0LessThan[2,10] = 1 -> Y0Group[2] - Y0Group[10]
   <= -1
 LinkY0Group1[3,0]: Y0LessThan[3,0] = 1 -> Y0Group[3] - Y0Group[0] <= -1
 LinkY0Group1[3,1]: Y0LessThan[3,1] = 1 -> Y0Group[3] - Y0Group[1] <= -1
 LinkY0Group1[3,2]: Y0LessThan[3,2] = 1 -> Y0Group[3] - Y0Group[2] <= -1
 LinkY0Group1[3,4]: Y0LessThan[3,4] = 1 -> Y0Group[3] - Y0Group[4] <= -1
 LinkY0Group1[3,5]: Y0LessThan[3,5] = 1 -> Y0Group[3] - Y0Group[5] <= -1
 LinkY0Group1[3,6]: Y0LessThan[3,6] = 1 -> Y0Group[3] - Y0Group[6] <= -1
 LinkY0Group1[3,7]: Y0LessThan[3,7] = 1 -> Y0Group[3] - Y0Group[7] <= -1
 LinkY0Group1[3,8]: Y0LessThan[3,8] = 1 -> Y0Group[3] - Y0Group[8] <= -1
 LinkY0Group1[3,9]: Y0LessThan[3,9] = 1 -> Y0Group[3] - Y0Group[9] <= -1
 LinkY0Group1[3,10]: Y0LessThan[3,10] = 1 -> Y0Group[3] - Y0Group[10]
   <= -1
 LinkY0Group1[4,0]: Y0LessThan[4,0] = 1 -> Y0Group[4] - Y0Group[0] <= -1
 LinkY0Group1[4,1]: Y0LessThan[4,1] = 1 -> Y0Group[4] - Y0Group[1] <= -1
 LinkY0Group1[4,2]: Y0LessThan[4,2] = 1 -> Y0Group[4] - Y0Group[2] <= -1
 LinkY0Group1[4,3]: Y0LessThan[4,3] = 1 -> Y0Group[4] - Y0Group[3] <= -1
 LinkY0Group1[4,5]: Y0LessThan[4,5] = 1 -> Y0Group[4] - Y0Group[5] <= -1
 LinkY0Group1[4,6]: Y0LessThan[4,6] = 1 -> Y0Group[4] - Y0Group[6] <= -1
 LinkY0Group1[4,7]: Y0LessThan[4,7] = 1 -> Y0Group[4] - Y0Group[7] <= -1
 LinkY0Group1[4,8]: Y0LessThan[4,8] = 1 -> Y0Group[4] - Y0Group[8] <= -1
 LinkY0Group1[4,9]: Y0LessThan[4,9] = 1 -> Y0Group[4] - Y0Group[9] <= -1
 LinkY0Group1[4,10]: Y0LessThan[4,10] = 1 -> Y0Group[4] - Y0Group[10]
   <= -1
 LinkY0Group1[5,0]: Y0LessThan[5,0] = 1 -> Y0Group[5] - Y0Group[0] <= -1
 LinkY0Group1[5,1]: Y0LessThan[5,1] = 1 -> Y0Group[5] - Y0Group[1] <= -1
 LinkY0Group1[5,2]: Y0LessThan[5,2] = 1 -> Y0Group[5] - Y0Group[2] <= -1
 LinkY0Group1[5,3]: Y0LessThan[5,3] = 1 -> Y0Group[5] - Y0Group[3] <= -1
 LinkY0Group1[5,4]: Y0LessThan[5,4] = 1 -> Y0Group[5] - Y0Group[4] <= -1
 LinkY0Group1[5,6]: Y0LessThan[5,6] = 1 -> Y0Group[5] - Y0Group[6] <= -1
 LinkY0Group1[5,7]: Y0LessThan[5,7] = 1 -> Y0Group[5] - Y0Group[7] <= -1
 LinkY0Group1[5,8]: Y0LessThan[5,8] = 1 -> Y0Group[5] - Y0Group[8] <= -1
 LinkY0Group1[5,9]: Y0LessThan[5,9] = 1 -> Y0Group[5] - Y0Group[9] <= -1
 LinkY0Group1[5,10]: Y0LessThan[5,10] = 1 -> Y0Group[5] - Y0Group[10]
   <= -1
 LinkY0Group1[6,0]: Y0LessThan[6,0] = 1 -> Y0Group[6] - Y0Group[0] <= -1
 LinkY0Group1[6,1]: Y0LessThan[6,1] = 1 -> Y0Group[6] - Y0Group[1] <= -1
 LinkY0Group1[6,2]: Y0LessThan[6,2] = 1 -> Y0Group[6] - Y0Group[2] <= -1
 LinkY0Group1[6,3]: Y0LessThan[6,3] = 1 -> Y0Group[6] - Y0Group[3] <= -1
 LinkY0Group1[6,4]: Y0LessThan[6,4] = 1 -> Y0Group[6] - Y0Group[4] <= -1
 LinkY0Group1[6,5]: Y0LessThan[6,5] = 1 -> Y0Group[6] - Y0Group[5] <= -1
 LinkY0Group1[6,7]: Y0LessThan[6,7] = 1 -> Y0Group[6] - Y0Group[7] <= -1
 LinkY0Group1[6,8]: Y0LessThan[6,8] = 1 -> Y0Group[6] - Y0Group[8] <= -1
 LinkY0Group1[6,9]: Y0LessThan[6,9] = 1 -> Y0Group[6] - Y0Group[9] <= -1
 LinkY0Group1[6,10]: Y0LessThan[6,10] = 1 -> Y0Group[6] - Y0Group[10]
   <= -1
 LinkY0Group1[7,0]: Y0LessThan[7,0] = 1 -> Y0Group[7] - Y0Group[0] <= -1
 LinkY0Group1[7,1]: Y0LessThan[7,1] = 1 -> Y0Group[7] - Y0Group[1] <= -1
 LinkY0Group1[7,2]: Y0LessThan[7,2] = 1 -> Y0Group[7] - Y0Group[2] <= -1
 LinkY0Group1[7,3]: Y0LessThan[7,3] = 1 -> Y0Group[7] - Y0Group[3] <= -1
 LinkY0Group1[7,4]: Y0LessThan[7,4] = 1 -> Y0Group[7] - Y0Group[4] <= -1
 LinkY0Group1[7,5]: Y0LessThan[7,5] = 1 -> Y0Group[7] - Y0Group[5] <= -1
 LinkY0Group1[7,6]: Y0LessThan[7,6] = 1 -> Y0Group[7] - Y0Group[6] <= -1
 LinkY0Group1[7,8]: Y0LessThan[7,8] = 1 -> Y0Group[7] - Y0Group[8] <= -1
 LinkY0Group1[7,9]: Y0LessThan[7,9] = 1 -> Y0Group[7] - Y0Group[9] <= -1
 LinkY0Group1[7,10]: Y0LessThan[7,10] = 1 -> Y0Group[7] - Y0Group[10]
   <= -1
 LinkY0Group1[8,0]: Y0LessThan[8,0] = 1 -> Y0Group[8] - Y0Group[0] <= -1
 LinkY0Group1[8,1]: Y0LessThan[8,1] = 1 -> Y0Group[8] - Y0Group[1] <= -1
 LinkY0Group1[8,2]: Y0LessThan[8,2] = 1 -> Y0Group[8] - Y0Group[2] <= -1
 LinkY0Group1[8,3]: Y0LessThan[8,3] = 1 -> Y0Group[8] - Y0Group[3] <= -1
 LinkY0Group1[8,4]: Y0LessThan[8,4] = 1 -> Y0Group[8] - Y0Group[4] <= -1
 LinkY0Group1[8,5]: Y0LessThan[8,5] = 1 -> Y0Group[8] - Y0Group[5] <= -1
 LinkY0Group1[8,6]: Y0LessThan[8,6] = 1 -> Y0Group[8] - Y0Group[6] <= -1
 LinkY0Group1[8,7]: Y0LessThan[8,7] = 1 -> Y0Group[8] - Y0Group[7] <= -1
 LinkY0Group1[8,9]: Y0LessThan[8,9] = 1 -> Y0Group[8] - Y0Group[9] <= -1
 LinkY0Group1[8,10]: Y0LessThan[8,10] = 1 -> Y0Group[8] - Y0Group[10]
   <= -1
 LinkY0Group1[9,0]: Y0LessThan[9,0] = 1 -> Y0Group[9] - Y0Group[0] <= -1
 LinkY0Group1[9,1]: Y0LessThan[9,1] = 1 -> Y0Group[9] - Y0Group[1] <= -1
 LinkY0Group1[9,2]: Y0LessThan[9,2] = 1 -> Y0Group[9] - Y0Group[2] <= -1
 LinkY0Group1[9,3]: Y0LessThan[9,3] = 1 -> Y0Group[9] - Y0Group[3] <= -1
 LinkY0Group1[9,4]: Y0LessThan[9,4] = 1 -> Y0Group[9] - Y0Group[4] <= -1
 LinkY0Group1[9,5]: Y0LessThan[9,5] = 1 -> Y0Group[9] - Y0Group[5] <= -1
 LinkY0Group1[9,6]: Y0LessThan[9,6] = 1 -> Y0Group[9] - Y0Group[6] <= -1
 LinkY0Group1[9,7]: Y0LessThan[9,7] = 1 -> Y0Group[9] - Y0Group[7] <= -1
 LinkY0Group1[9,8]: Y0LessThan[9,8] = 1 -> Y0Group[9] - Y0Group[8] <= -1
 LinkY0Group1[9,10]: Y0LessThan[9,10] = 1 -> Y0Group[9] - Y0Group[10]
   <= -1
 LinkY0Group1[10,0]: Y0LessThan[10,0] = 1 -> Y0Group[10] - Y0Group[0]
   <= -1
 LinkY0Group1[10,1]: Y0LessThan[10,1] = 1 -> Y0Group[10] - Y0Group[1]
   <= -1
 LinkY0Group1[10,2]: Y0LessThan[10,2] = 1 -> Y0Group[10] - Y0Group[2]
   <= -1
 LinkY0Group1[10,3]: Y0LessThan[10,3] = 1 -> Y0Group[10] - Y0Group[3]
   <= -1
 LinkY0Group1[10,4]: Y0LessThan[10,4] = 1 -> Y0Group[10] - Y0Group[4]
   <= -1
 LinkY0Group1[10,5]: Y0LessThan[10,5] = 1 -> Y0Group[10] - Y0Group[5]
   <= -1
 LinkY0Group1[10,6]: Y0LessThan[10,6] = 1 -> Y0Group[10] - Y0Group[6]
   <= -1
 LinkY0Group1[10,7]: Y0LessThan[10,7] = 1 -> Y0Group[10] - Y0Group[7]
   <= -1
 LinkY0Group1[10,8]: Y0LessThan[10,8] = 1 -> Y0Group[10] - Y0Group[8]
   <= -1
 LinkY0Group1[10,9]: Y0LessThan[10,9] = 1 -> Y0Group[10] - Y0Group[9]
   <= -1
 LinkY0Group2[0,1]: Y0LessThan[0,1] = 0 -> Y0Group[0] - Y0Group[1] >= -0
 LinkY0Group2[0,2]: Y0LessThan[0,2] = 0 -> Y0Group[0] - Y0Group[2] >= -0
 LinkY0Group2[0,3]: Y0LessThan[0,3] = 0 -> Y0Group[0] - Y0Group[3] >= -0
 LinkY0Group2[0,4]: Y0LessThan[0,4] = 0 -> Y0Group[0] - Y0Group[4] >= -0
 LinkY0Group2[0,5]: Y0LessThan[0,5] = 0 -> Y0Group[0] - Y0Group[5] >= -0
 LinkY0Group2[0,6]: Y0LessThan[0,6] = 0 -> Y0Group[0] - Y0Group[6] >= -0
 LinkY0Group2[0,7]: Y0LessThan[0,7] = 0 -> Y0Group[0] - Y0Group[7] >= -0
 LinkY0Group2[0,8]: Y0LessThan[0,8] = 0 -> Y0Group[0] - Y0Group[8] >= -0
 LinkY0Group2[0,9]: Y0LessThan[0,9] = 0 -> Y0Group[0] - Y0Group[9] >= -0
 LinkY0Group2[0,10]: Y0LessThan[0,10] = 0 -> Y0Group[0] - Y0Group[10]
   >= -0
 LinkY0Group2[1,0]: Y0LessThan[1,0] = 0 -> Y0Group[1] - Y0Group[0] >= -0
 LinkY0Group2[1,2]: Y0LessThan[1,2] = 0 -> Y0Group[1] - Y0Group[2] >= -0
 LinkY0Group2[1,3]: Y0LessThan[1,3] = 0 -> Y0Group[1] - Y0Group[3] >= -0
 LinkY0Group2[1,4]: Y0LessThan[1,4] = 0 -> Y0Group[1] - Y0Group[4] >= -0
 LinkY0Group2[1,5]: Y0LessThan[1,5] = 0 -> Y0Group[1] - Y0Group[5] >= -0
 LinkY0Group2[1,6]: Y0LessThan[1,6] = 0 -> Y0Group[1] - Y0Group[6] >= -0
 LinkY0Group2[1,7]: Y0LessThan[1,7] = 0 -> Y0Group[1] - Y0Group[7] >= -0
 LinkY0Group2[1,8]: Y0LessThan[1,8] = 0 -> Y0Group[1] - Y0Group[8] >= -0
 LinkY0Group2[1,9]: Y0LessThan[1,9] = 0 -> Y0Group[1] - Y0Group[9] >= -0
 LinkY0Group2[1,10]: Y0LessThan[1,10] = 0 -> Y0Group[1] - Y0Group[10]
   >= -0
 LinkY0Group2[2,0]: Y0LessThan[2,0] = 0 -> Y0Group[2] - Y0Group[0] >= -0
 LinkY0Group2[2,1]: Y0LessThan[2,1] = 0 -> Y0Group[2] - Y0Group[1] >= -0
 LinkY0Group2[2,3]: Y0LessThan[2,3] = 0 -> Y0Group[2] - Y0Group[3] >= -0
 LinkY0Group2[2,4]: Y0LessThan[2,4] = 0 -> Y0Group[2] - Y0Group[4] >= -0
 LinkY0Group2[2,5]: Y0LessThan[2,5] = 0 -> Y0Group[2] - Y0Group[5] >= -0
 LinkY0Group2[2,6]: Y0LessThan[2,6] = 0 -> Y0Group[2] - Y0Group[6] >= -0
 LinkY0Group2[2,7]: Y0LessThan[2,7] = 0 -> Y0Group[2] - Y0Group[7] >= -0
 LinkY0Group2[2,8]: Y0LessThan[2,8] = 0 -> Y0Group[2] - Y0Group[8] >= -0
 LinkY0Group2[2,9]: Y0LessThan[2,9] = 0 -> Y0Group[2] - Y0Group[9] >= -0
 LinkY0Group2[2,10]: Y0LessThan[2,10] = 0 -> Y0Group[2] - Y0Group[10]
   >= -0
 LinkY0Group2[3,0]: Y0LessThan[3,0] = 0 -> Y0Group[3] - Y0Group[0] >= -0
 LinkY0Group2[3,1]: Y0LessThan[3,1] = 0 -> Y0Group[3] - Y0Group[1] >= -0
 LinkY0Group2[3,2]: Y0LessThan[3,2] = 0 -> Y0Group[3] - Y0Group[2] >= -0
 LinkY0Group2[3,4]: Y0LessThan[3,4] = 0 -> Y0Group[3] - Y0Group[4] >= -0
 LinkY0Group2[3,5]: Y0LessThan[3,5] = 0 -> Y0Group[3] - Y0Group[5] >= -0
 LinkY0Group2[3,6]: Y0LessThan[3,6] = 0 -> Y0Group[3] - Y0Group[6] >= -0
 LinkY0Group2[3,7]: Y0LessThan[3,7] = 0 -> Y0Group[3] - Y0Group[7] >= -0
 LinkY0Group2[3,8]: Y0LessThan[3,8] = 0 -> Y0Group[3] - Y0Group[8] >= -0
 LinkY0Group2[3,9]: Y0LessThan[3,9] = 0 -> Y0Group[3] - Y0Group[9] >= -0
 LinkY0Group2[3,10]: Y0LessThan[3,10] = 0 -> Y0Group[3] - Y0Group[10]
   >= -0
 LinkY0Group2[4,0]: Y0LessThan[4,0] = 0 -> Y0Group[4] - Y0Group[0] >= -0
 LinkY0Group2[4,1]: Y0LessThan[4,1] = 0 -> Y0Group[4] - Y0Group[1] >= -0
 LinkY0Group2[4,2]: Y0LessThan[4,2] = 0 -> Y0Group[4] - Y0Group[2] >= -0
 LinkY0Group2[4,3]: Y0LessThan[4,3] = 0 -> Y0Group[4] - Y0Group[3] >= -0
 LinkY0Group2[4,5]: Y0LessThan[4,5] = 0 -> Y0Group[4] - Y0Group[5] >= -0
 LinkY0Group2[4,6]: Y0LessThan[4,6] = 0 -> Y0Group[4] - Y0Group[6] >= -0
 LinkY0Group2[4,7]: Y0LessThan[4,7] = 0 -> Y0Group[4] - Y0Group[7] >= -0
 LinkY0Group2[4,8]: Y0LessThan[4,8] = 0 -> Y0Group[4] - Y0Group[8] >= -0
 LinkY0Group2[4,9]: Y0LessThan[4,9] = 0 -> Y0Group[4] - Y0Group[9] >= -0
 LinkY0Group2[4,10]: Y0LessThan[4,10] = 0 -> Y0Group[4] - Y0Group[10]
   >= -0
 LinkY0Group2[5,0]: Y0LessThan[5,0] = 0 -> Y0Group[5] - Y0Group[0] >= -0
 LinkY0Group2[5,1]: Y0LessThan[5,1] = 0 -> Y0Group[5] - Y0Group[1] >= -0
 LinkY0Group2[5,2]: Y0LessThan[5,2] = 0 -> Y0Group[5] - Y0Group[2] >= -0
 LinkY0Group2[5,3]: Y0LessThan[5,3] = 0 -> Y0Group[5] - Y0Group[3] >= -0
 LinkY0Group2[5,4]: Y0LessThan[5,4] = 0 -> Y0Group[5] - Y0Group[4] >= -0
 LinkY0Group2[5,6]: Y0LessThan[5,6] = 0 -> Y0Group[5] - Y0Group[6] >= -0
 LinkY0Group2[5,7]: Y0LessThan[5,7] = 0 -> Y0Group[5] - Y0Group[7] >= -0
 LinkY0Group2[5,8]: Y0LessThan[5,8] = 0 -> Y0Group[5] - Y0Group[8] >= -0
 LinkY0Group2[5,9]: Y0LessThan[5,9] = 0 -> Y0Group[5] - Y0Group[9] >= -0
 LinkY0Group2[5,10]: Y0LessThan[5,10] = 0 -> Y0Group[5] - Y0Group[10]
   >= -0
 LinkY0Group2[6,0]: Y0LessThan[6,0] = 0 -> Y0Group[6] - Y0Group[0] >= -0
 LinkY0Group2[6,1]: Y0LessThan[6,1] = 0 -> Y0Group[6] - Y0Group[1] >= -0
 LinkY0Group2[6,2]: Y0LessThan[6,2] = 0 -> Y0Group[6] - Y0Group[2] >= -0
 LinkY0Group2[6,3]: Y0LessThan[6,3] = 0 -> Y0Group[6] - Y0Group[3] >= -0
 LinkY0Group2[6,4]: Y0LessThan[6,4] = 0 -> Y0Group[6] - Y0Group[4] >= -0
 LinkY0Group2[6,5]: Y0LessThan[6,5] = 0 -> Y0Group[6] - Y0Group[5] >= -0
 LinkY0Group2[6,7]: Y0LessThan[6,7] = 0 -> Y0Group[6] - Y0Group[7] >= -0
 LinkY0Group2[6,8]: Y0LessThan[6,8] = 0 -> Y0Group[6] - Y0Group[8] >= -0
 LinkY0Group2[6,9]: Y0LessThan[6,9] = 0 -> Y0Group[6] - Y0Group[9] >= -0
 LinkY0Group2[6,10]: Y0LessThan[6,10] = 0 -> Y0Group[6] - Y0Group[10]
   >= -0
 LinkY0Group2[7,0]: Y0LessThan[7,0] = 0 -> Y0Group[7] - Y0Group[0] >= -0
 LinkY0Group2[7,1]: Y0LessThan[7,1] = 0 -> Y0Group[7] - Y0Group[1] >= -0
 LinkY0Group2[7,2]: Y0LessThan[7,2] = 0 -> Y0Group[7] - Y0Group[2] >= -0
 LinkY0Group2[7,3]: Y0LessThan[7,3] = 0 -> Y0Group[7] - Y0Group[3] >= -0
 LinkY0Group2[7,4]: Y0LessThan[7,4] = 0 -> Y0Group[7] - Y0Group[4] >= -0
 LinkY0Group2[7,5]: Y0LessThan[7,5] = 0 -> Y0Group[7] - Y0Group[5] >= -0
 LinkY0Group2[7,6]: Y0LessThan[7,6] = 0 -> Y0Group[7] - Y0Group[6] >= -0
 LinkY0Group2[7,8]: Y0LessThan[7,8] = 0 -> Y0Group[7] - Y0Group[8] >= -0
 LinkY0Group2[7,9]: Y0LessThan[7,9] = 0 -> Y0Group[7] - Y0Group[9] >= -0
 LinkY0Group2[7,10]: Y0LessThan[7,10] = 0 -> Y0Group[7] - Y0Group[10]
   >= -0
 LinkY0Group2[8,0]: Y0LessThan[8,0] = 0 -> Y0Group[8] - Y0Group[0] >= -0
 LinkY0Group2[8,1]: Y0LessThan[8,1] = 0 -> Y0Group[8] - Y0Group[1] >= -0
 LinkY0Group2[8,2]: Y0LessThan[8,2] = 0 -> Y0Group[8] - Y0Group[2] >= -0
 LinkY0Group2[8,3]: Y0LessThan[8,3] = 0 -> Y0Group[8] - Y0Group[3] >= -0
 LinkY0Group2[8,4]: Y0LessThan[8,4] = 0 -> Y0Group[8] - Y0Group[4] >= -0
 LinkY0Group2[8,5]: Y0LessThan[8,5] = 0 -> Y0Group[8] - Y0Group[5] >= -0
 LinkY0Group2[8,6]: Y0LessThan[8,6] = 0 -> Y0Group[8] - Y0Group[6] >= -0
 LinkY0Group2[8,7]: Y0LessThan[8,7] = 0 -> Y0Group[8] - Y0Group[7] >= -0
 LinkY0Group2[8,9]: Y0LessThan[8,9] = 0 -> Y0Group[8] - Y0Group[9] >= -0
 LinkY0Group2[8,10]: Y0LessThan[8,10] = 0 -> Y0Group[8] - Y0Group[10]
   >= -0
 LinkY0Group2[9,0]: Y0LessThan[9,0] = 0 -> Y0Group[9] - Y0Group[0] >= -0
 LinkY0Group2[9,1]: Y0LessThan[9,1] = 0 -> Y0Group[9] - Y0Group[1] >= -0
 LinkY0Group2[9,2]: Y0LessThan[9,2] = 0 -> Y0Group[9] - Y0Group[2] >= -0
 LinkY0Group2[9,3]: Y0LessThan[9,3] = 0 -> Y0Group[9] - Y0Group[3] >= -0
 LinkY0Group2[9,4]: Y0LessThan[9,4] = 0 -> Y0Group[9] - Y0Group[4] >= -0
 LinkY0Group2[9,5]: Y0LessThan[9,5] = 0 -> Y0Group[9] - Y0Group[5] >= -0
 LinkY0Group2[9,6]: Y0LessThan[9,6] = 0 -> Y0Group[9] - Y0Group[6] >= -0
 LinkY0Group2[9,7]: Y0LessThan[9,7] = 0 -> Y0Group[9] - Y0Group[7] >= -0
 LinkY0Group2[9,8]: Y0LessThan[9,8] = 0 -> Y0Group[9] - Y0Group[8] >= -0
 LinkY0Group2[9,10]: Y0LessThan[9,10] = 0 -> Y0Group[9] - Y0Group[10]
   >= -0
 LinkY0Group2[10,0]: Y0LessThan[10,0] = 0 -> Y0Group[10] - Y0Group[0]
   >= -0
 LinkY0Group2[10,1]: Y0LessThan[10,1] = 0 -> Y0Group[10] - Y0Group[1]
   >= -0
 LinkY0Group2[10,2]: Y0LessThan[10,2] = 0 -> Y0Group[10] - Y0Group[2]
   >= -0
 LinkY0Group2[10,3]: Y0LessThan[10,3] = 0 -> Y0Group[10] - Y0Group[3]
   >= -0
 LinkY0Group2[10,4]: Y0LessThan[10,4] = 0 -> Y0Group[10] - Y0Group[4]
   >= -0
 LinkY0Group2[10,5]: Y0LessThan[10,5] = 0 -> Y0Group[10] - Y0Group[5]
   >= -0
 LinkY0Group2[10,6]: Y0LessThan[10,6] = 0 -> Y0Group[10] - Y0Group[6]
   >= -0
 LinkY0Group2[10,7]: Y0LessThan[10,7] = 0 -> Y0Group[10] - Y0Group[7]
   >= -0
 LinkY0Group2[10,8]: Y0LessThan[10,8] = 0 -> Y0Group[10] - Y0Group[8]
   >= -0
 LinkY0Group2[10,9]: Y0LessThan[10,9] = 0 -> Y0Group[10] - Y0Group[9]
   >= -0
 LinkX1Group1[0,1]: X1LessThan[0,1] = 1 -> X1Group[0] - X1Group[1] <= -1
 LinkX1Group1[0,2]: X1LessThan[0,2] = 1 -> X1Group[0] - X1Group[2] <= -1
 LinkX1Group1[0,3]: X1LessThan[0,3] = 1 -> X1Group[0] - X1Group[3] <= -1
 LinkX1Group1[0,4]: X1LessThan[0,4] = 1 -> X1Group[0] - X1Group[4] <= -1
 LinkX1Group1[0,5]: X1LessThan[0,5] = 1 -> X1Group[0] - X1Group[5] <= -1
 LinkX1Group1[0,6]: X1LessThan[0,6] = 1 -> X1Group[0] - X1Group[6] <= -1
 LinkX1Group1[0,7]: X1LessThan[0,7] = 1 -> X1Group[0] - X1Group[7] <= -1
 LinkX1Group1[0,8]: X1LessThan[0,8] = 1 -> X1Group[0] - X1Group[8] <= -1
 LinkX1Group1[0,9]: X1LessThan[0,9] = 1 -> X1Group[0] - X1Group[9] <= -1
 LinkX1Group1[0,10]: X1LessThan[0,10] = 1 -> X1Group[0] - X1Group[10]
   <= -1
 LinkX1Group1[1,0]: X1LessThan[1,0] = 1 -> X1Group[1] - X1Group[0] <= -1
 LinkX1Group1[1,2]: X1LessThan[1,2] = 1 -> X1Group[1] - X1Group[2] <= -1
 LinkX1Group1[1,3]: X1LessThan[1,3] = 1 -> X1Group[1] - X1Group[3] <= -1
 LinkX1Group1[1,4]: X1LessThan[1,4] = 1 -> X1Group[1] - X1Group[4] <= -1
 LinkX1Group1[1,5]: X1LessThan[1,5] = 1 -> X1Group[1] - X1Group[5] <= -1
 LinkX1Group1[1,6]: X1LessThan[1,6] = 1 -> X1Group[1] - X1Group[6] <= -1
 LinkX1Group1[1,7]: X1LessThan[1,7] = 1 -> X1Group[1] - X1Group[7] <= -1
 LinkX1Group1[1,8]: X1LessThan[1,8] = 1 -> X1Group[1] - X1Group[8] <= -1
 LinkX1Group1[1,9]: X1LessThan[1,9] = 1 -> X1Group[1] - X1Group[9] <= -1
 LinkX1Group1[1,10]: X1LessThan[1,10] = 1 -> X1Group[1] - X1Group[10]
   <= -1
 LinkX1Group1[2,0]: X1LessThan[2,0] = 1 -> X1Group[2] - X1Group[0] <= -1
 LinkX1Group1[2,1]: X1LessThan[2,1] = 1 -> X1Group[2] - X1Group[1] <= -1
 LinkX1Group1[2,3]: X1LessThan[2,3] = 1 -> X1Group[2] - X1Group[3] <= -1
 LinkX1Group1[2,4]: X1LessThan[2,4] = 1 -> X1Group[2] - X1Group[4] <= -1
 LinkX1Group1[2,5]: X1LessThan[2,5] = 1 -> X1Group[2] - X1Group[5] <= -1
 LinkX1Group1[2,6]: X1LessThan[2,6] = 1 -> X1Group[2] - X1Group[6] <= -1
 LinkX1Group1[2,7]: X1LessThan[2,7] = 1 -> X1Group[2] - X1Group[7] <= -1
 LinkX1Group1[2,8]: X1LessThan[2,8] = 1 -> X1Group[2] - X1Group[8] <= -1
 LinkX1Group1[2,9]: X1LessThan[2,9] = 1 -> X1Group[2] - X1Group[9] <= -1
 LinkX1Group1[2,10]: X1LessThan[2,10] = 1 -> X1Group[2] - X1Group[10]
   <= -1
 LinkX1Group1[3,0]: X1LessThan[3,0] = 1 -> X1Group[3] - X1Group[0] <= -1
 LinkX1Group1[3,1]: X1LessThan[3,1] = 1 -> X1Group[3] - X1Group[1] <= -1
 LinkX1Group1[3,2]: X1LessThan[3,2] = 1 -> X1Group[3] - X1Group[2] <= -1
 LinkX1Group1[3,4]: X1LessThan[3,4] = 1 -> X1Group[3] - X1Group[4] <= -1
 LinkX1Group1[3,5]: X1LessThan[3,5] = 1 -> X1Group[3] - X1Group[5] <= -1
 LinkX1Group1[3,6]: X1LessThan[3,6] = 1 -> X1Group[3] - X1Group[6] <= -1
 LinkX1Group1[3,7]: X1LessThan[3,7] = 1 -> X1Group[3] - X1Group[7] <= -1
 LinkX1Group1[3,8]: X1LessThan[3,8] = 1 -> X1Group[3] - X1Group[8] <= -1
 LinkX1Group1[3,9]: X1LessThan[3,9] = 1 -> X1Group[3] - X1Group[9] <= -1
 LinkX1Group1[3,10]: X1LessThan[3,10] = 1 -> X1Group[3] - X1Group[10]
   <= -1
 LinkX1Group1[4,0]: X1LessThan[4,0] = 1 -> X1Group[4] - X1Group[0] <= -1
 LinkX1Group1[4,1]: X1LessThan[4,1] = 1 -> X1Group[4] - X1Group[1] <= -1
 LinkX1Group1[4,2]: X1LessThan[4,2] = 1 -> X1Group[4] - X1Group[2] <= -1
 LinkX1Group1[4,3]: X1LessThan[4,3] = 1 -> X1Group[4] - X1Group[3] <= -1
 LinkX1Group1[4,5]: X1LessThan[4,5] = 1 -> X1Group[4] - X1Group[5] <= -1
 LinkX1Group1[4,6]: X1LessThan[4,6] = 1 -> X1Group[4] - X1Group[6] <= -1
 LinkX1Group1[4,7]: X1LessThan[4,7] = 1 -> X1Group[4] - X1Group[7] <= -1
 LinkX1Group1[4,8]: X1LessThan[4,8] = 1 -> X1Group[4] - X1Group[8] <= -1
 LinkX1Group1[4,9]: X1LessThan[4,9] = 1 -> X1Group[4] - X1Group[9] <= -1
 LinkX1Group1[4,10]: X1LessThan[4,10] = 1 -> X1Group[4] - X1Group[10]
   <= -1
 LinkX1Group1[5,0]: X1LessThan[5,0] = 1 -> X1Group[5] - X1Group[0] <= -1
 LinkX1Group1[5,1]: X1LessThan[5,1] = 1 -> X1Group[5] - X1Group[1] <= -1
 LinkX1Group1[5,2]: X1LessThan[5,2] = 1 -> X1Group[5] - X1Group[2] <= -1
 LinkX1Group1[5,3]: X1LessThan[5,3] = 1 -> X1Group[5] - X1Group[3] <= -1
 LinkX1Group1[5,4]: X1LessThan[5,4] = 1 -> X1Group[5] - X1Group[4] <= -1
 LinkX1Group1[5,6]: X1LessThan[5,6] = 1 -> X1Group[5] - X1Group[6] <= -1
 LinkX1Group1[5,7]: X1LessThan[5,7] = 1 -> X1Group[5] - X1Group[7] <= -1
 LinkX1Group1[5,8]: X1LessThan[5,8] = 1 -> X1Group[5] - X1Group[8] <= -1
 LinkX1Group1[5,9]: X1LessThan[5,9] = 1 -> X1Group[5] - X1Group[9] <= -1
 LinkX1Group1[5,10]: X1LessThan[5,10] = 1 -> X1Group[5] - X1Group[10]
   <= -1
 LinkX1Group1[6,0]: X1LessThan[6,0] = 1 -> X1Group[6] - X1Group[0] <= -1
 LinkX1Group1[6,1]: X1LessThan[6,1] = 1 -> X1Group[6] - X1Group[1] <= -1
 LinkX1Group1[6,2]: X1LessThan[6,2] = 1 -> X1Group[6] - X1Group[2] <= -1
 LinkX1Group1[6,3]: X1LessThan[6,3] = 1 -> X1Group[6] - X1Group[3] <= -1
 LinkX1Group1[6,4]: X1LessThan[6,4] = 1 -> X1Group[6] - X1Group[4] <= -1
 LinkX1Group1[6,5]: X1LessThan[6,5] = 1 -> X1Group[6] - X1Group[5] <= -1
 LinkX1Group1[6,7]: X1LessThan[6,7] = 1 -> X1Group[6] - X1Group[7] <= -1
 LinkX1Group1[6,8]: X1LessThan[6,8] = 1 -> X1Group[6] - X1Group[8] <= -1
 LinkX1Group1[6,9]: X1LessThan[6,9] = 1 -> X1Group[6] - X1Group[9] <= -1
 LinkX1Group1[6,10]: X1LessThan[6,10] = 1 -> X1Group[6] - X1Group[10]
   <= -1
 LinkX1Group1[7,0]: X1LessThan[7,0] = 1 -> X1Group[7] - X1Group[0] <= -1
 LinkX1Group1[7,1]: X1LessThan[7,1] = 1 -> X1Group[7] - X1Group[1] <= -1
 LinkX1Group1[7,2]: X1LessThan[7,2] = 1 -> X1Group[7] - X1Group[2] <= -1
 LinkX1Group1[7,3]: X1LessThan[7,3] = 1 -> X1Group[7] - X1Group[3] <= -1
 LinkX1Group1[7,4]: X1LessThan[7,4] = 1 -> X1Group[7] - X1Group[4] <= -1
 LinkX1Group1[7,5]: X1LessThan[7,5] = 1 -> X1Group[7] - X1Group[5] <= -1
 LinkX1Group1[7,6]: X1LessThan[7,6] = 1 -> X1Group[7] - X1Group[6] <= -1
 LinkX1Group1[7,8]: X1LessThan[7,8] = 1 -> X1Group[7] - X1Group[8] <= -1
 LinkX1Group1[7,9]: X1LessThan[7,9] = 1 -> X1Group[7] - X1Group[9] <= -1
 LinkX1Group1[7,10]: X1LessThan[7,10] = 1 -> X1Group[7] - X1Group[10]
   <= -1
 LinkX1Group1[8,0]: X1LessThan[8,0] = 1 -> X1Group[8] - X1Group[0] <= -1
 LinkX1Group1[8,1]: X1LessThan[8,1] = 1 -> X1Group[8] - X1Group[1] <= -1
 LinkX1Group1[8,2]: X1LessThan[8,2] = 1 -> X1Group[8] - X1Group[2] <= -1
 LinkX1Group1[8,3]: X1LessThan[8,3] = 1 -> X1Group[8] - X1Group[3] <= -1
 LinkX1Group1[8,4]: X1LessThan[8,4] = 1 -> X1Group[8] - X1Group[4] <= -1
 LinkX1Group1[8,5]: X1LessThan[8,5] = 1 -> X1Group[8] - X1Group[5] <= -1
 LinkX1Group1[8,6]: X1LessThan[8,6] = 1 -> X1Group[8] - X1Group[6] <= -1
 LinkX1Group1[8,7]: X1LessThan[8,7] = 1 -> X1Group[8] - X1Group[7] <= -1
 LinkX1Group1[8,9]: X1LessThan[8,9] = 1 -> X1Group[8] - X1Group[9] <= -1
 LinkX1Group1[8,10]: X1LessThan[8,10] = 1 -> X1Group[8] - X1Group[10]
   <= -1
 LinkX1Group1[9,0]: X1LessThan[9,0] = 1 -> X1Group[9] - X1Group[0] <= -1
 LinkX1Group1[9,1]: X1LessThan[9,1] = 1 -> X1Group[9] - X1Group[1] <= -1
 LinkX1Group1[9,2]: X1LessThan[9,2] = 1 -> X1Group[9] - X1Group[2] <= -1
 LinkX1Group1[9,3]: X1LessThan[9,3] = 1 -> X1Group[9] - X1Group[3] <= -1
 LinkX1Group1[9,4]: X1LessThan[9,4] = 1 -> X1Group[9] - X1Group[4] <= -1
 LinkX1Group1[9,5]: X1LessThan[9,5] = 1 -> X1Group[9] - X1Group[5] <= -1
 LinkX1Group1[9,6]: X1LessThan[9,6] = 1 -> X1Group[9] - X1Group[6] <= -1
 LinkX1Group1[9,7]: X1LessThan[9,7] = 1 -> X1Group[9] - X1Group[7] <= -1
 LinkX1Group1[9,8]: X1LessThan[9,8] = 1 -> X1Group[9] - X1Group[8] <= -1
 LinkX1Group1[9,10]: X1LessThan[9,10] = 1 -> X1Group[9] - X1Group[10]
   <= -1
 LinkX1Group1[10,0]: X1LessThan[10,0] = 1 -> X1Group[10] - X1Group[0]
   <= -1
 LinkX1Group1[10,1]: X1LessThan[10,1] = 1 -> X1Group[10] - X1Group[1]
   <= -1
 LinkX1Group1[10,2]: X1LessThan[10,2] = 1 -> X1Group[10] - X1Group[2]
   <= -1
 LinkX1Group1[10,3]: X1LessThan[10,3] = 1 -> X1Group[10] - X1Group[3]
   <= -1
 LinkX1Group1[10,4]: X1LessThan[10,4] = 1 -> X1Group[10] - X1Group[4]
   <= -1
 LinkX1Group1[10,5]: X1LessThan[10,5] = 1 -> X1Group[10] - X1Group[5]
   <= -1
 LinkX1Group1[10,6]: X1LessThan[10,6] = 1 -> X1Group[10] - X1Group[6]
   <= -1
 LinkX1Group1[10,7]: X1LessThan[10,7] = 1 -> X1Group[10] - X1Group[7]
   <= -1
 LinkX1Group1[10,8]: X1LessThan[10,8] = 1 -> X1Group[10] - X1Group[8]
   <= -1
 LinkX1Group1[10,9]: X1LessThan[10,9] = 1 -> X1Group[10] - X1Group[9]
   <= -1
 LinkX1Group2[0,1]: X1LessThan[0,1] = 0 -> X1Group[0] - X1Group[1] >= -0
 LinkX1Group2[0,2]: X1LessThan[0,2] = 0 -> X1Group[0] - X1Group[2] >= -0
 LinkX1Group2[0,3]: X1LessThan[0,3] = 0 -> X1Group[0] - X1Group[3] >= -0
 LinkX1Group2[0,4]: X1LessThan[0,4] = 0 -> X1Group[0] - X1Group[4] >= -0
 LinkX1Group2[0,5]: X1LessThan[0,5] = 0 -> X1Group[0] - X1Group[5] >= -0
 LinkX1Group2[0,6]: X1LessThan[0,6] = 0 -> X1Group[0] - X1Group[6] >= -0
 LinkX1Group2[0,7]: X1LessThan[0,7] = 0 -> X1Group[0] - X1Group[7] >= -0
 LinkX1Group2[0,8]: X1LessThan[0,8] = 0 -> X1Group[0] - X1Group[8] >= -0
 LinkX1Group2[0,9]: X1LessThan[0,9] = 0 -> X1Group[0] - X1Group[9] >= -0
 LinkX1Group2[0,10]: X1LessThan[0,10] = 0 -> X1Group[0] - X1Group[10]
   >= -0
 LinkX1Group2[1,0]: X1LessThan[1,0] = 0 -> X1Group[1] - X1Group[0] >= -0
 LinkX1Group2[1,2]: X1LessThan[1,2] = 0 -> X1Group[1] - X1Group[2] >= -0
 LinkX1Group2[1,3]: X1LessThan[1,3] = 0 -> X1Group[1] - X1Group[3] >= -0
 LinkX1Group2[1,4]: X1LessThan[1,4] = 0 -> X1Group[1] - X1Group[4] >= -0
 LinkX1Group2[1,5]: X1LessThan[1,5] = 0 -> X1Group[1] - X1Group[5] >= -0
 LinkX1Group2[1,6]: X1LessThan[1,6] = 0 -> X1Group[1] - X1Group[6] >= -0
 LinkX1Group2[1,7]: X1LessThan[1,7] = 0 -> X1Group[1] - X1Group[7] >= -0
 LinkX1Group2[1,8]: X1LessThan[1,8] = 0 -> X1Group[1] - X1Group[8] >= -0
 LinkX1Group2[1,9]: X1LessThan[1,9] = 0 -> X1Group[1] - X1Group[9] >= -0
 LinkX1Group2[1,10]: X1LessThan[1,10] = 0 -> X1Group[1] - X1Group[10]
   >= -0
 LinkX1Group2[2,0]: X1LessThan[2,0] = 0 -> X1Group[2] - X1Group[0] >= -0
 LinkX1Group2[2,1]: X1LessThan[2,1] = 0 -> X1Group[2] - X1Group[1] >= -0
 LinkX1Group2[2,3]: X1LessThan[2,3] = 0 -> X1Group[2] - X1Group[3] >= -0
 LinkX1Group2[2,4]: X1LessThan[2,4] = 0 -> X1Group[2] - X1Group[4] >= -0
 LinkX1Group2[2,5]: X1LessThan[2,5] = 0 -> X1Group[2] - X1Group[5] >= -0
 LinkX1Group2[2,6]: X1LessThan[2,6] = 0 -> X1Group[2] - X1Group[6] >= -0
 LinkX1Group2[2,7]: X1LessThan[2,7] = 0 -> X1Group[2] - X1Group[7] >= -0
 LinkX1Group2[2,8]: X1LessThan[2,8] = 0 -> X1Group[2] - X1Group[8] >= -0
 LinkX1Group2[2,9]: X1LessThan[2,9] = 0 -> X1Group[2] - X1Group[9] >= -0
 LinkX1Group2[2,10]: X1LessThan[2,10] = 0 -> X1Group[2] - X1Group[10]
   >= -0
 LinkX1Group2[3,0]: X1LessThan[3,0] = 0 -> X1Group[3] - X1Group[0] >= -0
 LinkX1Group2[3,1]: X1LessThan[3,1] = 0 -> X1Group[3] - X1Group[1] >= -0
 LinkX1Group2[3,2]: X1LessThan[3,2] = 0 -> X1Group[3] - X1Group[2] >= -0
 LinkX1Group2[3,4]: X1LessThan[3,4] = 0 -> X1Group[3] - X1Group[4] >= -0
 LinkX1Group2[3,5]: X1LessThan[3,5] = 0 -> X1Group[3] - X1Group[5] >= -0
 LinkX1Group2[3,6]: X1LessThan[3,6] = 0 -> X1Group[3] - X1Group[6] >= -0
 LinkX1Group2[3,7]: X1LessThan[3,7] = 0 -> X1Group[3] - X1Group[7] >= -0
 LinkX1Group2[3,8]: X1LessThan[3,8] = 0 -> X1Group[3] - X1Group[8] >= -0
 LinkX1Group2[3,9]: X1LessThan[3,9] = 0 -> X1Group[3] - X1Group[9] >= -0
 LinkX1Group2[3,10]: X1LessThan[3,10] = 0 -> X1Group[3] - X1Group[10]
   >= -0
 LinkX1Group2[4,0]: X1LessThan[4,0] = 0 -> X1Group[4] - X1Group[0] >= -0
 LinkX1Group2[4,1]: X1LessThan[4,1] = 0 -> X1Group[4] - X1Group[1] >= -0
 LinkX1Group2[4,2]: X1LessThan[4,2] = 0 -> X1Group[4] - X1Group[2] >= -0
 LinkX1Group2[4,3]: X1LessThan[4,3] = 0 -> X1Group[4] - X1Group[3] >= -0
 LinkX1Group2[4,5]: X1LessThan[4,5] = 0 -> X1Group[4] - X1Group[5] >= -0
 LinkX1Group2[4,6]: X1LessThan[4,6] = 0 -> X1Group[4] - X1Group[6] >= -0
 LinkX1Group2[4,7]: X1LessThan[4,7] = 0 -> X1Group[4] - X1Group[7] >= -0
 LinkX1Group2[4,8]: X1LessThan[4,8] = 0 -> X1Group[4] - X1Group[8] >= -0
 LinkX1Group2[4,9]: X1LessThan[4,9] = 0 -> X1Group[4] - X1Group[9] >= -0
 LinkX1Group2[4,10]: X1LessThan[4,10] = 0 -> X1Group[4] - X1Group[10]
   >= -0
 LinkX1Group2[5,0]: X1LessThan[5,0] = 0 -> X1Group[5] - X1Group[0] >= -0
 LinkX1Group2[5,1]: X1LessThan[5,1] = 0 -> X1Group[5] - X1Group[1] >= -0
 LinkX1Group2[5,2]: X1LessThan[5,2] = 0 -> X1Group[5] - X1Group[2] >= -0
 LinkX1Group2[5,3]: X1LessThan[5,3] = 0 -> X1Group[5] - X1Group[3] >= -0
 LinkX1Group2[5,4]: X1LessThan[5,4] = 0 -> X1Group[5] - X1Group[4] >= -0
 LinkX1Group2[5,6]: X1LessThan[5,6] = 0 -> X1Group[5] - X1Group[6] >= -0
 LinkX1Group2[5,7]: X1LessThan[5,7] = 0 -> X1Group[5] - X1Group[7] >= -0
 LinkX1Group2[5,8]: X1LessThan[5,8] = 0 -> X1Group[5] - X1Group[8] >= -0
 LinkX1Group2[5,9]: X1LessThan[5,9] = 0 -> X1Group[5] - X1Group[9] >= -0
 LinkX1Group2[5,10]: X1LessThan[5,10] = 0 -> X1Group[5] - X1Group[10]
   >= -0
 LinkX1Group2[6,0]: X1LessThan[6,0] = 0 -> X1Group[6] - X1Group[0] >= -0
 LinkX1Group2[6,1]: X1LessThan[6,1] = 0 -> X1Group[6] - X1Group[1] >= -0
 LinkX1Group2[6,2]: X1LessThan[6,2] = 0 -> X1Group[6] - X1Group[2] >= -0
 LinkX1Group2[6,3]: X1LessThan[6,3] = 0 -> X1Group[6] - X1Group[3] >= -0
 LinkX1Group2[6,4]: X1LessThan[6,4] = 0 -> X1Group[6] - X1Group[4] >= -0
 LinkX1Group2[6,5]: X1LessThan[6,5] = 0 -> X1Group[6] - X1Group[5] >= -0
 LinkX1Group2[6,7]: X1LessThan[6,7] = 0 -> X1Group[6] - X1Group[7] >= -0
 LinkX1Group2[6,8]: X1LessThan[6,8] = 0 -> X1Group[6] - X1Group[8] >= -0
 LinkX1Group2[6,9]: X1LessThan[6,9] = 0 -> X1Group[6] - X1Group[9] >= -0
 LinkX1Group2[6,10]: X1LessThan[6,10] = 0 -> X1Group[6] - X1Group[10]
   >= -0
 LinkX1Group2[7,0]: X1LessThan[7,0] = 0 -> X1Group[7] - X1Group[0] >= -0
 LinkX1Group2[7,1]: X1LessThan[7,1] = 0 -> X1Group[7] - X1Group[1] >= -0
 LinkX1Group2[7,2]: X1LessThan[7,2] = 0 -> X1Group[7] - X1Group[2] >= -0
 LinkX1Group2[7,3]: X1LessThan[7,3] = 0 -> X1Group[7] - X1Group[3] >= -0
 LinkX1Group2[7,4]: X1LessThan[7,4] = 0 -> X1Group[7] - X1Group[4] >= -0
 LinkX1Group2[7,5]: X1LessThan[7,5] = 0 -> X1Group[7] - X1Group[5] >= -0
 LinkX1Group2[7,6]: X1LessThan[7,6] = 0 -> X1Group[7] - X1Group[6] >= -0
 LinkX1Group2[7,8]: X1LessThan[7,8] = 0 -> X1Group[7] - X1Group[8] >= -0
 LinkX1Group2[7,9]: X1LessThan[7,9] = 0 -> X1Group[7] - X1Group[9] >= -0
 LinkX1Group2[7,10]: X1LessThan[7,10] = 0 -> X1Group[7] - X1Group[10]
   >= -0
 LinkX1Group2[8,0]: X1LessThan[8,0] = 0 -> X1Group[8] - X1Group[0] >= -0
 LinkX1Group2[8,1]: X1LessThan[8,1] = 0 -> X1Group[8] - X1Group[1] >= -0
 LinkX1Group2[8,2]: X1LessThan[8,2] = 0 -> X1Group[8] - X1Group[2] >= -0
 LinkX1Group2[8,3]: X1LessThan[8,3] = 0 -> X1Group[8] - X1Group[3] >= -0
 LinkX1Group2[8,4]: X1LessThan[8,4] = 0 -> X1Group[8] - X1Group[4] >= -0
 LinkX1Group2[8,5]: X1LessThan[8,5] = 0 -> X1Group[8] - X1Group[5] >= -0
 LinkX1Group2[8,6]: X1LessThan[8,6] = 0 -> X1Group[8] - X1Group[6] >= -0
 LinkX1Group2[8,7]: X1LessThan[8,7] = 0 -> X1Group[8] - X1Group[7] >= -0
 LinkX1Group2[8,9]: X1LessThan[8,9] = 0 -> X1Group[8] - X1Group[9] >= -0
 LinkX1Group2[8,10]: X1LessThan[8,10] = 0 -> X1Group[8] - X1Group[10]
   >= -0
 LinkX1Group2[9,0]: X1LessThan[9,0] = 0 -> X1Group[9] - X1Group[0] >= -0
 LinkX1Group2[9,1]: X1LessThan[9,1] = 0 -> X1Group[9] - X1Group[1] >= -0
 LinkX1Group2[9,2]: X1LessThan[9,2] = 0 -> X1Group[9] - X1Group[2] >= -0
 LinkX1Group2[9,3]: X1LessThan[9,3] = 0 -> X1Group[9] - X1Group[3] >= -0
 LinkX1Group2[9,4]: X1LessThan[9,4] = 0 -> X1Group[9] - X1Group[4] >= -0
 LinkX1Group2[9,5]: X1LessThan[9,5] = 0 -> X1Group[9] - X1Group[5] >= -0
 LinkX1Group2[9,6]: X1LessThan[9,6] = 0 -> X1Group[9] - X1Group[6] >= -0
 LinkX1Group2[9,7]: X1LessThan[9,7] = 0 -> X1Group[9] - X1Group[7] >= -0
 LinkX1Group2[9,8]: X1LessThan[9,8] = 0 -> X1Group[9] - X1Group[8] >= -0
 LinkX1Group2[9,10]: X1LessThan[9,10] = 0 -> X1Group[9] - X1Group[10]
   >= -0
 LinkX1Group2[10,0]: X1LessThan[10,0] = 0 -> X1Group[10] - X1Group[0]
   >= -0
 LinkX1Group2[10,1]: X1LessThan[10,1] = 0 -> X1Group[10] - X1Group[1]
   >= -0
 LinkX1Group2[10,2]: X1LessThan[10,2] = 0 -> X1Group[10] - X1Group[2]
   >= -0
 LinkX1Group2[10,3]: X1LessThan[10,3] = 0 -> X1Group[10] - X1Group[3]
   >= -0
 LinkX1Group2[10,4]: X1LessThan[10,4] = 0 -> X1Group[10] - X1Group[4]
   >= -0
 LinkX1Group2[10,5]: X1LessThan[10,5] = 0 -> X1Group[10] - X1Group[5]
   >= -0
 LinkX1Group2[10,6]: X1LessThan[10,6] = 0 -> X1Group[10] - X1Group[6]
   >= -0
 LinkX1Group2[10,7]: X1LessThan[10,7] = 0 -> X1Group[10] - X1Group[7]
   >= -0
 LinkX1Group2[10,8]: X1LessThan[10,8] = 0 -> X1Group[10] - X1Group[8]
   >= -0
 LinkX1Group2[10,9]: X1LessThan[10,9] = 0 -> X1Group[10] - X1Group[9]
   >= -0
 LinkY1Group1[0,1]: Y1LessThan[0,1] = 1 -> Y1Group[0] - Y1Group[1] <= -1
 LinkY1Group1[0,2]: Y1LessThan[0,2] = 1 -> Y1Group[0] - Y1Group[2] <= -1
 LinkY1Group1[0,3]: Y1LessThan[0,3] = 1 -> Y1Group[0] - Y1Group[3] <= -1
 LinkY1Group1[0,4]: Y1LessThan[0,4] = 1 -> Y1Group[0] - Y1Group[4] <= -1
 LinkY1Group1[0,5]: Y1LessThan[0,5] = 1 -> Y1Group[0] - Y1Group[5] <= -1
 LinkY1Group1[0,6]: Y1LessThan[0,6] = 1 -> Y1Group[0] - Y1Group[6] <= -1
 LinkY1Group1[0,7]: Y1LessThan[0,7] = 1 -> Y1Group[0] - Y1Group[7] <= -1
 LinkY1Group1[0,8]: Y1LessThan[0,8] = 1 -> Y1Group[0] - Y1Group[8] <= -1
 LinkY1Group1[0,9]: Y1LessThan[0,9] = 1 -> Y1Group[0] - Y1Group[9] <= -1
 LinkY1Group1[0,10]: Y1LessThan[0,10] = 1 -> Y1Group[0] - Y1Group[10]
   <= -1
 LinkY1Group1[1,0]: Y1LessThan[1,0] = 1 -> Y1Group[1] - Y1Group[0] <= -1
 LinkY1Group1[1,2]: Y1LessThan[1,2] = 1 -> Y1Group[1] - Y1Group[2] <= -1
 LinkY1Group1[1,3]: Y1LessThan[1,3] = 1 -> Y1Group[1] - Y1Group[3] <= -1
 LinkY1Group1[1,4]: Y1LessThan[1,4] = 1 -> Y1Group[1] - Y1Group[4] <= -1
 LinkY1Group1[1,5]: Y1LessThan[1,5] = 1 -> Y1Group[1] - Y1Group[5] <= -1
 LinkY1Group1[1,6]: Y1LessThan[1,6] = 1 -> Y1Group[1] - Y1Group[6] <= -1
 LinkY1Group1[1,7]: Y1LessThan[1,7] = 1 -> Y1Group[1] - Y1Group[7] <= -1
 LinkY1Group1[1,8]: Y1LessThan[1,8] = 1 -> Y1Group[1] - Y1Group[8] <= -1
 LinkY1Group1[1,9]: Y1LessThan[1,9] = 1 -> Y1Group[1] - Y1Group[9] <= -1
 LinkY1Group1[1,10]: Y1LessThan[1,10] = 1 -> Y1Group[1] - Y1Group[10]
   <= -1
 LinkY1Group1[2,0]: Y1LessThan[2,0] = 1 -> Y1Group[2] - Y1Group[0] <= -1
 LinkY1Group1[2,1]: Y1LessThan[2,1] = 1 -> Y1Group[2] - Y1Group[1] <= -1
 LinkY1Group1[2,3]: Y1LessThan[2,3] = 1 -> Y1Group[2] - Y1Group[3] <= -1
 LinkY1Group1[2,4]: Y1LessThan[2,4] = 1 -> Y1Group[2] - Y1Group[4] <= -1
 LinkY1Group1[2,5]: Y1LessThan[2,5] = 1 -> Y1Group[2] - Y1Group[5] <= -1
 LinkY1Group1[2,6]: Y1LessThan[2,6] = 1 -> Y1Group[2] - Y1Group[6] <= -1
 LinkY1Group1[2,7]: Y1LessThan[2,7] = 1 -> Y1Group[2] - Y1Group[7] <= -1
 LinkY1Group1[2,8]: Y1LessThan[2,8] = 1 -> Y1Group[2] - Y1Group[8] <= -1
 LinkY1Group1[2,9]: Y1LessThan[2,9] = 1 -> Y1Group[2] - Y1Group[9] <= -1
 LinkY1Group1[2,10]: Y1LessThan[2,10] = 1 -> Y1Group[2] - Y1Group[10]
   <= -1
 LinkY1Group1[3,0]: Y1LessThan[3,0] = 1 -> Y1Group[3] - Y1Group[0] <= -1
 LinkY1Group1[3,1]: Y1LessThan[3,1] = 1 -> Y1Group[3] - Y1Group[1] <= -1
 LinkY1Group1[3,2]: Y1LessThan[3,2] = 1 -> Y1Group[3] - Y1Group[2] <= -1
 LinkY1Group1[3,4]: Y1LessThan[3,4] = 1 -> Y1Group[3] - Y1Group[4] <= -1
 LinkY1Group1[3,5]: Y1LessThan[3,5] = 1 -> Y1Group[3] - Y1Group[5] <= -1
 LinkY1Group1[3,6]: Y1LessThan[3,6] = 1 -> Y1Group[3] - Y1Group[6] <= -1
 LinkY1Group1[3,7]: Y1LessThan[3,7] = 1 -> Y1Group[3] - Y1Group[7] <= -1
 LinkY1Group1[3,8]: Y1LessThan[3,8] = 1 -> Y1Group[3] - Y1Group[8] <= -1
 LinkY1Group1[3,9]: Y1LessThan[3,9] = 1 -> Y1Group[3] - Y1Group[9] <= -1
 LinkY1Group1[3,10]: Y1LessThan[3,10] = 1 -> Y1Group[3] - Y1Group[10]
   <= -1
 LinkY1Group1[4,0]: Y1LessThan[4,0] = 1 -> Y1Group[4] - Y1Group[0] <= -1
 LinkY1Group1[4,1]: Y1LessThan[4,1] = 1 -> Y1Group[4] - Y1Group[1] <= -1
 LinkY1Group1[4,2]: Y1LessThan[4,2] = 1 -> Y1Group[4] - Y1Group[2] <= -1
 LinkY1Group1[4,3]: Y1LessThan[4,3] = 1 -> Y1Group[4] - Y1Group[3] <= -1
 LinkY1Group1[4,5]: Y1LessThan[4,5] = 1 -> Y1Group[4] - Y1Group[5] <= -1
 LinkY1Group1[4,6]: Y1LessThan[4,6] = 1 -> Y1Group[4] - Y1Group[6] <= -1
 LinkY1Group1[4,7]: Y1LessThan[4,7] = 1 -> Y1Group[4] - Y1Group[7] <= -1
 LinkY1Group1[4,8]: Y1LessThan[4,8] = 1 -> Y1Group[4] - Y1Group[8] <= -1
 LinkY1Group1[4,9]: Y1LessThan[4,9] = 1 -> Y1Group[4] - Y1Group[9] <= -1
 LinkY1Group1[4,10]: Y1LessThan[4,10] = 1 -> Y1Group[4] - Y1Group[10]
   <= -1
 LinkY1Group1[5,0]: Y1LessThan[5,0] = 1 -> Y1Group[5] - Y1Group[0] <= -1
 LinkY1Group1[5,1]: Y1LessThan[5,1] = 1 -> Y1Group[5] - Y1Group[1] <= -1
 LinkY1Group1[5,2]: Y1LessThan[5,2] = 1 -> Y1Group[5] - Y1Group[2] <= -1
 LinkY1Group1[5,3]: Y1LessThan[5,3] = 1 -> Y1Group[5] - Y1Group[3] <= -1
 LinkY1Group1[5,4]: Y1LessThan[5,4] = 1 -> Y1Group[5] - Y1Group[4] <= -1
 LinkY1Group1[5,6]: Y1LessThan[5,6] = 1 -> Y1Group[5] - Y1Group[6] <= -1
 LinkY1Group1[5,7]: Y1LessThan[5,7] = 1 -> Y1Group[5] - Y1Group[7] <= -1
 LinkY1Group1[5,8]: Y1LessThan[5,8] = 1 -> Y1Group[5] - Y1Group[8] <= -1
 LinkY1Group1[5,9]: Y1LessThan[5,9] = 1 -> Y1Group[5] - Y1Group[9] <= -1
 LinkY1Group1[5,10]: Y1LessThan[5,10] = 1 -> Y1Group[5] - Y1Group[10]
   <= -1
 LinkY1Group1[6,0]: Y1LessThan[6,0] = 1 -> Y1Group[6] - Y1Group[0] <= -1
 LinkY1Group1[6,1]: Y1LessThan[6,1] = 1 -> Y1Group[6] - Y1Group[1] <= -1
 LinkY1Group1[6,2]: Y1LessThan[6,2] = 1 -> Y1Group[6] - Y1Group[2] <= -1
 LinkY1Group1[6,3]: Y1LessThan[6,3] = 1 -> Y1Group[6] - Y1Group[3] <= -1
 LinkY1Group1[6,4]: Y1LessThan[6,4] = 1 -> Y1Group[6] - Y1Group[4] <= -1
 LinkY1Group1[6,5]: Y1LessThan[6,5] = 1 -> Y1Group[6] - Y1Group[5] <= -1
 LinkY1Group1[6,7]: Y1LessThan[6,7] = 1 -> Y1Group[6] - Y1Group[7] <= -1
 LinkY1Group1[6,8]: Y1LessThan[6,8] = 1 -> Y1Group[6] - Y1Group[8] <= -1
 LinkY1Group1[6,9]: Y1LessThan[6,9] = 1 -> Y1Group[6] - Y1Group[9] <= -1
 LinkY1Group1[6,10]: Y1LessThan[6,10] = 1 -> Y1Group[6] - Y1Group[10]
   <= -1
 LinkY1Group1[7,0]: Y1LessThan[7,0] = 1 -> Y1Group[7] - Y1Group[0] <= -1
 LinkY1Group1[7,1]: Y1LessThan[7,1] = 1 -> Y1Group[7] - Y1Group[1] <= -1
 LinkY1Group1[7,2]: Y1LessThan[7,2] = 1 -> Y1Group[7] - Y1Group[2] <= -1
 LinkY1Group1[7,3]: Y1LessThan[7,3] = 1 -> Y1Group[7] - Y1Group[3] <= -1
 LinkY1Group1[7,4]: Y1LessThan[7,4] = 1 -> Y1Group[7] - Y1Group[4] <= -1
 LinkY1Group1[7,5]: Y1LessThan[7,5] = 1 -> Y1Group[7] - Y1Group[5] <= -1
 LinkY1Group1[7,6]: Y1LessThan[7,6] = 1 -> Y1Group[7] - Y1Group[6] <= -1
 LinkY1Group1[7,8]: Y1LessThan[7,8] = 1 -> Y1Group[7] - Y1Group[8] <= -1
 LinkY1Group1[7,9]: Y1LessThan[7,9] = 1 -> Y1Group[7] - Y1Group[9] <= -1
 LinkY1Group1[7,10]: Y1LessThan[7,10] = 1 -> Y1Group[7] - Y1Group[10]
   <= -1
 LinkY1Group1[8,0]: Y1LessThan[8,0] = 1 -> Y1Group[8] - Y1Group[0] <= -1
 LinkY1Group1[8,1]: Y1LessThan[8,1] = 1 -> Y1Group[8] - Y1Group[1] <= -1
 LinkY1Group1[8,2]: Y1LessThan[8,2] = 1 -> Y1Group[8] - Y1Group[2] <= -1
 LinkY1Group1[8,3]: Y1LessThan[8,3] = 1 -> Y1Group[8] - Y1Group[3] <= -1
 LinkY1Group1[8,4]: Y1LessThan[8,4] = 1 -> Y1Group[8] - Y1Group[4] <= -1
 LinkY1Group1[8,5]: Y1LessThan[8,5] = 1 -> Y1Group[8] - Y1Group[5] <= -1
 LinkY1Group1[8,6]: Y1LessThan[8,6] = 1 -> Y1Group[8] - Y1Group[6] <= -1
 LinkY1Group1[8,7]: Y1LessThan[8,7] = 1 -> Y1Group[8] - Y1Group[7] <= -1
 LinkY1Group1[8,9]: Y1LessThan[8,9] = 1 -> Y1Group[8] - Y1Group[9] <= -1
 LinkY1Group1[8,10]: Y1LessThan[8,10] = 1 -> Y1Group[8] - Y1Group[10]
   <= -1
 LinkY1Group1[9,0]: Y1LessThan[9,0] = 1 -> Y1Group[9] - Y1Group[0] <= -1
 LinkY1Group1[9,1]: Y1LessThan[9,1] = 1 -> Y1Group[9] - Y1Group[1] <= -1
 LinkY1Group1[9,2]: Y1LessThan[9,2] = 1 -> Y1Group[9] - Y1Group[2] <= -1
 LinkY1Group1[9,3]: Y1LessThan[9,3] = 1 -> Y1Group[9] - Y1Group[3] <= -1
 LinkY1Group1[9,4]: Y1LessThan[9,4] = 1 -> Y1Group[9] - Y1Group[4] <= -1
 LinkY1Group1[9,5]: Y1LessThan[9,5] = 1 -> Y1Group[9] - Y1Group[5] <= -1
 LinkY1Group1[9,6]: Y1LessThan[9,6] = 1 -> Y1Group[9] - Y1Group[6] <= -1
 LinkY1Group1[9,7]: Y1LessThan[9,7] = 1 -> Y1Group[9] - Y1Group[7] <= -1
 LinkY1Group1[9,8]: Y1LessThan[9,8] = 1 -> Y1Group[9] - Y1Group[8] <= -1
 LinkY1Group1[9,10]: Y1LessThan[9,10] = 1 -> Y1Group[9] - Y1Group[10]
   <= -1
 LinkY1Group1[10,0]: Y1LessThan[10,0] = 1 -> Y1Group[10] - Y1Group[0]
   <= -1
 LinkY1Group1[10,1]: Y1LessThan[10,1] = 1 -> Y1Group[10] - Y1Group[1]
   <= -1
 LinkY1Group1[10,2]: Y1LessThan[10,2] = 1 -> Y1Group[10] - Y1Group[2]
   <= -1
 LinkY1Group1[10,3]: Y1LessThan[10,3] = 1 -> Y1Group[10] - Y1Group[3]
   <= -1
 LinkY1Group1[10,4]: Y1LessThan[10,4] = 1 -> Y1Group[10] - Y1Group[4]
   <= -1
 LinkY1Group1[10,5]: Y1LessThan[10,5] = 1 -> Y1Group[10] - Y1Group[5]
   <= -1
 LinkY1Group1[10,6]: Y1LessThan[10,6] = 1 -> Y1Group[10] - Y1Group[6]
   <= -1
 LinkY1Group1[10,7]: Y1LessThan[10,7] = 1 -> Y1Group[10] - Y1Group[7]
   <= -1
 LinkY1Group1[10,8]: Y1LessThan[10,8] = 1 -> Y1Group[10] - Y1Group[8]
   <= -1
 LinkY1Group1[10,9]: Y1LessThan[10,9] = 1 -> Y1Group[10] - Y1Group[9]
   <= -1
 LinkY1Group2[0,1]: Y1LessThan[0,1] = 0 -> Y1Group[0] - Y1Group[1] >= -0
 LinkY1Group2[0,2]: Y1LessThan[0,2] = 0 -> Y1Group[0] - Y1Group[2] >= -0
 LinkY1Group2[0,3]: Y1LessThan[0,3] = 0 -> Y1Group[0] - Y1Group[3] >= -0
 LinkY1Group2[0,4]: Y1LessThan[0,4] = 0 -> Y1Group[0] - Y1Group[4] >= -0
 LinkY1Group2[0,5]: Y1LessThan[0,5] = 0 -> Y1Group[0] - Y1Group[5] >= -0
 LinkY1Group2[0,6]: Y1LessThan[0,6] = 0 -> Y1Group[0] - Y1Group[6] >= -0
 LinkY1Group2[0,7]: Y1LessThan[0,7] = 0 -> Y1Group[0] - Y1Group[7] >= -0
 LinkY1Group2[0,8]: Y1LessThan[0,8] = 0 -> Y1Group[0] - Y1Group[8] >= -0
 LinkY1Group2[0,9]: Y1LessThan[0,9] = 0 -> Y1Group[0] - Y1Group[9] >= -0
 LinkY1Group2[0,10]: Y1LessThan[0,10] = 0 -> Y1Group[0] - Y1Group[10]
   >= -0
 LinkY1Group2[1,0]: Y1LessThan[1,0] = 0 -> Y1Group[1] - Y1Group[0] >= -0
 LinkY1Group2[1,2]: Y1LessThan[1,2] = 0 -> Y1Group[1] - Y1Group[2] >= -0
 LinkY1Group2[1,3]: Y1LessThan[1,3] = 0 -> Y1Group[1] - Y1Group[3] >= -0
 LinkY1Group2[1,4]: Y1LessThan[1,4] = 0 -> Y1Group[1] - Y1Group[4] >= -0
 LinkY1Group2[1,5]: Y1LessThan[1,5] = 0 -> Y1Group[1] - Y1Group[5] >= -0
 LinkY1Group2[1,6]: Y1LessThan[1,6] = 0 -> Y1Group[1] - Y1Group[6] >= -0
 LinkY1Group2[1,7]: Y1LessThan[1,7] = 0 -> Y1Group[1] - Y1Group[7] >= -0
 LinkY1Group2[1,8]: Y1LessThan[1,8] = 0 -> Y1Group[1] - Y1Group[8] >= -0
 LinkY1Group2[1,9]: Y1LessThan[1,9] = 0 -> Y1Group[1] - Y1Group[9] >= -0
 LinkY1Group2[1,10]: Y1LessThan[1,10] = 0 -> Y1Group[1] - Y1Group[10]
   >= -0
 LinkY1Group2[2,0]: Y1LessThan[2,0] = 0 -> Y1Group[2] - Y1Group[0] >= -0
 LinkY1Group2[2,1]: Y1LessThan[2,1] = 0 -> Y1Group[2] - Y1Group[1] >= -0
 LinkY1Group2[2,3]: Y1LessThan[2,3] = 0 -> Y1Group[2] - Y1Group[3] >= -0
 LinkY1Group2[2,4]: Y1LessThan[2,4] = 0 -> Y1Group[2] - Y1Group[4] >= -0
 LinkY1Group2[2,5]: Y1LessThan[2,5] = 0 -> Y1Group[2] - Y1Group[5] >= -0
 LinkY1Group2[2,6]: Y1LessThan[2,6] = 0 -> Y1Group[2] - Y1Group[6] >= -0
 LinkY1Group2[2,7]: Y1LessThan[2,7] = 0 -> Y1Group[2] - Y1Group[7] >= -0
 LinkY1Group2[2,8]: Y1LessThan[2,8] = 0 -> Y1Group[2] - Y1Group[8] >= -0
 LinkY1Group2[2,9]: Y1LessThan[2,9] = 0 -> Y1Group[2] - Y1Group[9] >= -0
 LinkY1Group2[2,10]: Y1LessThan[2,10] = 0 -> Y1Group[2] - Y1Group[10]
   >= -0
 LinkY1Group2[3,0]: Y1LessThan[3,0] = 0 -> Y1Group[3] - Y1Group[0] >= -0
 LinkY1Group2[3,1]: Y1LessThan[3,1] = 0 -> Y1Group[3] - Y1Group[1] >= -0
 LinkY1Group2[3,2]: Y1LessThan[3,2] = 0 -> Y1Group[3] - Y1Group[2] >= -0
 LinkY1Group2[3,4]: Y1LessThan[3,4] = 0 -> Y1Group[3] - Y1Group[4] >= -0
 LinkY1Group2[3,5]: Y1LessThan[3,5] = 0 -> Y1Group[3] - Y1Group[5] >= -0
 LinkY1Group2[3,6]: Y1LessThan[3,6] = 0 -> Y1Group[3] - Y1Group[6] >= -0
 LinkY1Group2[3,7]: Y1LessThan[3,7] = 0 -> Y1Group[3] - Y1Group[7] >= -0
 LinkY1Group2[3,8]: Y1LessThan[3,8] = 0 -> Y1Group[3] - Y1Group[8] >= -0
 LinkY1Group2[3,9]: Y1LessThan[3,9] = 0 -> Y1Group[3] - Y1Group[9] >= -0
 LinkY1Group2[3,10]: Y1LessThan[3,10] = 0 -> Y1Group[3] - Y1Group[10]
   >= -0
 LinkY1Group2[4,0]: Y1LessThan[4,0] = 0 -> Y1Group[4] - Y1Group[0] >= -0
 LinkY1Group2[4,1]: Y1LessThan[4,1] = 0 -> Y1Group[4] - Y1Group[1] >= -0
 LinkY1Group2[4,2]: Y1LessThan[4,2] = 0 -> Y1Group[4] - Y1Group[2] >= -0
 LinkY1Group2[4,3]: Y1LessThan[4,3] = 0 -> Y1Group[4] - Y1Group[3] >= -0
 LinkY1Group2[4,5]: Y1LessThan[4,5] = 0 -> Y1Group[4] - Y1Group[5] >= -0
 LinkY1Group2[4,6]: Y1LessThan[4,6] = 0 -> Y1Group[4] - Y1Group[6] >= -0
 LinkY1Group2[4,7]: Y1LessThan[4,7] = 0 -> Y1Group[4] - Y1Group[7] >= -0
 LinkY1Group2[4,8]: Y1LessThan[4,8] = 0 -> Y1Group[4] - Y1Group[8] >= -0
 LinkY1Group2[4,9]: Y1LessThan[4,9] = 0 -> Y1Group[4] - Y1Group[9] >= -0
 LinkY1Group2[4,10]: Y1LessThan[4,10] = 0 -> Y1Group[4] - Y1Group[10]
   >= -0
 LinkY1Group2[5,0]: Y1LessThan[5,0] = 0 -> Y1Group[5] - Y1Group[0] >= -0
 LinkY1Group2[5,1]: Y1LessThan[5,1] = 0 -> Y1Group[5] - Y1Group[1] >= -0
 LinkY1Group2[5,2]: Y1LessThan[5,2] = 0 -> Y1Group[5] - Y1Group[2] >= -0
 LinkY1Group2[5,3]: Y1LessThan[5,3] = 0 -> Y1Group[5] - Y1Group[3] >= -0
 LinkY1Group2[5,4]: Y1LessThan[5,4] = 0 -> Y1Group[5] - Y1Group[4] >= -0
 LinkY1Group2[5,6]: Y1LessThan[5,6] = 0 -> Y1Group[5] - Y1Group[6] >= -0
 LinkY1Group2[5,7]: Y1LessThan[5,7] = 0 -> Y1Group[5] - Y1Group[7] >= -0
 LinkY1Group2[5,8]: Y1LessThan[5,8] = 0 -> Y1Group[5] - Y1Group[8] >= -0
 LinkY1Group2[5,9]: Y1LessThan[5,9] = 0 -> Y1Group[5] - Y1Group[9] >= -0
 LinkY1Group2[5,10]: Y1LessThan[5,10] = 0 -> Y1Group[5] - Y1Group[10]
   >= -0
 LinkY1Group2[6,0]: Y1LessThan[6,0] = 0 -> Y1Group[6] - Y1Group[0] >= -0
 LinkY1Group2[6,1]: Y1LessThan[6,1] = 0 -> Y1Group[6] - Y1Group[1] >= -0
 LinkY1Group2[6,2]: Y1LessThan[6,2] = 0 -> Y1Group[6] - Y1Group[2] >= -0
 LinkY1Group2[6,3]: Y1LessThan[6,3] = 0 -> Y1Group[6] - Y1Group[3] >= -0
 LinkY1Group2[6,4]: Y1LessThan[6,4] = 0 -> Y1Group[6] - Y1Group[4] >= -0
 LinkY1Group2[6,5]: Y1LessThan[6,5] = 0 -> Y1Group[6] - Y1Group[5] >= -0
 LinkY1Group2[6,7]: Y1LessThan[6,7] = 0 -> Y1Group[6] - Y1Group[7] >= -0
 LinkY1Group2[6,8]: Y1LessThan[6,8] = 0 -> Y1Group[6] - Y1Group[8] >= -0
 LinkY1Group2[6,9]: Y1LessThan[6,9] = 0 -> Y1Group[6] - Y1Group[9] >= -0
 LinkY1Group2[6,10]: Y1LessThan[6,10] = 0 -> Y1Group[6] - Y1Group[10]
   >= -0
 LinkY1Group2[7,0]: Y1LessThan[7,0] = 0 -> Y1Group[7] - Y1Group[0] >= -0
 LinkY1Group2[7,1]: Y1LessThan[7,1] = 0 -> Y1Group[7] - Y1Group[1] >= -0
 LinkY1Group2[7,2]: Y1LessThan[7,2] = 0 -> Y1Group[7] - Y1Group[2] >= -0
 LinkY1Group2[7,3]: Y1LessThan[7,3] = 0 -> Y1Group[7] - Y1Group[3] >= -0
 LinkY1Group2[7,4]: Y1LessThan[7,4] = 0 -> Y1Group[7] - Y1Group[4] >= -0
 LinkY1Group2[7,5]: Y1LessThan[7,5] = 0 -> Y1Group[7] - Y1Group[5] >= -0
 LinkY1Group2[7,6]: Y1LessThan[7,6] = 0 -> Y1Group[7] - Y1Group[6] >= -0
 LinkY1Group2[7,8]: Y1LessThan[7,8] = 0 -> Y1Group[7] - Y1Group[8] >= -0
 LinkY1Group2[7,9]: Y1LessThan[7,9] = 0 -> Y1Group[7] - Y1Group[9] >= -0
 LinkY1Group2[7,10]: Y1LessThan[7,10] = 0 -> Y1Group[7] - Y1Group[10]
   >= -0
 LinkY1Group2[8,0]: Y1LessThan[8,0] = 0 -> Y1Group[8] - Y1Group[0] >= -0
 LinkY1Group2[8,1]: Y1LessThan[8,1] = 0 -> Y1Group[8] - Y1Group[1] >= -0
 LinkY1Group2[8,2]: Y1LessThan[8,2] = 0 -> Y1Group[8] - Y1Group[2] >= -0
 LinkY1Group2[8,3]: Y1LessThan[8,3] = 0 -> Y1Group[8] - Y1Group[3] >= -0
 LinkY1Group2[8,4]: Y1LessThan[8,4] = 0 -> Y1Group[8] - Y1Group[4] >= -0
 LinkY1Group2[8,5]: Y1LessThan[8,5] = 0 -> Y1Group[8] - Y1Group[5] >= -0
 LinkY1Group2[8,6]: Y1LessThan[8,6] = 0 -> Y1Group[8] - Y1Group[6] >= -0
 LinkY1Group2[8,7]: Y1LessThan[8,7] = 0 -> Y1Group[8] - Y1Group[7] >= -0
 LinkY1Group2[8,9]: Y1LessThan[8,9] = 0 -> Y1Group[8] - Y1Group[9] >= -0
 LinkY1Group2[8,10]: Y1LessThan[8,10] = 0 -> Y1Group[8] - Y1Group[10]
   >= -0
 LinkY1Group2[9,0]: Y1LessThan[9,0] = 0 -> Y1Group[9] - Y1Group[0] >= -0
 LinkY1Group2[9,1]: Y1LessThan[9,1] = 0 -> Y1Group[9] - Y1Group[1] >= -0
 LinkY1Group2[9,2]: Y1LessThan[9,2] = 0 -> Y1Group[9] - Y1Group[2] >= -0
 LinkY1Group2[9,3]: Y1LessThan[9,3] = 0 -> Y1Group[9] - Y1Group[3] >= -0
 LinkY1Group2[9,4]: Y1LessThan[9,4] = 0 -> Y1Group[9] - Y1Group[4] >= -0
 LinkY1Group2[9,5]: Y1LessThan[9,5] = 0 -> Y1Group[9] - Y1Group[5] >= -0
 LinkY1Group2[9,6]: Y1LessThan[9,6] = 0 -> Y1Group[9] - Y1Group[6] >= -0
 LinkY1Group2[9,7]: Y1LessThan[9,7] = 0 -> Y1Group[9] - Y1Group[7] >= -0
 LinkY1Group2[9,8]: Y1LessThan[9,8] = 0 -> Y1Group[9] - Y1Group[8] >= -0
 LinkY1Group2[9,10]: Y1LessThan[9,10] = 0 -> Y1Group[9] - Y1Group[10]
   >= -0
 LinkY1Group2[10,0]: Y1LessThan[10,0] = 0 -> Y1Group[10] - Y1Group[0]
   >= -0
 LinkY1Group2[10,1]: Y1LessThan[10,1] = 0 -> Y1Group[10] - Y1Group[1]
   >= -0
 LinkY1Group2[10,2]: Y1LessThan[10,2] = 0 -> Y1Group[10] - Y1Group[2]
   >= -0
 LinkY1Group2[10,3]: Y1LessThan[10,3] = 0 -> Y1Group[10] - Y1Group[3]
   >= -0
 LinkY1Group2[10,4]: Y1LessThan[10,4] = 0 -> Y1Group[10] - Y1Group[4]
   >= -0
 LinkY1Group2[10,5]: Y1LessThan[10,5] = 0 -> Y1Group[10] - Y1Group[5]
   >= -0
 LinkY1Group2[10,6]: Y1LessThan[10,6] = 0 -> Y1Group[10] - Y1Group[6]
   >= -0
 LinkY1Group2[10,7]: Y1LessThan[10,7] = 0 -> Y1Group[10] - Y1Group[7]
   >= -0
 LinkY1Group2[10,8]: Y1LessThan[10,8] = 0 -> Y1Group[10] - Y1Group[8]
   >= -0
 LinkY1Group2[10,9]: Y1LessThan[10,9] = 0 -> Y1Group[10] - Y1Group[9]
   >= -0
 LinkStartColumn1[0,1]: X0LessThan[0,1] = 1 -> StartColumn[0]
   - StartColumn[1] <= -1
 LinkStartColumn1[0,2]: X0LessThan[0,2] = 1 -> StartColumn[0]
   - StartColumn[2] <= -1
 LinkStartColumn1[0,3]: X0LessThan[0,3] = 1 -> StartColumn[0]
   - StartColumn[3] <= -1
 LinkStartColumn1[0,4]: X0LessThan[0,4] = 1 -> StartColumn[0]
   - StartColumn[4] <= -1
 LinkStartColumn1[0,5]: X0LessThan[0,5] = 1 -> StartColumn[0]
   - StartColumn[5] <= -1
 LinkStartColumn1[0,6]: X0LessThan[0,6] = 1 -> StartColumn[0]
   - StartColumn[6] <= -1
 LinkStartColumn1[0,7]: X0LessThan[0,7] = 1 -> StartColumn[0]
   - StartColumn[7] <= -1
 LinkStartColumn1[0,8]: X0LessThan[0,8] = 1 -> StartColumn[0]
   - StartColumn[8] <= -1
 LinkStartColumn1[0,9]: X0LessThan[0,9] = 1 -> StartColumn[0]
   - StartColumn[9] <= -1
 LinkStartColumn1[0,10]: X0LessThan[0,10] = 1 -> StartColumn[0]
   - StartColumn[10] <= -1
 LinkStartColumn1[1,0]: X0LessThan[1,0] = 1 -> StartColumn[1]
   - StartColumn[0] <= -1
 LinkStartColumn1[1,2]: X0LessThan[1,2] = 1 -> StartColumn[1]
   - StartColumn[2] <= -1
 LinkStartColumn1[1,3]: X0LessThan[1,3] = 1 -> StartColumn[1]
   - StartColumn[3] <= -1
 LinkStartColumn1[1,4]: X0LessThan[1,4] = 1 -> StartColumn[1]
   - StartColumn[4] <= -1
 LinkStartColumn1[1,5]: X0LessThan[1,5] = 1 -> StartColumn[1]
   - StartColumn[5] <= -1
 LinkStartColumn1[1,6]: X0LessThan[1,6] = 1 -> StartColumn[1]
   - StartColumn[6] <= -1
 LinkStartColumn1[1,7]: X0LessThan[1,7] = 1 -> StartColumn[1]
   - StartColumn[7] <= -1
 LinkStartColumn1[1,8]: X0LessThan[1,8] = 1 -> StartColumn[1]
   - StartColumn[8] <= -1
 LinkStartColumn1[1,9]: X0LessThan[1,9] = 1 -> StartColumn[1]
   - StartColumn[9] <= -1
 LinkStartColumn1[1,10]: X0LessThan[1,10] = 1 -> StartColumn[1]
   - StartColumn[10] <= -1
 LinkStartColumn1[2,0]: X0LessThan[2,0] = 1 -> StartColumn[2]
   - StartColumn[0] <= -1
 LinkStartColumn1[2,1]: X0LessThan[2,1] = 1 -> StartColumn[2]
   - StartColumn[1] <= -1
 LinkStartColumn1[2,3]: X0LessThan[2,3] = 1 -> StartColumn[2]
   - StartColumn[3] <= -1
 LinkStartColumn1[2,4]: X0LessThan[2,4] = 1 -> StartColumn[2]
   - StartColumn[4] <= -1
 LinkStartColumn1[2,5]: X0LessThan[2,5] = 1 -> StartColumn[2]
   - StartColumn[5] <= -1
 LinkStartColumn1[2,6]: X0LessThan[2,6] = 1 -> StartColumn[2]
   - StartColumn[6] <= -1
 LinkStartColumn1[2,7]: X0LessThan[2,7] = 1 -> StartColumn[2]
   - StartColumn[7] <= -1
 LinkStartColumn1[2,8]: X0LessThan[2,8] = 1 -> StartColumn[2]
   - StartColumn[8] <= -1
 LinkStartColumn1[2,9]: X0LessThan[2,9] = 1 -> StartColumn[2]
   - StartColumn[9] <= -1
 LinkStartColumn1[2,10]: X0LessThan[2,10] = 1 -> StartColumn[2]
   - StartColumn[10] <= -1
 LinkStartColumn1[3,0]: X0LessThan[3,0] = 1 -> StartColumn[3]
   - StartColumn[0] <= -1
 LinkStartColumn1[3,1]: X0LessThan[3,1] = 1 -> StartColumn[3]
   - StartColumn[1] <= -1
 LinkStartColumn1[3,2]: X0LessThan[3,2] = 1 -> StartColumn[3]
   - StartColumn[2] <= -1
 LinkStartColumn1[3,4]: X0LessThan[3,4] = 1 -> StartColumn[3]
   - StartColumn[4] <= -1
 LinkStartColumn1[3,5]: X0LessThan[3,5] = 1 -> StartColumn[3]
   - StartColumn[5] <= -1
 LinkStartColumn1[3,6]: X0LessThan[3,6] = 1 -> StartColumn[3]
   - StartColumn[6] <= -1
 LinkStartColumn1[3,7]: X0LessThan[3,7] = 1 -> StartColumn[3]
   - StartColumn[7] <= -1
 LinkStartColumn1[3,8]: X0LessThan[3,8] = 1 -> StartColumn[3]
   - StartColumn[8] <= -1
 LinkStartColumn1[3,9]: X0LessThan[3,9] = 1 -> StartColumn[3]
   - StartColumn[9] <= -1
 LinkStartColumn1[3,10]: X0LessThan[3,10] = 1 -> StartColumn[3]
   - StartColumn[10] <= -1
 LinkStartColumn1[4,0]: X0LessThan[4,0] = 1 -> StartColumn[4]
   - StartColumn[0] <= -1
 LinkStartColumn1[4,1]: X0LessThan[4,1] = 1 -> StartColumn[4]
   - StartColumn[1] <= -1
 LinkStartColumn1[4,2]: X0LessThan[4,2] = 1 -> StartColumn[4]
   - StartColumn[2] <= -1
 LinkStartColumn1[4,3]: X0LessThan[4,3] = 1 -> StartColumn[4]
   - StartColumn[3] <= -1
 LinkStartColumn1[4,5]: X0LessThan[4,5] = 1 -> StartColumn[4]
   - StartColumn[5] <= -1
 LinkStartColumn1[4,6]: X0LessThan[4,6] = 1 -> StartColumn[4]
   - StartColumn[6] <= -1
 LinkStartColumn1[4,7]: X0LessThan[4,7] = 1 -> StartColumn[4]
   - StartColumn[7] <= -1
 LinkStartColumn1[4,8]: X0LessThan[4,8] = 1 -> StartColumn[4]
   - StartColumn[8] <= -1
 LinkStartColumn1[4,9]: X0LessThan[4,9] = 1 -> StartColumn[4]
   - StartColumn[9] <= -1
 LinkStartColumn1[4,10]: X0LessThan[4,10] = 1 -> StartColumn[4]
   - StartColumn[10] <= -1
 LinkStartColumn1[5,0]: X0LessThan[5,0] = 1 -> StartColumn[5]
   - StartColumn[0] <= -1
 LinkStartColumn1[5,1]: X0LessThan[5,1] = 1 -> StartColumn[5]
   - StartColumn[1] <= -1
 LinkStartColumn1[5,2]: X0LessThan[5,2] = 1 -> StartColumn[5]
   - StartColumn[2] <= -1
 LinkStartColumn1[5,3]: X0LessThan[5,3] = 1 -> StartColumn[5]
   - StartColumn[3] <= -1
 LinkStartColumn1[5,4]: X0LessThan[5,4] = 1 -> StartColumn[5]
   - StartColumn[4] <= -1
 LinkStartColumn1[5,6]: X0LessThan[5,6] = 1 -> StartColumn[5]
   - StartColumn[6] <= -1
 LinkStartColumn1[5,7]: X0LessThan[5,7] = 1 -> StartColumn[5]
   - StartColumn[7] <= -1
 LinkStartColumn1[5,8]: X0LessThan[5,8] = 1 -> StartColumn[5]
   - StartColumn[8] <= -1
 LinkStartColumn1[5,9]: X0LessThan[5,9] = 1 -> StartColumn[5]
   - StartColumn[9] <= -1
 LinkStartColumn1[5,10]: X0LessThan[5,10] = 1 -> StartColumn[5]
   - StartColumn[10] <= -1
 LinkStartColumn1[6,0]: X0LessThan[6,0] = 1 -> StartColumn[6]
   - StartColumn[0] <= -1
 LinkStartColumn1[6,1]: X0LessThan[6,1] = 1 -> StartColumn[6]
   - StartColumn[1] <= -1
 LinkStartColumn1[6,2]: X0LessThan[6,2] = 1 -> StartColumn[6]
   - StartColumn[2] <= -1
 LinkStartColumn1[6,3]: X0LessThan[6,3] = 1 -> StartColumn[6]
   - StartColumn[3] <= -1
 LinkStartColumn1[6,4]: X0LessThan[6,4] = 1 -> StartColumn[6]
   - StartColumn[4] <= -1
 LinkStartColumn1[6,5]: X0LessThan[6,5] = 1 -> StartColumn[6]
   - StartColumn[5] <= -1
 LinkStartColumn1[6,7]: X0LessThan[6,7] = 1 -> StartColumn[6]
   - StartColumn[7] <= -1
 LinkStartColumn1[6,8]: X0LessThan[6,8] = 1 -> StartColumn[6]
   - StartColumn[8] <= -1
 LinkStartColumn1[6,9]: X0LessThan[6,9] = 1 -> StartColumn[6]
   - StartColumn[9] <= -1
 LinkStartColumn1[6,10]: X0LessThan[6,10] = 1 -> StartColumn[6]
   - StartColumn[10] <= -1
 LinkStartColumn1[7,0]: X0LessThan[7,0] = 1 -> StartColumn[7]
   - StartColumn[0] <= -1
 LinkStartColumn1[7,1]: X0LessThan[7,1] = 1 -> StartColumn[7]
   - StartColumn[1] <= -1
 LinkStartColumn1[7,2]: X0LessThan[7,2] = 1 -> StartColumn[7]
   - StartColumn[2] <= -1
 LinkStartColumn1[7,3]: X0LessThan[7,3] = 1 -> StartColumn[7]
   - StartColumn[3] <= -1
 LinkStartColumn1[7,4]: X0LessThan[7,4] = 1 -> StartColumn[7]
   - StartColumn[4] <= -1
 LinkStartColumn1[7,5]: X0LessThan[7,5] = 1 -> StartColumn[7]
   - StartColumn[5] <= -1
 LinkStartColumn1[7,6]: X0LessThan[7,6] = 1 -> StartColumn[7]
   - StartColumn[6] <= -1
 LinkStartColumn1[7,8]: X0LessThan[7,8] = 1 -> StartColumn[7]
   - StartColumn[8] <= -1
 LinkStartColumn1[7,9]: X0LessThan[7,9] = 1 -> StartColumn[7]
   - StartColumn[9] <= -1
 LinkStartColumn1[7,10]: X0LessThan[7,10] = 1 -> StartColumn[7]
   - StartColumn[10] <= -1
 LinkStartColumn1[8,0]: X0LessThan[8,0] = 1 -> StartColumn[8]
   - StartColumn[0] <= -1
 LinkStartColumn1[8,1]: X0LessThan[8,1] = 1 -> StartColumn[8]
   - StartColumn[1] <= -1
 LinkStartColumn1[8,2]: X0LessThan[8,2] = 1 -> StartColumn[8]
   - StartColumn[2] <= -1
 LinkStartColumn1[8,3]: X0LessThan[8,3] = 1 -> StartColumn[8]
   - StartColumn[3] <= -1
 LinkStartColumn1[8,4]: X0LessThan[8,4] = 1 -> StartColumn[8]
   - StartColumn[4] <= -1
 LinkStartColumn1[8,5]: X0LessThan[8,5] = 1 -> StartColumn[8]
   - StartColumn[5] <= -1
 LinkStartColumn1[8,6]: X0LessThan[8,6] = 1 -> StartColumn[8]
   - StartColumn[6] <= -1
 LinkStartColumn1[8,7]: X0LessThan[8,7] = 1 -> StartColumn[8]
   - StartColumn[7] <= -1
 LinkStartColumn1[8,9]: X0LessThan[8,9] = 1 -> StartColumn[8]
   - StartColumn[9] <= -1
 LinkStartColumn1[8,10]: X0LessThan[8,10] = 1 -> StartColumn[8]
   - StartColumn[10] <= -1
 LinkStartColumn1[9,0]: X0LessThan[9,0] = 1 -> StartColumn[9]
   - StartColumn[0] <= -1
 LinkStartColumn1[9,1]: X0LessThan[9,1] = 1 -> StartColumn[9]
   - StartColumn[1] <= -1
 LinkStartColumn1[9,2]: X0LessThan[9,2] = 1 -> StartColumn[9]
   - StartColumn[2] <= -1
 LinkStartColumn1[9,3]: X0LessThan[9,3] = 1 -> StartColumn[9]
   - StartColumn[3] <= -1
 LinkStartColumn1[9,4]: X0LessThan[9,4] = 1 -> StartColumn[9]
   - StartColumn[4] <= -1
 LinkStartColumn1[9,5]: X0LessThan[9,5] = 1 -> StartColumn[9]
   - StartColumn[5] <= -1
 LinkStartColumn1[9,6]: X0LessThan[9,6] = 1 -> StartColumn[9]
   - StartColumn[6] <= -1
 LinkStartColumn1[9,7]: X0LessThan[9,7] = 1 -> StartColumn[9]
   - StartColumn[7] <= -1
 LinkStartColumn1[9,8]: X0LessThan[9,8] = 1 -> StartColumn[9]
   - StartColumn[8] <= -1
 LinkStartColumn1[9,10]: X0LessThan[9,10] = 1 -> StartColumn[9]
   - StartColumn[10] <= -1
 LinkStartColumn1[10,0]: X0LessThan[10,0] = 1 -> StartColumn[10]
   - StartColumn[0] <= -1
 LinkStartColumn1[10,1]: X0LessThan[10,1] = 1 -> StartColumn[10]
   - StartColumn[1] <= -1
 LinkStartColumn1[10,2]: X0LessThan[10,2] = 1 -> StartColumn[10]
   - StartColumn[2] <= -1
 LinkStartColumn1[10,3]: X0LessThan[10,3] = 1 -> StartColumn[10]
   - StartColumn[3] <= -1
 LinkStartColumn1[10,4]: X0LessThan[10,4] = 1 -> StartColumn[10]
   - StartColumn[4] <= -1
 LinkStartColumn1[10,5]: X0LessThan[10,5] = 1 -> StartColumn[10]
   - StartColumn[5] <= -1
 LinkStartColumn1[10,6]: X0LessThan[10,6] = 1 -> StartColumn[10]
   - StartColumn[6] <= -1
 LinkStartColumn1[10,7]: X0LessThan[10,7] = 1 -> StartColumn[10]
   - StartColumn[7] <= -1
 LinkStartColumn1[10,8]: X0LessThan[10,8] = 1 -> StartColumn[10]
   - StartColumn[8] <= -1
 LinkStartColumn1[10,9]: X0LessThan[10,9] = 1 -> StartColumn[10]
   - StartColumn[9] <= -1
 LinkStartColumn2[0,1]: X0LessThan[0,1] = 0 -> StartColumn[0]
   - StartColumn[1] >= -0
 LinkStartColumn2[0,2]: X0LessThan[0,2] = 0 -> StartColumn[0]
   - StartColumn[2] >= -0
 LinkStartColumn2[0,3]: X0LessThan[0,3] = 0 -> StartColumn[0]
   - StartColumn[3] >= -0
 LinkStartColumn2[0,4]: X0LessThan[0,4] = 0 -> StartColumn[0]
   - StartColumn[4] >= -0
 LinkStartColumn2[0,5]: X0LessThan[0,5] = 0 -> StartColumn[0]
   - StartColumn[5] >= -0
 LinkStartColumn2[0,6]: X0LessThan[0,6] = 0 -> StartColumn[0]
   - StartColumn[6] >= -0
 LinkStartColumn2[0,7]: X0LessThan[0,7] = 0 -> StartColumn[0]
   - StartColumn[7] >= -0
 LinkStartColumn2[0,8]: X0LessThan[0,8] = 0 -> StartColumn[0]
   - StartColumn[8] >= -0
 LinkStartColumn2[0,9]: X0LessThan[0,9] = 0 -> StartColumn[0]
   - StartColumn[9] >= -0
 LinkStartColumn2[0,10]: X0LessThan[0,10] = 0 -> StartColumn[0]
   - StartColumn[10] >= -0
 LinkStartColumn2[1,0]: X0LessThan[1,0] = 0 -> StartColumn[1]
   - StartColumn[0] >= -0
 LinkStartColumn2[1,2]: X0LessThan[1,2] = 0 -> StartColumn[1]
   - StartColumn[2] >= -0
 LinkStartColumn2[1,3]: X0LessThan[1,3] = 0 -> StartColumn[1]
   - StartColumn[3] >= -0
 LinkStartColumn2[1,4]: X0LessThan[1,4] = 0 -> StartColumn[1]
   - StartColumn[4] >= -0
 LinkStartColumn2[1,5]: X0LessThan[1,5] = 0 -> StartColumn[1]
   - StartColumn[5] >= -0
 LinkStartColumn2[1,6]: X0LessThan[1,6] = 0 -> StartColumn[1]
   - StartColumn[6] >= -0
 LinkStartColumn2[1,7]: X0LessThan[1,7] = 0 -> StartColumn[1]
   - StartColumn[7] >= -0
 LinkStartColumn2[1,8]: X0LessThan[1,8] = 0 -> StartColumn[1]
   - StartColumn[8] >= -0
 LinkStartColumn2[1,9]: X0LessThan[1,9] = 0 -> StartColumn[1]
   - StartColumn[9] >= -0
 LinkStartColumn2[1,10]: X0LessThan[1,10] = 0 -> StartColumn[1]
   - StartColumn[10] >= -0
 LinkStartColumn2[2,0]: X0LessThan[2,0] = 0 -> StartColumn[2]
   - StartColumn[0] >= -0
 LinkStartColumn2[2,1]: X0LessThan[2,1] = 0 -> StartColumn[2]
   - StartColumn[1] >= -0
 LinkStartColumn2[2,3]: X0LessThan[2,3] = 0 -> StartColumn[2]
   - StartColumn[3] >= -0
 LinkStartColumn2[2,4]: X0LessThan[2,4] = 0 -> StartColumn[2]
   - StartColumn[4] >= -0
 LinkStartColumn2[2,5]: X0LessThan[2,5] = 0 -> StartColumn[2]
   - StartColumn[5] >= -0
 LinkStartColumn2[2,6]: X0LessThan[2,6] = 0 -> StartColumn[2]
   - StartColumn[6] >= -0
 LinkStartColumn2[2,7]: X0LessThan[2,7] = 0 -> StartColumn[2]
   - StartColumn[7] >= -0
 LinkStartColumn2[2,8]: X0LessThan[2,8] = 0 -> StartColumn[2]
   - StartColumn[8] >= -0
 LinkStartColumn2[2,9]: X0LessThan[2,9] = 0 -> StartColumn[2]
   - StartColumn[9] >= -0
 LinkStartColumn2[2,10]: X0LessThan[2,10] = 0 -> StartColumn[2]
   - StartColumn[10] >= -0
 LinkStartColumn2[3,0]: X0LessThan[3,0] = 0 -> StartColumn[3]
   - StartColumn[0] >= -0
 LinkStartColumn2[3,1]: X0LessThan[3,1] = 0 -> StartColumn[3]
   - StartColumn[1] >= -0
 LinkStartColumn2[3,2]: X0LessThan[3,2] = 0 -> StartColumn[3]
   - StartColumn[2] >= -0
 LinkStartColumn2[3,4]: X0LessThan[3,4] = 0 -> StartColumn[3]
   - StartColumn[4] >= -0
 LinkStartColumn2[3,5]: X0LessThan[3,5] = 0 -> StartColumn[3]
   - StartColumn[5] >= -0
 LinkStartColumn2[3,6]: X0LessThan[3,6] = 0 -> StartColumn[3]
   - StartColumn[6] >= -0
 LinkStartColumn2[3,7]: X0LessThan[3,7] = 0 -> StartColumn[3]
   - StartColumn[7] >= -0
 LinkStartColumn2[3,8]: X0LessThan[3,8] = 0 -> StartColumn[3]
   - StartColumn[8] >= -0
 LinkStartColumn2[3,9]: X0LessThan[3,9] = 0 -> StartColumn[3]
   - StartColumn[9] >= -0
 LinkStartColumn2[3,10]: X0LessThan[3,10] = 0 -> StartColumn[3]
   - StartColumn[10] >= -0
 LinkStartColumn2[4,0]: X0LessThan[4,0] = 0 -> StartColumn[4]
   - StartColumn[0] >= -0
 LinkStartColumn2[4,1]: X0LessThan[4,1] = 0 -> StartColumn[4]
   - StartColumn[1] >= -0
 LinkStartColumn2[4,2]: X0LessThan[4,2] = 0 -> StartColumn[4]
   - StartColumn[2] >= -0
 LinkStartColumn2[4,3]: X0LessThan[4,3] = 0 -> StartColumn[4]
   - StartColumn[3] >= -0
 LinkStartColumn2[4,5]: X0LessThan[4,5] = 0 -> StartColumn[4]
   - StartColumn[5] >= -0
 LinkStartColumn2[4,6]: X0LessThan[4,6] = 0 -> StartColumn[4]
   - StartColumn[6] >= -0
 LinkStartColumn2[4,7]: X0LessThan[4,7] = 0 -> StartColumn[4]
   - StartColumn[7] >= -0
 LinkStartColumn2[4,8]: X0LessThan[4,8] = 0 -> StartColumn[4]
   - StartColumn[8] >= -0
 LinkStartColumn2[4,9]: X0LessThan[4,9] = 0 -> StartColumn[4]
   - StartColumn[9] >= -0
 LinkStartColumn2[4,10]: X0LessThan[4,10] = 0 -> StartColumn[4]
   - StartColumn[10] >= -0
 LinkStartColumn2[5,0]: X0LessThan[5,0] = 0 -> StartColumn[5]
   - StartColumn[0] >= -0
 LinkStartColumn2[5,1]: X0LessThan[5,1] = 0 -> StartColumn[5]
   - StartColumn[1] >= -0
 LinkStartColumn2[5,2]: X0LessThan[5,2] = 0 -> StartColumn[5]
   - StartColumn[2] >= -0
 LinkStartColumn2[5,3]: X0LessThan[5,3] = 0 -> StartColumn[5]
   - StartColumn[3] >= -0
 LinkStartColumn2[5,4]: X0LessThan[5,4] = 0 -> StartColumn[5]
   - StartColumn[4] >= -0
 LinkStartColumn2[5,6]: X0LessThan[5,6] = 0 -> StartColumn[5]
   - StartColumn[6] >= -0
 LinkStartColumn2[5,7]: X0LessThan[5,7] = 0 -> StartColumn[5]
   - StartColumn[7] >= -0
 LinkStartColumn2[5,8]: X0LessThan[5,8] = 0 -> StartColumn[5]
   - StartColumn[8] >= -0
 LinkStartColumn2[5,9]: X0LessThan[5,9] = 0 -> StartColumn[5]
   - StartColumn[9] >= -0
 LinkStartColumn2[5,10]: X0LessThan[5,10] = 0 -> StartColumn[5]
   - StartColumn[10] >= -0
 LinkStartColumn2[6,0]: X0LessThan[6,0] = 0 -> StartColumn[6]
   - StartColumn[0] >= -0
 LinkStartColumn2[6,1]: X0LessThan[6,1] = 0 -> StartColumn[6]
   - StartColumn[1] >= -0
 LinkStartColumn2[6,2]: X0LessThan[6,2] = 0 -> StartColumn[6]
   - StartColumn[2] >= -0
 LinkStartColumn2[6,3]: X0LessThan[6,3] = 0 -> StartColumn[6]
   - StartColumn[3] >= -0
 LinkStartColumn2[6,4]: X0LessThan[6,4] = 0 -> StartColumn[6]
   - StartColumn[4] >= -0
 LinkStartColumn2[6,5]: X0LessThan[6,5] = 0 -> StartColumn[6]
   - StartColumn[5] >= -0
 LinkStartColumn2[6,7]: X0LessThan[6,7] = 0 -> StartColumn[6]
   - StartColumn[7] >= -0
 LinkStartColumn2[6,8]: X0LessThan[6,8] = 0 -> StartColumn[6]
   - StartColumn[8] >= -0
 LinkStartColumn2[6,9]: X0LessThan[6,9] = 0 -> StartColumn[6]
   - StartColumn[9] >= -0
 LinkStartColumn2[6,10]: X0LessThan[6,10] = 0 -> StartColumn[6]
   - StartColumn[10] >= -0
 LinkStartColumn2[7,0]: X0LessThan[7,0] = 0 -> StartColumn[7]
   - StartColumn[0] >= -0
 LinkStartColumn2[7,1]: X0LessThan[7,1] = 0 -> StartColumn[7]
   - StartColumn[1] >= -0
 LinkStartColumn2[7,2]: X0LessThan[7,2] = 0 -> StartColumn[7]
   - StartColumn[2] >= -0
 LinkStartColumn2[7,3]: X0LessThan[7,3] = 0 -> StartColumn[7]
   - StartColumn[3] >= -0
 LinkStartColumn2[7,4]: X0LessThan[7,4] = 0 -> StartColumn[7]
   - StartColumn[4] >= -0
 LinkStartColumn2[7,5]: X0LessThan[7,5] = 0 -> StartColumn[7]
   - StartColumn[5] >= -0
 LinkStartColumn2[7,6]: X0LessThan[7,6] = 0 -> StartColumn[7]
   - StartColumn[6] >= -0
 LinkStartColumn2[7,8]: X0LessThan[7,8] = 0 -> StartColumn[7]
   - StartColumn[8] >= -0
 LinkStartColumn2[7,9]: X0LessThan[7,9] = 0 -> StartColumn[7]
   - StartColumn[9] >= -0
 LinkStartColumn2[7,10]: X0LessThan[7,10] = 0 -> StartColumn[7]
   - StartColumn[10] >= -0
 LinkStartColumn2[8,0]: X0LessThan[8,0] = 0 -> StartColumn[8]
   - StartColumn[0] >= -0
 LinkStartColumn2[8,1]: X0LessThan[8,1] = 0 -> StartColumn[8]
   - StartColumn[1] >= -0
 LinkStartColumn2[8,2]: X0LessThan[8,2] = 0 -> StartColumn[8]
   - StartColumn[2] >= -0
 LinkStartColumn2[8,3]: X0LessThan[8,3] = 0 -> StartColumn[8]
   - StartColumn[3] >= -0
 LinkStartColumn2[8,4]: X0LessThan[8,4] = 0 -> StartColumn[8]
   - StartColumn[4] >= -0
 LinkStartColumn2[8,5]: X0LessThan[8,5] = 0 -> StartColumn[8]
   - StartColumn[5] >= -0
 LinkStartColumn2[8,6]: X0LessThan[8,6] = 0 -> StartColumn[8]
   - StartColumn[6] >= -0
 LinkStartColumn2[8,7]: X0LessThan[8,7] = 0 -> StartColumn[8]
   - StartColumn[7] >= -0
 LinkStartColumn2[8,9]: X0LessThan[8,9] = 0 -> StartColumn[8]
   - StartColumn[9] >= -0
 LinkStartColumn2[8,10]: X0LessThan[8,10] = 0 -> StartColumn[8]
   - StartColumn[10] >= -0
 LinkStartColumn2[9,0]: X0LessThan[9,0] = 0 -> StartColumn[9]
   - StartColumn[0] >= -0
 LinkStartColumn2[9,1]: X0LessThan[9,1] = 0 -> StartColumn[9]
   - StartColumn[1] >= -0
 LinkStartColumn2[9,2]: X0LessThan[9,2] = 0 -> StartColumn[9]
   - StartColumn[2] >= -0
 LinkStartColumn2[9,3]: X0LessThan[9,3] = 0 -> StartColumn[9]
   - StartColumn[3] >= -0
 LinkStartColumn2[9,4]: X0LessThan[9,4] = 0 -> StartColumn[9]
   - StartColumn[4] >= -0
 LinkStartColumn2[9,5]: X0LessThan[9,5] = 0 -> StartColumn[9]
   - StartColumn[5] >= -0
 LinkStartColumn2[9,6]: X0LessThan[9,6] = 0 -> StartColumn[9]
   - StartColumn[6] >= -0
 LinkStartColumn2[9,7]: X0LessThan[9,7] = 0 -> StartColumn[9]
   - StartColumn[7] >= -0
 LinkStartColumn2[9,8]: X0LessThan[9,8] = 0 -> StartColumn[9]
   - StartColumn[8] >= -0
 LinkStartColumn2[9,10]: X0LessThan[9,10] = 0 -> StartColumn[9]
   - StartColumn[10] >= -0
 LinkStartColumn2[10,0]: X0LessThan[10,0] = 0 -> StartColumn[10]
   - StartColumn[0] >= -0
 LinkStartColumn2[10,1]: X0LessThan[10,1] = 0 -> StartColumn[10]
   - StartColumn[1] >= -0
 LinkStartColumn2[10,2]: X0LessThan[10,2] = 0 -> StartColumn[10]
   - StartColumn[2] >= -0
 LinkStartColumn2[10,3]: X0LessThan[10,3] = 0 -> StartColumn[10]
   - StartColumn[3] >= -0
 LinkStartColumn2[10,4]: X0LessThan[10,4] = 0 -> StartColumn[10]
   - StartColumn[4] >= -0
 LinkStartColumn2[10,5]: X0LessThan[10,5] = 0 -> StartColumn[10]
   - StartColumn[5] >= -0
 LinkStartColumn2[10,6]: X0LessThan[10,6] = 0 -> StartColumn[10]
   - StartColumn[6] >= -0
 LinkStartColumn2[10,7]: X0LessThan[10,7] = 0 -> StartColumn[10]
   - StartColumn[7] >= -0
 LinkStartColumn2[10,8]: X0LessThan[10,8] = 0 -> StartColumn[10]
   - StartColumn[8] >= -0
 LinkStartColumn2[10,9]: X0LessThan[10,9] = 0 -> StartColumn[10]
   - StartColumn[9] >= -0
 LinkEndColumn1[0,1]: X1LessThan[0,1] = 1 -> EndColumn[0] - EndColumn[1]
   <= -1
 LinkEndColumn1[0,2]: X1LessThan[0,2] = 1 -> EndColumn[0] - EndColumn[2]
   <= -1
 LinkEndColumn1[0,3]: X1LessThan[0,3] = 1 -> EndColumn[0] - EndColumn[3]
   <= -1
 LinkEndColumn1[0,4]: X1LessThan[0,4] = 1 -> EndColumn[0] - EndColumn[4]
   <= -1
 LinkEndColumn1[0,5]: X1LessThan[0,5] = 1 -> EndColumn[0] - EndColumn[5]
   <= -1
 LinkEndColumn1[0,6]: X1LessThan[0,6] = 1 -> EndColumn[0] - EndColumn[6]
   <= -1
 LinkEndColumn1[0,7]: X1LessThan[0,7] = 1 -> EndColumn[0] - EndColumn[7]
   <= -1
 LinkEndColumn1[0,8]: X1LessThan[0,8] = 1 -> EndColumn[0] - EndColumn[8]
   <= -1
 LinkEndColumn1[0,9]: X1LessThan[0,9] = 1 -> EndColumn[0] - EndColumn[9]
   <= -1
 LinkEndColumn1[0,10]: X1LessThan[0,10] = 1 -> EndColumn[0] - EndColumn[10]
   <= -1
 LinkEndColumn1[1,0]: X1LessThan[1,0] = 1 -> EndColumn[1] - EndColumn[0]
   <= -1
 LinkEndColumn1[1,2]: X1LessThan[1,2] = 1 -> EndColumn[1] - EndColumn[2]
   <= -1
 LinkEndColumn1[1,3]: X1LessThan[1,3] = 1 -> EndColumn[1] - EndColumn[3]
   <= -1
 LinkEndColumn1[1,4]: X1LessThan[1,4] = 1 -> EndColumn[1] - EndColumn[4]
   <= -1
 LinkEndColumn1[1,5]: X1LessThan[1,5] = 1 -> EndColumn[1] - EndColumn[5]
   <= -1
 LinkEndColumn1[1,6]: X1LessThan[1,6] = 1 -> EndColumn[1] - EndColumn[6]
   <= -1
 LinkEndColumn1[1,7]: X1LessThan[1,7] = 1 -> EndColumn[1] - EndColumn[7]
   <= -1
 LinkEndColumn1[1,8]: X1LessThan[1,8] = 1 -> EndColumn[1] - EndColumn[8]
   <= -1
 LinkEndColumn1[1,9]: X1LessThan[1,9] = 1 -> EndColumn[1] - EndColumn[9]
   <= -1
 LinkEndColumn1[1,10]: X1LessThan[1,10] = 1 -> EndColumn[1] - EndColumn[10]
   <= -1
 LinkEndColumn1[2,0]: X1LessThan[2,0] = 1 -> EndColumn[2] - EndColumn[0]
   <= -1
 LinkEndColumn1[2,1]: X1LessThan[2,1] = 1 -> EndColumn[2] - EndColumn[1]
   <= -1
 LinkEndColumn1[2,3]: X1LessThan[2,3] = 1 -> EndColumn[2] - EndColumn[3]
   <= -1
 LinkEndColumn1[2,4]: X1LessThan[2,4] = 1 -> EndColumn[2] - EndColumn[4]
   <= -1
 LinkEndColumn1[2,5]: X1LessThan[2,5] = 1 -> EndColumn[2] - EndColumn[5]
   <= -1
 LinkEndColumn1[2,6]: X1LessThan[2,6] = 1 -> EndColumn[2] - EndColumn[6]
   <= -1
 LinkEndColumn1[2,7]: X1LessThan[2,7] = 1 -> EndColumn[2] - EndColumn[7]
   <= -1
 LinkEndColumn1[2,8]: X1LessThan[2,8] = 1 -> EndColumn[2] - EndColumn[8]
   <= -1
 LinkEndColumn1[2,9]: X1LessThan[2,9] = 1 -> EndColumn[2] - EndColumn[9]
   <= -1
 LinkEndColumn1[2,10]: X1LessThan[2,10] = 1 -> EndColumn[2] - EndColumn[10]
   <= -1
 LinkEndColumn1[3,0]: X1LessThan[3,0] = 1 -> EndColumn[3] - EndColumn[0]
   <= -1
 LinkEndColumn1[3,1]: X1LessThan[3,1] = 1 -> EndColumn[3] - EndColumn[1]
   <= -1
 LinkEndColumn1[3,2]: X1LessThan[3,2] = 1 -> EndColumn[3] - EndColumn[2]
   <= -1
 LinkEndColumn1[3,4]: X1LessThan[3,4] = 1 -> EndColumn[3] - EndColumn[4]
   <= -1
 LinkEndColumn1[3,5]: X1LessThan[3,5] = 1 -> EndColumn[3] - EndColumn[5]
   <= -1
 LinkEndColumn1[3,6]: X1LessThan[3,6] = 1 -> EndColumn[3] - EndColumn[6]
   <= -1
 LinkEndColumn1[3,7]: X1LessThan[3,7] = 1 -> EndColumn[3] - EndColumn[7]
   <= -1
 LinkEndColumn1[3,8]: X1LessThan[3,8] = 1 -> EndColumn[3] - EndColumn[8]
   <= -1
 LinkEndColumn1[3,9]: X1LessThan[3,9] = 1 -> EndColumn[3] - EndColumn[9]
   <= -1
 LinkEndColumn1[3,10]: X1LessThan[3,10] = 1 -> EndColumn[3] - EndColumn[10]
   <= -1
 LinkEndColumn1[4,0]: X1LessThan[4,0] = 1 -> EndColumn[4] - EndColumn[0]
   <= -1
 LinkEndColumn1[4,1]: X1LessThan[4,1] = 1 -> EndColumn[4] - EndColumn[1]
   <= -1
 LinkEndColumn1[4,2]: X1LessThan[4,2] = 1 -> EndColumn[4] - EndColumn[2]
   <= -1
 LinkEndColumn1[4,3]: X1LessThan[4,3] = 1 -> EndColumn[4] - EndColumn[3]
   <= -1
 LinkEndColumn1[4,5]: X1LessThan[4,5] = 1 -> EndColumn[4] - EndColumn[5]
   <= -1
 LinkEndColumn1[4,6]: X1LessThan[4,6] = 1 -> EndColumn[4] - EndColumn[6]
   <= -1
 LinkEndColumn1[4,7]: X1LessThan[4,7] = 1 -> EndColumn[4] - EndColumn[7]
   <= -1
 LinkEndColumn1[4,8]: X1LessThan[4,8] = 1 -> EndColumn[4] - EndColumn[8]
   <= -1
 LinkEndColumn1[4,9]: X1LessThan[4,9] = 1 -> EndColumn[4] - EndColumn[9]
   <= -1
 LinkEndColumn1[4,10]: X1LessThan[4,10] = 1 -> EndColumn[4] - EndColumn[10]
   <= -1
 LinkEndColumn1[5,0]: X1LessThan[5,0] = 1 -> EndColumn[5] - EndColumn[0]
   <= -1
 LinkEndColumn1[5,1]: X1LessThan[5,1] = 1 -> EndColumn[5] - EndColumn[1]
   <= -1
 LinkEndColumn1[5,2]: X1LessThan[5,2] = 1 -> EndColumn[5] - EndColumn[2]
   <= -1
 LinkEndColumn1[5,3]: X1LessThan[5,3] = 1 -> EndColumn[5] - EndColumn[3]
   <= -1
 LinkEndColumn1[5,4]: X1LessThan[5,4] = 1 -> EndColumn[5] - EndColumn[4]
   <= -1
 LinkEndColumn1[5,6]: X1LessThan[5,6] = 1 -> EndColumn[5] - EndColumn[6]
   <= -1
 LinkEndColumn1[5,7]: X1LessThan[5,7] = 1 -> EndColumn[5] - EndColumn[7]
   <= -1
 LinkEndColumn1[5,8]: X1LessThan[5,8] = 1 -> EndColumn[5] - EndColumn[8]
   <= -1
 LinkEndColumn1[5,9]: X1LessThan[5,9] = 1 -> EndColumn[5] - EndColumn[9]
   <= -1
 LinkEndColumn1[5,10]: X1LessThan[5,10] = 1 -> EndColumn[5] - EndColumn[10]
   <= -1
 LinkEndColumn1[6,0]: X1LessThan[6,0] = 1 -> EndColumn[6] - EndColumn[0]
   <= -1
 LinkEndColumn1[6,1]: X1LessThan[6,1] = 1 -> EndColumn[6] - EndColumn[1]
   <= -1
 LinkEndColumn1[6,2]: X1LessThan[6,2] = 1 -> EndColumn[6] - EndColumn[2]
   <= -1
 LinkEndColumn1[6,3]: X1LessThan[6,3] = 1 -> EndColumn[6] - EndColumn[3]
   <= -1
 LinkEndColumn1[6,4]: X1LessThan[6,4] = 1 -> EndColumn[6] - EndColumn[4]
   <= -1
 LinkEndColumn1[6,5]: X1LessThan[6,5] = 1 -> EndColumn[6] - EndColumn[5]
   <= -1
 LinkEndColumn1[6,7]: X1LessThan[6,7] = 1 -> EndColumn[6] - EndColumn[7]
   <= -1
 LinkEndColumn1[6,8]: X1LessThan[6,8] = 1 -> EndColumn[6] - EndColumn[8]
   <= -1
 LinkEndColumn1[6,9]: X1LessThan[6,9] = 1 -> EndColumn[6] - EndColumn[9]
   <= -1
 LinkEndColumn1[6,10]: X1LessThan[6,10] = 1 -> EndColumn[6] - EndColumn[10]
   <= -1
 LinkEndColumn1[7,0]: X1LessThan[7,0] = 1 -> EndColumn[7] - EndColumn[0]
   <= -1
 LinkEndColumn1[7,1]: X1LessThan[7,1] = 1 -> EndColumn[7] - EndColumn[1]
   <= -1
 LinkEndColumn1[7,2]: X1LessThan[7,2] = 1 -> EndColumn[7] - EndColumn[2]
   <= -1
 LinkEndColumn1[7,3]: X1LessThan[7,3] = 1 -> EndColumn[7] - EndColumn[3]
   <= -1
 LinkEndColumn1[7,4]: X1LessThan[7,4] = 1 -> EndColumn[7] - EndColumn[4]
   <= -1
 LinkEndColumn1[7,5]: X1LessThan[7,5] = 1 -> EndColumn[7] - EndColumn[5]
   <= -1
 LinkEndColumn1[7,6]: X1LessThan[7,6] = 1 -> EndColumn[7] - EndColumn[6]
   <= -1
 LinkEndColumn1[7,8]: X1LessThan[7,8] = 1 -> EndColumn[7] - EndColumn[8]
   <= -1
 LinkEndColumn1[7,9]: X1LessThan[7,9] = 1 -> EndColumn[7] - EndColumn[9]
   <= -1
 LinkEndColumn1[7,10]: X1LessThan[7,10] = 1 -> EndColumn[7] - EndColumn[10]
   <= -1
 LinkEndColumn1[8,0]: X1LessThan[8,0] = 1 -> EndColumn[8] - EndColumn[0]
   <= -1
 LinkEndColumn1[8,1]: X1LessThan[8,1] = 1 -> EndColumn[8] - EndColumn[1]
   <= -1
 LinkEndColumn1[8,2]: X1LessThan[8,2] = 1 -> EndColumn[8] - EndColumn[2]
   <= -1
 LinkEndColumn1[8,3]: X1LessThan[8,3] = 1 -> EndColumn[8] - EndColumn[3]
   <= -1
 LinkEndColumn1[8,4]: X1LessThan[8,4] = 1 -> EndColumn[8] - EndColumn[4]
   <= -1
 LinkEndColumn1[8,5]: X1LessThan[8,5] = 1 -> EndColumn[8] - EndColumn[5]
   <= -1
 LinkEndColumn1[8,6]: X1LessThan[8,6] = 1 -> EndColumn[8] - EndColumn[6]
   <= -1
 LinkEndColumn1[8,7]: X1LessThan[8,7] = 1 -> EndColumn[8] - EndColumn[7]
   <= -1
 LinkEndColumn1[8,9]: X1LessThan[8,9] = 1 -> EndColumn[8] - EndColumn[9]
   <= -1
 LinkEndColumn1[8,10]: X1LessThan[8,10] = 1 -> EndColumn[8] - EndColumn[10]
   <= -1
 LinkEndColumn1[9,0]: X1LessThan[9,0] = 1 -> EndColumn[9] - EndColumn[0]
   <= -1
 LinkEndColumn1[9,1]: X1LessThan[9,1] = 1 -> EndColumn[9] - EndColumn[1]
   <= -1
 LinkEndColumn1[9,2]: X1LessThan[9,2] = 1 -> EndColumn[9] - EndColumn[2]
   <= -1
 LinkEndColumn1[9,3]: X1LessThan[9,3] = 1 -> EndColumn[9] - EndColumn[3]
   <= -1
 LinkEndColumn1[9,4]: X1LessThan[9,4] = 1 -> EndColumn[9] - EndColumn[4]
   <= -1
 LinkEndColumn1[9,5]: X1LessThan[9,5] = 1 -> EndColumn[9] - EndColumn[5]
   <= -1
 LinkEndColumn1[9,6]: X1LessThan[9,6] = 1 -> EndColumn[9] - EndColumn[6]
   <= -1
 LinkEndColumn1[9,7]: X1LessThan[9,7] = 1 -> EndColumn[9] - EndColumn[7]
   <= -1
 LinkEndColumn1[9,8]: X1LessThan[9,8] = 1 -> EndColumn[9] - EndColumn[8]
   <= -1
 LinkEndColumn1[9,10]: X1LessThan[9,10] = 1 -> EndColumn[9] - EndColumn[10]
   <= -1
 LinkEndColumn1[10,0]: X1LessThan[10,0] = 1 -> EndColumn[10] - EndColumn[0]
   <= -1
 LinkEndColumn1[10,1]: X1LessThan[10,1] = 1 -> EndColumn[10] - EndColumn[1]
   <= -1
 LinkEndColumn1[10,2]: X1LessThan[10,2] = 1 -> EndColumn[10] - EndColumn[2]
   <= -1
 LinkEndColumn1[10,3]: X1LessThan[10,3] = 1 -> EndColumn[10] - EndColumn[3]
   <= -1
 LinkEndColumn1[10,4]: X1LessThan[10,4] = 1 -> EndColumn[10] - EndColumn[4]
   <= -1
 LinkEndColumn1[10,5]: X1LessThan[10,5] = 1 -> EndColumn[10] - EndColumn[5]
   <= -1
 LinkEndColumn1[10,6]: X1LessThan[10,6] = 1 -> EndColumn[10] - EndColumn[6]
   <= -1
 LinkEndColumn1[10,7]: X1LessThan[10,7] = 1 -> EndColumn[10] - EndColumn[7]
   <= -1
 LinkEndColumn1[10,8]: X1LessThan[10,8] = 1 -> EndColumn[10] - EndColumn[8]
   <= -1
 LinkEndColumn1[10,9]: X1LessThan[10,9] = 1 -> EndColumn[10] - EndColumn[9]
   <= -1
 LinkEndColumn2[0,1]: X1LessThan[0,1] = 0 -> EndColumn[0] - EndColumn[1]
   >= -0
 LinkEndColumn2[0,2]: X1LessThan[0,2] = 0 -> EndColumn[0] - EndColumn[2]
   >= -0
 LinkEndColumn2[0,3]: X1LessThan[0,3] = 0 -> EndColumn[0] - EndColumn[3]
   >= -0
 LinkEndColumn2[0,4]: X1LessThan[0,4] = 0 -> EndColumn[0] - EndColumn[4]
   >= -0
 LinkEndColumn2[0,5]: X1LessThan[0,5] = 0 -> EndColumn[0] - EndColumn[5]
   >= -0
 LinkEndColumn2[0,6]: X1LessThan[0,6] = 0 -> EndColumn[0] - EndColumn[6]
   >= -0
 LinkEndColumn2[0,7]: X1LessThan[0,7] = 0 -> EndColumn[0] - EndColumn[7]
   >= -0
 LinkEndColumn2[0,8]: X1LessThan[0,8] = 0 -> EndColumn[0] - EndColumn[8]
   >= -0
 LinkEndColumn2[0,9]: X1LessThan[0,9] = 0 -> EndColumn[0] - EndColumn[9]
   >= -0
 LinkEndColumn2[0,10]: X1LessThan[0,10] = 0 -> EndColumn[0] - EndColumn[10]
   >= -0
 LinkEndColumn2[1,0]: X1LessThan[1,0] = 0 -> EndColumn[1] - EndColumn[0]
   >= -0
 LinkEndColumn2[1,2]: X1LessThan[1,2] = 0 -> EndColumn[1] - EndColumn[2]
   >= -0
 LinkEndColumn2[1,3]: X1LessThan[1,3] = 0 -> EndColumn[1] - EndColumn[3]
   >= -0
 LinkEndColumn2[1,4]: X1LessThan[1,4] = 0 -> EndColumn[1] - EndColumn[4]
   >= -0
 LinkEndColumn2[1,5]: X1LessThan[1,5] = 0 -> EndColumn[1] - EndColumn[5]
   >= -0
 LinkEndColumn2[1,6]: X1LessThan[1,6] = 0 -> EndColumn[1] - EndColumn[6]
   >= -0
 LinkEndColumn2[1,7]: X1LessThan[1,7] = 0 -> EndColumn[1] - EndColumn[7]
   >= -0
 LinkEndColumn2[1,8]: X1LessThan[1,8] = 0 -> EndColumn[1] - EndColumn[8]
   >= -0
 LinkEndColumn2[1,9]: X1LessThan[1,9] = 0 -> EndColumn[1] - EndColumn[9]
   >= -0
 LinkEndColumn2[1,10]: X1LessThan[1,10] = 0 -> EndColumn[1] - EndColumn[10]
   >= -0
 LinkEndColumn2[2,0]: X1LessThan[2,0] = 0 -> EndColumn[2] - EndColumn[0]
   >= -0
 LinkEndColumn2[2,1]: X1LessThan[2,1] = 0 -> EndColumn[2] - EndColumn[1]
   >= -0
 LinkEndColumn2[2,3]: X1LessThan[2,3] = 0 -> EndColumn[2] - EndColumn[3]
   >= -0
 LinkEndColumn2[2,4]: X1LessThan[2,4] = 0 -> EndColumn[2] - EndColumn[4]
   >= -0
 LinkEndColumn2[2,5]: X1LessThan[2,5] = 0 -> EndColumn[2] - EndColumn[5]
   >= -0
 LinkEndColumn2[2,6]: X1LessThan[2,6] = 0 -> EndColumn[2] - EndColumn[6]
   >= -0
 LinkEndColumn2[2,7]: X1LessThan[2,7] = 0 -> EndColumn[2] - EndColumn[7]
   >= -0
 LinkEndColumn2[2,8]: X1LessThan[2,8] = 0 -> EndColumn[2] - EndColumn[8]
   >= -0
 LinkEndColumn2[2,9]: X1LessThan[2,9] = 0 -> EndColumn[2] - EndColumn[9]
   >= -0
 LinkEndColumn2[2,10]: X1LessThan[2,10] = 0 -> EndColumn[2] - EndColumn[10]
   >= -0
 LinkEndColumn2[3,0]: X1LessThan[3,0] = 0 -> EndColumn[3] - EndColumn[0]
   >= -0
 LinkEndColumn2[3,1]: X1LessThan[3,1] = 0 -> EndColumn[3] - EndColumn[1]
   >= -0
 LinkEndColumn2[3,2]: X1LessThan[3,2] = 0 -> EndColumn[3] - EndColumn[2]
   >= -0
 LinkEndColumn2[3,4]: X1LessThan[3,4] = 0 -> EndColumn[3] - EndColumn[4]
   >= -0
 LinkEndColumn2[3,5]: X1LessThan[3,5] = 0 -> EndColumn[3] - EndColumn[5]
   >= -0
 LinkEndColumn2[3,6]: X1LessThan[3,6] = 0 -> EndColumn[3] - EndColumn[6]
   >= -0
 LinkEndColumn2[3,7]: X1LessThan[3,7] = 0 -> EndColumn[3] - EndColumn[7]
   >= -0
 LinkEndColumn2[3,8]: X1LessThan[3,8] = 0 -> EndColumn[3] - EndColumn[8]
   >= -0
 LinkEndColumn2[3,9]: X1LessThan[3,9] = 0 -> EndColumn[3] - EndColumn[9]
   >= -0
 LinkEndColumn2[3,10]: X1LessThan[3,10] = 0 -> EndColumn[3] - EndColumn[10]
   >= -0
 LinkEndColumn2[4,0]: X1LessThan[4,0] = 0 -> EndColumn[4] - EndColumn[0]
   >= -0
 LinkEndColumn2[4,1]: X1LessThan[4,1] = 0 -> EndColumn[4] - EndColumn[1]
   >= -0
 LinkEndColumn2[4,2]: X1LessThan[4,2] = 0 -> EndColumn[4] - EndColumn[2]
   >= -0
 LinkEndColumn2[4,3]: X1LessThan[4,3] = 0 -> EndColumn[4] - EndColumn[3]
   >= -0
 LinkEndColumn2[4,5]: X1LessThan[4,5] = 0 -> EndColumn[4] - EndColumn[5]
   >= -0
 LinkEndColumn2[4,6]: X1LessThan[4,6] = 0 -> EndColumn[4] - EndColumn[6]
   >= -0
 LinkEndColumn2[4,7]: X1LessThan[4,7] = 0 -> EndColumn[4] - EndColumn[7]
   >= -0
 LinkEndColumn2[4,8]: X1LessThan[4,8] = 0 -> EndColumn[4] - EndColumn[8]
   >= -0
 LinkEndColumn2[4,9]: X1LessThan[4,9] = 0 -> EndColumn[4] - EndColumn[9]
   >= -0
 LinkEndColumn2[4,10]: X1LessThan[4,10] = 0 -> EndColumn[4] - EndColumn[10]
   >= -0
 LinkEndColumn2[5,0]: X1LessThan[5,0] = 0 -> EndColumn[5] - EndColumn[0]
   >= -0
 LinkEndColumn2[5,1]: X1LessThan[5,1] = 0 -> EndColumn[5] - EndColumn[1]
   >= -0
 LinkEndColumn2[5,2]: X1LessThan[5,2] = 0 -> EndColumn[5] - EndColumn[2]
   >= -0
 LinkEndColumn2[5,3]: X1LessThan[5,3] = 0 -> EndColumn[5] - EndColumn[3]
   >= -0
 LinkEndColumn2[5,4]: X1LessThan[5,4] = 0 -> EndColumn[5] - EndColumn[4]
   >= -0
 LinkEndColumn2[5,6]: X1LessThan[5,6] = 0 -> EndColumn[5] - EndColumn[6]
   >= -0
 LinkEndColumn2[5,7]: X1LessThan[5,7] = 0 -> EndColumn[5] - EndColumn[7]
   >= -0
 LinkEndColumn2[5,8]: X1LessThan[5,8] = 0 -> EndColumn[5] - EndColumn[8]
   >= -0
 LinkEndColumn2[5,9]: X1LessThan[5,9] = 0 -> EndColumn[5] - EndColumn[9]
   >= -0
 LinkEndColumn2[5,10]: X1LessThan[5,10] = 0 -> EndColumn[5] - EndColumn[10]
   >= -0
 LinkEndColumn2[6,0]: X1LessThan[6,0] = 0 -> EndColumn[6] - EndColumn[0]
   >= -0
 LinkEndColumn2[6,1]: X1LessThan[6,1] = 0 -> EndColumn[6] - EndColumn[1]
   >= -0
 LinkEndColumn2[6,2]: X1LessThan[6,2] = 0 -> EndColumn[6] - EndColumn[2]
   >= -0
 LinkEndColumn2[6,3]: X1LessThan[6,3] = 0 -> EndColumn[6] - EndColumn[3]
   >= -0
 LinkEndColumn2[6,4]: X1LessThan[6,4] = 0 -> EndColumn[6] - EndColumn[4]
   >= -0
 LinkEndColumn2[6,5]: X1LessThan[6,5] = 0 -> EndColumn[6] - EndColumn[5]
   >= -0
 LinkEndColumn2[6,7]: X1LessThan[6,7] = 0 -> EndColumn[6] - EndColumn[7]
   >= -0
 LinkEndColumn2[6,8]: X1LessThan[6,8] = 0 -> EndColumn[6] - EndColumn[8]
   >= -0
 LinkEndColumn2[6,9]: X1LessThan[6,9] = 0 -> EndColumn[6] - EndColumn[9]
   >= -0
 LinkEndColumn2[6,10]: X1LessThan[6,10] = 0 -> EndColumn[6] - EndColumn[10]
   >= -0
 LinkEndColumn2[7,0]: X1LessThan[7,0] = 0 -> EndColumn[7] - EndColumn[0]
   >= -0
 LinkEndColumn2[7,1]: X1LessThan[7,1] = 0 -> EndColumn[7] - EndColumn[1]
   >= -0
 LinkEndColumn2[7,2]: X1LessThan[7,2] = 0 -> EndColumn[7] - EndColumn[2]
   >= -0
 LinkEndColumn2[7,3]: X1LessThan[7,3] = 0 -> EndColumn[7] - EndColumn[3]
   >= -0
 LinkEndColumn2[7,4]: X1LessThan[7,4] = 0 -> EndColumn[7] - EndColumn[4]
   >= -0
 LinkEndColumn2[7,5]: X1LessThan[7,5] = 0 -> EndColumn[7] - EndColumn[5]
   >= -0
 LinkEndColumn2[7,6]: X1LessThan[7,6] = 0 -> EndColumn[7] - EndColumn[6]
   >= -0
 LinkEndColumn2[7,8]: X1LessThan[7,8] = 0 -> EndColumn[7] - EndColumn[8]
   >= -0
 LinkEndColumn2[7,9]: X1LessThan[7,9] = 0 -> EndColumn[7] - EndColumn[9]
   >= -0
 LinkEndColumn2[7,10]: X1LessThan[7,10] = 0 -> EndColumn[7] - EndColumn[10]
   >= -0
 LinkEndColumn2[8,0]: X1LessThan[8,0] = 0 -> EndColumn[8] - EndColumn[0]
   >= -0
 LinkEndColumn2[8,1]: X1LessThan[8,1] = 0 -> EndColumn[8] - EndColumn[1]
   >= -0
 LinkEndColumn2[8,2]: X1LessThan[8,2] = 0 -> EndColumn[8] - EndColumn[2]
   >= -0
 LinkEndColumn2[8,3]: X1LessThan[8,3] = 0 -> EndColumn[8] - EndColumn[3]
   >= -0
 LinkEndColumn2[8,4]: X1LessThan[8,4] = 0 -> EndColumn[8] - EndColumn[4]
   >= -0
 LinkEndColumn2[8,5]: X1LessThan[8,5] = 0 -> EndColumn[8] - EndColumn[5]
   >= -0
 LinkEndColumn2[8,6]: X1LessThan[8,6] = 0 -> EndColumn[8] - EndColumn[6]
   >= -0
 LinkEndColumn2[8,7]: X1LessThan[8,7] = 0 -> EndColumn[8] - EndColumn[7]
   >= -0
 LinkEndColumn2[8,9]: X1LessThan[8,9] = 0 -> EndColumn[8] - EndColumn[9]
   >= -0
 LinkEndColumn2[8,10]: X1LessThan[8,10] = 0 -> EndColumn[8] - EndColumn[10]
   >= -0
 LinkEndColumn2[9,0]: X1LessThan[9,0] = 0 -> EndColumn[9] - EndColumn[0]
   >= -0
 LinkEndColumn2[9,1]: X1LessThan[9,1] = 0 -> EndColumn[9] - EndColumn[1]
   >= -0
 LinkEndColumn2[9,2]: X1LessThan[9,2] = 0 -> EndColumn[9] - EndColumn[2]
   >= -0
 LinkEndColumn2[9,3]: X1LessThan[9,3] = 0 -> EndColumn[9] - EndColumn[3]
   >= -0
 LinkEndColumn2[9,4]: X1LessThan[9,4] = 0 -> EndColumn[9] - EndColumn[4]
   >= -0
 LinkEndColumn2[9,5]: X1LessThan[9,5] = 0 -> EndColumn[9] - EndColumn[5]
   >= -0
 LinkEndColumn2[9,6]: X1LessThan[9,6] = 0 -> EndColumn[9] - EndColumn[6]
   >= -0
 LinkEndColumn2[9,7]: X1LessThan[9,7] = 0 -> EndColumn[9] - EndColumn[7]
   >= -0
 LinkEndColumn2[9,8]: X1LessThan[9,8] = 0 -> EndColumn[9] - EndColumn[8]
   >= -0
 LinkEndColumn2[9,10]: X1LessThan[9,10] = 0 -> EndColumn[9] - EndColumn[10]
   >= -0
 LinkEndColumn2[10,0]: X1LessThan[10,0] = 0 -> EndColumn[10] - EndColumn[0]
   >= -0
 LinkEndColumn2[10,1]: X1LessThan[10,1] = 0 -> EndColumn[10] - EndColumn[1]
   >= -0
 LinkEndColumn2[10,2]: X1LessThan[10,2] = 0 -> EndColumn[10] - EndColumn[2]
   >= -0
 LinkEndColumn2[10,3]: X1LessThan[10,3] = 0 -> EndColumn[10] - EndColumn[3]
   >= -0
 LinkEndColumn2[10,4]: X1LessThan[10,4] = 0 -> EndColumn[10] - EndColumn[4]
   >= -0
 LinkEndColumn2[10,5]: X1LessThan[10,5] = 0 -> EndColumn[10] - EndColumn[5]
   >= -0
 LinkEndColumn2[10,6]: X1LessThan[10,6] = 0 -> EndColumn[10] - EndColumn[6]
   >= -0
 LinkEndColumn2[10,7]: X1LessThan[10,7] = 0 -> EndColumn[10] - EndColumn[7]
   >= -0
 LinkEndColumn2[10,8]: X1LessThan[10,8] = 0 -> EndColumn[10] - EndColumn[8]
   >= -0
 LinkEndColumn2[10,9]: X1LessThan[10,9] = 0 -> EndColumn[10] - EndColumn[9]
   >= -0
 MaxColumnWidth1[0,1]: X0LessThan[0,1] = 1 -> X0Diff[1,0] - C2252 >= -0
 MaxColumnWidth1[0,2]: X0LessThan[0,2] = 1 -> X0Diff[2,0] - C2252 >= -0
 MaxColumnWidth1[0,3]: X0LessThan[0,3] = 1 -> X0Diff[3,0] - C2252 >= -0
 MaxColumnWidth1[0,4]: X0LessThan[0,4] = 1 -> X0Diff[4,0] - C2252 >= -0
 MaxColumnWidth1[0,5]: X0LessThan[0,5] = 1 -> X0Diff[5,0] - C2252 >= -0
 MaxColumnWidth1[0,6]: X0LessThan[0,6] = 1 -> X0Diff[6,0] - C2252 >= -0
 MaxColumnWidth1[0,7]: X0LessThan[0,7] = 1 -> X0Diff[7,0] - C2252 >= -0
 MaxColumnWidth1[0,8]: X0LessThan[0,8] = 1 -> X0Diff[8,0] - C2252 >= -0
 MaxColumnWidth1[0,9]: X0LessThan[0,9] = 1 -> X0Diff[9,0] - C2252 >= -0
 MaxColumnWidth1[0,10]: X0LessThan[0,10] = 1 -> X0Diff[10,0] - C2252 >= -0
 MaxColumnWidth1[1,0]: X0LessThan[1,0] = 1 -> X0Diff[0,1] - C2252 >= -0
 MaxColumnWidth1[1,2]: X0LessThan[1,2] = 1 -> X0Diff[2,1] - C2252 >= -0
 MaxColumnWidth1[1,3]: X0LessThan[1,3] = 1 -> X0Diff[3,1] - C2252 >= -0
 MaxColumnWidth1[1,4]: X0LessThan[1,4] = 1 -> X0Diff[4,1] - C2252 >= -0
 MaxColumnWidth1[1,5]: X0LessThan[1,5] = 1 -> X0Diff[5,1] - C2252 >= -0
 MaxColumnWidth1[1,6]: X0LessThan[1,6] = 1 -> X0Diff[6,1] - C2252 >= -0
 MaxColumnWidth1[1,7]: X0LessThan[1,7] = 1 -> X0Diff[7,1] - C2252 >= -0
 MaxColumnWidth1[1,8]: X0LessThan[1,8] = 1 -> X0Diff[8,1] - C2252 >= -0
 MaxColumnWidth1[1,9]: X0LessThan[1,9] = 1 -> X0Diff[9,1] - C2252 >= -0
 MaxColumnWidth1[1,10]: X0LessThan[1,10] = 1 -> X0Diff[10,1] - C2252 >= -0
 MaxColumnWidth1[2,0]: X0LessThan[2,0] = 1 -> X0Diff[0,2] - C2252 >= -0
 MaxColumnWidth1[2,1]: X0LessThan[2,1] = 1 -> X0Diff[1,2] - C2252 >= -0
 MaxColumnWidth1[2,3]: X0LessThan[2,3] = 1 -> X0Diff[3,2] - C2252 >= -0
 MaxColumnWidth1[2,4]: X0LessThan[2,4] = 1 -> X0Diff[4,2] - C2252 >= -0
 MaxColumnWidth1[2,5]: X0LessThan[2,5] = 1 -> X0Diff[5,2] - C2252 >= -0
 MaxColumnWidth1[2,6]: X0LessThan[2,6] = 1 -> X0Diff[6,2] - C2252 >= -0
 MaxColumnWidth1[2,7]: X0LessThan[2,7] = 1 -> X0Diff[7,2] - C2252 >= -0
 MaxColumnWidth1[2,8]: X0LessThan[2,8] = 1 -> X0Diff[8,2] - C2252 >= -0
 MaxColumnWidth1[2,9]: X0LessThan[2,9] = 1 -> X0Diff[9,2] - C2252 >= -0
 MaxColumnWidth1[2,10]: X0LessThan[2,10] = 1 -> X0Diff[10,2] - C2252 >= -0
 MaxColumnWidth1[3,0]: X0LessThan[3,0] = 1 -> X0Diff[0,3] - C2252 >= -0
 MaxColumnWidth1[3,1]: X0LessThan[3,1] = 1 -> X0Diff[1,3] - C2252 >= -0
 MaxColumnWidth1[3,2]: X0LessThan[3,2] = 1 -> X0Diff[2,3] - C2252 >= -0
 MaxColumnWidth1[3,4]: X0LessThan[3,4] = 1 -> X0Diff[4,3] - C2252 >= -0
 MaxColumnWidth1[3,5]: X0LessThan[3,5] = 1 -> X0Diff[5,3] - C2252 >= -0
 MaxColumnWidth1[3,6]: X0LessThan[3,6] = 1 -> X0Diff[6,3] - C2252 >= -0
 MaxColumnWidth1[3,7]: X0LessThan[3,7] = 1 -> X0Diff[7,3] - C2252 >= -0
 MaxColumnWidth1[3,8]: X0LessThan[3,8] = 1 -> X0Diff[8,3] - C2252 >= -0
 MaxColumnWidth1[3,9]: X0LessThan[3,9] = 1 -> X0Diff[9,3] - C2252 >= -0
 MaxColumnWidth1[3,10]: X0LessThan[3,10] = 1 -> X0Diff[10,3] - C2252 >= -0
 MaxColumnWidth1[4,0]: X0LessThan[4,0] = 1 -> X0Diff[0,4] - C2252 >= -0
 MaxColumnWidth1[4,1]: X0LessThan[4,1] = 1 -> X0Diff[1,4] - C2252 >= -0
 MaxColumnWidth1[4,2]: X0LessThan[4,2] = 1 -> X0Diff[2,4] - C2252 >= -0
 MaxColumnWidth1[4,3]: X0LessThan[4,3] = 1 -> X0Diff[3,4] - C2252 >= -0
 MaxColumnWidth1[4,5]: X0LessThan[4,5] = 1 -> X0Diff[5,4] - C2252 >= -0
 MaxColumnWidth1[4,6]: X0LessThan[4,6] = 1 -> X0Diff[6,4] - C2252 >= -0
 MaxColumnWidth1[4,7]: X0LessThan[4,7] = 1 -> X0Diff[7,4] - C2252 >= -0
 MaxColumnWidth1[4,8]: X0LessThan[4,8] = 1 -> X0Diff[8,4] - C2252 >= -0
 MaxColumnWidth1[4,9]: X0LessThan[4,9] = 1 -> X0Diff[9,4] - C2252 >= -0
 MaxColumnWidth1[4,10]: X0LessThan[4,10] = 1 -> X0Diff[10,4] - C2252 >= -0
 MaxColumnWidth1[5,0]: X0LessThan[5,0] = 1 -> X0Diff[0,5] - C2252 >= -0
 MaxColumnWidth1[5,1]: X0LessThan[5,1] = 1 -> X0Diff[1,5] - C2252 >= -0
 MaxColumnWidth1[5,2]: X0LessThan[5,2] = 1 -> X0Diff[2,5] - C2252 >= -0
 MaxColumnWidth1[5,3]: X0LessThan[5,3] = 1 -> X0Diff[3,5] - C2252 >= -0
 MaxColumnWidth1[5,4]: X0LessThan[5,4] = 1 -> X0Diff[4,5] - C2252 >= -0
 MaxColumnWidth1[5,6]: X0LessThan[5,6] = 1 -> X0Diff[6,5] - C2252 >= -0
 MaxColumnWidth1[5,7]: X0LessThan[5,7] = 1 -> X0Diff[7,5] - C2252 >= -0
 MaxColumnWidth1[5,8]: X0LessThan[5,8] = 1 -> X0Diff[8,5] - C2252 >= -0
 MaxColumnWidth1[5,9]: X0LessThan[5,9] = 1 -> X0Diff[9,5] - C2252 >= -0
 MaxColumnWidth1[5,10]: X0LessThan[5,10] = 1 -> X0Diff[10,5] - C2252 >= -0
 MaxColumnWidth1[6,0]: X0LessThan[6,0] = 1 -> X0Diff[0,6] - C2252 >= -0
 MaxColumnWidth1[6,1]: X0LessThan[6,1] = 1 -> X0Diff[1,6] - C2252 >= -0
 MaxColumnWidth1[6,2]: X0LessThan[6,2] = 1 -> X0Diff[2,6] - C2252 >= -0
 MaxColumnWidth1[6,3]: X0LessThan[6,3] = 1 -> X0Diff[3,6] - C2252 >= -0
 MaxColumnWidth1[6,4]: X0LessThan[6,4] = 1 -> X0Diff[4,6] - C2252 >= -0
 MaxColumnWidth1[6,5]: X0LessThan[6,5] = 1 -> X0Diff[5,6] - C2252 >= -0
 MaxColumnWidth1[6,7]: X0LessThan[6,7] = 1 -> X0Diff[7,6] - C2252 >= -0
 MaxColumnWidth1[6,8]: X0LessThan[6,8] = 1 -> X0Diff[8,6] - C2252 >= -0
 MaxColumnWidth1[6,9]: X0LessThan[6,9] = 1 -> X0Diff[9,6] - C2252 >= -0
 MaxColumnWidth1[6,10]: X0LessThan[6,10] = 1 -> X0Diff[10,6] - C2252 >= -0
 MaxColumnWidth1[7,0]: X0LessThan[7,0] = 1 -> X0Diff[0,7] - C2252 >= -0
 MaxColumnWidth1[7,1]: X0LessThan[7,1] = 1 -> X0Diff[1,7] - C2252 >= -0
 MaxColumnWidth1[7,2]: X0LessThan[7,2] = 1 -> X0Diff[2,7] - C2252 >= -0
 MaxColumnWidth1[7,3]: X0LessThan[7,3] = 1 -> X0Diff[3,7] - C2252 >= -0
 MaxColumnWidth1[7,4]: X0LessThan[7,4] = 1 -> X0Diff[4,7] - C2252 >= -0
 MaxColumnWidth1[7,5]: X0LessThan[7,5] = 1 -> X0Diff[5,7] - C2252 >= -0
 MaxColumnWidth1[7,6]: X0LessThan[7,6] = 1 -> X0Diff[6,7] - C2252 >= -0
 MaxColumnWidth1[7,8]: X0LessThan[7,8] = 1 -> X0Diff[8,7] - C2252 >= -0
 MaxColumnWidth1[7,9]: X0LessThan[7,9] = 1 -> X0Diff[9,7] - C2252 >= -0
 MaxColumnWidth1[7,10]: X0LessThan[7,10] = 1 -> X0Diff[10,7] - C2252 >= -0
 MaxColumnWidth1[8,0]: X0LessThan[8,0] = 1 -> X0Diff[0,8] - C2252 >= -0
 MaxColumnWidth1[8,1]: X0LessThan[8,1] = 1 -> X0Diff[1,8] - C2252 >= -0
 MaxColumnWidth1[8,2]: X0LessThan[8,2] = 1 -> X0Diff[2,8] - C2252 >= -0
 MaxColumnWidth1[8,3]: X0LessThan[8,3] = 1 -> X0Diff[3,8] - C2252 >= -0
 MaxColumnWidth1[8,4]: X0LessThan[8,4] = 1 -> X0Diff[4,8] - C2252 >= -0
 MaxColumnWidth1[8,5]: X0LessThan[8,5] = 1 -> X0Diff[5,8] - C2252 >= -0
 MaxColumnWidth1[8,6]: X0LessThan[8,6] = 1 -> X0Diff[6,8] - C2252 >= -0
 MaxColumnWidth1[8,7]: X0LessThan[8,7] = 1 -> X0Diff[7,8] - C2252 >= -0
 MaxColumnWidth1[8,9]: X0LessThan[8,9] = 1 -> X0Diff[9,8] - C2252 >= -0
 MaxColumnWidth1[8,10]: X0LessThan[8,10] = 1 -> X0Diff[10,8] - C2252 >= -0
 MaxColumnWidth1[9,0]: X0LessThan[9,0] = 1 -> X0Diff[0,9] - C2252 >= -0
 MaxColumnWidth1[9,1]: X0LessThan[9,1] = 1 -> X0Diff[1,9] - C2252 >= -0
 MaxColumnWidth1[9,2]: X0LessThan[9,2] = 1 -> X0Diff[2,9] - C2252 >= -0
 MaxColumnWidth1[9,3]: X0LessThan[9,3] = 1 -> X0Diff[3,9] - C2252 >= -0
 MaxColumnWidth1[9,4]: X0LessThan[9,4] = 1 -> X0Diff[4,9] - C2252 >= -0
 MaxColumnWidth1[9,5]: X0LessThan[9,5] = 1 -> X0Diff[5,9] - C2252 >= -0
 MaxColumnWidth1[9,6]: X0LessThan[9,6] = 1 -> X0Diff[6,9] - C2252 >= -0
 MaxColumnWidth1[9,7]: X0LessThan[9,7] = 1 -> X0Diff[7,9] - C2252 >= -0
 MaxColumnWidth1[9,8]: X0LessThan[9,8] = 1 -> X0Diff[8,9] - C2252 >= -0
 MaxColumnWidth1[9,10]: X0LessThan[9,10] = 1 -> X0Diff[10,9] - C2252 >= -0
 MaxColumnWidth1[10,0]: X0LessThan[10,0] = 1 -> X0Diff[0,10] - C2252 >= -0
 MaxColumnWidth1[10,1]: X0LessThan[10,1] = 1 -> X0Diff[1,10] - C2252 >= -0
 MaxColumnWidth1[10,2]: X0LessThan[10,2] = 1 -> X0Diff[2,10] - C2252 >= -0
 MaxColumnWidth1[10,3]: X0LessThan[10,3] = 1 -> X0Diff[3,10] - C2252 >= -0
 MaxColumnWidth1[10,4]: X0LessThan[10,4] = 1 -> X0Diff[4,10] - C2252 >= -0
 MaxColumnWidth1[10,5]: X0LessThan[10,5] = 1 -> X0Diff[5,10] - C2252 >= -0
 MaxColumnWidth1[10,6]: X0LessThan[10,6] = 1 -> X0Diff[6,10] - C2252 >= -0
 MaxColumnWidth1[10,7]: X0LessThan[10,7] = 1 -> X0Diff[7,10] - C2252 >= -0
 MaxColumnWidth1[10,8]: X0LessThan[10,8] = 1 -> X0Diff[8,10] - C2252 >= -0
 MaxColumnWidth1[10,9]: X0LessThan[10,9] = 1 -> X0Diff[9,10] - C2252 >= -0
 MaxColumnWidth2[0,1]: X1LessThan[0,1] = 1 -> X1Diff[1,0] - C2252 >= -0
 MaxColumnWidth2[0,2]: X1LessThan[0,2] = 1 -> X1Diff[2,0] - C2252 >= -0
 MaxColumnWidth2[0,3]: X1LessThan[0,3] = 1 -> X1Diff[3,0] - C2252 >= -0
 MaxColumnWidth2[0,4]: X1LessThan[0,4] = 1 -> X1Diff[4,0] - C2252 >= -0
 MaxColumnWidth2[0,5]: X1LessThan[0,5] = 1 -> X1Diff[5,0] - C2252 >= -0
 MaxColumnWidth2[0,6]: X1LessThan[0,6] = 1 -> X1Diff[6,0] - C2252 >= -0
 MaxColumnWidth2[0,7]: X1LessThan[0,7] = 1 -> X1Diff[7,0] - C2252 >= -0
 MaxColumnWidth2[0,8]: X1LessThan[0,8] = 1 -> X1Diff[8,0] - C2252 >= -0
 MaxColumnWidth2[0,9]: X1LessThan[0,9] = 1 -> X1Diff[9,0] - C2252 >= -0
 MaxColumnWidth2[0,10]: X1LessThan[0,10] = 1 -> X1Diff[10,0] - C2252 >= -0
 MaxColumnWidth2[1,0]: X1LessThan[1,0] = 1 -> X1Diff[0,1] - C2252 >= -0
 MaxColumnWidth2[1,2]: X1LessThan[1,2] = 1 -> X1Diff[2,1] - C2252 >= -0
 MaxColumnWidth2[1,3]: X1LessThan[1,3] = 1 -> X1Diff[3,1] - C2252 >= -0
 MaxColumnWidth2[1,4]: X1LessThan[1,4] = 1 -> X1Diff[4,1] - C2252 >= -0
 MaxColumnWidth2[1,5]: X1LessThan[1,5] = 1 -> X1Diff[5,1] - C2252 >= -0
 MaxColumnWidth2[1,6]: X1LessThan[1,6] = 1 -> X1Diff[6,1] - C2252 >= -0
 MaxColumnWidth2[1,7]: X1LessThan[1,7] = 1 -> X1Diff[7,1] - C2252 >= -0
 MaxColumnWidth2[1,8]: X1LessThan[1,8] = 1 -> X1Diff[8,1] - C2252 >= -0
 MaxColumnWidth2[1,9]: X1LessThan[1,9] = 1 -> X1Diff[9,1] - C2252 >= -0
 MaxColumnWidth2[1,10]: X1LessThan[1,10] = 1 -> X1Diff[10,1] - C2252 >= -0
 MaxColumnWidth2[2,0]: X1LessThan[2,0] = 1 -> X1Diff[0,2] - C2252 >= -0
 MaxColumnWidth2[2,1]: X1LessThan[2,1] = 1 -> X1Diff[1,2] - C2252 >= -0
 MaxColumnWidth2[2,3]: X1LessThan[2,3] = 1 -> X1Diff[3,2] - C2252 >= -0
 MaxColumnWidth2[2,4]: X1LessThan[2,4] = 1 -> X1Diff[4,2] - C2252 >= -0
 MaxColumnWidth2[2,5]: X1LessThan[2,5] = 1 -> X1Diff[5,2] - C2252 >= -0
 MaxColumnWidth2[2,6]: X1LessThan[2,6] = 1 -> X1Diff[6,2] - C2252 >= -0
 MaxColumnWidth2[2,7]: X1LessThan[2,7] = 1 -> X1Diff[7,2] - C2252 >= -0
 MaxColumnWidth2[2,8]: X1LessThan[2,8] = 1 -> X1Diff[8,2] - C2252 >= -0
 MaxColumnWidth2[2,9]: X1LessThan[2,9] = 1 -> X1Diff[9,2] - C2252 >= -0
 MaxColumnWidth2[2,10]: X1LessThan[2,10] = 1 -> X1Diff[10,2] - C2252 >= -0
 MaxColumnWidth2[3,0]: X1LessThan[3,0] = 1 -> X1Diff[0,3] - C2252 >= -0
 MaxColumnWidth2[3,1]: X1LessThan[3,1] = 1 -> X1Diff[1,3] - C2252 >= -0
 MaxColumnWidth2[3,2]: X1LessThan[3,2] = 1 -> X1Diff[2,3] - C2252 >= -0
 MaxColumnWidth2[3,4]: X1LessThan[3,4] = 1 -> X1Diff[4,3] - C2252 >= -0
 MaxColumnWidth2[3,5]: X1LessThan[3,5] = 1 -> X1Diff[5,3] - C2252 >= -0
 MaxColumnWidth2[3,6]: X1LessThan[3,6] = 1 -> X1Diff[6,3] - C2252 >= -0
 MaxColumnWidth2[3,7]: X1LessThan[3,7] = 1 -> X1Diff[7,3] - C2252 >= -0
 MaxColumnWidth2[3,8]: X1LessThan[3,8] = 1 -> X1Diff[8,3] - C2252 >= -0
 MaxColumnWidth2[3,9]: X1LessThan[3,9] = 1 -> X1Diff[9,3] - C2252 >= -0
 MaxColumnWidth2[3,10]: X1LessThan[3,10] = 1 -> X1Diff[10,3] - C2252 >= -0
 MaxColumnWidth2[4,0]: X1LessThan[4,0] = 1 -> X1Diff[0,4] - C2252 >= -0
 MaxColumnWidth2[4,1]: X1LessThan[4,1] = 1 -> X1Diff[1,4] - C2252 >= -0
 MaxColumnWidth2[4,2]: X1LessThan[4,2] = 1 -> X1Diff[2,4] - C2252 >= -0
 MaxColumnWidth2[4,3]: X1LessThan[4,3] = 1 -> X1Diff[3,4] - C2252 >= -0
 MaxColumnWidth2[4,5]: X1LessThan[4,5] = 1 -> X1Diff[5,4] - C2252 >= -0
 MaxColumnWidth2[4,6]: X1LessThan[4,6] = 1 -> X1Diff[6,4] - C2252 >= -0
 MaxColumnWidth2[4,7]: X1LessThan[4,7] = 1 -> X1Diff[7,4] - C2252 >= -0
 MaxColumnWidth2[4,8]: X1LessThan[4,8] = 1 -> X1Diff[8,4] - C2252 >= -0
 MaxColumnWidth2[4,9]: X1LessThan[4,9] = 1 -> X1Diff[9,4] - C2252 >= -0
 MaxColumnWidth2[4,10]: X1LessThan[4,10] = 1 -> X1Diff[10,4] - C2252 >= -0
 MaxColumnWidth2[5,0]: X1LessThan[5,0] = 1 -> X1Diff[0,5] - C2252 >= -0
 MaxColumnWidth2[5,1]: X1LessThan[5,1] = 1 -> X1Diff[1,5] - C2252 >= -0
 MaxColumnWidth2[5,2]: X1LessThan[5,2] = 1 -> X1Diff[2,5] - C2252 >= -0
 MaxColumnWidth2[5,3]: X1LessThan[5,3] = 1 -> X1Diff[3,5] - C2252 >= -0
 MaxColumnWidth2[5,4]: X1LessThan[5,4] = 1 -> X1Diff[4,5] - C2252 >= -0
 MaxColumnWidth2[5,6]: X1LessThan[5,6] = 1 -> X1Diff[6,5] - C2252 >= -0
 MaxColumnWidth2[5,7]: X1LessThan[5,7] = 1 -> X1Diff[7,5] - C2252 >= -0
 MaxColumnWidth2[5,8]: X1LessThan[5,8] = 1 -> X1Diff[8,5] - C2252 >= -0
 MaxColumnWidth2[5,9]: X1LessThan[5,9] = 1 -> X1Diff[9,5] - C2252 >= -0
 MaxColumnWidth2[5,10]: X1LessThan[5,10] = 1 -> X1Diff[10,5] - C2252 >= -0
 MaxColumnWidth2[6,0]: X1LessThan[6,0] = 1 -> X1Diff[0,6] - C2252 >= -0
 MaxColumnWidth2[6,1]: X1LessThan[6,1] = 1 -> X1Diff[1,6] - C2252 >= -0
 MaxColumnWidth2[6,2]: X1LessThan[6,2] = 1 -> X1Diff[2,6] - C2252 >= -0
 MaxColumnWidth2[6,3]: X1LessThan[6,3] = 1 -> X1Diff[3,6] - C2252 >= -0
 MaxColumnWidth2[6,4]: X1LessThan[6,4] = 1 -> X1Diff[4,6] - C2252 >= -0
 MaxColumnWidth2[6,5]: X1LessThan[6,5] = 1 -> X1Diff[5,6] - C2252 >= -0
 MaxColumnWidth2[6,7]: X1LessThan[6,7] = 1 -> X1Diff[7,6] - C2252 >= -0
 MaxColumnWidth2[6,8]: X1LessThan[6,8] = 1 -> X1Diff[8,6] - C2252 >= -0
 MaxColumnWidth2[6,9]: X1LessThan[6,9] = 1 -> X1Diff[9,6] - C2252 >= -0
 MaxColumnWidth2[6,10]: X1LessThan[6,10] = 1 -> X1Diff[10,6] - C2252 >= -0
 MaxColumnWidth2[7,0]: X1LessThan[7,0] = 1 -> X1Diff[0,7] - C2252 >= -0
 MaxColumnWidth2[7,1]: X1LessThan[7,1] = 1 -> X1Diff[1,7] - C2252 >= -0
 MaxColumnWidth2[7,2]: X1LessThan[7,2] = 1 -> X1Diff[2,7] - C2252 >= -0
 MaxColumnWidth2[7,3]: X1LessThan[7,3] = 1 -> X1Diff[3,7] - C2252 >= -0
 MaxColumnWidth2[7,4]: X1LessThan[7,4] = 1 -> X1Diff[4,7] - C2252 >= -0
 MaxColumnWidth2[7,5]: X1LessThan[7,5] = 1 -> X1Diff[5,7] - C2252 >= -0
 MaxColumnWidth2[7,6]: X1LessThan[7,6] = 1 -> X1Diff[6,7] - C2252 >= -0
 MaxColumnWidth2[7,8]: X1LessThan[7,8] = 1 -> X1Diff[8,7] - C2252 >= -0
 MaxColumnWidth2[7,9]: X1LessThan[7,9] = 1 -> X1Diff[9,7] - C2252 >= -0
 MaxColumnWidth2[7,10]: X1LessThan[7,10] = 1 -> X1Diff[10,7] - C2252 >= -0
 MaxColumnWidth2[8,0]: X1LessThan[8,0] = 1 -> X1Diff[0,8] - C2252 >= -0
 MaxColumnWidth2[8,1]: X1LessThan[8,1] = 1 -> X1Diff[1,8] - C2252 >= -0
 MaxColumnWidth2[8,2]: X1LessThan[8,2] = 1 -> X1Diff[2,8] - C2252 >= -0
 MaxColumnWidth2[8,3]: X1LessThan[8,3] = 1 -> X1Diff[3,8] - C2252 >= -0
 MaxColumnWidth2[8,4]: X1LessThan[8,4] = 1 -> X1Diff[4,8] - C2252 >= -0
 MaxColumnWidth2[8,5]: X1LessThan[8,5] = 1 -> X1Diff[5,8] - C2252 >= -0
 MaxColumnWidth2[8,6]: X1LessThan[8,6] = 1 -> X1Diff[6,8] - C2252 >= -0
 MaxColumnWidth2[8,7]: X1LessThan[8,7] = 1 -> X1Diff[7,8] - C2252 >= -0
 MaxColumnWidth2[8,9]: X1LessThan[8,9] = 1 -> X1Diff[9,8] - C2252 >= -0
 MaxColumnWidth2[8,10]: X1LessThan[8,10] = 1 -> X1Diff[10,8] - C2252 >= -0
 MaxColumnWidth2[9,0]: X1LessThan[9,0] = 1 -> X1Diff[0,9] - C2252 >= -0
 MaxColumnWidth2[9,1]: X1LessThan[9,1] = 1 -> X1Diff[1,9] - C2252 >= -0
 MaxColumnWidth2[9,2]: X1LessThan[9,2] = 1 -> X1Diff[2,9] - C2252 >= -0
 MaxColumnWidth2[9,3]: X1LessThan[9,3] = 1 -> X1Diff[3,9] - C2252 >= -0
 MaxColumnWidth2[9,4]: X1LessThan[9,4] = 1 -> X1Diff[4,9] - C2252 >= -0
 MaxColumnWidth2[9,5]: X1LessThan[9,5] = 1 -> X1Diff[5,9] - C2252 >= -0
 MaxColumnWidth2[9,6]: X1LessThan[9,6] = 1 -> X1Diff[6,9] - C2252 >= -0
 MaxColumnWidth2[9,7]: X1LessThan[9,7] = 1 -> X1Diff[7,9] - C2252 >= -0
 MaxColumnWidth2[9,8]: X1LessThan[9,8] = 1 -> X1Diff[8,9] - C2252 >= -0
 MaxColumnWidth2[9,10]: X1LessThan[9,10] = 1 -> X1Diff[10,9] - C2252 >= -0
 MaxColumnWidth2[10,0]: X1LessThan[10,0] = 1 -> X1Diff[0,10] - C2252 >= -0
 MaxColumnWidth2[10,1]: X1LessThan[10,1] = 1 -> X1Diff[1,10] - C2252 >= -0
 MaxColumnWidth2[10,2]: X1LessThan[10,2] = 1 -> X1Diff[2,10] - C2252 >= -0
 MaxColumnWidth2[10,3]: X1LessThan[10,3] = 1 -> X1Diff[3,10] - C2252 >= -0
 MaxColumnWidth2[10,4]: X1LessThan[10,4] = 1 -> X1Diff[4,10] - C2252 >= -0
 MaxColumnWidth2[10,5]: X1LessThan[10,5] = 1 -> X1Diff[5,10] - C2252 >= -0
 MaxColumnWidth2[10,6]: X1LessThan[10,6] = 1 -> X1Diff[6,10] - C2252 >= -0
 MaxColumnWidth2[10,7]: X1LessThan[10,7] = 1 -> X1Diff[7,10] - C2252 >= -0
 MaxColumnWidth2[10,8]: X1LessThan[10,8] = 1 -> X1Diff[8,10] - C2252 >= -0
 MaxColumnWidth2[10,9]: X1LessThan[10,9] = 1 -> X1Diff[9,10] - C2252 >= -0
 LinkOtherElementStartsInNextColumn2[0,1]: OtherElementStartsInNextColumn[0,1] = 1 ->
   StartColumn[1] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,2]: OtherElementStartsInNextColumn[0,2] = 1 ->
   StartColumn[2] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,3]: OtherElementStartsInNextColumn[0,3] = 1 ->
   StartColumn[3] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,4]: OtherElementStartsInNextColumn[0,4] = 1 ->
   StartColumn[4] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,5]: OtherElementStartsInNextColumn[0,5] = 1 ->
   StartColumn[5] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,6]: OtherElementStartsInNextColumn[0,6] = 1 ->
   StartColumn[6] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,7]: OtherElementStartsInNextColumn[0,7] = 1 ->
   StartColumn[7] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,8]: OtherElementStartsInNextColumn[0,8] = 1 ->
   StartColumn[8] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,9]: OtherElementStartsInNextColumn[0,9] = 1 ->
   StartColumn[9] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[0,10]: OtherElementStartsInNextColumn[0,10] = 1 ->
   StartColumn[10] - StartColumn[0] = 1
 LinkOtherElementStartsInNextColumn2[1,0]: OtherElementStartsInNextColumn[1,0] = 1 ->
   StartColumn[0] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,2]: OtherElementStartsInNextColumn[1,2] = 1 ->
   StartColumn[2] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,3]: OtherElementStartsInNextColumn[1,3] = 1 ->
   StartColumn[3] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,4]: OtherElementStartsInNextColumn[1,4] = 1 ->
   StartColumn[4] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,5]: OtherElementStartsInNextColumn[1,5] = 1 ->
   StartColumn[5] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,6]: OtherElementStartsInNextColumn[1,6] = 1 ->
   StartColumn[6] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,7]: OtherElementStartsInNextColumn[1,7] = 1 ->
   StartColumn[7] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,8]: OtherElementStartsInNextColumn[1,8] = 1 ->
   StartColumn[8] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,9]: OtherElementStartsInNextColumn[1,9] = 1 ->
   StartColumn[9] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[1,10]: OtherElementStartsInNextColumn[1,10] = 1 ->
   StartColumn[10] - StartColumn[1] = 1
 LinkOtherElementStartsInNextColumn2[2,0]: OtherElementStartsInNextColumn[2,0] = 1 ->
   StartColumn[0] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,1]: OtherElementStartsInNextColumn[2,1] = 1 ->
   StartColumn[1] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,3]: OtherElementStartsInNextColumn[2,3] = 1 ->
   StartColumn[3] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,4]: OtherElementStartsInNextColumn[2,4] = 1 ->
   StartColumn[4] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,5]: OtherElementStartsInNextColumn[2,5] = 1 ->
   StartColumn[5] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,6]: OtherElementStartsInNextColumn[2,6] = 1 ->
   StartColumn[6] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,7]: OtherElementStartsInNextColumn[2,7] = 1 ->
   StartColumn[7] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,8]: OtherElementStartsInNextColumn[2,8] = 1 ->
   StartColumn[8] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,9]: OtherElementStartsInNextColumn[2,9] = 1 ->
   StartColumn[9] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[2,10]: OtherElementStartsInNextColumn[2,10] = 1 ->
   StartColumn[10] - StartColumn[2] = 1
 LinkOtherElementStartsInNextColumn2[3,0]: OtherElementStartsInNextColumn[3,0] = 1 ->
   StartColumn[0] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,1]: OtherElementStartsInNextColumn[3,1] = 1 ->
   StartColumn[1] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,2]: OtherElementStartsInNextColumn[3,2] = 1 ->
   StartColumn[2] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,4]: OtherElementStartsInNextColumn[3,4] = 1 ->
   StartColumn[4] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,5]: OtherElementStartsInNextColumn[3,5] = 1 ->
   StartColumn[5] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,6]: OtherElementStartsInNextColumn[3,6] = 1 ->
   StartColumn[6] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,7]: OtherElementStartsInNextColumn[3,7] = 1 ->
   StartColumn[7] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,8]: OtherElementStartsInNextColumn[3,8] = 1 ->
   StartColumn[8] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,9]: OtherElementStartsInNextColumn[3,9] = 1 ->
   StartColumn[9] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[3,10]: OtherElementStartsInNextColumn[3,10] = 1 ->
   StartColumn[10] - StartColumn[3] = 1
 LinkOtherElementStartsInNextColumn2[4,0]: OtherElementStartsInNextColumn[4,0] = 1 ->
   StartColumn[0] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,1]: OtherElementStartsInNextColumn[4,1] = 1 ->
   StartColumn[1] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,2]: OtherElementStartsInNextColumn[4,2] = 1 ->
   StartColumn[2] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,3]: OtherElementStartsInNextColumn[4,3] = 1 ->
   StartColumn[3] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,5]: OtherElementStartsInNextColumn[4,5] = 1 ->
   StartColumn[5] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,6]: OtherElementStartsInNextColumn[4,6] = 1 ->
   StartColumn[6] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,7]: OtherElementStartsInNextColumn[4,7] = 1 ->
   StartColumn[7] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,8]: OtherElementStartsInNextColumn[4,8] = 1 ->
   StartColumn[8] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,9]: OtherElementStartsInNextColumn[4,9] = 1 ->
   StartColumn[9] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[4,10]: OtherElementStartsInNextColumn[4,10] = 1 ->
   StartColumn[10] - StartColumn[4] = 1
 LinkOtherElementStartsInNextColumn2[5,0]: OtherElementStartsInNextColumn[5,0] = 1 ->
   StartColumn[0] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,1]: OtherElementStartsInNextColumn[5,1] = 1 ->
   StartColumn[1] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,2]: OtherElementStartsInNextColumn[5,2] = 1 ->
   StartColumn[2] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,3]: OtherElementStartsInNextColumn[5,3] = 1 ->
   StartColumn[3] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,4]: OtherElementStartsInNextColumn[5,4] = 1 ->
   StartColumn[4] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,6]: OtherElementStartsInNextColumn[5,6] = 1 ->
   StartColumn[6] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,7]: OtherElementStartsInNextColumn[5,7] = 1 ->
   StartColumn[7] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,8]: OtherElementStartsInNextColumn[5,8] = 1 ->
   StartColumn[8] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,9]: OtherElementStartsInNextColumn[5,9] = 1 ->
   StartColumn[9] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[5,10]: OtherElementStartsInNextColumn[5,10] = 1 ->
   StartColumn[10] - StartColumn[5] = 1
 LinkOtherElementStartsInNextColumn2[6,0]: OtherElementStartsInNextColumn[6,0] = 1 ->
   StartColumn[0] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,1]: OtherElementStartsInNextColumn[6,1] = 1 ->
   StartColumn[1] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,2]: OtherElementStartsInNextColumn[6,2] = 1 ->
   StartColumn[2] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,3]: OtherElementStartsInNextColumn[6,3] = 1 ->
   StartColumn[3] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,4]: OtherElementStartsInNextColumn[6,4] = 1 ->
   StartColumn[4] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,5]: OtherElementStartsInNextColumn[6,5] = 1 ->
   StartColumn[5] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,7]: OtherElementStartsInNextColumn[6,7] = 1 ->
   StartColumn[7] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,8]: OtherElementStartsInNextColumn[6,8] = 1 ->
   StartColumn[8] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,9]: OtherElementStartsInNextColumn[6,9] = 1 ->
   StartColumn[9] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[6,10]: OtherElementStartsInNextColumn[6,10] = 1 ->
   StartColumn[10] - StartColumn[6] = 1
 LinkOtherElementStartsInNextColumn2[7,0]: OtherElementStartsInNextColumn[7,0] = 1 ->
   StartColumn[0] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,1]: OtherElementStartsInNextColumn[7,1] = 1 ->
   StartColumn[1] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,2]: OtherElementStartsInNextColumn[7,2] = 1 ->
   StartColumn[2] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,3]: OtherElementStartsInNextColumn[7,3] = 1 ->
   StartColumn[3] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,4]: OtherElementStartsInNextColumn[7,4] = 1 ->
   StartColumn[4] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,5]: OtherElementStartsInNextColumn[7,5] = 1 ->
   StartColumn[5] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,6]: OtherElementStartsInNextColumn[7,6] = 1 ->
   StartColumn[6] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,8]: OtherElementStartsInNextColumn[7,8] = 1 ->
   StartColumn[8] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,9]: OtherElementStartsInNextColumn[7,9] = 1 ->
   StartColumn[9] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[7,10]: OtherElementStartsInNextColumn[7,10] = 1 ->
   StartColumn[10] - StartColumn[7] = 1
 LinkOtherElementStartsInNextColumn2[8,0]: OtherElementStartsInNextColumn[8,0] = 1 ->
   StartColumn[0] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,1]: OtherElementStartsInNextColumn[8,1] = 1 ->
   StartColumn[1] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,2]: OtherElementStartsInNextColumn[8,2] = 1 ->
   StartColumn[2] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,3]: OtherElementStartsInNextColumn[8,3] = 1 ->
   StartColumn[3] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,4]: OtherElementStartsInNextColumn[8,4] = 1 ->
   StartColumn[4] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,5]: OtherElementStartsInNextColumn[8,5] = 1 ->
   StartColumn[5] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,6]: OtherElementStartsInNextColumn[8,6] = 1 ->
   StartColumn[6] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,7]: OtherElementStartsInNextColumn[8,7] = 1 ->
   StartColumn[7] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,9]: OtherElementStartsInNextColumn[8,9] = 1 ->
   StartColumn[9] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[8,10]: OtherElementStartsInNextColumn[8,10] = 1 ->
   StartColumn[10] - StartColumn[8] = 1
 LinkOtherElementStartsInNextColumn2[9,0]: OtherElementStartsInNextColumn[9,0] = 1 ->
   StartColumn[0] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,1]: OtherElementStartsInNextColumn[9,1] = 1 ->
   StartColumn[1] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,2]: OtherElementStartsInNextColumn[9,2] = 1 ->
   StartColumn[2] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,3]: OtherElementStartsInNextColumn[9,3] = 1 ->
   StartColumn[3] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,4]: OtherElementStartsInNextColumn[9,4] = 1 ->
   StartColumn[4] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,5]: OtherElementStartsInNextColumn[9,5] = 1 ->
   StartColumn[5] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,6]: OtherElementStartsInNextColumn[9,6] = 1 ->
   StartColumn[6] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,7]: OtherElementStartsInNextColumn[9,7] = 1 ->
   StartColumn[7] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,8]: OtherElementStartsInNextColumn[9,8] = 1 ->
   StartColumn[8] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[9,10]: OtherElementStartsInNextColumn[9,10] = 1 ->
   StartColumn[10] - StartColumn[9] = 1
 LinkOtherElementStartsInNextColumn2[10,0]: OtherElementStartsInNextColumn[10,0] = 1 ->
   StartColumn[0] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,1]: OtherElementStartsInNextColumn[10,1] = 1 ->
   StartColumn[1] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,2]: OtherElementStartsInNextColumn[10,2] = 1 ->
   StartColumn[2] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,3]: OtherElementStartsInNextColumn[10,3] = 1 ->
   StartColumn[3] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,4]: OtherElementStartsInNextColumn[10,4] = 1 ->
   StartColumn[4] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,5]: OtherElementStartsInNextColumn[10,5] = 1 ->
   StartColumn[5] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,6]: OtherElementStartsInNextColumn[10,6] = 1 ->
   StartColumn[6] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,7]: OtherElementStartsInNextColumn[10,7] = 1 ->
   StartColumn[7] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,8]: OtherElementStartsInNextColumn[10,8] = 1 ->
   StartColumn[8] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn2[10,9]: OtherElementStartsInNextColumn[10,9] = 1 ->
   StartColumn[9] - StartColumn[10] = 1
 LinkOtherElementStartsInNextColumn3[0,1]: X0LessThan[0,1] = 1 ->
   StartColumn[1] - StartColumn[0] + OtherElementStartsInNextColumn[0,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,2]: X0LessThan[0,2] = 1 ->
   StartColumn[2] - StartColumn[0] + OtherElementStartsInNextColumn[0,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,3]: X0LessThan[0,3] = 1 ->
   StartColumn[3] - StartColumn[0] + OtherElementStartsInNextColumn[0,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,4]: X0LessThan[0,4] = 1 ->
   StartColumn[4] - StartColumn[0] + OtherElementStartsInNextColumn[0,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,5]: X0LessThan[0,5] = 1 ->
   StartColumn[5] - StartColumn[0] + OtherElementStartsInNextColumn[0,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,6]: X0LessThan[0,6] = 1 ->
   StartColumn[6] - StartColumn[0] + OtherElementStartsInNextColumn[0,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,7]: X0LessThan[0,7] = 1 ->
   StartColumn[7] - StartColumn[0] + OtherElementStartsInNextColumn[0,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,8]: X0LessThan[0,8] = 1 ->
   StartColumn[8] - StartColumn[0] + OtherElementStartsInNextColumn[0,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,9]: X0LessThan[0,9] = 1 ->
   StartColumn[9] - StartColumn[0] + OtherElementStartsInNextColumn[0,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[0,10]: X0LessThan[0,10] = 1 ->
   StartColumn[10] - StartColumn[0] + OtherElementStartsInNextColumn[0,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,0]: X0LessThan[1,0] = 1 ->
   StartColumn[0] - StartColumn[1] + OtherElementStartsInNextColumn[1,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,2]: X0LessThan[1,2] = 1 ->
   StartColumn[2] - StartColumn[1] + OtherElementStartsInNextColumn[1,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,3]: X0LessThan[1,3] = 1 ->
   StartColumn[3] - StartColumn[1] + OtherElementStartsInNextColumn[1,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,4]: X0LessThan[1,4] = 1 ->
   StartColumn[4] - StartColumn[1] + OtherElementStartsInNextColumn[1,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,5]: X0LessThan[1,5] = 1 ->
   StartColumn[5] - StartColumn[1] + OtherElementStartsInNextColumn[1,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,6]: X0LessThan[1,6] = 1 ->
   StartColumn[6] - StartColumn[1] + OtherElementStartsInNextColumn[1,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,7]: X0LessThan[1,7] = 1 ->
   StartColumn[7] - StartColumn[1] + OtherElementStartsInNextColumn[1,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,8]: X0LessThan[1,8] = 1 ->
   StartColumn[8] - StartColumn[1] + OtherElementStartsInNextColumn[1,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,9]: X0LessThan[1,9] = 1 ->
   StartColumn[9] - StartColumn[1] + OtherElementStartsInNextColumn[1,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[1,10]: X0LessThan[1,10] = 1 ->
   StartColumn[10] - StartColumn[1] + OtherElementStartsInNextColumn[1,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,0]: X0LessThan[2,0] = 1 ->
   StartColumn[0] - StartColumn[2] + OtherElementStartsInNextColumn[2,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,1]: X0LessThan[2,1] = 1 ->
   StartColumn[1] - StartColumn[2] + OtherElementStartsInNextColumn[2,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,3]: X0LessThan[2,3] = 1 ->
   StartColumn[3] - StartColumn[2] + OtherElementStartsInNextColumn[2,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,4]: X0LessThan[2,4] = 1 ->
   StartColumn[4] - StartColumn[2] + OtherElementStartsInNextColumn[2,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,5]: X0LessThan[2,5] = 1 ->
   StartColumn[5] - StartColumn[2] + OtherElementStartsInNextColumn[2,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,6]: X0LessThan[2,6] = 1 ->
   StartColumn[6] - StartColumn[2] + OtherElementStartsInNextColumn[2,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,7]: X0LessThan[2,7] = 1 ->
   StartColumn[7] - StartColumn[2] + OtherElementStartsInNextColumn[2,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,8]: X0LessThan[2,8] = 1 ->
   StartColumn[8] - StartColumn[2] + OtherElementStartsInNextColumn[2,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,9]: X0LessThan[2,9] = 1 ->
   StartColumn[9] - StartColumn[2] + OtherElementStartsInNextColumn[2,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[2,10]: X0LessThan[2,10] = 1 ->
   StartColumn[10] - StartColumn[2] + OtherElementStartsInNextColumn[2,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,0]: X0LessThan[3,0] = 1 ->
   StartColumn[0] - StartColumn[3] + OtherElementStartsInNextColumn[3,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,1]: X0LessThan[3,1] = 1 ->
   StartColumn[1] - StartColumn[3] + OtherElementStartsInNextColumn[3,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,2]: X0LessThan[3,2] = 1 ->
   StartColumn[2] - StartColumn[3] + OtherElementStartsInNextColumn[3,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,4]: X0LessThan[3,4] = 1 ->
   StartColumn[4] - StartColumn[3] + OtherElementStartsInNextColumn[3,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,5]: X0LessThan[3,5] = 1 ->
   StartColumn[5] - StartColumn[3] + OtherElementStartsInNextColumn[3,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,6]: X0LessThan[3,6] = 1 ->
   StartColumn[6] - StartColumn[3] + OtherElementStartsInNextColumn[3,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,7]: X0LessThan[3,7] = 1 ->
   StartColumn[7] - StartColumn[3] + OtherElementStartsInNextColumn[3,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,8]: X0LessThan[3,8] = 1 ->
   StartColumn[8] - StartColumn[3] + OtherElementStartsInNextColumn[3,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,9]: X0LessThan[3,9] = 1 ->
   StartColumn[9] - StartColumn[3] + OtherElementStartsInNextColumn[3,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[3,10]: X0LessThan[3,10] = 1 ->
   StartColumn[10] - StartColumn[3] + OtherElementStartsInNextColumn[3,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,0]: X0LessThan[4,0] = 1 ->
   StartColumn[0] - StartColumn[4] + OtherElementStartsInNextColumn[4,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,1]: X0LessThan[4,1] = 1 ->
   StartColumn[1] - StartColumn[4] + OtherElementStartsInNextColumn[4,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,2]: X0LessThan[4,2] = 1 ->
   StartColumn[2] - StartColumn[4] + OtherElementStartsInNextColumn[4,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,3]: X0LessThan[4,3] = 1 ->
   StartColumn[3] - StartColumn[4] + OtherElementStartsInNextColumn[4,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,5]: X0LessThan[4,5] = 1 ->
   StartColumn[5] - StartColumn[4] + OtherElementStartsInNextColumn[4,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,6]: X0LessThan[4,6] = 1 ->
   StartColumn[6] - StartColumn[4] + OtherElementStartsInNextColumn[4,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,7]: X0LessThan[4,7] = 1 ->
   StartColumn[7] - StartColumn[4] + OtherElementStartsInNextColumn[4,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,8]: X0LessThan[4,8] = 1 ->
   StartColumn[8] - StartColumn[4] + OtherElementStartsInNextColumn[4,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,9]: X0LessThan[4,9] = 1 ->
   StartColumn[9] - StartColumn[4] + OtherElementStartsInNextColumn[4,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[4,10]: X0LessThan[4,10] = 1 ->
   StartColumn[10] - StartColumn[4] + OtherElementStartsInNextColumn[4,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,0]: X0LessThan[5,0] = 1 ->
   StartColumn[0] - StartColumn[5] + OtherElementStartsInNextColumn[5,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,1]: X0LessThan[5,1] = 1 ->
   StartColumn[1] - StartColumn[5] + OtherElementStartsInNextColumn[5,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,2]: X0LessThan[5,2] = 1 ->
   StartColumn[2] - StartColumn[5] + OtherElementStartsInNextColumn[5,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,3]: X0LessThan[5,3] = 1 ->
   StartColumn[3] - StartColumn[5] + OtherElementStartsInNextColumn[5,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,4]: X0LessThan[5,4] = 1 ->
   StartColumn[4] - StartColumn[5] + OtherElementStartsInNextColumn[5,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,6]: X0LessThan[5,6] = 1 ->
   StartColumn[6] - StartColumn[5] + OtherElementStartsInNextColumn[5,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,7]: X0LessThan[5,7] = 1 ->
   StartColumn[7] - StartColumn[5] + OtherElementStartsInNextColumn[5,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,8]: X0LessThan[5,8] = 1 ->
   StartColumn[8] - StartColumn[5] + OtherElementStartsInNextColumn[5,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,9]: X0LessThan[5,9] = 1 ->
   StartColumn[9] - StartColumn[5] + OtherElementStartsInNextColumn[5,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[5,10]: X0LessThan[5,10] = 1 ->
   StartColumn[10] - StartColumn[5] + OtherElementStartsInNextColumn[5,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,0]: X0LessThan[6,0] = 1 ->
   StartColumn[0] - StartColumn[6] + OtherElementStartsInNextColumn[6,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,1]: X0LessThan[6,1] = 1 ->
   StartColumn[1] - StartColumn[6] + OtherElementStartsInNextColumn[6,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,2]: X0LessThan[6,2] = 1 ->
   StartColumn[2] - StartColumn[6] + OtherElementStartsInNextColumn[6,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,3]: X0LessThan[6,3] = 1 ->
   StartColumn[3] - StartColumn[6] + OtherElementStartsInNextColumn[6,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,4]: X0LessThan[6,4] = 1 ->
   StartColumn[4] - StartColumn[6] + OtherElementStartsInNextColumn[6,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,5]: X0LessThan[6,5] = 1 ->
   StartColumn[5] - StartColumn[6] + OtherElementStartsInNextColumn[6,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,7]: X0LessThan[6,7] = 1 ->
   StartColumn[7] - StartColumn[6] + OtherElementStartsInNextColumn[6,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,8]: X0LessThan[6,8] = 1 ->
   StartColumn[8] - StartColumn[6] + OtherElementStartsInNextColumn[6,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,9]: X0LessThan[6,9] = 1 ->
   StartColumn[9] - StartColumn[6] + OtherElementStartsInNextColumn[6,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[6,10]: X0LessThan[6,10] = 1 ->
   StartColumn[10] - StartColumn[6] + OtherElementStartsInNextColumn[6,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,0]: X0LessThan[7,0] = 1 ->
   StartColumn[0] - StartColumn[7] + OtherElementStartsInNextColumn[7,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,1]: X0LessThan[7,1] = 1 ->
   StartColumn[1] - StartColumn[7] + OtherElementStartsInNextColumn[7,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,2]: X0LessThan[7,2] = 1 ->
   StartColumn[2] - StartColumn[7] + OtherElementStartsInNextColumn[7,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,3]: X0LessThan[7,3] = 1 ->
   StartColumn[3] - StartColumn[7] + OtherElementStartsInNextColumn[7,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,4]: X0LessThan[7,4] = 1 ->
   StartColumn[4] - StartColumn[7] + OtherElementStartsInNextColumn[7,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,5]: X0LessThan[7,5] = 1 ->
   StartColumn[5] - StartColumn[7] + OtherElementStartsInNextColumn[7,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,6]: X0LessThan[7,6] = 1 ->
   StartColumn[6] - StartColumn[7] + OtherElementStartsInNextColumn[7,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,8]: X0LessThan[7,8] = 1 ->
   StartColumn[8] - StartColumn[7] + OtherElementStartsInNextColumn[7,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,9]: X0LessThan[7,9] = 1 ->
   StartColumn[9] - StartColumn[7] + OtherElementStartsInNextColumn[7,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[7,10]: X0LessThan[7,10] = 1 ->
   StartColumn[10] - StartColumn[7] + OtherElementStartsInNextColumn[7,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,0]: X0LessThan[8,0] = 1 ->
   StartColumn[0] - StartColumn[8] + OtherElementStartsInNextColumn[8,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,1]: X0LessThan[8,1] = 1 ->
   StartColumn[1] - StartColumn[8] + OtherElementStartsInNextColumn[8,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,2]: X0LessThan[8,2] = 1 ->
   StartColumn[2] - StartColumn[8] + OtherElementStartsInNextColumn[8,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,3]: X0LessThan[8,3] = 1 ->
   StartColumn[3] - StartColumn[8] + OtherElementStartsInNextColumn[8,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,4]: X0LessThan[8,4] = 1 ->
   StartColumn[4] - StartColumn[8] + OtherElementStartsInNextColumn[8,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,5]: X0LessThan[8,5] = 1 ->
   StartColumn[5] - StartColumn[8] + OtherElementStartsInNextColumn[8,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,6]: X0LessThan[8,6] = 1 ->
   StartColumn[6] - StartColumn[8] + OtherElementStartsInNextColumn[8,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,7]: X0LessThan[8,7] = 1 ->
   StartColumn[7] - StartColumn[8] + OtherElementStartsInNextColumn[8,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,9]: X0LessThan[8,9] = 1 ->
   StartColumn[9] - StartColumn[8] + OtherElementStartsInNextColumn[8,9]
   >= 2
 LinkOtherElementStartsInNextColumn3[8,10]: X0LessThan[8,10] = 1 ->
   StartColumn[10] - StartColumn[8] + OtherElementStartsInNextColumn[8,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,0]: X0LessThan[9,0] = 1 ->
   StartColumn[0] - StartColumn[9] + OtherElementStartsInNextColumn[9,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,1]: X0LessThan[9,1] = 1 ->
   StartColumn[1] - StartColumn[9] + OtherElementStartsInNextColumn[9,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,2]: X0LessThan[9,2] = 1 ->
   StartColumn[2] - StartColumn[9] + OtherElementStartsInNextColumn[9,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,3]: X0LessThan[9,3] = 1 ->
   StartColumn[3] - StartColumn[9] + OtherElementStartsInNextColumn[9,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,4]: X0LessThan[9,4] = 1 ->
   StartColumn[4] - StartColumn[9] + OtherElementStartsInNextColumn[9,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,5]: X0LessThan[9,5] = 1 ->
   StartColumn[5] - StartColumn[9] + OtherElementStartsInNextColumn[9,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,6]: X0LessThan[9,6] = 1 ->
   StartColumn[6] - StartColumn[9] + OtherElementStartsInNextColumn[9,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,7]: X0LessThan[9,7] = 1 ->
   StartColumn[7] - StartColumn[9] + OtherElementStartsInNextColumn[9,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,8]: X0LessThan[9,8] = 1 ->
   StartColumn[8] - StartColumn[9] + OtherElementStartsInNextColumn[9,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[9,10]: X0LessThan[9,10] = 1 ->
   StartColumn[10] - StartColumn[9] + OtherElementStartsInNextColumn[9,10]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,0]: X0LessThan[10,0] = 1 ->
   StartColumn[0] - StartColumn[10] + OtherElementStartsInNextColumn[10,0]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,1]: X0LessThan[10,1] = 1 ->
   StartColumn[1] - StartColumn[10] + OtherElementStartsInNextColumn[10,1]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,2]: X0LessThan[10,2] = 1 ->
   StartColumn[2] - StartColumn[10] + OtherElementStartsInNextColumn[10,2]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,3]: X0LessThan[10,3] = 1 ->
   StartColumn[3] - StartColumn[10] + OtherElementStartsInNextColumn[10,3]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,4]: X0LessThan[10,4] = 1 ->
   StartColumn[4] - StartColumn[10] + OtherElementStartsInNextColumn[10,4]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,5]: X0LessThan[10,5] = 1 ->
   StartColumn[5] - StartColumn[10] + OtherElementStartsInNextColumn[10,5]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,6]: X0LessThan[10,6] = 1 ->
   StartColumn[6] - StartColumn[10] + OtherElementStartsInNextColumn[10,6]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,7]: X0LessThan[10,7] = 1 ->
   StartColumn[7] - StartColumn[10] + OtherElementStartsInNextColumn[10,7]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,8]: X0LessThan[10,8] = 1 ->
   StartColumn[8] - StartColumn[10] + OtherElementStartsInNextColumn[10,8]
   >= 2
 LinkOtherElementStartsInNextColumn3[10,9]: X0LessThan[10,9] = 1 ->
   StartColumn[9] - StartColumn[10] + OtherElementStartsInNextColumn[10,9]
   >= 2
 LinkOtherElementEndsInNextColumn2[0,1]: OtherElementEndsInNextColumn[0,1] = 1 ->
   EndColumn[1] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,2]: OtherElementEndsInNextColumn[0,2] = 1 ->
   EndColumn[2] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,3]: OtherElementEndsInNextColumn[0,3] = 1 ->
   EndColumn[3] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,4]: OtherElementEndsInNextColumn[0,4] = 1 ->
   EndColumn[4] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,5]: OtherElementEndsInNextColumn[0,5] = 1 ->
   EndColumn[5] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,6]: OtherElementEndsInNextColumn[0,6] = 1 ->
   EndColumn[6] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,7]: OtherElementEndsInNextColumn[0,7] = 1 ->
   EndColumn[7] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,8]: OtherElementEndsInNextColumn[0,8] = 1 ->
   EndColumn[8] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,9]: OtherElementEndsInNextColumn[0,9] = 1 ->
   EndColumn[9] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[0,10]: OtherElementEndsInNextColumn[0,10] = 1 ->
   EndColumn[10] - EndColumn[0] = 1
 LinkOtherElementEndsInNextColumn2[1,0]: OtherElementEndsInNextColumn[1,0] = 1 ->
   EndColumn[0] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,2]: OtherElementEndsInNextColumn[1,2] = 1 ->
   EndColumn[2] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,3]: OtherElementEndsInNextColumn[1,3] = 1 ->
   EndColumn[3] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,4]: OtherElementEndsInNextColumn[1,4] = 1 ->
   EndColumn[4] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,5]: OtherElementEndsInNextColumn[1,5] = 1 ->
   EndColumn[5] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,6]: OtherElementEndsInNextColumn[1,6] = 1 ->
   EndColumn[6] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,7]: OtherElementEndsInNextColumn[1,7] = 1 ->
   EndColumn[7] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,8]: OtherElementEndsInNextColumn[1,8] = 1 ->
   EndColumn[8] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,9]: OtherElementEndsInNextColumn[1,9] = 1 ->
   EndColumn[9] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[1,10]: OtherElementEndsInNextColumn[1,10] = 1 ->
   EndColumn[10] - EndColumn[1] = 1
 LinkOtherElementEndsInNextColumn2[2,0]: OtherElementEndsInNextColumn[2,0] = 1 ->
   EndColumn[0] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,1]: OtherElementEndsInNextColumn[2,1] = 1 ->
   EndColumn[1] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,3]: OtherElementEndsInNextColumn[2,3] = 1 ->
   EndColumn[3] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,4]: OtherElementEndsInNextColumn[2,4] = 1 ->
   EndColumn[4] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,5]: OtherElementEndsInNextColumn[2,5] = 1 ->
   EndColumn[5] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,6]: OtherElementEndsInNextColumn[2,6] = 1 ->
   EndColumn[6] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,7]: OtherElementEndsInNextColumn[2,7] = 1 ->
   EndColumn[7] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,8]: OtherElementEndsInNextColumn[2,8] = 1 ->
   EndColumn[8] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,9]: OtherElementEndsInNextColumn[2,9] = 1 ->
   EndColumn[9] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[2,10]: OtherElementEndsInNextColumn[2,10] = 1 ->
   EndColumn[10] - EndColumn[2] = 1
 LinkOtherElementEndsInNextColumn2[3,0]: OtherElementEndsInNextColumn[3,0] = 1 ->
   EndColumn[0] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,1]: OtherElementEndsInNextColumn[3,1] = 1 ->
   EndColumn[1] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,2]: OtherElementEndsInNextColumn[3,2] = 1 ->
   EndColumn[2] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,4]: OtherElementEndsInNextColumn[3,4] = 1 ->
   EndColumn[4] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,5]: OtherElementEndsInNextColumn[3,5] = 1 ->
   EndColumn[5] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,6]: OtherElementEndsInNextColumn[3,6] = 1 ->
   EndColumn[6] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,7]: OtherElementEndsInNextColumn[3,7] = 1 ->
   EndColumn[7] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,8]: OtherElementEndsInNextColumn[3,8] = 1 ->
   EndColumn[8] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,9]: OtherElementEndsInNextColumn[3,9] = 1 ->
   EndColumn[9] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[3,10]: OtherElementEndsInNextColumn[3,10] = 1 ->
   EndColumn[10] - EndColumn[3] = 1
 LinkOtherElementEndsInNextColumn2[4,0]: OtherElementEndsInNextColumn[4,0] = 1 ->
   EndColumn[0] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,1]: OtherElementEndsInNextColumn[4,1] = 1 ->
   EndColumn[1] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,2]: OtherElementEndsInNextColumn[4,2] = 1 ->
   EndColumn[2] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,3]: OtherElementEndsInNextColumn[4,3] = 1 ->
   EndColumn[3] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,5]: OtherElementEndsInNextColumn[4,5] = 1 ->
   EndColumn[5] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,6]: OtherElementEndsInNextColumn[4,6] = 1 ->
   EndColumn[6] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,7]: OtherElementEndsInNextColumn[4,7] = 1 ->
   EndColumn[7] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,8]: OtherElementEndsInNextColumn[4,8] = 1 ->
   EndColumn[8] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,9]: OtherElementEndsInNextColumn[4,9] = 1 ->
   EndColumn[9] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[4,10]: OtherElementEndsInNextColumn[4,10] = 1 ->
   EndColumn[10] - EndColumn[4] = 1
 LinkOtherElementEndsInNextColumn2[5,0]: OtherElementEndsInNextColumn[5,0] = 1 ->
   EndColumn[0] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,1]: OtherElementEndsInNextColumn[5,1] = 1 ->
   EndColumn[1] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,2]: OtherElementEndsInNextColumn[5,2] = 1 ->
   EndColumn[2] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,3]: OtherElementEndsInNextColumn[5,3] = 1 ->
   EndColumn[3] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,4]: OtherElementEndsInNextColumn[5,4] = 1 ->
   EndColumn[4] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,6]: OtherElementEndsInNextColumn[5,6] = 1 ->
   EndColumn[6] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,7]: OtherElementEndsInNextColumn[5,7] = 1 ->
   EndColumn[7] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,8]: OtherElementEndsInNextColumn[5,8] = 1 ->
   EndColumn[8] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,9]: OtherElementEndsInNextColumn[5,9] = 1 ->
   EndColumn[9] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[5,10]: OtherElementEndsInNextColumn[5,10] = 1 ->
   EndColumn[10] - EndColumn[5] = 1
 LinkOtherElementEndsInNextColumn2[6,0]: OtherElementEndsInNextColumn[6,0] = 1 ->
   EndColumn[0] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,1]: OtherElementEndsInNextColumn[6,1] = 1 ->
   EndColumn[1] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,2]: OtherElementEndsInNextColumn[6,2] = 1 ->
   EndColumn[2] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,3]: OtherElementEndsInNextColumn[6,3] = 1 ->
   EndColumn[3] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,4]: OtherElementEndsInNextColumn[6,4] = 1 ->
   EndColumn[4] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,5]: OtherElementEndsInNextColumn[6,5] = 1 ->
   EndColumn[5] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,7]: OtherElementEndsInNextColumn[6,7] = 1 ->
   EndColumn[7] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,8]: OtherElementEndsInNextColumn[6,8] = 1 ->
   EndColumn[8] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,9]: OtherElementEndsInNextColumn[6,9] = 1 ->
   EndColumn[9] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[6,10]: OtherElementEndsInNextColumn[6,10] = 1 ->
   EndColumn[10] - EndColumn[6] = 1
 LinkOtherElementEndsInNextColumn2[7,0]: OtherElementEndsInNextColumn[7,0] = 1 ->
   EndColumn[0] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,1]: OtherElementEndsInNextColumn[7,1] = 1 ->
   EndColumn[1] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,2]: OtherElementEndsInNextColumn[7,2] = 1 ->
   EndColumn[2] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,3]: OtherElementEndsInNextColumn[7,3] = 1 ->
   EndColumn[3] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,4]: OtherElementEndsInNextColumn[7,4] = 1 ->
   EndColumn[4] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,5]: OtherElementEndsInNextColumn[7,5] = 1 ->
   EndColumn[5] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,6]: OtherElementEndsInNextColumn[7,6] = 1 ->
   EndColumn[6] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,8]: OtherElementEndsInNextColumn[7,8] = 1 ->
   EndColumn[8] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,9]: OtherElementEndsInNextColumn[7,9] = 1 ->
   EndColumn[9] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[7,10]: OtherElementEndsInNextColumn[7,10] = 1 ->
   EndColumn[10] - EndColumn[7] = 1
 LinkOtherElementEndsInNextColumn2[8,0]: OtherElementEndsInNextColumn[8,0] = 1 ->
   EndColumn[0] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,1]: OtherElementEndsInNextColumn[8,1] = 1 ->
   EndColumn[1] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,2]: OtherElementEndsInNextColumn[8,2] = 1 ->
   EndColumn[2] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,3]: OtherElementEndsInNextColumn[8,3] = 1 ->
   EndColumn[3] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,4]: OtherElementEndsInNextColumn[8,4] = 1 ->
   EndColumn[4] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,5]: OtherElementEndsInNextColumn[8,5] = 1 ->
   EndColumn[5] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,6]: OtherElementEndsInNextColumn[8,6] = 1 ->
   EndColumn[6] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,7]: OtherElementEndsInNextColumn[8,7] = 1 ->
   EndColumn[7] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,9]: OtherElementEndsInNextColumn[8,9] = 1 ->
   EndColumn[9] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[8,10]: OtherElementEndsInNextColumn[8,10] = 1 ->
   EndColumn[10] - EndColumn[8] = 1
 LinkOtherElementEndsInNextColumn2[9,0]: OtherElementEndsInNextColumn[9,0] = 1 ->
   EndColumn[0] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,1]: OtherElementEndsInNextColumn[9,1] = 1 ->
   EndColumn[1] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,2]: OtherElementEndsInNextColumn[9,2] = 1 ->
   EndColumn[2] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,3]: OtherElementEndsInNextColumn[9,3] = 1 ->
   EndColumn[3] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,4]: OtherElementEndsInNextColumn[9,4] = 1 ->
   EndColumn[4] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,5]: OtherElementEndsInNextColumn[9,5] = 1 ->
   EndColumn[5] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,6]: OtherElementEndsInNextColumn[9,6] = 1 ->
   EndColumn[6] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,7]: OtherElementEndsInNextColumn[9,7] = 1 ->
   EndColumn[7] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,8]: OtherElementEndsInNextColumn[9,8] = 1 ->
   EndColumn[8] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[9,10]: OtherElementEndsInNextColumn[9,10] = 1 ->
   EndColumn[10] - EndColumn[9] = 1
 LinkOtherElementEndsInNextColumn2[10,0]: OtherElementEndsInNextColumn[10,0] = 1 ->
   EndColumn[0] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,1]: OtherElementEndsInNextColumn[10,1] = 1 ->
   EndColumn[1] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,2]: OtherElementEndsInNextColumn[10,2] = 1 ->
   EndColumn[2] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,3]: OtherElementEndsInNextColumn[10,3] = 1 ->
   EndColumn[3] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,4]: OtherElementEndsInNextColumn[10,4] = 1 ->
   EndColumn[4] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,5]: OtherElementEndsInNextColumn[10,5] = 1 ->
   EndColumn[5] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,6]: OtherElementEndsInNextColumn[10,6] = 1 ->
   EndColumn[6] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,7]: OtherElementEndsInNextColumn[10,7] = 1 ->
   EndColumn[7] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,8]: OtherElementEndsInNextColumn[10,8] = 1 ->
   EndColumn[8] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn2[10,9]: OtherElementEndsInNextColumn[10,9] = 1 ->
   EndColumn[9] - EndColumn[10] = 1
 LinkOtherElementEndsInNextColumn3[0,1]: X1LessThan[0,1] = 1 ->
   EndColumn[1] - EndColumn[0] + OtherElementEndsInNextColumn[0,1] >= 2
 LinkOtherElementEndsInNextColumn3[0,2]: X1LessThan[0,2] = 1 ->
   EndColumn[2] - EndColumn[0] + OtherElementEndsInNextColumn[0,2] >= 2
 LinkOtherElementEndsInNextColumn3[0,3]: X1LessThan[0,3] = 1 ->
   EndColumn[3] - EndColumn[0] + OtherElementEndsInNextColumn[0,3] >= 2
 LinkOtherElementEndsInNextColumn3[0,4]: X1LessThan[0,4] = 1 ->
   EndColumn[4] - EndColumn[0] + OtherElementEndsInNextColumn[0,4] >= 2
 LinkOtherElementEndsInNextColumn3[0,5]: X1LessThan[0,5] = 1 ->
   EndColumn[5] - EndColumn[0] + OtherElementEndsInNextColumn[0,5] >= 2
 LinkOtherElementEndsInNextColumn3[0,6]: X1LessThan[0,6] = 1 ->
   EndColumn[6] - EndColumn[0] + OtherElementEndsInNextColumn[0,6] >= 2
 LinkOtherElementEndsInNextColumn3[0,7]: X1LessThan[0,7] = 1 ->
   EndColumn[7] - EndColumn[0] + OtherElementEndsInNextColumn[0,7] >= 2
 LinkOtherElementEndsInNextColumn3[0,8]: X1LessThan[0,8] = 1 ->
   EndColumn[8] - EndColumn[0] + OtherElementEndsInNextColumn[0,8] >= 2
 LinkOtherElementEndsInNextColumn3[0,9]: X1LessThan[0,9] = 1 ->
   EndColumn[9] - EndColumn[0] + OtherElementEndsInNextColumn[0,9] >= 2
 LinkOtherElementEndsInNextColumn3[0,10]: X1LessThan[0,10] = 1 ->
   EndColumn[10] - EndColumn[0] + OtherElementEndsInNextColumn[0,10] >= 2
 LinkOtherElementEndsInNextColumn3[1,0]: X1LessThan[1,0] = 1 ->
   EndColumn[0] - EndColumn[1] + OtherElementEndsInNextColumn[1,0] >= 2
 LinkOtherElementEndsInNextColumn3[1,2]: X1LessThan[1,2] = 1 ->
   EndColumn[2] - EndColumn[1] + OtherElementEndsInNextColumn[1,2] >= 2
 LinkOtherElementEndsInNextColumn3[1,3]: X1LessThan[1,3] = 1 ->
   EndColumn[3] - EndColumn[1] + OtherElementEndsInNextColumn[1,3] >= 2
 LinkOtherElementEndsInNextColumn3[1,4]: X1LessThan[1,4] = 1 ->
   EndColumn[4] - EndColumn[1] + OtherElementEndsInNextColumn[1,4] >= 2
 LinkOtherElementEndsInNextColumn3[1,5]: X1LessThan[1,5] = 1 ->
   EndColumn[5] - EndColumn[1] + OtherElementEndsInNextColumn[1,5] >= 2
 LinkOtherElementEndsInNextColumn3[1,6]: X1LessThan[1,6] = 1 ->
   EndColumn[6] - EndColumn[1] + OtherElementEndsInNextColumn[1,6] >= 2
 LinkOtherElementEndsInNextColumn3[1,7]: X1LessThan[1,7] = 1 ->
   EndColumn[7] - EndColumn[1] + OtherElementEndsInNextColumn[1,7] >= 2
 LinkOtherElementEndsInNextColumn3[1,8]: X1LessThan[1,8] = 1 ->
   EndColumn[8] - EndColumn[1] + OtherElementEndsInNextColumn[1,8] >= 2
 LinkOtherElementEndsInNextColumn3[1,9]: X1LessThan[1,9] = 1 ->
   EndColumn[9] - EndColumn[1] + OtherElementEndsInNextColumn[1,9] >= 2
 LinkOtherElementEndsInNextColumn3[1,10]: X1LessThan[1,10] = 1 ->
   EndColumn[10] - EndColumn[1] + OtherElementEndsInNextColumn[1,10] >= 2
 LinkOtherElementEndsInNextColumn3[2,0]: X1LessThan[2,0] = 1 ->
   EndColumn[0] - EndColumn[2] + OtherElementEndsInNextColumn[2,0] >= 2
 LinkOtherElementEndsInNextColumn3[2,1]: X1LessThan[2,1] = 1 ->
   EndColumn[1] - EndColumn[2] + OtherElementEndsInNextColumn[2,1] >= 2
 LinkOtherElementEndsInNextColumn3[2,3]: X1LessThan[2,3] = 1 ->
   EndColumn[3] - EndColumn[2] + OtherElementEndsInNextColumn[2,3] >= 2
 LinkOtherElementEndsInNextColumn3[2,4]: X1LessThan[2,4] = 1 ->
   EndColumn[4] - EndColumn[2] + OtherElementEndsInNextColumn[2,4] >= 2
 LinkOtherElementEndsInNextColumn3[2,5]: X1LessThan[2,5] = 1 ->
   EndColumn[5] - EndColumn[2] + OtherElementEndsInNextColumn[2,5] >= 2
 LinkOtherElementEndsInNextColumn3[2,6]: X1LessThan[2,6] = 1 ->
   EndColumn[6] - EndColumn[2] + OtherElementEndsInNextColumn[2,6] >= 2
 LinkOtherElementEndsInNextColumn3[2,7]: X1LessThan[2,7] = 1 ->
   EndColumn[7] - EndColumn[2] + OtherElementEndsInNextColumn[2,7] >= 2
 LinkOtherElementEndsInNextColumn3[2,8]: X1LessThan[2,8] = 1 ->
   EndColumn[8] - EndColumn[2] + OtherElementEndsInNextColumn[2,8] >= 2
 LinkOtherElementEndsInNextColumn3[2,9]: X1LessThan[2,9] = 1 ->
   EndColumn[9] - EndColumn[2] + OtherElementEndsInNextColumn[2,9] >= 2
 LinkOtherElementEndsInNextColumn3[2,10]: X1LessThan[2,10] = 1 ->
   EndColumn[10] - EndColumn[2] + OtherElementEndsInNextColumn[2,10] >= 2
 LinkOtherElementEndsInNextColumn3[3,0]: X1LessThan[3,0] = 1 ->
   EndColumn[0] - EndColumn[3] + OtherElementEndsInNextColumn[3,0] >= 2
 LinkOtherElementEndsInNextColumn3[3,1]: X1LessThan[3,1] = 1 ->
   EndColumn[1] - EndColumn[3] + OtherElementEndsInNextColumn[3,1] >= 2
 LinkOtherElementEndsInNextColumn3[3,2]: X1LessThan[3,2] = 1 ->
   EndColumn[2] - EndColumn[3] + OtherElementEndsInNextColumn[3,2] >= 2
 LinkOtherElementEndsInNextColumn3[3,4]: X1LessThan[3,4] = 1 ->
   EndColumn[4] - EndColumn[3] + OtherElementEndsInNextColumn[3,4] >= 2
 LinkOtherElementEndsInNextColumn3[3,5]: X1LessThan[3,5] = 1 ->
   EndColumn[5] - EndColumn[3] + OtherElementEndsInNextColumn[3,5] >= 2
 LinkOtherElementEndsInNextColumn3[3,6]: X1LessThan[3,6] = 1 ->
   EndColumn[6] - EndColumn[3] + OtherElementEndsInNextColumn[3,6] >= 2
 LinkOtherElementEndsInNextColumn3[3,7]: X1LessThan[3,7] = 1 ->
   EndColumn[7] - EndColumn[3] + OtherElementEndsInNextColumn[3,7] >= 2
 LinkOtherElementEndsInNextColumn3[3,8]: X1LessThan[3,8] = 1 ->
   EndColumn[8] - EndColumn[3] + OtherElementEndsInNextColumn[3,8] >= 2
 LinkOtherElementEndsInNextColumn3[3,9]: X1LessThan[3,9] = 1 ->
   EndColumn[9] - EndColumn[3] + OtherElementEndsInNextColumn[3,9] >= 2
 LinkOtherElementEndsInNextColumn3[3,10]: X1LessThan[3,10] = 1 ->
   EndColumn[10] - EndColumn[3] + OtherElementEndsInNextColumn[3,10] >= 2
 LinkOtherElementEndsInNextColumn3[4,0]: X1LessThan[4,0] = 1 ->
   EndColumn[0] - EndColumn[4] + OtherElementEndsInNextColumn[4,0] >= 2
 LinkOtherElementEndsInNextColumn3[4,1]: X1LessThan[4,1] = 1 ->
   EndColumn[1] - EndColumn[4] + OtherElementEndsInNextColumn[4,1] >= 2
 LinkOtherElementEndsInNextColumn3[4,2]: X1LessThan[4,2] = 1 ->
   EndColumn[2] - EndColumn[4] + OtherElementEndsInNextColumn[4,2] >= 2
 LinkOtherElementEndsInNextColumn3[4,3]: X1LessThan[4,3] = 1 ->
   EndColumn[3] - EndColumn[4] + OtherElementEndsInNextColumn[4,3] >= 2
 LinkOtherElementEndsInNextColumn3[4,5]: X1LessThan[4,5] = 1 ->
   EndColumn[5] - EndColumn[4] + OtherElementEndsInNextColumn[4,5] >= 2
 LinkOtherElementEndsInNextColumn3[4,6]: X1LessThan[4,6] = 1 ->
   EndColumn[6] - EndColumn[4] + OtherElementEndsInNextColumn[4,6] >= 2
 LinkOtherElementEndsInNextColumn3[4,7]: X1LessThan[4,7] = 1 ->
   EndColumn[7] - EndColumn[4] + OtherElementEndsInNextColumn[4,7] >= 2
 LinkOtherElementEndsInNextColumn3[4,8]: X1LessThan[4,8] = 1 ->
   EndColumn[8] - EndColumn[4] + OtherElementEndsInNextColumn[4,8] >= 2
 LinkOtherElementEndsInNextColumn3[4,9]: X1LessThan[4,9] = 1 ->
   EndColumn[9] - EndColumn[4] + OtherElementEndsInNextColumn[4,9] >= 2
 LinkOtherElementEndsInNextColumn3[4,10]: X1LessThan[4,10] = 1 ->
   EndColumn[10] - EndColumn[4] + OtherElementEndsInNextColumn[4,10] >= 2
 LinkOtherElementEndsInNextColumn3[5,0]: X1LessThan[5,0] = 1 ->
   EndColumn[0] - EndColumn[5] + OtherElementEndsInNextColumn[5,0] >= 2
 LinkOtherElementEndsInNextColumn3[5,1]: X1LessThan[5,1] = 1 ->
   EndColumn[1] - EndColumn[5] + OtherElementEndsInNextColumn[5,1] >= 2
 LinkOtherElementEndsInNextColumn3[5,2]: X1LessThan[5,2] = 1 ->
   EndColumn[2] - EndColumn[5] + OtherElementEndsInNextColumn[5,2] >= 2
 LinkOtherElementEndsInNextColumn3[5,3]: X1LessThan[5,3] = 1 ->
   EndColumn[3] - EndColumn[5] + OtherElementEndsInNextColumn[5,3] >= 2
 LinkOtherElementEndsInNextColumn3[5,4]: X1LessThan[5,4] = 1 ->
   EndColumn[4] - EndColumn[5] + OtherElementEndsInNextColumn[5,4] >= 2
 LinkOtherElementEndsInNextColumn3[5,6]: X1LessThan[5,6] = 1 ->
   EndColumn[6] - EndColumn[5] + OtherElementEndsInNextColumn[5,6] >= 2
 LinkOtherElementEndsInNextColumn3[5,7]: X1LessThan[5,7] = 1 ->
   EndColumn[7] - EndColumn[5] + OtherElementEndsInNextColumn[5,7] >= 2
 LinkOtherElementEndsInNextColumn3[5,8]: X1LessThan[5,8] = 1 ->
   EndColumn[8] - EndColumn[5] + OtherElementEndsInNextColumn[5,8] >= 2
 LinkOtherElementEndsInNextColumn3[5,9]: X1LessThan[5,9] = 1 ->
   EndColumn[9] - EndColumn[5] + OtherElementEndsInNextColumn[5,9] >= 2
 LinkOtherElementEndsInNextColumn3[5,10]: X1LessThan[5,10] = 1 ->
   EndColumn[10] - EndColumn[5] + OtherElementEndsInNextColumn[5,10] >= 2
 LinkOtherElementEndsInNextColumn3[6,0]: X1LessThan[6,0] = 1 ->
   EndColumn[0] - EndColumn[6] + OtherElementEndsInNextColumn[6,0] >= 2
 LinkOtherElementEndsInNextColumn3[6,1]: X1LessThan[6,1] = 1 ->
   EndColumn[1] - EndColumn[6] + OtherElementEndsInNextColumn[6,1] >= 2
 LinkOtherElementEndsInNextColumn3[6,2]: X1LessThan[6,2] = 1 ->
   EndColumn[2] - EndColumn[6] + OtherElementEndsInNextColumn[6,2] >= 2
 LinkOtherElementEndsInNextColumn3[6,3]: X1LessThan[6,3] = 1 ->
   EndColumn[3] - EndColumn[6] + OtherElementEndsInNextColumn[6,3] >= 2
 LinkOtherElementEndsInNextColumn3[6,4]: X1LessThan[6,4] = 1 ->
   EndColumn[4] - EndColumn[6] + OtherElementEndsInNextColumn[6,4] >= 2
 LinkOtherElementEndsInNextColumn3[6,5]: X1LessThan[6,5] = 1 ->
   EndColumn[5] - EndColumn[6] + OtherElementEndsInNextColumn[6,5] >= 2
 LinkOtherElementEndsInNextColumn3[6,7]: X1LessThan[6,7] = 1 ->
   EndColumn[7] - EndColumn[6] + OtherElementEndsInNextColumn[6,7] >= 2
 LinkOtherElementEndsInNextColumn3[6,8]: X1LessThan[6,8] = 1 ->
   EndColumn[8] - EndColumn[6] + OtherElementEndsInNextColumn[6,8] >= 2
 LinkOtherElementEndsInNextColumn3[6,9]: X1LessThan[6,9] = 1 ->
   EndColumn[9] - EndColumn[6] + OtherElementEndsInNextColumn[6,9] >= 2
 LinkOtherElementEndsInNextColumn3[6,10]: X1LessThan[6,10] = 1 ->
   EndColumn[10] - EndColumn[6] + OtherElementEndsInNextColumn[6,10] >= 2
 LinkOtherElementEndsInNextColumn3[7,0]: X1LessThan[7,0] = 1 ->
   EndColumn[0] - EndColumn[7] + OtherElementEndsInNextColumn[7,0] >= 2
 LinkOtherElementEndsInNextColumn3[7,1]: X1LessThan[7,1] = 1 ->
   EndColumn[1] - EndColumn[7] + OtherElementEndsInNextColumn[7,1] >= 2
 LinkOtherElementEndsInNextColumn3[7,2]: X1LessThan[7,2] = 1 ->
   EndColumn[2] - EndColumn[7] + OtherElementEndsInNextColumn[7,2] >= 2
 LinkOtherElementEndsInNextColumn3[7,3]: X1LessThan[7,3] = 1 ->
   EndColumn[3] - EndColumn[7] + OtherElementEndsInNextColumn[7,3] >= 2
 LinkOtherElementEndsInNextColumn3[7,4]: X1LessThan[7,4] = 1 ->
   EndColumn[4] - EndColumn[7] + OtherElementEndsInNextColumn[7,4] >= 2
 LinkOtherElementEndsInNextColumn3[7,5]: X1LessThan[7,5] = 1 ->
   EndColumn[5] - EndColumn[7] + OtherElementEndsInNextColumn[7,5] >= 2
 LinkOtherElementEndsInNextColumn3[7,6]: X1LessThan[7,6] = 1 ->
   EndColumn[6] - EndColumn[7] + OtherElementEndsInNextColumn[7,6] >= 2
 LinkOtherElementEndsInNextColumn3[7,8]: X1LessThan[7,8] = 1 ->
   EndColumn[8] - EndColumn[7] + OtherElementEndsInNextColumn[7,8] >= 2
 LinkOtherElementEndsInNextColumn3[7,9]: X1LessThan[7,9] = 1 ->
   EndColumn[9] - EndColumn[7] + OtherElementEndsInNextColumn[7,9] >= 2
 LinkOtherElementEndsInNextColumn3[7,10]: X1LessThan[7,10] = 1 ->
   EndColumn[10] - EndColumn[7] + OtherElementEndsInNextColumn[7,10] >= 2
 LinkOtherElementEndsInNextColumn3[8,0]: X1LessThan[8,0] = 1 ->
   EndColumn[0] - EndColumn[8] + OtherElementEndsInNextColumn[8,0] >= 2
 LinkOtherElementEndsInNextColumn3[8,1]: X1LessThan[8,1] = 1 ->
   EndColumn[1] - EndColumn[8] + OtherElementEndsInNextColumn[8,1] >= 2
 LinkOtherElementEndsInNextColumn3[8,2]: X1LessThan[8,2] = 1 ->
   EndColumn[2] - EndColumn[8] + OtherElementEndsInNextColumn[8,2] >= 2
 LinkOtherElementEndsInNextColumn3[8,3]: X1LessThan[8,3] = 1 ->
   EndColumn[3] - EndColumn[8] + OtherElementEndsInNextColumn[8,3] >= 2
 LinkOtherElementEndsInNextColumn3[8,4]: X1LessThan[8,4] = 1 ->
   EndColumn[4] - EndColumn[8] + OtherElementEndsInNextColumn[8,4] >= 2
 LinkOtherElementEndsInNextColumn3[8,5]: X1LessThan[8,5] = 1 ->
   EndColumn[5] - EndColumn[8] + OtherElementEndsInNextColumn[8,5] >= 2
 LinkOtherElementEndsInNextColumn3[8,6]: X1LessThan[8,6] = 1 ->
   EndColumn[6] - EndColumn[8] + OtherElementEndsInNextColumn[8,6] >= 2
 LinkOtherElementEndsInNextColumn3[8,7]: X1LessThan[8,7] = 1 ->
   EndColumn[7] - EndColumn[8] + OtherElementEndsInNextColumn[8,7] >= 2
 LinkOtherElementEndsInNextColumn3[8,9]: X1LessThan[8,9] = 1 ->
   EndColumn[9] - EndColumn[8] + OtherElementEndsInNextColumn[8,9] >= 2
 LinkOtherElementEndsInNextColumn3[8,10]: X1LessThan[8,10] = 1 ->
   EndColumn[10] - EndColumn[8] + OtherElementEndsInNextColumn[8,10] >= 2
 LinkOtherElementEndsInNextColumn3[9,0]: X1LessThan[9,0] = 1 ->
   EndColumn[0] - EndColumn[9] + OtherElementEndsInNextColumn[9,0] >= 2
 LinkOtherElementEndsInNextColumn3[9,1]: X1LessThan[9,1] = 1 ->
   EndColumn[1] - EndColumn[9] + OtherElementEndsInNextColumn[9,1] >= 2
 LinkOtherElementEndsInNextColumn3[9,2]: X1LessThan[9,2] = 1 ->
   EndColumn[2] - EndColumn[9] + OtherElementEndsInNextColumn[9,2] >= 2
 LinkOtherElementEndsInNextColumn3[9,3]: X1LessThan[9,3] = 1 ->
   EndColumn[3] - EndColumn[9] + OtherElementEndsInNextColumn[9,3] >= 2
 LinkOtherElementEndsInNextColumn3[9,4]: X1LessThan[9,4] = 1 ->
   EndColumn[4] - EndColumn[9] + OtherElementEndsInNextColumn[9,4] >= 2
 LinkOtherElementEndsInNextColumn3[9,5]: X1LessThan[9,5] = 1 ->
   EndColumn[5] - EndColumn[9] + OtherElementEndsInNextColumn[9,5] >= 2
 LinkOtherElementEndsInNextColumn3[9,6]: X1LessThan[9,6] = 1 ->
   EndColumn[6] - EndColumn[9] + OtherElementEndsInNextColumn[9,6] >= 2
 LinkOtherElementEndsInNextColumn3[9,7]: X1LessThan[9,7] = 1 ->
   EndColumn[7] - EndColumn[9] + OtherElementEndsInNextColumn[9,7] >= 2
 LinkOtherElementEndsInNextColumn3[9,8]: X1LessThan[9,8] = 1 ->
   EndColumn[8] - EndColumn[9] + OtherElementEndsInNextColumn[9,8] >= 2
 LinkOtherElementEndsInNextColumn3[9,10]: X1LessThan[9,10] = 1 ->
   EndColumn[10] - EndColumn[9] + OtherElementEndsInNextColumn[9,10] >= 2
 LinkOtherElementEndsInNextColumn3[10,0]: X1LessThan[10,0] = 1 ->
   EndColumn[0] - EndColumn[10] + OtherElementEndsInNextColumn[10,0] >= 2
 LinkOtherElementEndsInNextColumn3[10,1]: X1LessThan[10,1] = 1 ->
   EndColumn[1] - EndColumn[10] + OtherElementEndsInNextColumn[10,1] >= 2
 LinkOtherElementEndsInNextColumn3[10,2]: X1LessThan[10,2] = 1 ->
   EndColumn[2] - EndColumn[10] + OtherElementEndsInNextColumn[10,2] >= 2
 LinkOtherElementEndsInNextColumn3[10,3]: X1LessThan[10,3] = 1 ->
   EndColumn[3] - EndColumn[10] + OtherElementEndsInNextColumn[10,3] >= 2
 LinkOtherElementEndsInNextColumn3[10,4]: X1LessThan[10,4] = 1 ->
   EndColumn[4] - EndColumn[10] + OtherElementEndsInNextColumn[10,4] >= 2
 LinkOtherElementEndsInNextColumn3[10,5]: X1LessThan[10,5] = 1 ->
   EndColumn[5] - EndColumn[10] + OtherElementEndsInNextColumn[10,5] >= 2
 LinkOtherElementEndsInNextColumn3[10,6]: X1LessThan[10,6] = 1 ->
   EndColumn[6] - EndColumn[10] + OtherElementEndsInNextColumn[10,6] >= 2
 LinkOtherElementEndsInNextColumn3[10,7]: X1LessThan[10,7] = 1 ->
   EndColumn[7] - EndColumn[10] + OtherElementEndsInNextColumn[10,7] >= 2
 LinkOtherElementEndsInNextColumn3[10,8]: X1LessThan[10,8] = 1 ->
   EndColumn[8] - EndColumn[10] + OtherElementEndsInNextColumn[10,8] >= 2
 LinkOtherElementEndsInNextColumn3[10,9]: X1LessThan[10,9] = 1 ->
   EndColumn[9] - EndColumn[10] + OtherElementEndsInNextColumn[10,9] >= 2
 EnforceColumnWidthOnAdjacentColumns1[0,0]: OtherElementStartsInNextColumn[0,0] = 1 ->
   X0Diff[0,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,1]: OtherElementStartsInNextColumn[0,1] = 1 ->
   X0Diff[1,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,2]: OtherElementStartsInNextColumn[0,2] = 1 ->
   X0Diff[2,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,3]: OtherElementStartsInNextColumn[0,3] = 1 ->
   X0Diff[3,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,4]: OtherElementStartsInNextColumn[0,4] = 1 ->
   X0Diff[4,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,5]: OtherElementStartsInNextColumn[0,5] = 1 ->
   X0Diff[5,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,6]: OtherElementStartsInNextColumn[0,6] = 1 ->
   X0Diff[6,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,7]: OtherElementStartsInNextColumn[0,7] = 1 ->
   X0Diff[7,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,8]: OtherElementStartsInNextColumn[0,8] = 1 ->
   X0Diff[8,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,9]: OtherElementStartsInNextColumn[0,9] = 1 ->
   X0Diff[9,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[0,10]: OtherElementStartsInNextColumn[0,10] = 1 ->
   X0Diff[10,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,0]: OtherElementStartsInNextColumn[1,0] = 1 ->
   X0Diff[0,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,1]: OtherElementStartsInNextColumn[1,1] = 1 ->
   X0Diff[1,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,2]: OtherElementStartsInNextColumn[1,2] = 1 ->
   X0Diff[2,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,3]: OtherElementStartsInNextColumn[1,3] = 1 ->
   X0Diff[3,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,4]: OtherElementStartsInNextColumn[1,4] = 1 ->
   X0Diff[4,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,5]: OtherElementStartsInNextColumn[1,5] = 1 ->
   X0Diff[5,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,6]: OtherElementStartsInNextColumn[1,6] = 1 ->
   X0Diff[6,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,7]: OtherElementStartsInNextColumn[1,7] = 1 ->
   X0Diff[7,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,8]: OtherElementStartsInNextColumn[1,8] = 1 ->
   X0Diff[8,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,9]: OtherElementStartsInNextColumn[1,9] = 1 ->
   X0Diff[9,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[1,10]: OtherElementStartsInNextColumn[1,10] = 1 ->
   X0Diff[10,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,0]: OtherElementStartsInNextColumn[2,0] = 1 ->
   X0Diff[0,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,1]: OtherElementStartsInNextColumn[2,1] = 1 ->
   X0Diff[1,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,2]: OtherElementStartsInNextColumn[2,2] = 1 ->
   X0Diff[2,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,3]: OtherElementStartsInNextColumn[2,3] = 1 ->
   X0Diff[3,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,4]: OtherElementStartsInNextColumn[2,4] = 1 ->
   X0Diff[4,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,5]: OtherElementStartsInNextColumn[2,5] = 1 ->
   X0Diff[5,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,6]: OtherElementStartsInNextColumn[2,6] = 1 ->
   X0Diff[6,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,7]: OtherElementStartsInNextColumn[2,7] = 1 ->
   X0Diff[7,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,8]: OtherElementStartsInNextColumn[2,8] = 1 ->
   X0Diff[8,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,9]: OtherElementStartsInNextColumn[2,9] = 1 ->
   X0Diff[9,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[2,10]: OtherElementStartsInNextColumn[2,10] = 1 ->
   X0Diff[10,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,0]: OtherElementStartsInNextColumn[3,0] = 1 ->
   X0Diff[0,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,1]: OtherElementStartsInNextColumn[3,1] = 1 ->
   X0Diff[1,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,2]: OtherElementStartsInNextColumn[3,2] = 1 ->
   X0Diff[2,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,3]: OtherElementStartsInNextColumn[3,3] = 1 ->
   X0Diff[3,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,4]: OtherElementStartsInNextColumn[3,4] = 1 ->
   X0Diff[4,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,5]: OtherElementStartsInNextColumn[3,5] = 1 ->
   X0Diff[5,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,6]: OtherElementStartsInNextColumn[3,6] = 1 ->
   X0Diff[6,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,7]: OtherElementStartsInNextColumn[3,7] = 1 ->
   X0Diff[7,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,8]: OtherElementStartsInNextColumn[3,8] = 1 ->
   X0Diff[8,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,9]: OtherElementStartsInNextColumn[3,9] = 1 ->
   X0Diff[9,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[3,10]: OtherElementStartsInNextColumn[3,10] = 1 ->
   X0Diff[10,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,0]: OtherElementStartsInNextColumn[4,0] = 1 ->
   X0Diff[0,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,1]: OtherElementStartsInNextColumn[4,1] = 1 ->
   X0Diff[1,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,2]: OtherElementStartsInNextColumn[4,2] = 1 ->
   X0Diff[2,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,3]: OtherElementStartsInNextColumn[4,3] = 1 ->
   X0Diff[3,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,4]: OtherElementStartsInNextColumn[4,4] = 1 ->
   X0Diff[4,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,5]: OtherElementStartsInNextColumn[4,5] = 1 ->
   X0Diff[5,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,6]: OtherElementStartsInNextColumn[4,6] = 1 ->
   X0Diff[6,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,7]: OtherElementStartsInNextColumn[4,7] = 1 ->
   X0Diff[7,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,8]: OtherElementStartsInNextColumn[4,8] = 1 ->
   X0Diff[8,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,9]: OtherElementStartsInNextColumn[4,9] = 1 ->
   X0Diff[9,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[4,10]: OtherElementStartsInNextColumn[4,10] = 1 ->
   X0Diff[10,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,0]: OtherElementStartsInNextColumn[5,0] = 1 ->
   X0Diff[0,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,1]: OtherElementStartsInNextColumn[5,1] = 1 ->
   X0Diff[1,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,2]: OtherElementStartsInNextColumn[5,2] = 1 ->
   X0Diff[2,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,3]: OtherElementStartsInNextColumn[5,3] = 1 ->
   X0Diff[3,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,4]: OtherElementStartsInNextColumn[5,4] = 1 ->
   X0Diff[4,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,5]: OtherElementStartsInNextColumn[5,5] = 1 ->
   X0Diff[5,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,6]: OtherElementStartsInNextColumn[5,6] = 1 ->
   X0Diff[6,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,7]: OtherElementStartsInNextColumn[5,7] = 1 ->
   X0Diff[7,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,8]: OtherElementStartsInNextColumn[5,8] = 1 ->
   X0Diff[8,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,9]: OtherElementStartsInNextColumn[5,9] = 1 ->
   X0Diff[9,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[5,10]: OtherElementStartsInNextColumn[5,10] = 1 ->
   X0Diff[10,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,0]: OtherElementStartsInNextColumn[6,0] = 1 ->
   X0Diff[0,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,1]: OtherElementStartsInNextColumn[6,1] = 1 ->
   X0Diff[1,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,2]: OtherElementStartsInNextColumn[6,2] = 1 ->
   X0Diff[2,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,3]: OtherElementStartsInNextColumn[6,3] = 1 ->
   X0Diff[3,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,4]: OtherElementStartsInNextColumn[6,4] = 1 ->
   X0Diff[4,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,5]: OtherElementStartsInNextColumn[6,5] = 1 ->
   X0Diff[5,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,6]: OtherElementStartsInNextColumn[6,6] = 1 ->
   X0Diff[6,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,7]: OtherElementStartsInNextColumn[6,7] = 1 ->
   X0Diff[7,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,8]: OtherElementStartsInNextColumn[6,8] = 1 ->
   X0Diff[8,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,9]: OtherElementStartsInNextColumn[6,9] = 1 ->
   X0Diff[9,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[6,10]: OtherElementStartsInNextColumn[6,10] = 1 ->
   X0Diff[10,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,0]: OtherElementStartsInNextColumn[7,0] = 1 ->
   X0Diff[0,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,1]: OtherElementStartsInNextColumn[7,1] = 1 ->
   X0Diff[1,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,2]: OtherElementStartsInNextColumn[7,2] = 1 ->
   X0Diff[2,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,3]: OtherElementStartsInNextColumn[7,3] = 1 ->
   X0Diff[3,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,4]: OtherElementStartsInNextColumn[7,4] = 1 ->
   X0Diff[4,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,5]: OtherElementStartsInNextColumn[7,5] = 1 ->
   X0Diff[5,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,6]: OtherElementStartsInNextColumn[7,6] = 1 ->
   X0Diff[6,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,7]: OtherElementStartsInNextColumn[7,7] = 1 ->
   X0Diff[7,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,8]: OtherElementStartsInNextColumn[7,8] = 1 ->
   X0Diff[8,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,9]: OtherElementStartsInNextColumn[7,9] = 1 ->
   X0Diff[9,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[7,10]: OtherElementStartsInNextColumn[7,10] = 1 ->
   X0Diff[10,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,0]: OtherElementStartsInNextColumn[8,0] = 1 ->
   X0Diff[0,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,1]: OtherElementStartsInNextColumn[8,1] = 1 ->
   X0Diff[1,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,2]: OtherElementStartsInNextColumn[8,2] = 1 ->
   X0Diff[2,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,3]: OtherElementStartsInNextColumn[8,3] = 1 ->
   X0Diff[3,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,4]: OtherElementStartsInNextColumn[8,4] = 1 ->
   X0Diff[4,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,5]: OtherElementStartsInNextColumn[8,5] = 1 ->
   X0Diff[5,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,6]: OtherElementStartsInNextColumn[8,6] = 1 ->
   X0Diff[6,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,7]: OtherElementStartsInNextColumn[8,7] = 1 ->
   X0Diff[7,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,8]: OtherElementStartsInNextColumn[8,8] = 1 ->
   X0Diff[8,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,9]: OtherElementStartsInNextColumn[8,9] = 1 ->
   X0Diff[9,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[8,10]: OtherElementStartsInNextColumn[8,10] = 1 ->
   X0Diff[10,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,0]: OtherElementStartsInNextColumn[9,0] = 1 ->
   X0Diff[0,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,1]: OtherElementStartsInNextColumn[9,1] = 1 ->
   X0Diff[1,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,2]: OtherElementStartsInNextColumn[9,2] = 1 ->
   X0Diff[2,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,3]: OtherElementStartsInNextColumn[9,3] = 1 ->
   X0Diff[3,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,4]: OtherElementStartsInNextColumn[9,4] = 1 ->
   X0Diff[4,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,5]: OtherElementStartsInNextColumn[9,5] = 1 ->
   X0Diff[5,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,6]: OtherElementStartsInNextColumn[9,6] = 1 ->
   X0Diff[6,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,7]: OtherElementStartsInNextColumn[9,7] = 1 ->
   X0Diff[7,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,8]: OtherElementStartsInNextColumn[9,8] = 1 ->
   X0Diff[8,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,9]: OtherElementStartsInNextColumn[9,9] = 1 ->
   X0Diff[9,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[9,10]: OtherElementStartsInNextColumn[9,10] = 1 ->
   X0Diff[10,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,0]: OtherElementStartsInNextColumn[10,0] = 1 ->
   X0Diff[0,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,1]: OtherElementStartsInNextColumn[10,1] = 1 ->
   X0Diff[1,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,2]: OtherElementStartsInNextColumn[10,2] = 1 ->
   X0Diff[2,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,3]: OtherElementStartsInNextColumn[10,3] = 1 ->
   X0Diff[3,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,4]: OtherElementStartsInNextColumn[10,4] = 1 ->
   X0Diff[4,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,5]: OtherElementStartsInNextColumn[10,5] = 1 ->
   X0Diff[5,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,6]: OtherElementStartsInNextColumn[10,6] = 1 ->
   X0Diff[6,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,7]: OtherElementStartsInNextColumn[10,7] = 1 ->
   X0Diff[7,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,8]: OtherElementStartsInNextColumn[10,8] = 1 ->
   X0Diff[8,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,9]: OtherElementStartsInNextColumn[10,9] = 1 ->
   X0Diff[9,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns1[10,10]: OtherElementStartsInNextColumn[10,10] = 1 ->
   X0Diff[10,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,0]: OtherElementEndsInNextColumn[0,0] = 1 ->
   X1Diff[0,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,1]: OtherElementEndsInNextColumn[0,1] = 1 ->
   X1Diff[1,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,2]: OtherElementEndsInNextColumn[0,2] = 1 ->
   X1Diff[2,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,3]: OtherElementEndsInNextColumn[0,3] = 1 ->
   X1Diff[3,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,4]: OtherElementEndsInNextColumn[0,4] = 1 ->
   X1Diff[4,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,5]: OtherElementEndsInNextColumn[0,5] = 1 ->
   X1Diff[5,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,6]: OtherElementEndsInNextColumn[0,6] = 1 ->
   X1Diff[6,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,7]: OtherElementEndsInNextColumn[0,7] = 1 ->
   X1Diff[7,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,8]: OtherElementEndsInNextColumn[0,8] = 1 ->
   X1Diff[8,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,9]: OtherElementEndsInNextColumn[0,9] = 1 ->
   X1Diff[9,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[0,10]: OtherElementEndsInNextColumn[0,10] = 1 ->
   X1Diff[10,0] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,0]: OtherElementEndsInNextColumn[1,0] = 1 ->
   X1Diff[0,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,1]: OtherElementEndsInNextColumn[1,1] = 1 ->
   X1Diff[1,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,2]: OtherElementEndsInNextColumn[1,2] = 1 ->
   X1Diff[2,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,3]: OtherElementEndsInNextColumn[1,3] = 1 ->
   X1Diff[3,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,4]: OtherElementEndsInNextColumn[1,4] = 1 ->
   X1Diff[4,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,5]: OtherElementEndsInNextColumn[1,5] = 1 ->
   X1Diff[5,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,6]: OtherElementEndsInNextColumn[1,6] = 1 ->
   X1Diff[6,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,7]: OtherElementEndsInNextColumn[1,7] = 1 ->
   X1Diff[7,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,8]: OtherElementEndsInNextColumn[1,8] = 1 ->
   X1Diff[8,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,9]: OtherElementEndsInNextColumn[1,9] = 1 ->
   X1Diff[9,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[1,10]: OtherElementEndsInNextColumn[1,10] = 1 ->
   X1Diff[10,1] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,0]: OtherElementEndsInNextColumn[2,0] = 1 ->
   X1Diff[0,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,1]: OtherElementEndsInNextColumn[2,1] = 1 ->
   X1Diff[1,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,2]: OtherElementEndsInNextColumn[2,2] = 1 ->
   X1Diff[2,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,3]: OtherElementEndsInNextColumn[2,3] = 1 ->
   X1Diff[3,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,4]: OtherElementEndsInNextColumn[2,4] = 1 ->
   X1Diff[4,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,5]: OtherElementEndsInNextColumn[2,5] = 1 ->
   X1Diff[5,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,6]: OtherElementEndsInNextColumn[2,6] = 1 ->
   X1Diff[6,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,7]: OtherElementEndsInNextColumn[2,7] = 1 ->
   X1Diff[7,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,8]: OtherElementEndsInNextColumn[2,8] = 1 ->
   X1Diff[8,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,9]: OtherElementEndsInNextColumn[2,9] = 1 ->
   X1Diff[9,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[2,10]: OtherElementEndsInNextColumn[2,10] = 1 ->
   X1Diff[10,2] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,0]: OtherElementEndsInNextColumn[3,0] = 1 ->
   X1Diff[0,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,1]: OtherElementEndsInNextColumn[3,1] = 1 ->
   X1Diff[1,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,2]: OtherElementEndsInNextColumn[3,2] = 1 ->
   X1Diff[2,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,3]: OtherElementEndsInNextColumn[3,3] = 1 ->
   X1Diff[3,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,4]: OtherElementEndsInNextColumn[3,4] = 1 ->
   X1Diff[4,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,5]: OtherElementEndsInNextColumn[3,5] = 1 ->
   X1Diff[5,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,6]: OtherElementEndsInNextColumn[3,6] = 1 ->
   X1Diff[6,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,7]: OtherElementEndsInNextColumn[3,7] = 1 ->
   X1Diff[7,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,8]: OtherElementEndsInNextColumn[3,8] = 1 ->
   X1Diff[8,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,9]: OtherElementEndsInNextColumn[3,9] = 1 ->
   X1Diff[9,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[3,10]: OtherElementEndsInNextColumn[3,10] = 1 ->
   X1Diff[10,3] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,0]: OtherElementEndsInNextColumn[4,0] = 1 ->
   X1Diff[0,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,1]: OtherElementEndsInNextColumn[4,1] = 1 ->
   X1Diff[1,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,2]: OtherElementEndsInNextColumn[4,2] = 1 ->
   X1Diff[2,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,3]: OtherElementEndsInNextColumn[4,3] = 1 ->
   X1Diff[3,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,4]: OtherElementEndsInNextColumn[4,4] = 1 ->
   X1Diff[4,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,5]: OtherElementEndsInNextColumn[4,5] = 1 ->
   X1Diff[5,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,6]: OtherElementEndsInNextColumn[4,6] = 1 ->
   X1Diff[6,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,7]: OtherElementEndsInNextColumn[4,7] = 1 ->
   X1Diff[7,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,8]: OtherElementEndsInNextColumn[4,8] = 1 ->
   X1Diff[8,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,9]: OtherElementEndsInNextColumn[4,9] = 1 ->
   X1Diff[9,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[4,10]: OtherElementEndsInNextColumn[4,10] = 1 ->
   X1Diff[10,4] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,0]: OtherElementEndsInNextColumn[5,0] = 1 ->
   X1Diff[0,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,1]: OtherElementEndsInNextColumn[5,1] = 1 ->
   X1Diff[1,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,2]: OtherElementEndsInNextColumn[5,2] = 1 ->
   X1Diff[2,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,3]: OtherElementEndsInNextColumn[5,3] = 1 ->
   X1Diff[3,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,4]: OtherElementEndsInNextColumn[5,4] = 1 ->
   X1Diff[4,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,5]: OtherElementEndsInNextColumn[5,5] = 1 ->
   X1Diff[5,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,6]: OtherElementEndsInNextColumn[5,6] = 1 ->
   X1Diff[6,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,7]: OtherElementEndsInNextColumn[5,7] = 1 ->
   X1Diff[7,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,8]: OtherElementEndsInNextColumn[5,8] = 1 ->
   X1Diff[8,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,9]: OtherElementEndsInNextColumn[5,9] = 1 ->
   X1Diff[9,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[5,10]: OtherElementEndsInNextColumn[5,10] = 1 ->
   X1Diff[10,5] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,0]: OtherElementEndsInNextColumn[6,0] = 1 ->
   X1Diff[0,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,1]: OtherElementEndsInNextColumn[6,1] = 1 ->
   X1Diff[1,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,2]: OtherElementEndsInNextColumn[6,2] = 1 ->
   X1Diff[2,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,3]: OtherElementEndsInNextColumn[6,3] = 1 ->
   X1Diff[3,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,4]: OtherElementEndsInNextColumn[6,4] = 1 ->
   X1Diff[4,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,5]: OtherElementEndsInNextColumn[6,5] = 1 ->
   X1Diff[5,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,6]: OtherElementEndsInNextColumn[6,6] = 1 ->
   X1Diff[6,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,7]: OtherElementEndsInNextColumn[6,7] = 1 ->
   X1Diff[7,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,8]: OtherElementEndsInNextColumn[6,8] = 1 ->
   X1Diff[8,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,9]: OtherElementEndsInNextColumn[6,9] = 1 ->
   X1Diff[9,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[6,10]: OtherElementEndsInNextColumn[6,10] = 1 ->
   X1Diff[10,6] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,0]: OtherElementEndsInNextColumn[7,0] = 1 ->
   X1Diff[0,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,1]: OtherElementEndsInNextColumn[7,1] = 1 ->
   X1Diff[1,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,2]: OtherElementEndsInNextColumn[7,2] = 1 ->
   X1Diff[2,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,3]: OtherElementEndsInNextColumn[7,3] = 1 ->
   X1Diff[3,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,4]: OtherElementEndsInNextColumn[7,4] = 1 ->
   X1Diff[4,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,5]: OtherElementEndsInNextColumn[7,5] = 1 ->
   X1Diff[5,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,6]: OtherElementEndsInNextColumn[7,6] = 1 ->
   X1Diff[6,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,7]: OtherElementEndsInNextColumn[7,7] = 1 ->
   X1Diff[7,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,8]: OtherElementEndsInNextColumn[7,8] = 1 ->
   X1Diff[8,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,9]: OtherElementEndsInNextColumn[7,9] = 1 ->
   X1Diff[9,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[7,10]: OtherElementEndsInNextColumn[7,10] = 1 ->
   X1Diff[10,7] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,0]: OtherElementEndsInNextColumn[8,0] = 1 ->
   X1Diff[0,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,1]: OtherElementEndsInNextColumn[8,1] = 1 ->
   X1Diff[1,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,2]: OtherElementEndsInNextColumn[8,2] = 1 ->
   X1Diff[2,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,3]: OtherElementEndsInNextColumn[8,3] = 1 ->
   X1Diff[3,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,4]: OtherElementEndsInNextColumn[8,4] = 1 ->
   X1Diff[4,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,5]: OtherElementEndsInNextColumn[8,5] = 1 ->
   X1Diff[5,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,6]: OtherElementEndsInNextColumn[8,6] = 1 ->
   X1Diff[6,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,7]: OtherElementEndsInNextColumn[8,7] = 1 ->
   X1Diff[7,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,8]: OtherElementEndsInNextColumn[8,8] = 1 ->
   X1Diff[8,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,9]: OtherElementEndsInNextColumn[8,9] = 1 ->
   X1Diff[9,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[8,10]: OtherElementEndsInNextColumn[8,10] = 1 ->
   X1Diff[10,8] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,0]: OtherElementEndsInNextColumn[9,0] = 1 ->
   X1Diff[0,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,1]: OtherElementEndsInNextColumn[9,1] = 1 ->
   X1Diff[1,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,2]: OtherElementEndsInNextColumn[9,2] = 1 ->
   X1Diff[2,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,3]: OtherElementEndsInNextColumn[9,3] = 1 ->
   X1Diff[3,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,4]: OtherElementEndsInNextColumn[9,4] = 1 ->
   X1Diff[4,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,5]: OtherElementEndsInNextColumn[9,5] = 1 ->
   X1Diff[5,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,6]: OtherElementEndsInNextColumn[9,6] = 1 ->
   X1Diff[6,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,7]: OtherElementEndsInNextColumn[9,7] = 1 ->
   X1Diff[7,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,8]: OtherElementEndsInNextColumn[9,8] = 1 ->
   X1Diff[8,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,9]: OtherElementEndsInNextColumn[9,9] = 1 ->
   X1Diff[9,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[9,10]: OtherElementEndsInNextColumn[9,10] = 1 ->
   X1Diff[10,9] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,0]: OtherElementEndsInNextColumn[10,0] = 1 ->
   X1Diff[0,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,1]: OtherElementEndsInNextColumn[10,1] = 1 ->
   X1Diff[1,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,2]: OtherElementEndsInNextColumn[10,2] = 1 ->
   X1Diff[2,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,3]: OtherElementEndsInNextColumn[10,3] = 1 ->
   X1Diff[3,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,4]: OtherElementEndsInNextColumn[10,4] = 1 ->
   X1Diff[4,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,5]: OtherElementEndsInNextColumn[10,5] = 1 ->
   X1Diff[5,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,6]: OtherElementEndsInNextColumn[10,6] = 1 ->
   X1Diff[6,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,7]: OtherElementEndsInNextColumn[10,7] = 1 ->
   X1Diff[7,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,8]: OtherElementEndsInNextColumn[10,8] = 1 ->
   X1Diff[8,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,9]: OtherElementEndsInNextColumn[10,9] = 1 ->
   X1Diff[9,10] - C2252 = -0
 EnforceColumnWidthOnAdjacentColumns2[10,10]: OtherElementEndsInNextColumn[10,10] = 1 ->
   X1Diff[10,10] - C2252 = -0
Bounds
 W[0] >= 1
 W[1] >= 1
 W[2] >= 1
 W[3] >= 1
 W[4] >= 1
 W[5] >= 1
 W[6] >= 1
 W[7] >= 1
 W[8] >= 1
 W[9] >= 1
 W[10] >= 1
 H[0] >= 1
 H[1] >= 1
 H[2] >= 1
 H[3] >= 1
 H[4] >= 1
 H[5] >= 1
 H[6] >= 1
 H[7] >= 1
 H[8] >= 1
 H[9] >= 1
 H[10] >= 1
 ResizeW[0] free
 ResizeW[1] free
 ResizeW[2] free
 ResizeW[3] free
 ResizeW[4] free
 ResizeW[5] free
 ResizeW[6] free
 ResizeW[7] free
 ResizeW[8] free
 ResizeW[9] free
 ResizeW[10] free
 ResizeH[0] free
 ResizeH[1] free
 ResizeH[2] free
 ResizeH[3] free
 ResizeH[4] free
 ResizeH[5] free
 ResizeH[6] free
 ResizeH[7] free
 ResizeH[8] free
 ResizeH[9] free
 ResizeH[10] free
 MoveX[0] free
 MoveX[1] free
 MoveX[2] free
 MoveX[3] free
 MoveX[4] free
 MoveX[5] free
 MoveX[6] free
 MoveX[7] free
 MoveX[8] free
 MoveX[9] free
 MoveX[10] free
 MoveY[0] free
 MoveY[1] free
 MoveY[2] free
 MoveY[3] free
 MoveY[4] free
 MoveY[5] free
 MoveY[6] free
 MoveY[7] free
 MoveY[8] free
 MoveY[9] free
 MoveY[10] free
 EdgeDistance[x0,0,0] free
 EdgeDistance[x0,0,1] free
 EdgeDistance[x0,0,2] free
 EdgeDistance[x0,0,3] free
 EdgeDistance[x0,0,4] free
 EdgeDistance[x0,0,5] free
 EdgeDistance[x0,0,6] free
 EdgeDistance[x0,0,7] free
 EdgeDistance[x0,0,8] free
 EdgeDistance[x0,0,9] free
 EdgeDistance[x0,0,10] free
 EdgeDistance[x0,1,0] free
 EdgeDistance[x0,1,1] free
 EdgeDistance[x0,1,2] free
 EdgeDistance[x0,1,3] free
 EdgeDistance[x0,1,4] free
 EdgeDistance[x0,1,5] free
 EdgeDistance[x0,1,6] free
 EdgeDistance[x0,1,7] free
 EdgeDistance[x0,1,8] free
 EdgeDistance[x0,1,9] free
 EdgeDistance[x0,1,10] free
 EdgeDistance[x0,2,0] free
 EdgeDistance[x0,2,1] free
 EdgeDistance[x0,2,2] free
 EdgeDistance[x0,2,3] free
 EdgeDistance[x0,2,4] free
 EdgeDistance[x0,2,5] free
 EdgeDistance[x0,2,6] free
 EdgeDistance[x0,2,7] free
 EdgeDistance[x0,2,8] free
 EdgeDistance[x0,2,9] free
 EdgeDistance[x0,2,10] free
 EdgeDistance[x0,3,0] free
 EdgeDistance[x0,3,1] free
 EdgeDistance[x0,3,2] free
 EdgeDistance[x0,3,3] free
 EdgeDistance[x0,3,4] free
 EdgeDistance[x0,3,5] free
 EdgeDistance[x0,3,6] free
 EdgeDistance[x0,3,7] free
 EdgeDistance[x0,3,8] free
 EdgeDistance[x0,3,9] free
 EdgeDistance[x0,3,10] free
 EdgeDistance[x0,4,0] free
 EdgeDistance[x0,4,1] free
 EdgeDistance[x0,4,2] free
 EdgeDistance[x0,4,3] free
 EdgeDistance[x0,4,4] free
 EdgeDistance[x0,4,5] free
 EdgeDistance[x0,4,6] free
 EdgeDistance[x0,4,7] free
 EdgeDistance[x0,4,8] free
 EdgeDistance[x0,4,9] free
 EdgeDistance[x0,4,10] free
 EdgeDistance[x0,5,0] free
 EdgeDistance[x0,5,1] free
 EdgeDistance[x0,5,2] free
 EdgeDistance[x0,5,3] free
 EdgeDistance[x0,5,4] free
 EdgeDistance[x0,5,5] free
 EdgeDistance[x0,5,6] free
 EdgeDistance[x0,5,7] free
 EdgeDistance[x0,5,8] free
 EdgeDistance[x0,5,9] free
 EdgeDistance[x0,5,10] free
 EdgeDistance[x0,6,0] free
 EdgeDistance[x0,6,1] free
 EdgeDistance[x0,6,2] free
 EdgeDistance[x0,6,3] free
 EdgeDistance[x0,6,4] free
 EdgeDistance[x0,6,5] free
 EdgeDistance[x0,6,6] free
 EdgeDistance[x0,6,7] free
 EdgeDistance[x0,6,8] free
 EdgeDistance[x0,6,9] free
 EdgeDistance[x0,6,10] free
 EdgeDistance[x0,7,0] free
 EdgeDistance[x0,7,1] free
 EdgeDistance[x0,7,2] free
 EdgeDistance[x0,7,3] free
 EdgeDistance[x0,7,4] free
 EdgeDistance[x0,7,5] free
 EdgeDistance[x0,7,6] free
 EdgeDistance[x0,7,7] free
 EdgeDistance[x0,7,8] free
 EdgeDistance[x0,7,9] free
 EdgeDistance[x0,7,10] free
 EdgeDistance[x0,8,0] free
 EdgeDistance[x0,8,1] free
 EdgeDistance[x0,8,2] free
 EdgeDistance[x0,8,3] free
 EdgeDistance[x0,8,4] free
 EdgeDistance[x0,8,5] free
 EdgeDistance[x0,8,6] free
 EdgeDistance[x0,8,7] free
 EdgeDistance[x0,8,8] free
 EdgeDistance[x0,8,9] free
 EdgeDistance[x0,8,10] free
 EdgeDistance[x0,9,0] free
 EdgeDistance[x0,9,1] free
 EdgeDistance[x0,9,2] free
 EdgeDistance[x0,9,3] free
 EdgeDistance[x0,9,4] free
 EdgeDistance[x0,9,5] free
 EdgeDistance[x0,9,6] free
 EdgeDistance[x0,9,7] free
 EdgeDistance[x0,9,8] free
 EdgeDistance[x0,9,9] free
 EdgeDistance[x0,9,10] free
 EdgeDistance[x0,10,0] free
 EdgeDistance[x0,10,1] free
 EdgeDistance[x0,10,2] free
 EdgeDistance[x0,10,3] free
 EdgeDistance[x0,10,4] free
 EdgeDistance[x0,10,5] free
 EdgeDistance[x0,10,6] free
 EdgeDistance[x0,10,7] free
 EdgeDistance[x0,10,8] free
 EdgeDistance[x0,10,9] free
 EdgeDistance[x0,10,10] free
 EdgeDistance[y0,0,0] free
 EdgeDistance[y0,0,1] free
 EdgeDistance[y0,0,2] free
 EdgeDistance[y0,0,3] free
 EdgeDistance[y0,0,4] free
 EdgeDistance[y0,0,5] free
 EdgeDistance[y0,0,6] free
 EdgeDistance[y0,0,7] free
 EdgeDistance[y0,0,8] free
 EdgeDistance[y0,0,9] free
 EdgeDistance[y0,0,10] free
 EdgeDistance[y0,1,0] free
 EdgeDistance[y0,1,1] free
 EdgeDistance[y0,1,2] free
 EdgeDistance[y0,1,3] free
 EdgeDistance[y0,1,4] free
 EdgeDistance[y0,1,5] free
 EdgeDistance[y0,1,6] free
 EdgeDistance[y0,1,7] free
 EdgeDistance[y0,1,8] free
 EdgeDistance[y0,1,9] free
 EdgeDistance[y0,1,10] free
 EdgeDistance[y0,2,0] free
 EdgeDistance[y0,2,1] free
 EdgeDistance[y0,2,2] free
 EdgeDistance[y0,2,3] free
 EdgeDistance[y0,2,4] free
 EdgeDistance[y0,2,5] free
 EdgeDistance[y0,2,6] free
 EdgeDistance[y0,2,7] free
 EdgeDistance[y0,2,8] free
 EdgeDistance[y0,2,9] free
 EdgeDistance[y0,2,10] free
 EdgeDistance[y0,3,0] free
 EdgeDistance[y0,3,1] free
 EdgeDistance[y0,3,2] free
 EdgeDistance[y0,3,3] free
 EdgeDistance[y0,3,4] free
 EdgeDistance[y0,3,5] free
 EdgeDistance[y0,3,6] free
 EdgeDistance[y0,3,7] free
 EdgeDistance[y0,3,8] free
 EdgeDistance[y0,3,9] free
 EdgeDistance[y0,3,10] free
 EdgeDistance[y0,4,0] free
 EdgeDistance[y0,4,1] free
 EdgeDistance[y0,4,2] free
 EdgeDistance[y0,4,3] free
 EdgeDistance[y0,4,4] free
 EdgeDistance[y0,4,5] free
 EdgeDistance[y0,4,6] free
 EdgeDistance[y0,4,7] free
 EdgeDistance[y0,4,8] free
 EdgeDistance[y0,4,9] free
 EdgeDistance[y0,4,10] free
 EdgeDistance[y0,5,0] free
 EdgeDistance[y0,5,1] free
 EdgeDistance[y0,5,2] free
 EdgeDistance[y0,5,3] free
 EdgeDistance[y0,5,4] free
 EdgeDistance[y0,5,5] free
 EdgeDistance[y0,5,6] free
 EdgeDistance[y0,5,7] free
 EdgeDistance[y0,5,8] free
 EdgeDistance[y0,5,9] free
 EdgeDistance[y0,5,10] free
 EdgeDistance[y0,6,0] free
 EdgeDistance[y0,6,1] free
 EdgeDistance[y0,6,2] free
 EdgeDistance[y0,6,3] free
 EdgeDistance[y0,6,4] free
 EdgeDistance[y0,6,5] free
 EdgeDistance[y0,6,6] free
 EdgeDistance[y0,6,7] free
 EdgeDistance[y0,6,8] free
 EdgeDistance[y0,6,9] free
 EdgeDistance[y0,6,10] free
 EdgeDistance[y0,7,0] free
 EdgeDistance[y0,7,1] free
 EdgeDistance[y0,7,2] free
 EdgeDistance[y0,7,3] free
 EdgeDistance[y0,7,4] free
 EdgeDistance[y0,7,5] free
 EdgeDistance[y0,7,6] free
 EdgeDistance[y0,7,7] free
 EdgeDistance[y0,7,8] free
 EdgeDistance[y0,7,9] free
 EdgeDistance[y0,7,10] free
 EdgeDistance[y0,8,0] free
 EdgeDistance[y0,8,1] free
 EdgeDistance[y0,8,2] free
 EdgeDistance[y0,8,3] free
 EdgeDistance[y0,8,4] free
 EdgeDistance[y0,8,5] free
 EdgeDistance[y0,8,6] free
 EdgeDistance[y0,8,7] free
 EdgeDistance[y0,8,8] free
 EdgeDistance[y0,8,9] free
 EdgeDistance[y0,8,10] free
 EdgeDistance[y0,9,0] free
 EdgeDistance[y0,9,1] free
 EdgeDistance[y0,9,2] free
 EdgeDistance[y0,9,3] free
 EdgeDistance[y0,9,4] free
 EdgeDistance[y0,9,5] free
 EdgeDistance[y0,9,6] free
 EdgeDistance[y0,9,7] free
 EdgeDistance[y0,9,8] free
 EdgeDistance[y0,9,9] free
 EdgeDistance[y0,9,10] free
 EdgeDistance[y0,10,0] free
 EdgeDistance[y0,10,1] free
 EdgeDistance[y0,10,2] free
 EdgeDistance[y0,10,3] free
 EdgeDistance[y0,10,4] free
 EdgeDistance[y0,10,5] free
 EdgeDistance[y0,10,6] free
 EdgeDistance[y0,10,7] free
 EdgeDistance[y0,10,8] free
 EdgeDistance[y0,10,9] free
 EdgeDistance[y0,10,10] free
 EdgeDistance[x1,0,0] free
 EdgeDistance[x1,0,1] free
 EdgeDistance[x1,0,2] free
 EdgeDistance[x1,0,3] free
 EdgeDistance[x1,0,4] free
 EdgeDistance[x1,0,5] free
 EdgeDistance[x1,0,6] free
 EdgeDistance[x1,0,7] free
 EdgeDistance[x1,0,8] free
 EdgeDistance[x1,0,9] free
 EdgeDistance[x1,0,10] free
 EdgeDistance[x1,1,0] free
 EdgeDistance[x1,1,1] free
 EdgeDistance[x1,1,2] free
 EdgeDistance[x1,1,3] free
 EdgeDistance[x1,1,4] free
 EdgeDistance[x1,1,5] free
 EdgeDistance[x1,1,6] free
 EdgeDistance[x1,1,7] free
 EdgeDistance[x1,1,8] free
 EdgeDistance[x1,1,9] free
 EdgeDistance[x1,1,10] free
 EdgeDistance[x1,2,0] free
 EdgeDistance[x1,2,1] free
 EdgeDistance[x1,2,2] free
 EdgeDistance[x1,2,3] free
 EdgeDistance[x1,2,4] free
 EdgeDistance[x1,2,5] free
 EdgeDistance[x1,2,6] free
 EdgeDistance[x1,2,7] free
 EdgeDistance[x1,2,8] free
 EdgeDistance[x1,2,9] free
 EdgeDistance[x1,2,10] free
 EdgeDistance[x1,3,0] free
 EdgeDistance[x1,3,1] free
 EdgeDistance[x1,3,2] free
 EdgeDistance[x1,3,3] free
 EdgeDistance[x1,3,4] free
 EdgeDistance[x1,3,5] free
 EdgeDistance[x1,3,6] free
 EdgeDistance[x1,3,7] free
 EdgeDistance[x1,3,8] free
 EdgeDistance[x1,3,9] free
 EdgeDistance[x1,3,10] free
 EdgeDistance[x1,4,0] free
 EdgeDistance[x1,4,1] free
 EdgeDistance[x1,4,2] free
 EdgeDistance[x1,4,3] free
 EdgeDistance[x1,4,4] free
 EdgeDistance[x1,4,5] free
 EdgeDistance[x1,4,6] free
 EdgeDistance[x1,4,7] free
 EdgeDistance[x1,4,8] free
 EdgeDistance[x1,4,9] free
 EdgeDistance[x1,4,10] free
 EdgeDistance[x1,5,0] free
 EdgeDistance[x1,5,1] free
 EdgeDistance[x1,5,2] free
 EdgeDistance[x1,5,3] free
 EdgeDistance[x1,5,4] free
 EdgeDistance[x1,5,5] free
 EdgeDistance[x1,5,6] free
 EdgeDistance[x1,5,7] free
 EdgeDistance[x1,5,8] free
 EdgeDistance[x1,5,9] free
 EdgeDistance[x1,5,10] free
 EdgeDistance[x1,6,0] free
 EdgeDistance[x1,6,1] free
 EdgeDistance[x1,6,2] free
 EdgeDistance[x1,6,3] free
 EdgeDistance[x1,6,4] free
 EdgeDistance[x1,6,5] free
 EdgeDistance[x1,6,6] free
 EdgeDistance[x1,6,7] free
 EdgeDistance[x1,6,8] free
 EdgeDistance[x1,6,9] free
 EdgeDistance[x1,6,10] free
 EdgeDistance[x1,7,0] free
 EdgeDistance[x1,7,1] free
 EdgeDistance[x1,7,2] free
 EdgeDistance[x1,7,3] free
 EdgeDistance[x1,7,4] free
 EdgeDistance[x1,7,5] free
 EdgeDistance[x1,7,6] free
 EdgeDistance[x1,7,7] free
 EdgeDistance[x1,7,8] free
 EdgeDistance[x1,7,9] free
 EdgeDistance[x1,7,10] free
 EdgeDistance[x1,8,0] free
 EdgeDistance[x1,8,1] free
 EdgeDistance[x1,8,2] free
 EdgeDistance[x1,8,3] free
 EdgeDistance[x1,8,4] free
 EdgeDistance[x1,8,5] free
 EdgeDistance[x1,8,6] free
 EdgeDistance[x1,8,7] free
 EdgeDistance[x1,8,8] free
 EdgeDistance[x1,8,9] free
 EdgeDistance[x1,8,10] free
 EdgeDistance[x1,9,0] free
 EdgeDistance[x1,9,1] free
 EdgeDistance[x1,9,2] free
 EdgeDistance[x1,9,3] free
 EdgeDistance[x1,9,4] free
 EdgeDistance[x1,9,5] free
 EdgeDistance[x1,9,6] free
 EdgeDistance[x1,9,7] free
 EdgeDistance[x1,9,8] free
 EdgeDistance[x1,9,9] free
 EdgeDistance[x1,9,10] free
 EdgeDistance[x1,10,0] free
 EdgeDistance[x1,10,1] free
 EdgeDistance[x1,10,2] free
 EdgeDistance[x1,10,3] free
 EdgeDistance[x1,10,4] free
 EdgeDistance[x1,10,5] free
 EdgeDistance[x1,10,6] free
 EdgeDistance[x1,10,7] free
 EdgeDistance[x1,10,8] free
 EdgeDistance[x1,10,9] free
 EdgeDistance[x1,10,10] free
 EdgeDistance[y1,0,0] free
 EdgeDistance[y1,0,1] free
 EdgeDistance[y1,0,2] free
 EdgeDistance[y1,0,3] free
 EdgeDistance[y1,0,4] free
 EdgeDistance[y1,0,5] free
 EdgeDistance[y1,0,6] free
 EdgeDistance[y1,0,7] free
 EdgeDistance[y1,0,8] free
 EdgeDistance[y1,0,9] free
 EdgeDistance[y1,0,10] free
 EdgeDistance[y1,1,0] free
 EdgeDistance[y1,1,1] free
 EdgeDistance[y1,1,2] free
 EdgeDistance[y1,1,3] free
 EdgeDistance[y1,1,4] free
 EdgeDistance[y1,1,5] free
 EdgeDistance[y1,1,6] free
 EdgeDistance[y1,1,7] free
 EdgeDistance[y1,1,8] free
 EdgeDistance[y1,1,9] free
 EdgeDistance[y1,1,10] free
 EdgeDistance[y1,2,0] free
 EdgeDistance[y1,2,1] free
 EdgeDistance[y1,2,2] free
 EdgeDistance[y1,2,3] free
 EdgeDistance[y1,2,4] free
 EdgeDistance[y1,2,5] free
 EdgeDistance[y1,2,6] free
 EdgeDistance[y1,2,7] free
 EdgeDistance[y1,2,8] free
 EdgeDistance[y1,2,9] free
 EdgeDistance[y1,2,10] free
 EdgeDistance[y1,3,0] free
 EdgeDistance[y1,3,1] free
 EdgeDistance[y1,3,2] free
 EdgeDistance[y1,3,3] free
 EdgeDistance[y1,3,4] free
 EdgeDistance[y1,3,5] free
 EdgeDistance[y1,3,6] free
 EdgeDistance[y1,3,7] free
 EdgeDistance[y1,3,8] free
 EdgeDistance[y1,3,9] free
 EdgeDistance[y1,3,10] free
 EdgeDistance[y1,4,0] free
 EdgeDistance[y1,4,1] free
 EdgeDistance[y1,4,2] free
 EdgeDistance[y1,4,3] free
 EdgeDistance[y1,4,4] free
 EdgeDistance[y1,4,5] free
 EdgeDistance[y1,4,6] free
 EdgeDistance[y1,4,7] free
 EdgeDistance[y1,4,8] free
 EdgeDistance[y1,4,9] free
 EdgeDistance[y1,4,10] free
 EdgeDistance[y1,5,0] free
 EdgeDistance[y1,5,1] free
 EdgeDistance[y1,5,2] free
 EdgeDistance[y1,5,3] free
 EdgeDistance[y1,5,4] free
 EdgeDistance[y1,5,5] free
 EdgeDistance[y1,5,6] free
 EdgeDistance[y1,5,7] free
 EdgeDistance[y1,5,8] free
 EdgeDistance[y1,5,9] free
 EdgeDistance[y1,5,10] free
 EdgeDistance[y1,6,0] free
 EdgeDistance[y1,6,1] free
 EdgeDistance[y1,6,2] free
 EdgeDistance[y1,6,3] free
 EdgeDistance[y1,6,4] free
 EdgeDistance[y1,6,5] free
 EdgeDistance[y1,6,6] free
 EdgeDistance[y1,6,7] free
 EdgeDistance[y1,6,8] free
 EdgeDistance[y1,6,9] free
 EdgeDistance[y1,6,10] free
 EdgeDistance[y1,7,0] free
 EdgeDistance[y1,7,1] free
 EdgeDistance[y1,7,2] free
 EdgeDistance[y1,7,3] free
 EdgeDistance[y1,7,4] free
 EdgeDistance[y1,7,5] free
 EdgeDistance[y1,7,6] free
 EdgeDistance[y1,7,7] free
 EdgeDistance[y1,7,8] free
 EdgeDistance[y1,7,9] free
 EdgeDistance[y1,7,10] free
 EdgeDistance[y1,8,0] free
 EdgeDistance[y1,8,1] free
 EdgeDistance[y1,8,2] free
 EdgeDistance[y1,8,3] free
 EdgeDistance[y1,8,4] free
 EdgeDistance[y1,8,5] free
 EdgeDistance[y1,8,6] free
 EdgeDistance[y1,8,7] free
 EdgeDistance[y1,8,8] free
 EdgeDistance[y1,8,9] free
 EdgeDistance[y1,8,10] free
 EdgeDistance[y1,9,0] free
 EdgeDistance[y1,9,1] free
 EdgeDistance[y1,9,2] free
 EdgeDistance[y1,9,3] free
 EdgeDistance[y1,9,4] free
 EdgeDistance[y1,9,5] free
 EdgeDistance[y1,9,6] free
 EdgeDistance[y1,9,7] free
 EdgeDistance[y1,9,8] free
 EdgeDistance[y1,9,9] free
 EdgeDistance[y1,9,10] free
 EdgeDistance[y1,10,0] free
 EdgeDistance[y1,10,1] free
 EdgeDistance[y1,10,2] free
 EdgeDistance[y1,10,3] free
 EdgeDistance[y1,10,4] free
 EdgeDistance[y1,10,5] free
 EdgeDistance[y1,10,6] free
 EdgeDistance[y1,10,7] free
 EdgeDistance[y1,10,8] free
 EdgeDistance[y1,10,9] free
 EdgeDistance[y1,10,10] free
 X0Diff[0,0] free
 X0Diff[0,1] free
 X0Diff[0,2] free
 X0Diff[0,3] free
 X0Diff[0,4] free
 X0Diff[0,5] free
 X0Diff[0,6] free
 X0Diff[0,7] free
 X0Diff[0,8] free
 X0Diff[0,9] free
 X0Diff[0,10] free
 X0Diff[1,0] free
 X0Diff[1,1] free
 X0Diff[1,2] free
 X0Diff[1,3] free
 X0Diff[1,4] free
 X0Diff[1,5] free
 X0Diff[1,6] free
 X0Diff[1,7] free
 X0Diff[1,8] free
 X0Diff[1,9] free
 X0Diff[1,10] free
 X0Diff[2,0] free
 X0Diff[2,1] free
 X0Diff[2,2] free
 X0Diff[2,3] free
 X0Diff[2,4] free
 X0Diff[2,5] free
 X0Diff[2,6] free
 X0Diff[2,7] free
 X0Diff[2,8] free
 X0Diff[2,9] free
 X0Diff[2,10] free
 X0Diff[3,0] free
 X0Diff[3,1] free
 X0Diff[3,2] free
 X0Diff[3,3] free
 X0Diff[3,4] free
 X0Diff[3,5] free
 X0Diff[3,6] free
 X0Diff[3,7] free
 X0Diff[3,8] free
 X0Diff[3,9] free
 X0Diff[3,10] free
 X0Diff[4,0] free
 X0Diff[4,1] free
 X0Diff[4,2] free
 X0Diff[4,3] free
 X0Diff[4,4] free
 X0Diff[4,5] free
 X0Diff[4,6] free
 X0Diff[4,7] free
 X0Diff[4,8] free
 X0Diff[4,9] free
 X0Diff[4,10] free
 X0Diff[5,0] free
 X0Diff[5,1] free
 X0Diff[5,2] free
 X0Diff[5,3] free
 X0Diff[5,4] free
 X0Diff[5,5] free
 X0Diff[5,6] free
 X0Diff[5,7] free
 X0Diff[5,8] free
 X0Diff[5,9] free
 X0Diff[5,10] free
 X0Diff[6,0] free
 X0Diff[6,1] free
 X0Diff[6,2] free
 X0Diff[6,3] free
 X0Diff[6,4] free
 X0Diff[6,5] free
 X0Diff[6,6] free
 X0Diff[6,7] free
 X0Diff[6,8] free
 X0Diff[6,9] free
 X0Diff[6,10] free
 X0Diff[7,0] free
 X0Diff[7,1] free
 X0Diff[7,2] free
 X0Diff[7,3] free
 X0Diff[7,4] free
 X0Diff[7,5] free
 X0Diff[7,6] free
 X0Diff[7,7] free
 X0Diff[7,8] free
 X0Diff[7,9] free
 X0Diff[7,10] free
 X0Diff[8,0] free
 X0Diff[8,1] free
 X0Diff[8,2] free
 X0Diff[8,3] free
 X0Diff[8,4] free
 X0Diff[8,5] free
 X0Diff[8,6] free
 X0Diff[8,7] free
 X0Diff[8,8] free
 X0Diff[8,9] free
 X0Diff[8,10] free
 X0Diff[9,0] free
 X0Diff[9,1] free
 X0Diff[9,2] free
 X0Diff[9,3] free
 X0Diff[9,4] free
 X0Diff[9,5] free
 X0Diff[9,6] free
 X0Diff[9,7] free
 X0Diff[9,8] free
 X0Diff[9,9] free
 X0Diff[9,10] free
 X0Diff[10,0] free
 X0Diff[10,1] free
 X0Diff[10,2] free
 X0Diff[10,3] free
 X0Diff[10,4] free
 X0Diff[10,5] free
 X0Diff[10,6] free
 X0Diff[10,7] free
 X0Diff[10,8] free
 X0Diff[10,9] free
 X0Diff[10,10] free
 Y0Diff[0,0] free
 Y0Diff[0,1] free
 Y0Diff[0,2] free
 Y0Diff[0,3] free
 Y0Diff[0,4] free
 Y0Diff[0,5] free
 Y0Diff[0,6] free
 Y0Diff[0,7] free
 Y0Diff[0,8] free
 Y0Diff[0,9] free
 Y0Diff[0,10] free
 Y0Diff[1,0] free
 Y0Diff[1,1] free
 Y0Diff[1,2] free
 Y0Diff[1,3] free
 Y0Diff[1,4] free
 Y0Diff[1,5] free
 Y0Diff[1,6] free
 Y0Diff[1,7] free
 Y0Diff[1,8] free
 Y0Diff[1,9] free
 Y0Diff[1,10] free
 Y0Diff[2,0] free
 Y0Diff[2,1] free
 Y0Diff[2,2] free
 Y0Diff[2,3] free
 Y0Diff[2,4] free
 Y0Diff[2,5] free
 Y0Diff[2,6] free
 Y0Diff[2,7] free
 Y0Diff[2,8] free
 Y0Diff[2,9] free
 Y0Diff[2,10] free
 Y0Diff[3,0] free
 Y0Diff[3,1] free
 Y0Diff[3,2] free
 Y0Diff[3,3] free
 Y0Diff[3,4] free
 Y0Diff[3,5] free
 Y0Diff[3,6] free
 Y0Diff[3,7] free
 Y0Diff[3,8] free
 Y0Diff[3,9] free
 Y0Diff[3,10] free
 Y0Diff[4,0] free
 Y0Diff[4,1] free
 Y0Diff[4,2] free
 Y0Diff[4,3] free
 Y0Diff[4,4] free
 Y0Diff[4,5] free
 Y0Diff[4,6] free
 Y0Diff[4,7] free
 Y0Diff[4,8] free
 Y0Diff[4,9] free
 Y0Diff[4,10] free
 Y0Diff[5,0] free
 Y0Diff[5,1] free
 Y0Diff[5,2] free
 Y0Diff[5,3] free
 Y0Diff[5,4] free
 Y0Diff[5,5] free
 Y0Diff[5,6] free
 Y0Diff[5,7] free
 Y0Diff[5,8] free
 Y0Diff[5,9] free
 Y0Diff[5,10] free
 Y0Diff[6,0] free
 Y0Diff[6,1] free
 Y0Diff[6,2] free
 Y0Diff[6,3] free
 Y0Diff[6,4] free
 Y0Diff[6,5] free
 Y0Diff[6,6] free
 Y0Diff[6,7] free
 Y0Diff[6,8] free
 Y0Diff[6,9] free
 Y0Diff[6,10] free
 Y0Diff[7,0] free
 Y0Diff[7,1] free
 Y0Diff[7,2] free
 Y0Diff[7,3] free
 Y0Diff[7,4] free
 Y0Diff[7,5] free
 Y0Diff[7,6] free
 Y0Diff[7,7] free
 Y0Diff[7,8] free
 Y0Diff[7,9] free
 Y0Diff[7,10] free
 Y0Diff[8,0] free
 Y0Diff[8,1] free
 Y0Diff[8,2] free
 Y0Diff[8,3] free
 Y0Diff[8,4] free
 Y0Diff[8,5] free
 Y0Diff[8,6] free
 Y0Diff[8,7] free
 Y0Diff[8,8] free
 Y0Diff[8,9] free
 Y0Diff[8,10] free
 Y0Diff[9,0] free
 Y0Diff[9,1] free
 Y0Diff[9,2] free
 Y0Diff[9,3] free
 Y0Diff[9,4] free
 Y0Diff[9,5] free
 Y0Diff[9,6] free
 Y0Diff[9,7] free
 Y0Diff[9,8] free
 Y0Diff[9,9] free
 Y0Diff[9,10] free
 Y0Diff[10,0] free
 Y0Diff[10,1] free
 Y0Diff[10,2] free
 Y0Diff[10,3] free
 Y0Diff[10,4] free
 Y0Diff[10,5] free
 Y0Diff[10,6] free
 Y0Diff[10,7] free
 Y0Diff[10,8] free
 Y0Diff[10,9] free
 Y0Diff[10,10] free
 X1Diff[0,0] free
 X1Diff[0,1] free
 X1Diff[0,2] free
 X1Diff[0,3] free
 X1Diff[0,4] free
 X1Diff[0,5] free
 X1Diff[0,6] free
 X1Diff[0,7] free
 X1Diff[0,8] free
 X1Diff[0,9] free
 X1Diff[0,10] free
 X1Diff[1,0] free
 X1Diff[1,1] free
 X1Diff[1,2] free
 X1Diff[1,3] free
 X1Diff[1,4] free
 X1Diff[1,5] free
 X1Diff[1,6] free
 X1Diff[1,7] free
 X1Diff[1,8] free
 X1Diff[1,9] free
 X1Diff[1,10] free
 X1Diff[2,0] free
 X1Diff[2,1] free
 X1Diff[2,2] free
 X1Diff[2,3] free
 X1Diff[2,4] free
 X1Diff[2,5] free
 X1Diff[2,6] free
 X1Diff[2,7] free
 X1Diff[2,8] free
 X1Diff[2,9] free
 X1Diff[2,10] free
 X1Diff[3,0] free
 X1Diff[3,1] free
 X1Diff[3,2] free
 X1Diff[3,3] free
 X1Diff[3,4] free
 X1Diff[3,5] free
 X1Diff[3,6] free
 X1Diff[3,7] free
 X1Diff[3,8] free
 X1Diff[3,9] free
 X1Diff[3,10] free
 X1Diff[4,0] free
 X1Diff[4,1] free
 X1Diff[4,2] free
 X1Diff[4,3] free
 X1Diff[4,4] free
 X1Diff[4,5] free
 X1Diff[4,6] free
 X1Diff[4,7] free
 X1Diff[4,8] free
 X1Diff[4,9] free
 X1Diff[4,10] free
 X1Diff[5,0] free
 X1Diff[5,1] free
 X1Diff[5,2] free
 X1Diff[5,3] free
 X1Diff[5,4] free
 X1Diff[5,5] free
 X1Diff[5,6] free
 X1Diff[5,7] free
 X1Diff[5,8] free
 X1Diff[5,9] free
 X1Diff[5,10] free
 X1Diff[6,0] free
 X1Diff[6,1] free
 X1Diff[6,2] free
 X1Diff[6,3] free
 X1Diff[6,4] free
 X1Diff[6,5] free
 X1Diff[6,6] free
 X1Diff[6,7] free
 X1Diff[6,8] free
 X1Diff[6,9] free
 X1Diff[6,10] free
 X1Diff[7,0] free
 X1Diff[7,1] free
 X1Diff[7,2] free
 X1Diff[7,3] free
 X1Diff[7,4] free
 X1Diff[7,5] free
 X1Diff[7,6] free
 X1Diff[7,7] free
 X1Diff[7,8] free
 X1Diff[7,9] free
 X1Diff[7,10] free
 X1Diff[8,0] free
 X1Diff[8,1] free
 X1Diff[8,2] free
 X1Diff[8,3] free
 X1Diff[8,4] free
 X1Diff[8,5] free
 X1Diff[8,6] free
 X1Diff[8,7] free
 X1Diff[8,8] free
 X1Diff[8,9] free
 X1Diff[8,10] free
 X1Diff[9,0] free
 X1Diff[9,1] free
 X1Diff[9,2] free
 X1Diff[9,3] free
 X1Diff[9,4] free
 X1Diff[9,5] free
 X1Diff[9,6] free
 X1Diff[9,7] free
 X1Diff[9,8] free
 X1Diff[9,9] free
 X1Diff[9,10] free
 X1Diff[10,0] free
 X1Diff[10,1] free
 X1Diff[10,2] free
 X1Diff[10,3] free
 X1Diff[10,4] free
 X1Diff[10,5] free
 X1Diff[10,6] free
 X1Diff[10,7] free
 X1Diff[10,8] free
 X1Diff[10,9] free
 X1Diff[10,10] free
 Y1Diff[0,0] free
 Y1Diff[0,1] free
 Y1Diff[0,2] free
 Y1Diff[0,3] free
 Y1Diff[0,4] free
 Y1Diff[0,5] free
 Y1Diff[0,6] free
 Y1Diff[0,7] free
 Y1Diff[0,8] free
 Y1Diff[0,9] free
 Y1Diff[0,10] free
 Y1Diff[1,0] free
 Y1Diff[1,1] free
 Y1Diff[1,2] free
 Y1Diff[1,3] free
 Y1Diff[1,4] free
 Y1Diff[1,5] free
 Y1Diff[1,6] free
 Y1Diff[1,7] free
 Y1Diff[1,8] free
 Y1Diff[1,9] free
 Y1Diff[1,10] free
 Y1Diff[2,0] free
 Y1Diff[2,1] free
 Y1Diff[2,2] free
 Y1Diff[2,3] free
 Y1Diff[2,4] free
 Y1Diff[2,5] free
 Y1Diff[2,6] free
 Y1Diff[2,7] free
 Y1Diff[2,8] free
 Y1Diff[2,9] free
 Y1Diff[2,10] free
 Y1Diff[3,0] free
 Y1Diff[3,1] free
 Y1Diff[3,2] free
 Y1Diff[3,3] free
 Y1Diff[3,4] free
 Y1Diff[3,5] free
 Y1Diff[3,6] free
 Y1Diff[3,7] free
 Y1Diff[3,8] free
 Y1Diff[3,9] free
 Y1Diff[3,10] free
 Y1Diff[4,0] free
 Y1Diff[4,1] free
 Y1Diff[4,2] free
 Y1Diff[4,3] free
 Y1Diff[4,4] free
 Y1Diff[4,5] free
 Y1Diff[4,6] free
 Y1Diff[4,7] free
 Y1Diff[4,8] free
 Y1Diff[4,9] free
 Y1Diff[4,10] free
 Y1Diff[5,0] free
 Y1Diff[5,1] free
 Y1Diff[5,2] free
 Y1Diff[5,3] free
 Y1Diff[5,4] free
 Y1Diff[5,5] free
 Y1Diff[5,6] free
 Y1Diff[5,7] free
 Y1Diff[5,8] free
 Y1Diff[5,9] free
 Y1Diff[5,10] free
 Y1Diff[6,0] free
 Y1Diff[6,1] free
 Y1Diff[6,2] free
 Y1Diff[6,3] free
 Y1Diff[6,4] free
 Y1Diff[6,5] free
 Y1Diff[6,6] free
 Y1Diff[6,7] free
 Y1Diff[6,8] free
 Y1Diff[6,9] free
 Y1Diff[6,10] free
 Y1Diff[7,0] free
 Y1Diff[7,1] free
 Y1Diff[7,2] free
 Y1Diff[7,3] free
 Y1Diff[7,4] free
 Y1Diff[7,5] free
 Y1Diff[7,6] free
 Y1Diff[7,7] free
 Y1Diff[7,8] free
 Y1Diff[7,9] free
 Y1Diff[7,10] free
 Y1Diff[8,0] free
 Y1Diff[8,1] free
 Y1Diff[8,2] free
 Y1Diff[8,3] free
 Y1Diff[8,4] free
 Y1Diff[8,5] free
 Y1Diff[8,6] free
 Y1Diff[8,7] free
 Y1Diff[8,8] free
 Y1Diff[8,9] free
 Y1Diff[8,10] free
 Y1Diff[9,0] free
 Y1Diff[9,1] free
 Y1Diff[9,2] free
 Y1Diff[9,3] free
 Y1Diff[9,4] free
 Y1Diff[9,5] free
 Y1Diff[9,6] free
 Y1Diff[9,7] free
 Y1Diff[9,8] free
 Y1Diff[9,9] free
 Y1Diff[9,10] free
 Y1Diff[10,0] free
 Y1Diff[10,1] free
 Y1Diff[10,2] free
 Y1Diff[10,3] free
 Y1Diff[10,4] free
 Y1Diff[10,5] free
 Y1Diff[10,6] free
 Y1Diff[10,7] free
 Y1Diff[10,8] free
 Y1Diff[10,9] free
 Y1Diff[10,10] free
 1 <= X0Group[0] <= 11
 1 <= X0Group[1] <= 11
 1 <= X0Group[2] <= 11
 1 <= X0Group[3] <= 11
 1 <= X0Group[4] <= 11
 1 <= X0Group[5] <= 11
 1 <= X0Group[6] <= 11
 1 <= X0Group[7] <= 11
 1 <= X0Group[8] <= 11
 1 <= X0Group[9] <= 11
 1 <= X0Group[10] <= 11
 1 <= Y0Group[0] <= 11
 1 <= Y0Group[1] <= 11
 1 <= Y0Group[2] <= 11
 1 <= Y0Group[3] <= 11
 1 <= Y0Group[4] <= 11
 1 <= Y0Group[5] <= 11
 1 <= Y0Group[6] <= 11
 1 <= Y0Group[7] <= 11
 1 <= Y0Group[8] <= 11
 1 <= Y0Group[9] <= 11
 1 <= Y0Group[10] <= 11
 1 <= X1Group[0] <= 11
 1 <= X1Group[1] <= 11
 1 <= X1Group[2] <= 11
 1 <= X1Group[3] <= 11
 1 <= X1Group[4] <= 11
 1 <= X1Group[5] <= 11
 1 <= X1Group[6] <= 11
 1 <= X1Group[7] <= 11
 1 <= X1Group[8] <= 11
 1 <= X1Group[9] <= 11
 1 <= X1Group[10] <= 11
 1 <= Y1Group[0] <= 11
 1 <= Y1Group[1] <= 11
 1 <= Y1Group[2] <= 11
 1 <= Y1Group[3] <= 11
 1 <= Y1Group[4] <= 11
 1 <= Y1Group[5] <= 11
 1 <= Y1Group[6] <= 11
 1 <= Y1Group[7] <= 11
 1 <= Y1Group[8] <= 11
 1 <= Y1Group[9] <= 11
 1 <= Y1Group[10] <= 11
 1 <= X0GroupCount <= 11
 1 <= Y0GroupCount <= 11
 1 <= X1GroupCount <= 11
 1 <= Y1GroupCount <= 11
 C2252 >= 1
 StartColumn[0] >= 1
 StartColumn[1] >= 1
 StartColumn[2] >= 1
 StartColumn[3] >= 1
 StartColumn[4] >= 1
 StartColumn[5] >= 1
 StartColumn[6] >= 1
 StartColumn[7] >= 1
 StartColumn[8] >= 1
 StartColumn[9] >= 1
 StartColumn[10] >= 1
 EndColumn[0] >= 1
 EndColumn[1] >= 1
 EndColumn[2] >= 1
 EndColumn[3] >= 1
 EndColumn[4] >= 1
 EndColumn[5] >= 1
 EndColumn[6] >= 1
 EndColumn[7] >= 1
 EndColumn[8] >= 1
 EndColumn[9] >= 1
 EndColumn[10] >= 1
 ColumnCount >= 1
 FirstColumn = 1
Binaries
 Above[0,0] Above[0,1] Above[0,2] Above[0,3] Above[0,4] Above[0,5]
 Above[0,6] Above[0,7] Above[0,8] Above[0,9] Above[0,10] Above[1,0]
 Above[1,1] Above[1,2] Above[1,3] Above[1,4] Above[1,5] Above[1,6]
 Above[1,7] Above[1,8] Above[1,9] Above[1,10] Above[2,0] Above[2,1]
 Above[2,2] Above[2,3] Above[2,4] Above[2,5] Above[2,6] Above[2,7]
 Above[2,8] Above[2,9] Above[2,10] Above[3,0] Above[3,1] Above[3,2]
 Above[3,3] Above[3,4] Above[3,5] Above[3,6] Above[3,7] Above[3,8]
 Above[3,9] Above[3,10] Above[4,0] Above[4,1] Above[4,2] Above[4,3]
 Above[4,4] Above[4,5] Above[4,6] Above[4,7] Above[4,8] Above[4,9]
 Above[4,10] Above[5,0] Above[5,1] Above[5,2] Above[5,3] Above[5,4]
 Above[5,5] Above[5,6] Above[5,7] Above[5,8] Above[5,9] Above[5,10]
 Above[6,0] Above[6,1] Above[6,2] Above[6,3] Above[6,4] Above[6,5]
 Above[6,6] Above[6,7] Above[6,8] Above[6,9] Above[6,10] Above[7,0]
 Above[7,1] Above[7,2] Above[7,3] Above[7,4] Above[7,5] Above[7,6]
 Above[7,7] Above[7,8] Above[7,9] Above[7,10] Above[8,0] Above[8,1]
 Above[8,2] Above[8,3] Above[8,4] Above[8,5] Above[8,6] Above[8,7]
 Above[8,8] Above[8,9] Above[8,10] Above[9,0] Above[9,1] Above[9,2]
 Above[9,3] Above[9,4] Above[9,5] Above[9,6] Above[9,7] Above[9,8]
 Above[9,9] Above[9,10] Above[10,0] Above[10,1] Above[10,2] Above[10,3]
 Above[10,4] Above[10,5] Above[10,6] Above[10,7] Above[10,8] Above[10,9]
 Above[10,10] OnLeft[0,0] OnLeft[0,1] OnLeft[0,2] OnLeft[0,3] OnLeft[0,4]
 OnLeft[0,5] OnLeft[0,6] OnLeft[0,7] OnLeft[0,8] OnLeft[0,9] OnLeft[0,10]
 OnLeft[1,0] OnLeft[1,1] OnLeft[1,2] OnLeft[1,3] OnLeft[1,4] OnLeft[1,5]
 OnLeft[1,6] OnLeft[1,7] OnLeft[1,8] OnLeft[1,9] OnLeft[1,10] OnLeft[2,0]
 OnLeft[2,1] OnLeft[2,2] OnLeft[2,3] OnLeft[2,4] OnLeft[2,5] OnLeft[2,6]
 OnLeft[2,7] OnLeft[2,8] OnLeft[2,9] OnLeft[2,10] OnLeft[3,0] OnLeft[3,1]
 OnLeft[3,2] OnLeft[3,3] OnLeft[3,4] OnLeft[3,5] OnLeft[3,6] OnLeft[3,7]
 OnLeft[3,8] OnLeft[3,9] OnLeft[3,10] OnLeft[4,0] OnLeft[4,1] OnLeft[4,2]
 OnLeft[4,3] OnLeft[4,4] OnLeft[4,5] OnLeft[4,6] OnLeft[4,7] OnLeft[4,8]
 OnLeft[4,9] OnLeft[4,10] OnLeft[5,0] OnLeft[5,1] OnLeft[5,2] OnLeft[5,3]
 OnLeft[5,4] OnLeft[5,5] OnLeft[5,6] OnLeft[5,7] OnLeft[5,8] OnLeft[5,9]
 OnLeft[5,10] OnLeft[6,0] OnLeft[6,1] OnLeft[6,2] OnLeft[6,3] OnLeft[6,4]
 OnLeft[6,5] OnLeft[6,6] OnLeft[6,7] OnLeft[6,8] OnLeft[6,9] OnLeft[6,10]
 OnLeft[7,0] OnLeft[7,1] OnLeft[7,2] OnLeft[7,3] OnLeft[7,4] OnLeft[7,5]
 OnLeft[7,6] OnLeft[7,7] OnLeft[7,8] OnLeft[7,9] OnLeft[7,10] OnLeft[8,0]
 OnLeft[8,1] OnLeft[8,2] OnLeft[8,3] OnLeft[8,4] OnLeft[8,5] OnLeft[8,6]
 OnLeft[8,7] OnLeft[8,8] OnLeft[8,9] OnLeft[8,10] OnLeft[9,0] OnLeft[9,1]
 OnLeft[9,2] OnLeft[9,3] OnLeft[9,4] OnLeft[9,5] OnLeft[9,6] OnLeft[9,7]
 OnLeft[9,8] OnLeft[9,9] OnLeft[9,10] OnLeft[10,0] OnLeft[10,1]
 OnLeft[10,2] OnLeft[10,3] OnLeft[10,4] OnLeft[10,5] OnLeft[10,6]
 OnLeft[10,7] OnLeft[10,8] OnLeft[10,9] OnLeft[10,10]
 HorizontalOverlap[0,1] HorizontalOverlap[0,2] HorizontalOverlap[0,3]
 HorizontalOverlap[0,4] HorizontalOverlap[0,5] HorizontalOverlap[0,6]
 HorizontalOverlap[0,7] HorizontalOverlap[0,8] HorizontalOverlap[0,9]
 HorizontalOverlap[0,10] HorizontalOverlap[1,0] HorizontalOverlap[1,2]
 HorizontalOverlap[1,3] HorizontalOverlap[1,4] HorizontalOverlap[1,5]
 HorizontalOverlap[1,6] HorizontalOverlap[1,7] HorizontalOverlap[1,8]
 HorizontalOverlap[1,9] HorizontalOverlap[1,10] HorizontalOverlap[2,0]
 HorizontalOverlap[2,1] HorizontalOverlap[2,3] HorizontalOverlap[2,4]
 HorizontalOverlap[2,5] HorizontalOverlap[2,6] HorizontalOverlap[2,7]
 HorizontalOverlap[2,8] HorizontalOverlap[2,9] HorizontalOverlap[2,10]
 HorizontalOverlap[3,0] HorizontalOverlap[3,1] HorizontalOverlap[3,2]
 HorizontalOverlap[3,4] HorizontalOverlap[3,5] HorizontalOverlap[3,6]
 HorizontalOverlap[3,7] HorizontalOverlap[3,8] HorizontalOverlap[3,9]
 HorizontalOverlap[3,10] HorizontalOverlap[4,0] HorizontalOverlap[4,1]
 HorizontalOverlap[4,2] HorizontalOverlap[4,3] HorizontalOverlap[4,5]
 HorizontalOverlap[4,6] HorizontalOverlap[4,7] HorizontalOverlap[4,8]
 HorizontalOverlap[4,9] HorizontalOverlap[4,10] HorizontalOverlap[5,0]
 HorizontalOverlap[5,1] HorizontalOverlap[5,2] HorizontalOverlap[5,3]
 HorizontalOverlap[5,4] HorizontalOverlap[5,6] HorizontalOverlap[5,7]
 HorizontalOverlap[5,8] HorizontalOverlap[5,9] HorizontalOverlap[5,10]
 HorizontalOverlap[6,0] HorizontalOverlap[6,1] HorizontalOverlap[6,2]
 HorizontalOverlap[6,3] HorizontalOverlap[6,4] HorizontalOverlap[6,5]
 HorizontalOverlap[6,7] HorizontalOverlap[6,8] HorizontalOverlap[6,9]
 HorizontalOverlap[6,10] HorizontalOverlap[7,0] HorizontalOverlap[7,1]
 HorizontalOverlap[7,2] HorizontalOverlap[7,3] HorizontalOverlap[7,4]
 HorizontalOverlap[7,5] HorizontalOverlap[7,6] HorizontalOverlap[7,8]
 HorizontalOverlap[7,9] HorizontalOverlap[7,10] HorizontalOverlap[8,0]
 HorizontalOverlap[8,1] HorizontalOverlap[8,2] HorizontalOverlap[8,3]
 HorizontalOverlap[8,4] HorizontalOverlap[8,5] HorizontalOverlap[8,6]
 HorizontalOverlap[8,7] HorizontalOverlap[8,9] HorizontalOverlap[8,10]
 HorizontalOverlap[9,0] HorizontalOverlap[9,1] HorizontalOverlap[9,2]
 HorizontalOverlap[9,3] HorizontalOverlap[9,4] HorizontalOverlap[9,5]
 HorizontalOverlap[9,6] HorizontalOverlap[9,7] HorizontalOverlap[9,8]
 HorizontalOverlap[9,10] HorizontalOverlap[10,0] HorizontalOverlap[10,1]
 HorizontalOverlap[10,2] HorizontalOverlap[10,3] HorizontalOverlap[10,4]
 HorizontalOverlap[10,5] HorizontalOverlap[10,6] HorizontalOverlap[10,7]
 HorizontalOverlap[10,8] HorizontalOverlap[10,9] VerticalOverlap[0,0]
 VerticalOverlap[0,1] VerticalOverlap[0,2] VerticalOverlap[0,3]
 VerticalOverlap[0,4] VerticalOverlap[0,5] VerticalOverlap[0,6]
 VerticalOverlap[0,7] VerticalOverlap[0,8] VerticalOverlap[0,9]
 VerticalOverlap[0,10] VerticalOverlap[1,0] VerticalOverlap[1,1]
 VerticalOverlap[1,2] VerticalOverlap[1,3] VerticalOverlap[1,4]
 VerticalOverlap[1,5] VerticalOverlap[1,6] VerticalOverlap[1,7]
 VerticalOverlap[1,8] VerticalOverlap[1,9] VerticalOverlap[1,10]
 VerticalOverlap[2,0] VerticalOverlap[2,1] VerticalOverlap[2,2]
 VerticalOverlap[2,3] VerticalOverlap[2,4] VerticalOverlap[2,5]
 VerticalOverlap[2,6] VerticalOverlap[2,7] VerticalOverlap[2,8]
 VerticalOverlap[2,9] VerticalOverlap[2,10] VerticalOverlap[3,0]
 VerticalOverlap[3,1] VerticalOverlap[3,2] VerticalOverlap[3,3]
 VerticalOverlap[3,4] VerticalOverlap[3,5] VerticalOverlap[3,6]
 VerticalOverlap[3,7] VerticalOverlap[3,8] VerticalOverlap[3,9]
 VerticalOverlap[3,10] VerticalOverlap[4,0] VerticalOverlap[4,1]
 VerticalOverlap[4,2] VerticalOverlap[4,3] VerticalOverlap[4,4]
 VerticalOverlap[4,5] VerticalOverlap[4,6] VerticalOverlap[4,7]
 VerticalOverlap[4,8] VerticalOverlap[4,9] VerticalOverlap[4,10]
 VerticalOverlap[5,0] VerticalOverlap[5,1] VerticalOverlap[5,2]
 VerticalOverlap[5,3] VerticalOverlap[5,4] VerticalOverlap[5,5]
 VerticalOverlap[5,6] VerticalOverlap[5,7] VerticalOverlap[5,8]
 VerticalOverlap[5,9] VerticalOverlap[5,10] VerticalOverlap[6,0]
 VerticalOverlap[6,1] VerticalOverlap[6,2] VerticalOverlap[6,3]
 VerticalOverlap[6,4] VerticalOverlap[6,5] VerticalOverlap[6,6]
 VerticalOverlap[6,7] VerticalOverlap[6,8] VerticalOverlap[6,9]
 VerticalOverlap[6,10] VerticalOverlap[7,0] VerticalOverlap[7,1]
 VerticalOverlap[7,2] VerticalOverlap[7,3] VerticalOverlap[7,4]
 VerticalOverlap[7,5] VerticalOverlap[7,6] VerticalOverlap[7,7]
 VerticalOverlap[7,8] VerticalOverlap[7,9] VerticalOverlap[7,10]
 VerticalOverlap[8,0] VerticalOverlap[8,1] VerticalOverlap[8,2]
 VerticalOverlap[8,3] VerticalOverlap[8,4] VerticalOverlap[8,5]
 VerticalOverlap[8,6] VerticalOverlap[8,7] VerticalOverlap[8,8]
 VerticalOverlap[8,9] VerticalOverlap[8,10] VerticalOverlap[9,0]
 VerticalOverlap[9,1] VerticalOverlap[9,2] VerticalOverlap[9,3]
 VerticalOverlap[9,4] VerticalOverlap[9,5] VerticalOverlap[9,6]
 VerticalOverlap[9,7] VerticalOverlap[9,8] VerticalOverlap[9,9]
 VerticalOverlap[9,10] VerticalOverlap[10,0] VerticalOverlap[10,1]
 VerticalOverlap[10,2] VerticalOverlap[10,3] VerticalOverlap[10,4]
 VerticalOverlap[10,5] VerticalOverlap[10,6] VerticalOverlap[10,7]
 VerticalOverlap[10,8] VerticalOverlap[10,9] VerticalOverlap[10,10]
 Overlap[0,0] Overlap[0,1] Overlap[0,2] Overlap[0,3] Overlap[0,4]
 Overlap[0,5] Overlap[0,6] Overlap[0,7] Overlap[0,8] Overlap[0,9]
 Overlap[0,10] Overlap[1,0] Overlap[1,1] Overlap[1,2] Overlap[1,3]
 Overlap[1,4] Overlap[1,5] Overlap[1,6] Overlap[1,7] Overlap[1,8]
 Overlap[1,9] Overlap[1,10] Overlap[2,0] Overlap[2,1] Overlap[2,2]
 Overlap[2,3] Overlap[2,4] Overlap[2,5] Overlap[2,6] Overlap[2,7]
 Overlap[2,8] Overlap[2,9] Overlap[2,10] Overlap[3,0] Overlap[3,1]
 Overlap[3,2] Overlap[3,3] Overlap[3,4] Overlap[3,5] Overlap[3,6]
 Overlap[3,7] Overlap[3,8] Overlap[3,9] Overlap[3,10] Overlap[4,0]
 Overlap[4,1] Overlap[4,2] Overlap[4,3] Overlap[4,4] Overlap[4,5]
 Overlap[4,6] Overlap[4,7] Overlap[4,8] Overlap[4,9] Overlap[4,10]
 Overlap[5,0] Overlap[5,1] Overlap[5,2] Overlap[5,3] Overlap[5,4]
 Overlap[5,5] Overlap[5,6] Overlap[5,7] Overlap[5,8] Overlap[5,9]
 Overlap[5,10] Overlap[6,0] Overlap[6,1] Overlap[6,2] Overlap[6,3]
 Overlap[6,4] Overlap[6,5] Overlap[6,6] Overlap[6,7] Overlap[6,8]
 Overlap[6,9] Overlap[6,10] Overlap[7,0] Overlap[7,1] Overlap[7,2]
 Overlap[7,3] Overlap[7,4] Overlap[7,5] Overlap[7,6] Overlap[7,7]
 Overlap[7,8] Overlap[7,9] Overlap[7,10] Overlap[8,0] Overlap[8,1]
 Overlap[8,2] Overlap[8,3] Overlap[8,4] Overlap[8,5] Overlap[8,6]
 Overlap[8,7] Overlap[8,8] Overlap[8,9] Overlap[8,10] Overlap[9,0]
 Overlap[9,1] Overlap[9,2] Overlap[9,3] Overlap[9,4] Overlap[9,5]
 Overlap[9,6] Overlap[9,7] Overlap[9,8] Overlap[9,9] Overlap[9,10]
 Overlap[10,0] Overlap[10,1] Overlap[10,2] Overlap[10,3] Overlap[10,4]
 Overlap[10,5] Overlap[10,6] Overlap[10,7] Overlap[10,8] Overlap[10,9]
 Overlap[10,10] X0LessThan[0,0] X0LessThan[0,1] X0LessThan[0,2]
 X0LessThan[0,3] X0LessThan[0,4] X0LessThan[0,5] X0LessThan[0,6]
 X0LessThan[0,7] X0LessThan[0,8] X0LessThan[0,9] X0LessThan[0,10]
 X0LessThan[1,0] X0LessThan[1,1] X0LessThan[1,2] X0LessThan[1,3]
 X0LessThan[1,4] X0LessThan[1,5] X0LessThan[1,6] X0LessThan[1,7]
 X0LessThan[1,8] X0LessThan[1,9] X0LessThan[1,10] X0LessThan[2,0]
 X0LessThan[2,1] X0LessThan[2,2] X0LessThan[2,3] X0LessThan[2,4]
 X0LessThan[2,5] X0LessThan[2,6] X0LessThan[2,7] X0LessThan[2,8]
 X0LessThan[2,9] X0LessThan[2,10] X0LessThan[3,0] X0LessThan[3,1]
 X0LessThan[3,2] X0LessThan[3,3] X0LessThan[3,4] X0LessThan[3,5]
 X0LessThan[3,6] X0LessThan[3,7] X0LessThan[3,8] X0LessThan[3,9]
 X0LessThan[3,10] X0LessThan[4,0] X0LessThan[4,1] X0LessThan[4,2]
 X0LessThan[4,3] X0LessThan[4,4] X0LessThan[4,5] X0LessThan[4,6]
 X0LessThan[4,7] X0LessThan[4,8] X0LessThan[4,9] X0LessThan[4,10]
 X0LessThan[5,0] X0LessThan[5,1] X0LessThan[5,2] X0LessThan[5,3]
 X0LessThan[5,4] X0LessThan[5,5] X0LessThan[5,6] X0LessThan[5,7]
 X0LessThan[5,8] X0LessThan[5,9] X0LessThan[5,10] X0LessThan[6,0]
 X0LessThan[6,1] X0LessThan[6,2] X0LessThan[6,3] X0LessThan[6,4]
 X0LessThan[6,5] X0LessThan[6,6] X0LessThan[6,7] X0LessThan[6,8]
 X0LessThan[6,9] X0LessThan[6,10] X0LessThan[7,0] X0LessThan[7,1]
 X0LessThan[7,2] X0LessThan[7,3] X0LessThan[7,4] X0LessThan[7,5]
 X0LessThan[7,6] X0LessThan[7,7] X0LessThan[7,8] X0LessThan[7,9]
 X0LessThan[7,10] X0LessThan[8,0] X0LessThan[8,1] X0LessThan[8,2]
 X0LessThan[8,3] X0LessThan[8,4] X0LessThan[8,5] X0LessThan[8,6]
 X0LessThan[8,7] X0LessThan[8,8] X0LessThan[8,9] X0LessThan[8,10]
 X0LessThan[9,0] X0LessThan[9,1] X0LessThan[9,2] X0LessThan[9,3]
 X0LessThan[9,4] X0LessThan[9,5] X0LessThan[9,6] X0LessThan[9,7]
 X0LessThan[9,8] X0LessThan[9,9] X0LessThan[9,10] X0LessThan[10,0]
 X0LessThan[10,1] X0LessThan[10,2] X0LessThan[10,3] X0LessThan[10,4]
 X0LessThan[10,5] X0LessThan[10,6] X0LessThan[10,7] X0LessThan[10,8]
 X0LessThan[10,9] X0LessThan[10,10] Y0LessThan[0,0] Y0LessThan[0,1]
 Y0LessThan[0,2] Y0LessThan[0,3] Y0LessThan[0,4] Y0LessThan[0,5]
 Y0LessThan[0,6] Y0LessThan[0,7] Y0LessThan[0,8] Y0LessThan[0,9]
 Y0LessThan[0,10] Y0LessThan[1,0] Y0LessThan[1,1] Y0LessThan[1,2]
 Y0LessThan[1,3] Y0LessThan[1,4] Y0LessThan[1,5] Y0LessThan[1,6]
 Y0LessThan[1,7] Y0LessThan[1,8] Y0LessThan[1,9] Y0LessThan[1,10]
 Y0LessThan[2,0] Y0LessThan[2,1] Y0LessThan[2,2] Y0LessThan[2,3]
 Y0LessThan[2,4] Y0LessThan[2,5] Y0LessThan[2,6] Y0LessThan[2,7]
 Y0LessThan[2,8] Y0LessThan[2,9] Y0LessThan[2,10] Y0LessThan[3,0]
 Y0LessThan[3,1] Y0LessThan[3,2] Y0LessThan[3,3] Y0LessThan[3,4]
 Y0LessThan[3,5] Y0LessThan[3,6] Y0LessThan[3,7] Y0LessThan[3,8]
 Y0LessThan[3,9] Y0LessThan[3,10] Y0LessThan[4,0] Y0LessThan[4,1]
 Y0LessThan[4,2] Y0LessThan[4,3] Y0LessThan[4,4] Y0LessThan[4,5]
 Y0LessThan[4,6] Y0LessThan[4,7] Y0LessThan[4,8] Y0LessThan[4,9]
 Y0LessThan[4,10] Y0LessThan[5,0] Y0LessThan[5,1] Y0LessThan[5,2]
 Y0LessThan[5,3] Y0LessThan[5,4] Y0LessThan[5,5] Y0LessThan[5,6]
 Y0LessThan[5,7] Y0LessThan[5,8] Y0LessThan[5,9] Y0LessThan[5,10]
 Y0LessThan[6,0] Y0LessThan[6,1] Y0LessThan[6,2] Y0LessThan[6,3]
 Y0LessThan[6,4] Y0LessThan[6,5] Y0LessThan[6,6] Y0LessThan[6,7]
 Y0LessThan[6,8] Y0LessThan[6,9] Y0LessThan[6,10] Y0LessThan[7,0]
 Y0LessThan[7,1] Y0LessThan[7,2] Y0LessThan[7,3] Y0LessThan[7,4]
 Y0LessThan[7,5] Y0LessThan[7,6] Y0LessThan[7,7] Y0LessThan[7,8]
 Y0LessThan[7,9] Y0LessThan[7,10] Y0LessThan[8,0] Y0LessThan[8,1]
 Y0LessThan[8,2] Y0LessThan[8,3] Y0LessThan[8,4] Y0LessThan[8,5]
 Y0LessThan[8,6] Y0LessThan[8,7] Y0LessThan[8,8] Y0LessThan[8,9]
 Y0LessThan[8,10] Y0LessThan[9,0] Y0LessThan[9,1] Y0LessThan[9,2]
 Y0LessThan[9,3] Y0LessThan[9,4] Y0LessThan[9,5] Y0LessThan[9,6]
 Y0LessThan[9,7] Y0LessThan[9,8] Y0LessThan[9,9] Y0LessThan[9,10]
 Y0LessThan[10,0] Y0LessThan[10,1] Y0LessThan[10,2] Y0LessThan[10,3]
 Y0LessThan[10,4] Y0LessThan[10,5] Y0LessThan[10,6] Y0LessThan[10,7]
 Y0LessThan[10,8] Y0LessThan[10,9] Y0LessThan[10,10] X1LessThan[0,0]
 X1LessThan[0,1] X1LessThan[0,2] X1LessThan[0,3] X1LessThan[0,4]
 X1LessThan[0,5] X1LessThan[0,6] X1LessThan[0,7] X1LessThan[0,8]
 X1LessThan[0,9] X1LessThan[0,10] X1LessThan[1,0] X1LessThan[1,1]
 X1LessThan[1,2] X1LessThan[1,3] X1LessThan[1,4] X1LessThan[1,5]
 X1LessThan[1,6] X1LessThan[1,7] X1LessThan[1,8] X1LessThan[1,9]
 X1LessThan[1,10] X1LessThan[2,0] X1LessThan[2,1] X1LessThan[2,2]
 X1LessThan[2,3] X1LessThan[2,4] X1LessThan[2,5] X1LessThan[2,6]
 X1LessThan[2,7] X1LessThan[2,8] X1LessThan[2,9] X1LessThan[2,10]
 X1LessThan[3,0] X1LessThan[3,1] X1LessThan[3,2] X1LessThan[3,3]
 X1LessThan[3,4] X1LessThan[3,5] X1LessThan[3,6] X1LessThan[3,7]
 X1LessThan[3,8] X1LessThan[3,9] X1LessThan[3,10] X1LessThan[4,0]
 X1LessThan[4,1] X1LessThan[4,2] X1LessThan[4,3] X1LessThan[4,4]
 X1LessThan[4,5] X1LessThan[4,6] X1LessThan[4,7] X1LessThan[4,8]
 X1LessThan[4,9] X1LessThan[4,10] X1LessThan[5,0] X1LessThan[5,1]
 X1LessThan[5,2] X1LessThan[5,3] X1LessThan[5,4] X1LessThan[5,5]
 X1LessThan[5,6] X1LessThan[5,7] X1LessThan[5,8] X1LessThan[5,9]
 X1LessThan[5,10] X1LessThan[6,0] X1LessThan[6,1] X1LessThan[6,2]
 X1LessThan[6,3] X1LessThan[6,4] X1LessThan[6,5] X1LessThan[6,6]
 X1LessThan[6,7] X1LessThan[6,8] X1LessThan[6,9] X1LessThan[6,10]
 X1LessThan[7,0] X1LessThan[7,1] X1LessThan[7,2] X1LessThan[7,3]
 X1LessThan[7,4] X1LessThan[7,5] X1LessThan[7,6] X1LessThan[7,7]
 X1LessThan[7,8] X1LessThan[7,9] X1LessThan[7,10] X1LessThan[8,0]
 X1LessThan[8,1] X1LessThan[8,2] X1LessThan[8,3] X1LessThan[8,4]
 X1LessThan[8,5] X1LessThan[8,6] X1LessThan[8,7] X1LessThan[8,8]
 X1LessThan[8,9] X1LessThan[8,10] X1LessThan[9,0] X1LessThan[9,1]
 X1LessThan[9,2] X1LessThan[9,3] X1LessThan[9,4] X1LessThan[9,5]
 X1LessThan[9,6] X1LessThan[9,7] X1LessThan[9,8] X1LessThan[9,9]
 X1LessThan[9,10] X1LessThan[10,0] X1LessThan[10,1] X1LessThan[10,2]
 X1LessThan[10,3] X1LessThan[10,4] X1LessThan[10,5] X1LessThan[10,6]
 X1LessThan[10,7] X1LessThan[10,8] X1LessThan[10,9] X1LessThan[10,10]
 Y1LessThan[0,0] Y1LessThan[0,1] Y1LessThan[0,2] Y1LessThan[0,3]
 Y1LessThan[0,4] Y1LessThan[0,5] Y1LessThan[0,6] Y1LessThan[0,7]
 Y1LessThan[0,8] Y1LessThan[0,9] Y1LessThan[0,10] Y1LessThan[1,0]
 Y1LessThan[1,1] Y1LessThan[1,2] Y1LessThan[1,3] Y1LessThan[1,4]
 Y1LessThan[1,5] Y1LessThan[1,6] Y1LessThan[1,7] Y1LessThan[1,8]
 Y1LessThan[1,9] Y1LessThan[1,10] Y1LessThan[2,0] Y1LessThan[2,1]
 Y1LessThan[2,2] Y1LessThan[2,3] Y1LessThan[2,4] Y1LessThan[2,5]
 Y1LessThan[2,6] Y1LessThan[2,7] Y1LessThan[2,8] Y1LessThan[2,9]
 Y1LessThan[2,10] Y1LessThan[3,0] Y1LessThan[3,1] Y1LessThan[3,2]
 Y1LessThan[3,3] Y1LessThan[3,4] Y1LessThan[3,5] Y1LessThan[3,6]
 Y1LessThan[3,7] Y1LessThan[3,8] Y1LessThan[3,9] Y1LessThan[3,10]
 Y1LessThan[4,0] Y1LessThan[4,1] Y1LessThan[4,2] Y1LessThan[4,3]
 Y1LessThan[4,4] Y1LessThan[4,5] Y1LessThan[4,6] Y1LessThan[4,7]
 Y1LessThan[4,8] Y1LessThan[4,9] Y1LessThan[4,10] Y1LessThan[5,0]
 Y1LessThan[5,1] Y1LessThan[5,2] Y1LessThan[5,3] Y1LessThan[5,4]
 Y1LessThan[5,5] Y1LessThan[5,6] Y1LessThan[5,7] Y1LessThan[5,8]
 Y1LessThan[5,9] Y1LessThan[5,10] Y1LessThan[6,0] Y1LessThan[6,1]
 Y1LessThan[6,2] Y1LessThan[6,3] Y1LessThan[6,4] Y1LessThan[6,5]
 Y1LessThan[6,6] Y1LessThan[6,7] Y1LessThan[6,8] Y1LessThan[6,9]
 Y1LessThan[6,10] Y1LessThan[7,0] Y1LessThan[7,1] Y1LessThan[7,2]
 Y1LessThan[7,3] Y1LessThan[7,4] Y1LessThan[7,5] Y1LessThan[7,6]
 Y1LessThan[7,7] Y1LessThan[7,8] Y1LessThan[7,9] Y1LessThan[7,10]
 Y1LessThan[8,0] Y1LessThan[8,1] Y1LessThan[8,2] Y1LessThan[8,3]
 Y1LessThan[8,4] Y1LessThan[8,5] Y1LessThan[8,6] Y1LessThan[8,7]
 Y1LessThan[8,8] Y1LessThan[8,9] Y1LessThan[8,10] Y1LessThan[9,0]
 Y1LessThan[9,1] Y1LessThan[9,2] Y1LessThan[9,3] Y1LessThan[9,4]
 Y1LessThan[9,5] Y1LessThan[9,6] Y1LessThan[9,7] Y1LessThan[9,8]
 Y1LessThan[9,9] Y1LessThan[9,10] Y1LessThan[10,0] Y1LessThan[10,1]
 Y1LessThan[10,2] Y1LessThan[10,3] Y1LessThan[10,4] Y1LessThan[10,5]
 Y1LessThan[10,6] Y1LessThan[10,7] Y1LessThan[10,8] Y1LessThan[10,9]
 Y1LessThan[10,10] OtherElementStartsInNextColumn[0,0]
 OtherElementStartsInNextColumn[0,1] OtherElementStartsInNextColumn[0,2]
 OtherElementStartsInNextColumn[0,3] OtherElementStartsInNextColumn[0,4]
 OtherElementStartsInNextColumn[0,5] OtherElementStartsInNextColumn[0,6]
 OtherElementStartsInNextColumn[0,7] OtherElementStartsInNextColumn[0,8]
 OtherElementStartsInNextColumn[0,9] OtherElementStartsInNextColumn[0,10]
 OtherElementStartsInNextColumn[1,0] OtherElementStartsInNextColumn[1,1]
 OtherElementStartsInNextColumn[1,2] OtherElementStartsInNextColumn[1,3]
 OtherElementStartsInNextColumn[1,4] OtherElementStartsInNextColumn[1,5]
 OtherElementStartsInNextColumn[1,6] OtherElementStartsInNextColumn[1,7]
 OtherElementStartsInNextColumn[1,8] OtherElementStartsInNextColumn[1,9]
 OtherElementStartsInNextColumn[1,10] OtherElementStartsInNextColumn[2,0]
 OtherElementStartsInNextColumn[2,1] OtherElementStartsInNextColumn[2,2]
 OtherElementStartsInNextColumn[2,3] OtherElementStartsInNextColumn[2,4]
 OtherElementStartsInNextColumn[2,5] OtherElementStartsInNextColumn[2,6]
 OtherElementStartsInNextColumn[2,7] OtherElementStartsInNextColumn[2,8]
 OtherElementStartsInNextColumn[2,9] OtherElementStartsInNextColumn[2,10]
 OtherElementStartsInNextColumn[3,0] OtherElementStartsInNextColumn[3,1]
 OtherElementStartsInNextColumn[3,2] OtherElementStartsInNextColumn[3,3]
 OtherElementStartsInNextColumn[3,4] OtherElementStartsInNextColumn[3,5]
 OtherElementStartsInNextColumn[3,6] OtherElementStartsInNextColumn[3,7]
 OtherElementStartsInNextColumn[3,8] OtherElementStartsInNextColumn[3,9]
 OtherElementStartsInNextColumn[3,10] OtherElementStartsInNextColumn[4,0]
 OtherElementStartsInNextColumn[4,1] OtherElementStartsInNextColumn[4,2]
 OtherElementStartsInNextColumn[4,3] OtherElementStartsInNextColumn[4,4]
 OtherElementStartsInNextColumn[4,5] OtherElementStartsInNextColumn[4,6]
 OtherElementStartsInNextColumn[4,7] OtherElementStartsInNextColumn[4,8]
 OtherElementStartsInNextColumn[4,9] OtherElementStartsInNextColumn[4,10]
 OtherElementStartsInNextColumn[5,0] OtherElementStartsInNextColumn[5,1]
 OtherElementStartsInNextColumn[5,2] OtherElementStartsInNextColumn[5,3]
 OtherElementStartsInNextColumn[5,4] OtherElementStartsInNextColumn[5,5]
 OtherElementStartsInNextColumn[5,6] OtherElementStartsInNextColumn[5,7]
 OtherElementStartsInNextColumn[5,8] OtherElementStartsInNextColumn[5,9]
 OtherElementStartsInNextColumn[5,10] OtherElementStartsInNextColumn[6,0]
 OtherElementStartsInNextColumn[6,1] OtherElementStartsInNextColumn[6,2]
 OtherElementStartsInNextColumn[6,3] OtherElementStartsInNextColumn[6,4]
 OtherElementStartsInNextColumn[6,5] OtherElementStartsInNextColumn[6,6]
 OtherElementStartsInNextColumn[6,7] OtherElementStartsInNextColumn[6,8]
 OtherElementStartsInNextColumn[6,9] OtherElementStartsInNextColumn[6,10]
 OtherElementStartsInNextColumn[7,0] OtherElementStartsInNextColumn[7,1]
 OtherElementStartsInNextColumn[7,2] OtherElementStartsInNextColumn[7,3]
 OtherElementStartsInNextColumn[7,4] OtherElementStartsInNextColumn[7,5]
 OtherElementStartsInNextColumn[7,6] OtherElementStartsInNextColumn[7,7]
 OtherElementStartsInNextColumn[7,8] OtherElementStartsInNextColumn[7,9]
 OtherElementStartsInNextColumn[7,10] OtherElementStartsInNextColumn[8,0]
 OtherElementStartsInNextColumn[8,1] OtherElementStartsInNextColumn[8,2]
 OtherElementStartsInNextColumn[8,3] OtherElementStartsInNextColumn[8,4]
 OtherElementStartsInNextColumn[8,5] OtherElementStartsInNextColumn[8,6]
 OtherElementStartsInNextColumn[8,7] OtherElementStartsInNextColumn[8,8]
 OtherElementStartsInNextColumn[8,9] OtherElementStartsInNextColumn[8,10]
 OtherElementStartsInNextColumn[9,0] OtherElementStartsInNextColumn[9,1]
 OtherElementStartsInNextColumn[9,2] OtherElementStartsInNextColumn[9,3]
 OtherElementStartsInNextColumn[9,4] OtherElementStartsInNextColumn[9,5]
 OtherElementStartsInNextColumn[9,6] OtherElementStartsInNextColumn[9,7]
 OtherElementStartsInNextColumn[9,8] OtherElementStartsInNextColumn[9,9]
 OtherElementStartsInNextColumn[9,10] OtherElementStartsInNextColumn[10,0]
 OtherElementStartsInNextColumn[10,1] OtherElementStartsInNextColumn[10,2]
 OtherElementStartsInNextColumn[10,3] OtherElementStartsInNextColumn[10,4]
 OtherElementStartsInNextColumn[10,5] OtherElementStartsInNextColumn[10,6]
 OtherElementStartsInNextColumn[10,7] OtherElementStartsInNextColumn[10,8]
 OtherElementStartsInNextColumn[10,9] OtherElementStartsInNextColumn[10,10]
 OtherElementEndsInNextColumn[0,0] OtherElementEndsInNextColumn[0,1]
 OtherElementEndsInNextColumn[0,2] OtherElementEndsInNextColumn[0,3]
 OtherElementEndsInNextColumn[0,4] OtherElementEndsInNextColumn[0,5]
 OtherElementEndsInNextColumn[0,6] OtherElementEndsInNextColumn[0,7]
 OtherElementEndsInNextColumn[0,8] OtherElementEndsInNextColumn[0,9]
 OtherElementEndsInNextColumn[0,10] OtherElementEndsInNextColumn[1,0]
 OtherElementEndsInNextColumn[1,1] OtherElementEndsInNextColumn[1,2]
 OtherElementEndsInNextColumn[1,3] OtherElementEndsInNextColumn[1,4]
 OtherElementEndsInNextColumn[1,5] OtherElementEndsInNextColumn[1,6]
 OtherElementEndsInNextColumn[1,7] OtherElementEndsInNextColumn[1,8]
 OtherElementEndsInNextColumn[1,9] OtherElementEndsInNextColumn[1,10]
 OtherElementEndsInNextColumn[2,0] OtherElementEndsInNextColumn[2,1]
 OtherElementEndsInNextColumn[2,2] OtherElementEndsInNextColumn[2,3]
 OtherElementEndsInNextColumn[2,4] OtherElementEndsInNextColumn[2,5]
 OtherElementEndsInNextColumn[2,6] OtherElementEndsInNextColumn[2,7]
 OtherElementEndsInNextColumn[2,8] OtherElementEndsInNextColumn[2,9]
 OtherElementEndsInNextColumn[2,10] OtherElementEndsInNextColumn[3,0]
 OtherElementEndsInNextColumn[3,1] OtherElementEndsInNextColumn[3,2]
 OtherElementEndsInNextColumn[3,3] OtherElementEndsInNextColumn[3,4]
 OtherElementEndsInNextColumn[3,5] OtherElementEndsInNextColumn[3,6]
 OtherElementEndsInNextColumn[3,7] OtherElementEndsInNextColumn[3,8]
 OtherElementEndsInNextColumn[3,9] OtherElementEndsInNextColumn[3,10]
 OtherElementEndsInNextColumn[4,0] OtherElementEndsInNextColumn[4,1]
 OtherElementEndsInNextColumn[4,2] OtherElementEndsInNextColumn[4,3]
 OtherElementEndsInNextColumn[4,4] OtherElementEndsInNextColumn[4,5]
 OtherElementEndsInNextColumn[4,6] OtherElementEndsInNextColumn[4,7]
 OtherElementEndsInNextColumn[4,8] OtherElementEndsInNextColumn[4,9]
 OtherElementEndsInNextColumn[4,10] OtherElementEndsInNextColumn[5,0]
 OtherElementEndsInNextColumn[5,1] OtherElementEndsInNextColumn[5,2]
 OtherElementEndsInNextColumn[5,3] OtherElementEndsInNextColumn[5,4]
 OtherElementEndsInNextColumn[5,5] OtherElementEndsInNextColumn[5,6]
 OtherElementEndsInNextColumn[5,7] OtherElementEndsInNextColumn[5,8]
 OtherElementEndsInNextColumn[5,9] OtherElementEndsInNextColumn[5,10]
 OtherElementEndsInNextColumn[6,0] OtherElementEndsInNextColumn[6,1]
 OtherElementEndsInNextColumn[6,2] OtherElementEndsInNextColumn[6,3]
 OtherElementEndsInNextColumn[6,4] OtherElementEndsInNextColumn[6,5]
 OtherElementEndsInNextColumn[6,6] OtherElementEndsInNextColumn[6,7]
 OtherElementEndsInNextColumn[6,8] OtherElementEndsInNextColumn[6,9]
 OtherElementEndsInNextColumn[6,10] OtherElementEndsInNextColumn[7,0]
 OtherElementEndsInNextColumn[7,1] OtherElementEndsInNextColumn[7,2]
 OtherElementEndsInNextColumn[7,3] OtherElementEndsInNextColumn[7,4]
 OtherElementEndsInNextColumn[7,5] OtherElementEndsInNextColumn[7,6]
 OtherElementEndsInNextColumn[7,7] OtherElementEndsInNextColumn[7,8]
 OtherElementEndsInNextColumn[7,9] OtherElementEndsInNextColumn[7,10]
 OtherElementEndsInNextColumn[8,0] OtherElementEndsInNextColumn[8,1]
 OtherElementEndsInNextColumn[8,2] OtherElementEndsInNextColumn[8,3]
 OtherElementEndsInNextColumn[8,4] OtherElementEndsInNextColumn[8,5]
 OtherElementEndsInNextColumn[8,6] OtherElementEndsInNextColumn[8,7]
 OtherElementEndsInNextColumn[8,8] OtherElementEndsInNextColumn[8,9]
 OtherElementEndsInNextColumn[8,10] OtherElementEndsInNextColumn[9,0]
 OtherElementEndsInNextColumn[9,1] OtherElementEndsInNextColumn[9,2]
 OtherElementEndsInNextColumn[9,3] OtherElementEndsInNextColumn[9,4]
 OtherElementEndsInNextColumn[9,5] OtherElementEndsInNextColumn[9,6]
 OtherElementEndsInNextColumn[9,7] OtherElementEndsInNextColumn[9,8]
 OtherElementEndsInNextColumn[9,9] OtherElementEndsInNextColumn[9,10]
 OtherElementEndsInNextColumn[10,0] OtherElementEndsInNextColumn[10,1]
 OtherElementEndsInNextColumn[10,2] OtherElementEndsInNextColumn[10,3]
 OtherElementEndsInNextColumn[10,4] OtherElementEndsInNextColumn[10,5]
 OtherElementEndsInNextColumn[10,6] OtherElementEndsInNextColumn[10,7]
 OtherElementEndsInNextColumn[10,8] OtherElementEndsInNextColumn[10,9]
 OtherElementEndsInNextColumn[10,10]
Generals
 X0[0] X0[1] X0[2] X0[3] X0[4] X0[5] X0[6] X0[7] X0[8] X0[9] X0[10] Y0[0]
 Y0[1] Y0[2] Y0[3] Y0[4] Y0[5] Y0[6] Y0[7] Y0[8] Y0[9] Y0[10] X1[0] X1[1]
 X1[2] X1[3] X1[4] X1[5] X1[6] X1[7] X1[8] X1[9] X1[10] Y1[0] Y1[1] Y1[2]
 Y1[3] Y1[4] Y1[5] Y1[6] Y1[7] Y1[8] Y1[9] Y1[10] W[0] W[1] W[2] W[3] W[4]
 W[5] W[6] W[7] W[8] W[9] W[10] H[0] H[1] H[2] H[3] H[4] H[5] H[6] H[7]
 H[8] H[9] H[10] ResizeW[0] ResizeW[1] ResizeW[2] ResizeW[3] ResizeW[4]
 ResizeW[5] ResizeW[6] ResizeW[7] ResizeW[8] ResizeW[9] ResizeW[10]
 ResizeWAbs[0] ResizeWAbs[1] ResizeWAbs[2] ResizeWAbs[3] ResizeWAbs[4]
 ResizeWAbs[5] ResizeWAbs[6] ResizeWAbs[7] ResizeWAbs[8] ResizeWAbs[9]
 ResizeWAbs[10] ResizeH[0] ResizeH[1] ResizeH[2] ResizeH[3] ResizeH[4]
 ResizeH[5] ResizeH[6] ResizeH[7] ResizeH[8] ResizeH[9] ResizeH[10]
 ResizeHAbs[0] ResizeHAbs[1] ResizeHAbs[2] ResizeHAbs[3] ResizeHAbs[4]
 ResizeHAbs[5] ResizeHAbs[6] ResizeHAbs[7] ResizeHAbs[8] ResizeHAbs[9]
 ResizeHAbs[10] MoveX[0] MoveX[1] MoveX[2] MoveX[3] MoveX[4] MoveX[5]
 MoveX[6] MoveX[7] MoveX[8] MoveX[9] MoveX[10] MoveXAbs[0] MoveXAbs[1]
 MoveXAbs[2] MoveXAbs[3] MoveXAbs[4] MoveXAbs[5] MoveXAbs[6] MoveXAbs[7]
 MoveXAbs[8] MoveXAbs[9] MoveXAbs[10] MoveY[0] MoveY[1] MoveY[2] MoveY[3]
 MoveY[4] MoveY[5] MoveY[6] MoveY[7] MoveY[8] MoveY[9] MoveY[10]
 MoveYAbs[0] MoveYAbs[1] MoveYAbs[2] MoveYAbs[3] MoveYAbs[4] MoveYAbs[5]
 MoveYAbs[6] MoveYAbs[7] MoveYAbs[8] MoveYAbs[9] MoveYAbs[10]
 EdgeDistance[x0,0,0] EdgeDistance[x0,0,1] EdgeDistance[x0,0,2]
 EdgeDistance[x0,0,3] EdgeDistance[x0,0,4] EdgeDistance[x0,0,5]
 EdgeDistance[x0,0,6] EdgeDistance[x0,0,7] EdgeDistance[x0,0,8]
 EdgeDistance[x0,0,9] EdgeDistance[x0,0,10] EdgeDistance[x0,1,0]
 EdgeDistance[x0,1,1] EdgeDistance[x0,1,2] EdgeDistance[x0,1,3]
 EdgeDistance[x0,1,4] EdgeDistance[x0,1,5] EdgeDistance[x0,1,6]
 EdgeDistance[x0,1,7] EdgeDistance[x0,1,8] EdgeDistance[x0,1,9]
 EdgeDistance[x0,1,10] EdgeDistance[x0,2,0] EdgeDistance[x0,2,1]
 EdgeDistance[x0,2,2] EdgeDistance[x0,2,3] EdgeDistance[x0,2,4]
 EdgeDistance[x0,2,5] EdgeDistance[x0,2,6] EdgeDistance[x0,2,7]
 EdgeDistance[x0,2,8] EdgeDistance[x0,2,9] EdgeDistance[x0,2,10]
 EdgeDistance[x0,3,0] EdgeDistance[x0,3,1] EdgeDistance[x0,3,2]
 EdgeDistance[x0,3,3] EdgeDistance[x0,3,4] EdgeDistance[x0,3,5]
 EdgeDistance[x0,3,6] EdgeDistance[x0,3,7] EdgeDistance[x0,3,8]
 EdgeDistance[x0,3,9] EdgeDistance[x0,3,10] EdgeDistance[x0,4,0]
 EdgeDistance[x0,4,1] EdgeDistance[x0,4,2] EdgeDistance[x0,4,3]
 EdgeDistance[x0,4,4] EdgeDistance[x0,4,5] EdgeDistance[x0,4,6]
 EdgeDistance[x0,4,7] EdgeDistance[x0,4,8] EdgeDistance[x0,4,9]
 EdgeDistance[x0,4,10] EdgeDistance[x0,5,0] EdgeDistance[x0,5,1]
 EdgeDistance[x0,5,2] EdgeDistance[x0,5,3] EdgeDistance[x0,5,4]
 EdgeDistance[x0,5,5] EdgeDistance[x0,5,6] EdgeDistance[x0,5,7]
 EdgeDistance[x0,5,8] EdgeDistance[x0,5,9] EdgeDistance[x0,5,10]
 EdgeDistance[x0,6,0] EdgeDistance[x0,6,1] EdgeDistance[x0,6,2]
 EdgeDistance[x0,6,3] EdgeDistance[x0,6,4] EdgeDistance[x0,6,5]
 EdgeDistance[x0,6,6] EdgeDistance[x0,6,7] EdgeDistance[x0,6,8]
 EdgeDistance[x0,6,9] EdgeDistance[x0,6,10] EdgeDistance[x0,7,0]
 EdgeDistance[x0,7,1] EdgeDistance[x0,7,2] EdgeDistance[x0,7,3]
 EdgeDistance[x0,7,4] EdgeDistance[x0,7,5] EdgeDistance[x0,7,6]
 EdgeDistance[x0,7,7] EdgeDistance[x0,7,8] EdgeDistance[x0,7,9]
 EdgeDistance[x0,7,10] EdgeDistance[x0,8,0] EdgeDistance[x0,8,1]
 EdgeDistance[x0,8,2] EdgeDistance[x0,8,3] EdgeDistance[x0,8,4]
 EdgeDistance[x0,8,5] EdgeDistance[x0,8,6] EdgeDistance[x0,8,7]
 EdgeDistance[x0,8,8] EdgeDistance[x0,8,9] EdgeDistance[x0,8,10]
 EdgeDistance[x0,9,0] EdgeDistance[x0,9,1] EdgeDistance[x0,9,2]
 EdgeDistance[x0,9,3] EdgeDistance[x0,9,4] EdgeDistance[x0,9,5]
 EdgeDistance[x0,9,6] EdgeDistance[x0,9,7] EdgeDistance[x0,9,8]
 EdgeDistance[x0,9,9] EdgeDistance[x0,9,10] EdgeDistance[x0,10,0]
 EdgeDistance[x0,10,1] EdgeDistance[x0,10,2] EdgeDistance[x0,10,3]
 EdgeDistance[x0,10,4] EdgeDistance[x0,10,5] EdgeDistance[x0,10,6]
 EdgeDistance[x0,10,7] EdgeDistance[x0,10,8] EdgeDistance[x0,10,9]
 EdgeDistance[x0,10,10] EdgeDistance[y0,0,0] EdgeDistance[y0,0,1]
 EdgeDistance[y0,0,2] EdgeDistance[y0,0,3] EdgeDistance[y0,0,4]
 EdgeDistance[y0,0,5] EdgeDistance[y0,0,6] EdgeDistance[y0,0,7]
 EdgeDistance[y0,0,8] EdgeDistance[y0,0,9] EdgeDistance[y0,0,10]
 EdgeDistance[y0,1,0] EdgeDistance[y0,1,1] EdgeDistance[y0,1,2]
 EdgeDistance[y0,1,3] EdgeDistance[y0,1,4] EdgeDistance[y0,1,5]
 EdgeDistance[y0,1,6] EdgeDistance[y0,1,7] EdgeDistance[y0,1,8]
 EdgeDistance[y0,1,9] EdgeDistance[y0,1,10] EdgeDistance[y0,2,0]
 EdgeDistance[y0,2,1] EdgeDistance[y0,2,2] EdgeDistance[y0,2,3]
 EdgeDistance[y0,2,4] EdgeDistance[y0,2,5] EdgeDistance[y0,2,6]
 EdgeDistance[y0,2,7] EdgeDistance[y0,2,8] EdgeDistance[y0,2,9]
 EdgeDistance[y0,2,10] EdgeDistance[y0,3,0] EdgeDistance[y0,3,1]
 EdgeDistance[y0,3,2] EdgeDistance[y0,3,3] EdgeDistance[y0,3,4]
 EdgeDistance[y0,3,5] EdgeDistance[y0,3,6] EdgeDistance[y0,3,7]
 EdgeDistance[y0,3,8] EdgeDistance[y0,3,9] EdgeDistance[y0,3,10]
 EdgeDistance[y0,4,0] EdgeDistance[y0,4,1] EdgeDistance[y0,4,2]
 EdgeDistance[y0,4,3] EdgeDistance[y0,4,4] EdgeDistance[y0,4,5]
 EdgeDistance[y0,4,6] EdgeDistance[y0,4,7] EdgeDistance[y0,4,8]
 EdgeDistance[y0,4,9] EdgeDistance[y0,4,10] EdgeDistance[y0,5,0]
 EdgeDistance[y0,5,1] EdgeDistance[y0,5,2] EdgeDistance[y0,5,3]
 EdgeDistance[y0,5,4] EdgeDistance[y0,5,5] EdgeDistance[y0,5,6]
 EdgeDistance[y0,5,7] EdgeDistance[y0,5,8] EdgeDistance[y0,5,9]
 EdgeDistance[y0,5,10] EdgeDistance[y0,6,0] EdgeDistance[y0,6,1]
 EdgeDistance[y0,6,2] EdgeDistance[y0,6,3] EdgeDistance[y0,6,4]
 EdgeDistance[y0,6,5] EdgeDistance[y0,6,6] EdgeDistance[y0,6,7]
 EdgeDistance[y0,6,8] EdgeDistance[y0,6,9] EdgeDistance[y0,6,10]
 EdgeDistance[y0,7,0] EdgeDistance[y0,7,1] EdgeDistance[y0,7,2]
 EdgeDistance[y0,7,3] EdgeDistance[y0,7,4] EdgeDistance[y0,7,5]
 EdgeDistance[y0,7,6] EdgeDistance[y0,7,7] EdgeDistance[y0,7,8]
 EdgeDistance[y0,7,9] EdgeDistance[y0,7,10] EdgeDistance[y0,8,0]
 EdgeDistance[y0,8,1] EdgeDistance[y0,8,2] EdgeDistance[y0,8,3]
 EdgeDistance[y0,8,4] EdgeDistance[y0,8,5] EdgeDistance[y0,8,6]
 EdgeDistance[y0,8,7] EdgeDistance[y0,8,8] EdgeDistance[y0,8,9]
 EdgeDistance[y0,8,10] EdgeDistance[y0,9,0] EdgeDistance[y0,9,1]
 EdgeDistance[y0,9,2] EdgeDistance[y0,9,3] EdgeDistance[y0,9,4]
 EdgeDistance[y0,9,5] EdgeDistance[y0,9,6] EdgeDistance[y0,9,7]
 EdgeDistance[y0,9,8] EdgeDistance[y0,9,9] EdgeDistance[y0,9,10]
 EdgeDistance[y0,10,0] EdgeDistance[y0,10,1] EdgeDistance[y0,10,2]
 EdgeDistance[y0,10,3] EdgeDistance[y0,10,4] EdgeDistance[y0,10,5]
 EdgeDistance[y0,10,6] EdgeDistance[y0,10,7] EdgeDistance[y0,10,8]
 EdgeDistance[y0,10,9] EdgeDistance[y0,10,10] EdgeDistance[x1,0,0]
 EdgeDistance[x1,0,1] EdgeDistance[x1,0,2] EdgeDistance[x1,0,3]
 EdgeDistance[x1,0,4] EdgeDistance[x1,0,5] EdgeDistance[x1,0,6]
 EdgeDistance[x1,0,7] EdgeDistance[x1,0,8] EdgeDistance[x1,0,9]
 EdgeDistance[x1,0,10] EdgeDistance[x1,1,0] EdgeDistance[x1,1,1]
 EdgeDistance[x1,1,2] EdgeDistance[x1,1,3] EdgeDistance[x1,1,4]
 EdgeDistance[x1,1,5] EdgeDistance[x1,1,6] EdgeDistance[x1,1,7]
 EdgeDistance[x1,1,8] EdgeDistance[x1,1,9] EdgeDistance[x1,1,10]
 EdgeDistance[x1,2,0] EdgeDistance[x1,2,1] EdgeDistance[x1,2,2]
 EdgeDistance[x1,2,3] EdgeDistance[x1,2,4] EdgeDistance[x1,2,5]
 EdgeDistance[x1,2,6] EdgeDistance[x1,2,7] EdgeDistance[x1,2,8]
 EdgeDistance[x1,2,9] EdgeDistance[x1,2,10] EdgeDistance[x1,3,0]
 EdgeDistance[x1,3,1] EdgeDistance[x1,3,2] EdgeDistance[x1,3,3]
 EdgeDistance[x1,3,4] EdgeDistance[x1,3,5] EdgeDistance[x1,3,6]
 EdgeDistance[x1,3,7] EdgeDistance[x1,3,8] EdgeDistance[x1,3,9]
 EdgeDistance[x1,3,10] EdgeDistance[x1,4,0] EdgeDistance[x1,4,1]
 EdgeDistance[x1,4,2] EdgeDistance[x1,4,3] EdgeDistance[x1,4,4]
 EdgeDistance[x1,4,5] EdgeDistance[x1,4,6] EdgeDistance[x1,4,7]
 EdgeDistance[x1,4,8] EdgeDistance[x1,4,9] EdgeDistance[x1,4,10]
 EdgeDistance[x1,5,0] EdgeDistance[x1,5,1] EdgeDistance[x1,5,2]
 EdgeDistance[x1,5,3] EdgeDistance[x1,5,4] EdgeDistance[x1,5,5]
 EdgeDistance[x1,5,6] EdgeDistance[x1,5,7] EdgeDistance[x1,5,8]
 EdgeDistance[x1,5,9] EdgeDistance[x1,5,10] EdgeDistance[x1,6,0]
 EdgeDistance[x1,6,1] EdgeDistance[x1,6,2] EdgeDistance[x1,6,3]
 EdgeDistance[x1,6,4] EdgeDistance[x1,6,5] EdgeDistance[x1,6,6]
 EdgeDistance[x1,6,7] EdgeDistance[x1,6,8] EdgeDistance[x1,6,9]
 EdgeDistance[x1,6,10] EdgeDistance[x1,7,0] EdgeDistance[x1,7,1]
 EdgeDistance[x1,7,2] EdgeDistance[x1,7,3] EdgeDistance[x1,7,4]
 EdgeDistance[x1,7,5] EdgeDistance[x1,7,6] EdgeDistance[x1,7,7]
 EdgeDistance[x1,7,8] EdgeDistance[x1,7,9] EdgeDistance[x1,7,10]
 EdgeDistance[x1,8,0] EdgeDistance[x1,8,1] EdgeDistance[x1,8,2]
 EdgeDistance[x1,8,3] EdgeDistance[x1,8,4] EdgeDistance[x1,8,5]
 EdgeDistance[x1,8,6] EdgeDistance[x1,8,7] EdgeDistance[x1,8,8]
 EdgeDistance[x1,8,9] EdgeDistance[x1,8,10] EdgeDistance[x1,9,0]
 EdgeDistance[x1,9,1] EdgeDistance[x1,9,2] EdgeDistance[x1,9,3]
 EdgeDistance[x1,9,4] EdgeDistance[x1,9,5] EdgeDistance[x1,9,6]
 EdgeDistance[x1,9,7] EdgeDistance[x1,9,8] EdgeDistance[x1,9,9]
 EdgeDistance[x1,9,10] EdgeDistance[x1,10,0] EdgeDistance[x1,10,1]
 EdgeDistance[x1,10,2] EdgeDistance[x1,10,3] EdgeDistance[x1,10,4]
 EdgeDistance[x1,10,5] EdgeDistance[x1,10,6] EdgeDistance[x1,10,7]
 EdgeDistance[x1,10,8] EdgeDistance[x1,10,9] EdgeDistance[x1,10,10]
 EdgeDistance[y1,0,0] EdgeDistance[y1,0,1] EdgeDistance[y1,0,2]
 EdgeDistance[y1,0,3] EdgeDistance[y1,0,4] EdgeDistance[y1,0,5]
 EdgeDistance[y1,0,6] EdgeDistance[y1,0,7] EdgeDistance[y1,0,8]
 EdgeDistance[y1,0,9] EdgeDistance[y1,0,10] EdgeDistance[y1,1,0]
 EdgeDistance[y1,1,1] EdgeDistance[y1,1,2] EdgeDistance[y1,1,3]
 EdgeDistance[y1,1,4] EdgeDistance[y1,1,5] EdgeDistance[y1,1,6]
 EdgeDistance[y1,1,7] EdgeDistance[y1,1,8] EdgeDistance[y1,1,9]
 EdgeDistance[y1,1,10] EdgeDistance[y1,2,0] EdgeDistance[y1,2,1]
 EdgeDistance[y1,2,2] EdgeDistance[y1,2,3] EdgeDistance[y1,2,4]
 EdgeDistance[y1,2,5] EdgeDistance[y1,2,6] EdgeDistance[y1,2,7]
 EdgeDistance[y1,2,8] EdgeDistance[y1,2,9] EdgeDistance[y1,2,10]
 EdgeDistance[y1,3,0] EdgeDistance[y1,3,1] EdgeDistance[y1,3,2]
 EdgeDistance[y1,3,3] EdgeDistance[y1,3,4] EdgeDistance[y1,3,5]
 EdgeDistance[y1,3,6] EdgeDistance[y1,3,7] EdgeDistance[y1,3,8]
 EdgeDistance[y1,3,9] EdgeDistance[y1,3,10] EdgeDistance[y1,4,0]
 EdgeDistance[y1,4,1] EdgeDistance[y1,4,2] EdgeDistance[y1,4,3]
 EdgeDistance[y1,4,4] EdgeDistance[y1,4,5] EdgeDistance[y1,4,6]
 EdgeDistance[y1,4,7] EdgeDistance[y1,4,8] EdgeDistance[y1,4,9]
 EdgeDistance[y1,4,10] EdgeDistance[y1,5,0] EdgeDistance[y1,5,1]
 EdgeDistance[y1,5,2] EdgeDistance[y1,5,3] EdgeDistance[y1,5,4]
 EdgeDistance[y1,5,5] EdgeDistance[y1,5,6] EdgeDistance[y1,5,7]
 EdgeDistance[y1,5,8] EdgeDistance[y1,5,9] EdgeDistance[y1,5,10]
 EdgeDistance[y1,6,0] EdgeDistance[y1,6,1] EdgeDistance[y1,6,2]
 EdgeDistance[y1,6,3] EdgeDistance[y1,6,4] EdgeDistance[y1,6,5]
 EdgeDistance[y1,6,6] EdgeDistance[y1,6,7] EdgeDistance[y1,6,8]
 EdgeDistance[y1,6,9] EdgeDistance[y1,6,10] EdgeDistance[y1,7,0]
 EdgeDistance[y1,7,1] EdgeDistance[y1,7,2] EdgeDistance[y1,7,3]
 EdgeDistance[y1,7,4] EdgeDistance[y1,7,5] EdgeDistance[y1,7,6]
 EdgeDistance[y1,7,7] EdgeDistance[y1,7,8] EdgeDistance[y1,7,9]
 EdgeDistance[y1,7,10] EdgeDistance[y1,8,0] EdgeDistance[y1,8,1]
 EdgeDistance[y1,8,2] EdgeDistance[y1,8,3] EdgeDistance[y1,8,4]
 EdgeDistance[y1,8,5] EdgeDistance[y1,8,6] EdgeDistance[y1,8,7]
 EdgeDistance[y1,8,8] EdgeDistance[y1,8,9] EdgeDistance[y1,8,10]
 EdgeDistance[y1,9,0] EdgeDistance[y1,9,1] EdgeDistance[y1,9,2]
 EdgeDistance[y1,9,3] EdgeDistance[y1,9,4] EdgeDistance[y1,9,5]
 EdgeDistance[y1,9,6] EdgeDistance[y1,9,7] EdgeDistance[y1,9,8]
 EdgeDistance[y1,9,9] EdgeDistance[y1,9,10] EdgeDistance[y1,10,0]
 EdgeDistance[y1,10,1] EdgeDistance[y1,10,2] EdgeDistance[y1,10,3]
 EdgeDistance[y1,10,4] EdgeDistance[y1,10,5] EdgeDistance[y1,10,6]
 EdgeDistance[y1,10,7] EdgeDistance[y1,10,8] EdgeDistance[y1,10,9]
 EdgeDistance[y1,10,10] X0Diff[0,0] X0Diff[0,1] X0Diff[0,2] X0Diff[0,3]
 X0Diff[0,4] X0Diff[0,5] X0Diff[0,6] X0Diff[0,7] X0Diff[0,8] X0Diff[0,9]
 X0Diff[0,10] X0Diff[1,0] X0Diff[1,1] X0Diff[1,2] X0Diff[1,3] X0Diff[1,4]
 X0Diff[1,5] X0Diff[1,6] X0Diff[1,7] X0Diff[1,8] X0Diff[1,9] X0Diff[1,10]
 X0Diff[2,0] X0Diff[2,1] X0Diff[2,2] X0Diff[2,3] X0Diff[2,4] X0Diff[2,5]
 X0Diff[2,6] X0Diff[2,7] X0Diff[2,8] X0Diff[2,9] X0Diff[2,10] X0Diff[3,0]
 X0Diff[3,1] X0Diff[3,2] X0Diff[3,3] X0Diff[3,4] X0Diff[3,5] X0Diff[3,6]
 X0Diff[3,7] X0Diff[3,8] X0Diff[3,9] X0Diff[3,10] X0Diff[4,0] X0Diff[4,1]
 X0Diff[4,2] X0Diff[4,3] X0Diff[4,4] X0Diff[4,5] X0Diff[4,6] X0Diff[4,7]
 X0Diff[4,8] X0Diff[4,9] X0Diff[4,10] X0Diff[5,0] X0Diff[5,1] X0Diff[5,2]
 X0Diff[5,3] X0Diff[5,4] X0Diff[5,5] X0Diff[5,6] X0Diff[5,7] X0Diff[5,8]
 X0Diff[5,9] X0Diff[5,10] X0Diff[6,0] X0Diff[6,1] X0Diff[6,2] X0Diff[6,3]
 X0Diff[6,4] X0Diff[6,5] X0Diff[6,6] X0Diff[6,7] X0Diff[6,8] X0Diff[6,9]
 X0Diff[6,10] X0Diff[7,0] X0Diff[7,1] X0Diff[7,2] X0Diff[7,3] X0Diff[7,4]
 X0Diff[7,5] X0Diff[7,6] X0Diff[7,7] X0Diff[7,8] X0Diff[7,9] X0Diff[7,10]
 X0Diff[8,0] X0Diff[8,1] X0Diff[8,2] X0Diff[8,3] X0Diff[8,4] X0Diff[8,5]
 X0Diff[8,6] X0Diff[8,7] X0Diff[8,8] X0Diff[8,9] X0Diff[8,10] X0Diff[9,0]
 X0Diff[9,1] X0Diff[9,2] X0Diff[9,3] X0Diff[9,4] X0Diff[9,5] X0Diff[9,6]
 X0Diff[9,7] X0Diff[9,8] X0Diff[9,9] X0Diff[9,10] X0Diff[10,0] X0Diff[10,1]
 X0Diff[10,2] X0Diff[10,3] X0Diff[10,4] X0Diff[10,5] X0Diff[10,6]
 X0Diff[10,7] X0Diff[10,8] X0Diff[10,9] X0Diff[10,10] Y0Diff[0,0]
 Y0Diff[0,1] Y0Diff[0,2] Y0Diff[0,3] Y0Diff[0,4] Y0Diff[0,5] Y0Diff[0,6]
 Y0Diff[0,7] Y0Diff[0,8] Y0Diff[0,9] Y0Diff[0,10] Y0Diff[1,0] Y0Diff[1,1]
 Y0Diff[1,2] Y0Diff[1,3] Y0Diff[1,4] Y0Diff[1,5] Y0Diff[1,6] Y0Diff[1,7]
 Y0Diff[1,8] Y0Diff[1,9] Y0Diff[1,10] Y0Diff[2,0] Y0Diff[2,1] Y0Diff[2,2]
 Y0Diff[2,3] Y0Diff[2,4] Y0Diff[2,5] Y0Diff[2,6] Y0Diff[2,7] Y0Diff[2,8]
 Y0Diff[2,9] Y0Diff[2,10] Y0Diff[3,0] Y0Diff[3,1] Y0Diff[3,2] Y0Diff[3,3]
 Y0Diff[3,4] Y0Diff[3,5] Y0Diff[3,6] Y0Diff[3,7] Y0Diff[3,8] Y0Diff[3,9]
 Y0Diff[3,10] Y0Diff[4,0] Y0Diff[4,1] Y0Diff[4,2] Y0Diff[4,3] Y0Diff[4,4]
 Y0Diff[4,5] Y0Diff[4,6] Y0Diff[4,7] Y0Diff[4,8] Y0Diff[4,9] Y0Diff[4,10]
 Y0Diff[5,0] Y0Diff[5,1] Y0Diff[5,2] Y0Diff[5,3] Y0Diff[5,4] Y0Diff[5,5]
 Y0Diff[5,6] Y0Diff[5,7] Y0Diff[5,8] Y0Diff[5,9] Y0Diff[5,10] Y0Diff[6,0]
 Y0Diff[6,1] Y0Diff[6,2] Y0Diff[6,3] Y0Diff[6,4] Y0Diff[6,5] Y0Diff[6,6]
 Y0Diff[6,7] Y0Diff[6,8] Y0Diff[6,9] Y0Diff[6,10] Y0Diff[7,0] Y0Diff[7,1]
 Y0Diff[7,2] Y0Diff[7,3] Y0Diff[7,4] Y0Diff[7,5] Y0Diff[7,6] Y0Diff[7,7]
 Y0Diff[7,8] Y0Diff[7,9] Y0Diff[7,10] Y0Diff[8,0] Y0Diff[8,1] Y0Diff[8,2]
 Y0Diff[8,3] Y0Diff[8,4] Y0Diff[8,5] Y0Diff[8,6] Y0Diff[8,7] Y0Diff[8,8]
 Y0Diff[8,9] Y0Diff[8,10] Y0Diff[9,0] Y0Diff[9,1] Y0Diff[9,2] Y0Diff[9,3]
 Y0Diff[9,4] Y0Diff[9,5] Y0Diff[9,6] Y0Diff[9,7] Y0Diff[9,8] Y0Diff[9,9]
 Y0Diff[9,10] Y0Diff[10,0] Y0Diff[10,1] Y0Diff[10,2] Y0Diff[10,3]
 Y0Diff[10,4] Y0Diff[10,5] Y0Diff[10,6] Y0Diff[10,7] Y0Diff[10,8]
 Y0Diff[10,9] Y0Diff[10,10] X1Diff[0,0] X1Diff[0,1] X1Diff[0,2] X1Diff[0,3]
 X1Diff[0,4] X1Diff[0,5] X1Diff[0,6] X1Diff[0,7] X1Diff[0,8] X1Diff[0,9]
 X1Diff[0,10] X1Diff[1,0] X1Diff[1,1] X1Diff[1,2] X1Diff[1,3] X1Diff[1,4]
 X1Diff[1,5] X1Diff[1,6] X1Diff[1,7] X1Diff[1,8] X1Diff[1,9] X1Diff[1,10]
 X1Diff[2,0] X1Diff[2,1] X1Diff[2,2] X1Diff[2,3] X1Diff[2,4] X1Diff[2,5]
 X1Diff[2,6] X1Diff[2,7] X1Diff[2,8] X1Diff[2,9] X1Diff[2,10] X1Diff[3,0]
 X1Diff[3,1] X1Diff[3,2] X1Diff[3,3] X1Diff[3,4] X1Diff[3,5] X1Diff[3,6]
 X1Diff[3,7] X1Diff[3,8] X1Diff[3,9] X1Diff[3,10] X1Diff[4,0] X1Diff[4,1]
 X1Diff[4,2] X1Diff[4,3] X1Diff[4,4] X1Diff[4,5] X1Diff[4,6] X1Diff[4,7]
 X1Diff[4,8] X1Diff[4,9] X1Diff[4,10] X1Diff[5,0] X1Diff[5,1] X1Diff[5,2]
 X1Diff[5,3] X1Diff[5,4] X1Diff[5,5] X1Diff[5,6] X1Diff[5,7] X1Diff[5,8]
 X1Diff[5,9] X1Diff[5,10] X1Diff[6,0] X1Diff[6,1] X1Diff[6,2] X1Diff[6,3]
 X1Diff[6,4] X1Diff[6,5] X1Diff[6,6] X1Diff[6,7] X1Diff[6,8] X1Diff[6,9]
 X1Diff[6,10] X1Diff[7,0] X1Diff[7,1] X1Diff[7,2] X1Diff[7,3] X1Diff[7,4]
 X1Diff[7,5] X1Diff[7,6] X1Diff[7,7] X1Diff[7,8] X1Diff[7,9] X1Diff[7,10]
 X1Diff[8,0] X1Diff[8,1] X1Diff[8,2] X1Diff[8,3] X1Diff[8,4] X1Diff[8,5]
 X1Diff[8,6] X1Diff[8,7] X1Diff[8,8] X1Diff[8,9] X1Diff[8,10] X1Diff[9,0]
 X1Diff[9,1] X1Diff[9,2] X1Diff[9,3] X1Diff[9,4] X1Diff[9,5] X1Diff[9,6]
 X1Diff[9,7] X1Diff[9,8] X1Diff[9,9] X1Diff[9,10] X1Diff[10,0] X1Diff[10,1]
 X1Diff[10,2] X1Diff[10,3] X1Diff[10,4] X1Diff[10,5] X1Diff[10,6]
 X1Diff[10,7] X1Diff[10,8] X1Diff[10,9] X1Diff[10,10] Y1Diff[0,0]
 Y1Diff[0,1] Y1Diff[0,2] Y1Diff[0,3] Y1Diff[0,4] Y1Diff[0,5] Y1Diff[0,6]
 Y1Diff[0,7] Y1Diff[0,8] Y1Diff[0,9] Y1Diff[0,10] Y1Diff[1,0] Y1Diff[1,1]
 Y1Diff[1,2] Y1Diff[1,3] Y1Diff[1,4] Y1Diff[1,5] Y1Diff[1,6] Y1Diff[1,7]
 Y1Diff[1,8] Y1Diff[1,9] Y1Diff[1,10] Y1Diff[2,0] Y1Diff[2,1] Y1Diff[2,2]
 Y1Diff[2,3] Y1Diff[2,4] Y1Diff[2,5] Y1Diff[2,6] Y1Diff[2,7] Y1Diff[2,8]
 Y1Diff[2,9] Y1Diff[2,10] Y1Diff[3,0] Y1Diff[3,1] Y1Diff[3,2] Y1Diff[3,3]
 Y1Diff[3,4] Y1Diff[3,5] Y1Diff[3,6] Y1Diff[3,7] Y1Diff[3,8] Y1Diff[3,9]
 Y1Diff[3,10] Y1Diff[4,0] Y1Diff[4,1] Y1Diff[4,2] Y1Diff[4,3] Y1Diff[4,4]
 Y1Diff[4,5] Y1Diff[4,6] Y1Diff[4,7] Y1Diff[4,8] Y1Diff[4,9] Y1Diff[4,10]
 Y1Diff[5,0] Y1Diff[5,1] Y1Diff[5,2] Y1Diff[5,3] Y1Diff[5,4] Y1Diff[5,5]
 Y1Diff[5,6] Y1Diff[5,7] Y1Diff[5,8] Y1Diff[5,9] Y1Diff[5,10] Y1Diff[6,0]
 Y1Diff[6,1] Y1Diff[6,2] Y1Diff[6,3] Y1Diff[6,4] Y1Diff[6,5] Y1Diff[6,6]
 Y1Diff[6,7] Y1Diff[6,8] Y1Diff[6,9] Y1Diff[6,10] Y1Diff[7,0] Y1Diff[7,1]
 Y1Diff[7,2] Y1Diff[7,3] Y1Diff[7,4] Y1Diff[7,5] Y1Diff[7,6] Y1Diff[7,7]
 Y1Diff[7,8] Y1Diff[7,9] Y1Diff[7,10] Y1Diff[8,0] Y1Diff[8,1] Y1Diff[8,2]
 Y1Diff[8,3] Y1Diff[8,4] Y1Diff[8,5] Y1Diff[8,6] Y1Diff[8,7] Y1Diff[8,8]
 Y1Diff[8,9] Y1Diff[8,10] Y1Diff[9,0] Y1Diff[9,1] Y1Diff[9,2] Y1Diff[9,3]
 Y1Diff[9,4] Y1Diff[9,5] Y1Diff[9,6] Y1Diff[9,7] Y1Diff[9,8] Y1Diff[9,9]
 Y1Diff[9,10] Y1Diff[10,0] Y1Diff[10,1] Y1Diff[10,2] Y1Diff[10,3]
 Y1Diff[10,4] Y1Diff[10,5] Y1Diff[10,6] Y1Diff[10,7] Y1Diff[10,8]
 Y1Diff[10,9] Y1Diff[10,10] X0Group[0] X0Group[1] X0Group[2] X0Group[3]
 X0Group[4] X0Group[5] X0Group[6] X0Group[7] X0Group[8] X0Group[9]
 X0Group[10] Y0Group[0] Y0Group[1] Y0Group[2] Y0Group[3] Y0Group[4]
 Y0Group[5] Y0Group[6] Y0Group[7] Y0Group[8] Y0Group[9] Y0Group[10]
 X1Group[0] X1Group[1] X1Group[2] X1Group[3] X1Group[4] X1Group[5]
 X1Group[6] X1Group[7] X1Group[8] X1Group[9] X1Group[10] Y1Group[0]
 Y1Group[1] Y1Group[2] Y1Group[3] Y1Group[4] Y1Group[5] Y1Group[6]
 Y1Group[7] Y1Group[8] Y1Group[9] Y1Group[10] X0GroupCount Y0GroupCount
 X1GroupCount Y1GroupCount LeftMargin MaxX1 MaxX1 MarginDiffLooseAbs C2252
 StartColumn[0] StartColumn[1] StartColumn[2] StartColumn[3] StartColumn[4]
 StartColumn[5] StartColumn[6] StartColumn[7] StartColumn[8] StartColumn[9]
 StartColumn[10] EndColumn[0] EndColumn[1] EndColumn[2] EndColumn[3]
 EndColumn[4] EndColumn[5] EndColumn[6] EndColumn[7] EndColumn[8]
 EndColumn[9] EndColumn[10] ColumnCount FirstColumn
General Constraints
 LinkResizeWAbs[0]: ResizeWAbs[0] = ABS ( ResizeW[0] )
 LinkResizeWAbs[1]: ResizeWAbs[1] = ABS ( ResizeW[1] )
 LinkResizeWAbs[2]: ResizeWAbs[2] = ABS ( ResizeW[2] )
 LinkResizeWAbs[3]: ResizeWAbs[3] = ABS ( ResizeW[3] )
 LinkResizeWAbs[4]: ResizeWAbs[4] = ABS ( ResizeW[4] )
 LinkResizeWAbs[5]: ResizeWAbs[5] = ABS ( ResizeW[5] )
 LinkResizeWAbs[6]: ResizeWAbs[6] = ABS ( ResizeW[6] )
 LinkResizeWAbs[7]: ResizeWAbs[7] = ABS ( ResizeW[7] )
 LinkResizeWAbs[8]: ResizeWAbs[8] = ABS ( ResizeW[8] )
 LinkResizeWAbs[9]: ResizeWAbs[9] = ABS ( ResizeW[9] )
 LinkResizeWAbs[10]: ResizeWAbs[10] = ABS ( ResizeW[10] )
 LinkResizeHAbs[0]: ResizeHAbs[0] = ABS ( ResizeH[0] )
 LinkResizeHAbs[1]: ResizeHAbs[1] = ABS ( ResizeH[1] )
 LinkResizeHAbs[2]: ResizeHAbs[2] = ABS ( ResizeH[2] )
 LinkResizeHAbs[3]: ResizeHAbs[3] = ABS ( ResizeH[3] )
 LinkResizeHAbs[4]: ResizeHAbs[4] = ABS ( ResizeH[4] )
 LinkResizeHAbs[5]: ResizeHAbs[5] = ABS ( ResizeH[5] )
 LinkResizeHAbs[6]: ResizeHAbs[6] = ABS ( ResizeH[6] )
 LinkResizeHAbs[7]: ResizeHAbs[7] = ABS ( ResizeH[7] )
 LinkResizeHAbs[8]: ResizeHAbs[8] = ABS ( ResizeH[8] )
 LinkResizeHAbs[9]: ResizeHAbs[9] = ABS ( ResizeH[9] )
 LinkResizeHAbs[10]: ResizeHAbs[10] = ABS ( ResizeH[10] )
 LinkMoveXAbs[0]: MoveXAbs[0] = ABS ( MoveX[0] )
 LinkMoveXAbs[1]: MoveXAbs[1] = ABS ( MoveX[1] )
 LinkMoveXAbs[2]: MoveXAbs[2] = ABS ( MoveX[2] )
 LinkMoveXAbs[3]: MoveXAbs[3] = ABS ( MoveX[3] )
 LinkMoveXAbs[4]: MoveXAbs[4] = ABS ( MoveX[4] )
 LinkMoveXAbs[5]: MoveXAbs[5] = ABS ( MoveX[5] )
 LinkMoveXAbs[6]: MoveXAbs[6] = ABS ( MoveX[6] )
 LinkMoveXAbs[7]: MoveXAbs[7] = ABS ( MoveX[7] )
 LinkMoveXAbs[8]: MoveXAbs[8] = ABS ( MoveX[8] )
 LinkMoveXAbs[9]: MoveXAbs[9] = ABS ( MoveX[9] )
 LinkMoveXAbs[10]: MoveXAbs[10] = ABS ( MoveX[10] )
 LinkMoveYAbs[0]: MoveYAbs[0] = ABS ( MoveY[0] )
 LinkMoveYAbs[1]: MoveYAbs[1] = ABS ( MoveY[1] )
 LinkMoveYAbs[2]: MoveYAbs[2] = ABS ( MoveY[2] )
 LinkMoveYAbs[3]: MoveYAbs[3] = ABS ( MoveY[3] )
 LinkMoveYAbs[4]: MoveYAbs[4] = ABS ( MoveY[4] )
 LinkMoveYAbs[5]: MoveYAbs[5] = ABS ( MoveY[5] )
 LinkMoveYAbs[6]: MoveYAbs[6] = ABS ( MoveY[6] )
 LinkMoveYAbs[7]: MoveYAbs[7] = ABS ( MoveY[7] )
 LinkMoveYAbs[8]: MoveYAbs[8] = ABS ( MoveY[8] )
 LinkMoveYAbs[9]: MoveYAbs[9] = ABS ( MoveY[9] )
 LinkMoveYAbs[10]: MoveYAbs[10] = ABS ( MoveY[10] )
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
 LinkOverlap[0,8]: Overlap[0,8] = AND ( HorizontalOverlap[0,8] ,
   VerticalOverlap[0,8] )
 LinkOverlap[0,9]: Overlap[0,9] = AND ( HorizontalOverlap[0,9] ,
   VerticalOverlap[0,9] )
 LinkOverlap[0,10]: Overlap[0,10] = AND ( HorizontalOverlap[0,10] ,
   VerticalOverlap[0,10] )
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
 LinkOverlap[1,8]: Overlap[1,8] = AND ( HorizontalOverlap[1,8] ,
   VerticalOverlap[1,8] )
 LinkOverlap[1,9]: Overlap[1,9] = AND ( HorizontalOverlap[1,9] ,
   VerticalOverlap[1,9] )
 LinkOverlap[1,10]: Overlap[1,10] = AND ( HorizontalOverlap[1,10] ,
   VerticalOverlap[1,10] )
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
 LinkOverlap[2,8]: Overlap[2,8] = AND ( HorizontalOverlap[2,8] ,
   VerticalOverlap[2,8] )
 LinkOverlap[2,9]: Overlap[2,9] = AND ( HorizontalOverlap[2,9] ,
   VerticalOverlap[2,9] )
 LinkOverlap[2,10]: Overlap[2,10] = AND ( HorizontalOverlap[2,10] ,
   VerticalOverlap[2,10] )
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
 LinkOverlap[3,8]: Overlap[3,8] = AND ( HorizontalOverlap[3,8] ,
   VerticalOverlap[3,8] )
 LinkOverlap[3,9]: Overlap[3,9] = AND ( HorizontalOverlap[3,9] ,
   VerticalOverlap[3,9] )
 LinkOverlap[3,10]: Overlap[3,10] = AND ( HorizontalOverlap[3,10] ,
   VerticalOverlap[3,10] )
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
 LinkOverlap[4,8]: Overlap[4,8] = AND ( HorizontalOverlap[4,8] ,
   VerticalOverlap[4,8] )
 LinkOverlap[4,9]: Overlap[4,9] = AND ( HorizontalOverlap[4,9] ,
   VerticalOverlap[4,9] )
 LinkOverlap[4,10]: Overlap[4,10] = AND ( HorizontalOverlap[4,10] ,
   VerticalOverlap[4,10] )
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
 LinkOverlap[5,8]: Overlap[5,8] = AND ( HorizontalOverlap[5,8] ,
   VerticalOverlap[5,8] )
 LinkOverlap[5,9]: Overlap[5,9] = AND ( HorizontalOverlap[5,9] ,
   VerticalOverlap[5,9] )
 LinkOverlap[5,10]: Overlap[5,10] = AND ( HorizontalOverlap[5,10] ,
   VerticalOverlap[5,10] )
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
 LinkOverlap[6,8]: Overlap[6,8] = AND ( HorizontalOverlap[6,8] ,
   VerticalOverlap[6,8] )
 LinkOverlap[6,9]: Overlap[6,9] = AND ( HorizontalOverlap[6,9] ,
   VerticalOverlap[6,9] )
 LinkOverlap[6,10]: Overlap[6,10] = AND ( HorizontalOverlap[6,10] ,
   VerticalOverlap[6,10] )
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
 LinkOverlap[7,8]: Overlap[7,8] = AND ( HorizontalOverlap[7,8] ,
   VerticalOverlap[7,8] )
 LinkOverlap[7,9]: Overlap[7,9] = AND ( HorizontalOverlap[7,9] ,
   VerticalOverlap[7,9] )
 LinkOverlap[7,10]: Overlap[7,10] = AND ( HorizontalOverlap[7,10] ,
   VerticalOverlap[7,10] )
 LinkOverlap[8,0]: Overlap[8,0] = AND ( HorizontalOverlap[8,0] ,
   VerticalOverlap[8,0] )
 LinkOverlap[8,1]: Overlap[8,1] = AND ( HorizontalOverlap[8,1] ,
   VerticalOverlap[8,1] )
 LinkOverlap[8,2]: Overlap[8,2] = AND ( HorizontalOverlap[8,2] ,
   VerticalOverlap[8,2] )
 LinkOverlap[8,3]: Overlap[8,3] = AND ( HorizontalOverlap[8,3] ,
   VerticalOverlap[8,3] )
 LinkOverlap[8,4]: Overlap[8,4] = AND ( HorizontalOverlap[8,4] ,
   VerticalOverlap[8,4] )
 LinkOverlap[8,5]: Overlap[8,5] = AND ( HorizontalOverlap[8,5] ,
   VerticalOverlap[8,5] )
 LinkOverlap[8,6]: Overlap[8,6] = AND ( HorizontalOverlap[8,6] ,
   VerticalOverlap[8,6] )
 LinkOverlap[8,7]: Overlap[8,7] = AND ( HorizontalOverlap[8,7] ,
   VerticalOverlap[8,7] )
 LinkOverlap[8,9]: Overlap[8,9] = AND ( HorizontalOverlap[8,9] ,
   VerticalOverlap[8,9] )
 LinkOverlap[8,10]: Overlap[8,10] = AND ( HorizontalOverlap[8,10] ,
   VerticalOverlap[8,10] )
 LinkOverlap[9,0]: Overlap[9,0] = AND ( HorizontalOverlap[9,0] ,
   VerticalOverlap[9,0] )
 LinkOverlap[9,1]: Overlap[9,1] = AND ( HorizontalOverlap[9,1] ,
   VerticalOverlap[9,1] )
 LinkOverlap[9,2]: Overlap[9,2] = AND ( HorizontalOverlap[9,2] ,
   VerticalOverlap[9,2] )
 LinkOverlap[9,3]: Overlap[9,3] = AND ( HorizontalOverlap[9,3] ,
   VerticalOverlap[9,3] )
 LinkOverlap[9,4]: Overlap[9,4] = AND ( HorizontalOverlap[9,4] ,
   VerticalOverlap[9,4] )
 LinkOverlap[9,5]: Overlap[9,5] = AND ( HorizontalOverlap[9,5] ,
   VerticalOverlap[9,5] )
 LinkOverlap[9,6]: Overlap[9,6] = AND ( HorizontalOverlap[9,6] ,
   VerticalOverlap[9,6] )
 LinkOverlap[9,7]: Overlap[9,7] = AND ( HorizontalOverlap[9,7] ,
   VerticalOverlap[9,7] )
 LinkOverlap[9,8]: Overlap[9,8] = AND ( HorizontalOverlap[9,8] ,
   VerticalOverlap[9,8] )
 LinkOverlap[9,10]: Overlap[9,10] = AND ( HorizontalOverlap[9,10] ,
   VerticalOverlap[9,10] )
 LinkOverlap[10,0]: Overlap[10,0] = AND ( HorizontalOverlap[10,0] ,
   VerticalOverlap[10,0] )
 LinkOverlap[10,1]: Overlap[10,1] = AND ( HorizontalOverlap[10,1] ,
   VerticalOverlap[10,1] )
 LinkOverlap[10,2]: Overlap[10,2] = AND ( HorizontalOverlap[10,2] ,
   VerticalOverlap[10,2] )
 LinkOverlap[10,3]: Overlap[10,3] = AND ( HorizontalOverlap[10,3] ,
   VerticalOverlap[10,3] )
 LinkOverlap[10,4]: Overlap[10,4] = AND ( HorizontalOverlap[10,4] ,
   VerticalOverlap[10,4] )
 LinkOverlap[10,5]: Overlap[10,5] = AND ( HorizontalOverlap[10,5] ,
   VerticalOverlap[10,5] )
 LinkOverlap[10,6]: Overlap[10,6] = AND ( HorizontalOverlap[10,6] ,
   VerticalOverlap[10,6] )
 LinkOverlap[10,7]: Overlap[10,7] = AND ( HorizontalOverlap[10,7] ,
   VerticalOverlap[10,7] )
 LinkOverlap[10,8]: Overlap[10,8] = AND ( HorizontalOverlap[10,8] ,
   VerticalOverlap[10,8] )
 LinkOverlap[10,9]: Overlap[10,9] = AND ( HorizontalOverlap[10,9] ,
   VerticalOverlap[10,9] )
 GC2442: X0GroupCount = MAX ( X0Group[0] , X0Group[1] , X0Group[2] ,
   X0Group[3] , X0Group[4] , X0Group[5] , X0Group[6] , X0Group[7] ,
   X0Group[8] , X0Group[9] , X0Group[10] )
 GC2443: Y0GroupCount = MAX ( Y0Group[0] , Y0Group[1] , Y0Group[2] ,
   Y0Group[3] , Y0Group[4] , Y0Group[5] , Y0Group[6] , Y0Group[7] ,
   Y0Group[8] , Y0Group[9] , Y0Group[10] )
 GC2444: X1GroupCount = MAX ( X1Group[0] , X1Group[1] , X1Group[2] ,
   X1Group[3] , X1Group[4] , X1Group[5] , X1Group[6] , X1Group[7] ,
   X1Group[8] , X1Group[9] , X1Group[10] )
 GC2445: Y1GroupCount = MAX ( Y1Group[0] , Y1Group[1] , Y1Group[2] ,
   Y1Group[3] , Y1Group[4] , Y1Group[5] , Y1Group[6] , Y1Group[7] ,
   Y1Group[8] , Y1Group[9] , Y1Group[10] )
 LinkLeftMargin: LeftMargin = MIN ( X0[0] , X0[1] , X0[2] , X0[3] , X0[4] ,
   X0[5] , X0[6] , X0[7] , X0[8] , X0[9] , X0[10] )
 LinkMaxX1: MaxX1 = MAX ( X1[0] , X1[1] , X1[2] , X1[3] , X1[4] , X1[5] ,
   X1[6] , X1[7] , X1[8] , X1[9] , X1[10] )
 GC3790: ColumnCount = MAX ( StartColumn[0] , StartColumn[1] ,
   StartColumn[2] , StartColumn[3] , StartColumn[4] , StartColumn[5] ,
   StartColumn[6] , StartColumn[7] , StartColumn[8] , StartColumn[9] ,
   StartColumn[10] , EndColumn[0] , EndColumn[1] , EndColumn[2] ,
   EndColumn[3] , EndColumn[4] , EndColumn[5] , EndColumn[6] ,
   EndColumn[7] , EndColumn[8] , EndColumn[9] , EndColumn[10] )
 GC3791: FirstColumn = MIN ( StartColumn[0] , StartColumn[1] ,
   StartColumn[2] , StartColumn[3] , StartColumn[4] , StartColumn[5] ,
   StartColumn[6] , StartColumn[7] , StartColumn[8] , StartColumn[9] ,
   StartColumn[10] )
End
