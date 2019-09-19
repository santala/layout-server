\ Model GLayoutQuality
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.1 ResizeWAbs[0] + 0.5 ResizeWAbs[1] + 0.25 ResizeWAbs[2]
   + 0.25 ResizeWAbs[3] + 0.25 ResizeWAbs[4] + 0.25 ResizeWAbs[5]
   + 0.25 ResizeWAbs[6] + 0.1 ResizeWAbs[7] + ResizeHAbs[0]
   + 0.1428571428571428 ResizeHAbs[1] + 0.3333333333333334 ResizeHAbs[2]
   + 0.3333333333333334 ResizeHAbs[3] + 0.25 ResizeHAbs[4]
   + 0.5 ResizeHAbs[5] + 0.5 ResizeHAbs[6]
   + 0.6666666666666667 ResizeHAbs[7] + 0.1 MoveXAbs[0] + 0.5 MoveXAbs[1]
   + 0.25 MoveXAbs[2] + 0.25 MoveXAbs[3] + 0.25 MoveXAbs[4]
   + 0.25 MoveXAbs[5] + 0.25 MoveXAbs[6] + 0.1 MoveXAbs[7] + MoveYAbs[0]
   + 0.1428571428571428 MoveYAbs[1] + 0.3333333333333334 MoveYAbs[2]
   + 0.3333333333333334 MoveYAbs[3] + 0.25 MoveYAbs[4] + 0.5 MoveYAbs[5]
   + 0.5 MoveYAbs[6] + 0.6666666666666667 MoveYAbs[7] + 0 Above[0,0]
   + 0 Above[1,1] + 0 Above[2,2] + 0 Above[3,3] + 0 Above[4,4]
   + 0 Above[5,5] + 0 Above[6,6] + 0 Above[7,7] + 0 OnLeft[0,0]
   + 0 OnLeft[1,1] + 0 OnLeft[2,2] + 0 OnLeft[3,3] + 0 OnLeft[4,4]
   + 0 OnLeft[5,5] + 0 OnLeft[6,6] + 0 OnLeft[7,7]
   + 0 HorizontalOverlap[0,0] + 0 HorizontalOverlap[1,1]
   + 0 HorizontalOverlap[2,2] + 0 HorizontalOverlap[3,3]
   + 0 HorizontalOverlap[4,4] + 0 HorizontalOverlap[5,5]
   + 0 HorizontalOverlap[6,6] + 0 HorizontalOverlap[7,7]
   + 0 VerticalOverlap[0,0] + 0 VerticalOverlap[1,1]
   + 0 VerticalOverlap[2,2] + 0 VerticalOverlap[3,3]
   + 0 VerticalOverlap[4,4] + 0 VerticalOverlap[5,5]
   + 0 VerticalOverlap[6,6] + 0 VerticalOverlap[7,7]
   + 0 EdgeDistanceAbs[x0,0,0] + 0 EdgeDistanceAbs[x0,0,1]
   + 0 EdgeDistanceAbs[x0,0,2] + 0 EdgeDistanceAbs[x0,0,3]
   + 0 EdgeDistanceAbs[x0,0,4] + 0 EdgeDistanceAbs[x0,0,5]
   + 0 EdgeDistanceAbs[x0,0,6] + 0 EdgeDistanceAbs[x0,0,7]
   + 0 EdgeDistanceAbs[x0,1,0] + 0 EdgeDistanceAbs[x0,1,1]
   + 0 EdgeDistanceAbs[x0,1,2] + 0 EdgeDistanceAbs[x0,1,3]
   + 0 EdgeDistanceAbs[x0,1,4] + 0 EdgeDistanceAbs[x0,1,5]
   + 0 EdgeDistanceAbs[x0,1,6] + 0 EdgeDistanceAbs[x0,1,7]
   + 0 EdgeDistanceAbs[x0,2,0] + 0 EdgeDistanceAbs[x0,2,1]
   + 0 EdgeDistanceAbs[x0,2,2] + 0 EdgeDistanceAbs[x0,2,3]
   + 0 EdgeDistanceAbs[x0,2,4] + 0 EdgeDistanceAbs[x0,2,5]
   + 0 EdgeDistanceAbs[x0,2,6] + 0 EdgeDistanceAbs[x0,2,7]
   + 0 EdgeDistanceAbs[x0,3,0] + 0 EdgeDistanceAbs[x0,3,1]
   + 0 EdgeDistanceAbs[x0,3,2] + 0 EdgeDistanceAbs[x0,3,3]
   + 0 EdgeDistanceAbs[x0,3,4] + 0 EdgeDistanceAbs[x0,3,5]
   + 0 EdgeDistanceAbs[x0,3,6] + 0 EdgeDistanceAbs[x0,3,7]
   + 0 EdgeDistanceAbs[x0,4,0] + 0 EdgeDistanceAbs[x0,4,1]
   + 0 EdgeDistanceAbs[x0,4,2] + 0 EdgeDistanceAbs[x0,4,3]
   + 0 EdgeDistanceAbs[x0,4,4] + 0 EdgeDistanceAbs[x0,4,5]
   + 0 EdgeDistanceAbs[x0,4,6] + 0 EdgeDistanceAbs[x0,4,7]
   + 0 EdgeDistanceAbs[x0,5,0] + 0 EdgeDistanceAbs[x0,5,1]
   + 0 EdgeDistanceAbs[x0,5,2] + 0 EdgeDistanceAbs[x0,5,3]
   + 0 EdgeDistanceAbs[x0,5,4] + 0 EdgeDistanceAbs[x0,5,5]
   + 0 EdgeDistanceAbs[x0,5,6] + 0 EdgeDistanceAbs[x0,5,7]
   + 0 EdgeDistanceAbs[x0,6,0] + 0 EdgeDistanceAbs[x0,6,1]
   + 0 EdgeDistanceAbs[x0,6,2] + 0 EdgeDistanceAbs[x0,6,3]
   + 0 EdgeDistanceAbs[x0,6,4] + 0 EdgeDistanceAbs[x0,6,5]
   + 0 EdgeDistanceAbs[x0,6,6] + 0 EdgeDistanceAbs[x0,6,7]
   + 0 EdgeDistanceAbs[x0,7,0] + 0 EdgeDistanceAbs[x0,7,1]
   + 0 EdgeDistanceAbs[x0,7,2] + 0 EdgeDistanceAbs[x0,7,3]
   + 0 EdgeDistanceAbs[x0,7,4] + 0 EdgeDistanceAbs[x0,7,5]
   + 0 EdgeDistanceAbs[x0,7,6] + 0 EdgeDistanceAbs[x0,7,7]
   + 0 EdgeDistanceAbs[y0,0,0] + 0 EdgeDistanceAbs[y0,0,1]
   + 0 EdgeDistanceAbs[y0,0,2] + 0 EdgeDistanceAbs[y0,0,3]
   + 0 EdgeDistanceAbs[y0,0,4] + 0 EdgeDistanceAbs[y0,0,5]
   + 0 EdgeDistanceAbs[y0,0,6] + 0 EdgeDistanceAbs[y0,0,7]
   + 0 EdgeDistanceAbs[y0,1,0] + 0 EdgeDistanceAbs[y0,1,1]
   + 0 EdgeDistanceAbs[y0,1,2] + 0 EdgeDistanceAbs[y0,1,3]
   + 0 EdgeDistanceAbs[y0,1,4] + 0 EdgeDistanceAbs[y0,1,5]
   + 0 EdgeDistanceAbs[y0,1,6] + 0 EdgeDistanceAbs[y0,1,7]
   + 0 EdgeDistanceAbs[y0,2,0] + 0 EdgeDistanceAbs[y0,2,1]
   + 0 EdgeDistanceAbs[y0,2,2] + 0 EdgeDistanceAbs[y0,2,3]
   + 0 EdgeDistanceAbs[y0,2,4] + 0 EdgeDistanceAbs[y0,2,5]
   + 0 EdgeDistanceAbs[y0,2,6] + 0 EdgeDistanceAbs[y0,2,7]
   + 0 EdgeDistanceAbs[y0,3,0] + 0 EdgeDistanceAbs[y0,3,1]
   + 0 EdgeDistanceAbs[y0,3,2] + 0 EdgeDistanceAbs[y0,3,3]
   + 0 EdgeDistanceAbs[y0,3,4] + 0 EdgeDistanceAbs[y0,3,5]
   + 0 EdgeDistanceAbs[y0,3,6] + 0 EdgeDistanceAbs[y0,3,7]
   + 0 EdgeDistanceAbs[y0,4,0] + 0 EdgeDistanceAbs[y0,4,1]
   + 0 EdgeDistanceAbs[y0,4,2] + 0 EdgeDistanceAbs[y0,4,3]
   + 0 EdgeDistanceAbs[y0,4,4] + 0 EdgeDistanceAbs[y0,4,5]
   + 0 EdgeDistanceAbs[y0,4,6] + 0 EdgeDistanceAbs[y0,4,7]
   + 0 EdgeDistanceAbs[y0,5,0] + 0 EdgeDistanceAbs[y0,5,1]
   + 0 EdgeDistanceAbs[y0,5,2] + 0 EdgeDistanceAbs[y0,5,3]
   + 0 EdgeDistanceAbs[y0,5,4] + 0 EdgeDistanceAbs[y0,5,5]
   + 0 EdgeDistanceAbs[y0,5,6] + 0 EdgeDistanceAbs[y0,5,7]
   + 0 EdgeDistanceAbs[y0,6,0] + 0 EdgeDistanceAbs[y0,6,1]
   + 0 EdgeDistanceAbs[y0,6,2] + 0 EdgeDistanceAbs[y0,6,3]
   + 0 EdgeDistanceAbs[y0,6,4] + 0 EdgeDistanceAbs[y0,6,5]
   + 0 EdgeDistanceAbs[y0,6,6] + 0 EdgeDistanceAbs[y0,6,7]
   + 0 EdgeDistanceAbs[y0,7,0] + 0 EdgeDistanceAbs[y0,7,1]
   + 0 EdgeDistanceAbs[y0,7,2] + 0 EdgeDistanceAbs[y0,7,3]
   + 0 EdgeDistanceAbs[y0,7,4] + 0 EdgeDistanceAbs[y0,7,5]
   + 0 EdgeDistanceAbs[y0,7,6] + 0 EdgeDistanceAbs[y0,7,7]
   + 0 EdgeDistanceAbs[x1,0,0] + 0 EdgeDistanceAbs[x1,0,1]
   + 0 EdgeDistanceAbs[x1,0,2] + 0 EdgeDistanceAbs[x1,0,3]
   + 0 EdgeDistanceAbs[x1,0,4] + 0 EdgeDistanceAbs[x1,0,5]
   + 0 EdgeDistanceAbs[x1,0,6] + 0 EdgeDistanceAbs[x1,0,7]
   + 0 EdgeDistanceAbs[x1,1,0] + 0 EdgeDistanceAbs[x1,1,1]
   + 0 EdgeDistanceAbs[x1,1,2] + 0 EdgeDistanceAbs[x1,1,3]
   + 0 EdgeDistanceAbs[x1,1,4] + 0 EdgeDistanceAbs[x1,1,5]
   + 0 EdgeDistanceAbs[x1,1,6] + 0 EdgeDistanceAbs[x1,1,7]
   + 0 EdgeDistanceAbs[x1,2,0] + 0 EdgeDistanceAbs[x1,2,1]
   + 0 EdgeDistanceAbs[x1,2,2] + 0 EdgeDistanceAbs[x1,2,3]
   + 0 EdgeDistanceAbs[x1,2,4] + 0 EdgeDistanceAbs[x1,2,5]
   + 0 EdgeDistanceAbs[x1,2,6] + 0 EdgeDistanceAbs[x1,2,7]
   + 0 EdgeDistanceAbs[x1,3,0] + 0 EdgeDistanceAbs[x1,3,1]
   + 0 EdgeDistanceAbs[x1,3,2] + 0 EdgeDistanceAbs[x1,3,3]
   + 0 EdgeDistanceAbs[x1,3,4] + 0 EdgeDistanceAbs[x1,3,5]
   + 0 EdgeDistanceAbs[x1,3,6] + 0 EdgeDistanceAbs[x1,3,7]
   + 0 EdgeDistanceAbs[x1,4,0] + 0 EdgeDistanceAbs[x1,4,1]
   + 0 EdgeDistanceAbs[x1,4,2] + 0 EdgeDistanceAbs[x1,4,3]
   + 0 EdgeDistanceAbs[x1,4,4] + 0 EdgeDistanceAbs[x1,4,5]
   + 0 EdgeDistanceAbs[x1,4,6] + 0 EdgeDistanceAbs[x1,4,7]
   + 0 EdgeDistanceAbs[x1,5,0] + 0 EdgeDistanceAbs[x1,5,1]
   + 0 EdgeDistanceAbs[x1,5,2] + 0 EdgeDistanceAbs[x1,5,3]
   + 0 EdgeDistanceAbs[x1,5,4] + 0 EdgeDistanceAbs[x1,5,5]
   + 0 EdgeDistanceAbs[x1,5,6] + 0 EdgeDistanceAbs[x1,5,7]
   + 0 EdgeDistanceAbs[x1,6,0] + 0 EdgeDistanceAbs[x1,6,1]
   + 0 EdgeDistanceAbs[x1,6,2] + 0 EdgeDistanceAbs[x1,6,3]
   + 0 EdgeDistanceAbs[x1,6,4] + 0 EdgeDistanceAbs[x1,6,5]
   + 0 EdgeDistanceAbs[x1,6,6] + 0 EdgeDistanceAbs[x1,6,7]
   + 0 EdgeDistanceAbs[x1,7,0] + 0 EdgeDistanceAbs[x1,7,1]
   + 0 EdgeDistanceAbs[x1,7,2] + 0 EdgeDistanceAbs[x1,7,3]
   + 0 EdgeDistanceAbs[x1,7,4] + 0 EdgeDistanceAbs[x1,7,5]
   + 0 EdgeDistanceAbs[x1,7,6] + 0 EdgeDistanceAbs[x1,7,7]
   + 0 EdgeDistanceAbs[y1,0,0] + 0 EdgeDistanceAbs[y1,0,1]
   + 0 EdgeDistanceAbs[y1,0,2] + 0 EdgeDistanceAbs[y1,0,3]
   + 0 EdgeDistanceAbs[y1,0,4] + 0 EdgeDistanceAbs[y1,0,5]
   + 0 EdgeDistanceAbs[y1,0,6] + 0 EdgeDistanceAbs[y1,0,7]
   + 0 EdgeDistanceAbs[y1,1,0] + 0 EdgeDistanceAbs[y1,1,1]
   + 0 EdgeDistanceAbs[y1,1,2] + 0 EdgeDistanceAbs[y1,1,3]
   + 0 EdgeDistanceAbs[y1,1,4] + 0 EdgeDistanceAbs[y1,1,5]
   + 0 EdgeDistanceAbs[y1,1,6] + 0 EdgeDistanceAbs[y1,1,7]
   + 0 EdgeDistanceAbs[y1,2,0] + 0 EdgeDistanceAbs[y1,2,1]
   + 0 EdgeDistanceAbs[y1,2,2] + 0 EdgeDistanceAbs[y1,2,3]
   + 0 EdgeDistanceAbs[y1,2,4] + 0 EdgeDistanceAbs[y1,2,5]
   + 0 EdgeDistanceAbs[y1,2,6] + 0 EdgeDistanceAbs[y1,2,7]
   + 0 EdgeDistanceAbs[y1,3,0] + 0 EdgeDistanceAbs[y1,3,1]
   + 0 EdgeDistanceAbs[y1,3,2] + 0 EdgeDistanceAbs[y1,3,3]
   + 0 EdgeDistanceAbs[y1,3,4] + 0 EdgeDistanceAbs[y1,3,5]
   + 0 EdgeDistanceAbs[y1,3,6] + 0 EdgeDistanceAbs[y1,3,7]
   + 0 EdgeDistanceAbs[y1,4,0] + 0 EdgeDistanceAbs[y1,4,1]
   + 0 EdgeDistanceAbs[y1,4,2] + 0 EdgeDistanceAbs[y1,4,3]
   + 0 EdgeDistanceAbs[y1,4,4] + 0 EdgeDistanceAbs[y1,4,5]
   + 0 EdgeDistanceAbs[y1,4,6] + 0 EdgeDistanceAbs[y1,4,7]
   + 0 EdgeDistanceAbs[y1,5,0] + 0 EdgeDistanceAbs[y1,5,1]
   + 0 EdgeDistanceAbs[y1,5,2] + 0 EdgeDistanceAbs[y1,5,3]
   + 0 EdgeDistanceAbs[y1,5,4] + 0 EdgeDistanceAbs[y1,5,5]
   + 0 EdgeDistanceAbs[y1,5,6] + 0 EdgeDistanceAbs[y1,5,7]
   + 0 EdgeDistanceAbs[y1,6,0] + 0 EdgeDistanceAbs[y1,6,1]
   + 0 EdgeDistanceAbs[y1,6,2] + 0 EdgeDistanceAbs[y1,6,3]
   + 0 EdgeDistanceAbs[y1,6,4] + 0 EdgeDistanceAbs[y1,6,5]
   + 0 EdgeDistanceAbs[y1,6,6] + 0 EdgeDistanceAbs[y1,6,7]
   + 0 EdgeDistanceAbs[y1,7,0] + 0 EdgeDistanceAbs[y1,7,1]
   + 0 EdgeDistanceAbs[y1,7,2] + 0 EdgeDistanceAbs[y1,7,3]
   + 0 EdgeDistanceAbs[y1,7,4] + 0 EdgeDistanceAbs[y1,7,5]
   + 0 EdgeDistanceAbs[y1,7,6] + 0 EdgeDistanceAbs[y1,7,7]
   + 0 EdgesAlign[x0,0,0] + 0 EdgesAlign[x0,1,0] + 0 EdgesAlign[x0,1,1]
   + 0 EdgesAlign[x0,2,0] + 0 EdgesAlign[x0,2,1] + 0 EdgesAlign[x0,2,2]
   + 0 EdgesAlign[x0,3,0] + 0 EdgesAlign[x0,3,1] + 0 EdgesAlign[x0,3,2]
   + 0 EdgesAlign[x0,3,3] + 0 EdgesAlign[x0,4,0] + 0 EdgesAlign[x0,4,1]
   + 0 EdgesAlign[x0,4,2] + 0 EdgesAlign[x0,4,3] + 0 EdgesAlign[x0,4,4]
   + 0 EdgesAlign[x0,5,0] + 0 EdgesAlign[x0,5,1] + 0 EdgesAlign[x0,5,2]
   + 0 EdgesAlign[x0,5,3] + 0 EdgesAlign[x0,5,4] + 0 EdgesAlign[x0,5,5]
   + 0 EdgesAlign[x0,6,0] + 0 EdgesAlign[x0,6,1] + 0 EdgesAlign[x0,6,2]
   + 0 EdgesAlign[x0,6,3] + 0 EdgesAlign[x0,6,4] + 0 EdgesAlign[x0,6,5]
   + 0 EdgesAlign[x0,6,6] + 0 EdgesAlign[x0,7,0] + 0 EdgesAlign[x0,7,1]
   + 0 EdgesAlign[x0,7,2] + 0 EdgesAlign[x0,7,3] + 0 EdgesAlign[x0,7,4]
   + 0 EdgesAlign[x0,7,5] + 0 EdgesAlign[x0,7,6] + 0 EdgesAlign[x0,7,7]
   + 0 EdgesAlign[y0,0,0] + 0 EdgesAlign[y0,1,0] + 0 EdgesAlign[y0,1,1]
   + 0 EdgesAlign[y0,2,0] + 0 EdgesAlign[y0,2,1] + 0 EdgesAlign[y0,2,2]
   + 0 EdgesAlign[y0,3,0] + 0 EdgesAlign[y0,3,1] + 0 EdgesAlign[y0,3,2]
   + 0 EdgesAlign[y0,3,3] + 0 EdgesAlign[y0,4,0] + 0 EdgesAlign[y0,4,1]
   + 0 EdgesAlign[y0,4,2] + 0 EdgesAlign[y0,4,3] + 0 EdgesAlign[y0,4,4]
   + 0 EdgesAlign[y0,5,0] + 0 EdgesAlign[y0,5,1] + 0 EdgesAlign[y0,5,2]
   + 0 EdgesAlign[y0,5,3] + 0 EdgesAlign[y0,5,4] + 0 EdgesAlign[y0,5,5]
   + 0 EdgesAlign[y0,6,0] + 0 EdgesAlign[y0,6,1] + 0 EdgesAlign[y0,6,2]
   + 0 EdgesAlign[y0,6,3] + 0 EdgesAlign[y0,6,4] + 0 EdgesAlign[y0,6,5]
   + 0 EdgesAlign[y0,6,6] + 0 EdgesAlign[y0,7,0] + 0 EdgesAlign[y0,7,1]
   + 0 EdgesAlign[y0,7,2] + 0 EdgesAlign[y0,7,3] + 0 EdgesAlign[y0,7,4]
   + 0 EdgesAlign[y0,7,5] + 0 EdgesAlign[y0,7,6] + 0 EdgesAlign[y0,7,7]
   + 0 EdgesAlign[x1,0,0] + 0 EdgesAlign[x1,1,0] + 0 EdgesAlign[x1,1,1]
   + 0 EdgesAlign[x1,2,0] + 0 EdgesAlign[x1,2,1] + 0 EdgesAlign[x1,2,2]
   + 0 EdgesAlign[x1,3,0] + 0 EdgesAlign[x1,3,1] + 0 EdgesAlign[x1,3,2]
   + 0 EdgesAlign[x1,3,3] + 0 EdgesAlign[x1,4,0] + 0 EdgesAlign[x1,4,1]
   + 0 EdgesAlign[x1,4,2] + 0 EdgesAlign[x1,4,3] + 0 EdgesAlign[x1,4,4]
   + 0 EdgesAlign[x1,5,0] + 0 EdgesAlign[x1,5,1] + 0 EdgesAlign[x1,5,2]
   + 0 EdgesAlign[x1,5,3] + 0 EdgesAlign[x1,5,4] + 0 EdgesAlign[x1,5,5]
   + 0 EdgesAlign[x1,6,0] + 0 EdgesAlign[x1,6,1] + 0 EdgesAlign[x1,6,2]
   + 0 EdgesAlign[x1,6,3] + 0 EdgesAlign[x1,6,4] + 0 EdgesAlign[x1,6,5]
   + 0 EdgesAlign[x1,6,6] + 0 EdgesAlign[x1,7,0] + 0 EdgesAlign[x1,7,1]
   + 0 EdgesAlign[x1,7,2] + 0 EdgesAlign[x1,7,3] + 0 EdgesAlign[x1,7,4]
   + 0 EdgesAlign[x1,7,5] + 0 EdgesAlign[x1,7,6] + 0 EdgesAlign[x1,7,7]
   + 0 EdgesAlign[y1,0,0] + 0 EdgesAlign[y1,1,0] + 0 EdgesAlign[y1,1,1]
   + 0 EdgesAlign[y1,2,0] + 0 EdgesAlign[y1,2,1] + 0 EdgesAlign[y1,2,2]
   + 0 EdgesAlign[y1,3,0] + 0 EdgesAlign[y1,3,1] + 0 EdgesAlign[y1,3,2]
   + 0 EdgesAlign[y1,3,3] + 0 EdgesAlign[y1,4,0] + 0 EdgesAlign[y1,4,1]
   + 0 EdgesAlign[y1,4,2] + 0 EdgesAlign[y1,4,3] + 0 EdgesAlign[y1,4,4]
   + 0 EdgesAlign[y1,5,0] + 0 EdgesAlign[y1,5,1] + 0 EdgesAlign[y1,5,2]
   + 0 EdgesAlign[y1,5,3] + 0 EdgesAlign[y1,5,4] + 0 EdgesAlign[y1,5,5]
   + 0 EdgesAlign[y1,6,0] + 0 EdgesAlign[y1,6,1] + 0 EdgesAlign[y1,6,2]
   + 0 EdgesAlign[y1,6,3] + 0 EdgesAlign[y1,6,4] + 0 EdgesAlign[y1,6,5]
   + 0 EdgesAlign[y1,6,6] + 0 EdgesAlign[y1,7,0] + 0 EdgesAlign[y1,7,1]
   + 0 EdgesAlign[y1,7,2] + 0 EdgesAlign[y1,7,3] + 0 EdgesAlign[y1,7,4]
   + 0 EdgesAlign[y1,7,5] + 0 EdgesAlign[y1,7,6] + 0 EdgesAlign[y1,7,7]
   + NumberOfGroups[x0] + NumberOfGroups[y0] + NumberOfGroups[x1]
   + NumberOfGroups[y1]
Subject To
 X0X1Sanity[0]: EdgeCoord[x0,0] - EdgeCoord[x1,0] <= -1
 X0X1Sanity[1]: EdgeCoord[x0,1] - EdgeCoord[x1,1] <= -1
 X0X1Sanity[2]: EdgeCoord[x0,2] - EdgeCoord[x1,2] <= -1
 X0X1Sanity[3]: EdgeCoord[x0,3] - EdgeCoord[x1,3] <= -1
 X0X1Sanity[4]: EdgeCoord[x0,4] - EdgeCoord[x1,4] <= -1
 X0X1Sanity[5]: EdgeCoord[x0,5] - EdgeCoord[x1,5] <= -1
 X0X1Sanity[6]: EdgeCoord[x0,6] - EdgeCoord[x1,6] <= -1
 X0X1Sanity[7]: EdgeCoord[x0,7] - EdgeCoord[x1,7] <= -1
 Y0Y1Sanity[0]: EdgeCoord[y0,0] - EdgeCoord[y1,0] <= -1
 Y0Y1Sanity[1]: EdgeCoord[y0,1] - EdgeCoord[y1,1] <= -1
 Y0Y1Sanity[2]: EdgeCoord[y0,2] - EdgeCoord[y1,2] <= -1
 Y0Y1Sanity[3]: EdgeCoord[y0,3] - EdgeCoord[y1,3] <= -1
 Y0Y1Sanity[4]: EdgeCoord[y0,4] - EdgeCoord[y1,4] <= -1
 Y0Y1Sanity[5]: EdgeCoord[y0,5] - EdgeCoord[y1,5] <= -1
 Y0Y1Sanity[6]: EdgeCoord[y0,6] - EdgeCoord[y1,6] <= -1
 Y0Y1Sanity[7]: EdgeCoord[y0,7] - EdgeCoord[y1,7] <= -1
 LimitToCanvasWidth[0]: EdgeCoord[x1,0] <= 125
 LimitToCanvasWidth[1]: EdgeCoord[x1,1] <= 125
 LimitToCanvasWidth[2]: EdgeCoord[x1,2] <= 125
 LimitToCanvasWidth[3]: EdgeCoord[x1,3] <= 125
 LimitToCanvasWidth[4]: EdgeCoord[x1,4] <= 125
 LimitToCanvasWidth[5]: EdgeCoord[x1,5] <= 125
 LimitToCanvasWidth[6]: EdgeCoord[x1,6] <= 125
 LimitToCanvasWidth[7]: EdgeCoord[x1,7] <= 125
 LimitToCanvasHeight[0]: EdgeCoord[y1,0] <= 119
 LimitToCanvasHeight[1]: EdgeCoord[y1,1] <= 119
 LimitToCanvasHeight[2]: EdgeCoord[y1,2] <= 119
 LimitToCanvasHeight[3]: EdgeCoord[y1,3] <= 119
 LimitToCanvasHeight[4]: EdgeCoord[y1,4] <= 119
 LimitToCanvasHeight[5]: EdgeCoord[y1,5] <= 119
 LimitToCanvasHeight[6]: EdgeCoord[y1,6] <= 119
 LimitToCanvasHeight[7]: EdgeCoord[y1,7] <= 119
 X1-X0=W[0]: - EdgeCoord[x0,0] + EdgeCoord[x1,0] - W[0] = 0
 X1-X0=W[1]: - EdgeCoord[x0,1] + EdgeCoord[x1,1] - W[1] = 0
 X1-X0=W[2]: - EdgeCoord[x0,2] + EdgeCoord[x1,2] - W[2] = 0
 X1-X0=W[3]: - EdgeCoord[x0,3] + EdgeCoord[x1,3] - W[3] = 0
 X1-X0=W[4]: - EdgeCoord[x0,4] + EdgeCoord[x1,4] - W[4] = 0
 X1-X0=W[5]: - EdgeCoord[x0,5] + EdgeCoord[x1,5] - W[5] = 0
 X1-X0=W[6]: - EdgeCoord[x0,6] + EdgeCoord[x1,6] - W[6] = 0
 X1-X0=W[7]: - EdgeCoord[x0,7] + EdgeCoord[x1,7] - W[7] = 0
 Y1-Y0=H[0]: - EdgeCoord[y0,0] + EdgeCoord[y1,0] - H[0] = 0
 Y1-Y0=H[1]: - EdgeCoord[y0,1] + EdgeCoord[y1,1] - H[1] = 0
 Y1-Y0=H[2]: - EdgeCoord[y0,2] + EdgeCoord[y1,2] - H[2] = 0
 Y1-Y0=H[3]: - EdgeCoord[y0,3] + EdgeCoord[y1,3] - H[3] = 0
 Y1-Y0=H[4]: - EdgeCoord[y0,4] + EdgeCoord[y1,4] - H[4] = 0
 Y1-Y0=H[5]: - EdgeCoord[y0,5] + EdgeCoord[y1,5] - H[5] = 0
 Y1-Y0=H[6]: - EdgeCoord[y0,6] + EdgeCoord[y1,6] - H[6] = 0
 Y1-Y0=H[7]: - EdgeCoord[y0,7] + EdgeCoord[y1,7] - H[7] = 0
 LinkResizeW[0,<tools.JSONLoader.Element_object_at_0x107c2ae48>]: - W[0]
   + ResizeW[0] = -125
 LinkResizeW[1,<tools.JSONLoader.Element_object_at_0x107c2ae80>]: - W[1]
   + ResizeW[1] = -25
 LinkResizeW[2,<tools.JSONLoader.Element_object_at_0x107c2aeb8>]: - W[2]
   + ResizeW[2] = -50
 LinkResizeW[3,<tools.JSONLoader.Element_object_at_0x107c2aef0>]: - W[3]
   + ResizeW[3] = -50
 LinkResizeW[4,<tools.JSONLoader.Element_object_at_0x107c2af28>]: - W[4]
   + ResizeW[4] = -50
 LinkResizeW[5,<tools.JSONLoader.Element_object_at_0x107c2af60>]: - W[5]
   + ResizeW[5] = -50
 LinkResizeW[6,<tools.JSONLoader.Element_object_at_0x107c2af98>]: - W[6]
   + ResizeW[6] = -50
 LinkResizeW[7,<tools.JSONLoader.Element_object_at_0x107c2afd0>]: - W[7]
   + ResizeW[7] = -125
 LinkResizeH[0,<tools.JSONLoader.Element_object_at_0x107c2ae48>]: - H[0]
   + ResizeH[0] = -12
 LinkResizeH[1,<tools.JSONLoader.Element_object_at_0x107c2ae80>]: - H[1]
   + ResizeH[1] = -88
 LinkResizeH[2,<tools.JSONLoader.Element_object_at_0x107c2aeb8>]: - H[2]
   + ResizeH[2] = -38
 LinkResizeH[3,<tools.JSONLoader.Element_object_at_0x107c2aef0>]: - H[3]
   + ResizeH[3] = -38
 LinkResizeH[4,<tools.JSONLoader.Element_object_at_0x107c2af28>]: - H[4]
   + ResizeH[4] = -50
 LinkResizeH[5,<tools.JSONLoader.Element_object_at_0x107c2af60>]: - H[5]
   + ResizeH[5] = -25
 LinkResizeH[6,<tools.JSONLoader.Element_object_at_0x107c2af98>]: - H[6]
   + ResizeH[6] = -25
 LinkResizeH[7,<tools.JSONLoader.Element_object_at_0x107c2afd0>]: - H[7]
   + ResizeH[7] = -19
 LinkMoveX[0,<tools.JSONLoader.Element_object_at_0x107c2ae48>]:
   - EdgeCoord[x0,0] + MoveX[0] = 0
 LinkMoveX[1,<tools.JSONLoader.Element_object_at_0x107c2ae80>]:
   - EdgeCoord[x0,1] + MoveX[1] = -100
 LinkMoveX[2,<tools.JSONLoader.Element_object_at_0x107c2aeb8>]:
   - EdgeCoord[x0,2] + MoveX[2] = 0
 LinkMoveX[3,<tools.JSONLoader.Element_object_at_0x107c2aef0>]:
   - EdgeCoord[x0,3] + MoveX[3] = -50
 LinkMoveX[4,<tools.JSONLoader.Element_object_at_0x107c2af28>]:
   - EdgeCoord[x0,4] + MoveX[4] = -50
 LinkMoveX[5,<tools.JSONLoader.Element_object_at_0x107c2af60>]:
   - EdgeCoord[x0,5] + MoveX[5] = 0
 LinkMoveX[6,<tools.JSONLoader.Element_object_at_0x107c2af98>]:
   - EdgeCoord[x0,6] + MoveX[6] = 0
 LinkMoveX[7,<tools.JSONLoader.Element_object_at_0x107c2afd0>]:
   - EdgeCoord[x0,7] + MoveX[7] = 0
 LinkMoveY[0,<tools.JSONLoader.Element_object_at_0x107c2ae48>]:
   - EdgeCoord[y0,0] + MoveY[0] = 0
 LinkMoveY[1,<tools.JSONLoader.Element_object_at_0x107c2ae80>]:
   - EdgeCoord[y0,1] + MoveY[1] = -12
 LinkMoveY[2,<tools.JSONLoader.Element_object_at_0x107c2aeb8>]:
   - EdgeCoord[y0,2] + MoveY[2] = -62
 LinkMoveY[3,<tools.JSONLoader.Element_object_at_0x107c2aef0>]:
   - EdgeCoord[y0,3] + MoveY[3] = -62
 LinkMoveY[4,<tools.JSONLoader.Element_object_at_0x107c2af28>]:
   - EdgeCoord[y0,4] + MoveY[4] = -12
 LinkMoveY[5,<tools.JSONLoader.Element_object_at_0x107c2af60>]:
   - EdgeCoord[y0,5] + MoveY[5] = -38
 LinkMoveY[6,<tools.JSONLoader.Element_object_at_0x107c2af98>]:
   - EdgeCoord[y0,6] + MoveY[6] = -12
 LinkMoveY[7,<tools.JSONLoader.Element_object_at_0x107c2afd0>]:
   - EdgeCoord[y0,7] + MoveY[7] = -100
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
 LinkEdgeDistance[x0,0,0]: EdgeDistance[x0,0,0] = 0
 LinkEdgeDistance[x0,0,1]: - EdgeCoord[x0,0] + EdgeCoord[x0,1]
   + EdgeDistance[x0,0,1] = 0
 LinkEdgeDistance[x0,0,2]: - EdgeCoord[x0,0] + EdgeCoord[x0,2]
   + EdgeDistance[x0,0,2] = 0
 LinkEdgeDistance[x0,0,3]: - EdgeCoord[x0,0] + EdgeCoord[x0,3]
   + EdgeDistance[x0,0,3] = 0
 LinkEdgeDistance[x0,0,4]: - EdgeCoord[x0,0] + EdgeCoord[x0,4]
   + EdgeDistance[x0,0,4] = 0
 LinkEdgeDistance[x0,0,5]: - EdgeCoord[x0,0] + EdgeCoord[x0,5]
   + EdgeDistance[x0,0,5] = 0
 LinkEdgeDistance[x0,0,6]: - EdgeCoord[x0,0] + EdgeCoord[x0,6]
   + EdgeDistance[x0,0,6] = 0
 LinkEdgeDistance[x0,0,7]: - EdgeCoord[x0,0] + EdgeCoord[x0,7]
   + EdgeDistance[x0,0,7] = 0
 LinkEdgeDistance[x0,1,0]: EdgeCoord[x0,0] - EdgeCoord[x0,1]
   + EdgeDistance[x0,1,0] = 0
 LinkEdgeDistance[x0,1,1]: EdgeDistance[x0,1,1] = 0
 LinkEdgeDistance[x0,1,2]: - EdgeCoord[x0,1] + EdgeCoord[x0,2]
   + EdgeDistance[x0,1,2] = 0
 LinkEdgeDistance[x0,1,3]: - EdgeCoord[x0,1] + EdgeCoord[x0,3]
   + EdgeDistance[x0,1,3] = 0
 LinkEdgeDistance[x0,1,4]: - EdgeCoord[x0,1] + EdgeCoord[x0,4]
   + EdgeDistance[x0,1,4] = 0
 LinkEdgeDistance[x0,1,5]: - EdgeCoord[x0,1] + EdgeCoord[x0,5]
   + EdgeDistance[x0,1,5] = 0
 LinkEdgeDistance[x0,1,6]: - EdgeCoord[x0,1] + EdgeCoord[x0,6]
   + EdgeDistance[x0,1,6] = 0
 LinkEdgeDistance[x0,1,7]: - EdgeCoord[x0,1] + EdgeCoord[x0,7]
   + EdgeDistance[x0,1,7] = 0
 LinkEdgeDistance[x0,2,0]: EdgeCoord[x0,0] - EdgeCoord[x0,2]
   + EdgeDistance[x0,2,0] = 0
 LinkEdgeDistance[x0,2,1]: EdgeCoord[x0,1] - EdgeCoord[x0,2]
   + EdgeDistance[x0,2,1] = 0
 LinkEdgeDistance[x0,2,2]: EdgeDistance[x0,2,2] = 0
 LinkEdgeDistance[x0,2,3]: - EdgeCoord[x0,2] + EdgeCoord[x0,3]
   + EdgeDistance[x0,2,3] = 0
 LinkEdgeDistance[x0,2,4]: - EdgeCoord[x0,2] + EdgeCoord[x0,4]
   + EdgeDistance[x0,2,4] = 0
 LinkEdgeDistance[x0,2,5]: - EdgeCoord[x0,2] + EdgeCoord[x0,5]
   + EdgeDistance[x0,2,5] = 0
 LinkEdgeDistance[x0,2,6]: - EdgeCoord[x0,2] + EdgeCoord[x0,6]
   + EdgeDistance[x0,2,6] = 0
 LinkEdgeDistance[x0,2,7]: - EdgeCoord[x0,2] + EdgeCoord[x0,7]
   + EdgeDistance[x0,2,7] = 0
 LinkEdgeDistance[x0,3,0]: EdgeCoord[x0,0] - EdgeCoord[x0,3]
   + EdgeDistance[x0,3,0] = 0
 LinkEdgeDistance[x0,3,1]: EdgeCoord[x0,1] - EdgeCoord[x0,3]
   + EdgeDistance[x0,3,1] = 0
 LinkEdgeDistance[x0,3,2]: EdgeCoord[x0,2] - EdgeCoord[x0,3]
   + EdgeDistance[x0,3,2] = 0
 LinkEdgeDistance[x0,3,3]: EdgeDistance[x0,3,3] = 0
 LinkEdgeDistance[x0,3,4]: - EdgeCoord[x0,3] + EdgeCoord[x0,4]
   + EdgeDistance[x0,3,4] = 0
 LinkEdgeDistance[x0,3,5]: - EdgeCoord[x0,3] + EdgeCoord[x0,5]
   + EdgeDistance[x0,3,5] = 0
 LinkEdgeDistance[x0,3,6]: - EdgeCoord[x0,3] + EdgeCoord[x0,6]
   + EdgeDistance[x0,3,6] = 0
 LinkEdgeDistance[x0,3,7]: - EdgeCoord[x0,3] + EdgeCoord[x0,7]
   + EdgeDistance[x0,3,7] = 0
 LinkEdgeDistance[x0,4,0]: EdgeCoord[x0,0] - EdgeCoord[x0,4]
   + EdgeDistance[x0,4,0] = 0
 LinkEdgeDistance[x0,4,1]: EdgeCoord[x0,1] - EdgeCoord[x0,4]
   + EdgeDistance[x0,4,1] = 0
 LinkEdgeDistance[x0,4,2]: EdgeCoord[x0,2] - EdgeCoord[x0,4]
   + EdgeDistance[x0,4,2] = 0
 LinkEdgeDistance[x0,4,3]: EdgeCoord[x0,3] - EdgeCoord[x0,4]
   + EdgeDistance[x0,4,3] = 0
 LinkEdgeDistance[x0,4,4]: EdgeDistance[x0,4,4] = 0
 LinkEdgeDistance[x0,4,5]: - EdgeCoord[x0,4] + EdgeCoord[x0,5]
   + EdgeDistance[x0,4,5] = 0
 LinkEdgeDistance[x0,4,6]: - EdgeCoord[x0,4] + EdgeCoord[x0,6]
   + EdgeDistance[x0,4,6] = 0
 LinkEdgeDistance[x0,4,7]: - EdgeCoord[x0,4] + EdgeCoord[x0,7]
   + EdgeDistance[x0,4,7] = 0
 LinkEdgeDistance[x0,5,0]: EdgeCoord[x0,0] - EdgeCoord[x0,5]
   + EdgeDistance[x0,5,0] = 0
 LinkEdgeDistance[x0,5,1]: EdgeCoord[x0,1] - EdgeCoord[x0,5]
   + EdgeDistance[x0,5,1] = 0
 LinkEdgeDistance[x0,5,2]: EdgeCoord[x0,2] - EdgeCoord[x0,5]
   + EdgeDistance[x0,5,2] = 0
 LinkEdgeDistance[x0,5,3]: EdgeCoord[x0,3] - EdgeCoord[x0,5]
   + EdgeDistance[x0,5,3] = 0
 LinkEdgeDistance[x0,5,4]: EdgeCoord[x0,4] - EdgeCoord[x0,5]
   + EdgeDistance[x0,5,4] = 0
 LinkEdgeDistance[x0,5,5]: EdgeDistance[x0,5,5] = 0
 LinkEdgeDistance[x0,5,6]: - EdgeCoord[x0,5] + EdgeCoord[x0,6]
   + EdgeDistance[x0,5,6] = 0
 LinkEdgeDistance[x0,5,7]: - EdgeCoord[x0,5] + EdgeCoord[x0,7]
   + EdgeDistance[x0,5,7] = 0
 LinkEdgeDistance[x0,6,0]: EdgeCoord[x0,0] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,0] = 0
 LinkEdgeDistance[x0,6,1]: EdgeCoord[x0,1] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,1] = 0
 LinkEdgeDistance[x0,6,2]: EdgeCoord[x0,2] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,2] = 0
 LinkEdgeDistance[x0,6,3]: EdgeCoord[x0,3] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,3] = 0
 LinkEdgeDistance[x0,6,4]: EdgeCoord[x0,4] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,4] = 0
 LinkEdgeDistance[x0,6,5]: EdgeCoord[x0,5] - EdgeCoord[x0,6]
   + EdgeDistance[x0,6,5] = 0
 LinkEdgeDistance[x0,6,6]: EdgeDistance[x0,6,6] = 0
 LinkEdgeDistance[x0,6,7]: - EdgeCoord[x0,6] + EdgeCoord[x0,7]
   + EdgeDistance[x0,6,7] = 0
 LinkEdgeDistance[x0,7,0]: EdgeCoord[x0,0] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,0] = 0
 LinkEdgeDistance[x0,7,1]: EdgeCoord[x0,1] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,1] = 0
 LinkEdgeDistance[x0,7,2]: EdgeCoord[x0,2] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,2] = 0
 LinkEdgeDistance[x0,7,3]: EdgeCoord[x0,3] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,3] = 0
 LinkEdgeDistance[x0,7,4]: EdgeCoord[x0,4] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,4] = 0
 LinkEdgeDistance[x0,7,5]: EdgeCoord[x0,5] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,5] = 0
 LinkEdgeDistance[x0,7,6]: EdgeCoord[x0,6] - EdgeCoord[x0,7]
   + EdgeDistance[x0,7,6] = 0
 LinkEdgeDistance[x0,7,7]: EdgeDistance[x0,7,7] = 0
 LinkEdgeDistance[y0,0,0]: EdgeDistance[y0,0,0] = 0
 LinkEdgeDistance[y0,0,1]: - EdgeCoord[y0,0] + EdgeCoord[y0,1]
   + EdgeDistance[y0,0,1] = 0
 LinkEdgeDistance[y0,0,2]: - EdgeCoord[y0,0] + EdgeCoord[y0,2]
   + EdgeDistance[y0,0,2] = 0
 LinkEdgeDistance[y0,0,3]: - EdgeCoord[y0,0] + EdgeCoord[y0,3]
   + EdgeDistance[y0,0,3] = 0
 LinkEdgeDistance[y0,0,4]: - EdgeCoord[y0,0] + EdgeCoord[y0,4]
   + EdgeDistance[y0,0,4] = 0
 LinkEdgeDistance[y0,0,5]: - EdgeCoord[y0,0] + EdgeCoord[y0,5]
   + EdgeDistance[y0,0,5] = 0
 LinkEdgeDistance[y0,0,6]: - EdgeCoord[y0,0] + EdgeCoord[y0,6]
   + EdgeDistance[y0,0,6] = 0
 LinkEdgeDistance[y0,0,7]: - EdgeCoord[y0,0] + EdgeCoord[y0,7]
   + EdgeDistance[y0,0,7] = 0
 LinkEdgeDistance[y0,1,0]: EdgeCoord[y0,0] - EdgeCoord[y0,1]
   + EdgeDistance[y0,1,0] = 0
 LinkEdgeDistance[y0,1,1]: EdgeDistance[y0,1,1] = 0
 LinkEdgeDistance[y0,1,2]: - EdgeCoord[y0,1] + EdgeCoord[y0,2]
   + EdgeDistance[y0,1,2] = 0
 LinkEdgeDistance[y0,1,3]: - EdgeCoord[y0,1] + EdgeCoord[y0,3]
   + EdgeDistance[y0,1,3] = 0
 LinkEdgeDistance[y0,1,4]: - EdgeCoord[y0,1] + EdgeCoord[y0,4]
   + EdgeDistance[y0,1,4] = 0
 LinkEdgeDistance[y0,1,5]: - EdgeCoord[y0,1] + EdgeCoord[y0,5]
   + EdgeDistance[y0,1,5] = 0
 LinkEdgeDistance[y0,1,6]: - EdgeCoord[y0,1] + EdgeCoord[y0,6]
   + EdgeDistance[y0,1,6] = 0
 LinkEdgeDistance[y0,1,7]: - EdgeCoord[y0,1] + EdgeCoord[y0,7]
   + EdgeDistance[y0,1,7] = 0
 LinkEdgeDistance[y0,2,0]: EdgeCoord[y0,0] - EdgeCoord[y0,2]
   + EdgeDistance[y0,2,0] = 0
 LinkEdgeDistance[y0,2,1]: EdgeCoord[y0,1] - EdgeCoord[y0,2]
   + EdgeDistance[y0,2,1] = 0
 LinkEdgeDistance[y0,2,2]: EdgeDistance[y0,2,2] = 0
 LinkEdgeDistance[y0,2,3]: - EdgeCoord[y0,2] + EdgeCoord[y0,3]
   + EdgeDistance[y0,2,3] = 0
 LinkEdgeDistance[y0,2,4]: - EdgeCoord[y0,2] + EdgeCoord[y0,4]
   + EdgeDistance[y0,2,4] = 0
 LinkEdgeDistance[y0,2,5]: - EdgeCoord[y0,2] + EdgeCoord[y0,5]
   + EdgeDistance[y0,2,5] = 0
 LinkEdgeDistance[y0,2,6]: - EdgeCoord[y0,2] + EdgeCoord[y0,6]
   + EdgeDistance[y0,2,6] = 0
 LinkEdgeDistance[y0,2,7]: - EdgeCoord[y0,2] + EdgeCoord[y0,7]
   + EdgeDistance[y0,2,7] = 0
 LinkEdgeDistance[y0,3,0]: EdgeCoord[y0,0] - EdgeCoord[y0,3]
   + EdgeDistance[y0,3,0] = 0
 LinkEdgeDistance[y0,3,1]: EdgeCoord[y0,1] - EdgeCoord[y0,3]
   + EdgeDistance[y0,3,1] = 0
 LinkEdgeDistance[y0,3,2]: EdgeCoord[y0,2] - EdgeCoord[y0,3]
   + EdgeDistance[y0,3,2] = 0
 LinkEdgeDistance[y0,3,3]: EdgeDistance[y0,3,3] = 0
 LinkEdgeDistance[y0,3,4]: - EdgeCoord[y0,3] + EdgeCoord[y0,4]
   + EdgeDistance[y0,3,4] = 0
 LinkEdgeDistance[y0,3,5]: - EdgeCoord[y0,3] + EdgeCoord[y0,5]
   + EdgeDistance[y0,3,5] = 0
 LinkEdgeDistance[y0,3,6]: - EdgeCoord[y0,3] + EdgeCoord[y0,6]
   + EdgeDistance[y0,3,6] = 0
 LinkEdgeDistance[y0,3,7]: - EdgeCoord[y0,3] + EdgeCoord[y0,7]
   + EdgeDistance[y0,3,7] = 0
 LinkEdgeDistance[y0,4,0]: EdgeCoord[y0,0] - EdgeCoord[y0,4]
   + EdgeDistance[y0,4,0] = 0
 LinkEdgeDistance[y0,4,1]: EdgeCoord[y0,1] - EdgeCoord[y0,4]
   + EdgeDistance[y0,4,1] = 0
 LinkEdgeDistance[y0,4,2]: EdgeCoord[y0,2] - EdgeCoord[y0,4]
   + EdgeDistance[y0,4,2] = 0
 LinkEdgeDistance[y0,4,3]: EdgeCoord[y0,3] - EdgeCoord[y0,4]
   + EdgeDistance[y0,4,3] = 0
 LinkEdgeDistance[y0,4,4]: EdgeDistance[y0,4,4] = 0
 LinkEdgeDistance[y0,4,5]: - EdgeCoord[y0,4] + EdgeCoord[y0,5]
   + EdgeDistance[y0,4,5] = 0
 LinkEdgeDistance[y0,4,6]: - EdgeCoord[y0,4] + EdgeCoord[y0,6]
   + EdgeDistance[y0,4,6] = 0
 LinkEdgeDistance[y0,4,7]: - EdgeCoord[y0,4] + EdgeCoord[y0,7]
   + EdgeDistance[y0,4,7] = 0
 LinkEdgeDistance[y0,5,0]: EdgeCoord[y0,0] - EdgeCoord[y0,5]
   + EdgeDistance[y0,5,0] = 0
 LinkEdgeDistance[y0,5,1]: EdgeCoord[y0,1] - EdgeCoord[y0,5]
   + EdgeDistance[y0,5,1] = 0
 LinkEdgeDistance[y0,5,2]: EdgeCoord[y0,2] - EdgeCoord[y0,5]
   + EdgeDistance[y0,5,2] = 0
 LinkEdgeDistance[y0,5,3]: EdgeCoord[y0,3] - EdgeCoord[y0,5]
   + EdgeDistance[y0,5,3] = 0
 LinkEdgeDistance[y0,5,4]: EdgeCoord[y0,4] - EdgeCoord[y0,5]
   + EdgeDistance[y0,5,4] = 0
 LinkEdgeDistance[y0,5,5]: EdgeDistance[y0,5,5] = 0
 LinkEdgeDistance[y0,5,6]: - EdgeCoord[y0,5] + EdgeCoord[y0,6]
   + EdgeDistance[y0,5,6] = 0
 LinkEdgeDistance[y0,5,7]: - EdgeCoord[y0,5] + EdgeCoord[y0,7]
   + EdgeDistance[y0,5,7] = 0
 LinkEdgeDistance[y0,6,0]: EdgeCoord[y0,0] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,0] = 0
 LinkEdgeDistance[y0,6,1]: EdgeCoord[y0,1] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,1] = 0
 LinkEdgeDistance[y0,6,2]: EdgeCoord[y0,2] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,2] = 0
 LinkEdgeDistance[y0,6,3]: EdgeCoord[y0,3] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,3] = 0
 LinkEdgeDistance[y0,6,4]: EdgeCoord[y0,4] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,4] = 0
 LinkEdgeDistance[y0,6,5]: EdgeCoord[y0,5] - EdgeCoord[y0,6]
   + EdgeDistance[y0,6,5] = 0
 LinkEdgeDistance[y0,6,6]: EdgeDistance[y0,6,6] = 0
 LinkEdgeDistance[y0,6,7]: - EdgeCoord[y0,6] + EdgeCoord[y0,7]
   + EdgeDistance[y0,6,7] = 0
 LinkEdgeDistance[y0,7,0]: EdgeCoord[y0,0] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,0] = 0
 LinkEdgeDistance[y0,7,1]: EdgeCoord[y0,1] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,1] = 0
 LinkEdgeDistance[y0,7,2]: EdgeCoord[y0,2] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,2] = 0
 LinkEdgeDistance[y0,7,3]: EdgeCoord[y0,3] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,3] = 0
 LinkEdgeDistance[y0,7,4]: EdgeCoord[y0,4] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,4] = 0
 LinkEdgeDistance[y0,7,5]: EdgeCoord[y0,5] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,5] = 0
 LinkEdgeDistance[y0,7,6]: EdgeCoord[y0,6] - EdgeCoord[y0,7]
   + EdgeDistance[y0,7,6] = 0
 LinkEdgeDistance[y0,7,7]: EdgeDistance[y0,7,7] = 0
 LinkEdgeDistance[x1,0,0]: EdgeDistance[x1,0,0] = 0
 LinkEdgeDistance[x1,0,1]: - EdgeCoord[x1,0] + EdgeCoord[x1,1]
   + EdgeDistance[x1,0,1] = 0
 LinkEdgeDistance[x1,0,2]: - EdgeCoord[x1,0] + EdgeCoord[x1,2]
   + EdgeDistance[x1,0,2] = 0
 LinkEdgeDistance[x1,0,3]: - EdgeCoord[x1,0] + EdgeCoord[x1,3]
   + EdgeDistance[x1,0,3] = 0
 LinkEdgeDistance[x1,0,4]: - EdgeCoord[x1,0] + EdgeCoord[x1,4]
   + EdgeDistance[x1,0,4] = 0
 LinkEdgeDistance[x1,0,5]: - EdgeCoord[x1,0] + EdgeCoord[x1,5]
   + EdgeDistance[x1,0,5] = 0
 LinkEdgeDistance[x1,0,6]: - EdgeCoord[x1,0] + EdgeCoord[x1,6]
   + EdgeDistance[x1,0,6] = 0
 LinkEdgeDistance[x1,0,7]: - EdgeCoord[x1,0] + EdgeCoord[x1,7]
   + EdgeDistance[x1,0,7] = 0
 LinkEdgeDistance[x1,1,0]: EdgeCoord[x1,0] - EdgeCoord[x1,1]
   + EdgeDistance[x1,1,0] = 0
 LinkEdgeDistance[x1,1,1]: EdgeDistance[x1,1,1] = 0
 LinkEdgeDistance[x1,1,2]: - EdgeCoord[x1,1] + EdgeCoord[x1,2]
   + EdgeDistance[x1,1,2] = 0
 LinkEdgeDistance[x1,1,3]: - EdgeCoord[x1,1] + EdgeCoord[x1,3]
   + EdgeDistance[x1,1,3] = 0
 LinkEdgeDistance[x1,1,4]: - EdgeCoord[x1,1] + EdgeCoord[x1,4]
   + EdgeDistance[x1,1,4] = 0
 LinkEdgeDistance[x1,1,5]: - EdgeCoord[x1,1] + EdgeCoord[x1,5]
   + EdgeDistance[x1,1,5] = 0
 LinkEdgeDistance[x1,1,6]: - EdgeCoord[x1,1] + EdgeCoord[x1,6]
   + EdgeDistance[x1,1,6] = 0
 LinkEdgeDistance[x1,1,7]: - EdgeCoord[x1,1] + EdgeCoord[x1,7]
   + EdgeDistance[x1,1,7] = 0
 LinkEdgeDistance[x1,2,0]: EdgeCoord[x1,0] - EdgeCoord[x1,2]
   + EdgeDistance[x1,2,0] = 0
 LinkEdgeDistance[x1,2,1]: EdgeCoord[x1,1] - EdgeCoord[x1,2]
   + EdgeDistance[x1,2,1] = 0
 LinkEdgeDistance[x1,2,2]: EdgeDistance[x1,2,2] = 0
 LinkEdgeDistance[x1,2,3]: - EdgeCoord[x1,2] + EdgeCoord[x1,3]
   + EdgeDistance[x1,2,3] = 0
 LinkEdgeDistance[x1,2,4]: - EdgeCoord[x1,2] + EdgeCoord[x1,4]
   + EdgeDistance[x1,2,4] = 0
 LinkEdgeDistance[x1,2,5]: - EdgeCoord[x1,2] + EdgeCoord[x1,5]
   + EdgeDistance[x1,2,5] = 0
 LinkEdgeDistance[x1,2,6]: - EdgeCoord[x1,2] + EdgeCoord[x1,6]
   + EdgeDistance[x1,2,6] = 0
 LinkEdgeDistance[x1,2,7]: - EdgeCoord[x1,2] + EdgeCoord[x1,7]
   + EdgeDistance[x1,2,7] = 0
 LinkEdgeDistance[x1,3,0]: EdgeCoord[x1,0] - EdgeCoord[x1,3]
   + EdgeDistance[x1,3,0] = 0
 LinkEdgeDistance[x1,3,1]: EdgeCoord[x1,1] - EdgeCoord[x1,3]
   + EdgeDistance[x1,3,1] = 0
 LinkEdgeDistance[x1,3,2]: EdgeCoord[x1,2] - EdgeCoord[x1,3]
   + EdgeDistance[x1,3,2] = 0
 LinkEdgeDistance[x1,3,3]: EdgeDistance[x1,3,3] = 0
 LinkEdgeDistance[x1,3,4]: - EdgeCoord[x1,3] + EdgeCoord[x1,4]
   + EdgeDistance[x1,3,4] = 0
 LinkEdgeDistance[x1,3,5]: - EdgeCoord[x1,3] + EdgeCoord[x1,5]
   + EdgeDistance[x1,3,5] = 0
 LinkEdgeDistance[x1,3,6]: - EdgeCoord[x1,3] + EdgeCoord[x1,6]
   + EdgeDistance[x1,3,6] = 0
 LinkEdgeDistance[x1,3,7]: - EdgeCoord[x1,3] + EdgeCoord[x1,7]
   + EdgeDistance[x1,3,7] = 0
 LinkEdgeDistance[x1,4,0]: EdgeCoord[x1,0] - EdgeCoord[x1,4]
   + EdgeDistance[x1,4,0] = 0
 LinkEdgeDistance[x1,4,1]: EdgeCoord[x1,1] - EdgeCoord[x1,4]
   + EdgeDistance[x1,4,1] = 0
 LinkEdgeDistance[x1,4,2]: EdgeCoord[x1,2] - EdgeCoord[x1,4]
   + EdgeDistance[x1,4,2] = 0
 LinkEdgeDistance[x1,4,3]: EdgeCoord[x1,3] - EdgeCoord[x1,4]
   + EdgeDistance[x1,4,3] = 0
 LinkEdgeDistance[x1,4,4]: EdgeDistance[x1,4,4] = 0
 LinkEdgeDistance[x1,4,5]: - EdgeCoord[x1,4] + EdgeCoord[x1,5]
   + EdgeDistance[x1,4,5] = 0
 LinkEdgeDistance[x1,4,6]: - EdgeCoord[x1,4] + EdgeCoord[x1,6]
   + EdgeDistance[x1,4,6] = 0
 LinkEdgeDistance[x1,4,7]: - EdgeCoord[x1,4] + EdgeCoord[x1,7]
   + EdgeDistance[x1,4,7] = 0
 LinkEdgeDistance[x1,5,0]: EdgeCoord[x1,0] - EdgeCoord[x1,5]
   + EdgeDistance[x1,5,0] = 0
 LinkEdgeDistance[x1,5,1]: EdgeCoord[x1,1] - EdgeCoord[x1,5]
   + EdgeDistance[x1,5,1] = 0
 LinkEdgeDistance[x1,5,2]: EdgeCoord[x1,2] - EdgeCoord[x1,5]
   + EdgeDistance[x1,5,2] = 0
 LinkEdgeDistance[x1,5,3]: EdgeCoord[x1,3] - EdgeCoord[x1,5]
   + EdgeDistance[x1,5,3] = 0
 LinkEdgeDistance[x1,5,4]: EdgeCoord[x1,4] - EdgeCoord[x1,5]
   + EdgeDistance[x1,5,4] = 0
 LinkEdgeDistance[x1,5,5]: EdgeDistance[x1,5,5] = 0
 LinkEdgeDistance[x1,5,6]: - EdgeCoord[x1,5] + EdgeCoord[x1,6]
   + EdgeDistance[x1,5,6] = 0
 LinkEdgeDistance[x1,5,7]: - EdgeCoord[x1,5] + EdgeCoord[x1,7]
   + EdgeDistance[x1,5,7] = 0
 LinkEdgeDistance[x1,6,0]: EdgeCoord[x1,0] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,0] = 0
 LinkEdgeDistance[x1,6,1]: EdgeCoord[x1,1] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,1] = 0
 LinkEdgeDistance[x1,6,2]: EdgeCoord[x1,2] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,2] = 0
 LinkEdgeDistance[x1,6,3]: EdgeCoord[x1,3] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,3] = 0
 LinkEdgeDistance[x1,6,4]: EdgeCoord[x1,4] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,4] = 0
 LinkEdgeDistance[x1,6,5]: EdgeCoord[x1,5] - EdgeCoord[x1,6]
   + EdgeDistance[x1,6,5] = 0
 LinkEdgeDistance[x1,6,6]: EdgeDistance[x1,6,6] = 0
 LinkEdgeDistance[x1,6,7]: - EdgeCoord[x1,6] + EdgeCoord[x1,7]
   + EdgeDistance[x1,6,7] = 0
 LinkEdgeDistance[x1,7,0]: EdgeCoord[x1,0] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,0] = 0
 LinkEdgeDistance[x1,7,1]: EdgeCoord[x1,1] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,1] = 0
 LinkEdgeDistance[x1,7,2]: EdgeCoord[x1,2] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,2] = 0
 LinkEdgeDistance[x1,7,3]: EdgeCoord[x1,3] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,3] = 0
 LinkEdgeDistance[x1,7,4]: EdgeCoord[x1,4] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,4] = 0
 LinkEdgeDistance[x1,7,5]: EdgeCoord[x1,5] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,5] = 0
 LinkEdgeDistance[x1,7,6]: EdgeCoord[x1,6] - EdgeCoord[x1,7]
   + EdgeDistance[x1,7,6] = 0
 LinkEdgeDistance[x1,7,7]: EdgeDistance[x1,7,7] = 0
 LinkEdgeDistance[y1,0,0]: EdgeDistance[y1,0,0] = 0
 LinkEdgeDistance[y1,0,1]: - EdgeCoord[y1,0] + EdgeCoord[y1,1]
   + EdgeDistance[y1,0,1] = 0
 LinkEdgeDistance[y1,0,2]: - EdgeCoord[y1,0] + EdgeCoord[y1,2]
   + EdgeDistance[y1,0,2] = 0
 LinkEdgeDistance[y1,0,3]: - EdgeCoord[y1,0] + EdgeCoord[y1,3]
   + EdgeDistance[y1,0,3] = 0
 LinkEdgeDistance[y1,0,4]: - EdgeCoord[y1,0] + EdgeCoord[y1,4]
   + EdgeDistance[y1,0,4] = 0
 LinkEdgeDistance[y1,0,5]: - EdgeCoord[y1,0] + EdgeCoord[y1,5]
   + EdgeDistance[y1,0,5] = 0
 LinkEdgeDistance[y1,0,6]: - EdgeCoord[y1,0] + EdgeCoord[y1,6]
   + EdgeDistance[y1,0,6] = 0
 LinkEdgeDistance[y1,0,7]: - EdgeCoord[y1,0] + EdgeCoord[y1,7]
   + EdgeDistance[y1,0,7] = 0
 LinkEdgeDistance[y1,1,0]: EdgeCoord[y1,0] - EdgeCoord[y1,1]
   + EdgeDistance[y1,1,0] = 0
 LinkEdgeDistance[y1,1,1]: EdgeDistance[y1,1,1] = 0
 LinkEdgeDistance[y1,1,2]: - EdgeCoord[y1,1] + EdgeCoord[y1,2]
   + EdgeDistance[y1,1,2] = 0
 LinkEdgeDistance[y1,1,3]: - EdgeCoord[y1,1] + EdgeCoord[y1,3]
   + EdgeDistance[y1,1,3] = 0
 LinkEdgeDistance[y1,1,4]: - EdgeCoord[y1,1] + EdgeCoord[y1,4]
   + EdgeDistance[y1,1,4] = 0
 LinkEdgeDistance[y1,1,5]: - EdgeCoord[y1,1] + EdgeCoord[y1,5]
   + EdgeDistance[y1,1,5] = 0
 LinkEdgeDistance[y1,1,6]: - EdgeCoord[y1,1] + EdgeCoord[y1,6]
   + EdgeDistance[y1,1,6] = 0
 LinkEdgeDistance[y1,1,7]: - EdgeCoord[y1,1] + EdgeCoord[y1,7]
   + EdgeDistance[y1,1,7] = 0
 LinkEdgeDistance[y1,2,0]: EdgeCoord[y1,0] - EdgeCoord[y1,2]
   + EdgeDistance[y1,2,0] = 0
 LinkEdgeDistance[y1,2,1]: EdgeCoord[y1,1] - EdgeCoord[y1,2]
   + EdgeDistance[y1,2,1] = 0
 LinkEdgeDistance[y1,2,2]: EdgeDistance[y1,2,2] = 0
 LinkEdgeDistance[y1,2,3]: - EdgeCoord[y1,2] + EdgeCoord[y1,3]
   + EdgeDistance[y1,2,3] = 0
 LinkEdgeDistance[y1,2,4]: - EdgeCoord[y1,2] + EdgeCoord[y1,4]
   + EdgeDistance[y1,2,4] = 0
 LinkEdgeDistance[y1,2,5]: - EdgeCoord[y1,2] + EdgeCoord[y1,5]
   + EdgeDistance[y1,2,5] = 0
 LinkEdgeDistance[y1,2,6]: - EdgeCoord[y1,2] + EdgeCoord[y1,6]
   + EdgeDistance[y1,2,6] = 0
 LinkEdgeDistance[y1,2,7]: - EdgeCoord[y1,2] + EdgeCoord[y1,7]
   + EdgeDistance[y1,2,7] = 0
 LinkEdgeDistance[y1,3,0]: EdgeCoord[y1,0] - EdgeCoord[y1,3]
   + EdgeDistance[y1,3,0] = 0
 LinkEdgeDistance[y1,3,1]: EdgeCoord[y1,1] - EdgeCoord[y1,3]
   + EdgeDistance[y1,3,1] = 0
 LinkEdgeDistance[y1,3,2]: EdgeCoord[y1,2] - EdgeCoord[y1,3]
   + EdgeDistance[y1,3,2] = 0
 LinkEdgeDistance[y1,3,3]: EdgeDistance[y1,3,3] = 0
 LinkEdgeDistance[y1,3,4]: - EdgeCoord[y1,3] + EdgeCoord[y1,4]
   + EdgeDistance[y1,3,4] = 0
 LinkEdgeDistance[y1,3,5]: - EdgeCoord[y1,3] + EdgeCoord[y1,5]
   + EdgeDistance[y1,3,5] = 0
 LinkEdgeDistance[y1,3,6]: - EdgeCoord[y1,3] + EdgeCoord[y1,6]
   + EdgeDistance[y1,3,6] = 0
 LinkEdgeDistance[y1,3,7]: - EdgeCoord[y1,3] + EdgeCoord[y1,7]
   + EdgeDistance[y1,3,7] = 0
 LinkEdgeDistance[y1,4,0]: EdgeCoord[y1,0] - EdgeCoord[y1,4]
   + EdgeDistance[y1,4,0] = 0
 LinkEdgeDistance[y1,4,1]: EdgeCoord[y1,1] - EdgeCoord[y1,4]
   + EdgeDistance[y1,4,1] = 0
 LinkEdgeDistance[y1,4,2]: EdgeCoord[y1,2] - EdgeCoord[y1,4]
   + EdgeDistance[y1,4,2] = 0
 LinkEdgeDistance[y1,4,3]: EdgeCoord[y1,3] - EdgeCoord[y1,4]
   + EdgeDistance[y1,4,3] = 0
 LinkEdgeDistance[y1,4,4]: EdgeDistance[y1,4,4] = 0
 LinkEdgeDistance[y1,4,5]: - EdgeCoord[y1,4] + EdgeCoord[y1,5]
   + EdgeDistance[y1,4,5] = 0
 LinkEdgeDistance[y1,4,6]: - EdgeCoord[y1,4] + EdgeCoord[y1,6]
   + EdgeDistance[y1,4,6] = 0
 LinkEdgeDistance[y1,4,7]: - EdgeCoord[y1,4] + EdgeCoord[y1,7]
   + EdgeDistance[y1,4,7] = 0
 LinkEdgeDistance[y1,5,0]: EdgeCoord[y1,0] - EdgeCoord[y1,5]
   + EdgeDistance[y1,5,0] = 0
 LinkEdgeDistance[y1,5,1]: EdgeCoord[y1,1] - EdgeCoord[y1,5]
   + EdgeDistance[y1,5,1] = 0
 LinkEdgeDistance[y1,5,2]: EdgeCoord[y1,2] - EdgeCoord[y1,5]
   + EdgeDistance[y1,5,2] = 0
 LinkEdgeDistance[y1,5,3]: EdgeCoord[y1,3] - EdgeCoord[y1,5]
   + EdgeDistance[y1,5,3] = 0
 LinkEdgeDistance[y1,5,4]: EdgeCoord[y1,4] - EdgeCoord[y1,5]
   + EdgeDistance[y1,5,4] = 0
 LinkEdgeDistance[y1,5,5]: EdgeDistance[y1,5,5] = 0
 LinkEdgeDistance[y1,5,6]: - EdgeCoord[y1,5] + EdgeCoord[y1,6]
   + EdgeDistance[y1,5,6] = 0
 LinkEdgeDistance[y1,5,7]: - EdgeCoord[y1,5] + EdgeCoord[y1,7]
   + EdgeDistance[y1,5,7] = 0
 LinkEdgeDistance[y1,6,0]: EdgeCoord[y1,0] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,0] = 0
 LinkEdgeDistance[y1,6,1]: EdgeCoord[y1,1] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,1] = 0
 LinkEdgeDistance[y1,6,2]: EdgeCoord[y1,2] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,2] = 0
 LinkEdgeDistance[y1,6,3]: EdgeCoord[y1,3] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,3] = 0
 LinkEdgeDistance[y1,6,4]: EdgeCoord[y1,4] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,4] = 0
 LinkEdgeDistance[y1,6,5]: EdgeCoord[y1,5] - EdgeCoord[y1,6]
   + EdgeDistance[y1,6,5] = 0
 LinkEdgeDistance[y1,6,6]: EdgeDistance[y1,6,6] = 0
 LinkEdgeDistance[y1,6,7]: - EdgeCoord[y1,6] + EdgeCoord[y1,7]
   + EdgeDistance[y1,6,7] = 0
 LinkEdgeDistance[y1,7,0]: EdgeCoord[y1,0] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,0] = 0
 LinkEdgeDistance[y1,7,1]: EdgeCoord[y1,1] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,1] = 0
 LinkEdgeDistance[y1,7,2]: EdgeCoord[y1,2] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,2] = 0
 LinkEdgeDistance[y1,7,3]: EdgeCoord[y1,3] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,3] = 0
 LinkEdgeDistance[y1,7,4]: EdgeCoord[y1,4] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,4] = 0
 LinkEdgeDistance[y1,7,5]: EdgeCoord[y1,5] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,5] = 0
 LinkEdgeDistance[y1,7,6]: EdgeCoord[y1,6] - EdgeCoord[y1,7]
   + EdgeDistance[y1,7,6] = 0
 LinkEdgeDistance[y1,7,7]: EdgeDistance[y1,7,7] = 0
 LinkIsElemFirstInGroup1[x0,0,1]: - EdgesAlign[x0,0,1]
   + IsElemFirstInGroup[x0,1] <= 0
 LinkIsElemFirstInGroup1[x0,0,2]: - EdgesAlign[x0,0,2]
   + IsElemFirstInGroup[x0,2] <= 0
 LinkIsElemFirstInGroup1[x0,0,3]: - EdgesAlign[x0,0,3]
   + IsElemFirstInGroup[x0,3] <= 0
 LinkIsElemFirstInGroup1[x0,0,4]: - EdgesAlign[x0,0,4]
   + IsElemFirstInGroup[x0,4] <= 0
 LinkIsElemFirstInGroup1[x0,0,5]: - EdgesAlign[x0,0,5]
   + IsElemFirstInGroup[x0,5] <= 0
 LinkIsElemFirstInGroup1[x0,0,6]: - EdgesAlign[x0,0,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,0,7]: - EdgesAlign[x0,0,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,1,2]: - EdgesAlign[x0,1,2]
   + IsElemFirstInGroup[x0,2] <= 0
 LinkIsElemFirstInGroup1[x0,1,3]: - EdgesAlign[x0,1,3]
   + IsElemFirstInGroup[x0,3] <= 0
 LinkIsElemFirstInGroup1[x0,1,4]: - EdgesAlign[x0,1,4]
   + IsElemFirstInGroup[x0,4] <= 0
 LinkIsElemFirstInGroup1[x0,1,5]: - EdgesAlign[x0,1,5]
   + IsElemFirstInGroup[x0,5] <= 0
 LinkIsElemFirstInGroup1[x0,1,6]: - EdgesAlign[x0,1,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,1,7]: - EdgesAlign[x0,1,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,2,3]: - EdgesAlign[x0,2,3]
   + IsElemFirstInGroup[x0,3] <= 0
 LinkIsElemFirstInGroup1[x0,2,4]: - EdgesAlign[x0,2,4]
   + IsElemFirstInGroup[x0,4] <= 0
 LinkIsElemFirstInGroup1[x0,2,5]: - EdgesAlign[x0,2,5]
   + IsElemFirstInGroup[x0,5] <= 0
 LinkIsElemFirstInGroup1[x0,2,6]: - EdgesAlign[x0,2,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,2,7]: - EdgesAlign[x0,2,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,3,4]: - EdgesAlign[x0,3,4]
   + IsElemFirstInGroup[x0,4] <= 0
 LinkIsElemFirstInGroup1[x0,3,5]: - EdgesAlign[x0,3,5]
   + IsElemFirstInGroup[x0,5] <= 0
 LinkIsElemFirstInGroup1[x0,3,6]: - EdgesAlign[x0,3,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,3,7]: - EdgesAlign[x0,3,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,4,5]: - EdgesAlign[x0,4,5]
   + IsElemFirstInGroup[x0,5] <= 0
 LinkIsElemFirstInGroup1[x0,4,6]: - EdgesAlign[x0,4,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,4,7]: - EdgesAlign[x0,4,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,5,6]: - EdgesAlign[x0,5,6]
   + IsElemFirstInGroup[x0,6] <= 0
 LinkIsElemFirstInGroup1[x0,5,7]: - EdgesAlign[x0,5,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[x0,6,7]: - EdgesAlign[x0,6,7]
   + IsElemFirstInGroup[x0,7] <= 0
 LinkIsElemFirstInGroup1[y0,0,1]: - EdgesAlign[y0,0,1]
   + IsElemFirstInGroup[y0,1] <= 0
 LinkIsElemFirstInGroup1[y0,0,2]: - EdgesAlign[y0,0,2]
   + IsElemFirstInGroup[y0,2] <= 0
 LinkIsElemFirstInGroup1[y0,0,3]: - EdgesAlign[y0,0,3]
   + IsElemFirstInGroup[y0,3] <= 0
 LinkIsElemFirstInGroup1[y0,0,4]: - EdgesAlign[y0,0,4]
   + IsElemFirstInGroup[y0,4] <= 0
 LinkIsElemFirstInGroup1[y0,0,5]: - EdgesAlign[y0,0,5]
   + IsElemFirstInGroup[y0,5] <= 0
 LinkIsElemFirstInGroup1[y0,0,6]: - EdgesAlign[y0,0,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,0,7]: - EdgesAlign[y0,0,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,1,2]: - EdgesAlign[y0,1,2]
   + IsElemFirstInGroup[y0,2] <= 0
 LinkIsElemFirstInGroup1[y0,1,3]: - EdgesAlign[y0,1,3]
   + IsElemFirstInGroup[y0,3] <= 0
 LinkIsElemFirstInGroup1[y0,1,4]: - EdgesAlign[y0,1,4]
   + IsElemFirstInGroup[y0,4] <= 0
 LinkIsElemFirstInGroup1[y0,1,5]: - EdgesAlign[y0,1,5]
   + IsElemFirstInGroup[y0,5] <= 0
 LinkIsElemFirstInGroup1[y0,1,6]: - EdgesAlign[y0,1,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,1,7]: - EdgesAlign[y0,1,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,2,3]: - EdgesAlign[y0,2,3]
   + IsElemFirstInGroup[y0,3] <= 0
 LinkIsElemFirstInGroup1[y0,2,4]: - EdgesAlign[y0,2,4]
   + IsElemFirstInGroup[y0,4] <= 0
 LinkIsElemFirstInGroup1[y0,2,5]: - EdgesAlign[y0,2,5]
   + IsElemFirstInGroup[y0,5] <= 0
 LinkIsElemFirstInGroup1[y0,2,6]: - EdgesAlign[y0,2,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,2,7]: - EdgesAlign[y0,2,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,3,4]: - EdgesAlign[y0,3,4]
   + IsElemFirstInGroup[y0,4] <= 0
 LinkIsElemFirstInGroup1[y0,3,5]: - EdgesAlign[y0,3,5]
   + IsElemFirstInGroup[y0,5] <= 0
 LinkIsElemFirstInGroup1[y0,3,6]: - EdgesAlign[y0,3,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,3,7]: - EdgesAlign[y0,3,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,4,5]: - EdgesAlign[y0,4,5]
   + IsElemFirstInGroup[y0,5] <= 0
 LinkIsElemFirstInGroup1[y0,4,6]: - EdgesAlign[y0,4,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,4,7]: - EdgesAlign[y0,4,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,5,6]: - EdgesAlign[y0,5,6]
   + IsElemFirstInGroup[y0,6] <= 0
 LinkIsElemFirstInGroup1[y0,5,7]: - EdgesAlign[y0,5,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[y0,6,7]: - EdgesAlign[y0,6,7]
   + IsElemFirstInGroup[y0,7] <= 0
 LinkIsElemFirstInGroup1[x1,0,1]: - EdgesAlign[x1,0,1]
   + IsElemFirstInGroup[x1,1] <= 0
 LinkIsElemFirstInGroup1[x1,0,2]: - EdgesAlign[x1,0,2]
   + IsElemFirstInGroup[x1,2] <= 0
 LinkIsElemFirstInGroup1[x1,0,3]: - EdgesAlign[x1,0,3]
   + IsElemFirstInGroup[x1,3] <= 0
 LinkIsElemFirstInGroup1[x1,0,4]: - EdgesAlign[x1,0,4]
   + IsElemFirstInGroup[x1,4] <= 0
 LinkIsElemFirstInGroup1[x1,0,5]: - EdgesAlign[x1,0,5]
   + IsElemFirstInGroup[x1,5] <= 0
 LinkIsElemFirstInGroup1[x1,0,6]: - EdgesAlign[x1,0,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,0,7]: - EdgesAlign[x1,0,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,1,2]: - EdgesAlign[x1,1,2]
   + IsElemFirstInGroup[x1,2] <= 0
 LinkIsElemFirstInGroup1[x1,1,3]: - EdgesAlign[x1,1,3]
   + IsElemFirstInGroup[x1,3] <= 0
 LinkIsElemFirstInGroup1[x1,1,4]: - EdgesAlign[x1,1,4]
   + IsElemFirstInGroup[x1,4] <= 0
 LinkIsElemFirstInGroup1[x1,1,5]: - EdgesAlign[x1,1,5]
   + IsElemFirstInGroup[x1,5] <= 0
 LinkIsElemFirstInGroup1[x1,1,6]: - EdgesAlign[x1,1,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,1,7]: - EdgesAlign[x1,1,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,2,3]: - EdgesAlign[x1,2,3]
   + IsElemFirstInGroup[x1,3] <= 0
 LinkIsElemFirstInGroup1[x1,2,4]: - EdgesAlign[x1,2,4]
   + IsElemFirstInGroup[x1,4] <= 0
 LinkIsElemFirstInGroup1[x1,2,5]: - EdgesAlign[x1,2,5]
   + IsElemFirstInGroup[x1,5] <= 0
 LinkIsElemFirstInGroup1[x1,2,6]: - EdgesAlign[x1,2,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,2,7]: - EdgesAlign[x1,2,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,3,4]: - EdgesAlign[x1,3,4]
   + IsElemFirstInGroup[x1,4] <= 0
 LinkIsElemFirstInGroup1[x1,3,5]: - EdgesAlign[x1,3,5]
   + IsElemFirstInGroup[x1,5] <= 0
 LinkIsElemFirstInGroup1[x1,3,6]: - EdgesAlign[x1,3,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,3,7]: - EdgesAlign[x1,3,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,4,5]: - EdgesAlign[x1,4,5]
   + IsElemFirstInGroup[x1,5] <= 0
 LinkIsElemFirstInGroup1[x1,4,6]: - EdgesAlign[x1,4,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,4,7]: - EdgesAlign[x1,4,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,5,6]: - EdgesAlign[x1,5,6]
   + IsElemFirstInGroup[x1,6] <= 0
 LinkIsElemFirstInGroup1[x1,5,7]: - EdgesAlign[x1,5,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[x1,6,7]: - EdgesAlign[x1,6,7]
   + IsElemFirstInGroup[x1,7] <= 0
 LinkIsElemFirstInGroup1[y1,0,1]: - EdgesAlign[y1,0,1]
   + IsElemFirstInGroup[y1,1] <= 0
 LinkIsElemFirstInGroup1[y1,0,2]: - EdgesAlign[y1,0,2]
   + IsElemFirstInGroup[y1,2] <= 0
 LinkIsElemFirstInGroup1[y1,0,3]: - EdgesAlign[y1,0,3]
   + IsElemFirstInGroup[y1,3] <= 0
 LinkIsElemFirstInGroup1[y1,0,4]: - EdgesAlign[y1,0,4]
   + IsElemFirstInGroup[y1,4] <= 0
 LinkIsElemFirstInGroup1[y1,0,5]: - EdgesAlign[y1,0,5]
   + IsElemFirstInGroup[y1,5] <= 0
 LinkIsElemFirstInGroup1[y1,0,6]: - EdgesAlign[y1,0,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,0,7]: - EdgesAlign[y1,0,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,1,2]: - EdgesAlign[y1,1,2]
   + IsElemFirstInGroup[y1,2] <= 0
 LinkIsElemFirstInGroup1[y1,1,3]: - EdgesAlign[y1,1,3]
   + IsElemFirstInGroup[y1,3] <= 0
 LinkIsElemFirstInGroup1[y1,1,4]: - EdgesAlign[y1,1,4]
   + IsElemFirstInGroup[y1,4] <= 0
 LinkIsElemFirstInGroup1[y1,1,5]: - EdgesAlign[y1,1,5]
   + IsElemFirstInGroup[y1,5] <= 0
 LinkIsElemFirstInGroup1[y1,1,6]: - EdgesAlign[y1,1,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,1,7]: - EdgesAlign[y1,1,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,2,3]: - EdgesAlign[y1,2,3]
   + IsElemFirstInGroup[y1,3] <= 0
 LinkIsElemFirstInGroup1[y1,2,4]: - EdgesAlign[y1,2,4]
   + IsElemFirstInGroup[y1,4] <= 0
 LinkIsElemFirstInGroup1[y1,2,5]: - EdgesAlign[y1,2,5]
   + IsElemFirstInGroup[y1,5] <= 0
 LinkIsElemFirstInGroup1[y1,2,6]: - EdgesAlign[y1,2,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,2,7]: - EdgesAlign[y1,2,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,3,4]: - EdgesAlign[y1,3,4]
   + IsElemFirstInGroup[y1,4] <= 0
 LinkIsElemFirstInGroup1[y1,3,5]: - EdgesAlign[y1,3,5]
   + IsElemFirstInGroup[y1,5] <= 0
 LinkIsElemFirstInGroup1[y1,3,6]: - EdgesAlign[y1,3,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,3,7]: - EdgesAlign[y1,3,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,4,5]: - EdgesAlign[y1,4,5]
   + IsElemFirstInGroup[y1,5] <= 0
 LinkIsElemFirstInGroup1[y1,4,6]: - EdgesAlign[y1,4,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,4,7]: - EdgesAlign[y1,4,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,5,6]: - EdgesAlign[y1,5,6]
   + IsElemFirstInGroup[y1,6] <= 0
 LinkIsElemFirstInGroup1[y1,5,7]: - EdgesAlign[y1,5,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkIsElemFirstInGroup1[y1,6,7]: - EdgesAlign[y1,6,7]
   + IsElemFirstInGroup[y1,7] <= 0
 LinkNumberOfGroups[x0]: - IsElemFirstInGroup[x0,0]
   - IsElemFirstInGroup[x0,1] - IsElemFirstInGroup[x0,2]
   - IsElemFirstInGroup[x0,3] - IsElemFirstInGroup[x0,4]
   - IsElemFirstInGroup[x0,5] - IsElemFirstInGroup[x0,6]
   - IsElemFirstInGroup[x0,7] + NumberOfGroups[x0] = 0
 LinkNumberOfGroups[y0]: - IsElemFirstInGroup[y0,0]
   - IsElemFirstInGroup[y0,1] - IsElemFirstInGroup[y0,2]
   - IsElemFirstInGroup[y0,3] - IsElemFirstInGroup[y0,4]
   - IsElemFirstInGroup[y0,5] - IsElemFirstInGroup[y0,6]
   - IsElemFirstInGroup[y0,7] + NumberOfGroups[y0] = 0
 LinkNumberOfGroups[x1]: - IsElemFirstInGroup[x1,0]
   - IsElemFirstInGroup[x1,1] - IsElemFirstInGroup[x1,2]
   - IsElemFirstInGroup[x1,3] - IsElemFirstInGroup[x1,4]
   - IsElemFirstInGroup[x1,5] - IsElemFirstInGroup[x1,6]
   - IsElemFirstInGroup[x1,7] + NumberOfGroups[x1] = 0
 LinkNumberOfGroups[y1]: - IsElemFirstInGroup[y1,0]
   - IsElemFirstInGroup[y1,1] - IsElemFirstInGroup[y1,2]
   - IsElemFirstInGroup[y1,3] - IsElemFirstInGroup[y1,4]
   - IsElemFirstInGroup[y1,5] - IsElemFirstInGroup[y1,6]
   - IsElemFirstInGroup[y1,7] + NumberOfGroups[y1] = 0
 PreventOvertOptimization: NumberOfGroups[x0] + NumberOfGroups[y0]
   + NumberOfGroups[x1] + NumberOfGroups[y1] >= 12
 R565: Overlap[0,0] + Overlap[0,1] + Overlap[0,2] + Overlap[0,3]
   + Overlap[0,4] + Overlap[0,5] + Overlap[0,6] + Overlap[0,7]
   + Overlap[1,0] + Overlap[1,1] + Overlap[1,2] + Overlap[1,3]
   + Overlap[1,4] + Overlap[1,5] + Overlap[1,6] + Overlap[1,7]
   + Overlap[2,0] + Overlap[2,1] + Overlap[2,2] + Overlap[2,3]
   + Overlap[2,4] + Overlap[2,5] + Overlap[2,6] + Overlap[2,7]
   + Overlap[3,0] + Overlap[3,1] + Overlap[3,2] + Overlap[3,3]
   + Overlap[3,4] + Overlap[3,5] + Overlap[3,6] + Overlap[3,7]
   + Overlap[4,0] + Overlap[4,1] + Overlap[4,2] + Overlap[4,3]
   + Overlap[4,4] + Overlap[4,5] + Overlap[4,6] + Overlap[4,7]
   + Overlap[5,0] + Overlap[5,1] + Overlap[5,2] + Overlap[5,3]
   + Overlap[5,4] + Overlap[5,5] + Overlap[5,6] + Overlap[5,7]
   + Overlap[6,0] + Overlap[6,1] + Overlap[6,2] + Overlap[6,3]
   + Overlap[6,4] + Overlap[6,5] + Overlap[6,6] + Overlap[6,7]
   + Overlap[7,0] + Overlap[7,1] + Overlap[7,2] + Overlap[7,3]
   + Overlap[7,4] + Overlap[7,5] + Overlap[7,6] + Overlap[7,7] = 0
 LinkAbove1[0,1]: Above[0,1] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[0,2]: Above[0,2] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[0,3]: Above[0,3] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[0,4]: Above[0,4] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[0,5]: Above[0,5] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[0,6]: Above[0,6] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[0,7]: Above[0,7] = 1 -> EdgeCoord[y1,0] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[1,0]: Above[1,0] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[1,2]: Above[1,2] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[1,3]: Above[1,3] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[1,4]: Above[1,4] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[1,5]: Above[1,5] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[1,6]: Above[1,6] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[1,7]: Above[1,7] = 1 -> EdgeCoord[y1,1] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[2,0]: Above[2,0] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[2,1]: Above[2,1] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[2,3]: Above[2,3] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[2,4]: Above[2,4] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[2,5]: Above[2,5] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[2,6]: Above[2,6] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[2,7]: Above[2,7] = 1 -> EdgeCoord[y1,2] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[3,0]: Above[3,0] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[3,1]: Above[3,1] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[3,2]: Above[3,2] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[3,4]: Above[3,4] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[3,5]: Above[3,5] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[3,6]: Above[3,6] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[3,7]: Above[3,7] = 1 -> EdgeCoord[y1,3] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[4,0]: Above[4,0] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[4,1]: Above[4,1] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[4,2]: Above[4,2] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[4,3]: Above[4,3] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[4,5]: Above[4,5] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[4,6]: Above[4,6] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[4,7]: Above[4,7] = 1 -> EdgeCoord[y1,4] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[5,0]: Above[5,0] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[5,1]: Above[5,1] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[5,2]: Above[5,2] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[5,3]: Above[5,3] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[5,4]: Above[5,4] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[5,6]: Above[5,6] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,6]
   <= -0
 LinkAbove1[5,7]: Above[5,7] = 1 -> EdgeCoord[y1,5] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[6,0]: Above[6,0] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[6,1]: Above[6,1] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[6,2]: Above[6,2] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[6,3]: Above[6,3] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[6,4]: Above[6,4] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[6,5]: Above[6,5] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[6,7]: Above[6,7] = 1 -> EdgeCoord[y1,6] - EdgeCoord[y0,7]
   <= -0
 LinkAbove1[7,0]: Above[7,0] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,0]
   <= -0
 LinkAbove1[7,1]: Above[7,1] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,1]
   <= -0
 LinkAbove1[7,2]: Above[7,2] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,2]
   <= -0
 LinkAbove1[7,3]: Above[7,3] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,3]
   <= -0
 LinkAbove1[7,4]: Above[7,4] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,4]
   <= -0
 LinkAbove1[7,5]: Above[7,5] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,5]
   <= -0
 LinkAbove1[7,6]: Above[7,6] = 1 -> EdgeCoord[y1,7] - EdgeCoord[y0,6]
   <= -0
 LinkAbove2[0,1]: Above[0,1] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[0,2]: Above[0,2] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[0,3]: Above[0,3] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[0,4]: Above[0,4] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[0,5]: Above[0,5] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[0,6]: Above[0,6] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[0,7]: Above[0,7] = 0 -> EdgeCoord[y1,0] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[1,0]: Above[1,0] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[1,2]: Above[1,2] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[1,3]: Above[1,3] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[1,4]: Above[1,4] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[1,5]: Above[1,5] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[1,6]: Above[1,6] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[1,7]: Above[1,7] = 0 -> EdgeCoord[y1,1] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[2,0]: Above[2,0] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[2,1]: Above[2,1] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[2,3]: Above[2,3] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[2,4]: Above[2,4] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[2,5]: Above[2,5] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[2,6]: Above[2,6] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[2,7]: Above[2,7] = 0 -> EdgeCoord[y1,2] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[3,0]: Above[3,0] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[3,1]: Above[3,1] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[3,2]: Above[3,2] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[3,4]: Above[3,4] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[3,5]: Above[3,5] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[3,6]: Above[3,6] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[3,7]: Above[3,7] = 0 -> EdgeCoord[y1,3] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[4,0]: Above[4,0] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[4,1]: Above[4,1] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[4,2]: Above[4,2] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[4,3]: Above[4,3] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[4,5]: Above[4,5] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[4,6]: Above[4,6] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[4,7]: Above[4,7] = 0 -> EdgeCoord[y1,4] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[5,0]: Above[5,0] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[5,1]: Above[5,1] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[5,2]: Above[5,2] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[5,3]: Above[5,3] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[5,4]: Above[5,4] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[5,6]: Above[5,6] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,6]
   >= -1
 LinkAbove2[5,7]: Above[5,7] = 0 -> EdgeCoord[y1,5] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[6,0]: Above[6,0] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[6,1]: Above[6,1] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[6,2]: Above[6,2] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[6,3]: Above[6,3] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[6,4]: Above[6,4] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[6,5]: Above[6,5] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[6,7]: Above[6,7] = 0 -> EdgeCoord[y1,6] - EdgeCoord[y0,7]
   >= -1
 LinkAbove2[7,0]: Above[7,0] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,0]
   >= -1
 LinkAbove2[7,1]: Above[7,1] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,1]
   >= -1
 LinkAbove2[7,2]: Above[7,2] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,2]
   >= -1
 LinkAbove2[7,3]: Above[7,3] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,3]
   >= -1
 LinkAbove2[7,4]: Above[7,4] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,4]
   >= -1
 LinkAbove2[7,5]: Above[7,5] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,5]
   >= -1
 LinkAbove2[7,6]: Above[7,6] = 0 -> EdgeCoord[y1,7] - EdgeCoord[y0,6]
   >= -1
 LinkOnLeft1[0,1]: OnLeft[0,1] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[0,2]: OnLeft[0,2] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[0,3]: OnLeft[0,3] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[0,4]: OnLeft[0,4] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[0,5]: OnLeft[0,5] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[0,6]: OnLeft[0,6] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[0,7]: OnLeft[0,7] = 1 -> EdgeCoord[x1,0] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[1,0]: OnLeft[1,0] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[1,2]: OnLeft[1,2] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[1,3]: OnLeft[1,3] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[1,4]: OnLeft[1,4] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[1,5]: OnLeft[1,5] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[1,6]: OnLeft[1,6] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[1,7]: OnLeft[1,7] = 1 -> EdgeCoord[x1,1] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[2,0]: OnLeft[2,0] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[2,1]: OnLeft[2,1] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[2,3]: OnLeft[2,3] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[2,4]: OnLeft[2,4] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[2,5]: OnLeft[2,5] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[2,6]: OnLeft[2,6] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[2,7]: OnLeft[2,7] = 1 -> EdgeCoord[x1,2] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[3,0]: OnLeft[3,0] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[3,1]: OnLeft[3,1] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[3,2]: OnLeft[3,2] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[3,4]: OnLeft[3,4] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[3,5]: OnLeft[3,5] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[3,6]: OnLeft[3,6] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[3,7]: OnLeft[3,7] = 1 -> EdgeCoord[x1,3] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[4,0]: OnLeft[4,0] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[4,1]: OnLeft[4,1] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[4,2]: OnLeft[4,2] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[4,3]: OnLeft[4,3] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[4,5]: OnLeft[4,5] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[4,6]: OnLeft[4,6] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[4,7]: OnLeft[4,7] = 1 -> EdgeCoord[x1,4] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[5,0]: OnLeft[5,0] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[5,1]: OnLeft[5,1] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[5,2]: OnLeft[5,2] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[5,3]: OnLeft[5,3] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[5,4]: OnLeft[5,4] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[5,6]: OnLeft[5,6] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft1[5,7]: OnLeft[5,7] = 1 -> EdgeCoord[x1,5] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[6,0]: OnLeft[6,0] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[6,1]: OnLeft[6,1] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[6,2]: OnLeft[6,2] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[6,3]: OnLeft[6,3] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[6,4]: OnLeft[6,4] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[6,5]: OnLeft[6,5] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[6,7]: OnLeft[6,7] = 1 -> EdgeCoord[x1,6] - EdgeCoord[x0,7]
   <= -0
 LinkOnLeft1[7,0]: OnLeft[7,0] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,0]
   <= -0
 LinkOnLeft1[7,1]: OnLeft[7,1] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,1]
   <= -0
 LinkOnLeft1[7,2]: OnLeft[7,2] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,2]
   <= -0
 LinkOnLeft1[7,3]: OnLeft[7,3] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,3]
   <= -0
 LinkOnLeft1[7,4]: OnLeft[7,4] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,4]
   <= -0
 LinkOnLeft1[7,5]: OnLeft[7,5] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,5]
   <= -0
 LinkOnLeft1[7,6]: OnLeft[7,6] = 1 -> EdgeCoord[x1,7] - EdgeCoord[x0,6]
   <= -0
 LinkOnLeft2[0,1]: OnLeft[0,1] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[0,2]: OnLeft[0,2] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[0,3]: OnLeft[0,3] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[0,4]: OnLeft[0,4] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[0,5]: OnLeft[0,5] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[0,6]: OnLeft[0,6] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[0,7]: OnLeft[0,7] = 0 -> EdgeCoord[x1,0] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[1,0]: OnLeft[1,0] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[1,2]: OnLeft[1,2] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[1,3]: OnLeft[1,3] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[1,4]: OnLeft[1,4] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[1,5]: OnLeft[1,5] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[1,6]: OnLeft[1,6] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[1,7]: OnLeft[1,7] = 0 -> EdgeCoord[x1,1] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[2,0]: OnLeft[2,0] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[2,1]: OnLeft[2,1] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[2,3]: OnLeft[2,3] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[2,4]: OnLeft[2,4] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[2,5]: OnLeft[2,5] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[2,6]: OnLeft[2,6] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[2,7]: OnLeft[2,7] = 0 -> EdgeCoord[x1,2] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[3,0]: OnLeft[3,0] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[3,1]: OnLeft[3,1] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[3,2]: OnLeft[3,2] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[3,4]: OnLeft[3,4] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[3,5]: OnLeft[3,5] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[3,6]: OnLeft[3,6] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[3,7]: OnLeft[3,7] = 0 -> EdgeCoord[x1,3] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[4,0]: OnLeft[4,0] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[4,1]: OnLeft[4,1] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[4,2]: OnLeft[4,2] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[4,3]: OnLeft[4,3] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[4,5]: OnLeft[4,5] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[4,6]: OnLeft[4,6] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[4,7]: OnLeft[4,7] = 0 -> EdgeCoord[x1,4] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[5,0]: OnLeft[5,0] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[5,1]: OnLeft[5,1] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[5,2]: OnLeft[5,2] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[5,3]: OnLeft[5,3] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[5,4]: OnLeft[5,4] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[5,6]: OnLeft[5,6] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,6]
   >= -1
 LinkOnLeft2[5,7]: OnLeft[5,7] = 0 -> EdgeCoord[x1,5] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[6,0]: OnLeft[6,0] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[6,1]: OnLeft[6,1] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[6,2]: OnLeft[6,2] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[6,3]: OnLeft[6,3] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[6,4]: OnLeft[6,4] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[6,5]: OnLeft[6,5] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[6,7]: OnLeft[6,7] = 0 -> EdgeCoord[x1,6] - EdgeCoord[x0,7]
   >= -1
 LinkOnLeft2[7,0]: OnLeft[7,0] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,0]
   >= -1
 LinkOnLeft2[7,1]: OnLeft[7,1] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,1]
   >= -1
 LinkOnLeft2[7,2]: OnLeft[7,2] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,2]
   >= -1
 LinkOnLeft2[7,3]: OnLeft[7,3] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,3]
   >= -1
 LinkOnLeft2[7,4]: OnLeft[7,4] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,4]
   >= -1
 LinkOnLeft2[7,5]: OnLeft[7,5] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,5]
   >= -1
 LinkOnLeft2[7,6]: OnLeft[7,6] = 0 -> EdgeCoord[x1,7] - EdgeCoord[x0,6]
   >= -1
 LinkIsElemFirstInGroup2[x0,0]: IsElemFirstInGroup[x0,0] = 0 -> <= -1
 LinkIsElemFirstInGroup2[x0,1]: IsElemFirstInGroup[x0,1] = 0 ->
   EdgesAlign[x0,0,1] <= -0
 LinkIsElemFirstInGroup2[x0,2]: IsElemFirstInGroup[x0,2] = 0 ->
   EdgesAlign[x0,0,2] + EdgesAlign[x0,1,2] <= 1
 LinkIsElemFirstInGroup2[x0,3]: IsElemFirstInGroup[x0,3] = 0 ->
   EdgesAlign[x0,0,3] + EdgesAlign[x0,1,3] + EdgesAlign[x0,2,3] <= 2
 LinkIsElemFirstInGroup2[x0,4]: IsElemFirstInGroup[x0,4] = 0 ->
   EdgesAlign[x0,0,4] + EdgesAlign[x0,1,4] + EdgesAlign[x0,2,4]
   + EdgesAlign[x0,3,4] <= 3
 LinkIsElemFirstInGroup2[x0,5]: IsElemFirstInGroup[x0,5] = 0 ->
   EdgesAlign[x0,0,5] + EdgesAlign[x0,1,5] + EdgesAlign[x0,2,5]
   + EdgesAlign[x0,3,5] + EdgesAlign[x0,4,5] <= 4
 LinkIsElemFirstInGroup2[x0,6]: IsElemFirstInGroup[x0,6] = 0 ->
   EdgesAlign[x0,0,6] + EdgesAlign[x0,1,6] + EdgesAlign[x0,2,6]
   + EdgesAlign[x0,3,6] + EdgesAlign[x0,4,6] + EdgesAlign[x0,5,6] <= 5
 LinkIsElemFirstInGroup2[x0,7]: IsElemFirstInGroup[x0,7] = 0 ->
   EdgesAlign[x0,0,7] + EdgesAlign[x0,1,7] + EdgesAlign[x0,2,7]
   + EdgesAlign[x0,3,7] + EdgesAlign[x0,4,7] + EdgesAlign[x0,5,7]
   + EdgesAlign[x0,6,7] <= 6
 LinkIsElemFirstInGroup2[y0,0]: IsElemFirstInGroup[y0,0] = 0 -> <= -1
 LinkIsElemFirstInGroup2[y0,1]: IsElemFirstInGroup[y0,1] = 0 ->
   EdgesAlign[y0,0,1] <= -0
 LinkIsElemFirstInGroup2[y0,2]: IsElemFirstInGroup[y0,2] = 0 ->
   EdgesAlign[y0,0,2] + EdgesAlign[y0,1,2] <= 1
 LinkIsElemFirstInGroup2[y0,3]: IsElemFirstInGroup[y0,3] = 0 ->
   EdgesAlign[y0,0,3] + EdgesAlign[y0,1,3] + EdgesAlign[y0,2,3] <= 2
 LinkIsElemFirstInGroup2[y0,4]: IsElemFirstInGroup[y0,4] = 0 ->
   EdgesAlign[y0,0,4] + EdgesAlign[y0,1,4] + EdgesAlign[y0,2,4]
   + EdgesAlign[y0,3,4] <= 3
 LinkIsElemFirstInGroup2[y0,5]: IsElemFirstInGroup[y0,5] = 0 ->
   EdgesAlign[y0,0,5] + EdgesAlign[y0,1,5] + EdgesAlign[y0,2,5]
   + EdgesAlign[y0,3,5] + EdgesAlign[y0,4,5] <= 4
 LinkIsElemFirstInGroup2[y0,6]: IsElemFirstInGroup[y0,6] = 0 ->
   EdgesAlign[y0,0,6] + EdgesAlign[y0,1,6] + EdgesAlign[y0,2,6]
   + EdgesAlign[y0,3,6] + EdgesAlign[y0,4,6] + EdgesAlign[y0,5,6] <= 5
 LinkIsElemFirstInGroup2[y0,7]: IsElemFirstInGroup[y0,7] = 0 ->
   EdgesAlign[y0,0,7] + EdgesAlign[y0,1,7] + EdgesAlign[y0,2,7]
   + EdgesAlign[y0,3,7] + EdgesAlign[y0,4,7] + EdgesAlign[y0,5,7]
   + EdgesAlign[y0,6,7] <= 6
 LinkIsElemFirstInGroup2[x1,0]: IsElemFirstInGroup[x1,0] = 0 -> <= -1
 LinkIsElemFirstInGroup2[x1,1]: IsElemFirstInGroup[x1,1] = 0 ->
   EdgesAlign[x1,0,1] <= -0
 LinkIsElemFirstInGroup2[x1,2]: IsElemFirstInGroup[x1,2] = 0 ->
   EdgesAlign[x1,0,2] + EdgesAlign[x1,1,2] <= 1
 LinkIsElemFirstInGroup2[x1,3]: IsElemFirstInGroup[x1,3] = 0 ->
   EdgesAlign[x1,0,3] + EdgesAlign[x1,1,3] + EdgesAlign[x1,2,3] <= 2
 LinkIsElemFirstInGroup2[x1,4]: IsElemFirstInGroup[x1,4] = 0 ->
   EdgesAlign[x1,0,4] + EdgesAlign[x1,1,4] + EdgesAlign[x1,2,4]
   + EdgesAlign[x1,3,4] <= 3
 LinkIsElemFirstInGroup2[x1,5]: IsElemFirstInGroup[x1,5] = 0 ->
   EdgesAlign[x1,0,5] + EdgesAlign[x1,1,5] + EdgesAlign[x1,2,5]
   + EdgesAlign[x1,3,5] + EdgesAlign[x1,4,5] <= 4
 LinkIsElemFirstInGroup2[x1,6]: IsElemFirstInGroup[x1,6] = 0 ->
   EdgesAlign[x1,0,6] + EdgesAlign[x1,1,6] + EdgesAlign[x1,2,6]
   + EdgesAlign[x1,3,6] + EdgesAlign[x1,4,6] + EdgesAlign[x1,5,6] <= 5
 LinkIsElemFirstInGroup2[x1,7]: IsElemFirstInGroup[x1,7] = 0 ->
   EdgesAlign[x1,0,7] + EdgesAlign[x1,1,7] + EdgesAlign[x1,2,7]
   + EdgesAlign[x1,3,7] + EdgesAlign[x1,4,7] + EdgesAlign[x1,5,7]
   + EdgesAlign[x1,6,7] <= 6
 LinkIsElemFirstInGroup2[y1,0]: IsElemFirstInGroup[y1,0] = 0 -> <= -1
 LinkIsElemFirstInGroup2[y1,1]: IsElemFirstInGroup[y1,1] = 0 ->
   EdgesAlign[y1,0,1] <= -0
 LinkIsElemFirstInGroup2[y1,2]: IsElemFirstInGroup[y1,2] = 0 ->
   EdgesAlign[y1,0,2] + EdgesAlign[y1,1,2] <= 1
 LinkIsElemFirstInGroup2[y1,3]: IsElemFirstInGroup[y1,3] = 0 ->
   EdgesAlign[y1,0,3] + EdgesAlign[y1,1,3] + EdgesAlign[y1,2,3] <= 2
 LinkIsElemFirstInGroup2[y1,4]: IsElemFirstInGroup[y1,4] = 0 ->
   EdgesAlign[y1,0,4] + EdgesAlign[y1,1,4] + EdgesAlign[y1,2,4]
   + EdgesAlign[y1,3,4] <= 3
 LinkIsElemFirstInGroup2[y1,5]: IsElemFirstInGroup[y1,5] = 0 ->
   EdgesAlign[y1,0,5] + EdgesAlign[y1,1,5] + EdgesAlign[y1,2,5]
   + EdgesAlign[y1,3,5] + EdgesAlign[y1,4,5] <= 4
 LinkIsElemFirstInGroup2[y1,6]: IsElemFirstInGroup[y1,6] = 0 ->
   EdgesAlign[y1,0,6] + EdgesAlign[y1,1,6] + EdgesAlign[y1,2,6]
   + EdgesAlign[y1,3,6] + EdgesAlign[y1,4,6] + EdgesAlign[y1,5,6] <= 5
 LinkIsElemFirstInGroup2[y1,7]: IsElemFirstInGroup[y1,7] = 0 ->
   EdgesAlign[y1,0,7] + EdgesAlign[y1,1,7] + EdgesAlign[y1,2,7]
   + EdgesAlign[y1,3,7] + EdgesAlign[y1,4,7] + EdgesAlign[y1,5,7]
   + EdgesAlign[y1,6,7] <= 6
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
 OnLeft[7,6] OnLeft[7,7] HorizontalOverlap[0,0] HorizontalOverlap[0,1]
 HorizontalOverlap[0,2] HorizontalOverlap[0,3] HorizontalOverlap[0,4]
 HorizontalOverlap[0,5] HorizontalOverlap[0,6] HorizontalOverlap[0,7]
 HorizontalOverlap[1,0] HorizontalOverlap[1,1] HorizontalOverlap[1,2]
 HorizontalOverlap[1,3] HorizontalOverlap[1,4] HorizontalOverlap[1,5]
 HorizontalOverlap[1,6] HorizontalOverlap[1,7] HorizontalOverlap[2,0]
 HorizontalOverlap[2,1] HorizontalOverlap[2,2] HorizontalOverlap[2,3]
 HorizontalOverlap[2,4] HorizontalOverlap[2,5] HorizontalOverlap[2,6]
 HorizontalOverlap[2,7] HorizontalOverlap[3,0] HorizontalOverlap[3,1]
 HorizontalOverlap[3,2] HorizontalOverlap[3,3] HorizontalOverlap[3,4]
 HorizontalOverlap[3,5] HorizontalOverlap[3,6] HorizontalOverlap[3,7]
 HorizontalOverlap[4,0] HorizontalOverlap[4,1] HorizontalOverlap[4,2]
 HorizontalOverlap[4,3] HorizontalOverlap[4,4] HorizontalOverlap[4,5]
 HorizontalOverlap[4,6] HorizontalOverlap[4,7] HorizontalOverlap[5,0]
 HorizontalOverlap[5,1] HorizontalOverlap[5,2] HorizontalOverlap[5,3]
 HorizontalOverlap[5,4] HorizontalOverlap[5,5] HorizontalOverlap[5,6]
 HorizontalOverlap[5,7] HorizontalOverlap[6,0] HorizontalOverlap[6,1]
 HorizontalOverlap[6,2] HorizontalOverlap[6,3] HorizontalOverlap[6,4]
 HorizontalOverlap[6,5] HorizontalOverlap[6,6] HorizontalOverlap[6,7]
 HorizontalOverlap[7,0] HorizontalOverlap[7,1] HorizontalOverlap[7,2]
 HorizontalOverlap[7,3] HorizontalOverlap[7,4] HorizontalOverlap[7,5]
 HorizontalOverlap[7,6] HorizontalOverlap[7,7] VerticalOverlap[0,0]
 VerticalOverlap[0,1] VerticalOverlap[0,2] VerticalOverlap[0,3]
 VerticalOverlap[0,4] VerticalOverlap[0,5] VerticalOverlap[0,6]
 VerticalOverlap[0,7] VerticalOverlap[1,0] VerticalOverlap[1,1]
 VerticalOverlap[1,2] VerticalOverlap[1,3] VerticalOverlap[1,4]
 VerticalOverlap[1,5] VerticalOverlap[1,6] VerticalOverlap[1,7]
 VerticalOverlap[2,0] VerticalOverlap[2,1] VerticalOverlap[2,2]
 VerticalOverlap[2,3] VerticalOverlap[2,4] VerticalOverlap[2,5]
 VerticalOverlap[2,6] VerticalOverlap[2,7] VerticalOverlap[3,0]
 VerticalOverlap[3,1] VerticalOverlap[3,2] VerticalOverlap[3,3]
 VerticalOverlap[3,4] VerticalOverlap[3,5] VerticalOverlap[3,6]
 VerticalOverlap[3,7] VerticalOverlap[4,0] VerticalOverlap[4,1]
 VerticalOverlap[4,2] VerticalOverlap[4,3] VerticalOverlap[4,4]
 VerticalOverlap[4,5] VerticalOverlap[4,6] VerticalOverlap[4,7]
 VerticalOverlap[5,0] VerticalOverlap[5,1] VerticalOverlap[5,2]
 VerticalOverlap[5,3] VerticalOverlap[5,4] VerticalOverlap[5,5]
 VerticalOverlap[5,6] VerticalOverlap[5,7] VerticalOverlap[6,0]
 VerticalOverlap[6,1] VerticalOverlap[6,2] VerticalOverlap[6,3]
 VerticalOverlap[6,4] VerticalOverlap[6,5] VerticalOverlap[6,6]
 VerticalOverlap[6,7] VerticalOverlap[7,0] VerticalOverlap[7,1]
 VerticalOverlap[7,2] VerticalOverlap[7,3] VerticalOverlap[7,4]
 VerticalOverlap[7,5] VerticalOverlap[7,6] VerticalOverlap[7,7]
 Overlap[0,0] Overlap[0,1] Overlap[0,2] Overlap[0,3] Overlap[0,4]
 Overlap[0,5] Overlap[0,6] Overlap[0,7] Overlap[1,0] Overlap[1,1]
 Overlap[1,2] Overlap[1,3] Overlap[1,4] Overlap[1,5] Overlap[1,6]
 Overlap[1,7] Overlap[2,0] Overlap[2,1] Overlap[2,2] Overlap[2,3]
 Overlap[2,4] Overlap[2,5] Overlap[2,6] Overlap[2,7] Overlap[3,0]
 Overlap[3,1] Overlap[3,2] Overlap[3,3] Overlap[3,4] Overlap[3,5]
 Overlap[3,6] Overlap[3,7] Overlap[4,0] Overlap[4,1] Overlap[4,2]
 Overlap[4,3] Overlap[4,4] Overlap[4,5] Overlap[4,6] Overlap[4,7]
 Overlap[5,0] Overlap[5,1] Overlap[5,2] Overlap[5,3] Overlap[5,4]
 Overlap[5,5] Overlap[5,6] Overlap[5,7] Overlap[6,0] Overlap[6,1]
 Overlap[6,2] Overlap[6,3] Overlap[6,4] Overlap[6,5] Overlap[6,6]
 Overlap[6,7] Overlap[7,0] Overlap[7,1] Overlap[7,2] Overlap[7,3]
 Overlap[7,4] Overlap[7,5] Overlap[7,6] Overlap[7,7] EdgesAlign[x0,0,0]
 EdgesAlign[x0,0,1] EdgesAlign[x0,0,2] EdgesAlign[x0,0,3]
 EdgesAlign[x0,0,4] EdgesAlign[x0,0,5] EdgesAlign[x0,0,6]
 EdgesAlign[x0,0,7] EdgesAlign[x0,1,0] EdgesAlign[x0,1,1]
 EdgesAlign[x0,1,2] EdgesAlign[x0,1,3] EdgesAlign[x0,1,4]
 EdgesAlign[x0,1,5] EdgesAlign[x0,1,6] EdgesAlign[x0,1,7]
 EdgesAlign[x0,2,0] EdgesAlign[x0,2,1] EdgesAlign[x0,2,2]
 EdgesAlign[x0,2,3] EdgesAlign[x0,2,4] EdgesAlign[x0,2,5]
 EdgesAlign[x0,2,6] EdgesAlign[x0,2,7] EdgesAlign[x0,3,0]
 EdgesAlign[x0,3,1] EdgesAlign[x0,3,2] EdgesAlign[x0,3,3]
 EdgesAlign[x0,3,4] EdgesAlign[x0,3,5] EdgesAlign[x0,3,6]
 EdgesAlign[x0,3,7] EdgesAlign[x0,4,0] EdgesAlign[x0,4,1]
 EdgesAlign[x0,4,2] EdgesAlign[x0,4,3] EdgesAlign[x0,4,4]
 EdgesAlign[x0,4,5] EdgesAlign[x0,4,6] EdgesAlign[x0,4,7]
 EdgesAlign[x0,5,0] EdgesAlign[x0,5,1] EdgesAlign[x0,5,2]
 EdgesAlign[x0,5,3] EdgesAlign[x0,5,4] EdgesAlign[x0,5,5]
 EdgesAlign[x0,5,6] EdgesAlign[x0,5,7] EdgesAlign[x0,6,0]
 EdgesAlign[x0,6,1] EdgesAlign[x0,6,2] EdgesAlign[x0,6,3]
 EdgesAlign[x0,6,4] EdgesAlign[x0,6,5] EdgesAlign[x0,6,6]
 EdgesAlign[x0,6,7] EdgesAlign[x0,7,0] EdgesAlign[x0,7,1]
 EdgesAlign[x0,7,2] EdgesAlign[x0,7,3] EdgesAlign[x0,7,4]
 EdgesAlign[x0,7,5] EdgesAlign[x0,7,6] EdgesAlign[x0,7,7]
 EdgesAlign[y0,0,0] EdgesAlign[y0,0,1] EdgesAlign[y0,0,2]
 EdgesAlign[y0,0,3] EdgesAlign[y0,0,4] EdgesAlign[y0,0,5]
 EdgesAlign[y0,0,6] EdgesAlign[y0,0,7] EdgesAlign[y0,1,0]
 EdgesAlign[y0,1,1] EdgesAlign[y0,1,2] EdgesAlign[y0,1,3]
 EdgesAlign[y0,1,4] EdgesAlign[y0,1,5] EdgesAlign[y0,1,6]
 EdgesAlign[y0,1,7] EdgesAlign[y0,2,0] EdgesAlign[y0,2,1]
 EdgesAlign[y0,2,2] EdgesAlign[y0,2,3] EdgesAlign[y0,2,4]
 EdgesAlign[y0,2,5] EdgesAlign[y0,2,6] EdgesAlign[y0,2,7]
 EdgesAlign[y0,3,0] EdgesAlign[y0,3,1] EdgesAlign[y0,3,2]
 EdgesAlign[y0,3,3] EdgesAlign[y0,3,4] EdgesAlign[y0,3,5]
 EdgesAlign[y0,3,6] EdgesAlign[y0,3,7] EdgesAlign[y0,4,0]
 EdgesAlign[y0,4,1] EdgesAlign[y0,4,2] EdgesAlign[y0,4,3]
 EdgesAlign[y0,4,4] EdgesAlign[y0,4,5] EdgesAlign[y0,4,6]
 EdgesAlign[y0,4,7] EdgesAlign[y0,5,0] EdgesAlign[y0,5,1]
 EdgesAlign[y0,5,2] EdgesAlign[y0,5,3] EdgesAlign[y0,5,4]
 EdgesAlign[y0,5,5] EdgesAlign[y0,5,6] EdgesAlign[y0,5,7]
 EdgesAlign[y0,6,0] EdgesAlign[y0,6,1] EdgesAlign[y0,6,2]
 EdgesAlign[y0,6,3] EdgesAlign[y0,6,4] EdgesAlign[y0,6,5]
 EdgesAlign[y0,6,6] EdgesAlign[y0,6,7] EdgesAlign[y0,7,0]
 EdgesAlign[y0,7,1] EdgesAlign[y0,7,2] EdgesAlign[y0,7,3]
 EdgesAlign[y0,7,4] EdgesAlign[y0,7,5] EdgesAlign[y0,7,6]
 EdgesAlign[y0,7,7] EdgesAlign[x1,0,0] EdgesAlign[x1,0,1]
 EdgesAlign[x1,0,2] EdgesAlign[x1,0,3] EdgesAlign[x1,0,4]
 EdgesAlign[x1,0,5] EdgesAlign[x1,0,6] EdgesAlign[x1,0,7]
 EdgesAlign[x1,1,0] EdgesAlign[x1,1,1] EdgesAlign[x1,1,2]
 EdgesAlign[x1,1,3] EdgesAlign[x1,1,4] EdgesAlign[x1,1,5]
 EdgesAlign[x1,1,6] EdgesAlign[x1,1,7] EdgesAlign[x1,2,0]
 EdgesAlign[x1,2,1] EdgesAlign[x1,2,2] EdgesAlign[x1,2,3]
 EdgesAlign[x1,2,4] EdgesAlign[x1,2,5] EdgesAlign[x1,2,6]
 EdgesAlign[x1,2,7] EdgesAlign[x1,3,0] EdgesAlign[x1,3,1]
 EdgesAlign[x1,3,2] EdgesAlign[x1,3,3] EdgesAlign[x1,3,4]
 EdgesAlign[x1,3,5] EdgesAlign[x1,3,6] EdgesAlign[x1,3,7]
 EdgesAlign[x1,4,0] EdgesAlign[x1,4,1] EdgesAlign[x1,4,2]
 EdgesAlign[x1,4,3] EdgesAlign[x1,4,4] EdgesAlign[x1,4,5]
 EdgesAlign[x1,4,6] EdgesAlign[x1,4,7] EdgesAlign[x1,5,0]
 EdgesAlign[x1,5,1] EdgesAlign[x1,5,2] EdgesAlign[x1,5,3]
 EdgesAlign[x1,5,4] EdgesAlign[x1,5,5] EdgesAlign[x1,5,6]
 EdgesAlign[x1,5,7] EdgesAlign[x1,6,0] EdgesAlign[x1,6,1]
 EdgesAlign[x1,6,2] EdgesAlign[x1,6,3] EdgesAlign[x1,6,4]
 EdgesAlign[x1,6,5] EdgesAlign[x1,6,6] EdgesAlign[x1,6,7]
 EdgesAlign[x1,7,0] EdgesAlign[x1,7,1] EdgesAlign[x1,7,2]
 EdgesAlign[x1,7,3] EdgesAlign[x1,7,4] EdgesAlign[x1,7,5]
 EdgesAlign[x1,7,6] EdgesAlign[x1,7,7] EdgesAlign[y1,0,0]
 EdgesAlign[y1,0,1] EdgesAlign[y1,0,2] EdgesAlign[y1,0,3]
 EdgesAlign[y1,0,4] EdgesAlign[y1,0,5] EdgesAlign[y1,0,6]
 EdgesAlign[y1,0,7] EdgesAlign[y1,1,0] EdgesAlign[y1,1,1]
 EdgesAlign[y1,1,2] EdgesAlign[y1,1,3] EdgesAlign[y1,1,4]
 EdgesAlign[y1,1,5] EdgesAlign[y1,1,6] EdgesAlign[y1,1,7]
 EdgesAlign[y1,2,0] EdgesAlign[y1,2,1] EdgesAlign[y1,2,2]
 EdgesAlign[y1,2,3] EdgesAlign[y1,2,4] EdgesAlign[y1,2,5]
 EdgesAlign[y1,2,6] EdgesAlign[y1,2,7] EdgesAlign[y1,3,0]
 EdgesAlign[y1,3,1] EdgesAlign[y1,3,2] EdgesAlign[y1,3,3]
 EdgesAlign[y1,3,4] EdgesAlign[y1,3,5] EdgesAlign[y1,3,6]
 EdgesAlign[y1,3,7] EdgesAlign[y1,4,0] EdgesAlign[y1,4,1]
 EdgesAlign[y1,4,2] EdgesAlign[y1,4,3] EdgesAlign[y1,4,4]
 EdgesAlign[y1,4,5] EdgesAlign[y1,4,6] EdgesAlign[y1,4,7]
 EdgesAlign[y1,5,0] EdgesAlign[y1,5,1] EdgesAlign[y1,5,2]
 EdgesAlign[y1,5,3] EdgesAlign[y1,5,4] EdgesAlign[y1,5,5]
 EdgesAlign[y1,5,6] EdgesAlign[y1,5,7] EdgesAlign[y1,6,0]
 EdgesAlign[y1,6,1] EdgesAlign[y1,6,2] EdgesAlign[y1,6,3]
 EdgesAlign[y1,6,4] EdgesAlign[y1,6,5] EdgesAlign[y1,6,6]
 EdgesAlign[y1,6,7] EdgesAlign[y1,7,0] EdgesAlign[y1,7,1]
 EdgesAlign[y1,7,2] EdgesAlign[y1,7,3] EdgesAlign[y1,7,4]
 EdgesAlign[y1,7,5] EdgesAlign[y1,7,6] EdgesAlign[y1,7,7]
 IsElemFirstInGroup[x0,0] IsElemFirstInGroup[x0,1] IsElemFirstInGroup[x0,2]
 IsElemFirstInGroup[x0,3] IsElemFirstInGroup[x0,4] IsElemFirstInGroup[x0,5]
 IsElemFirstInGroup[x0,6] IsElemFirstInGroup[x0,7] IsElemFirstInGroup[y0,0]
 IsElemFirstInGroup[y0,1] IsElemFirstInGroup[y0,2] IsElemFirstInGroup[y0,3]
 IsElemFirstInGroup[y0,4] IsElemFirstInGroup[y0,5] IsElemFirstInGroup[y0,6]
 IsElemFirstInGroup[y0,7] IsElemFirstInGroup[x1,0] IsElemFirstInGroup[x1,1]
 IsElemFirstInGroup[x1,2] IsElemFirstInGroup[x1,3] IsElemFirstInGroup[x1,4]
 IsElemFirstInGroup[x1,5] IsElemFirstInGroup[x1,6] IsElemFirstInGroup[x1,7]
 IsElemFirstInGroup[y1,0] IsElemFirstInGroup[y1,1] IsElemFirstInGroup[y1,2]
 IsElemFirstInGroup[y1,3] IsElemFirstInGroup[y1,4] IsElemFirstInGroup[y1,5]
 IsElemFirstInGroup[y1,6] IsElemFirstInGroup[y1,7]
Generals
 EdgeCoord[x0,0] EdgeCoord[x0,1] EdgeCoord[x0,2] EdgeCoord[x0,3]
 EdgeCoord[x0,4] EdgeCoord[x0,5] EdgeCoord[x0,6] EdgeCoord[x0,7]
 EdgeCoord[y0,0] EdgeCoord[y0,1] EdgeCoord[y0,2] EdgeCoord[y0,3]
 EdgeCoord[y0,4] EdgeCoord[y0,5] EdgeCoord[y0,6] EdgeCoord[y0,7]
 EdgeCoord[x1,0] EdgeCoord[x1,1] EdgeCoord[x1,2] EdgeCoord[x1,3]
 EdgeCoord[x1,4] EdgeCoord[x1,5] EdgeCoord[x1,6] EdgeCoord[x1,7]
 EdgeCoord[y1,0] EdgeCoord[y1,1] EdgeCoord[y1,2] EdgeCoord[y1,3]
 EdgeCoord[y1,4] EdgeCoord[y1,5] EdgeCoord[y1,6] EdgeCoord[y1,7] W[0] W[1]
 W[2] W[3] W[4] W[5] W[6] W[7] H[0] H[1] H[2] H[3] H[4] H[5] H[6] H[7]
 ResizeW[0] ResizeW[1] ResizeW[2] ResizeW[3] ResizeW[4] ResizeW[5]
 ResizeW[6] ResizeW[7] ResizeWAbs[0] ResizeWAbs[1] ResizeWAbs[2]
 ResizeWAbs[3] ResizeWAbs[4] ResizeWAbs[5] ResizeWAbs[6] ResizeWAbs[7]
 ResizeH[0] ResizeH[1] ResizeH[2] ResizeH[3] ResizeH[4] ResizeH[5]
 ResizeH[6] ResizeH[7] ResizeHAbs[0] ResizeHAbs[1] ResizeHAbs[2]
 ResizeHAbs[3] ResizeHAbs[4] ResizeHAbs[5] ResizeHAbs[6] ResizeHAbs[7]
 MoveX[0] MoveX[1] MoveX[2] MoveX[3] MoveX[4] MoveX[5] MoveX[6] MoveX[7]
 MoveXAbs[0] MoveXAbs[1] MoveXAbs[2] MoveXAbs[3] MoveXAbs[4] MoveXAbs[5]
 MoveXAbs[6] MoveXAbs[7] MoveY[0] MoveY[1] MoveY[2] MoveY[3] MoveY[4]
 MoveY[5] MoveY[6] MoveY[7] MoveYAbs[0] MoveYAbs[1] MoveYAbs[2] MoveYAbs[3]
 MoveYAbs[4] MoveYAbs[5] MoveYAbs[6] MoveYAbs[7] EdgeDistance[x0,0,0]
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
 EdgeDistance[y1,7,5] EdgeDistance[y1,7,6] EdgeDistance[y1,7,7]
 EdgeDistanceAbs[x0,0,0] EdgeDistanceAbs[x0,0,1] EdgeDistanceAbs[x0,0,2]
 EdgeDistanceAbs[x0,0,3] EdgeDistanceAbs[x0,0,4] EdgeDistanceAbs[x0,0,5]
 EdgeDistanceAbs[x0,0,6] EdgeDistanceAbs[x0,0,7] EdgeDistanceAbs[x0,1,0]
 EdgeDistanceAbs[x0,1,1] EdgeDistanceAbs[x0,1,2] EdgeDistanceAbs[x0,1,3]
 EdgeDistanceAbs[x0,1,4] EdgeDistanceAbs[x0,1,5] EdgeDistanceAbs[x0,1,6]
 EdgeDistanceAbs[x0,1,7] EdgeDistanceAbs[x0,2,0] EdgeDistanceAbs[x0,2,1]
 EdgeDistanceAbs[x0,2,2] EdgeDistanceAbs[x0,2,3] EdgeDistanceAbs[x0,2,4]
 EdgeDistanceAbs[x0,2,5] EdgeDistanceAbs[x0,2,6] EdgeDistanceAbs[x0,2,7]
 EdgeDistanceAbs[x0,3,0] EdgeDistanceAbs[x0,3,1] EdgeDistanceAbs[x0,3,2]
 EdgeDistanceAbs[x0,3,3] EdgeDistanceAbs[x0,3,4] EdgeDistanceAbs[x0,3,5]
 EdgeDistanceAbs[x0,3,6] EdgeDistanceAbs[x0,3,7] EdgeDistanceAbs[x0,4,0]
 EdgeDistanceAbs[x0,4,1] EdgeDistanceAbs[x0,4,2] EdgeDistanceAbs[x0,4,3]
 EdgeDistanceAbs[x0,4,4] EdgeDistanceAbs[x0,4,5] EdgeDistanceAbs[x0,4,6]
 EdgeDistanceAbs[x0,4,7] EdgeDistanceAbs[x0,5,0] EdgeDistanceAbs[x0,5,1]
 EdgeDistanceAbs[x0,5,2] EdgeDistanceAbs[x0,5,3] EdgeDistanceAbs[x0,5,4]
 EdgeDistanceAbs[x0,5,5] EdgeDistanceAbs[x0,5,6] EdgeDistanceAbs[x0,5,7]
 EdgeDistanceAbs[x0,6,0] EdgeDistanceAbs[x0,6,1] EdgeDistanceAbs[x0,6,2]
 EdgeDistanceAbs[x0,6,3] EdgeDistanceAbs[x0,6,4] EdgeDistanceAbs[x0,6,5]
 EdgeDistanceAbs[x0,6,6] EdgeDistanceAbs[x0,6,7] EdgeDistanceAbs[x0,7,0]
 EdgeDistanceAbs[x0,7,1] EdgeDistanceAbs[x0,7,2] EdgeDistanceAbs[x0,7,3]
 EdgeDistanceAbs[x0,7,4] EdgeDistanceAbs[x0,7,5] EdgeDistanceAbs[x0,7,6]
 EdgeDistanceAbs[x0,7,7] EdgeDistanceAbs[y0,0,0] EdgeDistanceAbs[y0,0,1]
 EdgeDistanceAbs[y0,0,2] EdgeDistanceAbs[y0,0,3] EdgeDistanceAbs[y0,0,4]
 EdgeDistanceAbs[y0,0,5] EdgeDistanceAbs[y0,0,6] EdgeDistanceAbs[y0,0,7]
 EdgeDistanceAbs[y0,1,0] EdgeDistanceAbs[y0,1,1] EdgeDistanceAbs[y0,1,2]
 EdgeDistanceAbs[y0,1,3] EdgeDistanceAbs[y0,1,4] EdgeDistanceAbs[y0,1,5]
 EdgeDistanceAbs[y0,1,6] EdgeDistanceAbs[y0,1,7] EdgeDistanceAbs[y0,2,0]
 EdgeDistanceAbs[y0,2,1] EdgeDistanceAbs[y0,2,2] EdgeDistanceAbs[y0,2,3]
 EdgeDistanceAbs[y0,2,4] EdgeDistanceAbs[y0,2,5] EdgeDistanceAbs[y0,2,6]
 EdgeDistanceAbs[y0,2,7] EdgeDistanceAbs[y0,3,0] EdgeDistanceAbs[y0,3,1]
 EdgeDistanceAbs[y0,3,2] EdgeDistanceAbs[y0,3,3] EdgeDistanceAbs[y0,3,4]
 EdgeDistanceAbs[y0,3,5] EdgeDistanceAbs[y0,3,6] EdgeDistanceAbs[y0,3,7]
 EdgeDistanceAbs[y0,4,0] EdgeDistanceAbs[y0,4,1] EdgeDistanceAbs[y0,4,2]
 EdgeDistanceAbs[y0,4,3] EdgeDistanceAbs[y0,4,4] EdgeDistanceAbs[y0,4,5]
 EdgeDistanceAbs[y0,4,6] EdgeDistanceAbs[y0,4,7] EdgeDistanceAbs[y0,5,0]
 EdgeDistanceAbs[y0,5,1] EdgeDistanceAbs[y0,5,2] EdgeDistanceAbs[y0,5,3]
 EdgeDistanceAbs[y0,5,4] EdgeDistanceAbs[y0,5,5] EdgeDistanceAbs[y0,5,6]
 EdgeDistanceAbs[y0,5,7] EdgeDistanceAbs[y0,6,0] EdgeDistanceAbs[y0,6,1]
 EdgeDistanceAbs[y0,6,2] EdgeDistanceAbs[y0,6,3] EdgeDistanceAbs[y0,6,4]
 EdgeDistanceAbs[y0,6,5] EdgeDistanceAbs[y0,6,6] EdgeDistanceAbs[y0,6,7]
 EdgeDistanceAbs[y0,7,0] EdgeDistanceAbs[y0,7,1] EdgeDistanceAbs[y0,7,2]
 EdgeDistanceAbs[y0,7,3] EdgeDistanceAbs[y0,7,4] EdgeDistanceAbs[y0,7,5]
 EdgeDistanceAbs[y0,7,6] EdgeDistanceAbs[y0,7,7] EdgeDistanceAbs[x1,0,0]
 EdgeDistanceAbs[x1,0,1] EdgeDistanceAbs[x1,0,2] EdgeDistanceAbs[x1,0,3]
 EdgeDistanceAbs[x1,0,4] EdgeDistanceAbs[x1,0,5] EdgeDistanceAbs[x1,0,6]
 EdgeDistanceAbs[x1,0,7] EdgeDistanceAbs[x1,1,0] EdgeDistanceAbs[x1,1,1]
 EdgeDistanceAbs[x1,1,2] EdgeDistanceAbs[x1,1,3] EdgeDistanceAbs[x1,1,4]
 EdgeDistanceAbs[x1,1,5] EdgeDistanceAbs[x1,1,6] EdgeDistanceAbs[x1,1,7]
 EdgeDistanceAbs[x1,2,0] EdgeDistanceAbs[x1,2,1] EdgeDistanceAbs[x1,2,2]
 EdgeDistanceAbs[x1,2,3] EdgeDistanceAbs[x1,2,4] EdgeDistanceAbs[x1,2,5]
 EdgeDistanceAbs[x1,2,6] EdgeDistanceAbs[x1,2,7] EdgeDistanceAbs[x1,3,0]
 EdgeDistanceAbs[x1,3,1] EdgeDistanceAbs[x1,3,2] EdgeDistanceAbs[x1,3,3]
 EdgeDistanceAbs[x1,3,4] EdgeDistanceAbs[x1,3,5] EdgeDistanceAbs[x1,3,6]
 EdgeDistanceAbs[x1,3,7] EdgeDistanceAbs[x1,4,0] EdgeDistanceAbs[x1,4,1]
 EdgeDistanceAbs[x1,4,2] EdgeDistanceAbs[x1,4,3] EdgeDistanceAbs[x1,4,4]
 EdgeDistanceAbs[x1,4,5] EdgeDistanceAbs[x1,4,6] EdgeDistanceAbs[x1,4,7]
 EdgeDistanceAbs[x1,5,0] EdgeDistanceAbs[x1,5,1] EdgeDistanceAbs[x1,5,2]
 EdgeDistanceAbs[x1,5,3] EdgeDistanceAbs[x1,5,4] EdgeDistanceAbs[x1,5,5]
 EdgeDistanceAbs[x1,5,6] EdgeDistanceAbs[x1,5,7] EdgeDistanceAbs[x1,6,0]
 EdgeDistanceAbs[x1,6,1] EdgeDistanceAbs[x1,6,2] EdgeDistanceAbs[x1,6,3]
 EdgeDistanceAbs[x1,6,4] EdgeDistanceAbs[x1,6,5] EdgeDistanceAbs[x1,6,6]
 EdgeDistanceAbs[x1,6,7] EdgeDistanceAbs[x1,7,0] EdgeDistanceAbs[x1,7,1]
 EdgeDistanceAbs[x1,7,2] EdgeDistanceAbs[x1,7,3] EdgeDistanceAbs[x1,7,4]
 EdgeDistanceAbs[x1,7,5] EdgeDistanceAbs[x1,7,6] EdgeDistanceAbs[x1,7,7]
 EdgeDistanceAbs[y1,0,0] EdgeDistanceAbs[y1,0,1] EdgeDistanceAbs[y1,0,2]
 EdgeDistanceAbs[y1,0,3] EdgeDistanceAbs[y1,0,4] EdgeDistanceAbs[y1,0,5]
 EdgeDistanceAbs[y1,0,6] EdgeDistanceAbs[y1,0,7] EdgeDistanceAbs[y1,1,0]
 EdgeDistanceAbs[y1,1,1] EdgeDistanceAbs[y1,1,2] EdgeDistanceAbs[y1,1,3]
 EdgeDistanceAbs[y1,1,4] EdgeDistanceAbs[y1,1,5] EdgeDistanceAbs[y1,1,6]
 EdgeDistanceAbs[y1,1,7] EdgeDistanceAbs[y1,2,0] EdgeDistanceAbs[y1,2,1]
 EdgeDistanceAbs[y1,2,2] EdgeDistanceAbs[y1,2,3] EdgeDistanceAbs[y1,2,4]
 EdgeDistanceAbs[y1,2,5] EdgeDistanceAbs[y1,2,6] EdgeDistanceAbs[y1,2,7]
 EdgeDistanceAbs[y1,3,0] EdgeDistanceAbs[y1,3,1] EdgeDistanceAbs[y1,3,2]
 EdgeDistanceAbs[y1,3,3] EdgeDistanceAbs[y1,3,4] EdgeDistanceAbs[y1,3,5]
 EdgeDistanceAbs[y1,3,6] EdgeDistanceAbs[y1,3,7] EdgeDistanceAbs[y1,4,0]
 EdgeDistanceAbs[y1,4,1] EdgeDistanceAbs[y1,4,2] EdgeDistanceAbs[y1,4,3]
 EdgeDistanceAbs[y1,4,4] EdgeDistanceAbs[y1,4,5] EdgeDistanceAbs[y1,4,6]
 EdgeDistanceAbs[y1,4,7] EdgeDistanceAbs[y1,5,0] EdgeDistanceAbs[y1,5,1]
 EdgeDistanceAbs[y1,5,2] EdgeDistanceAbs[y1,5,3] EdgeDistanceAbs[y1,5,4]
 EdgeDistanceAbs[y1,5,5] EdgeDistanceAbs[y1,5,6] EdgeDistanceAbs[y1,5,7]
 EdgeDistanceAbs[y1,6,0] EdgeDistanceAbs[y1,6,1] EdgeDistanceAbs[y1,6,2]
 EdgeDistanceAbs[y1,6,3] EdgeDistanceAbs[y1,6,4] EdgeDistanceAbs[y1,6,5]
 EdgeDistanceAbs[y1,6,6] EdgeDistanceAbs[y1,6,7] EdgeDistanceAbs[y1,7,0]
 EdgeDistanceAbs[y1,7,1] EdgeDistanceAbs[y1,7,2] EdgeDistanceAbs[y1,7,3]
 EdgeDistanceAbs[y1,7,4] EdgeDistanceAbs[y1,7,5] EdgeDistanceAbs[y1,7,6]
 EdgeDistanceAbs[y1,7,7] NumberOfGroups[x0] NumberOfGroups[y0]
 NumberOfGroups[x1] NumberOfGroups[y1]
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
 LinkEdgeDistanceAbs[x0,0,0]: EdgeDistanceAbs[x0,0,0] = ABS (
   EdgeDistance[x0,0,0] )
 LinkEdgeDistanceAbs[x0,0,1]: EdgeDistanceAbs[x0,0,1] = ABS (
   EdgeDistance[x0,0,1] )
 LinkEdgeDistanceAbs[x0,0,2]: EdgeDistanceAbs[x0,0,2] = ABS (
   EdgeDistance[x0,0,2] )
 LinkEdgeDistanceAbs[x0,0,3]: EdgeDistanceAbs[x0,0,3] = ABS (
   EdgeDistance[x0,0,3] )
 LinkEdgeDistanceAbs[x0,0,4]: EdgeDistanceAbs[x0,0,4] = ABS (
   EdgeDistance[x0,0,4] )
 LinkEdgeDistanceAbs[x0,0,5]: EdgeDistanceAbs[x0,0,5] = ABS (
   EdgeDistance[x0,0,5] )
 LinkEdgeDistanceAbs[x0,0,6]: EdgeDistanceAbs[x0,0,6] = ABS (
   EdgeDistance[x0,0,6] )
 LinkEdgeDistanceAbs[x0,0,7]: EdgeDistanceAbs[x0,0,7] = ABS (
   EdgeDistance[x0,0,7] )
 LinkEdgeDistanceAbs[x0,1,0]: EdgeDistanceAbs[x0,1,0] = ABS (
   EdgeDistance[x0,1,0] )
 LinkEdgeDistanceAbs[x0,1,1]: EdgeDistanceAbs[x0,1,1] = ABS (
   EdgeDistance[x0,1,1] )
 LinkEdgeDistanceAbs[x0,1,2]: EdgeDistanceAbs[x0,1,2] = ABS (
   EdgeDistance[x0,1,2] )
 LinkEdgeDistanceAbs[x0,1,3]: EdgeDistanceAbs[x0,1,3] = ABS (
   EdgeDistance[x0,1,3] )
 LinkEdgeDistanceAbs[x0,1,4]: EdgeDistanceAbs[x0,1,4] = ABS (
   EdgeDistance[x0,1,4] )
 LinkEdgeDistanceAbs[x0,1,5]: EdgeDistanceAbs[x0,1,5] = ABS (
   EdgeDistance[x0,1,5] )
 LinkEdgeDistanceAbs[x0,1,6]: EdgeDistanceAbs[x0,1,6] = ABS (
   EdgeDistance[x0,1,6] )
 LinkEdgeDistanceAbs[x0,1,7]: EdgeDistanceAbs[x0,1,7] = ABS (
   EdgeDistance[x0,1,7] )
 LinkEdgeDistanceAbs[x0,2,0]: EdgeDistanceAbs[x0,2,0] = ABS (
   EdgeDistance[x0,2,0] )
 LinkEdgeDistanceAbs[x0,2,1]: EdgeDistanceAbs[x0,2,1] = ABS (
   EdgeDistance[x0,2,1] )
 LinkEdgeDistanceAbs[x0,2,2]: EdgeDistanceAbs[x0,2,2] = ABS (
   EdgeDistance[x0,2,2] )
 LinkEdgeDistanceAbs[x0,2,3]: EdgeDistanceAbs[x0,2,3] = ABS (
   EdgeDistance[x0,2,3] )
 LinkEdgeDistanceAbs[x0,2,4]: EdgeDistanceAbs[x0,2,4] = ABS (
   EdgeDistance[x0,2,4] )
 LinkEdgeDistanceAbs[x0,2,5]: EdgeDistanceAbs[x0,2,5] = ABS (
   EdgeDistance[x0,2,5] )
 LinkEdgeDistanceAbs[x0,2,6]: EdgeDistanceAbs[x0,2,6] = ABS (
   EdgeDistance[x0,2,6] )
 LinkEdgeDistanceAbs[x0,2,7]: EdgeDistanceAbs[x0,2,7] = ABS (
   EdgeDistance[x0,2,7] )
 LinkEdgeDistanceAbs[x0,3,0]: EdgeDistanceAbs[x0,3,0] = ABS (
   EdgeDistance[x0,3,0] )
 LinkEdgeDistanceAbs[x0,3,1]: EdgeDistanceAbs[x0,3,1] = ABS (
   EdgeDistance[x0,3,1] )
 LinkEdgeDistanceAbs[x0,3,2]: EdgeDistanceAbs[x0,3,2] = ABS (
   EdgeDistance[x0,3,2] )
 LinkEdgeDistanceAbs[x0,3,3]: EdgeDistanceAbs[x0,3,3] = ABS (
   EdgeDistance[x0,3,3] )
 LinkEdgeDistanceAbs[x0,3,4]: EdgeDistanceAbs[x0,3,4] = ABS (
   EdgeDistance[x0,3,4] )
 LinkEdgeDistanceAbs[x0,3,5]: EdgeDistanceAbs[x0,3,5] = ABS (
   EdgeDistance[x0,3,5] )
 LinkEdgeDistanceAbs[x0,3,6]: EdgeDistanceAbs[x0,3,6] = ABS (
   EdgeDistance[x0,3,6] )
 LinkEdgeDistanceAbs[x0,3,7]: EdgeDistanceAbs[x0,3,7] = ABS (
   EdgeDistance[x0,3,7] )
 LinkEdgeDistanceAbs[x0,4,0]: EdgeDistanceAbs[x0,4,0] = ABS (
   EdgeDistance[x0,4,0] )
 LinkEdgeDistanceAbs[x0,4,1]: EdgeDistanceAbs[x0,4,1] = ABS (
   EdgeDistance[x0,4,1] )
 LinkEdgeDistanceAbs[x0,4,2]: EdgeDistanceAbs[x0,4,2] = ABS (
   EdgeDistance[x0,4,2] )
 LinkEdgeDistanceAbs[x0,4,3]: EdgeDistanceAbs[x0,4,3] = ABS (
   EdgeDistance[x0,4,3] )
 LinkEdgeDistanceAbs[x0,4,4]: EdgeDistanceAbs[x0,4,4] = ABS (
   EdgeDistance[x0,4,4] )
 LinkEdgeDistanceAbs[x0,4,5]: EdgeDistanceAbs[x0,4,5] = ABS (
   EdgeDistance[x0,4,5] )
 LinkEdgeDistanceAbs[x0,4,6]: EdgeDistanceAbs[x0,4,6] = ABS (
   EdgeDistance[x0,4,6] )
 LinkEdgeDistanceAbs[x0,4,7]: EdgeDistanceAbs[x0,4,7] = ABS (
   EdgeDistance[x0,4,7] )
 LinkEdgeDistanceAbs[x0,5,0]: EdgeDistanceAbs[x0,5,0] = ABS (
   EdgeDistance[x0,5,0] )
 LinkEdgeDistanceAbs[x0,5,1]: EdgeDistanceAbs[x0,5,1] = ABS (
   EdgeDistance[x0,5,1] )
 LinkEdgeDistanceAbs[x0,5,2]: EdgeDistanceAbs[x0,5,2] = ABS (
   EdgeDistance[x0,5,2] )
 LinkEdgeDistanceAbs[x0,5,3]: EdgeDistanceAbs[x0,5,3] = ABS (
   EdgeDistance[x0,5,3] )
 LinkEdgeDistanceAbs[x0,5,4]: EdgeDistanceAbs[x0,5,4] = ABS (
   EdgeDistance[x0,5,4] )
 LinkEdgeDistanceAbs[x0,5,5]: EdgeDistanceAbs[x0,5,5] = ABS (
   EdgeDistance[x0,5,5] )
 LinkEdgeDistanceAbs[x0,5,6]: EdgeDistanceAbs[x0,5,6] = ABS (
   EdgeDistance[x0,5,6] )
 LinkEdgeDistanceAbs[x0,5,7]: EdgeDistanceAbs[x0,5,7] = ABS (
   EdgeDistance[x0,5,7] )
 LinkEdgeDistanceAbs[x0,6,0]: EdgeDistanceAbs[x0,6,0] = ABS (
   EdgeDistance[x0,6,0] )
 LinkEdgeDistanceAbs[x0,6,1]: EdgeDistanceAbs[x0,6,1] = ABS (
   EdgeDistance[x0,6,1] )
 LinkEdgeDistanceAbs[x0,6,2]: EdgeDistanceAbs[x0,6,2] = ABS (
   EdgeDistance[x0,6,2] )
 LinkEdgeDistanceAbs[x0,6,3]: EdgeDistanceAbs[x0,6,3] = ABS (
   EdgeDistance[x0,6,3] )
 LinkEdgeDistanceAbs[x0,6,4]: EdgeDistanceAbs[x0,6,4] = ABS (
   EdgeDistance[x0,6,4] )
 LinkEdgeDistanceAbs[x0,6,5]: EdgeDistanceAbs[x0,6,5] = ABS (
   EdgeDistance[x0,6,5] )
 LinkEdgeDistanceAbs[x0,6,6]: EdgeDistanceAbs[x0,6,6] = ABS (
   EdgeDistance[x0,6,6] )
 LinkEdgeDistanceAbs[x0,6,7]: EdgeDistanceAbs[x0,6,7] = ABS (
   EdgeDistance[x0,6,7] )
 LinkEdgeDistanceAbs[x0,7,0]: EdgeDistanceAbs[x0,7,0] = ABS (
   EdgeDistance[x0,7,0] )
 LinkEdgeDistanceAbs[x0,7,1]: EdgeDistanceAbs[x0,7,1] = ABS (
   EdgeDistance[x0,7,1] )
 LinkEdgeDistanceAbs[x0,7,2]: EdgeDistanceAbs[x0,7,2] = ABS (
   EdgeDistance[x0,7,2] )
 LinkEdgeDistanceAbs[x0,7,3]: EdgeDistanceAbs[x0,7,3] = ABS (
   EdgeDistance[x0,7,3] )
 LinkEdgeDistanceAbs[x0,7,4]: EdgeDistanceAbs[x0,7,4] = ABS (
   EdgeDistance[x0,7,4] )
 LinkEdgeDistanceAbs[x0,7,5]: EdgeDistanceAbs[x0,7,5] = ABS (
   EdgeDistance[x0,7,5] )
 LinkEdgeDistanceAbs[x0,7,6]: EdgeDistanceAbs[x0,7,6] = ABS (
   EdgeDistance[x0,7,6] )
 LinkEdgeDistanceAbs[x0,7,7]: EdgeDistanceAbs[x0,7,7] = ABS (
   EdgeDistance[x0,7,7] )
 LinkEdgeDistanceAbs[y0,0,0]: EdgeDistanceAbs[y0,0,0] = ABS (
   EdgeDistance[y0,0,0] )
 LinkEdgeDistanceAbs[y0,0,1]: EdgeDistanceAbs[y0,0,1] = ABS (
   EdgeDistance[y0,0,1] )
 LinkEdgeDistanceAbs[y0,0,2]: EdgeDistanceAbs[y0,0,2] = ABS (
   EdgeDistance[y0,0,2] )
 LinkEdgeDistanceAbs[y0,0,3]: EdgeDistanceAbs[y0,0,3] = ABS (
   EdgeDistance[y0,0,3] )
 LinkEdgeDistanceAbs[y0,0,4]: EdgeDistanceAbs[y0,0,4] = ABS (
   EdgeDistance[y0,0,4] )
 LinkEdgeDistanceAbs[y0,0,5]: EdgeDistanceAbs[y0,0,5] = ABS (
   EdgeDistance[y0,0,5] )
 LinkEdgeDistanceAbs[y0,0,6]: EdgeDistanceAbs[y0,0,6] = ABS (
   EdgeDistance[y0,0,6] )
 LinkEdgeDistanceAbs[y0,0,7]: EdgeDistanceAbs[y0,0,7] = ABS (
   EdgeDistance[y0,0,7] )
 LinkEdgeDistanceAbs[y0,1,0]: EdgeDistanceAbs[y0,1,0] = ABS (
   EdgeDistance[y0,1,0] )
 LinkEdgeDistanceAbs[y0,1,1]: EdgeDistanceAbs[y0,1,1] = ABS (
   EdgeDistance[y0,1,1] )
 LinkEdgeDistanceAbs[y0,1,2]: EdgeDistanceAbs[y0,1,2] = ABS (
   EdgeDistance[y0,1,2] )
 LinkEdgeDistanceAbs[y0,1,3]: EdgeDistanceAbs[y0,1,3] = ABS (
   EdgeDistance[y0,1,3] )
 LinkEdgeDistanceAbs[y0,1,4]: EdgeDistanceAbs[y0,1,4] = ABS (
   EdgeDistance[y0,1,4] )
 LinkEdgeDistanceAbs[y0,1,5]: EdgeDistanceAbs[y0,1,5] = ABS (
   EdgeDistance[y0,1,5] )
 LinkEdgeDistanceAbs[y0,1,6]: EdgeDistanceAbs[y0,1,6] = ABS (
   EdgeDistance[y0,1,6] )
 LinkEdgeDistanceAbs[y0,1,7]: EdgeDistanceAbs[y0,1,7] = ABS (
   EdgeDistance[y0,1,7] )
 LinkEdgeDistanceAbs[y0,2,0]: EdgeDistanceAbs[y0,2,0] = ABS (
   EdgeDistance[y0,2,0] )
 LinkEdgeDistanceAbs[y0,2,1]: EdgeDistanceAbs[y0,2,1] = ABS (
   EdgeDistance[y0,2,1] )
 LinkEdgeDistanceAbs[y0,2,2]: EdgeDistanceAbs[y0,2,2] = ABS (
   EdgeDistance[y0,2,2] )
 LinkEdgeDistanceAbs[y0,2,3]: EdgeDistanceAbs[y0,2,3] = ABS (
   EdgeDistance[y0,2,3] )
 LinkEdgeDistanceAbs[y0,2,4]: EdgeDistanceAbs[y0,2,4] = ABS (
   EdgeDistance[y0,2,4] )
 LinkEdgeDistanceAbs[y0,2,5]: EdgeDistanceAbs[y0,2,5] = ABS (
   EdgeDistance[y0,2,5] )
 LinkEdgeDistanceAbs[y0,2,6]: EdgeDistanceAbs[y0,2,6] = ABS (
   EdgeDistance[y0,2,6] )
 LinkEdgeDistanceAbs[y0,2,7]: EdgeDistanceAbs[y0,2,7] = ABS (
   EdgeDistance[y0,2,7] )
 LinkEdgeDistanceAbs[y0,3,0]: EdgeDistanceAbs[y0,3,0] = ABS (
   EdgeDistance[y0,3,0] )
 LinkEdgeDistanceAbs[y0,3,1]: EdgeDistanceAbs[y0,3,1] = ABS (
   EdgeDistance[y0,3,1] )
 LinkEdgeDistanceAbs[y0,3,2]: EdgeDistanceAbs[y0,3,2] = ABS (
   EdgeDistance[y0,3,2] )
 LinkEdgeDistanceAbs[y0,3,3]: EdgeDistanceAbs[y0,3,3] = ABS (
   EdgeDistance[y0,3,3] )
 LinkEdgeDistanceAbs[y0,3,4]: EdgeDistanceAbs[y0,3,4] = ABS (
   EdgeDistance[y0,3,4] )
 LinkEdgeDistanceAbs[y0,3,5]: EdgeDistanceAbs[y0,3,5] = ABS (
   EdgeDistance[y0,3,5] )
 LinkEdgeDistanceAbs[y0,3,6]: EdgeDistanceAbs[y0,3,6] = ABS (
   EdgeDistance[y0,3,6] )
 LinkEdgeDistanceAbs[y0,3,7]: EdgeDistanceAbs[y0,3,7] = ABS (
   EdgeDistance[y0,3,7] )
 LinkEdgeDistanceAbs[y0,4,0]: EdgeDistanceAbs[y0,4,0] = ABS (
   EdgeDistance[y0,4,0] )
 LinkEdgeDistanceAbs[y0,4,1]: EdgeDistanceAbs[y0,4,1] = ABS (
   EdgeDistance[y0,4,1] )
 LinkEdgeDistanceAbs[y0,4,2]: EdgeDistanceAbs[y0,4,2] = ABS (
   EdgeDistance[y0,4,2] )
 LinkEdgeDistanceAbs[y0,4,3]: EdgeDistanceAbs[y0,4,3] = ABS (
   EdgeDistance[y0,4,3] )
 LinkEdgeDistanceAbs[y0,4,4]: EdgeDistanceAbs[y0,4,4] = ABS (
   EdgeDistance[y0,4,4] )
 LinkEdgeDistanceAbs[y0,4,5]: EdgeDistanceAbs[y0,4,5] = ABS (
   EdgeDistance[y0,4,5] )
 LinkEdgeDistanceAbs[y0,4,6]: EdgeDistanceAbs[y0,4,6] = ABS (
   EdgeDistance[y0,4,6] )
 LinkEdgeDistanceAbs[y0,4,7]: EdgeDistanceAbs[y0,4,7] = ABS (
   EdgeDistance[y0,4,7] )
 LinkEdgeDistanceAbs[y0,5,0]: EdgeDistanceAbs[y0,5,0] = ABS (
   EdgeDistance[y0,5,0] )
 LinkEdgeDistanceAbs[y0,5,1]: EdgeDistanceAbs[y0,5,1] = ABS (
   EdgeDistance[y0,5,1] )
 LinkEdgeDistanceAbs[y0,5,2]: EdgeDistanceAbs[y0,5,2] = ABS (
   EdgeDistance[y0,5,2] )
 LinkEdgeDistanceAbs[y0,5,3]: EdgeDistanceAbs[y0,5,3] = ABS (
   EdgeDistance[y0,5,3] )
 LinkEdgeDistanceAbs[y0,5,4]: EdgeDistanceAbs[y0,5,4] = ABS (
   EdgeDistance[y0,5,4] )
 LinkEdgeDistanceAbs[y0,5,5]: EdgeDistanceAbs[y0,5,5] = ABS (
   EdgeDistance[y0,5,5] )
 LinkEdgeDistanceAbs[y0,5,6]: EdgeDistanceAbs[y0,5,6] = ABS (
   EdgeDistance[y0,5,6] )
 LinkEdgeDistanceAbs[y0,5,7]: EdgeDistanceAbs[y0,5,7] = ABS (
   EdgeDistance[y0,5,7] )
 LinkEdgeDistanceAbs[y0,6,0]: EdgeDistanceAbs[y0,6,0] = ABS (
   EdgeDistance[y0,6,0] )
 LinkEdgeDistanceAbs[y0,6,1]: EdgeDistanceAbs[y0,6,1] = ABS (
   EdgeDistance[y0,6,1] )
 LinkEdgeDistanceAbs[y0,6,2]: EdgeDistanceAbs[y0,6,2] = ABS (
   EdgeDistance[y0,6,2] )
 LinkEdgeDistanceAbs[y0,6,3]: EdgeDistanceAbs[y0,6,3] = ABS (
   EdgeDistance[y0,6,3] )
 LinkEdgeDistanceAbs[y0,6,4]: EdgeDistanceAbs[y0,6,4] = ABS (
   EdgeDistance[y0,6,4] )
 LinkEdgeDistanceAbs[y0,6,5]: EdgeDistanceAbs[y0,6,5] = ABS (
   EdgeDistance[y0,6,5] )
 LinkEdgeDistanceAbs[y0,6,6]: EdgeDistanceAbs[y0,6,6] = ABS (
   EdgeDistance[y0,6,6] )
 LinkEdgeDistanceAbs[y0,6,7]: EdgeDistanceAbs[y0,6,7] = ABS (
   EdgeDistance[y0,6,7] )
 LinkEdgeDistanceAbs[y0,7,0]: EdgeDistanceAbs[y0,7,0] = ABS (
   EdgeDistance[y0,7,0] )
 LinkEdgeDistanceAbs[y0,7,1]: EdgeDistanceAbs[y0,7,1] = ABS (
   EdgeDistance[y0,7,1] )
 LinkEdgeDistanceAbs[y0,7,2]: EdgeDistanceAbs[y0,7,2] = ABS (
   EdgeDistance[y0,7,2] )
 LinkEdgeDistanceAbs[y0,7,3]: EdgeDistanceAbs[y0,7,3] = ABS (
   EdgeDistance[y0,7,3] )
 LinkEdgeDistanceAbs[y0,7,4]: EdgeDistanceAbs[y0,7,4] = ABS (
   EdgeDistance[y0,7,4] )
 LinkEdgeDistanceAbs[y0,7,5]: EdgeDistanceAbs[y0,7,5] = ABS (
   EdgeDistance[y0,7,5] )
 LinkEdgeDistanceAbs[y0,7,6]: EdgeDistanceAbs[y0,7,6] = ABS (
   EdgeDistance[y0,7,6] )
 LinkEdgeDistanceAbs[y0,7,7]: EdgeDistanceAbs[y0,7,7] = ABS (
   EdgeDistance[y0,7,7] )
 LinkEdgeDistanceAbs[x1,0,0]: EdgeDistanceAbs[x1,0,0] = ABS (
   EdgeDistance[x1,0,0] )
 LinkEdgeDistanceAbs[x1,0,1]: EdgeDistanceAbs[x1,0,1] = ABS (
   EdgeDistance[x1,0,1] )
 LinkEdgeDistanceAbs[x1,0,2]: EdgeDistanceAbs[x1,0,2] = ABS (
   EdgeDistance[x1,0,2] )
 LinkEdgeDistanceAbs[x1,0,3]: EdgeDistanceAbs[x1,0,3] = ABS (
   EdgeDistance[x1,0,3] )
 LinkEdgeDistanceAbs[x1,0,4]: EdgeDistanceAbs[x1,0,4] = ABS (
   EdgeDistance[x1,0,4] )
 LinkEdgeDistanceAbs[x1,0,5]: EdgeDistanceAbs[x1,0,5] = ABS (
   EdgeDistance[x1,0,5] )
 LinkEdgeDistanceAbs[x1,0,6]: EdgeDistanceAbs[x1,0,6] = ABS (
   EdgeDistance[x1,0,6] )
 LinkEdgeDistanceAbs[x1,0,7]: EdgeDistanceAbs[x1,0,7] = ABS (
   EdgeDistance[x1,0,7] )
 LinkEdgeDistanceAbs[x1,1,0]: EdgeDistanceAbs[x1,1,0] = ABS (
   EdgeDistance[x1,1,0] )
 LinkEdgeDistanceAbs[x1,1,1]: EdgeDistanceAbs[x1,1,1] = ABS (
   EdgeDistance[x1,1,1] )
 LinkEdgeDistanceAbs[x1,1,2]: EdgeDistanceAbs[x1,1,2] = ABS (
   EdgeDistance[x1,1,2] )
 LinkEdgeDistanceAbs[x1,1,3]: EdgeDistanceAbs[x1,1,3] = ABS (
   EdgeDistance[x1,1,3] )
 LinkEdgeDistanceAbs[x1,1,4]: EdgeDistanceAbs[x1,1,4] = ABS (
   EdgeDistance[x1,1,4] )
 LinkEdgeDistanceAbs[x1,1,5]: EdgeDistanceAbs[x1,1,5] = ABS (
   EdgeDistance[x1,1,5] )
 LinkEdgeDistanceAbs[x1,1,6]: EdgeDistanceAbs[x1,1,6] = ABS (
   EdgeDistance[x1,1,6] )
 LinkEdgeDistanceAbs[x1,1,7]: EdgeDistanceAbs[x1,1,7] = ABS (
   EdgeDistance[x1,1,7] )
 LinkEdgeDistanceAbs[x1,2,0]: EdgeDistanceAbs[x1,2,0] = ABS (
   EdgeDistance[x1,2,0] )
 LinkEdgeDistanceAbs[x1,2,1]: EdgeDistanceAbs[x1,2,1] = ABS (
   EdgeDistance[x1,2,1] )
 LinkEdgeDistanceAbs[x1,2,2]: EdgeDistanceAbs[x1,2,2] = ABS (
   EdgeDistance[x1,2,2] )
 LinkEdgeDistanceAbs[x1,2,3]: EdgeDistanceAbs[x1,2,3] = ABS (
   EdgeDistance[x1,2,3] )
 LinkEdgeDistanceAbs[x1,2,4]: EdgeDistanceAbs[x1,2,4] = ABS (
   EdgeDistance[x1,2,4] )
 LinkEdgeDistanceAbs[x1,2,5]: EdgeDistanceAbs[x1,2,5] = ABS (
   EdgeDistance[x1,2,5] )
 LinkEdgeDistanceAbs[x1,2,6]: EdgeDistanceAbs[x1,2,6] = ABS (
   EdgeDistance[x1,2,6] )
 LinkEdgeDistanceAbs[x1,2,7]: EdgeDistanceAbs[x1,2,7] = ABS (
   EdgeDistance[x1,2,7] )
 LinkEdgeDistanceAbs[x1,3,0]: EdgeDistanceAbs[x1,3,0] = ABS (
   EdgeDistance[x1,3,0] )
 LinkEdgeDistanceAbs[x1,3,1]: EdgeDistanceAbs[x1,3,1] = ABS (
   EdgeDistance[x1,3,1] )
 LinkEdgeDistanceAbs[x1,3,2]: EdgeDistanceAbs[x1,3,2] = ABS (
   EdgeDistance[x1,3,2] )
 LinkEdgeDistanceAbs[x1,3,3]: EdgeDistanceAbs[x1,3,3] = ABS (
   EdgeDistance[x1,3,3] )
 LinkEdgeDistanceAbs[x1,3,4]: EdgeDistanceAbs[x1,3,4] = ABS (
   EdgeDistance[x1,3,4] )
 LinkEdgeDistanceAbs[x1,3,5]: EdgeDistanceAbs[x1,3,5] = ABS (
   EdgeDistance[x1,3,5] )
 LinkEdgeDistanceAbs[x1,3,6]: EdgeDistanceAbs[x1,3,6] = ABS (
   EdgeDistance[x1,3,6] )
 LinkEdgeDistanceAbs[x1,3,7]: EdgeDistanceAbs[x1,3,7] = ABS (
   EdgeDistance[x1,3,7] )
 LinkEdgeDistanceAbs[x1,4,0]: EdgeDistanceAbs[x1,4,0] = ABS (
   EdgeDistance[x1,4,0] )
 LinkEdgeDistanceAbs[x1,4,1]: EdgeDistanceAbs[x1,4,1] = ABS (
   EdgeDistance[x1,4,1] )
 LinkEdgeDistanceAbs[x1,4,2]: EdgeDistanceAbs[x1,4,2] = ABS (
   EdgeDistance[x1,4,2] )
 LinkEdgeDistanceAbs[x1,4,3]: EdgeDistanceAbs[x1,4,3] = ABS (
   EdgeDistance[x1,4,3] )
 LinkEdgeDistanceAbs[x1,4,4]: EdgeDistanceAbs[x1,4,4] = ABS (
   EdgeDistance[x1,4,4] )
 LinkEdgeDistanceAbs[x1,4,5]: EdgeDistanceAbs[x1,4,5] = ABS (
   EdgeDistance[x1,4,5] )
 LinkEdgeDistanceAbs[x1,4,6]: EdgeDistanceAbs[x1,4,6] = ABS (
   EdgeDistance[x1,4,6] )
 LinkEdgeDistanceAbs[x1,4,7]: EdgeDistanceAbs[x1,4,7] = ABS (
   EdgeDistance[x1,4,7] )
 LinkEdgeDistanceAbs[x1,5,0]: EdgeDistanceAbs[x1,5,0] = ABS (
   EdgeDistance[x1,5,0] )
 LinkEdgeDistanceAbs[x1,5,1]: EdgeDistanceAbs[x1,5,1] = ABS (
   EdgeDistance[x1,5,1] )
 LinkEdgeDistanceAbs[x1,5,2]: EdgeDistanceAbs[x1,5,2] = ABS (
   EdgeDistance[x1,5,2] )
 LinkEdgeDistanceAbs[x1,5,3]: EdgeDistanceAbs[x1,5,3] = ABS (
   EdgeDistance[x1,5,3] )
 LinkEdgeDistanceAbs[x1,5,4]: EdgeDistanceAbs[x1,5,4] = ABS (
   EdgeDistance[x1,5,4] )
 LinkEdgeDistanceAbs[x1,5,5]: EdgeDistanceAbs[x1,5,5] = ABS (
   EdgeDistance[x1,5,5] )
 LinkEdgeDistanceAbs[x1,5,6]: EdgeDistanceAbs[x1,5,6] = ABS (
   EdgeDistance[x1,5,6] )
 LinkEdgeDistanceAbs[x1,5,7]: EdgeDistanceAbs[x1,5,7] = ABS (
   EdgeDistance[x1,5,7] )
 LinkEdgeDistanceAbs[x1,6,0]: EdgeDistanceAbs[x1,6,0] = ABS (
   EdgeDistance[x1,6,0] )
 LinkEdgeDistanceAbs[x1,6,1]: EdgeDistanceAbs[x1,6,1] = ABS (
   EdgeDistance[x1,6,1] )
 LinkEdgeDistanceAbs[x1,6,2]: EdgeDistanceAbs[x1,6,2] = ABS (
   EdgeDistance[x1,6,2] )
 LinkEdgeDistanceAbs[x1,6,3]: EdgeDistanceAbs[x1,6,3] = ABS (
   EdgeDistance[x1,6,3] )
 LinkEdgeDistanceAbs[x1,6,4]: EdgeDistanceAbs[x1,6,4] = ABS (
   EdgeDistance[x1,6,4] )
 LinkEdgeDistanceAbs[x1,6,5]: EdgeDistanceAbs[x1,6,5] = ABS (
   EdgeDistance[x1,6,5] )
 LinkEdgeDistanceAbs[x1,6,6]: EdgeDistanceAbs[x1,6,6] = ABS (
   EdgeDistance[x1,6,6] )
 LinkEdgeDistanceAbs[x1,6,7]: EdgeDistanceAbs[x1,6,7] = ABS (
   EdgeDistance[x1,6,7] )
 LinkEdgeDistanceAbs[x1,7,0]: EdgeDistanceAbs[x1,7,0] = ABS (
   EdgeDistance[x1,7,0] )
 LinkEdgeDistanceAbs[x1,7,1]: EdgeDistanceAbs[x1,7,1] = ABS (
   EdgeDistance[x1,7,1] )
 LinkEdgeDistanceAbs[x1,7,2]: EdgeDistanceAbs[x1,7,2] = ABS (
   EdgeDistance[x1,7,2] )
 LinkEdgeDistanceAbs[x1,7,3]: EdgeDistanceAbs[x1,7,3] = ABS (
   EdgeDistance[x1,7,3] )
 LinkEdgeDistanceAbs[x1,7,4]: EdgeDistanceAbs[x1,7,4] = ABS (
   EdgeDistance[x1,7,4] )
 LinkEdgeDistanceAbs[x1,7,5]: EdgeDistanceAbs[x1,7,5] = ABS (
   EdgeDistance[x1,7,5] )
 LinkEdgeDistanceAbs[x1,7,6]: EdgeDistanceAbs[x1,7,6] = ABS (
   EdgeDistance[x1,7,6] )
 LinkEdgeDistanceAbs[x1,7,7]: EdgeDistanceAbs[x1,7,7] = ABS (
   EdgeDistance[x1,7,7] )
 LinkEdgeDistanceAbs[y1,0,0]: EdgeDistanceAbs[y1,0,0] = ABS (
   EdgeDistance[y1,0,0] )
 LinkEdgeDistanceAbs[y1,0,1]: EdgeDistanceAbs[y1,0,1] = ABS (
   EdgeDistance[y1,0,1] )
 LinkEdgeDistanceAbs[y1,0,2]: EdgeDistanceAbs[y1,0,2] = ABS (
   EdgeDistance[y1,0,2] )
 LinkEdgeDistanceAbs[y1,0,3]: EdgeDistanceAbs[y1,0,3] = ABS (
   EdgeDistance[y1,0,3] )
 LinkEdgeDistanceAbs[y1,0,4]: EdgeDistanceAbs[y1,0,4] = ABS (
   EdgeDistance[y1,0,4] )
 LinkEdgeDistanceAbs[y1,0,5]: EdgeDistanceAbs[y1,0,5] = ABS (
   EdgeDistance[y1,0,5] )
 LinkEdgeDistanceAbs[y1,0,6]: EdgeDistanceAbs[y1,0,6] = ABS (
   EdgeDistance[y1,0,6] )
 LinkEdgeDistanceAbs[y1,0,7]: EdgeDistanceAbs[y1,0,7] = ABS (
   EdgeDistance[y1,0,7] )
 LinkEdgeDistanceAbs[y1,1,0]: EdgeDistanceAbs[y1,1,0] = ABS (
   EdgeDistance[y1,1,0] )
 LinkEdgeDistanceAbs[y1,1,1]: EdgeDistanceAbs[y1,1,1] = ABS (
   EdgeDistance[y1,1,1] )
 LinkEdgeDistanceAbs[y1,1,2]: EdgeDistanceAbs[y1,1,2] = ABS (
   EdgeDistance[y1,1,2] )
 LinkEdgeDistanceAbs[y1,1,3]: EdgeDistanceAbs[y1,1,3] = ABS (
   EdgeDistance[y1,1,3] )
 LinkEdgeDistanceAbs[y1,1,4]: EdgeDistanceAbs[y1,1,4] = ABS (
   EdgeDistance[y1,1,4] )
 LinkEdgeDistanceAbs[y1,1,5]: EdgeDistanceAbs[y1,1,5] = ABS (
   EdgeDistance[y1,1,5] )
 LinkEdgeDistanceAbs[y1,1,6]: EdgeDistanceAbs[y1,1,6] = ABS (
   EdgeDistance[y1,1,6] )
 LinkEdgeDistanceAbs[y1,1,7]: EdgeDistanceAbs[y1,1,7] = ABS (
   EdgeDistance[y1,1,7] )
 LinkEdgeDistanceAbs[y1,2,0]: EdgeDistanceAbs[y1,2,0] = ABS (
   EdgeDistance[y1,2,0] )
 LinkEdgeDistanceAbs[y1,2,1]: EdgeDistanceAbs[y1,2,1] = ABS (
   EdgeDistance[y1,2,1] )
 LinkEdgeDistanceAbs[y1,2,2]: EdgeDistanceAbs[y1,2,2] = ABS (
   EdgeDistance[y1,2,2] )
 LinkEdgeDistanceAbs[y1,2,3]: EdgeDistanceAbs[y1,2,3] = ABS (
   EdgeDistance[y1,2,3] )
 LinkEdgeDistanceAbs[y1,2,4]: EdgeDistanceAbs[y1,2,4] = ABS (
   EdgeDistance[y1,2,4] )
 LinkEdgeDistanceAbs[y1,2,5]: EdgeDistanceAbs[y1,2,5] = ABS (
   EdgeDistance[y1,2,5] )
 LinkEdgeDistanceAbs[y1,2,6]: EdgeDistanceAbs[y1,2,6] = ABS (
   EdgeDistance[y1,2,6] )
 LinkEdgeDistanceAbs[y1,2,7]: EdgeDistanceAbs[y1,2,7] = ABS (
   EdgeDistance[y1,2,7] )
 LinkEdgeDistanceAbs[y1,3,0]: EdgeDistanceAbs[y1,3,0] = ABS (
   EdgeDistance[y1,3,0] )
 LinkEdgeDistanceAbs[y1,3,1]: EdgeDistanceAbs[y1,3,1] = ABS (
   EdgeDistance[y1,3,1] )
 LinkEdgeDistanceAbs[y1,3,2]: EdgeDistanceAbs[y1,3,2] = ABS (
   EdgeDistance[y1,3,2] )
 LinkEdgeDistanceAbs[y1,3,3]: EdgeDistanceAbs[y1,3,3] = ABS (
   EdgeDistance[y1,3,3] )
 LinkEdgeDistanceAbs[y1,3,4]: EdgeDistanceAbs[y1,3,4] = ABS (
   EdgeDistance[y1,3,4] )
 LinkEdgeDistanceAbs[y1,3,5]: EdgeDistanceAbs[y1,3,5] = ABS (
   EdgeDistance[y1,3,5] )
 LinkEdgeDistanceAbs[y1,3,6]: EdgeDistanceAbs[y1,3,6] = ABS (
   EdgeDistance[y1,3,6] )
 LinkEdgeDistanceAbs[y1,3,7]: EdgeDistanceAbs[y1,3,7] = ABS (
   EdgeDistance[y1,3,7] )
 LinkEdgeDistanceAbs[y1,4,0]: EdgeDistanceAbs[y1,4,0] = ABS (
   EdgeDistance[y1,4,0] )
 LinkEdgeDistanceAbs[y1,4,1]: EdgeDistanceAbs[y1,4,1] = ABS (
   EdgeDistance[y1,4,1] )
 LinkEdgeDistanceAbs[y1,4,2]: EdgeDistanceAbs[y1,4,2] = ABS (
   EdgeDistance[y1,4,2] )
 LinkEdgeDistanceAbs[y1,4,3]: EdgeDistanceAbs[y1,4,3] = ABS (
   EdgeDistance[y1,4,3] )
 LinkEdgeDistanceAbs[y1,4,4]: EdgeDistanceAbs[y1,4,4] = ABS (
   EdgeDistance[y1,4,4] )
 LinkEdgeDistanceAbs[y1,4,5]: EdgeDistanceAbs[y1,4,5] = ABS (
   EdgeDistance[y1,4,5] )
 LinkEdgeDistanceAbs[y1,4,6]: EdgeDistanceAbs[y1,4,6] = ABS (
   EdgeDistance[y1,4,6] )
 LinkEdgeDistanceAbs[y1,4,7]: EdgeDistanceAbs[y1,4,7] = ABS (
   EdgeDistance[y1,4,7] )
 LinkEdgeDistanceAbs[y1,5,0]: EdgeDistanceAbs[y1,5,0] = ABS (
   EdgeDistance[y1,5,0] )
 LinkEdgeDistanceAbs[y1,5,1]: EdgeDistanceAbs[y1,5,1] = ABS (
   EdgeDistance[y1,5,1] )
 LinkEdgeDistanceAbs[y1,5,2]: EdgeDistanceAbs[y1,5,2] = ABS (
   EdgeDistance[y1,5,2] )
 LinkEdgeDistanceAbs[y1,5,3]: EdgeDistanceAbs[y1,5,3] = ABS (
   EdgeDistance[y1,5,3] )
 LinkEdgeDistanceAbs[y1,5,4]: EdgeDistanceAbs[y1,5,4] = ABS (
   EdgeDistance[y1,5,4] )
 LinkEdgeDistanceAbs[y1,5,5]: EdgeDistanceAbs[y1,5,5] = ABS (
   EdgeDistance[y1,5,5] )
 LinkEdgeDistanceAbs[y1,5,6]: EdgeDistanceAbs[y1,5,6] = ABS (
   EdgeDistance[y1,5,6] )
 LinkEdgeDistanceAbs[y1,5,7]: EdgeDistanceAbs[y1,5,7] = ABS (
   EdgeDistance[y1,5,7] )
 LinkEdgeDistanceAbs[y1,6,0]: EdgeDistanceAbs[y1,6,0] = ABS (
   EdgeDistance[y1,6,0] )
 LinkEdgeDistanceAbs[y1,6,1]: EdgeDistanceAbs[y1,6,1] = ABS (
   EdgeDistance[y1,6,1] )
 LinkEdgeDistanceAbs[y1,6,2]: EdgeDistanceAbs[y1,6,2] = ABS (
   EdgeDistance[y1,6,2] )
 LinkEdgeDistanceAbs[y1,6,3]: EdgeDistanceAbs[y1,6,3] = ABS (
   EdgeDistance[y1,6,3] )
 LinkEdgeDistanceAbs[y1,6,4]: EdgeDistanceAbs[y1,6,4] = ABS (
   EdgeDistance[y1,6,4] )
 LinkEdgeDistanceAbs[y1,6,5]: EdgeDistanceAbs[y1,6,5] = ABS (
   EdgeDistance[y1,6,5] )
 LinkEdgeDistanceAbs[y1,6,6]: EdgeDistanceAbs[y1,6,6] = ABS (
   EdgeDistance[y1,6,6] )
 LinkEdgeDistanceAbs[y1,6,7]: EdgeDistanceAbs[y1,6,7] = ABS (
   EdgeDistance[y1,6,7] )
 LinkEdgeDistanceAbs[y1,7,0]: EdgeDistanceAbs[y1,7,0] = ABS (
   EdgeDistance[y1,7,0] )
 LinkEdgeDistanceAbs[y1,7,1]: EdgeDistanceAbs[y1,7,1] = ABS (
   EdgeDistance[y1,7,1] )
 LinkEdgeDistanceAbs[y1,7,2]: EdgeDistanceAbs[y1,7,2] = ABS (
   EdgeDistance[y1,7,2] )
 LinkEdgeDistanceAbs[y1,7,3]: EdgeDistanceAbs[y1,7,3] = ABS (
   EdgeDistance[y1,7,3] )
 LinkEdgeDistanceAbs[y1,7,4]: EdgeDistanceAbs[y1,7,4] = ABS (
   EdgeDistance[y1,7,4] )
 LinkEdgeDistanceAbs[y1,7,5]: EdgeDistanceAbs[y1,7,5] = ABS (
   EdgeDistance[y1,7,5] )
 LinkEdgeDistanceAbs[y1,7,6]: EdgeDistanceAbs[y1,7,6] = ABS (
   EdgeDistance[y1,7,6] )
 LinkEdgeDistanceAbs[y1,7,7]: EdgeDistanceAbs[y1,7,7] = ABS (
   EdgeDistance[y1,7,7] )
 LinkEdgesAlign[x0,0,0]: EdgesAlign[x0,0,0] = MIN ( EdgeDistanceAbs[x0,0,0]
   , 1 )
 LinkEdgesAlign[x0,0,1]: EdgesAlign[x0,0,1] = MIN ( EdgeDistanceAbs[x0,0,1]
   , 1 )
 LinkEdgesAlign[x0,0,2]: EdgesAlign[x0,0,2] = MIN ( EdgeDistanceAbs[x0,0,2]
   , 1 )
 LinkEdgesAlign[x0,0,3]: EdgesAlign[x0,0,3] = MIN ( EdgeDistanceAbs[x0,0,3]
   , 1 )
 LinkEdgesAlign[x0,0,4]: EdgesAlign[x0,0,4] = MIN ( EdgeDistanceAbs[x0,0,4]
   , 1 )
 LinkEdgesAlign[x0,0,5]: EdgesAlign[x0,0,5] = MIN ( EdgeDistanceAbs[x0,0,5]
   , 1 )
 LinkEdgesAlign[x0,0,6]: EdgesAlign[x0,0,6] = MIN ( EdgeDistanceAbs[x0,0,6]
   , 1 )
 LinkEdgesAlign[x0,0,7]: EdgesAlign[x0,0,7] = MIN ( EdgeDistanceAbs[x0,0,7]
   , 1 )
 LinkEdgesAlign[x0,1,0]: EdgesAlign[x0,1,0] = MIN ( EdgeDistanceAbs[x0,1,0]
   , 1 )
 LinkEdgesAlign[x0,1,1]: EdgesAlign[x0,1,1] = MIN ( EdgeDistanceAbs[x0,1,1]
   , 1 )
 LinkEdgesAlign[x0,1,2]: EdgesAlign[x0,1,2] = MIN ( EdgeDistanceAbs[x0,1,2]
   , 1 )
 LinkEdgesAlign[x0,1,3]: EdgesAlign[x0,1,3] = MIN ( EdgeDistanceAbs[x0,1,3]
   , 1 )
 LinkEdgesAlign[x0,1,4]: EdgesAlign[x0,1,4] = MIN ( EdgeDistanceAbs[x0,1,4]
   , 1 )
 LinkEdgesAlign[x0,1,5]: EdgesAlign[x0,1,5] = MIN ( EdgeDistanceAbs[x0,1,5]
   , 1 )
 LinkEdgesAlign[x0,1,6]: EdgesAlign[x0,1,6] = MIN ( EdgeDistanceAbs[x0,1,6]
   , 1 )
 LinkEdgesAlign[x0,1,7]: EdgesAlign[x0,1,7] = MIN ( EdgeDistanceAbs[x0,1,7]
   , 1 )
 LinkEdgesAlign[x0,2,0]: EdgesAlign[x0,2,0] = MIN ( EdgeDistanceAbs[x0,2,0]
   , 1 )
 LinkEdgesAlign[x0,2,1]: EdgesAlign[x0,2,1] = MIN ( EdgeDistanceAbs[x0,2,1]
   , 1 )
 LinkEdgesAlign[x0,2,2]: EdgesAlign[x0,2,2] = MIN ( EdgeDistanceAbs[x0,2,2]
   , 1 )
 LinkEdgesAlign[x0,2,3]: EdgesAlign[x0,2,3] = MIN ( EdgeDistanceAbs[x0,2,3]
   , 1 )
 LinkEdgesAlign[x0,2,4]: EdgesAlign[x0,2,4] = MIN ( EdgeDistanceAbs[x0,2,4]
   , 1 )
 LinkEdgesAlign[x0,2,5]: EdgesAlign[x0,2,5] = MIN ( EdgeDistanceAbs[x0,2,5]
   , 1 )
 LinkEdgesAlign[x0,2,6]: EdgesAlign[x0,2,6] = MIN ( EdgeDistanceAbs[x0,2,6]
   , 1 )
 LinkEdgesAlign[x0,2,7]: EdgesAlign[x0,2,7] = MIN ( EdgeDistanceAbs[x0,2,7]
   , 1 )
 LinkEdgesAlign[x0,3,0]: EdgesAlign[x0,3,0] = MIN ( EdgeDistanceAbs[x0,3,0]
   , 1 )
 LinkEdgesAlign[x0,3,1]: EdgesAlign[x0,3,1] = MIN ( EdgeDistanceAbs[x0,3,1]
   , 1 )
 LinkEdgesAlign[x0,3,2]: EdgesAlign[x0,3,2] = MIN ( EdgeDistanceAbs[x0,3,2]
   , 1 )
 LinkEdgesAlign[x0,3,3]: EdgesAlign[x0,3,3] = MIN ( EdgeDistanceAbs[x0,3,3]
   , 1 )
 LinkEdgesAlign[x0,3,4]: EdgesAlign[x0,3,4] = MIN ( EdgeDistanceAbs[x0,3,4]
   , 1 )
 LinkEdgesAlign[x0,3,5]: EdgesAlign[x0,3,5] = MIN ( EdgeDistanceAbs[x0,3,5]
   , 1 )
 LinkEdgesAlign[x0,3,6]: EdgesAlign[x0,3,6] = MIN ( EdgeDistanceAbs[x0,3,6]
   , 1 )
 LinkEdgesAlign[x0,3,7]: EdgesAlign[x0,3,7] = MIN ( EdgeDistanceAbs[x0,3,7]
   , 1 )
 LinkEdgesAlign[x0,4,0]: EdgesAlign[x0,4,0] = MIN ( EdgeDistanceAbs[x0,4,0]
   , 1 )
 LinkEdgesAlign[x0,4,1]: EdgesAlign[x0,4,1] = MIN ( EdgeDistanceAbs[x0,4,1]
   , 1 )
 LinkEdgesAlign[x0,4,2]: EdgesAlign[x0,4,2] = MIN ( EdgeDistanceAbs[x0,4,2]
   , 1 )
 LinkEdgesAlign[x0,4,3]: EdgesAlign[x0,4,3] = MIN ( EdgeDistanceAbs[x0,4,3]
   , 1 )
 LinkEdgesAlign[x0,4,4]: EdgesAlign[x0,4,4] = MIN ( EdgeDistanceAbs[x0,4,4]
   , 1 )
 LinkEdgesAlign[x0,4,5]: EdgesAlign[x0,4,5] = MIN ( EdgeDistanceAbs[x0,4,5]
   , 1 )
 LinkEdgesAlign[x0,4,6]: EdgesAlign[x0,4,6] = MIN ( EdgeDistanceAbs[x0,4,6]
   , 1 )
 LinkEdgesAlign[x0,4,7]: EdgesAlign[x0,4,7] = MIN ( EdgeDistanceAbs[x0,4,7]
   , 1 )
 LinkEdgesAlign[x0,5,0]: EdgesAlign[x0,5,0] = MIN ( EdgeDistanceAbs[x0,5,0]
   , 1 )
 LinkEdgesAlign[x0,5,1]: EdgesAlign[x0,5,1] = MIN ( EdgeDistanceAbs[x0,5,1]
   , 1 )
 LinkEdgesAlign[x0,5,2]: EdgesAlign[x0,5,2] = MIN ( EdgeDistanceAbs[x0,5,2]
   , 1 )
 LinkEdgesAlign[x0,5,3]: EdgesAlign[x0,5,3] = MIN ( EdgeDistanceAbs[x0,5,3]
   , 1 )
 LinkEdgesAlign[x0,5,4]: EdgesAlign[x0,5,4] = MIN ( EdgeDistanceAbs[x0,5,4]
   , 1 )
 LinkEdgesAlign[x0,5,5]: EdgesAlign[x0,5,5] = MIN ( EdgeDistanceAbs[x0,5,5]
   , 1 )
 LinkEdgesAlign[x0,5,6]: EdgesAlign[x0,5,6] = MIN ( EdgeDistanceAbs[x0,5,6]
   , 1 )
 LinkEdgesAlign[x0,5,7]: EdgesAlign[x0,5,7] = MIN ( EdgeDistanceAbs[x0,5,7]
   , 1 )
 LinkEdgesAlign[x0,6,0]: EdgesAlign[x0,6,0] = MIN ( EdgeDistanceAbs[x0,6,0]
   , 1 )
 LinkEdgesAlign[x0,6,1]: EdgesAlign[x0,6,1] = MIN ( EdgeDistanceAbs[x0,6,1]
   , 1 )
 LinkEdgesAlign[x0,6,2]: EdgesAlign[x0,6,2] = MIN ( EdgeDistanceAbs[x0,6,2]
   , 1 )
 LinkEdgesAlign[x0,6,3]: EdgesAlign[x0,6,3] = MIN ( EdgeDistanceAbs[x0,6,3]
   , 1 )
 LinkEdgesAlign[x0,6,4]: EdgesAlign[x0,6,4] = MIN ( EdgeDistanceAbs[x0,6,4]
   , 1 )
 LinkEdgesAlign[x0,6,5]: EdgesAlign[x0,6,5] = MIN ( EdgeDistanceAbs[x0,6,5]
   , 1 )
 LinkEdgesAlign[x0,6,6]: EdgesAlign[x0,6,6] = MIN ( EdgeDistanceAbs[x0,6,6]
   , 1 )
 LinkEdgesAlign[x0,6,7]: EdgesAlign[x0,6,7] = MIN ( EdgeDistanceAbs[x0,6,7]
   , 1 )
 LinkEdgesAlign[x0,7,0]: EdgesAlign[x0,7,0] = MIN ( EdgeDistanceAbs[x0,7,0]
   , 1 )
 LinkEdgesAlign[x0,7,1]: EdgesAlign[x0,7,1] = MIN ( EdgeDistanceAbs[x0,7,1]
   , 1 )
 LinkEdgesAlign[x0,7,2]: EdgesAlign[x0,7,2] = MIN ( EdgeDistanceAbs[x0,7,2]
   , 1 )
 LinkEdgesAlign[x0,7,3]: EdgesAlign[x0,7,3] = MIN ( EdgeDistanceAbs[x0,7,3]
   , 1 )
 LinkEdgesAlign[x0,7,4]: EdgesAlign[x0,7,4] = MIN ( EdgeDistanceAbs[x0,7,4]
   , 1 )
 LinkEdgesAlign[x0,7,5]: EdgesAlign[x0,7,5] = MIN ( EdgeDistanceAbs[x0,7,5]
   , 1 )
 LinkEdgesAlign[x0,7,6]: EdgesAlign[x0,7,6] = MIN ( EdgeDistanceAbs[x0,7,6]
   , 1 )
 LinkEdgesAlign[x0,7,7]: EdgesAlign[x0,7,7] = MIN ( EdgeDistanceAbs[x0,7,7]
   , 1 )
 LinkEdgesAlign[y0,0,0]: EdgesAlign[y0,0,0] = MIN ( EdgeDistanceAbs[y0,0,0]
   , 1 )
 LinkEdgesAlign[y0,0,1]: EdgesAlign[y0,0,1] = MIN ( EdgeDistanceAbs[y0,0,1]
   , 1 )
 LinkEdgesAlign[y0,0,2]: EdgesAlign[y0,0,2] = MIN ( EdgeDistanceAbs[y0,0,2]
   , 1 )
 LinkEdgesAlign[y0,0,3]: EdgesAlign[y0,0,3] = MIN ( EdgeDistanceAbs[y0,0,3]
   , 1 )
 LinkEdgesAlign[y0,0,4]: EdgesAlign[y0,0,4] = MIN ( EdgeDistanceAbs[y0,0,4]
   , 1 )
 LinkEdgesAlign[y0,0,5]: EdgesAlign[y0,0,5] = MIN ( EdgeDistanceAbs[y0,0,5]
   , 1 )
 LinkEdgesAlign[y0,0,6]: EdgesAlign[y0,0,6] = MIN ( EdgeDistanceAbs[y0,0,6]
   , 1 )
 LinkEdgesAlign[y0,0,7]: EdgesAlign[y0,0,7] = MIN ( EdgeDistanceAbs[y0,0,7]
   , 1 )
 LinkEdgesAlign[y0,1,0]: EdgesAlign[y0,1,0] = MIN ( EdgeDistanceAbs[y0,1,0]
   , 1 )
 LinkEdgesAlign[y0,1,1]: EdgesAlign[y0,1,1] = MIN ( EdgeDistanceAbs[y0,1,1]
   , 1 )
 LinkEdgesAlign[y0,1,2]: EdgesAlign[y0,1,2] = MIN ( EdgeDistanceAbs[y0,1,2]
   , 1 )
 LinkEdgesAlign[y0,1,3]: EdgesAlign[y0,1,3] = MIN ( EdgeDistanceAbs[y0,1,3]
   , 1 )
 LinkEdgesAlign[y0,1,4]: EdgesAlign[y0,1,4] = MIN ( EdgeDistanceAbs[y0,1,4]
   , 1 )
 LinkEdgesAlign[y0,1,5]: EdgesAlign[y0,1,5] = MIN ( EdgeDistanceAbs[y0,1,5]
   , 1 )
 LinkEdgesAlign[y0,1,6]: EdgesAlign[y0,1,6] = MIN ( EdgeDistanceAbs[y0,1,6]
   , 1 )
 LinkEdgesAlign[y0,1,7]: EdgesAlign[y0,1,7] = MIN ( EdgeDistanceAbs[y0,1,7]
   , 1 )
 LinkEdgesAlign[y0,2,0]: EdgesAlign[y0,2,0] = MIN ( EdgeDistanceAbs[y0,2,0]
   , 1 )
 LinkEdgesAlign[y0,2,1]: EdgesAlign[y0,2,1] = MIN ( EdgeDistanceAbs[y0,2,1]
   , 1 )
 LinkEdgesAlign[y0,2,2]: EdgesAlign[y0,2,2] = MIN ( EdgeDistanceAbs[y0,2,2]
   , 1 )
 LinkEdgesAlign[y0,2,3]: EdgesAlign[y0,2,3] = MIN ( EdgeDistanceAbs[y0,2,3]
   , 1 )
 LinkEdgesAlign[y0,2,4]: EdgesAlign[y0,2,4] = MIN ( EdgeDistanceAbs[y0,2,4]
   , 1 )
 LinkEdgesAlign[y0,2,5]: EdgesAlign[y0,2,5] = MIN ( EdgeDistanceAbs[y0,2,5]
   , 1 )
 LinkEdgesAlign[y0,2,6]: EdgesAlign[y0,2,6] = MIN ( EdgeDistanceAbs[y0,2,6]
   , 1 )
 LinkEdgesAlign[y0,2,7]: EdgesAlign[y0,2,7] = MIN ( EdgeDistanceAbs[y0,2,7]
   , 1 )
 LinkEdgesAlign[y0,3,0]: EdgesAlign[y0,3,0] = MIN ( EdgeDistanceAbs[y0,3,0]
   , 1 )
 LinkEdgesAlign[y0,3,1]: EdgesAlign[y0,3,1] = MIN ( EdgeDistanceAbs[y0,3,1]
   , 1 )
 LinkEdgesAlign[y0,3,2]: EdgesAlign[y0,3,2] = MIN ( EdgeDistanceAbs[y0,3,2]
   , 1 )
 LinkEdgesAlign[y0,3,3]: EdgesAlign[y0,3,3] = MIN ( EdgeDistanceAbs[y0,3,3]
   , 1 )
 LinkEdgesAlign[y0,3,4]: EdgesAlign[y0,3,4] = MIN ( EdgeDistanceAbs[y0,3,4]
   , 1 )
 LinkEdgesAlign[y0,3,5]: EdgesAlign[y0,3,5] = MIN ( EdgeDistanceAbs[y0,3,5]
   , 1 )
 LinkEdgesAlign[y0,3,6]: EdgesAlign[y0,3,6] = MIN ( EdgeDistanceAbs[y0,3,6]
   , 1 )
 LinkEdgesAlign[y0,3,7]: EdgesAlign[y0,3,7] = MIN ( EdgeDistanceAbs[y0,3,7]
   , 1 )
 LinkEdgesAlign[y0,4,0]: EdgesAlign[y0,4,0] = MIN ( EdgeDistanceAbs[y0,4,0]
   , 1 )
 LinkEdgesAlign[y0,4,1]: EdgesAlign[y0,4,1] = MIN ( EdgeDistanceAbs[y0,4,1]
   , 1 )
 LinkEdgesAlign[y0,4,2]: EdgesAlign[y0,4,2] = MIN ( EdgeDistanceAbs[y0,4,2]
   , 1 )
 LinkEdgesAlign[y0,4,3]: EdgesAlign[y0,4,3] = MIN ( EdgeDistanceAbs[y0,4,3]
   , 1 )
 LinkEdgesAlign[y0,4,4]: EdgesAlign[y0,4,4] = MIN ( EdgeDistanceAbs[y0,4,4]
   , 1 )
 LinkEdgesAlign[y0,4,5]: EdgesAlign[y0,4,5] = MIN ( EdgeDistanceAbs[y0,4,5]
   , 1 )
 LinkEdgesAlign[y0,4,6]: EdgesAlign[y0,4,6] = MIN ( EdgeDistanceAbs[y0,4,6]
   , 1 )
 LinkEdgesAlign[y0,4,7]: EdgesAlign[y0,4,7] = MIN ( EdgeDistanceAbs[y0,4,7]
   , 1 )
 LinkEdgesAlign[y0,5,0]: EdgesAlign[y0,5,0] = MIN ( EdgeDistanceAbs[y0,5,0]
   , 1 )
 LinkEdgesAlign[y0,5,1]: EdgesAlign[y0,5,1] = MIN ( EdgeDistanceAbs[y0,5,1]
   , 1 )
 LinkEdgesAlign[y0,5,2]: EdgesAlign[y0,5,2] = MIN ( EdgeDistanceAbs[y0,5,2]
   , 1 )
 LinkEdgesAlign[y0,5,3]: EdgesAlign[y0,5,3] = MIN ( EdgeDistanceAbs[y0,5,3]
   , 1 )
 LinkEdgesAlign[y0,5,4]: EdgesAlign[y0,5,4] = MIN ( EdgeDistanceAbs[y0,5,4]
   , 1 )
 LinkEdgesAlign[y0,5,5]: EdgesAlign[y0,5,5] = MIN ( EdgeDistanceAbs[y0,5,5]
   , 1 )
 LinkEdgesAlign[y0,5,6]: EdgesAlign[y0,5,6] = MIN ( EdgeDistanceAbs[y0,5,6]
   , 1 )
 LinkEdgesAlign[y0,5,7]: EdgesAlign[y0,5,7] = MIN ( EdgeDistanceAbs[y0,5,7]
   , 1 )
 LinkEdgesAlign[y0,6,0]: EdgesAlign[y0,6,0] = MIN ( EdgeDistanceAbs[y0,6,0]
   , 1 )
 LinkEdgesAlign[y0,6,1]: EdgesAlign[y0,6,1] = MIN ( EdgeDistanceAbs[y0,6,1]
   , 1 )
 LinkEdgesAlign[y0,6,2]: EdgesAlign[y0,6,2] = MIN ( EdgeDistanceAbs[y0,6,2]
   , 1 )
 LinkEdgesAlign[y0,6,3]: EdgesAlign[y0,6,3] = MIN ( EdgeDistanceAbs[y0,6,3]
   , 1 )
 LinkEdgesAlign[y0,6,4]: EdgesAlign[y0,6,4] = MIN ( EdgeDistanceAbs[y0,6,4]
   , 1 )
 LinkEdgesAlign[y0,6,5]: EdgesAlign[y0,6,5] = MIN ( EdgeDistanceAbs[y0,6,5]
   , 1 )
 LinkEdgesAlign[y0,6,6]: EdgesAlign[y0,6,6] = MIN ( EdgeDistanceAbs[y0,6,6]
   , 1 )
 LinkEdgesAlign[y0,6,7]: EdgesAlign[y0,6,7] = MIN ( EdgeDistanceAbs[y0,6,7]
   , 1 )
 LinkEdgesAlign[y0,7,0]: EdgesAlign[y0,7,0] = MIN ( EdgeDistanceAbs[y0,7,0]
   , 1 )
 LinkEdgesAlign[y0,7,1]: EdgesAlign[y0,7,1] = MIN ( EdgeDistanceAbs[y0,7,1]
   , 1 )
 LinkEdgesAlign[y0,7,2]: EdgesAlign[y0,7,2] = MIN ( EdgeDistanceAbs[y0,7,2]
   , 1 )
 LinkEdgesAlign[y0,7,3]: EdgesAlign[y0,7,3] = MIN ( EdgeDistanceAbs[y0,7,3]
   , 1 )
 LinkEdgesAlign[y0,7,4]: EdgesAlign[y0,7,4] = MIN ( EdgeDistanceAbs[y0,7,4]
   , 1 )
 LinkEdgesAlign[y0,7,5]: EdgesAlign[y0,7,5] = MIN ( EdgeDistanceAbs[y0,7,5]
   , 1 )
 LinkEdgesAlign[y0,7,6]: EdgesAlign[y0,7,6] = MIN ( EdgeDistanceAbs[y0,7,6]
   , 1 )
 LinkEdgesAlign[y0,7,7]: EdgesAlign[y0,7,7] = MIN ( EdgeDistanceAbs[y0,7,7]
   , 1 )
 LinkEdgesAlign[x1,0,0]: EdgesAlign[x1,0,0] = MIN ( EdgeDistanceAbs[x1,0,0]
   , 1 )
 LinkEdgesAlign[x1,0,1]: EdgesAlign[x1,0,1] = MIN ( EdgeDistanceAbs[x1,0,1]
   , 1 )
 LinkEdgesAlign[x1,0,2]: EdgesAlign[x1,0,2] = MIN ( EdgeDistanceAbs[x1,0,2]
   , 1 )
 LinkEdgesAlign[x1,0,3]: EdgesAlign[x1,0,3] = MIN ( EdgeDistanceAbs[x1,0,3]
   , 1 )
 LinkEdgesAlign[x1,0,4]: EdgesAlign[x1,0,4] = MIN ( EdgeDistanceAbs[x1,0,4]
   , 1 )
 LinkEdgesAlign[x1,0,5]: EdgesAlign[x1,0,5] = MIN ( EdgeDistanceAbs[x1,0,5]
   , 1 )
 LinkEdgesAlign[x1,0,6]: EdgesAlign[x1,0,6] = MIN ( EdgeDistanceAbs[x1,0,6]
   , 1 )
 LinkEdgesAlign[x1,0,7]: EdgesAlign[x1,0,7] = MIN ( EdgeDistanceAbs[x1,0,7]
   , 1 )
 LinkEdgesAlign[x1,1,0]: EdgesAlign[x1,1,0] = MIN ( EdgeDistanceAbs[x1,1,0]
   , 1 )
 LinkEdgesAlign[x1,1,1]: EdgesAlign[x1,1,1] = MIN ( EdgeDistanceAbs[x1,1,1]
   , 1 )
 LinkEdgesAlign[x1,1,2]: EdgesAlign[x1,1,2] = MIN ( EdgeDistanceAbs[x1,1,2]
   , 1 )
 LinkEdgesAlign[x1,1,3]: EdgesAlign[x1,1,3] = MIN ( EdgeDistanceAbs[x1,1,3]
   , 1 )
 LinkEdgesAlign[x1,1,4]: EdgesAlign[x1,1,4] = MIN ( EdgeDistanceAbs[x1,1,4]
   , 1 )
 LinkEdgesAlign[x1,1,5]: EdgesAlign[x1,1,5] = MIN ( EdgeDistanceAbs[x1,1,5]
   , 1 )
 LinkEdgesAlign[x1,1,6]: EdgesAlign[x1,1,6] = MIN ( EdgeDistanceAbs[x1,1,6]
   , 1 )
 LinkEdgesAlign[x1,1,7]: EdgesAlign[x1,1,7] = MIN ( EdgeDistanceAbs[x1,1,7]
   , 1 )
 LinkEdgesAlign[x1,2,0]: EdgesAlign[x1,2,0] = MIN ( EdgeDistanceAbs[x1,2,0]
   , 1 )
 LinkEdgesAlign[x1,2,1]: EdgesAlign[x1,2,1] = MIN ( EdgeDistanceAbs[x1,2,1]
   , 1 )
 LinkEdgesAlign[x1,2,2]: EdgesAlign[x1,2,2] = MIN ( EdgeDistanceAbs[x1,2,2]
   , 1 )
 LinkEdgesAlign[x1,2,3]: EdgesAlign[x1,2,3] = MIN ( EdgeDistanceAbs[x1,2,3]
   , 1 )
 LinkEdgesAlign[x1,2,4]: EdgesAlign[x1,2,4] = MIN ( EdgeDistanceAbs[x1,2,4]
   , 1 )
 LinkEdgesAlign[x1,2,5]: EdgesAlign[x1,2,5] = MIN ( EdgeDistanceAbs[x1,2,5]
   , 1 )
 LinkEdgesAlign[x1,2,6]: EdgesAlign[x1,2,6] = MIN ( EdgeDistanceAbs[x1,2,6]
   , 1 )
 LinkEdgesAlign[x1,2,7]: EdgesAlign[x1,2,7] = MIN ( EdgeDistanceAbs[x1,2,7]
   , 1 )
 LinkEdgesAlign[x1,3,0]: EdgesAlign[x1,3,0] = MIN ( EdgeDistanceAbs[x1,3,0]
   , 1 )
 LinkEdgesAlign[x1,3,1]: EdgesAlign[x1,3,1] = MIN ( EdgeDistanceAbs[x1,3,1]
   , 1 )
 LinkEdgesAlign[x1,3,2]: EdgesAlign[x1,3,2] = MIN ( EdgeDistanceAbs[x1,3,2]
   , 1 )
 LinkEdgesAlign[x1,3,3]: EdgesAlign[x1,3,3] = MIN ( EdgeDistanceAbs[x1,3,3]
   , 1 )
 LinkEdgesAlign[x1,3,4]: EdgesAlign[x1,3,4] = MIN ( EdgeDistanceAbs[x1,3,4]
   , 1 )
 LinkEdgesAlign[x1,3,5]: EdgesAlign[x1,3,5] = MIN ( EdgeDistanceAbs[x1,3,5]
   , 1 )
 LinkEdgesAlign[x1,3,6]: EdgesAlign[x1,3,6] = MIN ( EdgeDistanceAbs[x1,3,6]
   , 1 )
 LinkEdgesAlign[x1,3,7]: EdgesAlign[x1,3,7] = MIN ( EdgeDistanceAbs[x1,3,7]
   , 1 )
 LinkEdgesAlign[x1,4,0]: EdgesAlign[x1,4,0] = MIN ( EdgeDistanceAbs[x1,4,0]
   , 1 )
 LinkEdgesAlign[x1,4,1]: EdgesAlign[x1,4,1] = MIN ( EdgeDistanceAbs[x1,4,1]
   , 1 )
 LinkEdgesAlign[x1,4,2]: EdgesAlign[x1,4,2] = MIN ( EdgeDistanceAbs[x1,4,2]
   , 1 )
 LinkEdgesAlign[x1,4,3]: EdgesAlign[x1,4,3] = MIN ( EdgeDistanceAbs[x1,4,3]
   , 1 )
 LinkEdgesAlign[x1,4,4]: EdgesAlign[x1,4,4] = MIN ( EdgeDistanceAbs[x1,4,4]
   , 1 )
 LinkEdgesAlign[x1,4,5]: EdgesAlign[x1,4,5] = MIN ( EdgeDistanceAbs[x1,4,5]
   , 1 )
 LinkEdgesAlign[x1,4,6]: EdgesAlign[x1,4,6] = MIN ( EdgeDistanceAbs[x1,4,6]
   , 1 )
 LinkEdgesAlign[x1,4,7]: EdgesAlign[x1,4,7] = MIN ( EdgeDistanceAbs[x1,4,7]
   , 1 )
 LinkEdgesAlign[x1,5,0]: EdgesAlign[x1,5,0] = MIN ( EdgeDistanceAbs[x1,5,0]
   , 1 )
 LinkEdgesAlign[x1,5,1]: EdgesAlign[x1,5,1] = MIN ( EdgeDistanceAbs[x1,5,1]
   , 1 )
 LinkEdgesAlign[x1,5,2]: EdgesAlign[x1,5,2] = MIN ( EdgeDistanceAbs[x1,5,2]
   , 1 )
 LinkEdgesAlign[x1,5,3]: EdgesAlign[x1,5,3] = MIN ( EdgeDistanceAbs[x1,5,3]
   , 1 )
 LinkEdgesAlign[x1,5,4]: EdgesAlign[x1,5,4] = MIN ( EdgeDistanceAbs[x1,5,4]
   , 1 )
 LinkEdgesAlign[x1,5,5]: EdgesAlign[x1,5,5] = MIN ( EdgeDistanceAbs[x1,5,5]
   , 1 )
 LinkEdgesAlign[x1,5,6]: EdgesAlign[x1,5,6] = MIN ( EdgeDistanceAbs[x1,5,6]
   , 1 )
 LinkEdgesAlign[x1,5,7]: EdgesAlign[x1,5,7] = MIN ( EdgeDistanceAbs[x1,5,7]
   , 1 )
 LinkEdgesAlign[x1,6,0]: EdgesAlign[x1,6,0] = MIN ( EdgeDistanceAbs[x1,6,0]
   , 1 )
 LinkEdgesAlign[x1,6,1]: EdgesAlign[x1,6,1] = MIN ( EdgeDistanceAbs[x1,6,1]
   , 1 )
 LinkEdgesAlign[x1,6,2]: EdgesAlign[x1,6,2] = MIN ( EdgeDistanceAbs[x1,6,2]
   , 1 )
 LinkEdgesAlign[x1,6,3]: EdgesAlign[x1,6,3] = MIN ( EdgeDistanceAbs[x1,6,3]
   , 1 )
 LinkEdgesAlign[x1,6,4]: EdgesAlign[x1,6,4] = MIN ( EdgeDistanceAbs[x1,6,4]
   , 1 )
 LinkEdgesAlign[x1,6,5]: EdgesAlign[x1,6,5] = MIN ( EdgeDistanceAbs[x1,6,5]
   , 1 )
 LinkEdgesAlign[x1,6,6]: EdgesAlign[x1,6,6] = MIN ( EdgeDistanceAbs[x1,6,6]
   , 1 )
 LinkEdgesAlign[x1,6,7]: EdgesAlign[x1,6,7] = MIN ( EdgeDistanceAbs[x1,6,7]
   , 1 )
 LinkEdgesAlign[x1,7,0]: EdgesAlign[x1,7,0] = MIN ( EdgeDistanceAbs[x1,7,0]
   , 1 )
 LinkEdgesAlign[x1,7,1]: EdgesAlign[x1,7,1] = MIN ( EdgeDistanceAbs[x1,7,1]
   , 1 )
 LinkEdgesAlign[x1,7,2]: EdgesAlign[x1,7,2] = MIN ( EdgeDistanceAbs[x1,7,2]
   , 1 )
 LinkEdgesAlign[x1,7,3]: EdgesAlign[x1,7,3] = MIN ( EdgeDistanceAbs[x1,7,3]
   , 1 )
 LinkEdgesAlign[x1,7,4]: EdgesAlign[x1,7,4] = MIN ( EdgeDistanceAbs[x1,7,4]
   , 1 )
 LinkEdgesAlign[x1,7,5]: EdgesAlign[x1,7,5] = MIN ( EdgeDistanceAbs[x1,7,5]
   , 1 )
 LinkEdgesAlign[x1,7,6]: EdgesAlign[x1,7,6] = MIN ( EdgeDistanceAbs[x1,7,6]
   , 1 )
 LinkEdgesAlign[x1,7,7]: EdgesAlign[x1,7,7] = MIN ( EdgeDistanceAbs[x1,7,7]
   , 1 )
 LinkEdgesAlign[y1,0,0]: EdgesAlign[y1,0,0] = MIN ( EdgeDistanceAbs[y1,0,0]
   , 1 )
 LinkEdgesAlign[y1,0,1]: EdgesAlign[y1,0,1] = MIN ( EdgeDistanceAbs[y1,0,1]
   , 1 )
 LinkEdgesAlign[y1,0,2]: EdgesAlign[y1,0,2] = MIN ( EdgeDistanceAbs[y1,0,2]
   , 1 )
 LinkEdgesAlign[y1,0,3]: EdgesAlign[y1,0,3] = MIN ( EdgeDistanceAbs[y1,0,3]
   , 1 )
 LinkEdgesAlign[y1,0,4]: EdgesAlign[y1,0,4] = MIN ( EdgeDistanceAbs[y1,0,4]
   , 1 )
 LinkEdgesAlign[y1,0,5]: EdgesAlign[y1,0,5] = MIN ( EdgeDistanceAbs[y1,0,5]
   , 1 )
 LinkEdgesAlign[y1,0,6]: EdgesAlign[y1,0,6] = MIN ( EdgeDistanceAbs[y1,0,6]
   , 1 )
 LinkEdgesAlign[y1,0,7]: EdgesAlign[y1,0,7] = MIN ( EdgeDistanceAbs[y1,0,7]
   , 1 )
 LinkEdgesAlign[y1,1,0]: EdgesAlign[y1,1,0] = MIN ( EdgeDistanceAbs[y1,1,0]
   , 1 )
 LinkEdgesAlign[y1,1,1]: EdgesAlign[y1,1,1] = MIN ( EdgeDistanceAbs[y1,1,1]
   , 1 )
 LinkEdgesAlign[y1,1,2]: EdgesAlign[y1,1,2] = MIN ( EdgeDistanceAbs[y1,1,2]
   , 1 )
 LinkEdgesAlign[y1,1,3]: EdgesAlign[y1,1,3] = MIN ( EdgeDistanceAbs[y1,1,3]
   , 1 )
 LinkEdgesAlign[y1,1,4]: EdgesAlign[y1,1,4] = MIN ( EdgeDistanceAbs[y1,1,4]
   , 1 )
 LinkEdgesAlign[y1,1,5]: EdgesAlign[y1,1,5] = MIN ( EdgeDistanceAbs[y1,1,5]
   , 1 )
 LinkEdgesAlign[y1,1,6]: EdgesAlign[y1,1,6] = MIN ( EdgeDistanceAbs[y1,1,6]
   , 1 )
 LinkEdgesAlign[y1,1,7]: EdgesAlign[y1,1,7] = MIN ( EdgeDistanceAbs[y1,1,7]
   , 1 )
 LinkEdgesAlign[y1,2,0]: EdgesAlign[y1,2,0] = MIN ( EdgeDistanceAbs[y1,2,0]
   , 1 )
 LinkEdgesAlign[y1,2,1]: EdgesAlign[y1,2,1] = MIN ( EdgeDistanceAbs[y1,2,1]
   , 1 )
 LinkEdgesAlign[y1,2,2]: EdgesAlign[y1,2,2] = MIN ( EdgeDistanceAbs[y1,2,2]
   , 1 )
 LinkEdgesAlign[y1,2,3]: EdgesAlign[y1,2,3] = MIN ( EdgeDistanceAbs[y1,2,3]
   , 1 )
 LinkEdgesAlign[y1,2,4]: EdgesAlign[y1,2,4] = MIN ( EdgeDistanceAbs[y1,2,4]
   , 1 )
 LinkEdgesAlign[y1,2,5]: EdgesAlign[y1,2,5] = MIN ( EdgeDistanceAbs[y1,2,5]
   , 1 )
 LinkEdgesAlign[y1,2,6]: EdgesAlign[y1,2,6] = MIN ( EdgeDistanceAbs[y1,2,6]
   , 1 )
 LinkEdgesAlign[y1,2,7]: EdgesAlign[y1,2,7] = MIN ( EdgeDistanceAbs[y1,2,7]
   , 1 )
 LinkEdgesAlign[y1,3,0]: EdgesAlign[y1,3,0] = MIN ( EdgeDistanceAbs[y1,3,0]
   , 1 )
 LinkEdgesAlign[y1,3,1]: EdgesAlign[y1,3,1] = MIN ( EdgeDistanceAbs[y1,3,1]
   , 1 )
 LinkEdgesAlign[y1,3,2]: EdgesAlign[y1,3,2] = MIN ( EdgeDistanceAbs[y1,3,2]
   , 1 )
 LinkEdgesAlign[y1,3,3]: EdgesAlign[y1,3,3] = MIN ( EdgeDistanceAbs[y1,3,3]
   , 1 )
 LinkEdgesAlign[y1,3,4]: EdgesAlign[y1,3,4] = MIN ( EdgeDistanceAbs[y1,3,4]
   , 1 )
 LinkEdgesAlign[y1,3,5]: EdgesAlign[y1,3,5] = MIN ( EdgeDistanceAbs[y1,3,5]
   , 1 )
 LinkEdgesAlign[y1,3,6]: EdgesAlign[y1,3,6] = MIN ( EdgeDistanceAbs[y1,3,6]
   , 1 )
 LinkEdgesAlign[y1,3,7]: EdgesAlign[y1,3,7] = MIN ( EdgeDistanceAbs[y1,3,7]
   , 1 )
 LinkEdgesAlign[y1,4,0]: EdgesAlign[y1,4,0] = MIN ( EdgeDistanceAbs[y1,4,0]
   , 1 )
 LinkEdgesAlign[y1,4,1]: EdgesAlign[y1,4,1] = MIN ( EdgeDistanceAbs[y1,4,1]
   , 1 )
 LinkEdgesAlign[y1,4,2]: EdgesAlign[y1,4,2] = MIN ( EdgeDistanceAbs[y1,4,2]
   , 1 )
 LinkEdgesAlign[y1,4,3]: EdgesAlign[y1,4,3] = MIN ( EdgeDistanceAbs[y1,4,3]
   , 1 )
 LinkEdgesAlign[y1,4,4]: EdgesAlign[y1,4,4] = MIN ( EdgeDistanceAbs[y1,4,4]
   , 1 )
 LinkEdgesAlign[y1,4,5]: EdgesAlign[y1,4,5] = MIN ( EdgeDistanceAbs[y1,4,5]
   , 1 )
 LinkEdgesAlign[y1,4,6]: EdgesAlign[y1,4,6] = MIN ( EdgeDistanceAbs[y1,4,6]
   , 1 )
 LinkEdgesAlign[y1,4,7]: EdgesAlign[y1,4,7] = MIN ( EdgeDistanceAbs[y1,4,7]
   , 1 )
 LinkEdgesAlign[y1,5,0]: EdgesAlign[y1,5,0] = MIN ( EdgeDistanceAbs[y1,5,0]
   , 1 )
 LinkEdgesAlign[y1,5,1]: EdgesAlign[y1,5,1] = MIN ( EdgeDistanceAbs[y1,5,1]
   , 1 )
 LinkEdgesAlign[y1,5,2]: EdgesAlign[y1,5,2] = MIN ( EdgeDistanceAbs[y1,5,2]
   , 1 )
 LinkEdgesAlign[y1,5,3]: EdgesAlign[y1,5,3] = MIN ( EdgeDistanceAbs[y1,5,3]
   , 1 )
 LinkEdgesAlign[y1,5,4]: EdgesAlign[y1,5,4] = MIN ( EdgeDistanceAbs[y1,5,4]
   , 1 )
 LinkEdgesAlign[y1,5,5]: EdgesAlign[y1,5,5] = MIN ( EdgeDistanceAbs[y1,5,5]
   , 1 )
 LinkEdgesAlign[y1,5,6]: EdgesAlign[y1,5,6] = MIN ( EdgeDistanceAbs[y1,5,6]
   , 1 )
 LinkEdgesAlign[y1,5,7]: EdgesAlign[y1,5,7] = MIN ( EdgeDistanceAbs[y1,5,7]
   , 1 )
 LinkEdgesAlign[y1,6,0]: EdgesAlign[y1,6,0] = MIN ( EdgeDistanceAbs[y1,6,0]
   , 1 )
 LinkEdgesAlign[y1,6,1]: EdgesAlign[y1,6,1] = MIN ( EdgeDistanceAbs[y1,6,1]
   , 1 )
 LinkEdgesAlign[y1,6,2]: EdgesAlign[y1,6,2] = MIN ( EdgeDistanceAbs[y1,6,2]
   , 1 )
 LinkEdgesAlign[y1,6,3]: EdgesAlign[y1,6,3] = MIN ( EdgeDistanceAbs[y1,6,3]
   , 1 )
 LinkEdgesAlign[y1,6,4]: EdgesAlign[y1,6,4] = MIN ( EdgeDistanceAbs[y1,6,4]
   , 1 )
 LinkEdgesAlign[y1,6,5]: EdgesAlign[y1,6,5] = MIN ( EdgeDistanceAbs[y1,6,5]
   , 1 )
 LinkEdgesAlign[y1,6,6]: EdgesAlign[y1,6,6] = MIN ( EdgeDistanceAbs[y1,6,6]
   , 1 )
 LinkEdgesAlign[y1,6,7]: EdgesAlign[y1,6,7] = MIN ( EdgeDistanceAbs[y1,6,7]
   , 1 )
 LinkEdgesAlign[y1,7,0]: EdgesAlign[y1,7,0] = MIN ( EdgeDistanceAbs[y1,7,0]
   , 1 )
 LinkEdgesAlign[y1,7,1]: EdgesAlign[y1,7,1] = MIN ( EdgeDistanceAbs[y1,7,1]
   , 1 )
 LinkEdgesAlign[y1,7,2]: EdgesAlign[y1,7,2] = MIN ( EdgeDistanceAbs[y1,7,2]
   , 1 )
 LinkEdgesAlign[y1,7,3]: EdgesAlign[y1,7,3] = MIN ( EdgeDistanceAbs[y1,7,3]
   , 1 )
 LinkEdgesAlign[y1,7,4]: EdgesAlign[y1,7,4] = MIN ( EdgeDistanceAbs[y1,7,4]
   , 1 )
 LinkEdgesAlign[y1,7,5]: EdgesAlign[y1,7,5] = MIN ( EdgeDistanceAbs[y1,7,5]
   , 1 )
 LinkEdgesAlign[y1,7,6]: EdgesAlign[y1,7,6] = MIN ( EdgeDistanceAbs[y1,7,6]
   , 1 )
 LinkEdgesAlign[y1,7,7]: EdgesAlign[y1,7,7] = MIN ( EdgeDistanceAbs[y1,7,7]
   , 1 )
End
