#include "space.h"
#include "mympi.h"
#include "error.h"
#include "parameter.h"
#include "lattice.h"
#include <stdlib.h>

// 空间分解，将模拟的体系分解成若干个部分，每个部分由一个进程处理
void initSpace(struct ParameterStr* para, struct LatticeStr* lattice, struct SpacialStr** spa){
	
	int myRank = getMyRank();

	// 整个体系各方向的长度
	double3 globalLength;
	globalLength[0] = para->xLat *lattice->latticeConst;
	globalLength[1] = para->yLat *lattice->latticeConst;
	globalLength[2] = para->zLat *lattice->latticeConst;

	// 进程xyz参数错误
	if(para->xProc * para->yProc * para->zProc != getRankNums()){
		errorInfo(procNum);
		exit(procNum);
	}

	*spa = (Spacial*)malloc(sizeof(Spacial));
	Spacial* space = *spa;
	
	space->globalProcNum[0] = para->xProc;
	space->globalProcNum[1] = para->yProc;
	space->globalProcNum[2] = para->zProc;

	//计算本进程对应的空间在体系中的位置
	
   	space->position[0] = myRank % space->globalProcNum[0];
   	myRank /= space->globalProcNum[0];
   	space->position[1] = myRank % space->globalProcNum[1];
   	space->position[2] = myRank / space->globalProcNum[1];

   	// 初始化空间各坐标
   	for (int i = 0; i < 3; i++)
   	{
     	space->globalMin[i] = 0;
      	space->globalMax[i] = globalLength[i];
      	space->globalLength[i] = space->globalMax[i] - space->globalMin[i];
   	}
   	for (int i = 0; i < 3; i++)
   	{
      	space->myLength[i] = space->globalLength[i] / space->globalProcNum[i];
      	space->myMin[i] = space->globalMin[i] +  space->position[i]    * space->myLength[i];
      	space->myMax[i] = space->globalMin[i] + (space->position[i]+1) * space->myLength[i];
   	}

}