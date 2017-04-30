#include "datacomm.h"
#include "cell.h"
#include "atom.h"
#include "system.h"
#include "mympi.h"

#include <stdlib.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

// 初始化结构体
void initComm(DataComm** comm, struct SpacialStr* space, struct CellStr* cells){

	*comm = (DataComm*)malloc(sizeof(DataComm));
    DataComm* datacomm = *comm;

    int* myPos = space->position;
    int* globalProcNum = space->globalProcNum;
    int* xyzCellNum = cells->xyzCellNum;

    // 计算邻居进程号
    datacomm->neighborProc[X_NEG] = (myPos[0] -1 + globalProcNum[0]) % globalProcNum[0]
    	+ globalProcNum[0] *(myPos[1] + globalProcNum[1]*myPos[2]);
    datacomm->neighborProc[X_POS] = (myPos[0] +1 + globalProcNum[0]) % globalProcNum[0]
    	+ globalProcNum[0] *(myPos[1] + globalProcNum[1]*myPos[2]);

    datacomm->neighborProc[Y_NEG] = myPos[0] + globalProcNum[0] *
    	( (myPos[1] -1 + globalProcNum[1]) % globalProcNum[1] + globalProcNum[1]*myPos[2]);
    datacomm->neighborProc[Y_POS] = myPos[0] + globalProcNum[0] *
    	( (myPos[1] +1 + globalProcNum[1]) % globalProcNum[1] + globalProcNum[1]*myPos[2]);

    datacomm->neighborProc[Z_NEG] = myPos[0] + globalProcNum[0] *
    	( myPos[1] + globalProcNum[1]*((myPos[2] -1 + globalProcNum[2]) % globalProcNum[2]));
    datacomm->neighborProc[Z_POS] = myPos[0] + globalProcNum[0] *
    	( myPos[1] + globalProcNum[1]*((myPos[2] +1 + globalProcNum[2]) % globalProcNum[2]));

    // if (ifZeroRank())
    // 	for(int i=0;i<6;i++)
    // 		printf("%d ",datacomm->neighborProc[i]);

    // 各方向需要通信的细胞数的最大值
    // int maxComm = MAX((xyzCellNum[0]+2)*(xyzCellNum[1]+2),
    // 	MAX((xyzCellNum[1]+2)*(xyzCellNum[2]+2),
    // 		(xyzCellNum[0]+2)*(xyzCellNum[2]+2)));
    // datacomm->bufSize = 2*maxComm*MAXPERCELL*sizeof(AtomData); //可改进为每个面需要自己的细胞数量

    datacomm->commCellNum[X_NEG] = 2*(xyzCellNum[1]+2)*(xyzCellNum[2]+2);
   	datacomm->commCellNum[X_POS] = 2*(xyzCellNum[1]+2)*(xyzCellNum[2]+2);
   	datacomm->commCellNum[Y_NEG] = 2*(xyzCellNum[0]+2)*(xyzCellNum[2]+2);
   	datacomm->commCellNum[Y_POS]  = 2*(xyzCellNum[0]+2)*(xyzCellNum[2]+2);
   	datacomm->commCellNum[Z_NEG]  = 2*(xyzCellNum[0]+2)*(xyzCellNum[1]+2);
   	datacomm->commCellNum[Z_POS]  = 2*(xyzCellNum[0]+2)*(xyzCellNum[1]+2);

   	for (int dimen=0; dimen<6; dimen++)
      datacomm->commCells[dimen] = findCommCells(cells, dimen, datacomm->commCellNum[dimen]);
}

// 找出指定维度上所有通信部分的细胞
int* findCommCells(struct CellStr* cells, enum Neighbor dimen, int num){
	
	int* commcells = malloc(num*sizeof(int));
   	int xBegin = -1;
   	int xEnd   = cells->xyzCellNum[0]+1;
   	int yBegin = -1;
   	int yEnd   = cells->xyzCellNum[1]+1;
   	int zBegin = -1;
   	int zEnd   = cells->xyzCellNum[2]+1;

   	if (dimen == X_NEG) xEnd = xBegin+2;
   	if (dimen == X_POS) xBegin = xEnd-2;
   	if (dimen == Y_NEG) yEnd = yBegin+2;
   	if (dimen == Y_POS) yBegin = yEnd-2;
   	if (dimen == Z_NEG) zEnd = zBegin+2;
   	if (dimen == Z_POS) zBegin = zEnd-2;

   	int n = 0;
   	int3 xyz;
   	for (xyz[0]=xBegin; xyz[0]<xEnd; xyz[0]++)
      	for (xyz[1]=yBegin; xyz[1]<yEnd; xyz[1]++)
         	for (xyz[2]=zBegin; xyz[2]<zEnd; xyz[2]++)
            	commcells[n++] = findCellByXYZ(cells, xyz);
   	//assert
   	return commcells;
}

// 将待发送的原子数据加入缓冲区内,返回加入缓冲区内的数据个数
 int addSendData(struct SystemStr* sys, void* buf, enum Neighbor dimen){

	int num = 0;
   	AtomData* buffer = (AtomData*) buf; // 可改进为拥有自己的缓冲区
    
   	int commCellNum = sys->datacomm->commCellNum[dimen]; 	
   	int* commCells = sys->datacomm->commCells[dimen];
   	int* spacePos = sys->space->position;
   	int* spaceNum = sys->space->globalProcNum;

   	double3 boundaryAdjust;
   	for(int i=0;i<3;i++)
   		boundaryAdjust[i]= 0.0;

   	// if (ifZeroRank())
    // {
    // 	printf ("commCellNum:%d\n",commCellNum);
    // 	printf("commcell:\n");
    // 	for(int i=0;i<commCellNum;i++){
    // 		printf("%d ",commCells[i]);
    // 	}
    // }
   	if(spacePos[0] == 0 && dimen == X_NEG)
   		boundaryAdjust[0] = sys->space->globalLength[0];
   	if(spacePos[0] == spaceNum[0]-1 && dimen == X_POS)
   		boundaryAdjust[0] = -1.0*sys->space->globalLength[0];
   	if(spacePos[1] == 0 && dimen == Y_NEG)
   		boundaryAdjust[1] = sys->space->globalLength[1];
   	if(spacePos[1] == spaceNum[1]-1 && dimen == Y_POS)
   		boundaryAdjust[1] = -1.0*sys->space->globalLength[1];
   	if(spacePos[2] == 0 && dimen == Z_NEG)
   		boundaryAdjust[2] = sys->space->globalLength[2];
   	if(spacePos[2] == spaceNum[2]-1 && dimen == Z_POS)
   		boundaryAdjust[2] = -1.0*sys->space->globalLength[2];
   
   	for (int nCell=0; nCell<commCellNum; nCell++)
   	{
      	int cell = commCells[nCell];
      	for (int n=cell*MAXPERCELL,count=0; count<sys->cells->atomNum[cell]; n++,count++)
      	{
      		for(int i=0;i<3;i++){
      			buffer[num].pos[i] = sys->atoms->pos[n][i]+boundaryAdjust[i];
      			buffer[num].momenta[i] = sys->atoms->momenta[n][i];
      		}
        	buffer[num].id  = sys->atoms->id[n];
         	num++;
      	}
   	}
   return num;
}

// 处理已接收的其他进程的原子数据
void procRecvData(struct SystemStr* sys, void* buf, int size){
	
	AtomData* buffer = (AtomData*) buf;

	double3 pos; //原子坐标
	double3 momenta; //原子动量

	int id;
	for (int num=0; num<size; num++)
   	{     	
      	for(int i=0;i<3;i++)
      	{
      		pos[i] = buffer[num].pos[i];
      		momenta[i] = buffer[num].momenta[i];
      	}
      	id = buffer[num].id;
      	
      	// 将原子分配至对应的细胞中
      	assignAtom(id, pos, sys, momenta);
   	}
}
