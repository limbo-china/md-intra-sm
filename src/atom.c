#include "atom.h"
#include "timer.h"
#include "parameter.h"
#include "cell.h"
#include "system.h"
#include "lattice.h"
#include "energy.h"
#include "datacomm.h"
#include "random.h"
#include "info.h"
#include "mympi.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>

// 初始化原子信息结构体
void initAtoms(struct CellStr* cells, Atom** ato){

	*ato = (Atom*)malloc(sizeof(Atom));
    Atom* atoms = *ato;

   	int maxAtomNum = MAXPERCELL*cells->totalCellNum;
	
	atoms->myNum = 0;
   	atoms->totalNum = 0;

   	atoms->pos = (double3*) malloc(maxAtomNum*sizeof(double3));
   	atoms->momenta = (double3*) malloc(maxAtomNum*sizeof(double3));
   	atoms->force = (double3*) malloc(maxAtomNum*sizeof(double3));
   	atoms->pot = (double*)malloc(maxAtomNum*sizeof(double));
   	atoms->id = (int*)malloc(maxAtomNum*sizeof(int));

   	for (int i = 0; i < maxAtomNum; i++)
   	{
      	for(int j = 0; j< 3; j++){
      		atoms->pos[i][j] = 0.0;
      		atoms->momenta[i][j] = 0.0;
      		atoms->force[i][j] = 0.0;
      	}
      	atoms->pot[i] = 0.0;
      	atoms->id[i] = 0;
   	}
}

// 分配各原子到对应的细胞中
void distributeAtoms(struct SystemStr* sys, struct ParameterStr* para){
 
   	double latticeConst = sys->lattice->latticeConst;
   	int yLat = para->yLat;
   	int zLat = para->zLat;
   	double* myMin = sys->space->myMin;
   	double* myMax = sys->space->myMax;

   	double3 xyzpos;  // 原子坐标

   	double3 momenta; // 原子动量
   	momenta[0] = 0.0;
   	momenta[1] = 0.0;
   	momenta[2] = 0.0;
   
   	int n = 4;  // 每个晶胞4个原子
   	double3 displace[4] = { {0.25, 0.25, 0.25},
      	{0.25, 0.75, 0.75},
      	{0.75, 0.25, 0.75},
      	{0.75, 0.75, 0.25} };

   	// 分配原子
   	int begin[3];
   	int end[3];
   	for (int i = 0; i < 3; i++)
   	{
      	begin[i] = floor(myMin[i]/latticeConst);
      	end[i]   = ceil (myMax[i]/latticeConst);
   	}

   	for (int ix=begin[0]; ix<end[0]; ++ix)
      	for (int iy=begin[1]; iy<end[1]; ++iy)
         	for (int iz=begin[2]; iz<end[2]; ++iz)
            	for (int ib=0; ib<n; ++ib)
            	{
               		double xpos = (ix+displace[ib][0]) * latticeConst;
               		double ypos = (iy+displace[ib][1]) * latticeConst;
               		double zpos = (iz+displace[ib][2]) * latticeConst;
               		if (xpos < myMin[0] || xpos >= myMax[0]) continue;
               		if (ypos < myMin[1] || ypos >= myMax[1]) continue;
               		if (zpos < myMin[2] || zpos >= myMax[2]) continue;

               		// 计算原子的id
               		int id = ib+n*(iz+zLat*(iy+yLat*(ix)));

               		xyzpos[0] = xpos;
               		xyzpos[1] = ypos;
               		xyzpos[2] = zpos;

               		// 将此原子置于对应的细胞中,并初始化动量为0
               		assignAtom(id, xyzpos, sys, momenta);
            	}

   	// 利用mpi的reduce计算所有进程的总原子数量
   	//beginTimer(reduce);
   	MPI_Allreduce(&sys->atoms->myNum, &sys->atoms->totalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   	//endTimer(reduce);
    //printTotalAtom(stdout,sys->atoms);

   	//assert(s->atoms->nGlobal == nb*nx*ny*nz);
}

// 将指定原子分配到对应的细胞中
void assignAtom(int id, double3 xyzpos, struct SystemStr* sys, double3 momenta){
    
    // 根据原子坐标找到对应的细胞
    int cell = findCellByCoord(sys->cells, sys->space, xyzpos);

    // 计算此原子为本空间第几个原子
    int n = cell*MAXPERCELL;
    n = n + sys->cells->atomNum[cell];
   
    // 若不在通信区域中，本空间总原子数加1
    if (cell < sys->cells->myCellNum)
        sys->atoms->myNum++;

    // 当前细胞中的原子数加1
    sys->cells->atomNum[cell]++;

    sys->atoms->id[n] = id;

    // 对原子的位置坐标、动量赋值
    for(int i =0; i<3 ;i++){
        sys->atoms->pos[n][i] = xyzpos[i];
        sys->atoms->momenta[n][i] = momenta[i];
    }
}

// 初始化体系的温度，即原子的速度
void initTemperature(struct SystemStr* sys, struct ParameterStr* para){

    // 指定温度
    double temper = para->initTemper;
    // 原子质量
    double atomM = sys->lattice->atomM; 
 
    // 本空间所有原子总动量
    double3 myMomenta = {0.0,0.0,0.0};

    // 整个体系所有原子总动量
    double3 globalMomenta = {0.0,0.0,0.0};

    // 给定原子一个随机的速度及动量，并计算本空间的所有原子总动量
    for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
        for (int n=MAXPERCELL*nCell, count=0; count<sys->cells->atomNum[nCell]; count++, n++)
        {
            double sigma = sqrt(kB * temper/atomM);
            uint64_t seed = mkSeed(sys->atoms->id[n], 123);
            sys->atoms->momenta[n][0] = atomM * sigma * gasdev(&seed);
            sys->atoms->momenta[n][1] = atomM * sigma * gasdev(&seed);
            sys->atoms->momenta[n][2] = atomM * sigma * gasdev(&seed);

            myMomenta[0] += sys->atoms->momenta[n][0];
            myMomenta[1] += sys->atoms->momenta[n][1];
            myMomenta[2] += sys->atoms->momenta[n][2];
        }

    // 保证体系的总动量为0，在计算力之前需要调整为0

    // AllReduce, 得到整个体系的总动量
    MPI_Allreduce(myMomenta, globalMomenta, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // 每个原子需要中和的动量值
    double3 adjustMomenta;
    for(int i=0; i<3; i++)
        adjustMomenta[i] = -1 * globalMomenta[i]/sys->atoms->totalNum;

    // 调整各原子动量
    for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
        for (int n=MAXPERCELL*nCell, count=0; count<sys->cells->atomNum[nCell]; count++, n++)
            for(int i=0 ;i<3 ;i++)
                sys->atoms->momenta[n][i] += adjustMomenta[i];

    // 调整总动量为0后，需要调整体系的温度为指定温度
    computeTotalKinetic(sys);
    // 调整前的系统温度
    double t = (2*sys->energy->kineticEnergy)/(sys->atoms->totalNum*kB*3); 
    // 校正因子
    double factor = sqrt(temper/t);

    // 调整温度,乘以校正因子
    for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
        for (int n=MAXPERCELL*nCell, count=0; count<sys->cells->atomNum[nCell]; count++, n++)
            for(int i=0 ;i<3 ;i++)
                sys->atoms->momenta[n][i] *= factor; 

    // 计算调整后的总动能
    computeTotalKinetic(sys);
    //printTemper(stdout, sys->energy, sys->atoms->totalNum);
}

// 调整原子所在细胞，并进行原子数据通信(去掉了序号排序)
void adjustAtoms(struct SystemStr* sys){

    // 清空本空间外的细胞
    for (int i=sys->cells->myCellNum; i<sys->cells->totalCellNum; i++)
        sys->cells->atomNum[i] = 0;

    // 调整原子所在细胞
    for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
        for (int n = nCell*MAXPERCELL,count=0; count< sys->cells->atomNum[nCell];)
        {
            int nCell2 = findCellByCoord(sys->cells, sys->space,sys->atoms->pos[n+count]);
            if (nCell2 == nCell){
                count++;
                continue;
            }   
            moveAtom(sys->cells, sys->atoms, count, nCell, nCell2);           
        }

    //int haloatoms=0;
    //for (int i=sys->cells->myCellNum; i<sys->cells->totalCellNum; i++)
    //    haloatoms+=sys->cells->atomNum[i];
    //printf("haloatoms:%d\n",haloatoms);
    MPI_Allreduce(&sys->atoms->myNum, &sys->atoms->totalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //printTotalAtom(stdout,sys->atoms);
    //printf("adjust\n");

    // 与各邻居进程进行通信
    enum Neighbor dimen_POSI, dimen_NEGA;
    int neighbor_NEGA,neighbor_POSI;

    // 4个缓冲区，正负轴上发送缓冲区和接收缓冲区
    int bufsize = sys->datacomm->bufSize;
    char* negSendBuf = malloc(bufsize);
    char* posSendBuf = malloc(bufsize);
    char* negRecvBuf = malloc(bufsize);
    char* posRecvBuf = malloc(bufsize);

    beginTimer(communication);
    for(int dimen = 0;dimen<3;dimen++){
        dimen_NEGA = 2*dimen;
        dimen_POSI = 2*dimen + 1;

        neighbor_NEGA = sys->datacomm->neighborProc[dimen_NEGA];
        neighbor_POSI = sys->datacomm->neighborProc[dimen_POSI];

        // 将数据加入发送缓冲区
        int neg_send = addSendData(sys, negSendBuf, dimen_NEGA);
        int pos_send = addSendData(sys, posSendBuf, dimen_POSI);
        //printf("addsend\n");
        // if (ifZeroRank())
        // {
        //     printf("%d %d %d %d\n",neg_send,pos_send,sizeof(AtomData),bufsize);
        // }

        // 调用mpi_sendrecv函数，与邻居进程发送与接收原子数据
        int neg_recv,pos_recv;

        
        MPI_Status status1,status2;
        MPI_Sendrecv(negSendBuf, neg_send*sizeof(AtomData), MPI_BYTE, neighbor_NEGA, 0,
                posRecvBuf, bufsize, MPI_BYTE, neighbor_POSI, 0,
                MPI_COMM_WORLD, &status1);
        MPI_Get_count(&status1, MPI_BYTE, &pos_recv);
        MPI_Sendrecv(posSendBuf, pos_send*sizeof(AtomData), MPI_BYTE, neighbor_POSI, 0,
                negRecvBuf, bufsize, MPI_BYTE, neighbor_NEGA, 0,
                MPI_COMM_WORLD, &status2);
        MPI_Get_count(&status2, MPI_BYTE, &neg_recv);
       
        //printf("sendrecv\n");

        // if (ifZeroRank())
        // {
        //     printf("pos_recv:%d neg_recv:%d\n",pos_recv,neg_recv);
        // }

        // 处理接收到的原子数据，将原子分配至细胞中
        procRecvData(sys, posRecvBuf, pos_recv/sizeof(AtomData));
        procRecvData(sys, negRecvBuf, neg_recv/sizeof(AtomData));
    }
     endTimer(communication);

    // 通信结束，释放缓冲区
    free(negSendBuf);free(posSendBuf);free(negRecvBuf);free(posRecvBuf);
}

// 将cell1中的第N个原子移动到cell2中
void moveAtom(struct CellStr* cells, Atom* atoms, int n, int cell1, int cell2){

    // 先将原子数据写入细胞cell2中
    int n1 = MAXPERCELL*cell1+n;
    int n2 = MAXPERCELL*cell2+cells->atomNum[cell2];
    for(int i=0;i<3;i++){
        atoms->pos[n2][i]=atoms->pos[n1][i];
        atoms->momenta[n2][i]=atoms->momenta[n1][i];
        atoms->force[n2][i]=atoms->force[n1][i];
    }
    atoms->pot[n2] = atoms->pot[n1];
    atoms->id[n2] = atoms->id[n1];

    // cell2中原子总数加1
    cells->atomNum[cell2]++;

    //assert

    // cell1中原子总数减1
    cells->atomNum[cell1]--;
    
    // 若cell1中还有原子，则将最后一个原子数据填补至被移动的原子处
    if (cells->atomNum[cell1]){
        n1 = MAXPERCELL*cell1+cells->atomNum[cell1];
        n2 = MAXPERCELL*cell1+n;
        for(int i=0;i<3;i++){
            atoms->pos[n2][i]=atoms->pos[n1][i];
            atoms->momenta[n2][i]=atoms->momenta[n1][i];
            atoms->force[n2][i]=atoms->force[n1][i];
        }
        atoms->pot[n2] = atoms->pot[n1];
        atoms->id[n2] = atoms->id[n1];
    }

    // 若原子移动出了本空间,则本空间总原子数减1
    if (cell2 > cells->myCellNum) //此处不应该是>=??????
        atoms->myNum--;
}
