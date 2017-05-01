#include "timer.h"
#include "mympi.h"
#include "parameter.h"
#include "info.h"
#include "atom.h"
#include "potential.h"
#include "system.h"

#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

void updateMomenta(System* sys, Parameter* para); 
void updatePosition(System* sys, Parameter* para);

int main(int argc, char** argv){
	
	MPI_Init(&argc, &argv);
	initRank();

	char processor_name[20];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	fprintf(stdout, "%s\n",processor_name);
	//fprintf(stdout, "rankNums: %d\n", getRankNums());
	//fprintf(stdout, "myRank: %d\n", getMyRank());

	// 同步不起作用？
	//parallelBarrier("Begin to read parameters.");

	beginTimer(total);

	Parameter* para = readParameter();
	printPara(stdout,para);

	beginTimer(loop);
	//sleep(5);
	System* sys = initSystem(para);

	char* PutBuf = NULL;
	MPI_Win win;

	//printf("size: %d\n",sys->datacomm->bufSize );
	MPI_Win_allocate_shared(sys->datacomm->bufSize+2*sizeof(int), sizeof(char),
          MPI_INFO_NULL,MPI_COMM_WORLD, &PutBuf, &win);

	adjustAtoms(sys,PutBuf,&win);
	computeForce(sys);

	for(int i=1;i<=para->stepNums;i++){
    	updateMomenta(sys, para); 

    	updatePosition(sys, para);

    	//beginTimer(adjustatom);
    	adjustAtoms(sys,PutBuf,&win);
    	//endTimer(adjustatom);

    	beginTimer(force);
    	computeForce(sys);
    	endTimer(force);

    	updateMomenta(sys, para); 
    	if(i%para->printNums == 0){

    	//MPI_Allreduce(&sys->atoms->myNum, &sys->atoms->totalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    	//printTotalAtom(stdout,sys->atoms);
    		if(ifZeroRank())
			{
				printf("当前步数: %d 		",i);
			}		
    		computeTotalKinetic(sys);	
    		printTemper(stdout,sys->energy,sys->atoms->totalNum);
    	}
    }
	endTimer(loop);
	
	MPI_Win_free(&win);

	endTimer(total);

	double looptime = getGlobalTime(loop);
	double forcetime =  getGlobalTime(force);
	double commtime = getGlobalTime(communication);
	double globalloop=0.0,globalforce=0.0,globalcomm=0.0;

	MPI_Allreduce(&looptime, &globalloop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&forcetime, &globalforce, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&commtime, &globalcomm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if(ifZeroRank())
	{
		//fprintf(stdout, "total time: %g\n",getGlobalTime(total));
		fprintf(stdout, "\n------\n总循环时间: %g 平均: %g\n",globalloop,globalloop/getRankNums());
		//fprintf(stdout, "adjust time: %g\n",getGlobalTime(adjustatom));
		fprintf(stdout, "通信时间: %g 平均: %g\n",globalcomm,globalcomm/getRankNums());
		fprintf(stdout, "计算力时间: %g 平均: %g\n------\n",globalforce,globalforce/getRankNums());
		//fprintf(stdout, "test time: %g\n",getGlobalTime(test));
	}

	MPI_Finalize();
	return 0;
}

void updateMomenta(System* sys, Parameter* para){

	double t = 0.5*para->stepTime;

	for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
      	for (int n=MAXPERCELL*nCell,count=0; count<sys->cells->atomNum[nCell]; count++,n++)
      		for(int i=0;i<3;i++)
         		sys->atoms->momenta[n][i] += t*sys->atoms->force[n][i];
}
void updatePosition(System* sys, Parameter* para){

	double t = para->stepTime;
	double m = sys->lattice->atomM;

	for (int nCell=0; nCell<sys->cells->myCellNum; nCell++)
      	for (int n=MAXPERCELL*nCell,count=0; count<sys->cells->atomNum[nCell]; count++,n++)
      		for(int i=0;i<3;i++)
         		sys->atoms->pos[n][i] += t*sys->atoms->momenta[n][i]/m;
}