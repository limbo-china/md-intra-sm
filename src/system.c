#include "system.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

//初始化模拟体系
System* initSystem(Parameter* para){

	System* sys = (System*)malloc(sizeof(System));
	memset(sys, 0, sizeof(System));

    initEnergy(&sys->energy);
   	initPotInfo(&sys->potential);
    printPotential(stdout, sys->potential);
   	initLatticeInfo(&sys->lattice);
    //printLattice(stdout, sys->lattice);
    initSpace(para, sys->lattice, &sys->space);
    initCells(sys->space, sys->potential, &sys->cells);
    initAtoms(sys->cells, &sys->atoms);

    distributeAtoms(sys, para);
    initTemperature(sys, para);

    initComm(&sys->datacomm, sys->space, sys->cells);

    adjustAtoms(sys);

    //MPI_Allreduce(&sys->atoms->myNum, &sys->atoms->totalNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //printTotalAtom(stdout,sys->atoms);

    computeForce(sys);


    return sys;
}