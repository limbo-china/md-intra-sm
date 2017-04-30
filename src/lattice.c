#include "lattice.h"

#include <stdlib.h>
#include <string.h>

// 初始化晶格结构体
void initLatticeInfo(Lattice** lat){

	*lat = (Lattice*)malloc(sizeof(Lattice));
	Lattice* lattice = *lat;

	strcpy(lattice->latticeType, "FCC");
	strcpy(lattice->atomName, "Cu");
	lattice->atomM = 63.55;
	lattice->latticeConst = 3.615;
}