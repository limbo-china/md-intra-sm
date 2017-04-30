// initsystem.h
// 初始化体系的条件和状态，如体系的初始温度、原子初始坐标和速度、时间步长等

#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "potential.h"
#include "lattice.h"
#include "parameter.h"
#include "space.h"
#include "cell.h"
#include "atom.h"
#include "info.h"
#include "energy.h"
#include "datacomm.h"

// 所模拟的分子体系对于的结构体
typedef struct SystemStr
{  
	Energy* energy;     // 系统总能量

	Potential* potential;	  // 势函数相关数据

	Lattice* lattice;   // 所模拟的晶格相关信息

   	Spacial* space;        // 空间分解法所需的必要信息

   	DataComm* datacomm;  // 通信区域，交换原子信息

   	Cell* cells;       	// 选取linked-cell数据结构, 存储各cell相关信息数据

   	Atom* atoms;          // 存储原子的相关信息数据
     	 
} System;

//初始化模拟体系
System* initSystem();

#endif