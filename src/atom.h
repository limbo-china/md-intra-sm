// atom.h
// 原子结构体，存储原子的作用力，势能，速度等信息

#ifndef ATOM_H_
#define ATOM_H_

// 每个细胞中的原子数最大值
#define MAXPERCELL 64
#define kB (8.6173324e-5) //波尔兹曼常数

#include "mytype.h"
#include <mpi.h>

struct CellStr;
struct SystemStr;
struct ParameterStr;

typedef struct AtomStr{

	double3*  pos;     // 原子坐标
   	double3*  momenta;     // 原子动量
   	double3*  force;     // 原子受到的作用力 
   	double*  pot;     // 原子势能

	int myNum; // 本进程空间中的总原子数
	int totalNum; // 整个体系的总原子数

	int* id;      // 各原子id

}Atom;

// 初始化原子信息
void initAtoms(struct CellStr* cells, Atom** ato);

// 分配各原子到对应的细胞中
void distributeAtoms(struct SystemStr* sys, struct ParameterStr* para);

// 将指定原子根据其坐标，分配到对应的细胞中
void assignAtom(int id, double3 xyzpos, struct SystemStr* sys, double3 momenta);

// 初始化体系的温度，即原子的速度
void initTemperature(struct SystemStr* sys, struct ParameterStr* para);

// 调整原子所在细胞，并进行原子数据通信
void adjustAtoms(struct SystemStr* sys, void * buf, MPI_Win *win);

// 将cell1中的第N个原子移动到cell2中
void moveAtom(struct CellStr* cells, Atom* atoms, int n, int cell1, int cell2);

#endif