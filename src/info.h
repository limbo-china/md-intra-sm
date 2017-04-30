// info.h
// 打印模拟时各部分的相关信息，展现给用户

#ifndef INFO_H_
#define INFO_H_

#include "parameter.h"
#include "potential.h"
#include "lattice.h"
#include "atom.h"
#include "energy.h"

#include <stdio.h>

// 打印模拟时所需的各参数信息
void printPara(FILE* f, Parameter* para);

// 打印势函数的相关信息
void printPotential(FILE* f, Potential* potential);

// 打印所模拟晶格的相关信息
void printLattice(FILE* f, Lattice* lattice);

// 跟踪模拟体系的总原子数
void printTotalAtom(FILE* f, Atom* atoms);

// 输出体系的温度
void printTemper(FILE*f, Energy* ener, int totalAtom);
#endif