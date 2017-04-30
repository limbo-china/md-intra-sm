// space.h
// 空间分解法

#ifndef SPACE_H_
#define SPACE_H_

#include "mytype.h"

struct ParameterStr;
struct LatticeStr;
struct SpacialStr;

//空间分解结构体
typedef struct SpacialStr{

	double3 myMin; // 本进程对应空间坐标最小值
	double3 myMax; // 本进程对应空间坐标最大值
	double3 myLength; // 本进程对应空间的长度

	double3 globalMin; //整个体系坐标最小值
	double3 globalMax; //整个体系坐标最大值
	double3 globalLength; //整个体系的长度

	int3 globalProcNum; // 各坐标轴上分解的空间数
	int3 position; // 本进程对应的空间位置
}Spacial;

// 空间分解，将模拟的体系分解成若干个部分，每个部分由一个进程处理
void initSpace(struct ParameterStr* para, struct LatticeStr* lattice, struct SpacialStr** spa);


#endif