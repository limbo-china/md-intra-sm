// cell.h
// 用细胞链表数据结构来存储原子信息

#ifndef CELL_H_
#define CELL_H_

#include "mytype.h"

struct SpacialStr;
struct PotentialStr;
struct CellStr;

// 细胞链表结构体
typedef struct CellStr{

	int* atomNum;         // 本空间各细胞中的原子数

	int3 xyzCellNum;     // xyz各维度上的细胞数
   	int myCellNum;     // 本空间的细胞数
   	int commCellNum;      // 本空间用于通信的细胞数
   	int totalCellNum;   // 总细胞数 = 实际细胞数 + 通信细胞数

   	double3 cellLength;       // 细胞在各维度上的长度
   	

}Cell;

// 初始化细胞链表
void initCells(struct SpacialStr* space, struct PotentialStr* potential, struct CellStr** cel);


// 根据坐标找到所在的细胞，返回细胞序号，即该空间中第几个细胞
int findCellByCoord(Cell* cells, struct SpacialStr* space, double3 coord);

// 根据细胞位置xyz返回细胞序号，即该空间中第几个细胞
int findCellByXYZ(Cell* cells, int* xyz);

// 根据细胞序号返回细胞位置xyz,与函数findCellByXYZ互为逆过程
void getXYZByCell(Cell* cells,int* xyz, int num);

#endif