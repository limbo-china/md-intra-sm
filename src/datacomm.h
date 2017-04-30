// datacomm.h
// 原子的位置、动量等数据通信
#ifndef DATACOMM_H_
#define DATACOMM_H_

#include "mytype.h"

struct SpacialStr;
struct CellStr;
struct SystemStr;

typedef struct DataCommStr{

	// 邻居进程的序号
	int neighborProc[6];

	// 缓冲区大小
	int bufSize;

	// 各方向上需要通信的细胞数量
	int commCellNum[6];

	// 各方向上通信的细胞链表
	int *commCells[6];

}DataComm;

// 需要通信的原子数据
typedef struct atomDataStr{

	int id;
	double3 pos; // 原子坐标
	double3 momenta; // 原子动量

}AtomData;

enum Neighbor{
	X_NEG,X_POS,Y_NEG,Y_POS,Z_NEG,Z_POS
};

// 初始化结构体
void initComm(DataComm** comm, struct SpacialStr* space, struct CellStr* cells);

// 找出指定维度上所有通信部分的细胞
int* findCommCells(struct CellStr* cells, enum Neighbor dimen, int num);

// 将待发送的原子数据加入缓冲区内,返回加入缓冲区内的数据个数
int addSendData(struct SystemStr* sys, void* buf, enum Neighbor dimen);

// 处理已接收的其他进程的原子数据
void procRecvData(struct SystemStr* sys, void* buf, int size);

#endif