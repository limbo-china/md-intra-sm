// mympi.h
// 程序并行的一些基本函数

#ifndef MYMPI_H_
#define MYMPI_H_

#include <mpi.h>

// 获取并行的总进程数
int getRankNums();

// 获取当前进程编号
int getMyRank();

// 判断当前进程是否为0进程
int ifZeroRank();

// 获取进程数和编号
void initRank();

// 同步所有进程
void parallelBarrier(const char* info);
#endif