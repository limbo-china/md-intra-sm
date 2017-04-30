// potential.h
// 计算原子间作用力时所必需的势函数

#ifndef POTENTIAL_H_
#define POTENTIAL_H_

struct SystemStr;

typedef struct PotentialStr{
	
	char potentialType[30]; //势函数类型
	
	 // double De;
  //   	double re;
  //   	double Beta;

    	double cutoff; //截断距离
	double sigma;
   double epsilon;

	//void  (*computeforce)(struct SystemStr* sys); // 计算相互作用力的函数
   	//可以单独拿出来 void (*print)(FILE* f, Potential* potential); // 打印势函数相关信息的函数
   	//void (*free)(Potential* potential); // 释放结构体空间的函数
   	
}Potential;

// 初始化势函数结构体
void initPotInfo(Potential** pot);

// 释放结构体空间
void potentialFree(Potential* potential);

// 根据势函数，求原子间的相互作用力
void computeForce(struct SystemStr* sys);

#endif