// lattice.h
// 所模拟的晶格的相关参数

#ifndef LATTICE_H_
#define LATTICE_H_

// 晶格结构体
typedef struct LatticeStr{
	
	char latticeType[10]; // 晶格类型
	char atomName[4];	   // 原子名称
	double atomM;  			// 元素的相对原子质量

   	double latticeConst;   // 晶格常数
 	
}Lattice;

// 初始化晶格结构体
void initLatticeInfo(Lattice** lat);

#endif
