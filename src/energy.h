// energe.h
// 体系的能量信息，包括动能、势能

#ifndef ENERGY_H_
#define ENERGY_H_

struct SystemStr;
typedef struct EnergyStr{

	double kineticEnergy;
	//double potentialEnergy;

}Energy;

// 初始化结构体
void initEnergy(Energy** ener);

// 计算体系的总动能
void computeTotalKinetic(struct SystemStr* sys);

#endif