// parameter.h
// 定义模拟所需要的各参数

#ifndef PARAMETER_H_
#define PARAMETER_H_

#define INPUTFILE_PATH "./input/parameter"

#include "mytype.h"

typedef struct ParameterStr{

   	char potentialName[128]; // 势函数名
   	int xLat;             // X轴上的晶格数
   	int yLat;             // Y轴上的晶格数
   	int zLat;             // Z轴上的晶格数
   	int xProc;          // X轴上的进程数
   	int yProc;          // Y轴上的进程数
   	int zProc;          // Z轴上的进程数
   	int stepNums;         // 模拟的总步数
   	int printNums;      // 每多少步打印一次信息
   	double stepTime;          // 步长（飞秒）
   	double initTemper; // 初始温度

}Parameter;

// 从文件中解析出各参数
Parameter* readParameter(); 

#endif