#include "parameter.h"
#include "getinput.h"

#include <stdlib.h>

// 从文件中解析出各参数
Parameter* readParameter(){

	Parameter* para = (Parameter*)malloc(sizeof(Parameter));

	// 初始化参数结构体（默认值）
	memset(para->potentialName, 0, 128);
	strcpy(para->potentialName, "Morse");
	para->xLat = 10;
	para->yLat = 10;
	para->zLat = 10;
	para->xProc = 2;
	para->yProc = 2;
	para->zProc = 2;
	para->stepNums = 100;
	para->printNums = 10;
	para->stepTime = 1.0;
	para->initTemper = 600.0;

	//可改进：参数值的格式检查-----------------

	// 从文件中获取参数，当没有对应的参数或指定使用默认值时，使用默认参数值
	char value_buff[VALUE_MAX_LENGTH];
	if(getInputValue(INPUTFILE_PATH, "potentialName", value_buff) == 1)
		strcpy(para->potentialName, value_buff);

	if(getInputValue(INPUTFILE_PATH, "xLatticeNum", value_buff) == 1)
		para->xLat = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "yLatticeNum", value_buff) == 1)
		para->yLat = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "zLatticeNum", value_buff) == 1)
		para->zLat = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "xProcessNum", value_buff) == 1)
		para->xProc = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "yProcessNum", value_buff) == 1)
		para->yProc = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "zProcessNum", value_buff) == 1)
		para->zProc = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "stepNums", value_buff) == 1)
		para->stepNums = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "printNums", value_buff) == 1)
		para->printNums = atoi(value_buff);

	if(getInputValue(INPUTFILE_PATH, "stepTime", value_buff) == 1)
		para->stepTime = strtod(value_buff, NULL);

	if(getInputValue(INPUTFILE_PATH, "initialTemperature", value_buff) == 1)
		para->initTemper = strtod(value_buff, NULL);

	return para;
}