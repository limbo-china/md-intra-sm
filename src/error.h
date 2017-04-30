// error.h
// 错误和异常的定义

#ifndef ERROR_H_
#define ERROR_H_

enum ErrorPtr{

	normal,
	procNum,
	errNums
};


// 提示错误信息
void errorInfo(const enum ErrorPtr ptr);

#endif