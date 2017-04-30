//timer.h 
//作为程序各计算部分的计时器，用于性能分析

#ifndef TIMER_H_
#define TIMER_H_

#include <inttypes.h>

//计时器数组指针
enum TimerPtr{
	total,
	loop,
	force,
	communication,
	adjustatom,
	reduce,
	test,
	timerNums
};

//计时器结构体,单位均为us
typedef struct TimerStr{
	uint64_t begin; //开始时间
	uint64_t global; //所有调用的总时间消耗
	uint64_t delta; //最近一次调用的时间消耗
}Timer;

// 启动定时器
void beginTimer(const enum TimerPtr ptr); 

// 停止定时器
void endTimer(const enum TimerPtr ptr);

// 获取定时器总时间
double getGlobalTime(const enum TimerPtr ptr);

#endif