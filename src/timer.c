#include "timer.h"

#include <stdio.h>
#include <sys/time.h>
#include <string.h>

//计时器数组
static Timer timers[timerNums]; 

//获取当前时间(us)
static uint64_t getUsTime(); 
//us转化为s
static double usecToSec(uint64_t t); 
 
static uint64_t getUsTime(){
	struct timeval t;
   	gettimeofday(&t, (struct timezone *)NULL);
   	return ((uint64_t)1000000)*(uint64_t)t.tv_sec + (uint64_t)t.tv_usec; 
}

static double usecToSec(uint64_t t){
	return t*1.0e-6;
}

// 启动定时器
void beginTimer(const enum TimerPtr ptr){
	timers[ptr].begin = getUsTime();
}

// 停止定时器
void endTimer(const enum TimerPtr ptr){
	timers[ptr].delta = getUsTime() - timers[ptr].begin;
	timers[ptr].global = timers[ptr].global + timers[ptr].delta;
}

// 获取定时器总时间
double getGlobalTime(const enum TimerPtr ptr){
	return usecToSec(timers[ptr].global);
}