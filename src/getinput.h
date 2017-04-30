// getinput.h
// 从文件中读取参数

#ifndef GETINPUT_H_
#define GETINPUT_H_

#include <stdio.h>
#include <string.h>

#define LINE_MAX_LENGTH 256 //每行最大字符数
#define NAME_MAX_LENGTH 50 //每个参数名最大字符数
#define VALUE_MAX_LENGTH 50 //每个参数值最大字符数


// 从输入文件中读取单独一个参数对应的值                                    
// 参数：文件路径; 参数名称; 存储空间                        
// 返回：0,未找到; 1,找到符合名称的值; 2,使用默认值                                     
int getInputValue(const char *file_path, const char *para_name, char *buff);

#endif
