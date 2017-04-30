// 从输入文件中读取参数配置信息                                           
// 格式：                                                         
//              每行一个参数值                                          
//              # 开头为注释行                               
//              如果一行中不包含 = 号，则视为注释行                  
//              参数名字与参数值用 = 号连接                                           
//               = 号前后不能有空格                                       
//              每行最大字符数为 LINE_MAX_LENGTH                          
//              参数名最大字符数为 NAME_MAX_LENGTH,超长将丢弃该参数       
//              参数值最大字符数为 VALUE_MAX_LENGTH,超长将截断  
//              当参数值为'default'时，使用默认参数值       
/**************************************************************************/
#include "getinput.h"

// 从输入文件中读取单独一个参数对应的值                                    
// 参数：文件路径; 参数名称; 存储空间                        
// 返回：0,未找到; 1,找到符合名称的值; 2,使用默认值
int getInputValue(const char* file_path, const char* para_name,
    char* buff)
{
    char linebuf[LINE_MAX_LENGTH];
    char line_name[NAME_MAX_LENGTH];
    char* sign = "=";
    char* leave_line;
    FILE* f;
    int flag = 0, len;

    f = fopen(file_path, "r");
    if (f == NULL) {
        fprintf(stdout,"Open InputFile Failed:%s\n", file_path);
        return 0;
    }

    fseek(f, 0, SEEK_SET);
    while (fgets(linebuf, LINE_MAX_LENGTH, f) != NULL) {
        if (strlen(linebuf) < 4) { //去除空行 
            continue;
        }

        if (linebuf[0] == '#') { //去除注释行 "#"
            continue;
        }

        if (linebuf[strlen(linebuf) - 1] == 10 || linebuf[strlen(linebuf) - 1] == 13 ) {
            linebuf[strlen(linebuf) - 1] = '\0';
        }

        memset(line_name, 0, sizeof(line_name));
        leave_line = strstr(linebuf, sign);
        if (leave_line == NULL) { //去除无"="的情况
            continue;
        }

        int leave_num = leave_line - linebuf;
        if (leave_num > NAME_MAX_LENGTH) {
            continue;
        }

        strncpy(line_name, linebuf, leave_num);
        if (strcmp(line_name, para_name) == 0) {
            len = strlen(linebuf) - leave_num - 1;
            len = len > VALUE_MAX_LENGTH ? VALUE_MAX_LENGTH : len;
            strncpy(buff, linebuf + (leave_num + 1),
                len);
            *(buff + len) = '\0';
            if(strcmp(buff,"default") == 0){
                flag = 2;
                break;
            }
            flag = 1;
            break;
        }

        if (fgetc(f) == EOF){
            break;
        }

        fseek(f, -1, SEEK_CUR);
        memset(linebuf, 0, sizeof(linebuf));
    }

    fclose(f);
    return flag;
}
