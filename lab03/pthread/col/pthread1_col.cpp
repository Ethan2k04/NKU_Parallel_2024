/*
 * File: col/pthread1_col.cpp
 * Author: Ethan
 * Date: 2024-5-25
 * Description: pthread 动态线程
 *
 * Usage: g++ -o pthread1_col .\pthread1_col.cpp -pthread
 *
 * Note: 该版本为pthread普通高斯消去 + 按列划分任务
 */

#include <iostream>
#include <pthread.h>
#include <Windows.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
int thread_num = 4;
const int times = 10;

// 线程参数结构体
typedef struct
{
    int k;    // 消去的轮次
    int t_id; // 线程id
} threadParam;

// 生成测试用例
void print();
void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            m[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                m[i][j] += m[k][j];
            }
        }
    }
}

// 输出数组
void print()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
}

void *threadFunc_Col(void *param)
{
    threadParam *p = (threadParam *)param;
    int k = p->k;
    int t_id = p->t_id;
    // 获取自己的计算任务
    for (int i = k + 1; i < N; i++)
    {
        for (int j = k + 1 + t_id; j < N; j += thread_num)
        {
            m[i][j] -= (m[i][k] * m[k][j]);
        }
    }
    pthread_exit(NULL);
    return 0;
}

void threadMain_Col()
{
    for (int k = 0; k < N; k++)
    {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        // 创建子线程，进行消去操作
        pthread_t *handles = new pthread_t[thread_num];   // 创建对应句柄
        threadParam *param = new threadParam[thread_num]; // 创建对应参数
        // 分配任务
        for (int t_id = 0; t_id < thread_num; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // 创建线程
        for (int t_id = 0; t_id < thread_num; t_id++)
        {
            pthread_create(&handles[t_id], NULL, threadFunc_Col, (void *)&param[t_id]);
        }
        // 主线程等待回收所有子线程
        for (int t_id = 0; t_id < thread_num; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        // 赋值m[i][k]为0
        for (int i = k + 1; i < N; i++)
        {
            m[i][k] = 0;
        }
        // 释放分配的空间
        delete[] handles;
        delete[] param;
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        threadMain_Col();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " pthread1_Col:  " << timeuse / times << "ms" << endl;
    return 0;
}