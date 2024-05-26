/*
 * File: ./LU.cpp
 * Author: Ethan
 * Date: 2024-5-24
 * Description: 串行算法（对照）
 *
 * Usage: g++ -o LU .\LU.cpp
 *
 * Note: 请保持问题规模与并行版本一致
 */

#include <iostream>
#include <stdlib.h>
#include <iostream>
#include <Windows.h>
using namespace std;
const int N = 128;
float m[N][N];
const int times = 10;

// 生成测试用例
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

// 普通高斯消元串行算法
void serial_LU()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] /= m[k][k]; // 将第k行对角线元素变为1
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k]; // 从第i行消去第k行
            }
            m[i][k] = 0;
        }
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        serial_LU();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " serial_LU:  " << timeuse1 / times << "ms" << endl;
    return 0;
}