#include <iostream>
#include <Windows.h>
#include <omp.h>
using namespace std;
const int N = 512;
float m[N][N];
const int NUM_THREADS = 4;
const int times = 10;

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
    // print();
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

// 普通高斯消元算法
void serial_LU()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }
}

void Openmp_Col()
{
    int i, j, k;
    float tmp;
    // 创建线程
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, tmp)
    for (k = 0; k < N; k++)
    {
        // 除法部分，一个线程执行
#pragma omp single
        {
            tmp = m[k][k];
            for (j = k + 1; j < N; j++)
            {
                m[k][j] /= tmp;
            }
            m[k][k] = 1.0;
        }
        // 消去部分，使用列划分
#pragma omp for
        for (j = k + 1; j < N; j++)
        {
            tmp = m[k][j];
            for (i = k + 1; i < N; i++)
            {
                m[i][j] -= (m[i][k] * tmp);
            }
        }
        // 将m[i][k]置为0
#pragma omp single
        {
            for (i = k + 1; i < N; i++)
            {
                m[i][k] = 0;
            }
        }
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    // 普通高斯消元计时
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        serial_LU();
        // print();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " Serial:  " << timeuse1 / times << "ms" << endl;
    // OpenMP按行划分计时
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        Openmp_Col();
        // print();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " OPenMP_Col:  " << timeuse2 / times << "ms" << endl;
    return 0;
}