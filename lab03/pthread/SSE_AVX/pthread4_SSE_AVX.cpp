/*
 * File: SSE_AVX/pthread4_SSE_AVX.cpp
 * Author: Ethan
 * Date: 2024-5-24
 * Description: pthread 静态线程 + barrier 同步 + SSE + AVX
 *
 * Usage: g++ -g -march=native -o pthread4_SSE_AVX .\pthread4_SSE_AVX.cpp
 *
 * Note: 该版本为 pthread + SSE/AVX 普通高斯消去
 */

#include <iostream>
#include <pthread.h>
#include <Windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;
const int N = 512;
float m[N][N];
const int thread_num = 4;
const int times = 10;

// 线程参数结构体
typedef struct
{
    int t_id;
} threadParam;

// barrier
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

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

void *threadFunc_SSE(void *param)
{
    threadParam *p = (threadParam *)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        // 0线程进行除法操作
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
        }
        // 第一个同步点
        pthread_barrier_wait(&barrier_Divsion);
        // 所有线程进行消去操作
        for (int i = k + 1 + t_id; i < N; i += thread_num)
        {
            float tmp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            __m128 tmp_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, num = j)
            {
                __m128 tmp_ij = _mm_loadu_ps(m[i] + j);
                __m128 tmp_kj = _mm_loadu_ps(m[k] + j);
                tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
                _mm_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = num; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return 0;
}

void threadMain_SSE()
{
    // 初始化barrier
    pthread_barrier_init(&barrier_Divsion, NULL, thread_num); // thread_num个线程到达后才会执行
    pthread_barrier_init(&barrier_Elimination, NULL, thread_num);
    // 创建线程
    pthread_t handles[thread_num];
    threadParam param[thread_num];
    for (int t_id = 0; t_id < thread_num; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_SSE, (void *)&param[t_id]);
    }
    // 等待回收线程
    for (int t_id = 0; t_id < thread_num; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁barrier
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}

void *threadFunc_AVX(void *param)
{
    threadParam *p = (threadParam *)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        // 0线程进行除法操作
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
        }
        // 第一个同步点
        pthread_barrier_wait(&barrier_Divsion);
        // 所有线程进行消去操作
        for (int i = k + 1 + t_id; i < N; i += thread_num)
        {
            float tmp[8] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            __m256 tmp_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            for (int j = k + 1; j + 8 <= N; j += 8, num = j)
            {
                __m256 tmp_ij = _mm256_loadu_ps(m[i] + j);
                __m256 tmp_kj = _mm256_loadu_ps(m[k] + j);
                tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
                _mm256_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = num; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return 0;
}

void threadMain_AVX()
{
    // 初始化barrier
    pthread_barrier_init(&barrier_Divsion, NULL, thread_num); // thread_num个线程到达后才会执行
    pthread_barrier_init(&barrier_Elimination, NULL, thread_num);
    // 创建线程
    pthread_t handles[thread_num];
    threadParam param[thread_num];
    for (int t_id = 0; t_id < thread_num; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_AVX, (void *)&param[t_id]);
    }
    // 等待回收线程
    for (int t_id = 0; t_id < thread_num; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁barrier
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    // SSE
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        threadMain_SSE();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " pthread4_SSE:  " << timeuse1 / times << "ms" << endl;
    // AVX
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        threadMain_AVX();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " pthread4_AVX:  " << timeuse2 / times << "ms" << endl;
    return 0;
}