#include <iostream>
#include <Windows.h>
#include <omp.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX
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

void Openmp_SSE()
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
        // 消去部分，使用行划分
#pragma omp for
        for (i = k + 1; i < N; i++)
        {
            // SSE
            float temp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            __m128 tmp_ik = _mm_loadu_ps(temp);
            int num = k + 1;
            for (j = k + 1; j + 4 <= N; j += 4, num = j)
            {
                __m128 tmp_ij = _mm_loadu_ps(m[i] + j);
                __m128 tmp_kj = _mm_loadu_ps(m[k] + j);
                tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
                _mm_storeu_ps(m[i] + j, tmp_ij);
            }
            for (j = num; j < N; j++)
            {
                m[i][j] -= (m[i][k] * m[k][j]);
            }
            m[i][k] = 0;
        }
    }
}

void Openmp_AVX()
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
        // 消去部分，使用行划分
#pragma omp for
        for (i = k + 1; i < N; i++)
        {
            // AVX
            float temp[8] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            __m256 tmp_ik = _mm256_loadu_ps(temp);
            int num = k + 1;
            for (j = k + 1; j + 8 <= N; j += 8, num = j)
            {
                __m256 tmp_ij = _mm256_loadu_ps(m[i] + j);
                __m256 tmp_kj = _mm256_loadu_ps(m[k] + j);
                tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
                _mm256_storeu_ps(m[i] + j, tmp_ij);
            }
            for (j = num; j < N; j++)
            {
                m[i][j] -= (m[i][k] * m[k][j]);
            }
            m[i][k] = 0;
        }
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        Openmp_SSE();
        // print();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " OpenMP_SSE:  " << timeuse1 / times << "ms" << endl;
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        Openmp_AVX();
        // print();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " OPenMP_AVX:  " << timeuse2 / times << "ms" << endl;
    return 0;
}