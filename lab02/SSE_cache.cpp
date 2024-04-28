#include <iostream>
#include <stdlib.h>
#include <Windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX

using namespace std;

const int N = 256;
float m[N][N];
float m_i[N][N];
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

void serial_LU_cache()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m_i[j][i] = m[i][j];
            m[i][j] = 0;
        }
    }
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
                m[i][j] -= m[k][j] * m_i[k][i]; // 从第i行消去第k行
            }
            m[i][k] = 0;
        }
    }
}

void parallel_SSE()
{
    for (int k = 0; k < N; k++)
    {
        float tmp1[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        __m128 tmp_kk = _mm_loadu_ps(tmp1);
        int rem1 = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, rem1 = j)
        {
            __m128 tmp_kj = _mm_loadu_ps(m[k] + j);
            tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
            _mm_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = rem1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp2[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            __m128 tmp_ik = _mm_loadu_ps(tmp2);
            int rem2 = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, rem2 = j)
            {
                __m128 tmp_ij = _mm_loadu_ps(m[i] + j);
                __m128 tmp_kj = _mm_loadu_ps(m[k] + j);
                tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
                _mm_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = rem2; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }
}

void parallel_SSE_cache()
{
    __m128 tmp_kk, tmp_kj, tmp_ik, tmp_ij;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m_i[j][i] = m[i][j];
            m[i][j] = 0;
        }
    }

    for (int k = 0; k < N; k++)
    {
        float tmp[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        int rem1 = k + 1;
        for (int j = k + 1; j < N; j += 4, rem1 = j)
        {
            tmp_kj = _mm_loadu_ps(m[k] + j);
            tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
            _mm_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = rem1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp2[4] = {m_i[k][i], m_i[k][i], m_i[k][i], m_i[k][i]};
            tmp_ik = _mm_loadu_ps(tmp2);
            int rem2 = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, rem2 = j)
            {
                tmp_ij = _mm_loadu_ps(m[i] + j);
                tmp_kj = _mm_loadu_ps(m[k] + j);
                tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
                _mm_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = rem2; j < N; j++)
            {
                m[i][j] -= m[k][j] * m_i[k][i];
            }
            m[i][k] = 0;
        }
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    // 对未 cache 优化串行算法进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        serial_LU();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " serial_LU Uncache:  " << timeuse1 / times << "ms" << endl;
    // 对 cache 优化串行算法进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        serial_LU_cache();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " serial_LU Cache:  " << timeuse2 / times << "ms" << endl;
    // 对未 cache 优化并行算法进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_SSE();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse3 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " parallel Uncache:  " << timeuse3 / times << "ms" << endl;
    // 对 cache 优化并行算法进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_SSE_cache();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse4 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " parallel Cache:  " << timeuse4 / times << "ms" << endl;

    cin.get();

    return 0;
}