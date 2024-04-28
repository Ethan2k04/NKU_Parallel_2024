#include <iostream>
#include <stdlib.h>
#include <Windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX

using namespace std;

const int N = 64;
float m[N][N], x[N], b[N];
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

void b_reset()
{
    for (int i = 0; i < N; i++)
    {
        x[i] = rand() % 10;
    }

    for (int i = 0; i < N; i++)
    {
        b[i] = 0;
        for (int j = 0; j < N; j++)
        {
            b[i] += m[i][j] * x[j];
        }
    }

    for (int i = 0; i < N; i++)
    {
        x[i] = 0;
    }
}

// 普通高斯消元串行算法
void serial_LU()
{
    // 消去过程
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

/*
// 普通高斯消元串行算法
void serial_LU()
{
    float factor, sum;
    // 消去过程
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            factor = m[i][k] / m[k][k];

            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= factor * m[k][j];
            }
            b[i] = b[i] - factor * b[k];
        }
    }

    // 回代过程
    x[N - 1] = b[N - 1] / m[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--)
    {
        sum = b[i];
        for (int j = i + 1; j < N; j++)
        {
            x[i] = sum / m[i][i];
        }
    }
}
*/

// 对第一部分进行SSE并行化
void parallel_SSE_Part1()
{
    __m128 tmp_kk, tmp_kj;
    for (int k = 0; k < N; k++)
    {
        float tmp[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        tmp_kk = _mm_loadu_ps(tmp);
        int num = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, num = j)
        {
            tmp_kj = _mm_loadu_ps(m[k] + j);
            tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
            _mm_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = num; j < N; j++)
        {
            m[k][j] /= m[k][k];
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

// 对第二部分进行SSE并行化
void parallel_SSE_Part2()
{
    __m128 tmp_ik, tmp_ij, tmp_kj;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] /= m[k][k]; // 将第k行对角线元素变为1
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            tmp_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, num = j)
            {
                tmp_ij = _mm_loadu_ps(m[i] + j);
                tmp_kj = _mm_loadu_ps(m[k] + j);
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
    }
}

// 两个部分进行SSE并行化
void parallel_SSE()
{
    __m128 tmp_kk, tmp_kj, tmp_ik, tmp_ij;
    for (int k = 0; k < N; k++)
    {
        float tmp1[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        tmp_kk = _mm_loadu_ps(tmp1);
        int num1 = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, num1 = j)
        {
            tmp_kj = _mm_loadu_ps(m[k] + j);
            tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
            _mm_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = num1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp2[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            tmp_ik = _mm_loadu_ps(tmp2);
            int num2 = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, num2 = j)
            {
                tmp_ij = _mm_loadu_ps(m[i] + j);
                tmp_kj = _mm_loadu_ps(m[k] + j);
                tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
                _mm_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = num2; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }
}

// 对第一部分进行AVX并行化
void parallel_AVX_Part1()
{
    __m256 tmp_kk, tmp_kj;
    for (int k = 0; k < N; k++)
    {
        float tmp[8] = {m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]};
        tmp_kk = _mm256_loadu_ps(tmp);
        int num = k + 1;
        for (int j = k + 1; j + 8 <= N; j += 8, num = j)
        {
            tmp_kj = _mm256_loadu_ps(m[k] + j);
            tmp_kj = _mm256_div_ps(tmp_kj, tmp_kk);
            _mm256_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = num; j < N; j++)
        {
            m[k][j] /= m[k][k];
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

// 对第二部分进行AVX并行化
void parallel_AVX_Part2()
{
    __m256 tmp_ik, tmp_ij, tmp_kj;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] /= m[k][k]; // 将第k行对角线元素变为1
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp[8] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            tmp_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            for (int j = k + 1; j + 8 <= N; j += 8, num = j)
            {
                tmp_ij = _mm256_loadu_ps(m[i] + j);
                tmp_kj = _mm256_loadu_ps(m[k] + j);
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
    }
}

// 对AVX两个部分进行并行化
void parallel_AVX()
{
    __m256 tmp_kk, tmp_kj, tmp_ik, tmp_ij;
    for (int k = 0; k < N; k++)
    {
        float tmp1[8] = {m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]};
        tmp_kk = _mm256_loadu_ps(tmp1);
        int num1 = k + 1;
        for (int j = k + 1; j + 8 <= N; j += 8, num1 = j)
        {
            tmp_kj = _mm256_loadu_ps(m[k] + j);
            tmp_kj = _mm256_div_ps(tmp_kj, tmp_kk);
            _mm256_storeu_ps(m[k] + j, tmp_kj);
        }
        for (int j = num1; j < N; j++)
        {
            m[k][j] /= m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float tmp2[8] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            tmp_ik = _mm256_loadu_ps(tmp2);
            int num2 = k + 1;
            for (int j = k + 1; j + 8 <= N; j += 8, num2 = j)
            {
                tmp_ij = _mm256_loadu_ps(m[i] + j);
                tmp_kj = _mm256_loadu_ps(m[k] + j);
                tmp_kj = _mm256_mul_ps(tmp_kj, tmp_ik);
                tmp_ij = _mm256_sub_ps(tmp_ij, tmp_kj);
                _mm256_storeu_ps(m[i] + j, tmp_ij);
            }
            for (int j = num2; j < N; j++)
            {
                m[i][j] -= m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }
}

int main()
{
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0, timeuse5 = 0, timeuse6 = 0, timeuse7 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 对串行算法进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        serial_LU();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << " serial_LU:  " << timeuse1 / times << "ms" << endl;
    // 对第一部分SSE并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_SSE_Part1();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  SSE_Part1:  " << timeuse2 / times << "ms" << endl;
    // 对第二部分SSE并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_SSE_Part2();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse3 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  SSE_Part2:  " << timeuse3 / times << "ms" << endl;
    // 对SSE两个部分并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_SSE();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse4 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  SSE_All_Part:  " << timeuse4 / times << "ms" << endl;
    // 对第一部分AVX并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_AVX_Part1();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse5 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  AVX_Part1:  " << timeuse5 / times << "ms" << endl;
    // 对第二部分AVX并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_AVX_Part2();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse6 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  AVX_Part2:  " << timeuse6 / times << "ms" << endl;
    // 对AVX两个部分并行化进行时间测试
    for (int i = 0; i < times; i++)
    {
        m_reset();
        QueryPerformanceCounter((LARGE_INTEGER *)&begin);
        parallel_AVX();
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse7 += (end - begin) * 1000.0 / freq;
    }
    cout << "n=" << N << "  AVX_All_Part:  " << timeuse7 / times << "ms" << endl;

    cin.get();
    return 0;
}