#include <iostream>
#include <stdlib.h>
#include <Windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX

using namespace std;

const int N = 256;
float m[N][N];
const int times = 10;

// 生成测试用例（静态分配）
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

// 生成测试用例（动态分配，对齐）
float **m_reset_dynamic(float **m)
{
	const int alignment = 16;
	const int size = N;
	m = (float **)_aligned_malloc(size * size * sizeof(float), alignment);
	for (int i = 0; i < size; i++)
	{
		m[i] = (float *)_aligned_malloc(size * sizeof(float), alignment);
		for (int j = 0; j < i; j++)
		{
			m[i][j] = 0;
		}
		m[i][i] = 1.0;
		for (int j = i + 1; j < size; j++)
		{
			m[i][j] = rand();
		}
	}
	for (int k = 0; k < size; k++)
	{
		for (int i = k + 1; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				m[i][j] += m[k][j];
			}
		}
	}

	return m;
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

// 对SSE两个部分进行并行化(不对齐)
void parallel_SSE_Unalign()
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

// 对SSE两个部分进行并行化(对齐，静态)
void parallel_SSE_Align()
{
	for (int k = 0; k < N; k++)
	{
		// 1.处理不对齐部分
		int pre1 = 4 - (k + 1) % 4;
		for (int j = k + 1; j < k + 1 + pre1; j++)
		{
			m[k][j] /= m[k][k];
		}
		// 2.处理对齐部分
		float tmp1[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
		__m128 tmp_kk = _mm_load_ps(tmp1);
		int rem1 = pre1 + k + 1;
		for (int j = pre1 + k + 1; j + 4 <= N; j += 4, rem1 = j)
		{
			__m128 tmp_kj = _mm_load_ps(m[k] + j);
			tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
			_mm_store_ps(m[k] + j, tmp_kj);
		}
		// 3.处理剩余部分
		for (int j = rem1; j < N; j++)
		{
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			// 1.处理不对齐部分
			int pre2 = 4 - (k + 1) % 4;
			for (int j = k + 1; j < k + 1 + pre2; j++)
			{
				m[i][j] -= m[k][j] * m[i][k];
			}
			// 2.处理对齐部分
			float tmp2[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
			__m128 tmp_ik = _mm_load_ps(tmp2);
			int rem2 = pre2 + k + 1;
			for (int j = pre2 + k + 1; j + 4 <= N; j += 4, rem2 = j)
			{
				__m128 tmp_ij = _mm_load_ps(m[i] + j);
				__m128 tmp_kj = _mm_load_ps(m[k] + j);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_store_ps(m[i] + j, tmp_ij);
			}
			// 3.处理剩余部分
			for (int j = rem2; j < N; j++)
			{
				m[i][j] -= m[k][j] * m[i][k];
			}
			m[i][k] = 0;
		}
	}
}

// 对SSE两个部分进行并行化(对齐，动态)
void parallel_SSE_Align_Dynamic(float **m)
{
	for (int k = 0; k < N; k++)
	{
		float tmp1[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
		__m128 tmp_kk = _mm_load_ps(tmp1);
		int pre = k + 1;
		for (int j = k + 1; j + 4 <= N; j += 4, pre = j)
		{
			__m128 tmp_kj = _mm_load_ps(m[k] + j - k - 1);
			tmp_kj = _mm_div_ps(tmp_kj, tmp_kk);
			_mm_store_ps(m[k] + j - k - 1, tmp_kj);
		}
		for (int j = pre; j < N; j++)
		{
			m[k][j] /= m[k][k];
		}
		m[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			float tmp2[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
			__m128 tmp_ik = _mm_load_ps(tmp2);
			int rem = k + 1;
			for (int j = k + 1; j + 4 <= N; j += 4, rem = j)
			{
				__m128 tmp_ij = _mm_load_ps(m[i] + j - k - 1);
				__m128 tmp_kj = _mm_load_ps(m[k] + j - k - 1);
				tmp_kj = _mm_mul_ps(tmp_kj, tmp_ik);
				tmp_ij = _mm_sub_ps(tmp_ij, tmp_kj);
				_mm_store_ps(m[i] + j - k - 1, tmp_ij);
			}
			for (int j = rem; j < N; j++)
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
	double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0, timeuse5 = 0, timeuse6 = 0;
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
	// 对SSE两个部分（不对齐）并行化进行时间测试
	for (int i = 0; i < times; i++)
	{
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER *)&begin);
		parallel_SSE_Unalign();
		QueryPerformanceCounter((LARGE_INTEGER *)&end);
		timeuse2 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  SSE_All_Part_Unalign:  " << timeuse2 / times << "ms" << endl;

	// 对SSE两个部分（对齐，静态）并行化进行时间测试
	for (int i = 0; i < times; i++)
	{
		m_reset();
		QueryPerformanceCounter((LARGE_INTEGER *)&begin);
		parallel_SSE_Align();
		QueryPerformanceCounter((LARGE_INTEGER *)&end);
		timeuse3 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  SSE_All_Part_Align:  " << timeuse3 / times << "ms" << endl;

	// 对SSE两个部分（对齐，动态）并行化进行时间测试
	float **m_;
	for (int i = 0; i < times; i++)
	{
		m_ = m_reset_dynamic(m_);
		QueryPerformanceCounter((LARGE_INTEGER *)&begin);
		parallel_SSE_Align_Dynamic(m_);
		QueryPerformanceCounter((LARGE_INTEGER *)&end);
		timeuse4 += (end - begin) * 1000.0 / freq;
	}
	cout << "n=" << N << "  SSE_All_Part_Align_Dynamic:  " << timeuse4 / times << "ms" << endl;

	cin.get();

	return 0;
}