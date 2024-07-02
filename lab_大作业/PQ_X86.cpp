#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <iomanip>
#include <Windows.h>
#include <immintrin.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

const int times = 1000;
typedef vector<float> Vector;
typedef vector<Vector> Codebook;

// 计算两个向量之间的欧氏距离
float ComputeDistance(const Vector &a, const Vector &b)
{
    float distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

// 计算两个向量之间的欧氏距离（SSE优化）
float ComputeDistanceSSE(const Vector &a, const Vector &b)
{
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    for (; i < a.size(); i += 4)
    {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }
    float result[4];
    _mm_storeu_ps(result, sum);

    float final_sum = result[0] + result[1] + result[2] + result[3];

    // 处理剩余的元素
    for (; i < a.size(); ++i)
    {
        final_sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return sqrt(final_sum);
}

// 计算两个向量之间的欧氏距离（AVX优化）
float ComputeDistanceAVX(const Vector &a, const Vector &b)
{
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < a.size(); i += 8)
    {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    float final_sum = result[0] + result[1] + result[2] + result[3] +
                      result[4] + result[5] + result[6] + result[7];

    // 处理剩余的元素
    for (; i < a.size(); ++i)
    {
        final_sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return sqrt(final_sum);
}

// 计算两个向量之间的欧氏距离（AVX512优化）
float ComputeDistanceAVX512(const Vector &a, const Vector &b)
{
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < a.size(); i += 16)
    {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }
    float result[16];
    _mm512_storeu_ps(result, sum);
    float distance = 0.0;
    for (int j = 0; j < 16; ++j)
    {
        distance += result[j];
    }
    // 处理剩余的元素
    for (; i < a.size(); ++i)
    {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

struct ThreadData
{
    const std::vector<float> *a;
    const std::vector<float> *b;
    size_t start;
    size_t end;
    float result;
};

// 线程函数，计算部分向量的距离
void *partial_distance(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    const std::vector<float> &a = *(data->a);
    const std::vector<float> &b = *(data->b);
    float distance = 0.0;
    for (size_t i = data->start; i < data->end; ++i)
    {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    data->result = distance;
    pthread_exit(nullptr);
    return nullptr;
}

// 计算两个向量之间的欧氏距离（pthread优化）
float ComputeDistancePTH(const Vector &a, const Vector &b)
{
    const int num_threads = 4; // 假设我们使用4个线程
    pthread_t threads[num_threads];
    ThreadData data[num_threads];
    size_t length = a.size() / num_threads;
    float total_distance = 0.0;

    // 创建线程计算各部分的距离
    for (int i = 0; i < num_threads; ++i)
    {
        data[i] = {&a, &b, i * length, (i + 1) * length};
        if (i == num_threads - 1)
            data[i].end = a.size(); // 确保最后一个线程处理所有剩余的元素
        pthread_create(&threads[i], nullptr, partial_distance, (void *)&data[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(threads[i], nullptr);
        total_distance += data[i].result;
    }

    return sqrt(total_distance);
}

// 计算两个向量之间的欧氏距离（OpenMP优化）
float ComputeDistanceOMP(const Vector &a, const Vector &b)
{
    float distance = 0.0;

#pragma omp parallel for reduction(+ : distance)
    for (size_t i = 0; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

// 计算两个向量之间的欧氏距离（MPI优化）
float ComputeDistanceMPI(const std::vector<float> &a, const std::vector<float> &b)
{

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = a.size();
    int local_n = n / size;
    int remainder = n % size;
    int start = rank * local_n + std::min(rank, remainder);
    int end = start + local_n + (rank < remainder ? 1 : 0);

    float partialDistance = 0.0;
    for (int i = start; i < end; ++i)
    {
        partialDistance += (a[i] - b[i]) * (a[i] - b[i]);
    }

    float totalDistance;
    MPI_Reduce(&partialDistance, &totalDistance, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        totalDistance = sqrt(totalDistance);
    }

    return (rank == 0) ? totalDistance : 0.0;
}

// K-Means聚类算法
Codebook KMeans(const vector<Vector> &data, int k, int iterations = 100, int switch_opt = 0)
{
    int dim = data[0].size();
    Codebook centroids(k, Vector(dim, 0.0));
    vector<int> labels(data.size(), 0);

    // 初始化中心点
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, data.size() - 1);
    for (int i = 0; i < k; ++i)
    {
        centroids[i] = data[dis(gen)];
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        // 分配每个点到最近的中心点
        for (size_t i = 0; i < data.size(); ++i)
        {
            float minDist = numeric_limits<float>::max();
            for (int j = 0; j < k; ++j)
            {
                float dist = 0.0;
                if (switch_opt == 0)
                {
                    dist += ComputeDistance(data[i], centroids[j]);
                }
                else if (switch_opt == 1)
                {
                    dist += ComputeDistanceSSE(data[i], centroids[j]);
                }
                else if (switch_opt == 2)
                {
                    dist += ComputeDistanceAVX(data[i], centroids[j]);
                }
                else if (switch_opt == 3)
                {
                    distance += ComputeDistanceAVX512(data[i], centroids[j]);
                }
                else if (switch_opt == 4)
                {
                    dist += ComputeDistancePTH(data[i], centroids[j]);
                }
                else if (switch_opt == 5)
                {
                    dist += ComputeDistanceOMP(data[i], centroids[j]);
                }
                else if (switch_opt == 6)
                {
                    dist += ComputeDistanceMPI(data[i], centroids[j]);
                }
                if (dist < minDist)
                {
                    minDist = dist;
                    labels[i] = j;
                }
            }
        }

        // 重新计算中心点
        vector<Vector> newCentroids(k, Vector(dim, 0.0));
        vector<int> counts(k, 0);
        for (size_t i = 0; i < data.size(); ++i)
        {
            for (int d = 0; d < dim; ++d)
            {
                newCentroids[labels[i]][d] += data[i][d];
            }
            counts[labels[i]]++;
        }
        for (int j = 0; j < k; ++j)
        {
            if (counts[j] > 0)
            {
                for (int d = 0; d < dim; ++d)
                {
                    newCentroids[j][d] /= counts[j];
                }
            }
        }
        centroids = newCentroids;
    }

    return centroids;
}

// 对向量进行量化
int Quantize(const Vector &vec, const Codebook &codebook, int switch_opt)
{
    int bestIndex = 0;
    float bestDistance = ComputeDistance(vec, codebook[0]);
    for (size_t i = 1; i < codebook.size(); ++i)
    {
        float distance = 0.0;
        if (switch_opt == 0)
        {
            distance += ComputeDistance(vec, codebook[i]);
        }
        else if (switch_opt == 1)
        {
            distance += ComputeDistanceSSE(vec, codebook[i]);
        }
        else if (switch_opt == 2)
        {
            distance += ComputeDistanceAVX(vec, codebook[i]);
        }
        else if (switch_opt == 3)
        {
            distance += ComputeDistanceAVX512(vec, codebook[i]);
        }
        else if (switch_opt == 4)
        {
            distance += ComputeDistancePTH(vec, codebook[i]);
        }
        else if (switch_opt == 5)
        {
            distance += ComputeDistanceOMP(vec, codebook[i]);
        }
        else if (switch_opt == 6)
        {
            distance += ComputeDistanceMPI(vec, codebook[i]);
        }
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}

// PQ编码
vector<int> PQEncoding(const vector<Vector> &data, const vector<Codebook> &codebooks, int switch_opt)
{
    vector<int> codes;
    for (const auto &vec : data)
    {
        for (size_t i = 0; i < codebooks.size(); ++i)
        {
            Vector subvec(vec.begin() + i * vec.size() / codebooks.size(),
                          vec.begin() + (i + 1) * vec.size() / codebooks.size());
            codes.push_back(Quantize(subvec, codebooks[i], switch_opt));
        }
    }
    return codes;
}

// PQ解码
vector<Vector> PQDecoding(const vector<int> &codes, const vector<Codebook> &codebooks, int originalDim)
{
    vector<Vector> decodedVectors(codes.size() / codebooks.size(), Vector(originalDim, 0.0));
    for (size_t i = 0; i < decodedVectors.size(); ++i)
    {
        for (size_t j = 0; j < codebooks.size(); ++j)
        {
            int codeIndex = i * codebooks.size() + j;
            const Vector &subvec = codebooks[j][codes[codeIndex]];
            for (size_t k = 0; k < subvec.size(); ++k)
            {
                decodedVectors[i][j * subvec.size() + k] = subvec[k];
            }
        }
    }
    return decodedVectors;
}

// 近邻查询（ADC）
int PQNearestNeighbor(const Vector &query, const vector<int> &codes, const vector<Codebook> &codebooks, const vector<Vector> &data, int switch_opt)
{
    vector<int> queryCodes;
    for (size_t i = 0; i < codebooks.size(); ++i)
    {
        Vector subvec(query.begin() + i * query.size() / codebooks.size(),
                      query.begin() + (i + 1) * query.size() / codebooks.size());
        queryCodes.push_back(Quantize(subvec, codebooks[i], switch_opt));
    }

    int bestIndex = 0;
    float bestDistance = numeric_limits<float>::max();
    for (size_t i = 0; i < data.size(); ++i)
    {
        float distance = 0.0;
        for (size_t j = 0; j < codebooks.size(); ++j)
        {
            int codeIndex = i * codebooks.size() + j;
            if (switch_opt == 0)
            {
                distance += ComputeDistance(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 1)
            {
                distance += ComputeDistanceSSE(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 2)
            {
                distance += ComputeDistanceAVX(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 3)
            {
                distance += ComputeDistanceAVX512(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 4)
            {
                distance += ComputeDistancePTH(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 5)
            {
                distance += ComputeDistanceOMP(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
            }
            else if (switch_opt == 6)
            {
                distance += ComputeDistanceMPI(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]])
            }
        }
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}

int main()
{
    // 计时相关
    long long begin, end, freq;
    double timeuse1 = 0, timeuse2 = 0, timeuse3 = 0, timeuse4 = 0, timeuse5 = 0, timeuse6 = 0, timeuse7 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 生成示例数据集（1024个数据点，每个数据点有32维）
    int datasize = 1024, dim = 32, groupnum = 8, K = 32;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 10);
    vector<Vector> data(datasize, Vector(dim));
    Vector query;
    vector<Vector> query_array;
    int nearestIndex;

    // 生成测试集
    for (auto &vec : data)
    {
        for (auto &val : vec)
        {
            val = dis(gen);
        }
    }

    // 生成所有查询向量
    for (int i = 0; i < times; i++)
    {
        query.clear();
        // 生成一个查询向量
        for (int j = 0; j < dim; ++j)
        {
            query.push_back(rand() % 10);
        }
        query_array.push_back(query);
    }

    // 开始计时（串行）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 0)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 0);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 0);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse1 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（SSE优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 1)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 1);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 1);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse2 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（AVX优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 2)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 2);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 2);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse3 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（AVX512优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 3)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 3);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 3);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse4 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（pthread优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 4)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 4);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 4);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse5 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（openmp优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 5)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 5);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 5);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse6 += (end - begin) * 1000.0 / freq;
    }

    // 开始计时（mpi优化）
    QueryPerformanceCounter((LARGE_INTEGER *)&begin);

    MPI_Init(NULL, NULL);

    for (int i = 0; i < times; ++i)
    {
        // 将数据分成4个子空间
        int subspaceDim = data[0].size() / groupnum;
        vector<vector<Vector>> subspaces(groupnum);
        for (const auto &vec : data)
        {
            for (int i = 0; i < groupnum; ++i)
            {
                subspaces[i].emplace_back(vec.begin() + i * subspaceDim, vec.begin() + (i + 1) * subspaceDim);
            }
        }

        // 为每个子空间生成码本
        vector<Codebook> codebooks;
        for (const auto &subspace : subspaces)
        {
            codebooks.push_back(KMeans(subspace, K, 6)); // 生成K个聚类中心
        }

        // 编码
        vector<int> codes = PQEncoding(data, codebooks, 6);

        // 获得查询
        query = query_array[i];

        // 查找最近邻
        nearestIndex = PQNearestNeighbor(query, codes, codebooks, data, 6);

        // 结束计时
        QueryPerformanceCounter((LARGE_INTEGER *)&end);
        timeuse7 += (end - begin) * 1000.0 / freq;
    }

    MPI_Finalize();

    cout << "Non-SSE optimized time used:" << timeuse1 / times << "ms" << endl;

    cout << "SSE optimized time used:" << timeuse2 / times << "ms" << endl;

    cout << "AVX optimized time used:" << timeuse3 / times << "ms" << endl;

    cout << "AVX512 optimized time used:" << timeuse4 / times << "ms" << endl;

    cout << "pthread optimized time used:" << timeuse5 / times << "ms" << endl;

    cout << "OMP optimized time used:" << timeuse6 / times << "ms" << endl;

    cout << "MPI optimized time used:" << timeuse7 / times << "ms" << endl;

    return 0;
}
