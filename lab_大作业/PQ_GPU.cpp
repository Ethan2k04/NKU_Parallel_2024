#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <iomanip>
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

const int times = 1000;
size_t dim = 32;
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

__global__ void DistanceKernel(const float *a, const float *b, float *result, size_t size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        float diff = a[index] - b[index];
        result[index] = diff * diff;
    }
}

float ComputeDistanceGPU(const float *a, const float *b)
{
    size_t size = dim;
    float *dev_a, *dev_b, *dev_partial_result;
    float *partial_result = (float *)malloc(size * sizeof(float));
    float distance = 0.0;

    // Allocate GPU memory
    cudaMalloc((void **)&dev_a, size * sizeof(float));
    cudaMalloc((void **)&dev_b, size * sizeof(float));
    cudaMalloc((void **)&dev_partial_result, size * sizeof(float));

    // Copy vectors to device
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Setup block and grid dimensions
    int blockSize = 256; // You can tune this parameter to optimize
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    DistanceKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_partial_result, size);

    // Copy results back to host
    cudaMemcpy(partial_result, dev_partial_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up results
    for (size_t i = 0; i < size; ++i)
    {
        distance += partial_result[i];
    }

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_result);
    free(partial_result);

    // Return the square root of the summed distances
    return sqrt(distance);
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
                    dist += ComputeDistanceGPU(data[i], centroids[j]);
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
            distance += ComputeDistanceGPU(vec, codebook[i]);
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
                distance += ComputeDistanceGPU(codebooks[j][queryCodes[j]], codebooks[j][codes[codeIndex]]);
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
    double timeuse1 = 0, timeuse2 = 0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 生成示例数据集（1024个数据点）
    int datasize = 1024, groupnum = 8, K = 32;
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

    // 开始计时（GPU优化）
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

    cout << "Non-Parrel optimized time used:" << timeuse1 / times << "ms" << endl;

    cout << "GPU optimized time used:" << timeuse2 / times << "ms" << endl;

    return 0;
}
