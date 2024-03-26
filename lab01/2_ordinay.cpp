#include <iostream>
#include <Windows.h>
#include <fstream>

using namespace std;

void writeArrayToCSV(double **arr, int rows, int cols, const std::string &filename)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // 写入二维数组数据到 CSV 文件
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outputFile << arr[i][j];
            if (j < cols - 1)
            {
                outputFile << ","; // 用逗号分隔每个元素
            }
        }
        outputFile << std::endl; // 换行表示下一行数据
    }

    outputFile.close();
    std::cout << "CSV written in: " << filename << std::endl;
}

void run(int n, int k, double **arr, int count)
{
    // 定义累加的数组
    int *a = new int[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
    // 定义累加结果
    int sum;
    long long head, tail, freq; // timers
    double *record = new double[k];
    // 获取计时器频率
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    // 开始计时
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    // 调用递归函数进行累加
    for (int count = 0; count < k; count++)
    {
        long long tick_h, tick_t;
        // 开始单次计时
        QueryPerformanceCounter((LARGE_INTEGER *)&tick_h);
        for (int count = 0; count < k; count++)
        {
            sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a[i];
            }
        }
        // 截取时间
        QueryPerformanceCounter((LARGE_INTEGER *)&tick_t);
        record[count] = (double)(tick_t - tick_h) * 1000.0 / freq;
    }
    // 结束计时
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    // 输出结果
    cout << "n: " << n << " "
         << "k: " << k << endl;
    cout << "Total time: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    cout << "Average time: " << (tail - head) * 1000.0 / (freq * k) << "ms" << endl;
    cout << "================================================================" << endl;
    arr[count] = new double[k];
    for (int i = 0; i < k; i++)
    {
        arr[count][i] = record[i];
    }
    // 释放内存
    delete[] a;
}

int main()
{
    cout << "***********Ordinary Way***************" << endl;
    double **arr = new double *[8];
    int n[8] = {16, 32, 64, 128, 256, 512, 1024, 2048}, k[8] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    for (int i = 0; i < 8; i++)
    {
        run(n[i], k[i], arr, i);
    }
    writeArrayToCSV(arr, 8, 1000, "2_ordinary.csv");
    return 0;
}