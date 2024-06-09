#include <iostream>
#include <mpi.h>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <omp.h>

using namespace std;

const int MATRIX_SIZE = 2048;
float matrix[MATRIX_SIZE][MATRIX_SIZE];
int THREAD_COUNT = 8; // 线程数
const int ITERATIONS = 10;
double duration_MPI = 0, duration_MPI_NEON = 0, duration_MPI_OMP = 0, duration_MPI_NEON_OMP = 0;

// 输出数组
void print_matrix()
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// 生成测试用例
void reset_matrix()
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < i; j++)
        {
            matrix[i][j] = 0;
        }
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < MATRIX_SIZE; j++)
        {
            matrix[i][j] = rand();
        }
    }
    for (int k = 0; k < MATRIX_SIZE; k++)
    {
        for (int i = k + 1; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                matrix[i][j] += matrix[k][j];
            }
        }
    }
}

// 普通MPI
void MPI_process()
{
    double start_time, end_time;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int task_count; // 每个进程的任务量
    if (rank < MATRIX_SIZE % size)
    {
        task_count = MATRIX_SIZE / size + 1;
    }
    else
    {
        task_count = MATRIX_SIZE / size;
    }
    float *buffer = new float[task_count * MATRIX_SIZE];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < MATRIX_SIZE; i += size)
            {
                for (int j = 0; j < MATRIX_SIZE; j++)
                {
                    buffer[i / size * MATRIX_SIZE + j] = matrix[i][j];
                }
            }
            int recv_task_count = p < MATRIX_SIZE % size ? MATRIX_SIZE / size + 1 : MATRIX_SIZE / size;
            MPI_Send(buffer, recv_task_count * MATRIX_SIZE, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&matrix[rank][0], task_count * MATRIX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_count; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    for (int k = 0; k < MATRIX_SIZE; k++)
    {
        if (k % size == rank)
        {
            for (int j = k + 1; j < MATRIX_SIZE; j++)
            {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int i = 0; i < size; i++)
            {
                if (i != rank)
                {
                    MPI_Send(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            MPI_Recv(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        int begin = MATRIX_SIZE / size * size + rank < MATRIX_SIZE ? MATRIX_SIZE / size * size + rank : MATRIX_SIZE / size * size + rank - size;
        for (int i = begin; i > k; i -= size)
        {
            for (int j = k + 1; j < MATRIX_SIZE; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    if (rank == 0)
    {
        duration_MPI += (end_time - start_time) * 1000.0;
    }
    return;
}

// MPI和NEON结合
void MPI_NEON_process()
{
    double start_time, end_time;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();
    int task_count; // 每个进程的任务量
    if (rank < MATRIX_SIZE % size)
    {
        task_count = MATRIX_SIZE / size + 1;
    }
    else
    {
        task_count = MATRIX_SIZE / size;
    }
    float *buffer = new float[task_count * MATRIX_SIZE];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < MATRIX_SIZE; i += size)
            {
                for (int j = 0; j < MATRIX_SIZE; j++)
                {
                    buffer[i / size * MATRIX_SIZE + j] = matrix[i][j];
                }
            }
            int recv_task_count = p < MATRIX_SIZE % size ? MATRIX_SIZE / size + 1 : MATRIX_SIZE / size;
            MPI_Send(buffer, recv_task_count * MATRIX_SIZE, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&matrix[rank][0], task_count * MATRIX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_count; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    for (int k = 0; k < MATRIX_SIZE; k++)
    {
        if (k % size == rank)
        {
            float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
            float32x4_t vec_temp = vld1q_f32(temp);
            vrecpeq_f32(vec_temp);
            int index = k + 1;
            for (int j = k + 1; j + 4 <= MATRIX_SIZE; j += 4, index = j)
            {
                float32x4_t vec_a;
                vec_a = vld1q_f32(matrix[k] + j);
                vec_a = vmulq_f32(vec_a, vec_temp);
                vst1q_f32(matrix[k] + j, vec_a);
            }
            for (int j = index; j < MATRIX_SIZE; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
            for (int p = 0; p < size; p++)
            {
                if (p != rank)
                {
                    MPI_Send(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            MPI_Recv(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        int begin = k + 1;
        while (begin % size != rank)
        {
            begin++;
        }
        for (int i = begin; i < MATRIX_SIZE; i += size)
        {
            float32x4_t vaik;
            float temp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            vaik = vld1q_f32(temp);
            int index = k + 1;
            for (int j = k + 1; j + 4 <= MATRIX_SIZE; j += 4, index = j)
            {
                float32x4_t vakj = vld1q_f32(matrix[k] + j);
                float32x4_t vaij = vld1q_f32(matrix[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(matrix[i] + j, vaij);
            }
            for (int j = index; j < MATRIX_SIZE; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[k][j] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    if (rank == 0)
    {
        duration_MPI_NEON += (end_time - start_time) * 1000;
    }
}

// MPI和OMP结合
void MPI_OMP_process()
{
    double start_time, end_time;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int task_count; // 每个进程的任务量
    if (rank < MATRIX_SIZE % size)
    {
        task_count = MATRIX_SIZE / size + 1;
    }
    else
    {
        task_count = MATRIX_SIZE / size;
    }
    float *buffer = new float[task_count * MATRIX_SIZE];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < MATRIX_SIZE; i += size)
            {
                for (int j = 0; j < MATRIX_SIZE; j++)
                {
                    buffer[i / size * MATRIX_SIZE + j] = matrix[i][j];
                }
            }
            int recv_task_count = p < MATRIX_SIZE % size ? MATRIX_SIZE / size + 1 : MATRIX_SIZE / size;
            MPI_Send(buffer, recv_task_count * MATRIX_SIZE, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&matrix[rank][0], task_count * MATRIX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_count; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    int i, j, k;
#pragma omp parallel num_threads(THREAD_COUNT) default(none) private(i, j, k) shared(matrix, MATRIX_SIZE, size, rank)
    for (k = 0; k < MATRIX_SIZE; k++)
    {
#pragma omp single
        {
            if (k % size == rank)
            {
                for (j = k + 1; j < MATRIX_SIZE; j++)
                {
                    matrix[k][j] /= matrix[k][k];
                }
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++)
                {
                    if (p != rank)
                    {
                        MPI_Send(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                    }
                }
            }
            else
            {
                MPI_Recv(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int begin = k + 1;
        while (begin % size != rank)
        {
            begin++;
        }
#pragma omp for schedule(simd : guided)
        for (int i = begin; i < MATRIX_SIZE; i += size)
        {
            for (j = k + 1; j < MATRIX_SIZE; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    if (rank == 0)
    {
        duration_MPI_OMP += (end_time - start_time) * 1000;
    }
}

// MPI和NEON、OMP结合
void MPI_NEON_OMP_process()
{
    double start_time, end_time;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int task_count; // 每个进程的任务量
    if (rank < MATRIX_SIZE % size)
    {
        task_count = MATRIX_SIZE / size + 1;
    }
    else
    {
        task_count = MATRIX_SIZE / size;
    }
    float *buffer = new float[task_count * MATRIX_SIZE];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < MATRIX_SIZE; i += size)
            {
                for (int j = 0; j < MATRIX_SIZE; j++)
                {
                    buffer[i / size * MATRIX_SIZE + j] = matrix[i][j];
                }
            }
            int recv_task_count = p < MATRIX_SIZE % size ? MATRIX_SIZE / size + 1 : MATRIX_SIZE / size;
            MPI_Send(buffer, recv_task_count * MATRIX_SIZE, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&matrix[rank][0], task_count * MATRIX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_count; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    int i, j, k;
#pragma omp parallel num_threads(THREAD_COUNT) default(none) private(i, j, k) shared(matrix, MATRIX_SIZE, size, rank)
    for (k = 0; k < MATRIX_SIZE; k++)
    {
#pragma omp single
        {
            if (k % size == rank)
            {
                float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
                float32x4_t vec_temp = vld1q_f32(temp);
                vrecpeq_f32(vec_temp);
                int index = k + 1;
                for (int j = k + 1; j + 4 <= MATRIX_SIZE; j += 4, index = j)
                {
                    float32x4_t vec_a;
                    vec_a = vld1q_f32(matrix[k] + j);
                    vec_a = vmulq_f32(vec_a, vec_temp);
                    vst1q_f32(matrix[k] + j, vec_a);
                }
                for (int j = index; j < MATRIX_SIZE; j++)
                {
                    matrix[k][j] = matrix[k][j] / matrix[k][k];
                }
                matrix[k][k] = 1.0;
                for (int p = 0; p < size; p++)
                {
                    if (p != rank)
                    {
                        MPI_Send(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                    }
                }
            }
            else
            {
                MPI_Recv(&matrix[k][0], MATRIX_SIZE, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int begin = k + 1;
        while (begin % size != rank)
        {
            begin++;
        }
#pragma omp for schedule(simd : guided)
        for (int i = begin; i < MATRIX_SIZE; i += size)
        {
            float32x4_t vaik;
            float temp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            vaik = vld1q_f32(temp);
            int index = k + 1;
            for (int j = k + 1; j + 4 <= MATRIX_SIZE; j += 4, index = j)
            {
                float32x4_t vakj = vld1q_f32(matrix[k] + j);
                float32x4_t vaij = vld1q_f32(matrix[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(matrix[i] + j, vaij);
            }
            for (int j = index; j < MATRIX_SIZE; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[k][j] * matrix[i][k];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    if (rank == 0)
    {
        duration_MPI_NEON_
                duration_MPI_NEON_OMP += (end_time - start_time) * 1000;
    }
}

int main()
{
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 测试MPI_cycle
    for (int i = 0; i < ITERATIONS; i++)
    {
        reset_matrix();
        MPI_process();
    }
    if (rank == 0)
    {
        cout << "MATRIX_SIZE=" << MATRIX_SIZE << " MPI_process: " << duration_MPI / ITERATIONS << "ms" << endl;
    }

    // 测试MPI_NEON
    for (int i = 0; i < ITERATIONS; i++)
    {
        reset_matrix();
        MPI_NEON_process();
    }
    if (rank == 0)
    {
        cout << "MATRIX_SIZE=" << MATRIX_SIZE << " MPI_NEON_process: " << duration_MPI_NEON / ITERATIONS << "ms" << endl;
    }

    // 测试MPI_OMP
    for (int i = 0; i < ITERATIONS; i++)
    {
        reset_matrix();
        MPI_OMP_process();
    }
    if (rank == 0)
    {
        cout << "MATRIX_SIZE=" << MATRIX_SIZE << " MPI_OMP_process: " << duration_MPI_OMP / ITERATIONS << "ms" << endl;
    }

    // 测试MPI_NEON_OMP
    for (int i = 0; i < ITERATIONS; i++)
    {
        reset_matrix();
        MPI_NEON_OMP_process();
    }
    if (rank == 0)
    {
        cout << "MATRIX_SIZE=" << MATRIX_SIZE << " MPI_NEON_OMP_process: " << duration_MPI_NEON_OMP / ITERATIONS << "ms" << endl;
    }

    MPI_Finalize();
    return 0;
}
