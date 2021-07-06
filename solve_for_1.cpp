#include <iostream>
#include <ctime>
#include <cmath>
#include <mpi.h>
#include <cstdio>
#include <stdlib.h>
#include <cstring>
#define K 20
#define LEN 1.0

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>

enum slice_type {x_slice, y_slice, z_slice};

class Matrix {
	float* data;
	int x;
	int y;
	int z;
public:
	Matrix(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {
		data = new float[x_ * y_ * z_];
	}
	float operator()(int i, int j, int k) const {
		return data[k * x * y + j * x + i];
	}
	float& operator()(int i, int j, int k) {
		return data[k * x * y + j * x + i];
	}
	void swap(Matrix& matrix) {
		float* tmp = matrix.data;
		matrix.data = data;
		data = tmp;
		x = matrix.x;
		y = matrix.y;
		z = matrix.z;
	}
	int X() {
		return x;
	}
	int Y() {
		return y;
	}
	int Z() {
		return z;
	}
	float* slice_get(slice_type sls, int number) {
		float* slice;
		switch (sls) {
			case x_slice:
				slice = new float[y * z];
				for (int k = 0; k < z; k++) {
					for (int j = 0; j < y; j++) {
						slice[k * y + j] = this->operator()(number, j, k);
					}
				}
				return slice;
			case y_slice:
				slice = new float[x * z];
				for (int k = 0; k < z; k++) {
					for (int i = 0; i < x; i++) {
						slice[k * x + i] = this->operator()(i, number, k);
					}
				}
				return slice;
			case z_slice:
				slice = new float[x * y];
				for (int j = 0; j < y; j++) {
					for (int i = 0; i < x; i++) {
						slice[j * x + i] = this->operator()(i, j, number);
					}
				}
				return slice;
		} 
		return 0;
	}
	void slice_set(float* slice, slice_type sls, int number) {
		switch (sls) {
			case x_slice:
				for (int k = 0; k < z; k++) {
					for (int j = 0; j < y; j++) {
						data[k * x * y + j * x + number] = slice[k * y + j];
					}
				}
				break;
			case y_slice:
				for (int k = 0; k < z; k++) {
					for (int i = 0; i < x; i++) {
						data[k * x * y + number * x + i] = slice[k * x + i];
					}
				}
				break;
			case z_slice:
				for (int j = 0; j < y; j++) {
					for (int i = 0; i < x; i++) {
						data[number * x * y + j * x + i] = slice[j * x + i];
					}
				}
				break;
		} 
	}
	~Matrix() {
		delete[] data;
	}
};

struct Process {
	MPI_Comm comm;	//коммуникатор
	int left_n;		//соседи
	int right_n;
	int up_n;
	int down_n;
	int front_n;
	int back_n;
	int mine;
	int dims[3];	
};

float fi(float x, float y, float z)
{
	return sin(M_PI * x / LEN) * 
		   sin(2.0 * M_PI * y / LEN) * 
		   sin(3.0 * M_PI * z / LEN); 
}

float u_analytical(float x, float y, float z, float t)
{
	float a = M_PI * sqrt((1.0 + 4.0 + 9.0)/ (LEN * LEN));
	return sin(M_PI * x / LEN) * 
		   sin(2.0 * M_PI * y / LEN) * 
		   sin(3.0 * M_PI * z / LEN) * 
		   cos(a * t);
}

float delta(float x_1, float x_2, float y_1, float y_2,
			float z_1, float z_2, float u, float h)
{
	return (x_1 - 6 * u + x_2 + y_1 + y_2 + z_1 + z_2) / (h * h);
}

char* create_path(Process proc, int N, int core)
{
	char* path = new char[80];
	sprintf(path, "%s","solution_L1_");
	char str_n[80];
	char str_core[80];
	char str_file[80];
	sprintf(str_n, "N%d_", N);
	sprintf(str_core, "c%d/", core);
	strcat(path, str_n);
	strcat(path, str_core);
	struct stat s = { 0 };
	if (stat(path, &s) == -1 && mkdir(path, ACCESSPERMS) != 0)
    {
        printf("Directore was created\n");
    }
	sprintf(str_file, "res_proc%d.csv", proc.mine);
	strcat(path, str_file);
	std::cout << path << std::endl;
	return path;
}

FILE* print_matrix(Matrix& arr, Process proc, int start[3], int N, int core, float inaccuracy[K])
{
	int st_left = proc.left_n < 0 ? 0 : 1;
	int st_back = proc.back_n < 0 ? 0 : 1;
	int st_down = proc.down_n < 0 ? 0 : 1;
	int fn_right = proc.right_n < 0 ? 0 : 1;
	int fn_front = proc.front_n < 0 ? 0 : 1;
	int fn_up = proc.up_n < 0 ? 0 : 1;

	char* path = create_path(proc, N, core);
	FILE* fout = fopen(path, "w");
	fprintf(fout, "N = %d, core = %d, L = %f\n", N, core, LEN);
	fprintf(fout, "process rank = %d\n", proc.mine);
	fprintf(fout, "left_neighbour = %d\n", proc.left_n);
	fprintf(fout, "right_neighbour = %d\n", proc.right_n);
	fprintf(fout, "up_neighbour = %d\n", proc.up_n);
	fprintf(fout, "down_neighbour = %d\n", proc.down_n);
	fprintf(fout, "front_neighbour = %d\n", proc.front_n);
	fprintf(fout, "back_neighbour = %d\n", proc.back_n);
	fprintf(fout, "x_start = %d, y_start = %d, z_start = %d\n", 
			start[0] + st_back, start[1] + st_left, start[2] + st_down);
	fprintf(fout, "x_size = %d, y_size = %d, z_size = %d\n", 
			arr.X() - fn_front - st_back, arr.Y() - fn_right - st_left, arr.Z() - fn_up - st_down);
	fprintf(fout, "inaccuracy: ");
	for(int t = 0; t < K; t++) {
		fprintf(fout, "%f ", inaccuracy[t]);
	}
	putc('\n', fout);

	for (int i = st_back; i < arr.X() - fn_front; i++) {
		for (int j = st_left; j < arr.Y() - fn_right; j++) {
			for (int k = st_down; k < arr.Z() - fn_up; k++) {
				fprintf(fout, "%f ", arr(i, j, k));
			}
			putc('\n', fout);
		}
		putc('\n', fout);
	}
	delete[] path;

	return fout;
}

void define_u0_u1(Process& proc, 
				  Matrix& u0, 
				  Matrix& u1, 
				  int start[3], 
				  float h, 
				  float tau)
{
	//calculation inside points
    for(int i = start[0] + 1; i < start[0] + u0.X() - 1; i++) {
    	for (int j = start[1] + 1; j < start[1] + u0.Y() - 1; j++) {
    		for (int k = start[2] + 1; k < start[2] + u0.Z() - 1; k++) {
	    			u0(i - start[0], j - start[1], k - start[2]) = 
	    				fi(i * h, j * h, k * h);
	    			u1(i - start[0], j - start[1], k - start[2]) = 
	    			// u_analytical(i * h, j * h, k * h, tau * 2);
	    				u0(i - start[0], j - start[1], k - start[2]) +
	    					(tau * tau) / 2 *
								delta(fi((i - 1) * h, j * h, k * h), 
									  fi((i + 1) * h, j * h, k * h), 
									  fi(i * h, (j - 1) * h, k * h), 
									  fi(i * h, (j + 1) * h, k * h),
									  fi(i * h, j * h, (k - 1) * h), 
									  fi(i * h, j * h, (k + 1) * h), 
									  fi(i * h, j * h, k * h), 
									  h);
			}
		}
	}
	//calculation boundary points
    if (proc.down_n < 0) {
	    for (int i = start[0]; i < start[0] + u0.X(); i++) {
	    	for (int j = start[1]; j < start[1] + u0.Y(); j++) {

	    		u0(i - start[0], j - start[1], 0) = 
	    			u_analytical(i * h, j * h, 0.0, 0.0);

    			u1(i - start[0], j - start[1], 0) = 
    				u_analytical(i * h, j * h, 0.0, tau);
	    	}
	    }
	}
	if (proc.up_n < 0) {
	    for (int i = start[0]; i < u0.X(); i++) {
	    	for (int j = start[1]; j < u0.Y(); j++) {

	    		u0(i - start[0], j - start[1], u0.Z() - 1) = 
	    			u_analytical(i * h, j * h, (u0.Z() - 1 + start[2]) * h, 0.0);

    			u1(i - start[0], j - start[1], u0.Z() - 1) = 
    				u_analytical(i * h, j * h, (u0.Z() - 1 + start[2]) * h, tau);
	    	}
	    }
	}
	if (proc.left_n < 0) {
	    for (int i = start[0]; i < start[0] + u0.X(); i++) {
	    	for (int k = start[2]; k < start[2] + u0.Z(); k++) {

	    		u0(i - start[0], 0, k - start[2]) = 
	    			u_analytical(i * h, 0.0, k * h, 0.0);

    			u1(i - start[0], 0, k - start[2]) = 
    				u_analytical(i * h, 0.0, k * h, tau);
	    	}
	    }
	}
	if (proc.right_n < 0) {
	    for (int i = start[0]; i < start[0] + u0.X(); i++) {
	    	for (int k = start[2]; k < start[2] + u0.Z(); k++) {
	    		
	    		u0(i - start[0], u0.Y() - 1, k - start[2]) = 
	    			u_analytical(i * h, (u0.Y() - 1 + start[1]) * h, k * h, 0.0);

    			u1(i - start[0], u0.Y() - 1, k - start[2]) = 
    				u_analytical(i * h, (u0.Y() - 1 + start[1]) * h, k * h, tau);
	    	}
	    }
	}
	if (proc.back_n < 0) {
	    for (int j = start[1]; j < start[1] + u0.Y(); j++) {
	    	for (int k = start[2]; k < start[2] + u0.Z(); k++) {
	    		
	    		u0(0, j - start[1], k - start[2]) = 
	    			u_analytical(0.0, j * h, k * h, 0.0);

    			u1(0, j - start[1], k - start[2]) = 
    				u_analytical(0.0, j * h, k * h, tau);
	    	}
	    }
	}
	if (proc.front_n < 0) {
	    for (int j = start[1]; j < start[1] + u0.Y(); j++) {
	    	for (int k = start[2]; k < start[2] + u0.Z(); k++) {
	    		
	    		u0(u0.X() - 1, j - start[1], k - start[2]) = 
	    			u_analytical((u0.X() - 1 + start[0]) * h, j * h, k * h, 0.0);

    			u1(u0.X() - 1, j - start[1], k - start[2]) = 
    				u_analytical((u0.X() - 1 + start[0]) * h, j * h, k * h, tau);
	    	}
	    }
	}
}

void step(Process& proc, 
		  Matrix& previos_arr, 
		  Matrix& current_arr, 
		  Matrix& next_arr, 
		  float tau, 
		  float h, int t,
		  float& inaccuracy,
		  int start[3])
{ 
	//calculation inside points
	for(int i = 1; i < next_arr.X() - 1; i++) {
    	for (int j = 1; j < next_arr.Y() - 1; j++) {
    		for (int k = 1; k < next_arr.Z() - 1; k++) {
    			next_arr(i, j, k) = 2 * current_arr(i, j, k) - 
    				previos_arr(i, j, k) + tau * tau * 
    				delta(current_arr(i - 1, j, k), current_arr(i + 1, j, k),
	    				  current_arr(i, j - 1, k), current_arr(i, j + 1, k),
	    				  current_arr(i, j, k - 1), current_arr(i, j, k + 1),
	    				  current_arr(i, j, k), h);
    			//std::cout << "u_n = " << next_arr(i, j, k) << std::endl;
    			if (std::abs(next_arr(i, j, k) - u_analytical(i * h, j * h, k * h, tau * t)) > inaccuracy) {
    				inaccuracy = std::abs(next_arr(i, j, k) - u_analytical((i + start[0]) * h, (j + start[1]) * h, (k + start[2]) * h, tau * t));
    			}	
    		}
    	}		
    }

    //calculation boundary points
    if (proc.down_n < 0) {
	    for (int i = 0; i < next_arr.X(); i++) {
	    	for (int j = 0; j < next_arr.Y(); j++) {
				next_arr(i, j, 0) = 0.0;
	    	}
	    }
	}
	if (proc.up_n < 0) {
	    for (int i = 0; i < next_arr.X(); i++) {
	    	for (int j = 0; j < next_arr.Y(); j++) {
				next_arr(i, j, next_arr.Z() - 1) = 0.0;
	    	}
	    }
	}
	if (proc.left_n < 0) {
	    for (int i = 0; i < next_arr.X(); i++) {
	    	for (int k = 0; k < next_arr.Z(); k++) {
	    		next_arr(i, 0, k) = 
	    			(next_arr(i, 1, k) - next_arr(i, next_arr.Y() - 2, k)) / 2;
	    	}
	    }
	}
	if (proc.right_n < 0) {
	    for (int i = 0; i < next_arr.X(); i++) {
	    	for (int k = 0; k < next_arr.Z(); k++) {
	    		next_arr(i, next_arr.Y() - 1, k) = 
	    			(next_arr(i, 1, k) - next_arr(i, next_arr.Y() - 2, k)) / 2;
	    	}
	    }
	}
	if (proc.back_n < 0) {
	    for (int j = 0; j < next_arr.Y(); j++) {
	    	for (int k = 0; k < next_arr.Z(); k++) {
	    		next_arr(0, j, k) = 0.0;
	    	}
	    }
	}
	if (proc.front_n < 0) {
	    for (int j = 0; j < next_arr.Y(); j++) {
	    	for (int k = 0; k < next_arr.Z(); k++) {
	    		next_arr(next_arr.X() - 1, j, k) = 0.0;
	    	}
	    }
	}
}

Process MPI_Start()
{
	Process proc;
	int period[3] = {0, 0, 0};
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	for (int i = 0; i < 3; i++) {
		proc.dims[i] = 0;
	}
	MPI_Dims_create(size, 3, proc.dims);
	std::cout << proc.dims[0] << proc.dims[1] << proc.dims[2] << std::endl;
	MPI_Cart_create(MPI_COMM_WORLD, 3, proc.dims, period, 1, &(proc.comm));
	if (proc.comm == MPI_COMM_NULL) {
		return proc;
	}
	int mine;
	MPI_Comm_rank(proc.comm, &(proc.mine));
	MPI_Cart_shift(proc.comm, 1, -1, &(mine), &(proc.left_n));		
	// std::cout << proc.mine << ' ' << "left = " << proc.left_n << std::endl;
	MPI_Cart_shift(proc.comm, 1, +1, &(mine), &(proc.right_n));
	// std::cout << proc.mine << ' ' << "right = " << proc.right_n << std::endl;
	MPI_Cart_shift(proc.comm, 2, -1, &(mine), &(proc.down_n));
	// std::cout << proc.mine << ' ' << "down = " << proc.down_n << std::endl;
	MPI_Cart_shift(proc.comm, 2, +1, &(mine), &(proc.up_n));
	// std::cout << proc.mine << ' ' << "up = " << proc.up_n << std::endl;
	MPI_Cart_shift(proc.comm, 0, -1, &(mine), &(proc.back_n));
	// std::cout << proc.mine << ' ' << "back = " << proc.back_n << std::endl;
	MPI_Cart_shift(proc.comm, 0, +1, &(mine), &(proc.front_n));
	// std::cout << proc.mine << ' ' << "front = " << proc.front_n << std::endl;
	// std::cout << proc.mine << ' ' << "rank = " << proc.mine << std::endl;
	return proc;
}

void calculate_size(const Process& proc, int start[3], int size[3], int N)
{
	int coords[3];
	// std::cout << proc.mine << ' ' << "N = " << N << std::endl;
	std::cout << proc.mine << ' ' << "my rank = " << proc.mine << std::endl;
	// std::cout << proc.mine << ' ' << "mpi proc null = " << MPI_PROC_NULL << std::endl;	
	MPI_Cart_coords(proc.comm, proc.mine, 3, coords);
	// std::cout << proc.mine << ' ' << "coords: " << coords[0] << ' ' << coords[1] 
	// 		  << ' ' << coords[2] << ' ' << std::endl;
	for (int i = 0; i < 3; i++) {
		int m = ceil((float)N / proc.dims[i]);
		// std::cout << proc.mine << ' ' << "m = " << m << std::endl;
		start[i] = m * coords[i];
		if (start[i] != 0)
			start[i]--;
		int end = std::min(m * (coords[i] + 1) - 1, N - 1);
		if (end != N - 1)
			end++;
		size[i] = end - start[i] + 1;
	}
}

void mpi_swap(const Process& proc, Matrix& next_arr)
{
	// int max_size = std::max(next_arr.X(), std::max(next_arr.Y(), next_arr.X()));
	// float* in_buffer = new float[max_size * max_size];

	//left_neighbour
	if (proc.left_n >= 0) {
		// std::cout << proc.mine << ' ' << "left_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(y_slice, 1);
		float* in_buffer = new float[next_arr.X() * next_arr.Z()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.X() * next_arr.Z(), MPI_FLOAT, proc.left_n, 0, 
					 in_buffer, next_arr.X() * next_arr.Z(), MPI_FLOAT, proc.left_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, y_slice, 0);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	//right_neighbour
	if (proc.right_n >= 0) {
		// std::cout << proc.mine << ' ' << "right_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(y_slice, next_arr.Y() - 2);
		float* in_buffer = new float[next_arr.X() * next_arr.Z()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.X() * next_arr.Z(), MPI_FLOAT, proc.right_n, 0, 
					 in_buffer, next_arr.X() * next_arr.Z(), MPI_FLOAT, proc.right_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, y_slice, next_arr.Y() - 1);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	//down_neighbour
	if (proc.down_n >= 0) {
		// std::cout << proc.mine << ' ' << "down_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(z_slice, 1);
		float* in_buffer = new float[next_arr.X() * next_arr.Y()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.X() * next_arr.Y(), MPI_FLOAT, proc.down_n, 0, 
					 in_buffer, next_arr.X() * next_arr.Y(), MPI_FLOAT, proc.down_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, z_slice, 0);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	//up_neighbour
	if (proc.up_n >= 0) {
		// std::cout << proc.mine << ' ' << "up_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(z_slice, next_arr.Z() - 2);
		float* in_buffer = new float[next_arr.X() * next_arr.Y()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.X() * next_arr.Y(), MPI_FLOAT, proc.up_n, 0, 
					 in_buffer, next_arr.X() * next_arr.Y(), MPI_FLOAT, proc.up_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, z_slice, next_arr.Z() - 1);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	//back_neighbour
	if (proc.back_n >= 0) {
		// std::cout << proc.mine << ' ' << "back_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(x_slice, 1);
		float* in_buffer = new float[next_arr.Y() * next_arr.Z()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.Y() * next_arr.Z(), MPI_FLOAT, proc.back_n, 0, 
					 in_buffer, next_arr.Y() * next_arr.Z(), MPI_FLOAT, proc.back_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, x_slice, 0);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	//front_neighbour
	if (proc.front_n >= 0) {
		// std::cout << proc.mine << ' ' << "front_neighbour" << std::endl;
		float* out_buffer = next_arr.slice_get(x_slice, next_arr.X() - 2);
		float* in_buffer = new float[next_arr.Y() * next_arr.Z()];
		MPI_Status tmp;
		MPI_Sendrecv(out_buffer, next_arr.Y() * next_arr.Z(), MPI_FLOAT, proc.front_n, 0, 
					 in_buffer, next_arr.Y() * next_arr.Z(), MPI_FLOAT, proc.front_n, 0, 
					 proc.comm, &tmp);
		next_arr.slice_set(in_buffer, x_slice, next_arr.X() - 1);
		delete[] out_buffer;
		delete[] in_buffer;
	}

	// delete[] in_buffer;
}

//input: N, L_x=L_y=L_z={1, M_PI}
int main(int argc, char** argv) {
	if (MPI_Init(&argc, &argv) != 0)
    {
        printf("Cannot init MPI \n");
        return 1;
    }
	if (argc < 3) {
		std::cout << "no parameter N or core number" << std::endl;
		MPI_Finalize();
		return 1;
	}
	int N;
	int core;
	sscanf(argv[1], "%d", &N);
	sscanf(argv[2], "%d", &core);
	const float h = float(LEN) / float(N - 1);
    const float tau = h * h / 2;

	double start_time = MPI_Wtime();

    Process proc = MPI_Start();
    if (proc.comm == MPI_COMM_NULL) {
    	MPI_Finalize();
    	printf("No communicator in process\n");
    	return 1;
    }
    int start[3];
    int size[3];
    float inaccuracy[K];
    calculate_size(proc, start, size, N);
    Matrix previos_arr(size[0], size[1], size[2]);
    Matrix current_arr(size[0], size[1], size[2]);
    Matrix next_arr(size[0], size[1], size[2]);
    inaccuracy[0] = inaccuracy[1] = 0.0;

    define_u0_u1(proc, previos_arr, current_arr, start, h, tau);

    for(int t = 2; t < K; t++) {
    	inaccuracy[t] = 0.0;
		step(proc, previos_arr, current_arr, next_arr, tau, h, t, inaccuracy[t], start);
		mpi_swap(proc, current_arr);
		previos_arr.swap(current_arr);
		current_arr.swap(next_arr);
		double sendbuf = inaccuracy[t];
		double recvbuf;
		MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, proc.comm);
		inaccuracy[t] = recvbuf;
		// if (t == 2) {
		// 	fout = print_matrix(current_arr, proc, start, N, core, inaccuracy, tau * t, h);
		// }
	}
	double end_time = MPI_Wtime() - start_time;

	FILE* fout = print_matrix(current_arr, proc, start, N, core, inaccuracy);


	// std::cout << "end printing" << std::endl;
	double sendbuf = end_time;
	double recvbuf;
	MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, proc.comm);
	end_time = recvbuf;
	fprintf(fout, "time = %f\n", end_time);
	fclose(fout);
	// std::cout << "count time" << std::endl;

	if (proc.mine == 0) {
		printf("inaccuracy: ");
		for(int l = 0; l < K; l++) {
			printf("%f ", inaccuracy[l]);
		}
		printf("\n time = %f\n", end_time);
		printf("start: %d %d %d\n",  start[0], start[1], start[2]);
		printf("size: %d %d %d\n", size[0], size[1], size[2]);
	}
	MPI_Finalize();
	return 0;
}
