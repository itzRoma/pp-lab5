#include <iostream>
#include <mpi.h>

using namespace std;

/*
	Паралельне програмування: Лабораторна робота №5 (ЛР5)

	Варіант: 18, P = 8
	Функція: A = (B + C * MM) + Z * (MX * MU) * min(Z)

	Автор: Бондаренко Роман Ігорович, група ІО-03
	Дата: 07/12/2022
*/

const int N = 16;
const int P = 8;
const int H = N / P;

// primary resources
int A[N], B[N], C[N], Z[N];
int MM[N][N], MX[N][N], MU[N][N];

// additional resources
int D[N];
int m;
int MA[N][N];

// resources per proccess
int D_H[H], B_H[H], Z_H[H], A_H[H];
int MM_H[N][H], MA_H[N][H], MU_H[N][H];
int m_i;

void fill_matrix(int matrix[N][N]);
void fill_vector(int vector[N]);

int vector_min(int vector[H]);

void print_vector(int t, const char name[], int vector[N]);

int main(int args, char* argv[]) {

	MPI_Init(&args, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int index[] = { 1, 7, 8, 9, 10, 11, 12, 13 };
	int edges[] = { 1, 0, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 1, 1 };
	MPI_Comm graph;
	MPI_Graph_create(MPI_COMM_WORLD, P, index, edges, 1, &graph);

	cout << "Thread T" << rank + 1 << " started" << endl;

	switch (rank) {
		case 0: {
			// 1.1. Введення: MX, C.
			fill_matrix(MX);
			fill_vector(C);

			break;
		}
		case 1: {
			// 1.2. Введення: MM, Z.
			fill_matrix(MM);
			fill_vector(Z);

			break;
		}
		case 3: {
			// 1.4. Введення: MU, B.
			fill_matrix(MU);
			fill_vector(B);

			break;
		}
		default: break;
	}

	MPI_Barrier(graph); // Do I really need this?

	switch (rank) {
		case 0: {
			// 2.1. Надіслати задачі Т2: MX, C.
			MPI_Send(&MX, N * N, MPI_INT, 1, 0, graph);
			MPI_Send(&C, N, MPI_INT, 1, 0, graph);

			break;
		}
		case 1: {
			// 2.2. Прийняти від задачі Т1: MX, C.
			MPI_Recv(&MX, N * N, MPI_INT, 0, 0, graph, MPI_STATUSES_IGNORE);
			MPI_Recv(&C, N, MPI_INT, 0, 0, graph, MPI_STATUSES_IGNORE);

			// 2.3. Прийняти від задачі Т4: MU, B.
			MPI_Recv(&MU, N * N, MPI_INT, 3, 0, graph, MPI_STATUSES_IGNORE);
			MPI_Recv(&B, N, MPI_INT, 3, 0, graph, MPI_STATUSES_IGNORE);

			break;
		}
		case 3: {
			// 2.4. Надіслати задачі Т2: MU, B.
			MPI_Send(&MU, N * N, MPI_INT, 1, 0, graph);
			MPI_Send(&B, N, MPI_INT, 1, 0, graph);

			break;
		}
		default: break;
	}

	// 4.2. Надіслати всім задачам: BH, C, MMH, MX, MUH, ZH, AH, Z.
	MPI_Scatter(&B, H, MPI_INT, &B_H, H, MPI_INT, 1, graph);
	MPI_Bcast(&C, N, MPI_INT, 1, graph);
	MPI_Scatter(&MM, N * H, MPI_INT, &MM_H, N * H, MPI_INT, 1, graph);
	MPI_Bcast(&MX, N * N, MPI_INT, 1, graph);
	MPI_Scatter(&MU, N * H, MPI_INT, &MU_H, N * H, MPI_INT, 1, graph);
	MPI_Scatter(&Z, H, MPI_INT, &Z_H, H, MPI_INT, 1, graph);
	MPI_Scatter(&A, H, MPI_INT, &A_H, H, MPI_INT, 1, graph);
	MPI_Bcast(&Z, N, MPI_INT, 1, graph);

	// (4.1, 5.2, 2.3, 4.4, 2.5, 2.6, 2.7, 2.8). Обчислення 1: DH = BH + C * MMH.
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < N; j++) {
			D_H[i] += C[j] * MM_H[j][i];
		}
		D_H[i] += B_H[i];
	}

	// (5.1, 6.2, 3.3, 5.4, 3.5, 3.6, 3.7, 3.8). Обчислення 2: MAH = MX * MUH.
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < H; j++) {
			int res = 0;
			for (int k = 0; k < N; k++) {
				res += MX[i][k] * MU_H[k][j];
			}
			MA_H[i][j] = res;
		}
	}

	// (6.1, 7.2, 4.3, 6.4, 4.5, 4.6, 4.7, 4.8). Обчислення 3: mi = min(ZH).
	m_i = vector_min(Z_H);

	MPI_Barrier(graph); // Do I really need this?

	// (7.1, 8.2, 5.3, 7.4, 5.5, 5.6, 5.7, 5.8). Обчислення 4: m = min(m, m5).
	MPI_Allreduce(&m_i, &m, 1, MPI_INT, MPI_MIN, graph);

	// (8.1, 9.2, 6.3, 8.4, 6.5, 6.6, 6.7, 6.8). Обчислення 5: AH = DH + Z * MAH * m.
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < N; j++) {
			A_H[i] += Z[j] * MA_H[j][i] * m;
		}
		A_H[i] += D_H[i];
	}

	MPI_Barrier(graph); // Do I really need this?

	// (9.1, 7.3, 9.4, 7.5, 7.6, 7.7, 7.8). Надіслати задачі Т2: AH.
	// 10.2. Прийняти, об’єднати і надіслати всім задачам: AH.
	MPI_Gather(&A_H, H, MPI_INT, &A, H, MPI_INT, 1, graph);

	if (rank == 1) {
		// 11.2. Надіслати задачі Т4: A.
		MPI_Send(&A, N, MPI_INT, 3, 0, graph);
	}
	else if (rank == 3) {
		// 10.4. Прийняти від задачі Т2: A.
		MPI_Recv(&A, N, MPI_INT, 1, 0, graph, MPI_STATUSES_IGNORE);

		// 11.4. Виведення результату A.
		print_vector(rank + 1, "A", A);
	}

	cout << "Thread T" << rank + 1 << " finished" << endl;

	MPI_Finalize();

	return 0;
}

void fill_matrix(int matrix[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = 1;
		}
	}
}

void fill_vector(int vector[N]) {
	for (int i = 0; i < N; i++) {
		vector[i] = 1;
	}
}

int vector_min(int vector[H]) {
	int min = vector[0];
	for (int i = 0; i < H; i++) {
		if (vector[i] < min) {
			min = vector[i];
		}
	}
	return min;
}

void print_vector(int t, const char name[], int vector[N]) {
	cout << "Thread T" << t << " - Answer " << name << endl;
	for (int i = 0; i < N; i++) {
		cout << vector[i];
		if (i != N - 1) {
			cout << ", ";
		}
	}
	cout << endl;
}