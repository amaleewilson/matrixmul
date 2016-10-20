
all:
	nvcc -lopenblas -lcublas -o matrix_mul mat_mul.cu
