mppSort:
	nvcc -ccbin clang++ main.cu -o mppSort -arch=sm_50
