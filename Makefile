all:
	g++ -c Subset.cpp
	g++ -o main main.cpp Subset.o