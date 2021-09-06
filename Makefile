all:
	g++ -c Obstacle.cpp
	g++ -c Graph.cpp
	g++ -o main main.cpp Graph.o Obstacle.o