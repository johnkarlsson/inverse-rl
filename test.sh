clang++ -std=c++11 $(find src | grep .cc$) -lgsl -lgslcblas -o bin/main.o; ./bin/main.o
