clang++ -std=c++11 -g $(find src | grep .cc$) -lgsl -lgslcblas -o bin/main.o;
