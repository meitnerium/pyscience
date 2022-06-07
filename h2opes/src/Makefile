h2opes: main.o pes.o h2opot.o
	gfortran -lstdc++ $^ -o $@

main.o: main.cpp
	g++ -c $< -o $@

pes.o: pes.cpp
	g++ -c $< -o $@

h2opot.o: h2opot.f90
	gfortran -c $< -o $@

clean:
	rm *.o