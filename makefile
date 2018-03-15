MCC=mpic++
MCFLAGS = -std=c++11 -O3 -Wno-unused-result
MLFLAGS = -lmpi -lstdc++

all: mugc muca

mugc: mugc.cpp
	$(MCC) $(MCFLAGS) mugc.cpp -o run_mugc $(LDFLAGS) $(LDLIBS)

muca: muca.cpp
	$(MCC) $(MCFLAGS) muca.cpp -o run_muca $(LDFLAGS) $(LDLIBS)

clean:
	rm -f ./run*
