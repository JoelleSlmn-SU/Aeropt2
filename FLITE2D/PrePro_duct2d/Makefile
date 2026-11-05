OBJS = Aggl2d_Aggl.o  Aggl2d_def.o  Aggl2d_main.o

# --- compiler settings -----------------------------------

CC    = icc
#CC    = gcc
COPT  = -D BIT64 -O3 -ansi -align -malign-double -mtune=core2
CINCL = -I/opt/intel/include 

CF    = ifort
#CF    = gfortran
FOPT  = -O2 -r8 -mtune=core2 -align all -check bounds
FINCL = -I/opt/intel/include
LIB   = 

# --- commands -------------------------------------------------------

default: $(OBJS)
	$(CF) *.o $(FOPT) $(LIB) -o Aggl2d

conly: *.o

%.o : %.f90
	$(CF) -c $(FOPT) $(FINCL) $<

clean:
	find .. -name '*.o' -delete 



