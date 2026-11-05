#Compilers and flags

MF	= ifort
MC	= cc
ML	= ar
RM	= rm

FFLAGS	=  -r8 -O3 
CFLAGS	=  -r8 -O3 
#FFLAGS	= -g 
#CFLAGS	= -g 
LFLAGS	= crv

#C and Fortran Object Files
FOBJS	=  mgns2d_defPrimC.o mgns2d_mg.o mgns2d_main.o 


OBJS	=  $(FOBJS) 

#Compile and link

default:
	make -s comment

all:  $(OBJS)
	$(MF) $(FFLAGS) -o mgns2d $(OBJS) 
#      $(RM) -f $(FFOBJS)

comment:
		echo " "
		echo "   Makefile for mgns2d        "
		echo "   ------------------------- "
		echo "    clean  to remove object files "              
		echo "    all    to compile and link "
		echo " "
		
clean:
	$(RM) -f $(OBJS) mgns2d
mgns2d_defPrimC.o:
	$(MF) $(FFLAGS) -c mgns2d_defPrimC.f90

mgns2d_mg.o:
	$(MF) $(FFLAGS) -c mgns2d_mg.f90
mgns2d_main.o:
	$(MF) $(FFLAGS) -c mgns2d_main.f90
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
