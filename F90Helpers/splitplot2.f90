
program plot
 IMPLICIT NONE

 character :: filename*80,caseName*80,inFile*80,fm*4,char_ipr*4
 integer :: numberOfNodes,nameLength,i,j,k,xlen,numberOfDomains
 integer :: numberOfLocalNodes,numberOfToNodes
 real :: x1(4),x2(4),maxdiff(4)
 real,pointer :: u(:,:),unk(:,:)
 integer,pointer :: transArray(:,:)
 integer :: maxInt(4),globInd
 integer :: RUNFILE,maxind

 character :: problemName*80,eFormat*4,fileExtension*8
 integer :: physicalTimestepNumber,numberOfPhysicalTimesteps,problemNameLength,extLen

 logical :: includeTurbulence


 write(*,*) "Welcome to splitplot"
 write(*,'(A)',advance="no") "Register file: "
 read(*,'(A)') fileName
 write(*,'(A)',advance="no") "Infile: "
 read(*,'(A)') inFile
 write(*,'(A)',advance="no") "Result name: "
 read(*,'(A)') caseName

 write(*,'(A)',advance="no") "Include turbulence?: "
 read(*,*) includeTurbulence

  nameLength = nameLen(fileName)
  write(*,*) "opening ",fileName(1:nameLength)
  open(15,file=fileName(1:nameLength),form='unformatted',status='old')
  nameLength = nameLen(caseName)

  read(15) numberOfNodes,numberOfDomains

  write(*,*) "number of nodes and domains: ",numberOfNodes,numberOfDomains

  allocate(u(numberOfNodes,6))

  ! read unknown file

  nameLength = nameLen(inFile)
  write(*,*) "opening ",inFile(1:nameLength)
  open(17,file=inFile(1:nameLength),form='unformatted',status='old')

  read(17) numberOfNodes
  if(includeTurbulence) then 
   read(17) ((u(i,j),i=1,numberOfNodes),j=1,6)
  else
   read(17) ((u(i,j),i=1,numberOfNodes),j=1,5)
  end if
  close(17)

  write(*,*) "starting splitting..."

 
 do i=1,numberOfDomains
  read(15) numberOfLocalNodes

  write(*,*) "number of local nodes for domain: ",numberOfLocalNodes

  allocate(transArray(numberOfLocalNodes,2))

  transArray = 0

  do j=1,numberOfLocalNodes
   read(15) transArray(j,:)
  end do

  maxind = 0
  do j=1,numberOfLocalNodes
   if(transArray(j,1)>maxind) maxind = transArray(j,1)
  end do


  allocate(unk(maxind,6))

  do j=1,numberOfLocalNodes
   unk(transArray(j,1),:) = u(transArray(j,2),:)
  end do 

  if (i.le.9) then
   fm='(i1)'
   xlen = 1
  else if (i.le.99) then
   fm='(i2)'
   xlen = 2
  else if (i.le.999) then
   fm='(i3)'
   xlen = 3
  endif

  write(char_ipr,fm) i

  nameLength = nameLen(caseName)
  write(*,*) "opening ",caseName(1:nameLength)//'_'//char_ipr(1:xlen)
  open(16,file=caseName(1:nameLength)//'_'//char_ipr(1:xlen),form='unformatted',status='unknown')

  write(16) numberOfLocalNodes,maxind
  write(16) (transArray(j,1),j=1,numberOfLocalNodes)
!  write(16) ((unk(j,k),j=1,numberOfLocalNodes),k=1,5)
  if(includeTurbulence) then 
    print*,'Including SA turbulence parameter'
    write(16) ((unk(transArray(j,1),k),j=1,numberOfLocalNodes),k=1,6)
  else
    write(16) ((unk(transArray(j,1),k),j=1,numberOfLocalNodes),k=1,5)
  end if
  close(16)

  deallocate(unk)
  deallocate(transArray)
 end do

 deallocate(u)

 close(15)

 write(*,*) "done" 

 stop

contains 

integer function nameLen(fn)
IMPLICIT NONE
 character*80 :: fn
 integer :: i
 do i = 80,1,-1
  nameLen = i 
  if(fn(i:i)/=' ') GOTO 77
 end do
 nameLen = 0
77 end function nameLen
end

