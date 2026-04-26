
program plot
 IMPLICIT NONE

 character :: filename*80,caseName*80,outFile*80,fm*4,char_ipr*4
 integer :: numberOfNodes,nameLength,i,j,k,xlen,numberOfDomains
 integer :: numberOfLocalNodes,numberOfToNodes
 logical :: extractTurbulence,hasTurbulence
 real :: x1(4),x2(4),maxdiff(4),buffr
 real,pointer :: u(:,:),unk(:,:)
 integer,pointer :: transArray(:,:)
 integer :: maxInt(4),globInd
 ! read filenames
 write(*,*) "Welcome to makeplot"
 write(*,'(A)',advance="no") "Register file: "
 read(*,'(A)') fileName
 write(*,'(A)',advance="no") "Result name: "
 read(*,'(A)') caseName
 write(*,'(A)',advance="no") "Outfile: "
 read(*,'(A)') outFile 
 write(*,'(A)',advance="no") "Extract turbulence as 5'th variable (T/F): "
 read(*,*) extractTurbulence
 if(.not.extractTurbulence) then 
  write(*,'(A)',advance="no") "Include turbulent field (T/F): "
  read(*,*) hasTurbulence
 end if

  nameLength = nameLen(fileName)
  write(*,*) "opening ",fileName(1:nameLength)
  open(15,file=fileName(1:nameLength),form='unformatted',status='old')
  nameLength = nameLen(caseName)

  read(15) numberOfNodes,numberOfDomains

  write(*,*) "number of nodes and domains: ",numberOfNodes,numberOfDomains

  allocate(u(numberOfNodes,6))

  write(*,*) "starting merging..."

  ! read translation array

 do i=1,numberOfDomains
  read(15) numberOfLocalNodes

  write(*,*) "number of local nodes for domain: ",numberOfLocalNodes

  allocate(transArray(numberOfLocalNodes,2))
 
  transArray = 0

  do j=1,numberOfLocalNodes
   read(15) transArray(j,:)
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

  write(*,*) "opening ",caseName(1:nameLength)//'_'//char_ipr(1:xlen)
  open(16,file=caseName(1:nameLength)//'_'//char_ipr(1:xlen),form='unformatted',status='old')

  read(16) numberOfToNodes
  allocate(unk(numberOfToNodes,6))
  if(extractTurbulence) then 
   read(16) ((unk(j,k),j=1,numberOfToNodes),k=1,4),(buffr,j=1,numberOfToNodes),(unk(j,5),j=1,numberOfToNodes)
  else if(hasTurbulence) then 
   read(16) ((unk(j,k),j=1,numberOfToNodes),k=1,6)
  else
   read(16) ((unk(j,k),j=1,numberOfToNodes),k=1,5)
  end if
  close(16)

  do j=1,numberOfLocalNodes
   u(transArray(j,2),:) = unk(transArray(j,1),:) 
  end do
 
  deallocate(unk)
 end do

 close(15)
 deallocate(transArray) 
 nameLength = nameLen(outFile)
 write(*,*) "opening ",outFile(1:nameLength)
 open(17,file=outFile(1:nameLength),form='unformatted',status='unknown')

 write(17) numberOfNodes
 if(hasTurbulence) then 
  write(17) ((u(i,j),i=1,numberOfNodes),j=1,6)
 else
  write(17) ((u(i,j),i=1,numberOfNodes),j=1,5)
 end if 
 close(17)

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

