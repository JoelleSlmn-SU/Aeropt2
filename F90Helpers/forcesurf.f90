program forcesurf
implicit none
character*4 istep
character*50 groupname,filename,text
integer :: ne,np,nbf,i,j,k,nsurf,test,ie,ip,it,in
integer :: isurf,ip1,ip2,ip3,nssrf,nlines,num1,num2,csurf
real*8 :: veca(3),vecb(3),gammaMinusOne,press1,press2,press3,press,prod(3),datum(3)
real*8 :: LIFT,DRAG,LATF,LIFTTOT,DRAGTOT,LATFTOT,force(3),mag,areasum,Minf,gamma
real*8 :: LATM, LATM1, LATM2, LONGM, ROLLM,  xcoor, ycoor, zcoor,LATMTOT,LONGMTOT
integer, allocatable :: ieltet(:,:),iface(:,:),surf(:,:)
real*8, allocatable :: coor(:,:),u(:,:),norm(:,:),area(:)

write(6,*) '*******************************************************' 
write(6,*) '*********   SURFACE FORCE BREAKDOWN    ****************'
write(6,*) '*******************************************************'
write(6,*) 
write(6,*) 'Enter the filegroup name: '
read(5,'(a)') groupname
write(6,*) 'OK, reading file group: ', groupname(1:len_trim(groupname))
! READ THE UNFORMATTED .plt FILE
    write(6,*) 'Opening the unformatted .plt file: ', groupname(1:len_trim(groupname))//'.plt'
    open(10,file=groupname(1:len_trim(groupname))//'.plt',form='unformatted',status='old')
    write(*,*) 'opened' 
    read(10) ne,np,nbf
    write(*,*) 'read in ne,np,nbf' 
    write(*,*) 'ne=',ne
    write(*,*) 'np=',np
    write(*,*) 'nbf=',nbf
    allocate (ieltet(4,ne),coor(3,np),iface(5,nbf),u(np,5))
    allocate (norm(3,nbf),area(nbf))
    read(10) 
!  ((ieltet(in,ie),ie=1,ne),in=1,4) 
    read(10) ((coor(in,ip),ip=1,np),in=1,3)
    read(10) ((iface(in,it),it=1,nbf),in=1,5)
! READ THE UNFORMATTED .unk FILE
  write(6,*) 'Opening the unformatted .unk file: ', groupname(1:len_trim(groupname))//'.unk'
  open(10,file=groupname(1:len_trim(groupname))//'.unk',form='unformatted',status='old')
  write(*,*) 'opened' 
  read(10) np
  read(10) ((u(i,j),i=1,np),j=1,5)
  write(*,*) 'read in unknowns' 
  close(10)
! RETURN UNK VARIABLES TO SOLVER VARIABLES
 ! do i=1,np
 !   do j=2,5
 !     u(i,j) = u(i,j)*u(i,1)
 !   enddo
 ! enddo
! READ THE .BCO FILE
  write(6,*) 'Enter the .bco filename: '
  read(5,'(a)') filename
  open(10,file=filename,form='formatted',status='unknown')
  read(10,*) text
  read(10,*) nssrf,nlines
  read(10,*) text
  allocate(surf(2,nssrf))
  do i=1,nssrf
    read(10,'(A)',iostat=ios) line
    if (ios /= 0) then
      write(*,*) 'ERROR: unexpected EOF while reading surfaces at i=', i
      stop
    end if
    ! Try 4-column: id, num1, num2, class
    read(line,*,iostat=ios) surf(1,i), num1, num2, surf(2,i)
    if (ios /= 0) then
      ! Fallback to 3-column: id, num1, class
      read(line,*,iostat=ios) surf(1,i), num1, surf(2,i)
      if (ios /= 0) then
        write(*,*) 'ERROR: bad BCO surface line at i=', i, ': ', trim(line)
        stop
      end if
      num2 = 0
    end if
    write(*,*)  surf(1,i), num1, num2, surf(2,i)
  enddo
  read(10,*) text
  write(*,*) 'text'
  close(10)
! GET THE FREESTREAM MACH NUMBER
  write(6,*) 'Enter the freestream Mach number: '
  read(5,*) Minf
! GET THE DATUM FOR COMPUTATION OF MOMENTS
  write(6,*) 'Enter the datum for moment computation: '
  read(5,*) datum(1),datum(2),datum(3)
! FIND THE MAX NUMBER OF CAR SURFS
  csurf = 0
  do i=1,nssrf
    test = surf(2,i)
    if(test.gt.csurf) csurf = test
  enddo
  print*,'CSURF =',CSURF 
! OPEN THE OUTPUT FILE
  write(6,*) 'Enter the output filename: '
  read(5,'(a)') filename
  open(10,file=filename,form='formatted',status='unknown')
! COMPUTE THE NUMBER OF MESH SURFACES
  nsurf = 0
  do i=1,nbf
    test = iface(5,i)
    if(test.gt.nsurf)nsurf=test
  enddo
  if(nsurf.ne.nssrf)print*,'ERROR IN NUMBER OF SURFACES!'
  print*,'NSURF = ',nsurf
! LOOP OVER BOUNDARY FACES TO COMPUTE NORMALS AND AREAS
  do it=1,nbf
    do i=1,3
      veca(i) = coor(i,iface(3,it))-coor(i,iface(1,it))
      vecb(i) = coor(i,iface(2,it))-coor(i,iface(1,it))
    enddo
!
    call vcprod(vecb,veca,norm(:,it))
    mag = sqrt(norm(1,it)**2+norm(2,it)**2+norm(3,it)**2)
    area(it) = 0.5*mag
    norm(1,it) = norm(1,it)/mag
    norm(2,it) = norm(2,it)/mag
    norm(3,it) = norm(3,it)/mag
  enddo
! LOOP OVER SURFACES AND COMPUTE LIFT AND DRAG
  gammaMinusOne = 0.4
  gamma = 1.4
  LIFTTOT = 0.0
  DRAGTOT = 0.0
  LATFTOT = 0.0 
  LATMTOT = 0.0
  LONGMTOT = 0.0
  do isurf = -2,CSURF 
    LIFT = 0.0
    DRAG = 0.0
    LATF = 0.0
    LATM = 0.0
    LATM1 = 0.0
    LATM2 = 0.0
    LONGM = 0.0 
    ROLLM = 0.0
    areasum=0.0
    do it = 1,nbf
      test = iface(5,it)
      if(surf(2,test).eq.isurf)then
        ip1 = iface(1,it)
        ip2 = iface(2,it)
        ip3 = iface(3,it)
        xcoor = (1./3.)*(coor(1,ip1)+coor(1,ip2)+coor(1,ip3))
        ycoor = (1./3.)*(coor(2,ip1)+coor(2,ip2)+coor(2,ip3))
        zcoor = (1./3.)*(coor(3,ip1)+coor(3,ip2)+coor(3,ip3))
! compute pressure
        press1 = gammaMinusOne*u(ip1,1)*(u(ip1,5)-0.5*(u(ip1,2)**2+u(ip1,3)**2+u(ip1,4)**2))
        press2 = gammaMinusOne*u(ip2,1)*(u(ip2,5)-0.5*(u(ip2,2)**2+u(ip2,3)**2+u(ip2,4)**2))
        press3 = gammaMinusOne*u(ip3,1)*(u(ip3,5)-0.5*(u(ip3,2)**2+u(ip3,3)**2+u(ip3,4)**2))
        press = (1./3.)*(press1+press2+press3) - (1.0/(gamma*Minf*Minf))    !this is now a pressure coefficient
! compute force
 !       if((ycoor.lt.-4.2).AND.(ycoor.gt.-8.3))then
        LIFT = LIFT - area(it)*press*norm(3,it)
        DRAG = DRAG - area(it)*press*norm(1,it)
        LATF = LATF - area(it)*press*norm(2,it)
        LATM = LATM + area(it)*press*norm(2,it)*(xcoor-datum(1))-area(it)*press*norm(1,it)*(ycoor-datum(2))
 !       LATM1 = LATM1 - area(it)*press*norm(2,it)*(xcoor-datum(1))
        LATM2 = LATM2 + area(it)*press*norm(1,it)*(ycoor-datum(2))
 !       LONGM = LONGM + area(it)*press*norm(2,it)*(zcoor-datum(3))-area(it)*press*norm(3,it)*(xcoor-datum(1))
        LONGM = LONGM + area(it)*press*norm(1,it)*(zcoor-datum(3))
        ROLLM = ROLLM + area(it)*press*norm(3,it)*(ycoor-datum(2))+area(it)*press*norm(2,it)*(zcoor-datum(3))
 !       ROLLM = ROLLM + area(it)*press*norm(3,it)*(ycoor-datum(2))
 !       LONGM = LONGM  -area(it)*press*norm(3,it)*(xcoor-datum(1))
        areasum = areasum + area(it)
 !       endif
!        
      endif
    enddo
    LIFT = 2.0*LIFT
    DRAG = 2.0*DRAG
    LATF = 2.0*LATF
    LATM = 2.0*LATM
    LATM1 = 2.0*LATM1
    LATM2 = 2.0*LATM2
    LONGM = 2.0*LONGM
    ROLLM = 2.0*ROLLM
! add up the totals
    if(isurf.ge.1)then
      LIFTTOT = LIFTTOT + LIFT
      DRAGTOT = DRAGTOT + DRAG
      LATFTOT = LATFTOT + LATF
      LATMTOT = LATMTOT + LATM
      LONGMTOT = LONGMTOT + LONGM
    endif
! print
    print*,'Car Surface:',isurf,'LIFT,DRAG,LATF,LATM2,LONGM, (area):',LIFT,DRAG,LATF,LATM,LONGM,'(',areasum,')'
    write(10,100) isurf,LIFT,DRAG,LATF,LATM,LONGM,ROLLM
  enddo
  close(10)
! write totals to screen
  print*,'TOTALS:'
  print*
  print*,'TOTAL LIFT:',LIFTTOT
  print*,'TOTAL DRAG:',DRAGTOT
  print*,'TOTAL LATF:',LATFTOT
  print*,'TOTAL LATM:',LATMTOT
  print*,'TOTAL LONGM:',LONGMTOT
! 
100 format(I10,6E14.5)
  stop
  end

!------------------------------------------------------------------------------------------------------------
 subroutine scprod(v1,v2,prod)
 implicit none
 
 real :: prod,v1(3),v2(3)

 prod = v1(1)*v2(1)+v1(2)*v2(2)+v1(3)*v2(3)

 end subroutine
!-------------------------------------------------------------------------------------------------------------
 subroutine vcprod(v1,v2,v3)
 implicit none
 
 real :: v1(3),v2(3),v3(3)

 v3(1) = v1(2)*v2(3) - v1(3)*v2(2)
 v3(2) = v1(3)*v2(1) - v1(1)*v2(3)
 v3(3) = v1(1)*v2(2) - v1(2)*v2(1)

 end subroutine
!-----------------------------------------------------------------------------------------------------------


