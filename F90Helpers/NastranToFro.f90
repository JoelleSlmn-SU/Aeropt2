program NastranToFro

     implicit none 
     character*4 istep,fileExtension,dummy 
     character*50 nasfile,frofile
     character*80 string,string1, string2
     integer ::  i,j,ne,np,nbf,ie,in,ip,it,iz,nstart,nstop,nsample,is,np2,isur
     integer :: ihex,ipri,ipyr,itet,nfaceq,nfacet,icount,itest,tmp,int,nsegm 
     integer :: ip1, ip2, ip3,nPshell
     integer :: nsurf,test,surfno,nbp,isurf,itri,i1,i2,i3,i4,i5,i6,ipx,nsur
     integer, allocatable :: ielhex(:,:),ielpri(:,:),ielpyr(:,:),trino(:),surf_con(:,:) 
     integer, allocatable :: ieltet(:,:),iface(:,:),iel(:,:),gloloc(:),iface_new(:,:) 
     real*4 :: Mach,ss,T,gamma,uinf,press,R,cp,dens
     real*4 :: x,y,z
     real*4, allocatable :: u(:,:),uabs(:,:),coor(:,:),temporary(:,:),uloc(:,:),coor_new(:,:)
     logical :: onemesh,unformatted 
!
     write(6,*) '*******************************************************' 
     write(6,*) '*****  SURFACE MESH .nas -> .fro CONVERTER  ***********'
     write(6,*) '*******************************************************'
     write(6,*) 
     write(6,*) 'Enter the .nas filename: '
     read(5,'(a)') nasfile

     write(6,*) 'Opening the .nas file: ', nasfile
     open(10,file=nasfile,form='formatted',status='old')
     write(*,*) 'opened' 
     read(10,*) string
     read(10,*)
     write(*,*) 'Reading: ',string
     write(6,*) 'Enter the number of PSHELLs: '
     read(5,*) nPshell
     write(6,*) 'nPshell = ',nPshell
     do i=1,nPshell
       read(10,*) string
       write(*,*) 'Reading: ',string
     enddo
     write(6,*) 'Enter the number of nodes in the surface mesh:'
     read(5,*) np
     allocate (coor(3,np))
     do ip=1,np
       read(10,*)   string,j,dummy,x,y,z
       write(*,*) 'Reading point ',j,'x=',x,'y=',y,'z=',z
       coor(1,ip)=x
       coor(2,ip)=y
       coor(3,ip)=z
     enddo
     write(6,*) 'Enter the number of boundary faces in the surface mesh:'
     read(5,*) nbf
     allocate (iface(5,nbf))
     nsurf=0
     do i=1,nbf
       read(10,*) string,j,iface(5,i),iface(2,i),iface(3,i),iface(4,i)
       iface(1,i)=i
       if(iface(5,i).gt.nsurf) nsurf=iface(5,i)
       write(*,*) 'Reading face',j,'iface(:,*)=',iface(1,i),iface(2,i),iface(3,i),iface(4,i),iface(5,i)
     enddo
!
     close(10)
!
! Ask for the new .fro filename
!
    write(6,*) 'Enter the new .fro filename: '
    read(5,'(a)') frofile
    open(11,file=frofile,form='formatted',status='unknown')
!
! write out fro file
!
    write(6,*) 'Writing out the new .fro file'
    write(11,*) nbf,np,1,0,0,nsurf,0,0
    do ip=1,np
      write(11,*) ip,(coor(in,ip),in=1,3)
    enddo
    do it=1,nbf
      write(11,*) (iface(in,it),in=1,5)
    enddo
    close(11)
    write(6,*) 'Finished - good luck with this!'

    end


