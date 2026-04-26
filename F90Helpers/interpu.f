c
      program path
c
c *** program for the computation of particle trajectories.
c
      parameter (mxpoi=1000000,mxele=5000000,mxbou=250000)
      parameter (mxvar=1000000)
      real*8 coord(3,mxpoi),intma(4,mxele),unkno(6,mxpoi) 
      dimension ibsid(5,mxbou),lbg(-20:6*mxele)
      real*8 xi(3,mxvar),st(6,mxvar)
      logical :: doTurb
      common /geocon/ npoin,nelem,nboun,doTurb,numComp
   
c

      character*80 filenam,textread,fname
      character*80 text1
c
c *** open files.
c
      inp1 = 1
      inp2 = 2
      iout = 3
      inp3 = 4
c
      ierr = 0
  333 continue
      if(ierr.eq.1) write(*,'(/,a)') '  Could not open '//fname
      ierr = 1
      write(*,'(//)')
      filenam = textread('  Enter problem name: ')
      l = namlen(filenam)
      write(*,*) "Do turbulence: "
      read(*,*) doTurb
      if(doTurb) then 
       numComp=6
      else
       numComp=5
      end if
c
c *** 3D tetrahedral mesh filename *.plt
c
      fname = filenam(1:l)//'.plt'
      open(inp1,file=fname,err=333,status='old',form='unformatted')
c
c *** Flowfield data filename *.unk
c
      fname = filenam(1:l)//'.unk'
      open(inp2,file=fname,err=333,status='old',form='unformatted')
      write(*,*) "TT: ",fname
c
  533 continue
      write(*,'(//)')
      filenam = textread('  Enter new problem name: ')
      l = namlen(filenam)
      fname = filenam(1:l)//'.plt'
      open(inp3,file=fname,err=533,status='old',form='unformatted')
      fname = filenam(1:l)//'.unk'
      open(iout,file=fname,err=533,status='unknown',form='unformatted')
c
c *** read the input data.
c
      print 400
      read(inp3) nelem,npar,nboun
      npoin=npar
      call gtinpt1(inp3,intma,xi,ibsid)
      read(inp1) nelem,npoin,nboun
      if(nelem.gt.mxele) stop ' path > increase mxele'
      if(npoin.gt.mxpoi) stop ' path > increase mxpoi'
      if(nboun.gt.mxbou) stop ' path > increase mxbou'
      read(inp2) npoin
      call gtinpt(inp1,inp2,intma,coord,unkno,ibsid)
c
c *** set up the tree structure.
c
      print 450
      melem = mxele
      call setup(melem,coord,intma,lbg)
c
c *** computes the tracks of the particles.
c
      print 700
      call trajec(npar,npos,dist,xi,coord,unkno,lbg,st)
c
c *** output of results in the required format.
c
      print 800
      call output(iout,npar,st)
c ...
  100 format(//,'    ****************************************',/,
     .          '    ***** program for particle tracing *****',/,
     .          '    ****************************************',//,
     .          '          geometry  file ?: ',$)
  101 format(/, '          new file       ?: ',$)
  200 format(/, '          unknowns file  ?: ',$)
  300 format(//,'          output  file   ?: ',$)
  400 format(//,'  path > reading input data.')
  450 format(//,'  path > setting up the tree.')
  500 format(/ ,'  Enter nr. of particles ( max = ',i3,' ): ',$)
  510 format(/ ,'  Enter nr. of positions to calculate: ',$)
  520 format(/ ,'  Enter space interval to plot: ',$)
  600 format(/ ,'  Initial coordinates of particle ',i3,' ?: ',$)
  700 format(//,'  path > computing particle trajectories.')
  800 format(//,'  path > writing output file.')
c ...
      stop
      end
c
c ----------------------------------------------------------------------
c
      subroutine gtinpt(inp1,inp2,intma,coord,unkno,ibsid)
c
      common /geocon/ npoin,nelem,nboun,doTurb,numComp
c
      real*8 coord(3,1),unkno(6,1)
      integer intma(4,1),ibsid(5,1)
c
c *** reads the geometry data.
c
      read(inp1) ((intma(j,i),i=1,nelem),j=1,4)
      read(inp1) ((coord(j,i),i=1,npoin),j=1,3)
      read(inp1) ((ibsid(j,i),i=1,nboun),j=1,5)
c
c *** reads the unknowns.
c
      read(inp2) ((unkno(j,i),i=1,npoin),j=1,numComp)
      write(*,*) "A: ",unkno(1,1),npoin,numComp
c
      close(inp1)
      close(inp2)
c
      return
      end
      subroutine gtinpt1(inp1,intma,coord,unkno)
c
      common /geocon/ npoin,nelem,nboun,doTurb,numComp
c
      real*8 coord(3,1),unkno(6,1)
      integer intma(4,1)
c
c *** reads the geometry data.
c
      read(inp1) ((intma(j,i),i=1,nelem),j=1,4)
      read(inp1) ((coord(j,i),i=1,npoin),j=1,3)
c
c *** reads the unknowns.
c
      close(inp1)
c
      return
      end
c
c ...................................................................
c
      subroutine setup(melem,coord,intma,lbg)
c
c *** reading element connectivities and setting up the tree.
c
      common /geocon/ npoin,nelem,nboun,doTurb,numComp
c
      real*8 coord(3,1)
      integer intma(4,1),lbg(-20:1),kel(4)
      real  xel(6)
c
      call xlim(npoin,coord,lbg,lbg,melem)
      do 100 ie = 1,nelem
        do 50 in=1,4
          kel(in) = intma(in,ie)
   50   continue
        call alim4(coord,kel(1),kel(2),kel(3),kel(4),xel   )
        call adtrb(lbg  ,lbg   ,xel   ,kel   )
  100 continue
      return
      end
c
c....................................................................
c
      subroutine trajec(npar,npos,dist,xi,coord,unkno,lbg,st)
c
c *** this subr. computes the trajectories of the particles.
c
      real*8 xi(3,1),coord(3,1),unkno(6,1),st(6,1)
      dimension lbg(-20:1),xr(3),lst(1)
      logical :: doTurb
      common /geocon/ npoin,nelem,nboun,doTurb,numComp
c
      dt(x1,x2,x3,y1,y2,y3,z1,z2,z3) = x1*(y2*z3-y3*z2)+
     .                                 x2*(y3*z1-y1*z3)+
     .                                 x3*(y1*z2-y2*z1)
c
      kp = 0
      lst(1) = kp+1
c
c     sec0=second()
      do 100 ip=1,npar
       if(1000*(ip/1000).eq.ip)then
        print *,' # of points processed = ',ip
c       print *,st(1,ip-1),st(2,ip-1),st(6,ip-1),doTurb
c         print*,' iteration time',second()-sec0
c         sec0=second()
       endif
        xr(1) = xi(1,ip)
        xr(2) = xi(2,ip)
        xr(3) = xi(3,ip)
        call trsear(xr,lbg,lbg,coord,j1,j2,j3,j4,a1,a2,a3,a4,
     .              x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)
        if(a1.lt.0) a1 = 0
        if(a2.lt.0) a2 = 0
        if(a3.lt.0) a3 = 0
        if(a4.lt.0) a4 = 0
        if(a1.gt.1) a1 = 1
        if(a2.gt.1) a2 = 1
        if(a3.gt.1) a3 = 1
        if(a4.gt.1) a4 = 1
        at = (a1+a2+a3+a4)
        if(at.eq.0) then
          print *,'can not find element for point:', ip
          dst =1000000.
          do 10 i = 1 , npoin
           dm = sqrt((coord(1,i)-xr(1))**2+(coord(2,i)-xr(2))**2
     &              +(coord(3,i)-xr(3))**2)
           if(dm.lt.dst) then
             dst = dm
             j1 = i
           end if
  10      continue
          print *,' recovered by using point :',j1
          j2 = j1
          j3 = j2
          j4 = j3
          a1 = 0.25
          a2 = 0.25
          a3 = 0.25
          a4 = 0.25
        else
          a1 = a1 / at
          a2 = a2 / at
          a3 = a3 / at
          a4 = a4 / at
        end if
        dens = a1*unkno(1,j1)+a2*unkno(1,j2)+
     .         a3*unkno(1,j3)+a4*unkno(1,j4)
        velx = a1*unkno(2,j1)+a2*unkno(2,j2)+
     .         a3*unkno(2,j3)+a4*unkno(2,j4)
        vely = a1*unkno(3,j1)+a2*unkno(3,j2)+
     .         a3*unkno(3,j3)+a4*unkno(3,j4)
        velz = a1*unkno(4,j1)+a2*unkno(4,j2)+
     .         a3*unkno(4,j3)+a4*unkno(4,j4)
        ener = a1*unkno(5,j1)+a2*unkno(5,j2)+
     .         a3*unkno(5,j3)+a4*unkno(5,j4)
        if(doTurb) then 
        turb = a1*unkno(6,j1)+a2*unkno(6,j2)+
     .         a3*unkno(6,j3)+a4*unkno(6,j4)
        end if
c
        st(1,ip) = dens
        st(2,ip) = velx
        st(3,ip) = vely
        st(4,ip) = velz
        st(5,ip) = ener
        st(6,ip) = turb
 100   continue
      return
      end
c
c....................................................................
c
      subroutine output(iou,npar,st)
c
c *** this subr. writes the variables for particle tracing.
c
      real*8 st(6,1)
c
       write(iou)npar
          vz = st(4,ip)
          write(iou)((st(i,j),j=1,npar),i=1,6)
  200 continue
      return
      end
c
c....................................................................
c                   **************************
c                   ***  tree subroutines  ***
c                   **************************
c....................................................................
c
      subroutine xlim (npoin, coob ,trb  ,arb  ,melem)
      parameter (mxlev = 120 ,toler=1.e-02)
      integer     trb(-20:1)
      real        arb(-20:1)
      real*8      coob(3,1)  
      real        xli(2,3)
c
      trb(-20) =  6*melem
      trb(-19) =    mxlev
      trb(-18) =        4
      trb(-17) =        4
      trb(-16) =        6
      trb(-15) =        0
      trb(-14) =        1
      trb(-13) =        1
      trb(  0) =        0
      trb(  1) =        0
c
      do 100 id = 1,3
       xli(1,id) =  1.e+10
       xli(2,id) = -1.e+10
  100 continue
      do 101 ip = 1,npoin
       do 102 id = 1,3
        xc        = coob(id,ip)
        xli(1,id) = min(xli(1,id),xc)
        xli(2,id) = max(xli(2,id),xc)
  102  continue
  101 continue
      do 103 id = 1,3
       xc        = xli(2,id)-xli(1,id)
       xli(1,id) = xli(1,id)-xc*toler
       xli(2,id) = xli(2,id)+xc*toler
  103 continue
      do 104 id = 1,3
       ad        = xli(2,id)-xli(1,id)
       xli(2,id) = 1./ad
  104 continue
      do 105 id = 1,3
       arb(-id)   = xli(1,id)
       arb(-id-3) = xli(1,id)
       arb(-id-6) = xli(2,id)
       arb(-id-9) = xli(2,id)
  105 continue
      return
      end
c
c................................................................
c
      subroutine adtrb(trb,arb,xr,k)
      integer     trb(-20:1),k(1)
      real        arb(-20:1),xr(1),x(6)
c
      data  nbox,ndim,nkey/6,6,4/
c
      mxtr    = trb(-20)
      mxle    = trb(-19)
      nele    = trb(-15)
      iava    = trb(-14)
      iend    = trb(-13)
      ipoi    = trb(  0)
      ifth    = 0
c
      do 10 id = 1,ndim
       orig = arb(-id)
       scal = arb(-id-6)
       val  = (xr(id)-orig)*scal
       if(val.lt.0.0.or.val.gt.1.0) stop 'cd060'
       x(id) = val
   10 continue
c
      il = 0
 1000 if(ipoi.eq.0) then
c ***                             add element in available location
         ipoi       = iava
         trb(ifth)  = ipoi
         iava       = trb(ipoi)
         nele       = nele+1
         if(iava.eq.0) then
            iend       = iend+nbox
            iava       = iend
            trb(iava)  = 0
            if(iend.gt.mxtr-nbox) stop 'cd070'
         endif
         trb(ipoi)   = 0
         trb(ipoi+1) = 0
         trb(ipoi+2) = k(1)
         trb(ipoi+3) = k(2)
         trb(ipoi+4) = k(3)
         trb(ipoi+5) = k(4)
      else
c ***                             work way down the tree
         id    = mod(il,ndim)+1
         x(id) = 2.*x(id)
         if(x(id).lt.1.0) then
            ifth  = ipoi
         else
            ifth  = ipoi+1
            x(id) = x(id)-1.
         endif
         il    = il+1
         ipoi  = trb(ifth)
         goto                     1000
      endif
c
      trb(-13) = iend
      trb(-14) = iava
      trb(-15) = nele
      if(il.gt.mxle) stop 'cd080'
      return
      end
c
c...................................................................
c
      subroutine trsear(xr,tr,ar,coob,l1,l2,l3,l4,p1,p2,p3,p4,
     -                    x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)
      parameter(mxlev=120)
      integer tr(-20:1),stak(8*mxlev)
      real*8  coob(3,1)
      real    ar(-20:1),xr(1)
      real    box(6),xl(6),xel(6)
c
      dt(x1,x2,x3,y1,y2,y3,z1,z2,z3) = x1*(y2*z3-y3*z2)+
     -                                 x2*(y3*z1-y1*z3)+
     -                                 x3*(y1*z2-y2*z1)
c
      data   ndim,ncar/6,8/
c
      nkey    = tr(-17)
      ipoi    = tr(  0)
      ifth    = 0
      istk    = 1
      nn      = 0
      ac      = -1.e+6
c
      do 10 id = 1,3
       xl(id)    = 0.
       xl(id+3)  = 0.
       orig      = ar(-id)
       scal      = ar(-id-6)
       box(id  ) = (xr(id)-orig)*scal
       box(id+3) = box(id)
   10 continue
c
      il = 0
 1000 if(ipoi.eq.0) then
         if(istk.eq.1) return
         ipoi   = stak(istk)
         il     = stak(istk+1)
         do 101 id = 1,ndim
          xl(id) = rtr(stak(istk+1+id))
  101    continue
         istk   = istk-ncar
         ipoi   = tr(ipoi+1)
         if(ipoi.ne.0) then
            id     = mod(il,ndim)+1
            amov   = 1./2.**((il/ndim)+1)
            xl(id) = xl(id)+amov
            il     = il+1
            if(id.le.3) then
               if(xl(id)     .gt.box(id)) ipoi = 0
            else
               if(xl(id)+amov.lt.box(id)) ipoi = 0
            endif
         endif
      else
c                           visit node ipoi
         j1 = tr(ipoi+2)
         j2 = tr(ipoi+3)
         j3 = tr(ipoi+4)
         j4 = tr(ipoi+5)
         call alim4(coob,j1,j2,j3,j4,xel)
         do 20 id = 1,ndim
          orig    = ar(-id)
          scal    = ar(-id-6)
          xel(id) = (xel(id)-orig)*scal
   20    continue
         if(xel(1).gt.box(1))    goto 105
         if(xel(2).gt.box(2))    goto 105
         if(xel(3).gt.box(3))    goto 105
         if(xel(4).lt.box(4))    goto 105
         if(xel(5).lt.box(5))    goto 105
         if(xel(6).lt.box(6))    goto 105
c
         x1  = coob(1,j1)
         y1  = coob(2,j1)
         z1  = coob(3,j1)
         x2  = coob(1,j2)
         y2  = coob(2,j2)
         z2  = coob(3,j2)
         x3  = coob(1,j3)
         y3  = coob(2,j3)
         z3  = coob(3,j3)
         x4  = coob(1,j4)
         y4  = coob(2,j4)
         z4  = coob(3,j4)
c
         vl = dt(x2-x1,y2-y1,z2-z1,x3-x1,y3-y1,z3-z1,x4-x1,y4-y1,z4-z1)
         a1 = dt(x2-xr(1),y2-xr(2),z2-xr(3),x3-xr(1),y3-xr(2),z3-xr(3),
     -           x4-xr(1),y4-xr(2),z4-xr(3))/vl
         a2 = dt(x1-xr(1),y1-xr(2),z1-xr(3),x4-xr(1),y4-xr(2),z4-xr(3),
     -           x3-xr(1),y3-xr(2),z3-xr(3))/vl
         a3 = dt(x1-xr(1),y1-xr(2),z1-xr(3),x2-xr(1),y2-xr(2),z2-xr(3),
     -           x4-xr(1),y4-xr(2),z4-xr(3))/vl
         a4 = 1.-a1-a2-a3
c
         am = min(a1,a2,a3,a4)
         if(am.gt.ac) then
c ...... this is the point !.
          ac = am
          l1 = j1
          l2 = j2
          l3 = j3
          l4 = j4
          p1 = a1
          p2 = a2
          p3 = a3
          p4 = a4
         endif
c
         if(ac.ge.0.) return
c
  105    continue
c
         istk         = istk+ncar
         stak(istk)   = ipoi
         stak(istk+1) = il
         do 102 id = 1,ndim
          stak(istk+1+id) = itr(xl(id))
  102    continue
         ipoi         = tr(ipoi)
         if(ipoi.ne.0) then
            id     = mod(il,ndim)+1
            amov   = 1./2.**((il/ndim)+1)
            il     = il+1
            if(id.le.3) then
               if(xl(id)     .gt.box(id)) ipoi = 0
            else
               if(xl(id)+amov.lt.box(id)) ipoi = 0
            endif
         endif
      endif
c
      goto                        1000
c
      end
c
c......................................................................
c
      subroutine alim4(coob,i1,i2,i3,i4,xel)
      real*8 coob(3,1)
      real   xel(1)
      x1     = coob(1,i1)
      y1     = coob(2,i1)
      z1     = coob(3,i1)
      x2     = coob(1,i2)
      y2     = coob(2,i2)
      z2     = coob(3,i2)
      x3     = coob(1,i3)
      y3     = coob(2,i3)
      z3     = coob(3,i3)
      x4     = coob(1,i4)
      y4     = coob(2,i4)
      z4     = coob(3,i4)
      xel(1) = min( x1, x2, x3, x4)
      xel(2) = min( y1, y2, y3, y4)
      xel(3) = min( z1, z2, z3, z4)
      xel(4) = max( x1, x2, x3, x4)
      xel(5) = max( y1, y2, y3, y4)
      xel(6) = max( z1, z2, z3, z4)
      return
      end
c
c...................................................................
c
      function rtr(n)
      equivalence (a,m)
      m   = n
      rtr = a
      return
      end
c
c...................................................................
c
      function itr(a)
      equivalence (b,n)
      b   = a
      itr = n
      return
      end
c
c*-----------------------------------------------------------*
c*    [namlen] determines the length of a character string   *
c*-----------------------------------------------------------*
c
      integer function namlen(filenam)
      character*80 filenam
c
      namlen = 0
      do 100 i = 80,1,-1
       if(filenam(i:i).eq.' ') goto 100
       namlen = i
       goto 200
  100 continue
  200 continue
c
      return
      end
c
c*--------------------------------------------------------------*
c*    [textread] outputs a prompt and reads a character string  *
c*--------------------------------------------------------------*
c
      character*80 function textread( prompt)
      character*(*) prompt
c
      write(*,'(/,a,$)') prompt
      read(*,'(a)') textread
c
      return
      end

