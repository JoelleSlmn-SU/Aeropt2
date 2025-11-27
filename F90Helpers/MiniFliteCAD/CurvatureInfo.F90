!
!  Copyright (C) 2017 College of Engineering, Swansea University
!
!  This file is part of the SwanSim FLITE suite of tools.
!
!  SwanSim FLITE is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  SwanSim FLITE is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with this SwanSim FLITE product.
!  If not, see <http://www.gnu.org/licenses/>.
!



!*******************************************************************************
!>
!!  curvature data and function
!<
!*******************************************************************************


!#define JWJ_PRINT

!>
!!
SUBROUTINE Curvature_Input()

  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  USE SurfaceCurvatureCADfix
  USE SurfaceCurvatureManager
  USE Spacing_Parameters
  USE SpacingStorageGen
  IMPLICIT NONE
  REAL*8  :: alpha, tMin, tMax, t, P(3), UBOT, VBOT, UTOP, VTOP, u, v, P2(3), dist, d, uv(2), P1(3)
  REAL*8  :: maxC, minC
  LOGICAL :: ex,ex2,ex3,ex4,ex5
  INTEGER :: i, is, icur, c, fileNum, j, k, ip, in1, in2, in3, in4
  INTEGER :: numCurveNodes,numPointPerCurve, numCurveT, netot,nptot,nqtot
  TYPE(IntQueueType) :: IC
  REAL*8 :: Ru(3),Rv(3),Ruv(3),Ruu(3),Rvv(3)
  TYPE(SurfaceCurvatureType)    :: CurvBack

  CHARACTER*255  entityName

!
  numPointPerCurve = 25
  fileNum = 2015
!
   IF(Curvature_Type==1)THEN
     INQUIRE(file = JobName(1:JobNameLength)//'.dat', EXIST=ex)
     IF(.NOT. ex)THEN
        WRITE(29,*) 'Error---  The file '//JobName(1:JobNameLength)//'.dat does NOT exist.'
        WRITE(*, *) 'Error--- The file '//JobName(1:JobNameLength)//'.dat does NOT exist.'
        INQUIRE(file = JobName(1:JobNameLength)//'.fbm', EXIST=ex)
        IF(ex)THEN
            Curvature_Type=2
        ELSE
            Curvature_Type=4
      ENDIF
     ELSE
           CALL SurfaceCurvature_Input(JobName,JobNameLength,Curvt)
           CALL SurfaceCurvature_BuildTangent(Curvt)
           NumCurves  = Curvt%NB_Curve
           NumRegions = Curvt%NB_Region
           IF(Debug_Display>2)THEN
              !--- checks the maximum stretching of the mesh
              ! CALL SurfaceCurvature_Check( Curvt)
           ENDIF
     ENDIF
   END IF

   IF(Curvature_Type==2 .OR. Curvature_Type==3)THEN
    CALL CADFIX_IsClientMode( ex )
    IF(.NOT. ex)THEN
       INQUIRE(file = JobName(1:JobNameLength)//'.fbm', EXIST=ex)
       IF(.NOT. ex)THEN
          WRITE(29,*) 'Error--- the file '//JobName(1:JobNameLength)//'.fbm does NOT exist.'
          WRITE(*, *) 'Error--- the file '//JobName(1:JobNameLength)//'.fbm does NOT exist.'
          CALL Error_Stop( ' Curvature_Input ::'  )
              INQUIRE(file = JobName(1:JobNameLength)//'.dat', EXIST=ex)
            IF(ex)THEN
               WRITE(29,*) '     But, since you have *.dat file, try setting Curvature_Type=1 then.'
               WRITE(*, *) '     But, since you have *.dat file, try setting Curvature_Type=1 then.'
           ENDIF
       ENDIF
       CALL CADFIX_LoadGeom( JobName )
       CALL CADFIX_AnalyseModel
       CALL CADFIX_GetNumCurves( NumCurves )
       CALL CADFIX_GetNumSurfaces( NumRegions )

    ENDIF
   ENDIF

   IF(Curvature_Type==4)THEN
    ! TOLG = Curvature_Factors(3)*1.d-4
    ! TOLG = 1.d-10
      TOLG = 1.d-4 *  GLOBE_GRIDSIZE
!     write(*,*) 'TOLG for reading:',TOLG,  GLOBE_GRIDSIZE
     !CALL CADFIX_IsClientMode( ex )
     !IF(.NOT. ex) THEN
      INQUIRE(file = JobName(1:JobNameLength)//'.igs', EXIST=ex) !IGES File
      IF (ex) THEN
        WRITE(29,*) 'Reading '//JobName(1:JobNameLength)//'.igs'
        WRITE(*, *) 'Reading '//JobName(1:JobNameLength)//'.igs'
        CALL OCCT_LoadIGES(JobName(1:LEN_TRIM(JobName))//'.igs',TOLG)
      ELSE
        INQUIRE(file = JobName(1:JobNameLength)//'.stp', EXIST=ex2) !STEP File
        IF (ex2) THEN
           WRITE(29,*) 'Reading '//JobName(1:JobNameLength)//'.stp'
           WRITE(*, *) 'Reading '//JobName(1:JobNameLength)//'.stp'
           !CALL OCCT_LoadSTEP('cube.stp')
           CALL OCCT_LoadSTEP(JobName(1:LEN_TRIM(JobName))//'.stp',TOLG )
        ELSE
           INQUIRE(file = JobName(1:JobNameLength)//'.iges',EXIST=ex3) !IGES file
           IF (ex3) THEN
              WRITE(29,*) 'Reading '//JobName(1:JobNameLength)//'.iges'
              WRITE(*,*)  'Reading '//JobName(1:JobNameLength)//'.iges'
              CALL OCCT_LoadIGES(JobName(1:LEN_TRIM(JobName))//'.iges',TOLG)
           ELSE
              INQUIRE(file = JobName(1:JobNameLength)//'.step',EXIST=ex4) !STEP file
              IF (ex4) THEN
                 WRITE(29,*) 'Reading '//JobName(1:JobNameLength)//'.step'
                 WRITE(*,*)  'Reading '//JobName(1:JobNameLength)//'.step'
                 CALL OCCT_LoadSTEP(JobName(1:LEN_TRIM(JobName))//'.step',TOLG )
              ELSE
                INQUIRE(file = JobName(1:JobNameLength)//'.brep',EXIST=ex5) !STEP file
                IF (ex5) THEN
                   WRITE(29,*) 'Reading '//JobName(1:JobNameLength)//'.brep'
                   WRITE(*,*)  'Reading '//JobName(1:JobNameLength)//'.brep'
                   CALL OCCT_LoadBRep(JobName(1:LEN_TRIM(JobName))//'.brep',TOLG )
                ELSE
                   CALL Error_Stop( ' Curvature_Input ::'  )
                   INQUIRE(file = JobName(1:JobNameLength)//'.dat', EXIST=ex)
                   IF(ex)THEN
                     WRITE(29,*) '     But, since you have *.dat file, try setting Curvature_Type=1 then.'
                     WRITE(*, *) '     But, since you have *.dat file, try setting Curvature_Type=1 then.'
                   ENDIF
                ENDIF
              ENDIF
           ENDIF
        ENDIF
      ENDIF
      !CALL CADFIX_LoadGeom( JobName )
      !Call CADFIX_GetTolerance( tolg )
      !CALL CADFIX_AnalyseModel
      CALL OCCT_GetNumCurves( NumCurves )
      CALL OCCT_GetNumSurfaces( NumRegions )

      ALLOCATE(SurfMaxUV(NumRegions,8))
      DO is = 1,NumRegions
        CALL OCCT_GetSurfaceUVBox(is, UBOT, VBOT, UTOP, VTOP)
        SurfMaxUV(is,1) = UBOT
        SurfMaxUV(is,2) = VBOT
        SurfMaxUV(is,3) = UTOP
        SurfMaxUV(is,4) = VTOP

      !We now want to calculate the width and height of this surface
        dist = 0.0d0
        DO k = 1,numPointPerCurve
          d = 0.0d0
          DO j = 2,numPointPerCurve
            !First the last point
            uv(1) = UBOT + (REAL(j-2)/REAL(numPointPerCurve-1))*(UTOP-UBOT)
            uv(2) = VBOT + (REAL(k-1)/REAL(numPointPerCurve-1))*(VTOP-VBOT)
            CALL OCCT_GetUVPointInfoAll(is,uv(1),uv(2),P,Ru,Rv,Ruv,Ruu,Rvv)
          !Now the next
            uv(1) = UBOT + (REAL(j-1)/REAL(numPointPerCurve-1))*(UTOP-UBOT)
            uv(2) = VBOT + (REAL(k-1)/REAL(numPointPerCurve-1))*(VTOP-VBOT)
            CALL OCCT_GetUVPointInfoAll(is,uv(1),uv(2),P2,Ru,Rv,Ruv,Ruu,Rvv)
            d = d + Geo3D_Distance(P,P2)
          END DO
          dist = MAX(d,dist)
        END DO

        SurfMaxUV(is,5) = dist
        SurfMaxUV(is,7) = 1.d0/dist

        dist = 0.0d0
        DO j = 1,numPointPerCurve
          d = 0.0d0
          DO k = 2,numPointPerCurve
          !First the last point
            uv(1) = UBOT + (REAL(j-1)/REAL(numPointPerCurve-1))*(UTOP-UBOT)
            uv(2) = VBOT + (REAL(k-2)/REAL(numPointPerCurve-1))*(VTOP-VBOT)
            CALL OCCT_GetUVPointInfoAll(is,uv(1),uv(2),P,Ru,Rv,Ruv,Ruu,Rvv)
          !Now the next
            uv(1) = UBOT + (REAL(j-1)/REAL(numPointPerCurve-1))*(UTOP-UBOT)
            uv(2) = VBOT + (REAL(k-1)/REAL(numPointPerCurve-1))*(VTOP-VBOT)
            CALL OCCT_GetUVPointInfoAll(is,uv(1),uv(2),P2,Ru,Rv,Ruv,Ruu,Rvv)
            d = d + Geo3D_Distance(P,P2)
          END DO
          dist = MAX(d,dist)
        END DO

        SurfMaxUV(is,6) = dist
        SurfMaxUV(is,8) = 1.d0/dist

      END DO


      DO is=1,NumRegions
         CALL GetRegionName(is,entityName)
      END DO

      IF (Output_dat==1) THEN

        write(fileNum+3,'(a)') ' Geometry'
        write(fileNum+3,'(2I7)') NumCurves,NumRegions
        write(fileNum+3,'(a)') ' Curves'
           !Get max and min U
        do c = 1 , NumCurves
         CALL OCCT_GetLineTBox(c,tMin,tMax)
         write(fileNum+3,'(2I8)') c,1
         write(fileNum+3,'(2I8)') numPointPerCurve
         DO j = 1,numPointPerCurve
          t = tMin + (REAL(j-1)/(REAL(numPointPerCurve-1)))*(tMax-tMin)
          CALL OCCT_GetLineXYZFromT(c,t,P)
          WRITE(fileNum+3,'(3E15.7)')P
         END DO
        END DO

        write(fileNum+3,'(a)') ' Surfaces'
        DO is = 1,NumRegions
         !Now sample the region
         CALL GetRegionUVBox(is, UBOT, VBOT, UTOP, VTOP)
         write(fileNum+3,'(2I8)') is,1
         write(fileNum+3,'(2I8)') numPointPerCurve, numPointPerCurve

         DO j = 1,numPointPerCurve
          DO k = 1,numPointPerCurve
           u = UBOT + (REAL(j-1)/(REAL(numPointPerCurve-1)))*(UTOP-UBOT)
           v = VBOT + (REAL(k-1)/(REAL(numPointPerCurve-1)))*(VTOP-VBOT)
           CALL GetUVPointInfo0(is,u,v,P)
           ip = (is-1)*numPointPerCurve**2+(j-1)*numPointPerCurve+k
           WRITE(fileNum+3,'(3E15.7)')P
          END DO
         END DO

        END DO

        write(fileNum+3,'(a)') ' Mesh Generation'
        write(fileNum+3,'(4I7)') NumCurves,NumRegions,0,0
        write(fileNum+3,'(a)' )' Segment Curves '
        do c = 1 , NumCurves
         write(fileNum+3,'(3I8)') c,c,1
        END DO

        write(fileNum+3,'(a)') ' Surfaces'
        DO is = 1,NumRegions
         write(fileNum+3,'(4I8)') is,is,1,1
         CALL GetRegionCurveList(is,IC)
         write(fileNum+3,'(2I8)') ic%numNodes
         write(fileNum+3,'(8I8)') (IC%Nodes(icur),icur=1,ic%numNodes)
        END DO

        numCurveT = 0
        DO is = 1,NumRegions
         CALL GetRegionCurveList(is,IC)
         numCurveT = numCurveT+IC%numNodes
         print *,'is:',is,' #Curves:',IC%numNodes,numCurveT
        end do

        nqtot  = 2*NumRegions*(numPointPerCurve-1)**2
        netot  = nqtot+numCurveT*(numPointPerCurve-1)
        nptot  = NumRegions*numPointPerCurve**2+numCurveT*numPointPerCurve
        WRITE(fileNum,'(8I9)') netot,nptot

    !   !Test by writing out surface and curves

        DO is = 1,NumRegions
         !Now sample the region
         CALL GetRegionUVBox(is, UBOT, VBOT, UTOP, VTOP)
         DO j = 1,numPointPerCurve
          DO k = 1,numPointPerCurve
           u = UBOT + (REAL(j-1)/(REAL(numPointPerCurve-1)))*(UTOP-UBOT)
           v = VBOT + (REAL(k-1)/(REAL(numPointPerCurve-1)))*(VTOP-VBOT)
           CALL GetUVPointInfo0(is,u,v,P)
           ip = (is-1)*numPointPerCurve**2+(j-1)*numPointPerCurve+k
           WRITE(fileNum,'(i9,5E15.7)')ip,P,u,v
          END DO
         END DO
         print *,' Surface: ',is,' # points:' ,ip
        END DO
        numCurveT = 0
        DO is = 1,NumRegions
         !Get the list of curves surrounding this region
         CALL GetRegionCurveList(is,IC)
         !We are going to plot curves
         DO icur = 1,IC%numNodes
          !Get each curve number
          c = IC%Nodes(icur)
          !Get max and min U
          CALL OCCT_GetLineTBox(c,tMin,tMax)
          DO j = 1,numPointPerCurve
           t = tMin + (REAL(j-1)/(REAL(numPointPerCurve-1)))*(tMax-tMin)
           CALL OCCT_GetLineXYZFromT(c,t,P)
           ip = NumRegions*numPointPerCurve**2+numCurveT*numPointPerCurve+(icur-1)*numPointPerCurve+j
           WRITE(fileNum,'(i9,5E15.7)')ip,P,t
          END DO
         END DO
         numCurveT = numCurveT + iC%numNodes
         print *,' Surface + curves: ',is, IC%numNodes,' # points:' ,ip

        END DO

        DO is = 1,NumRegions
         DO j = 1,numPointPerCurve-1
          DO k = 1,numPointPerCurve-1
           in1 = (is-1)*numPointPerCurve**2+(j-1)*numPointPerCurve+k
           in2 = (is-1)*numPointPerCurve**2+(j-1)*numPointPerCurve+k+1
           in3 = (is-1)*numPointPerCurve**2+ j*numPointPerCurve+k+1
           in4 = (is-1)*numPointPerCurve**2+ j*numPointPerCurve+k
           ip = 2*(is-1)*(numPointPerCurve-1)**2+2*(j-1)*(numPointPerCurve-1)+2*k-1
           WRITE(fileNum,'(7I9)')ip,in1,in2,in3,is,is
           WRITE(fileNum,'(7I9)')ip+1,in3,in4,in1,is,is
          END DO
         END DO
         print *,' Surface: ',is,' # elements:' ,ip
        END DO

        numCurveT = 0
        DO is = 1,NumRegions
         !Get the list of curves surrounding this region
         CALL GetRegionCurveList(is,IC)
         !We are going to plot curves
         DO icur = 1,IC%numNodes
          DO j = 1,numPointPerCurve-1
           ip = numCurveT*(numPointPerCurve-1)+2*NumRegions*(numPointPerCurve-1)**2+(icur-1)*(numPointPerCurve-1)+j
           in1 = numCurveT*numPointPerCurve+NumRegions*numPointPerCurve**2+(icur-1)*numPointPerCurve+j
           WRITE(fileNum,'(7I9)')ip,in1,in1+1,in1+1,is*1000+icur,is*1000+icur
          END DO
         END DO
         numCurveT = numCurveT + iC%numNodes

         print *,' Surface + curves: ',is, IC%numNodes,' # elements:' ,ip
        END DO

      END IF
      !       ! Output the surface trianglations if debugging
      !       IF(Debug_Display>3) THEN
      !         DO is=1,NumRegions
      !             CALL OCCT_WriteSurfVis(is)
      !          END DO
      !       ENDIF
      !       IF(Debug_Display>3) THEN
      !          DO is=1,NumCurves
      !             CALL OCCT_WriteCurvVis(is)
      !          ENDDO
      !       end if


      close(2015)
      close(2015+3)
    ENDIF

    IF(Curvature_Type==3)THEN
     alpha = MIN(Curvature_Factors(1), 0.1d0)
     CALL SurfaceCurvature_from_CADfix(alpha, Curvt, 11)
     CALL CADFIX_Exit()

     CALL SurfaceCurvature_Output(JobName,JobNameLength, Curvt)
     CALL SurfaceCurvature_BuildTangent(Curvt)
     Curvature_Type = 1
    ENDIF

    ALLOCATE(RegionMark(NumRegions), CurveMark(NumCurves))
    IF(ListGs%numNodes==0 .AND. LEN_TRIM(NameGs)==0)THEN
        RegionMark(:) = 1
    ELSE
        RegionMark(:) = 0
        DO i = 1, ListGs%numNodes
           is = ListGs%Nodes(i)
           IF(is>0 .AND. is<=NumRegions) RegionMark(is) = 1
        ENDDO
        DO is = 1, NumRegions
           CALL GetRegionName(is, entityName)
           IF(CHAR_Contains(NameGs,entityName)) RegionMark(is) = 1
        ENDDO
    ENDIF

    CurveMark(:)  = 0
    DO is = 1, NumRegions
     IF(RegionMark(is) == 1)THEN
        CALL GetRegionCurveList(is, IC)
        CurveMark(IC%Nodes(1:IC%numNodes)) = 1
     ENDIF
    ENDDO

    IF(Curvature_Type==4) THEN
    !Now output info on the max and min curve lengths to help people set curvature control
      maxC = 0.0d0
      minC = HUGE(0.0d0)
      WRITE(*,*) '================================='
      WRITE(29,*) '================================='
      WRITE(*,*) 'Measuring curve lengths',TOLG
      WRITE(29,*) 'Measuring curve lengths'
      DO c = 1,NumCurves
       !Get max and min U
       CALL OCCT_GetLineTBox(c,tMin,tMax)

       d = 0.0d0
       CALL OCCT_GetLineXYZFromT(c,tMin,P1)

       DO j = 2,numPointPerCurve
         t = tMin + (REAL(j-1)/(numPointPerCurve-1))*(tMax-tMin)
         CALL OCCT_GetLineXYZFromT(c,t,P2)

         d = d + Geo3D_Distance(P1,P2)
         P1(:) = P2(:)
       END DO

      !if(d.le.TOLG) then
      !  CurveMark(c) = 0
      !  print *,'   '
      !  print *,' *** Curve:',c,' is removed as it is shorter than the tolerece'
      !  print *,'   '
      !else

         maxC = MAX(d, maxC)
         minC = MIN(d, minC)
      !end if

      END DO

      WRITE(*,*)'Max Curve Length: ',maxC
      WRITE(*,*)'Min Curve Length: ',minC
      WRITE(29,*)'Max Curve Length: ',maxC
      WRITE(29,*)'Min Curve Length: ',minC

      WRITE(*,*) '================================='
      WRITE(29,*) '================================='

    END IF

    CALL IntQueue_Clear(IC)

END SUBROUTINE Curvature_Input

!>
!!
SUBROUTINE GetRegionUVBox(RegionID, UBOT, VBOT, UTOP, VTOP)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  REAL*8,  INTENT(OUT) :: UBOT, VBOT, UTOP, VTOP
  IF(Curvature_Type==1)THEN
     UBOT = 0.d0
     VBOT = 0.d0
     UTOP = Curvt%Regions(RegionID)%numNodeU - 1
     VTOP = Curvt%Regions(RegionID)%numNodeV - 1
  ELSE IF(Curvature_Type==4) THEN
     !CALL OCCT_GetSurfaceUVBox( RegionID, UBOT, VBOT, UTOP, VTOP )
    UBOT = SurfMaxUV(RegionID,1)
    VBOT = SurfMaxUV(RegionID,2)
    UTOP = SurfMaxUV(RegionID,3)
    VTOP = SurfMaxUV(RegionID,4)

    IF (0==1) THEN !Normalise paramterspace
     UTOP = ((UTOP - UBOT)/(UTOP-UBOT))*SurfMaxUV(RegionID,5)
     VTOP = ((VTOP - VBOT)/(VTOP-VBOT))*SurfMaxUV(RegionID,6)
     UBOT = 0.0d0
     VBOT = 0.0d0
    END IF

  ELSE
       CALL CADFIX_GetSurfaceUVBox( RegionID, UBOT, VBOT, UTOP, VTOP )
  ENDIF
END SUBROUTINE GetRegionUVBox

!>
!!
SUBROUTINE GetRegionUVmax(RegionID, IUTOP, IVTOP)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  INTEGER, INTENT(OUT) :: IUTOP, IVTOP
  REAL*8 :: UBOT, VBOT, UTOP, VTOP
  IF(Curvature_Type==1)THEN
     IUTOP = Curvt%Regions(RegionID)%numNodeU
     IVTOP = Curvt%Regions(RegionID)%numNodeV
  ELSE IF(Curvature_Type==4) THEN
     CALL GetRegionUVBox( RegionID, UBOT, VBOT, UTOP, VTOP )
     IUTOP = UTOP - UBOT + 1.5
     IVTOP = VTOP - VBOT + 1.5
  ELSE
        CALL CADFIX_GetSurfaceUVBox( RegionID, UBOT, VBOT, UTOP, VTOP )
     IUTOP = UTOP - UBOT + 1.5
        IVTOP = VTOP - VBOT + 1.5
  ENDIF
END SUBROUTINE GetRegionUVmax

!>
!!
SUBROUTINE GetRegionName(RegionID, entityName)
  USE control_Parameters
  USE surface_Parameters
  USE occt_fortran  !SPW

  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  CHARACTER(LEN=*), INTENT(OUT)  :: entityName
  IF(Curvature_Type==1)THEN
     entityName = Curvt%Regions(RegionID)%theName
  ELSE IF(Curvature_Type==4) THEN
     CALL OCCT_GetSurfaceName( RegionID, entityName )
  ELSE
     CALL CADFIX_GetSurfaceName( RegionID, entityName )
  ENDIF
  IF(LEN_TRIM(entityName)==0) entityName = INT_to_CHAR (RegionID, 4)
END SUBROUTINE GetRegionName


!>
!!
SUBROUTINE GetRegionGeoType(RegionID, GeoType)
  USE control_Parameters
  USE surface_Parameters
  USE occt_fortran  !SPW

  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  INTEGER, INTENT(OUT) :: GeoType
  Integer              :: i,j
  IF(Curvature_Type==1)THEN
     GeoType = Curvt%Regions(RegionID)%GeoType
  ELSE IF(Curvature_Type==4) THEN
     GeoType = 1
  ELSE
     GeoType = 1
     CALL CADFIX_GetSurfaceTopoType( RegionID, GeoType)
     IF(GeoType/=-1) GeoType = 1
  ENDIF
  DO i = 1, ListQd%numNodes
    j = ListQd%Nodes(i)
    IF(j.eq.RegionID) then
      GeoType = -1
      exit
    end if
  END DO
END SUBROUTINE GetRegionGeoType


!>
!!
SUBROUTINE GetRegionNumCurves(RegionID, nc)
  USE control_Parameters
  USE occt_fortran  !SPW
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  INTEGER, INTENT(OUT) :: nc
  IF(Curvature_Type==1)THEN
     nc = Curvt%Regions(RegionID)%numCurve
  ELSE IF(Curvature_Type==4) THEN
     CALL OCCT_GetSurfaceNumCurves(RegionID, nc)
  ELSE
     CALL CADFIX_GetSurfaceNumCurves(RegionID, nc)
  ENDIF
END SUBROUTINE GetRegionNumCurves

!>
!!
SUBROUTINE GetRegionCurveList(RegionID, IC)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  USE Queue
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  TYPE(IntQueueType), INTENT(INOUT) :: IC
  INTEGER :: nc, List(10000)
  IF(Curvature_Type==1)THEN
     CALL IntQueue_Set(IC, Curvt%Regions(RegionID)%numCurve, Curvt%Regions(RegionID)%IC)
  ELSE IF(Curvature_Type == 4) THEN
     CALL OCCT_GetSurfaceCurves(RegionID, nc, List)
     IF(nc>10000) CALL Error_Stop (' GetRegionCurveList: nc>10000 ')
     CALL IntQueue_Set(IC, nc, List)
  ELSE
    CALL CADFIX_GetSurfaceCurves(RegionID, nc, List)
      IF(nc>10000) CALL Error_Stop (' GetRegionCurveList: nc>10000 ')
      CALL IntQueue_Set(IC, nc, List)
  ENDIF
END SUBROUTINE GetRegionCurveList



!>
!!
SUBROUTINE GetRegionXYZFromT(CurveID, t, P)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: CurveID
  REAL*8, INTENT(IN)   :: t
  REAL*8, INTENT(OUT)  :: P(3)

  IF (Curvature_Type==1) THEN
    CALL CurveType_Interpolate(Curvt%Curves(CurveID), 0, t, P)
  ELSE IF (Curvature_Type==4) THEN
    CALL OCCT_GetLineXYZFromT(CurveID, t, P)
  ELSE
    CALL CADFIX_GetLineXYZFromT(CurveID, t, P)
  END IF

END SUBROUTINE GetRegionXYZFromT

!>
!!
SUBROUTINE GetUVPointInfo0(RegionID, u, v, P)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  REAL*8,  INTENT(IN)  :: u, v
  REAL*8,  INTENT(OUT) :: P(3)
  REAL*8 :: uv(2), Pu(3), Pv(3), Puv(3),Puu(3),Pvv(3),UBOT, VBOT, UTOP, VTOP, ut, vt
  IF(Curvature_Type==1)THEN
     CALL RegionType_Interpolate(Curvt%Regions(RegionID), 0, u, v, P)
  ELSE IF(Curvature_Type==4) THEN
    IF (0==1) THEN  !Normalise parameter space
      UBOT = SurfMaxUV(RegionID,1)
      VBOT = SurfMaxUV(RegionID,2)
      UTOP = SurfMaxUV(RegionID,3)
      VTOP = SurfMaxUV(RegionID,4)
      ut = ((u*SurfMaxUV(RegionID,7))*(UTOP-UBOT) + UBOT)
      vt = ((v*SurfMaxUV(RegionID,8))*(VTOP-VBOT) + VBOT)
      CALL OCCT_GetUVPointInfoAll(RegionID,ut,vt,P,Pu,Pv,Puv,Puu,Pvv)
    ELSE
      CALL OCCT_GetUVPointInfoAll(RegionID,u,v,P,Pu,Pv,Puv,Puu,Pvv)
    END IF

  ELSE
    uv(:) = (/u,v/)
    CALL CADFIX_GetXYZFromUV1( RegionID, uv, P )
  ENDIF
END SUBROUTINE GetUVPointInfo0

!>
!!
SUBROUTINE GetUVPointInfo1(RegionID, u, v, P, Pu, Pv, Puv)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID
  REAL*8,  INTENT(IN)  :: u, v
  REAL*8,  INTENT(OUT) :: P(3), Pu(3), Pv(3), Puv(3)
  REAL*8 :: info(18),Puu(3),Pvv(3),UBOT, VBOT, UTOP, VTOP, ut, vt
#ifdef JWJ_PRINT
  write(99,*) 'Region',RegionID, u, v
#endif
  IF(Curvature_Type==1)THEN
     CALL RegionType_Interpolate(Curvt%Regions(RegionID), 1, u, v, P, Pu, Pv, Puv)
  ELSE IF(Curvature_Type==4) THEN

    IF (0==1) THEN
      UBOT = SurfMaxUV(RegionID,1)
      VBOT = SurfMaxUV(RegionID,2)
      UTOP = SurfMaxUV(RegionID,3)
      VTOP = SurfMaxUV(RegionID,4)
      ut = ((u*SurfMaxUV(RegionID,7))*(UTOP-UBOT) + UBOT)
      vt = ((v*SurfMaxUV(RegionID,8))*(VTOP-VBOT) + VBOT)
      CALL OCCT_GetUVPointInfoAll(RegionID,ut,vt,P,Pu,Pv,Puv,Puu,Pvv)
    ELSE
      CALL OCCT_GetUVPointInfoAll(RegionID,u,v,P,Pu,Pv,Puv,Puu,Pvv)
    END IF



  ELSE
       CALL CADFIX_GetUVPointInfo( RegionID, u, v, info )
     P(1:3)   = info(1:3)
     Pu(1:3)  = info(4:6)
     Pv(1:3)  = info(7:9)
     Puv(1:3) = info(10:12)
  ENDIF
#ifdef JWJ_PRINT
  write(99,*) 'XYZ',P(1),P(2),P(3)
  write(99,*) 'dU',Pu(1),Pu(2),Pu(3)
  write(99,*) 'dV',Pv(1),Pv(2),Pv(3)
  write(99,*) 'dUV',Puv(1),Puv(2),Puv(3)
#endif
END SUBROUTINE GetUVPointInfo1


!>
!!
SUBROUTINE GetUVFromXYZ(RegionID, npl, xyz, uv, TOLG)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  USE SurfaceCurvatureManager
  IMPLICIT NONE
  INTEGER, INTENT(IN)  :: RegionID, npl
  REAL*8,  INTENT(IN)  :: xyz(3,*), TOLG
  REAL*8,  INTENT(OUT) :: uv(2,*)
  REAL*8 :: drm_max, UBOT, VBOT, UTOP, VTOP
  INTEGER :: np, ip

  IF(Curvature_Type==1)THEN
     CALL RegionType_GetUVFromXYZ(Curvt%Regions(RegionID), npl, xyz, uv, TOLG, drm_max)
     IF( drm_max > TOLG ) THEN
        WRITE(29,'(/,a,    /)') ' LOCUV >        !!! *** WARNING *** !!!           '
        WRITE(29,'(  a,e12.5)') '        The maximum distance between the target is: ', drm_max
        WRITE(29,'(  a      )') '        This is BIGGER than the tolerance used by '
        WRITE(29,'(  a,e12.5)') '        SURFACE for coincident points which is:   ', TOLG
        WRITE(29,'(  a      )') '        The run continues but you are advised to  '
        WRITE(29,'(  a      )') '        check your input geometry data for errors '
     ENDIF
  ELSE IF(Curvature_Type==4) THEN
    CALL OCCT_GetUVFromXYZ( RegionID, npl, xyz, uv, TOLG)


     IF (0==1) THEN
      UBOT = SurfMaxUV(RegionID,1)
      VBOT = SurfMaxUV(RegionID,2)
      UTOP = SurfMaxUV(RegionID,3)
      VTOP = SurfMaxUV(RegionID,4)

       DO ip = 1,npl
        uv(1,ip) = ((uv(1,ip)- UBOT)/(UTOP-UBOT))*SurfMaxUV(RegionID,5)
        uv(2,ip) = ((uv(2,ip)- VBOT)/(VTOP-VBOT))*SurfMaxUV(RegionID,6)
       END DO

     END IF

  ELSE
     CALL CADFIX_GetUVFromXYZ( RegionID, npl, xyz, uv )
  ENDIF
END SUBROUTINE GetUVFromXYZ

!>
!!
SUBROUTINE ProjectToRegionFromXYZ(RegionID, xyz, uv, TOLG, onSurf, drm)
  USE occt_fortran  !SPW
  USE control_Parameters
  USE surface_Parameters
  USE SurfaceCurvatureManager
  IMPLICIT NONE
  INTEGER, INTENT(IN)    :: RegionID
  REAL*8,  INTENT(INOUT) :: xyz(3), uv(2)
  REAL*8,  INTENT(IN)    :: TOLG
  INTEGER, INTENT(in)    :: onSurf
  REAL*8,  INTENT(OUT)   :: drm
  REAL*8 :: Pt(3),UBOT, VBOT, UTOP, VTOP
  INTEGER :: np
  IF(Curvature_Type==1)THEN
     CALL RegionType_GetUVFromXYZ1(Curvt%Regions(RegionID), xyz, uv, TOLG, onSurf, drm, Pt)
  ELSE IF(Curvature_Type==4) THEN
     !WRITE(*,*)xyz
     CALL OCCT_GetUVFromXYZ(  RegionID, 1, xyz, uv, TOLG )
     CALL GetUVPointInfo0( RegionID, uv(1),uv(2), Pt )
     drm = Geo3D_Distance(xyz, Pt)

     IF (0==1) THEN
      UBOT = SurfMaxUV(RegionID,1)
      VBOT = SurfMaxUV(RegionID,2)
      UTOP = SurfMaxUV(RegionID,3)
      VTOP = SurfMaxUV(RegionID,4)
      uv(1) = ((uv(1)- UBOT)/(UTOP-UBOT))*SurfMaxUV(RegionID,5)
      uv(2) = ((uv(2)- VBOT)/(VTOP-VBOT))*SurfMaxUV(RegionID,6)

     END IF

  ELSE
     CALL CADFIX_GetUVFromXYZ(  RegionID, 1, xyz, uv )

     CALL CADFIX_GetXYZFromUV1( RegionID, uv, Pt )
       drm = Geo3D_Distance(xyz, Pt)
  ENDIF
  xyz(:) = Pt(:)
END SUBROUTINE ProjectToRegionFromXYZ




SUBROUTINE OCCT_To_FLITE(Curv)
  !This converts a CAD file to a .dat
  USE occt_fortran
  USE SurfaceCurvature
  USE SurfaceCurvatureManager
  USE Spacing_Parameters
  USE Queue

  TYPE(SurfaceCurvatureType), INTENT(OUT) :: Curv

  INTEGER :: NBs, maxNodes,s,i,j,k, NCs, c, iu, iv,numU, numV
  TYPE(IntQueueType) :: IC
  REAL*8 :: UBOT, VBOT, UTOP, VTOP,u,v,P(3), P1(3), P2(3), P3(3)
  REAL*8 :: t, tMin, tMax, dist, uv(2)

  LOGICAL :: complete, flag
  TYPE(Point3dQueueType) :: param, physical, paramU, paramV



  CALL OCCT_GetNumSurfaces( NBs )

  Curv%NB_Region = NBs
  ALLOCATE(Curv%Regions(Curv%NB_Region))

  CALL OCCT_GetNumCurves(NCs)

  ALLOCATE(Curv%Curves(NCs))
  Curv%NB_Curve = NCs


  numCuts = 25

  DO s = 1,NBs

    !Set up basic info
    Curv%Regions(s)%ID = s
    Curv%Regions(s)%TopoType = 1
    Curv%Regions(s)%numNodeU = numCuts
    Curv%Regions(s)%numNodeV = numCuts

    ALLOCATE(Curv%Regions(s)%Posit(3,numCuts,numCuts))

    !Get the list of curves surrounding this region
    CALL GetRegionCurveList(s,IC)
    Curv%Regions(s)%numCurve = IC%numNodes
    ALLOCATE(Curv%Regions(s)%IC(IC%numNodes))
    DO i = 1,IC%numNodes
      Curv%Regions(s)%IC(i) = IC%Nodes(i)
    END DO
    CALL IntQueue_Clear(IC)

    !Now sample the region
     CALL GetRegionUVBox(s, UBOT, VBOT, UTOP, VTOP)
     DO j = 1,numCuts
       DO k = 1,numCuts
         u = UBOT + (REAL(j-1)/REAL(numCuts-1.0d0))*(UTOP-UBOT)
         v = VBOT + (REAL(k-1)/REAL(numCuts-1.0d0))*(VTOP-VBOT)

         CALL GetUVPointInfo0(s,u,v,P)
         Curv%Regions(s)%Posit(:,j,k) = P(:)


       END DO
     END DO

  END DO


  DO c = 1,NCs

    Curv%Curves(c)%ID       = c
    Curv%Curves(c)%TopoType = 1
    Curv%Curves(c)%numNodes = numCuts

    ALLOCATE(Curv%Curves(c)%Posit(3,numCuts))
    CALL OCCT_GetLineTBox(c,tMin,tMax)
    DO j = 1,numCuts
     t = tMin + (REAL(j-1)/REAL(numCuts-1.0d0))*(tMax-tMin)
     CALL OCCT_GetLineXYZFromT(c,t,P)
     Curv%Curves(c)%Posit(:,j) = P(:)
    END DO



  END DO

  ! DO c = 1,NCs

  !   !We are going to pick the number of nodes in this curve using curvature control
  !   WRITE(*,*)'Curve ',c
  !   CALL Point3dQueue_Clear(param)
  !   CALL Point3dQueue_Clear(physical)

  !   !First get the initial and final node
  !   CALL OCCT_GetLineTBox(c,tMin,tMax)

  !   CALL OCCT_GetLineXYZFromT(c,tMin,P)
  !   P1(:) = 0.0d0
  !   P1(1) = tMin
  !   CALL Point3dQueue_Push(physical, P)
  !   CALL Point3dQueue_Push(param, P1)

  !   CALL OCCT_GetLineXYZFromT(c,tMax,P)
  !   P1(:) = 0.0d0
  !   P1(1) = tMax
  !   CALL Point3dQueue_Push(physical, P)
  !   CALL Point3dQueue_Push(param, P1)

  !   !Variable to keep track of progress
  !   complete = .FALSE.

  !   DO WHILE (.NOT.complete)
  !     flag = .TRUE.
  !     DO i = 1,physical%numNodes-1
  !       !Project midpoint of two nodes to curve
  !       P2(:) = 0.5d0*(physical%Pt(:,i)+physical%Pt(:,i+1))
  !       CALL OCCT_GetUFromXYZ(c,P2,t)
  !       CALL OCCT_GetLineXYZFromT(c,t,P3)
  !       dist = Geo3D_Distance(P2,P3)
  !       !WRITE(*,*) c, i, dist
  !       IF (dist.GT.BGSpacing%MinSize) THEN
  !         t = 0.5d0*(param%Pt(1,i)+param%Pt(1,i+1))
  !         CALL OCCT_GetLineXYZFromT(c,t,P3)
  !         P1(:) = 0.0d0
  !         P1(1) = t
  !         CALL Point3dQueue_Add(physical, i+1, P3)
  !         CALL Point3dQueue_Add(param, i+1, P1)
  !         flag = .FALSE.
  !         EXIT
  !       END IF
  !     END DO

  !     !Check for completion
  !     IF (flag.OR.(physical%numNodes.EQ.maxNodes)) THEN
  !       complete = .TRUE.
  !       EXIT
  !     END IF
  !   END DO

  !   Curv%Curves(c)%ID       = c
  !   Curv%Curves(c)%TopoType = 1
  !   Curv%Curves(c)%numNodes = param%numNodes

  !   ALLOCATE(Curv%Curves(c)%Posit(3,param%numNodes))

  !   DO j = 1,param%numNodes
  !    t = param%Pt(1,j)
  !    CALL OCCT_GetLineXYZFromT(c,t,P)
  !    Curv%Curves(c)%Posit(:,j) = P(:)
  !   END DO



  ! END DO

  ! !Might be better to do this using curvature?



  ! DO s = 1,NBs
  !   WRITE(*,*) 'Surface ',s
  !   !Set up basic info
  !   Curv%Regions(s)%ID = s
  !   Curv%Regions(s)%TopoType = 1



  !   !Now we are going to pick the number of nodes in each direction
  !   !using curvature control
  !   CALL Point3dQueue_Clear(paramU)
  !   CALL Point3dQueue_Clear(paramV)
  !   CALL GetRegionUVBox(s, UBOT, VBOT, UTOP, VTOP)

  !   P1(:) = 0.0d0
  !   P1(1) = UBOT
  !   CALL Point3dQueue_Push(paramU, P1)
  !   P1(:) = 0.0d0
  !   P1(1) = UTOP
  !   CALL Point3dQueue_Push(paramU, P1)
  !   numU = 2

  !   P1(:) = 0.0d0
  !   P1(1) = VBOT
  !   CALL Point3dQueue_Push(paramV, P1)
  !   P1(:) = 0.0d0
  !   P1(1) = VTOP
  !   CALL Point3dQueue_Push(paramV, P1)
  !   numV = 2

  !   !Variable to keep track of progress
  !   complete = .FALSE.

  !   DO WHILE (.NOT.complete)
  !     flag = .TRUE.
  !     !First do the u's
  !     WRITE(*,*)paramU%numNodes,paramV%numNodes
  !     IF (paramV%numNodes<maxNodes) THEN
  !       uloop: DO iu = 1,paramU%numNodes
  !         !Each v is going to be at the same u
  !         u = paramU%Pt(1,iu)

  !         CALL Point3dQueue_Clear(physical)
  !         !First fill the physical space array
  !         DO i = 1,paramV%numNodes
  !           v = paramV%Pt(1,i)
  !           !WRITE(*,*) u,v
  !           CALL GetUVPointInfo0(s,u,v,P)
  !           CALL Point3dQueue_Push(physical, P)
  !         END DO

  !         !Now check for refinement
  !         DO i = 1,physical%numNodes-1
  !           !Project midpoint of two nodes to curve
  !           P2(:) = 0.5d0*(physical%Pt(:,i)+physical%Pt(:,i+1))
  !           CALL ProjectToRegionFromXYZ(s,P2,uv,TOLG,0,dist)
  !           !WRITE(*,*)s,iu,i,dist
  !           IF (dist.GT.BGSpacing%MinSize) THEN
  !             v = 0.5d0*(paramV%Pt(1,i)+paramV%Pt(1,i+1))
  !             CALL GetUVPointInfo0(s,u,v,P3)
  !             P1(:) = 0.0d0
  !             P1(1) = v
  !             CALL Point3dQueue_Add(physical, i+1, P3)
  !             CALL Point3dQueue_Add(paramV, i+1, P1)
  !             numV = numV + 1
  !              WRITE(*,*)dist
  !             ! CALL Point3dQueue_Clear(paramV)
  !             ! DO k = 1,numV
  !             !   v = VBOT + (REAL(k-1)/REAL(numV-1))*(VTOP-VBOT)
  !             !   P1(:) = 0.0d0
  !             !   P1(1) = v
  !             !   CALL Point3dQueue_Push(paramV, P1)
  !             ! END DO
  !             flag = .FALSE.
  !             EXIT uloop
  !           END IF
  !         END DO
  !       END DO uloop
  !     END IF

  !     IF (flag.AND.(paramU%numNodes<maxNodes)) THEN
  !       !Now the v's
  !       vloop: DO iv = 1,paramV%numNodes
  !         !Each u is going to be at the same v
  !         v = paramV%Pt(1,iv)

  !         CALL Point3dQueue_Clear(physical)
  !         !Fill physical space array
  !         DO i = 1,paramU%numNodes
  !           u = paramU%Pt(1,i)
  !           !WRITE(*,*) u,v
  !           CALL GetUVPointInfo0(s,u,v,P)
  !           CALL Point3dQueue_Push(physical, P)
  !         END DO

  !         !Check for refinement
  !         DO i = 1,physical%numNodes-1
  !           !Project midpoint of two nodes to curve
  !           P2(:) = 0.5d0*(physical%Pt(:,i)+physical%Pt(:,i+1))
  !           CALL ProjectToRegionFromXYZ(s,P2,uv,TOLG,0,dist)
  !           !WRITE(*,*)s,iv,i,dist

  !           IF (dist.GT.BGSpacing%MinSize) THEN
  !             u = 0.5d0*(paramU%Pt(1,i)+paramU%Pt(1,i+1))
  !             CALL GetUVPointInfo0(s,u,v,P3)
  !             P1(:) = 0.0d0
  !             P1(1) = u
  !             CALL Point3dQueue_Add(physical, i+1, P3)
  !             CALL Point3dQueue_Add(paramU, i+1, P1)
  !             flag = .FALSE.
  !             numU = numU + 1
  !             WRITE(*,*)dist
  !             ! CALL Point3dQueue_Clear(paramU)
  !             ! DO k = 1,numU
  !             !   u = UBOT + (REAL(k-1)/REAL(numU-1))*(UTOP-UBOT)
  !             !   P1(:) = 0.0d0
  !             !   P1(1) = u
  !             !   CALL Point3dQueue_Push(paramU, P1)
  !             ! END DO
  !             ! flag = .FALSE.
  !             EXIT vloop
  !           END IF
  !         END DO
  !       END DO vloop
  !     END IF

  !     !Check for completion
  !     IF (flag) THEN
  !       complete = .TRUE.
  !       EXIT
  !     END IF

  !   END DO

  !   Curv%Regions(s)%numNodeU = paramU%numNodes
  !   Curv%Regions(s)%numNodeV = paramV%numNodes

  !   ALLOCATE(Curv%Regions(s)%Posit(3,paramU%numNodes,paramV%numNodes))

  !   !Get the list of curves surrounding this region
  !   CALL GetRegionCurveList(s,IC)
  !   Curv%Regions(s)%numCurve = IC%numNodes
  !   ALLOCATE(Curv%Regions(s)%IC(IC%numNodes))
  !   DO i = 1,IC%numNodes
  !     Curv%Regions(s)%IC(i) = IC%Nodes(i)
  !   END DO

  !   !Now sample the region

  !    DO j = 1,paramU%numNodes
  !      DO k = 1,paramV%numNodes
  !        u = paramU%Pt(1,j)
  !        v = paramV%Pt(1,k)

  !        CALL GetUVPointInfo0(s,u,v,P)
  !        Curv%Regions(s)%Posit(:,j,k) = P(:)


  !      END DO
  !    END DO

  ! END DO




! CALL SurfaceCurvature_Output('igsToDat',8,Curv)
  CALL SurfaceCurvature_BuildTangent(Curv)


END SUBROUTINE










SUBROUTINE SurfaceCurvature_from_occt(alpha, Curv, NReStr)

    USE occt_fortran
    USE SurfaceCurvature
    USE SurfaceMeshStorage
    USE SpacingStorage
    IMPLICIT NONE
    REAL*8,  INTENT(IN) ::  alpha
    TYPE(SurfaceCurvatureType), INTENT(INOUT) :: Curv
    INTEGER, INTENT(IN) ::  NReStr
    INTEGER :: icv, isf, id, nu, nv, ip, iu, iv, nn, imax, Isucc, nt
    REAL*8  :: dd, du, dv, dmin, dmax, StrMax
    INTEGER, PARAMETER :: maxN = 1001
    INTEGER :: im(maxN), it(maxN)
    REAL*8  :: ui(maxN), vi(maxN), ddd(maxN), uv(2), tt(maxN), ttd(maxN)
    REAL*8  :: p1(3), p2(3), p3(3), p4(3)
    REAL*8  :: UBOT, VBOT, UTOP, VTOP, u, v
    REAL*8  :: R(3), Ru(3),Rv(3),Ruv(3),Ruu(3),Rvv(3), ck
    LOGICAL :: CarryOn
    CHARACTER*256 :: theName

    CALL OCCT_GetNumCurves( Curv%NB_Curve )
    CALL OCCT_GetNumSurfaces( Curv%NB_Region )
    ALLOCATE(Curv%Curves (Curv%NB_Curve) )
    ALLOCATE(Curv%Regions(Curv%NB_Region))
    WRITE(*,*) 'SurfaceCurvature_Input_CADfix :: Curv% NB_Curve, NB_Region='
    WRITE(*,*) Curv%NB_Curve, Curv%NB_Region

    dmin   = 0.7d0 * alpha
    StrMax = 300.d0

    !--- generate curves

    DO icv = 1, Curv%NB_Curve
       CALL OCCT_GetCurveName( icv, theName )
!        WRITE(104,*) icv,theName
       Curv%Curves(icv)%theName  = theName
       Curv%Curves(icv)%ID       = icv
       Curv%Curves(icv)%TopoType = 1

       CALL OCCT_GetLineTBox( icv, uBOT, uTOP )

       nu = 9
       du = (UTOP - UBOT) / (nu-1)
       DO iu=1,nu
          ui(iu) = (iu-1) * du + uBOT
       ENDDO

       ddd(:) = -1
       DO WHILE(nu<MaxN/2)

          dmax = 0
          DO iu = 1, nu-1
             IF(ddd(iu)<0)THEN
                u = (ui(iu) + ui(iu+1) ) / 2.d0
                CALL OCCT_GetLinePointDeriv( icv, u, Ru,Ruu)
      !      WRITE(102,*) icv,u,Ru,Ruu
                CALL DiffGeo_CalcLineCurvature(Ru, Ruu, ck, Isucc)
                CALL OCCT_GetLineXYZFromT ( icv, ui(iu), p1 )
                CALL OCCT_GetLineXYZFromT ( icv, ui(iu+1), p2 )
                dd = Geo3D_Distance(p1,p2)
                ddd(iu) = ABS(ck)*dd
             ENDIF
             IF(dmax<ddd(iu)) dmax = ddd(iu)
          ENDDO

          IF(dmax<=dmin) EXIT

          tt(1:nu)  = ui(1:nu)
          ttd(1:nu) = ddd(1:nu)
          imax = 0
          DO iu = 1, nu
             imax = imax+1
             ui(imax) = tt(iu)
             ddd(imax) = ttd(iu)
             IF(iu<nu .AND. ttd(iu)>dmin)THEN
                imax = imax+1
                ui(imax) = (tt(iu)+tt(iu+1))/2.d0
                ddd(imax-1) = -1
                ddd(imax) = -1
             ENDIF
          ENDDO
          nu = imax
       ENDDO

       !--- smooth
       DO ip =1,nu-1
          ddd(ip) = ui(ip+1) - ui(ip)
       ENDDO

       CarryOn = .TRUE.
       DO WHILE(CarryOn)
          CarryOn = .FALSE.
          DO ip =1,nu-2
             IF(ddd(ip)>1.2d0*ddd(ip+1))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip)   = 0.545d0 * dd
                ddd(ip+1) = dd - ddd(ip)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
          DO ip =nu-2, 1, -1
             IF(ddd(ip+1)>1.2d0*ddd(ip))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip+1) = 0.545d0 * dd
                ddd(ip)   = dd - ddd(ip+1)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
       ENDDO

       DO ip =2,nu-1
          ui(ip) = ui(ip-1) + ddd(ip-1)
       ENDDO

       !--- preject

       Curv%Curves(icv)%numNodes = nu
       ALLOCATE(Curv%Curves(icv)%Posit(3,nu))
       DO ip=1,nu
          u = ui(ip)
          CALL OCCT_GetLineXYZFromT ( icv, u, R )
          Curv%Curves(icv)%Posit(:,ip) = R(1:3)
       ENDDO
    ENDDO

    !--- generate regions

    DO isf = 1,Curv%NB_Region

       !CALL CADFIX_GetSurfaceName( isf, theName )
       Curv%Regions(isf)%theName  = 'OCCT'
       Curv%Regions(isf)%ID       =  isf
       Curv%Regions(isf)%TopoType = 1
       CALL OCCT_GetSurfaceUVBox( isf, UBOT, VBOT, UTOP, VTOP )

       nu = 9
       nv = 9
       du = (UTOP - UBOT) / (nu-1)
       dv = (VTOP - VBOT) / (nv-1)
       DO iu=1,nu
          ui(iu) = (iu-1) * du + uBOT
       ENDDO
       DO iv=1,nv
          vi(iv) = (iv-1) * dv + vBOT
       ENDDO

       !--- set ui

       ddd(:) = -1
       DO WHILE(nu<MaxN/2)

          dmax = 0
          DO iu = 1, nu-1
             IF(ddd(iu)<0)THEN
                DO iv = 1, nv-1
                   u = (ui(iu) + ui(iu+1)) / 2.d0
                   v = (vi(iv) + vi(iv+1)) / 2.d0
                   CALL OCCT_GetUVPointInfoAll( isf, u, v, R,Ru,Rv,Ruv,Ruu,Rvv )
                   CALL DiffGeo_CalcNormalCurvature(Ru, Rv, Ruv, Ruu, Rvv, Ru, ck, Isucc)
                   IF(Isucc==0) CALL Error_STOP ( '--- sharp point?')
                   uv = (/ui(iu),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                   uv = (/ui(iu+1),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                   dd = ABS(ck) * Geo3D_Distance(p1,p2)
                   ddd(iu) = MAX(ddd(iu), dd)
                ENDDO
             ENDIF
             IF(dmax<ddd(iu)) dmax = ddd(iu)
          ENDDO

          IF(dmax<=dmin) EXIT
          tt(1:nu) = ui(1:nu)
          ttd(1:nu) = ddd(1:nu)
          imax = 0
          DO iu = 1, nu
             imax = imax+1
             ui(imax) = tt(iu)
             ddd(imax) = ttd(iu)
             IF(iu<nu .AND. ttd(iu)>dmin)THEN
                imax = imax+1
                ui(imax) = (tt(iu)+tt(iu+1))/2.d0
                ddd(imax-1) = -1
                ddd(imax) = -1
             ENDIF
          ENDDO
          nu = imax

       ENDDO

       !--- smooth
       DO ip =1,nu-1
          ddd(ip) = ui(ip+1) - ui(ip)
       ENDDO

       CarryOn = .TRUE.
       DO WHILE(CarryOn)
          CarryOn = .FALSE.
          DO ip =1,nu-2
             IF(ddd(ip)>1.2d0*ddd(ip+1))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip)   = 0.545d0 * dd
                ddd(ip+1) = dd - ddd(ip)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
          DO ip =nu-2, 1, -1
             IF(ddd(ip+1)>1.2d0*ddd(ip))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip+1) = 0.545d0 * dd
                ddd(ip)   = dd - ddd(ip+1)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
       ENDDO

       DO ip =2,nu-1
          ui(ip) = ui(ip-1) + ddd(ip-1)
       ENDDO

       !--- set vi

       ddd(:) = -1
       DO WHILE(nv<MaxN/2)

          dmax = 0
          DO iv = 1, nv-1
             IF(ddd(iv)<0)THEN
                DO iu = 1, nu-1
                   u = (ui(iu) + ui(iu+1)) / 2.d0
                   v = (vi(iv) + vi(iv+1)) / 2.d0
                   CALL OCCT_GetUVPointInfoAll( isf, u, v, R,Ru,Rv,Ruv,Ruu,Rvv )
                   CALL DiffGeo_CalcNormalCurvature(Ru, Rv, Ruv, Ruu, Rvv, Rv, ck, Isucc)
                   IF(Isucc==0) CALL Error_STOP ( '--- sharp point?')
                   uv = (/u,vi(iv)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                   uv = (/u,vi(iv+1)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                   dd = ABS(ck) * Geo3D_Distance(p1,p2)
                   ddd(iv) = MAX(ddd(iv), dd)
                ENDDO
             ENDIF
             IF(dmax<ddd(iv)) dmax = ddd(iv)
          ENDDO

          IF(dmax<=dmin) EXIT
          tt(1:nv) = vi(1:nv)
          ttd(1:nv) = ddd(1:nv)
          imax = 0
          DO iv = 1, nv
             imax = imax+1
             vi(imax) = tt(iv)
             ddd(imax) = ttd(iv)
             IF(iv<nv .AND. ttd(iv)>dmin)THEN
                imax = imax+1
                vi(imax) = (tt(iv)+tt(iv+1))/2.d0
                ddd(imax-1) = -1
                ddd(imax) = -1
             ENDIF
          ENDDO
          nv = imax

       ENDDO


       !--- smooth
       DO ip =1,nv-1
          ddd(ip) = vi(ip+1) - vi(ip)
       ENDDO

       CarryOn = .TRUE.
       DO WHILE(CarryOn)
          CarryOn = .FALSE.
          DO ip =1,nv-2
             IF(ddd(ip)>1.2d0*ddd(ip+1))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip)   = 0.545d0 * dd
                ddd(ip+1) = dd - ddd(ip)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
          DO ip =nv-2, 1, -1
             IF(ddd(ip+1)>1.2d0*ddd(ip))THEN
                dd        = ddd(ip) + ddd(ip+1)
                ddd(ip+1) = 0.545d0 * dd
                ddd(ip)   = dd - ddd(ip+1)
                CarryOn = .TRUE.
             ENDIF
          ENDDO
       ENDDO

       DO ip =2,nv-1
          vi(ip) = vi(ip-1) + ddd(ip-1)
       ENDDO


       !--- reset ui to avoid big stretching
       nt = 0
       DO WHILE(nu<MaxN/2 .AND. nt<NReStr)

          nt = nt + 1
          dmax = 0
          DO iu = 1, nu-1
             IF(nt==1)THEN
                ddd(iu) = 0
                im(iu)  = 0
                DO iv = 1, nv-1
                   u = (ui(iu) + ui(iu+1)) / 2.d0
                   v = (vi(iv) + vi(iv+1)) / 2.d0
                   uv = (/ui(iu),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                   uv = (/ui(iu+1),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                   uv = (/u,vi(iv)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p3 )
                   uv = (/u,vi(iv+1)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p4 )
                   dd = Geo3D_Distance_SQ(p1,p2) / Geo3D_Distance_SQ(p3,p4)
                   IF(dd>ddd(iu) .AND. dd>StrMax)THEN
                      ddd(iu) = dd
                      im(iu)  = iv
                   ENDIF
                ENDDO
             ELSE IF(im(iu)>0)THEN
                iv = im(iu)
                u = (ui(iu) + ui(iu+1)) / 2.d0
                v = (vi(iv) + vi(iv+1)) / 2.d0
                uv = (/ui(iu),v/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                uv = (/ui(iu+1),v/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                uv = (/u,vi(iv)/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p3 )
                uv = (/u,vi(iv+1)/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p4 )
                dd = Geo3D_Distance_SQ(p1,p2) / Geo3D_Distance_SQ(p3,p4)
                IF(dd>StrMax)THEN
                   ddd(iu) = dd
                ELSE
                   im(iu) = 0
                ENDIF
             ENDIF
             IF(im(iu)>0 .AND. dmax<ddd(iu)) dmax = ddd(iu)
          ENDDO

          IF(dmax<=StrMax) EXIT
          tt(1:nu) = ui(1:nu)
          it(1:nu) = im(1:nu)
          imax = 0
          DO iu = 1, nu
             imax = imax+1
             ui(imax) = tt(iu)
             im(imax) = it(iu)
             IF(iu<nu .AND. it(iu)>0)THEN
                imax = imax+1
                ui(imax) = (tt(iu)+tt(iu+1))/2.d0
                im(imax) = it(iu)
             ENDIF
          ENDDO
          nu = imax

       ENDDO

       IF(nt>1)THEN
          !--- smooth
          DO ip =1,nu-1
             ddd(ip) = ui(ip+1) - ui(ip)
          ENDDO

          CarryOn = .TRUE.
          DO WHILE(CarryOn)
             CarryOn = .FALSE.
             DO ip =1,nu-2
                IF(ddd(ip)>1.2d0*ddd(ip+1))THEN
                   dd        = ddd(ip) + ddd(ip+1)
                   ddd(ip)   = 0.545d0 * dd
                   ddd(ip+1) = dd - ddd(ip)
                   CarryOn = .TRUE.
                ENDIF
             ENDDO
             DO ip =nu-2, 1, -1
                IF(ddd(ip+1)>1.2d0*ddd(ip))THEN
                   dd        = ddd(ip) + ddd(ip+1)
                   ddd(ip+1) = 0.545d0 * dd
                   ddd(ip)   = dd - ddd(ip+1)
                   CarryOn = .TRUE.
                ENDIF
             ENDDO
          ENDDO

          DO ip =2,nu-1
             ui(ip) = ui(ip-1) + ddd(ip-1)
          ENDDO
       ENDIF

       !--- reset vi to avoid big stretching

       nt = 0
       DO WHILE(nv<MaxN/2 .AND. nt<NReStr)

          nt = nt + 1
          dmax = 0
          DO iv = 1, nv-1
             IF(nt==1)THEN
                ddd(iv) = 0
                im(iv)  = 0
                DO iu = 1, nu-1
                   u = (ui(iu) + ui(iu+1)) / 2.d0
                   v = (vi(iv) + vi(iv+1)) / 2.d0
                   uv = (/ui(iu),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                   uv = (/ui(iu+1),v/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                   uv = (/u,vi(iv)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p3 )
                   uv = (/u,vi(iv+1)/)
                   CALL OCCT_GetXYZFromUV1( isf, uv, p4 )
                   dd = Geo3D_Distance_SQ(p3,p4) / Geo3D_Distance_SQ(p1,p2)
                   IF(dd>ddd(iv) .AND. dd>StrMax)THEN
                      ddd(iv) = dd
                      im(iv)  = iu
                   ENDIF
                ENDDO
             ELSE IF(im(iv)>0)THEN
                iu = im(iv)
                u = (ui(iu) + ui(iu+1)) / 2.d0
                v = (vi(iv) + vi(iv+1)) / 2.d0
                uv = (/ui(iu),v/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p1 )
                uv = (/ui(iu+1),v/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p2 )
                uv = (/u,vi(iv)/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p3 )
                uv = (/u,vi(iv+1)/)
                CALL OCCT_GetXYZFromUV1( isf, uv, p4 )
                dd = Geo3D_Distance_SQ(p3,p4) / Geo3D_Distance_SQ(p1,p2)
                IF(dd>StrMax)THEN
                   ddd(iv) = dd
                ELSE
                   im(iv) = 0
                ENDIF
             ENDIF
             IF(im(iv)>0 .AND. dmax<ddd(iv)) dmax = ddd(iv)
          ENDDO

          IF(dmax<=StrMax) EXIT
          tt(1:nv) = vi(1:nv)
          it(1:nv) = im(1:nv)
          imax = 0
          DO iv = 1, nv
             imax = imax+1
             vi(imax) = tt(iv)
             im(imax) = it(iv)
             IF(iv<nv .AND. it(iv)>0)THEN
                imax = imax+1
                vi(imax) = (tt(iv)+tt(iv+1))/2.d0
                im(imax) = it(iv)
             ENDIF
          ENDDO
          nv = imax

       ENDDO


       IF(nt>1)THEN
          !--- smooth
          DO ip =1,nv-1
             ddd(ip) = vi(ip+1) - vi(ip)
          ENDDO

          CarryOn = .TRUE.
          DO WHILE(CarryOn)
             CarryOn = .FALSE.
             DO ip =1,nv-2
                IF(ddd(ip)>1.2d0*ddd(ip+1))THEN
                   dd        = ddd(ip) + ddd(ip+1)
                   ddd(ip)   = 0.545d0 * dd
                   ddd(ip+1) = dd - ddd(ip)
                   CarryOn = .TRUE.
                ENDIF
             ENDDO
             DO ip =nv-2, 1, -1
                IF(ddd(ip+1)>1.2d0*ddd(ip))THEN
                   dd        = ddd(ip) + ddd(ip+1)
                   ddd(ip+1) = 0.545d0 * dd
                   ddd(ip)   = dd - ddd(ip+1)
                   CarryOn = .TRUE.
                ENDIF
             ENDDO
          ENDDO

          DO ip =2,nv-1
             vi(ip) = vi(ip-1) + ddd(ip-1)
          ENDDO
       ENDIF

       !--- project

       Curv%Regions(isf)%numNodeU = nu
       Curv%Regions(isf)%numNodeV = nv
       ALLOCATE(Curv%Regions(isf)%Posit(3,nu,nv))
       DO iv=1,nv
          DO iu=1,nu
             uv = (/ ui(iu), vi(iv) /)
             CALL OCCT_GetXYZFromUV1( isf, uv, R(1:3) )
             Curv%Regions(isf)%Posit(:,iu,iv) = R(1:3)
          ENDDO
       ENDDO


       Curv%Regions(isf)%GeoType = 1
       CALL OCCT_GetSurfaceNumCurves(isf, nn)

       Curv%Regions(isf)%numCurve = nn
       ALLOCATE(Curv%Regions(isf)%IC(nn))
       CALL OCCT_GetSurfaceCurves(isf, nn, Curv%Regions(isf)%IC)
    ENDDO

    RETURN

  END SUBROUTINE SurfaceCurvature_from_occt

