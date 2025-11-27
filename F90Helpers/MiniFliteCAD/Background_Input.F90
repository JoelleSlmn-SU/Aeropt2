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
!!   Read background according to surface_Parameters::Background_Model.
!<
!*******************************************************************************
SUBROUTINE Background_Input()

  USE occt_fortran !SPW
  USE control_Parameters
  USE surface_Parameters
  USE Spacing_Parameters
  USE Geometry3DAll
  USE SpacingStorageGen
  USE SurfaceCurvatureCADfix
  USE Number_Char_Transfer
  IMPLICIT NONE
  REAL*8  :: Stretch(6), alpha1, p(3)
  INTEGER :: i, j, Isucc, numCuts
  LOGICAL :: ex, isotr
  TYPE(SurfaceMeshStorageType)  :: SurfBack           !<   Surface mesh for background.
  TYPE(SurfaceCurvatureType)    :: CurvBack           !<   Surface curvature for background.
  TYPE(IntQueueType) :: IC  !< for reading surface curves
  REAL*8  :: d, maxd, mind, tMin, tMax, t, P1(3), P2(3), meand
  BGSpacing%Model            = 1
  BGSpacing%CheckLevel       = Debug_Display
  BGSpacing%Interp_Method    = Mapping_Interp_Model
  BGSpacing%OctCell_Vary     = Interpolate_Oct_Mapping
  BGSpacing%Gradation_Factor = Gradation_Factor


 !TOLG   = 3.d-4
 !IF (Curvature_Type.NE.4) THEN

    TOLG   = 1.d-4 *  GLOBE_GRIDSIZE

 !END IF

  WRITE(*,*) 'TOLG=',TOLG, GLOBE_GRIDSIZE


  IF(ABS(Background_Model)==7)THEN

     !--- read octree background
     INQUIRE(file = JobName(1:JobNameLength)//'.Obac', EXIST=ex)
     IF(ex)THEN
        CALL SpacingStorage_read_Obac(JobName,JobNameLength,BGSpacing)
        IF(BGSpacing%Model==-3 .AND. Background_Model>0)THEN
           WRITE(29,*) ' '
           WRITE(29,*) ' Warning---- Background_Model against .Obac file input.'
           WRITE(29,*) ' Set Background_Model = -7'
           Background_Model = -7
        ENDIF
     ELSE
        WRITE(29,*)' Error---- no .Obac file exsiting. '
        CALL Error_Stop ('  Background_Input :: ')
     ENDIF

  ELSE IF( ABS(Background_Model)>=2 .AND. ABS(Background_Model)<=6 )THEN

     !--- Read background mesh .bac
     IF(Background_Model<0)THEN
        isotr = .FALSE.
     ELSE
        isotr = .TRUE.
     ENDIF

     Isucc = 0

     IF(Curvature_Type==2)THEN
        CALL SpacingStorage_from_CADfix(BGSpacing, isotr, Isucc)
     ENDIF
     IF(Isucc==0)THEN
        IF(Globe_GridSize>0)THEN
           !--- Globe_GridSize has been read from *.bpp file
           BGSpacing%BasicSize = Globe_GridSize
           BGSpacing%MinSize   = Globe_GridSize
        ENDIF
        INQUIRE(file = JobName(1:JobNameLength)//'.bac', EXIST=ex)
        IF(ex)THEN
           CALL SpacingStorage_read_bac(JobName,JobNameLength,BGSpacing,isotr, Stretch_Limit)
        ELSE
           IF(Background_Model<0)THEN
              BGSpacing%Model = -1
           ELSE
              BGSpacing%Model = 1
           ENDIF

        ENDIF
     ENDIF

     IF( ABS(Background_Model)==5 .OR. ABS(Background_Model)==6) THEN
        !--- Add point source from surface curvature.
        !    Present  background will be refered.

        IF(ABS(Background_Model)==5)THEN
           IF(Curvature_Type==4)THEN
              CALL OCCT_To_FLITE(CurvBack)
             !CALL SpacingStorage_SourceCurvature(BGSpacing, CurvBack, Curvature_Factors, Stretch_Limit)
           ELSE IF(Curvature_Type==2) THEN
              alpha1 = MIN(Curvature_Factors(1), 0.1d0)
              CALL SurfaceCurvature_from_CADfix(alpha1, CurvBack, 0)
           ELSE
              CALL SurfaceCurvature_Input(JobName,JobNameLength,CurvBack)
           ENDIF
           CALL SurfaceCurvature_BuildTangent(CurvBack)

           DO i = 1, CurvBack%NB_Curve
              CurvBack%Curves(i)%TopoType = -1
           ENDDO
           DO i = 1,CurvBack%NB_Region
              CurvBack%Regions(i)%TopoType = -1
           ENDDO

          !print *,' Unchecked lists info:',ListUc%numNodes, ListUs%numNodes
           DO i = 1, ListUc%numNodes
              j = ListUc%Nodes(i)
              IF(j<=0 .OR. j>CurvBack%NB_Curve) CYCLE
              CurvBack%Curves(j)%TopoType = 1
           ENDDO
           DO i = 1, ListUs%numNodes
              j = ListUs%Nodes(i)
              IF(j<=0 .OR. j>CurvBack%NB_Region) CYCLE
              CurvBack%Regions(j)%TopoType = 1
        ! Need to add the curves on this surface
              CALL GetRegionCurveList(j,IC)
              CurvBack%Curves(IC%Nodes(1:IC%numNodes))%TopoType = 1;
           ENDDO
           DO i = 1, CurvBack%NB_Curve
              IF(CHAR_Contains(NameUc,CurvBack%Curves(i)%theName))   &
                   CurvBack%Curves(i)%TopoType = 1
           ENDDO
           DO i = 1, CurvBack%NB_Region
              IF(CHAR_Contains(NameUs,CurvBack%Regions(i)%theName)) then
                   CurvBack%Regions(i)%TopoType = 1
         ! Need to add the curves on this surface
                  CALL GetRegionCurveList(i,IC)
                  CurvBack%Curves(IC%Nodes(1:IC%numNodes))%TopoType = 1;
               end if
           ENDDO

         ! write(29,*) ' Curves marker'
         ! DO i = 1, CurvBack%NB_Curve
         !    write(29,*) CurvBack%Curves(i)%TopoType
         ! ENDDO
         ! write(29,*) ' Surface marker'
         ! DO i = 1,CurvBack%NB_Region
         !    write(29,*) CurvBack%Regions(i)%TopoType
         ! ENDDO


           CALL SpacingStorage_SourceCurvature(BGSpacing, CurvBack, Curvature_Factors, &
               Stretch_Limit,CurveRefine_Limit,Curvature_Type,TOLG)
        ELSE IF(ABS(Background_Model)==6)THEN
           CALL SurfaceMeshStorage_Input(JobName(1:JobNameLength)//'_0', JobNameLength+2,1, SurfBack)
           CALL SpacingStorage_SourceSurface(BGSpacing, SurfBack, Curvature_Factors, Stretch_Limit,ListUs,SurfOctRef)
           CALL SurfaceMeshStorage_Clear(SurfBack)
        ENDIF

        IF(Debug_Display>1)THEN
           CALL SourceGroup_info(BGSpacing%Sources)
           WRITE(*,*)'     number of BOX sources: ',BGSpacing%NB_BoxSource
        ENDIF
     ENDIF

     IF( ABS(BGSpacing%Model)==4 .AND. (.NOT. SourceGroup_isReady(BGSpacing%Sources)) )THEN
        CALL Error_Stop (' --- not expected here==== ')
     ENDIF

     IF(ABS(Background_Model)>=4 .AND. ABS(Background_Model)<=6)THEN
        !--- build octree background using present tet. background
        !--- Parameter BGSpacing%Model will be changed.
        IF(ABS(BGSpacing%Model)==1 .AND. BGSpacing%NB_BoxSource==0)THEN
           WRITE(29,*)' '
           WRITE(29,*)' Informing--- No Oct. background built for a EVEN domain.'
        ELSE
           CALL SpacingStorage_BuildOctree(BGSpacing)
        ENDIF
        if(Output_Bac.gt.0) CALL SpacingStorage_write_Obac(JobName,JobNameLength,BGSpacing)
     ENDIF

  ELSE
     WRITE(29,*)' '
     WRITE(29,*)' Error--- illegal Background_Model'
     CALL Error_Stop ('  Background_Input ::')
  ENDIF

  CALL SpacingStorage_Clean(BGSpacing)

  BGSpacing%BasicSize = BGSpacing%BasicSize * GridSize_Adjustor

  WRITE(29,*)'============================='
  CALL SpacingStorage_info(BGSpacing, 29, ex)
  IF(ABS(GridSize_Adjustor-1.d0)>1.d-6)   &
       WRITE(29,*)' GridSize_Adjustor involved: ',GridSize_Adjustor
  WRITE(29,*)'============================='
  IF(.NOT. ex)THEN
     WRITE(29,*)' '
     WRITE(29,*)' Error--- Background spacing is not ready'
     CALL Error_Stop (' Background_Input :: ')
  ENDIF


  RETURN
END SUBROUTINE Background_Input
