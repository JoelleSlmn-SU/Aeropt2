!==============================================================
! CadSurfaceSampler.F90
!   Fortran driver to sample CAD surfaces using OCCT and
!   write one simple text file per surface:
!       surf_0001.dat, surf_0002.dat, ...
!
! Usage from command line:
!   occt_sampler  cad_file.stp  output_directory
!
! Each surf_xxxx.dat has:
!   line 1: sid numU numV
!   lines:  i j x y z
!==============================================================
PROGRAM CadSurfaceSampler
  USE occt_fortran          ! <-- your existing OCCT-Fortran binding
  IMPLICIT NONE

  CHARACTER(512) :: cadFile, outDir
  INTEGER        :: istat

  CALL GetCommandArgument(1, cadFile)
  CALL GetCommandArgument(2, outDir)

  IF (LEN_TRIM(cadFile) == 0) THEN
     WRITE(*,*) 'Usage: occt_sampler cad_file.stp output_dir'
     STOP 1
  END IF

  IF (LEN_TRIM(outDir) == 0) THEN
     WRITE(*,*) 'Usage: occt_sampler cad_file.stp output_dir'
     STOP 1
  END IF

  !---------------------------------------------
  ! Initialise OCCT with the CAD file.
  ! Replace OCCT_InitFromFile with *whatever*
  ! you currently use to load the geometry.
  !---------------------------------------------
  WRITE(*,*) ' [OCCT] Loading CAD file: ', TRIM(cadFile)
  CALL OCCT_InitFromFile(TRIM(cadFile), istat)    ! <<< CHANGE TO YOUR ACTUAL INIT
  IF (istat /= 0) THEN
     WRITE(*,*) ' [OCCT] ERROR: failed to load CAD file; istat=', istat
     STOP 2
  END IF

  CALL WriteSurfaceGrids(outDir)

END PROGRAM CadSurfaceSampler


!==============================================================
!  WriteSurfaceGrids
!    - queries OCCT for # of surfaces
!    - for each surface s, samples a (numU x numV) grid in UV
!    - writes surf_xxxx.dat under outDir
!==============================================================
SUBROUTINE WriteSurfaceGrids(outDir)
  USE occt_fortran
  IMPLICIT NONE

  CHARACTER(*), INTENT(IN) :: outDir

  INTEGER :: NBs, s
  INTEGER :: numU, numV, iu, iv, unit
  REAL*8  :: UBOT, VBOT, UTOP, VTOP
  REAL*8  :: u, v, P(3)
  CHARACTER(512) :: fname

  ! You can tweak this resolution or even make it depend on curvature later
  numU = 11
  numV = 11

  CALL OCCT_GetNumSurfaces(NBs)
  WRITE(*,*) ' [OCCT] Number of surfaces: ', NBs

  DO s = 1, NBs

    ! --- UV box for region s ---
    CALL OCCT_GetRegionUVBox(s, UBOT, VBOT, UTOP, VTOP)

    WRITE(fname, '(A,"/surf_",I4.4,".dat")') TRIM(outDir), s
    OPEN(NEWUNIT=unit, FILE=TRIM(fname), STATUS='REPLACE', ACTION='WRITE')

    ! Header: sid, numU, numV
    WRITE(unit, '(3I10)') s, numU, numV

    DO iu = 1, numU
      u = UBOT + (DBLE(iu-1)/DBLE(numU-1))*(UTOP - UBOT)
      DO iv = 1, numV
        v = VBOT + (DBLE(iv-1)/DBLE(numV-1))*(VTOP - VBOT)

        CALL OCCT_GetRegionXYZFromUV(s, u, v, P)

        ! i, j, x, y, z
        WRITE(unit, '(2I6,3ES25.16)') iu, iv, P(1), P(2), P(3)
      END DO
    END DO

    CLOSE(unit)
    WRITE(*,*) ' [OCCT] Wrote ', TRIM(fname)

  END DO

END SUBROUTINE WriteSurfaceGrids
