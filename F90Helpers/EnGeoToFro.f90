program EnGeoSurfaceToFro
! Read an EnSight Gold ASCII .geo that already contains *surface*
! elements
! (parts with TRIA3 / QUAD4) and write a .fro file.
! Keeps original variable names used in your codebase.

  implicit none
  integer, parameter :: RK4 = selected_real_kind(6), RK8 =
selected_real_kind(15)
  character(len=255) :: geofilename, frofilename, line
  integer :: i, j, k, ip, ie, npart, nsurf, ios, ntr, nqd, q
  integer :: ipcount, ifcount, part_id, surf_id
  integer, allocatable :: faces_n1(:), faces_n2(:), faces_n3(:),
faces_sid(:)
  real(RK4), allocatable :: coorsurf(:,:)
  integer, allocatable :: nodemap(:)      ! identity map (reserved for
future filtering)
  integer :: unit_in, unit_out
  logical :: in_coords, in_part, ascii_ok
  integer :: nnode, loc_nnode
  integer :: tri_count_total, quad_count_total
  integer :: nread, node_id_dummy

  ! Your existing arrays that the writer expects
  integer, allocatable :: gloconnect(:,:)   ! (4, ifcount): n1, n2, n3,
surf_id

  print *, 'Input the .geo filename (including .geo extension):'
  read(*,'(A)') geofilename
  print *, 'Input the output .fro filename:'
  read(*,'(A)') frofilename

  unit_in  = 11
  unit_out = 10

  open(unit_in, file=trim(geofilename), status='old', action='read',
iostat=ios)
  if (ios /= 0) then
     print *, 'ERROR: cannot open ', trim(geofilename)
     stop 1
  end if

  ! --- Quick sanity: EnSight Gold ASCII starts with a FORMAT line
  ! typically
  ascii_ok = .false.
  do
     read(unit_in,'(A)',iostat=ios) line
     if (ios /= 0) exit
     if (index(to_lower(line),'binary') > 0) then
        print *, 'ERROR: This converter expects ASCII EnSight Gold .geo,
not Binary.'
        close(unit_in); stop 2
     endif
     if (index(to_lower(line),'node id assign') > 0 .or.
index(to_lower(line),'coordinates') > 0) then
        ! Rewind to start to parse properly
        rewind(unit_in)
        ascii_ok = .true.
        exit
     endif
  end do
  if (.not. ascii_ok) then
     print *, 'ERROR: could not detect ASCII EnSight Gold header.'
     close(unit_in); stop 3
  end if

  ! --- First pass: count nodes and total surface elements so we can
  ! allocate
  ipcount = 0
  ifcount = 0
  nsurf   = 0
  in_coords = .false.; in_part = .false.
  tri_count_total  = 0
  quad_count_total = 0

  do
     read(unit_in,'(A)',iostat=ios) line
     if (ios /= 0) exit

     if (index(to_lower(line),'coordinates') > 0) then
        ! Next line should be number of nodes
        read(unit_in,*,iostat=ios) nnode
        if (ios /= 0) then
           print *, 'ERROR: failed to read node count after COORDINATES'
           close(unit_in); stop 4
        endif
        ipcount = nnode
        ! Skip the node block this pass
        do i = 1, nnode
           read(unit_in,'(A)',iostat=ios) line
           if (ios /= 0) then
              print *, 'ERROR: truncated node block.'
              close(unit_in); stop 5
           endif
        end do
     endif

     if (index(to_lower(line),'part') == 1) then
        nsurf = nsurf + 1
        cycle
     endif

     ! Count surface elements
     if (index(to_lower(line),'tria3') == 1) then
        read(unit_in,*,iostat=ios) ntr
        if (ios /= 0) then
           print *, 'ERROR: tria3 count read failed'
           close(unit_in); stop 6
        endif
        tri_count_total = tri_count_total + ntr
        do i=1,ntr
           read(unit_in,'(A)',iostat=ios) line
        end do
     else if (index(to_lower(line),'quad4') == 1) then
        read(unit_in,*,iostat=ios) nqd
        if (ios /= 0) then
           print *, 'ERROR: quad4 count read failed'
           close(unit_in); stop 7
        endif
        quad_count_total = quad_count_total + nqd
        do i=1,nqd
           read(unit_in,'(A)',iostat=ios) line
        end do
     end if
  end do

  ! Triangles from TRIA3 + 2x triangles per QUAD4
  ifcount = tri_count_total + 2*quad_count_total

  ! --- Allocate
  allocate(coorsurf(3, ipcount))
  allocate(faces_n1(ifcount), faces_n2(ifcount), faces_n3(ifcount),
faces_sid(ifcount))
  allocate(gloconnect(4, ifcount))
  allocate(nodemap(ipcount))
  do i=1,ipcount
     nodemap(i) = i
  end do

  ! --- Second pass: actually read coordinates + faces
  rewind(unit_in)
  surf_id = 0
  ie = 0

  do
     read(unit_in,'(A)',iostat=ios) line
     if (ios /= 0) exit

     if (index(to_lower(line),'coordinates') > 0) then
        read(unit_in,*,iostat=ios) nnode
        if (nnode /= ipcount) then
           print *, 'ERROR: node count changed between passes.'
           close(unit_in); stop 8
        end if
        do ip = 1, ipcount
           read(unit_in,*,iostat=ios) coorsurf(1,ip), coorsurf(2,ip),
coorsurf(3,ip)
           if (ios /= 0) then
              print *, 'ERROR: reading node ', ip
              close(unit_in); stop 9
           endif
        end do

     else if (index(to_lower(line),'part') == 1) then
        surf_id = surf_id + 1
        ! Next line is usually "description" – skip it safely
        read(unit_in,'(A)',iostat=ios) line

     else if (index(to_lower(line),'tria3') == 1) then
        read(unit_in,*,iostat=ios) ntr
        do i=1,ntr
           read(unit_in,*,iostat=ios) j, k, q
           if (ios /= 0) then
              print *, 'ERROR: reading TRIA3 at ', i
              close(unit_in); stop 10
           endif
           ie = ie + 1
           faces_n1(ie)  = j
           faces_n2(ie)  = k
           faces_n3(ie)  = q
           faces_sid(ie) = surf_id
        end do

     else if (index(to_lower(line),'quad4') == 1) then
        read(unit_in,*,iostat=ios) nqd
        do i=1,nqd
           read(unit_in,*,iostat=ios) j, k, q, node_id_dummy
           if (ios /= 0) then
              print *, 'ERROR: reading QUAD4 at ', i
              close(unit_in); stop 11
           endif
           ! Split quad (j,k,q,r) into two triangles: (j,k,q) and
           ! (j,q,r)
           ie = ie + 1
           faces_n1(ie)  = j
           faces_n2(ie)  = k
           faces_n3(ie)  = q
           faces_sid(ie) = surf_id

           ie = ie + 1
           faces_n1(ie)  = j
           faces_n2(ie)  = q
           faces_n3(ie)  = node_id_dummy
           faces_sid(ie) = surf_id
        end do
     end if
  end do
  close(unit_in)

  if (ie /= ifcount) then
     print *, 'WARNING: counted faces =', ifcount, ' but filled =', ie
     ifcount = ie
  end if

  ! Fill gloconnect the way your writer expects: (n1,n2,n3,surf_id)
  do i = 1, ifcount
     gloconnect(1,i) = faces_n1(i)
     gloconnect(2,i) = faces_n2(i)
     gloconnect(3,i) = faces_n3(i)
     gloconnect(4,i) = faces_sid(i)
  end do

  ! --- Write .fro (keeps your header fields)
  open(unit_out, file=trim(frofilename), status='replace',
action='write', iostat=ios)
  if (ios /= 0) then
     print *, 'ERROR: cannot open output ', trim(frofilename)
     stop 12
  end if

  print *, 'Number of faces =', ifcount
  print *, 'Number of nodes =', ipcount

  write(unit_out,*) ifcount, ipcount, 1, 0, 0, nsurf, 0, 0

  do ip = 1, ipcount
     write(unit_out,*) ip, coorsurf(1,ip), coorsurf(2,ip),
coorsurf(3,ip)
  end do

  do ie = 1, ifcount
     ! NOTE: fix from your original – use `ie` not `ifcount` as the
     ! first field
     write(unit_out,*) ie, gloconnect(1,ie), gloconnect(2,ie),
gloconnect(3,ie), gloconnect(4,ie)
  end do

  close(unit_out)
  print *, 'Wrote ', trim(frofilename)
end program EnGeoSurfaceToFro

contains
  pure function to_lower(s) result(t)
    character(len=*), intent(in) :: s
    character(len=len(s)) :: t
    integer :: i
    do i=1,len(s)
       select case (iachar(s(i:i)))
       case(65:90); t(i:i) = achar(iachar(s(i:i))+32)
       case default; t(i:i) = s(i:i)
       end select
    end do
  end function to_lower
end

