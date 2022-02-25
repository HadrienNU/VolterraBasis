!************************************************************
!*      Fortran implementation of inversion of first
!*      and second order Volterra Intergral equation
!************************************************************


module lapackMod
contains

! Returns the inverse of a matrix calculated by finding the LU
! decomposition.  Depends on LAPACK.
function inv(A) result(Ainv)
  implicit none
  double precision,intent(in) :: A(:,:)
  double precision            :: Ainv(size(A,1),size(A,2))
  double precision            :: work(size(A,1))            ! work array for LAPACK
  integer         :: n,info,ipiv(size(A,1))     ! pivot indices

  ! Store A in Ainv to prevent it from being overwritten by LAPACK
  Ainv = A
  n = size(A,1)
  ! SGETRF computes an LU factorization of a general M-by-N matrix A
  ! using partial pivoting with row interchanges.
  call DGETRF(n,n,Ainv,n,ipiv,info)
  if (info.ne.0) stop 'Matrix is numerically singular!'
  ! SGETRI computes the inverse of a matrix using the LU factorization
  ! computed by SGETRF.
  call DGETRI(n,Ainv,n,ipiv,work,n,info)
  if (info.ne.0) stop 'Matrix inversion failed!'
end function inv
end module lapackMod


subroutine rect_integral(res,dt,n,B,kernel,dim_basis,dim_x)
  implicit none
  integer,intent(in)::n,dim_basis,dim_x
  double precision,dimension(0:n, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:n, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(out)::res
  double precision,intent(in)::dt
  integer::j
  res=0.
  do j=0,n-1
     res=res+dt*matmul(B(n-j,:,:),kernel(j,:,:))
  end do
end subroutine rect_integral


subroutine midpoint_integral(res,dt,n,B,kernel,dim_basis,dim_x)
  implicit none
  integer,intent(in)::n,dim_basis,dim_x
  double precision,dimension(0:2*n, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:n, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(out)::res
  double precision,intent(in)::dt
  integer::j
  res=0.
  do j=0,n-1
     res=res+dt*matmul(B(2*(n-j),:,:),kernel(j,:,:))
  end do
end subroutine midpoint_integral


 ! Get  int_0^end_int B(t-s)kernel(s) ds using trapezoidal rule
subroutine trapz_integral(res,dt,n,B,kernel,dim_basis,dim_x)
  implicit none
  integer,intent(in)::n,dim_basis,dim_x
  double precision,dimension(0:n, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:n, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(out)::res
  double precision,intent(in)::dt
  integer::j
  res=0.5*dt*matmul(B(n,:,:),kernel(0,:,:))
  do j=1,n-1
     res=res+dt*matmul(B(n-j,:,:),kernel(j,:,:))
  end do
end subroutine trapz_integral

!!$ ! Get  int_0^end_int B(t-s)kernel(s) ds using simpson rule
subroutine simpson_integral(res,dt,n,B,kernel,dim_basis,dim_x)
  implicit none
  integer,intent(in)::n,dim_basis,dim_x
  double precision,dimension(0:n, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:n, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(out)::res
  double precision,intent(in)::dt
  integer::j,start_j
  double precision::h
  h=dt/3.
  !n even use trapezodial rule for first point
  if(mod(n,2) ==1) then
     res=0.5*dt*matmul(B(n,:,:),kernel(0,:,:))+(0.5*dt+h)*matmul(B(n-1,:,:),kernel(1,:,:)) ! Do trapz rule on first interval
     start_j=2
  else
     res=h*matmul(B(n,:,:),kernel(0,:,:))
     start_j=1
  end if

  do j= start_j,n-2,2
   ! write(*,*) n,j,j+1
     res=res+4*h*matmul(B(n-j,:,:),kernel(j,:,:))+2*h*matmul(B(n-j-1,:,:),kernel(j+1,:,:))
  end do
  res=res+4*h*matmul(B(1,:,:),kernel(n-1,:,:))
  !write(*,*) '--'
end subroutine simpson_integral


subroutine kernel_first_kind_rect(lenTraj, dim_basis, dim_x, kernel,  B, DxB,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxB
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(dt*B(1,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=-1*matmul(invB0,DxB(1,:,:))

  do i=1,lenTraj-1 !! for i in range(1, lenTraj):
     call rect_integral(num,dt,i,B(1:i+1,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=-1*matmul(invB0,num+DxB(i+1,:,:))
  end do


end subroutine kernel_first_kind_rect

subroutine kernel_first_kind_midpoint(lenTraj, dim_basis,dim_x, kernel,  B, DxB,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:(lenTraj-1)/2, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxB
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(2*dt*B(1,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=-1*matmul(invB0,DxB(1,:,:))

  do i=1,lenTraj/2-1 !! for i in range(1, lenTraj):
     call midpoint_integral(num,2*dt,i,B(0:2*i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=-1*matmul(invB0,num+DxB(2*i+1,:,:))
  end do


end subroutine kernel_first_kind_midpoint

subroutine kernel_first_kind_trapz(lenTraj, dim_basis,dim_x, kernel, k0, B, DxB,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxB
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(0.5*dt*B(0,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=k0

  do i=1,lenTraj !! for i in range(1, lenTraj):
     call trapz_integral(num,dt,i,B(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=-1*matmul(invB0,num+DxB(i,:,:))
  end do


end subroutine kernel_first_kind_trapz

subroutine kernel_first_kind_simpson(lenTraj, dim_basis,dim_x, kernel, k0, B, DxB,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::B
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxB
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(0.5*dt*B(0,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=k0

  kernel(1,:,:)=matmul(invB0,-1*DxB(1,:,:)-0.5*dt*matmul(B(1,:,:),kernel(0,:,:))) ! First point trapz rule

  invB0=inv(dt*B(0,:,:)/3.)

  do i=2,lenTraj
     call simpson_integral(num,dt,i,B(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=-1*matmul(invB0,num+DxB(i,:,:))
  end do


end subroutine kernel_first_kind_simpson


subroutine kernel_second_kind(lenTraj, dim_basis,dim_x, kernel, k0, B0, Bdot, DxBdot,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(dim_basis,dim_basis),intent(in)::B0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::Bdot
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxBdot
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(B0+0.5*dt*Bdot(0,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=k0

  do i=1,lenTraj !! for i in range(1, lenTraj):
     call trapz_integral(num,dt,i,Bdot(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=matmul(invB0,num+DxBdot(i,:,:))
  end do


end subroutine kernel_second_kind


subroutine kernel_second_kind_rect(lenTraj, dim_basis,dim_x, kernel, k0, B0, Bdot, DxBdot,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(dim_basis,dim_basis),intent(in)::B0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::Bdot
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxBdot
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(B0) ! Update this depending of integration rule

  kernel(0,:,:)=k0

  do i=1,lenTraj !! for i in range(1, lenTraj):
     call rect_integral(num,dt,i,Bdot(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=matmul(invB0,num+DxBdot(i,:,:))
  end do


end subroutine kernel_second_kind_rect


subroutine kernel_second_kind_trapz(lenTraj, dim_basis,dim_x, kernel, k0, B0, Bdot, DxBdot,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(dim_basis,dim_basis),intent(in)::B0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::Bdot
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxBdot
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(B0-0.5*dt*Bdot(0,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=k0

  do i=1,lenTraj !! for i in range(1, lenTraj):
     call trapz_integral(num,dt,i,Bdot(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=matmul(invB0,num+DxBdot(i,:,:))
  end do


end subroutine kernel_second_kind_trapz

subroutine kernel_second_kind_simpson(lenTraj, dim_basis,dim_x, kernel, k0, B0, Bdot, DxBdot,dt)
  use lapackMod
  implicit none
  integer,intent(in)::lenTraj,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(out)::kernel
  double precision,dimension(dim_basis,dim_x),intent(in)::k0
  double precision,dimension(dim_basis,dim_basis),intent(in)::B0
  double precision,dimension(0:lenTraj, dim_basis, dim_basis),intent(in)::Bdot
  double precision,dimension(0:lenTraj, dim_basis,dim_x),intent(in)::DxBdot
  double precision,intent(in)::dt
  double precision,dimension(dim_basis,dim_basis)::invB0
  double precision,dimension(dim_basis,dim_x)::num
  integer::i

  invB0=inv(B0+0.5*dt*Bdot(0,:,:)) ! Update this depending of integration rule

  kernel(0,:,:)=k0
  kernel(1,:,:)=matmul(invB0,DxBdot(1,:,:)+0.5*dt*matmul(Bdot(1,:,:),kernel(0,:,:))) ! First point trapz rule
  invB0=inv(B0+dt*Bdot(0,:,:)/3) ! Update this depending of integration rule

  do i=2,lenTraj !! for i in range(1, lenTraj):
     call simpson_integral(num,dt,i,Bdot(0:i,:,:),kernel(0:i,:,:),dim_basis,dim_x)
     kernel(i,:,:)=matmul(invB0,num+DxBdot(i,:,:))
  end do

end subroutine kernel_second_kind_simpson


! to_integrate = np.einsum("ik,ikl->il", E[: n + 1, :][::-1, :], self.kernel[: n + 1, :, :])
! memory[n] = -1 * trapezoid(to_integrate, dx=dt, axis=0)

subroutine memory_rect(lenTraj,len_mem, dim_basis, dim_x, memory, kernel, E, dt)
  implicit none
  integer,intent(in)::lenTraj,len_mem,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_x),intent(out)::memory
  double precision,dimension(0:len_mem, dim_basis,dim_x),intent(in)::kernel
  double precision,dimension(0:lenTraj, dim_basis),intent(in)::E
  double precision,intent(in)::dt
  integer::i,j

  memory(:,:)=0.

  do i=1,lenTraj !! for i in range(1, lenTraj):
    do j=0,min(i-1,len_mem)
       memory(i,:)=memory(i,:)-dt*matmul(E(i-j,:),kernel(j,:,:))
    end do
  end do


end subroutine memory_rect

subroutine memory_trapz(lenTraj,len_mem, dim_basis, dim_x, memory, kernel, E, dt)
  implicit none
  integer,intent(in)::lenTraj,len_mem,dim_basis,dim_x
  double precision,dimension(0:lenTraj, dim_x),intent(out)::memory
  double precision,dimension(0:len_mem, dim_basis,dim_x),intent(in)::kernel
  double precision,dimension(0:lenTraj, dim_basis),intent(in)::E
  double precision,intent(in)::dt
  integer::i,j

  memory(0,:)=0.

  do i=1,lenTraj !! for i in range(1, lenTraj):
    memory(i,:)=-0.5*dt*matmul(E(i,:),kernel(0,:,:))
    do j=1,min(i-1,len_mem)
       memory(i,:)=memory(i,:)-dt*matmul(E(i-j,:),kernel(j,:,:))
    end do
   memory(i,:)=memory(i,:)-0.5*dt*matmul(E(i-min(i,len_mem),:),kernel(min(i,len_mem),:,:))
  end do


end subroutine memory_trapz
