module qif_sfa

double precision :: PI = 4.0*atan(1.0)

contains


subroutine qif_run(t,y,dy,tau,Delta,I_ext,eta,tau_a,alpha,weight)

implicit none

double precision, intent(in) :: t
double precision, intent(in) :: y(4)
double precision :: r
double precision :: v
double precision :: a
double precision :: x
double precision :: r_in
double precision, intent(inout) :: dy(4)
double precision, intent(in) :: tau
double precision, intent(in) :: Delta
double precision, intent(in) :: I_ext
double precision, intent(in) :: eta
double precision, intent(in) :: tau_a
double precision, intent(in) :: alpha
double precision, intent(in) :: weight


r = y(1)
v = y(2)
a = y(3)
x = y(4)
r_in = r*weight

dy(1) = (Delta/(pi*tau) + 2.0*r*v)/tau
dy(2) = (I_ext - a + eta - pi**2*r**2*tau**2 + r_in*tau + v**2)/tau
dy(3) = x/tau_a
dy(4) = -a/tau_a + alpha*r - 2.0*x/tau_a

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use qif_sfa
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call qif_run(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7))

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = 1.0  ! tau
args(2) = 2.0  ! Delta
args(3) = 0.0  ! I_ext
args(4) = -10.0  ! eta
args(5) = 10.0  ! tau_a
args(6) = 1.0  ! alpha
args(7) = 21.213203435596427  ! weight
y(1) = 0.01  ! r
y(2) = -2.0  ! v
y(3) = 0.0  ! a
y(4) = 0.0  ! x

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
