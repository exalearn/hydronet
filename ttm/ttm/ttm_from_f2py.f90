
subroutine ttm_from_f2py(modelNumber, numWaters, pos, derivatives, finalEnergy)
    use potential_mod
    implicit none

    integer :: modelNumber, numWaters
    double precision, dimension(3, 3*numWaters) :: pos
    double precision, dimension(3, 3*numWaters) :: derivatives
    double precision :: finalEnergy

!f2py    intent(in) modelNumber
!f2py    intent(in) numWaters
!f2py    intent(in) pos
!f2py    intent(out) derivatives
!f2py    intent(out) finalEnergy
    
    imodel=modelNumber

    call potential(numWaters, pos, derivatives, finalEnergy)

    return
end subroutine ttm_from_f2py
