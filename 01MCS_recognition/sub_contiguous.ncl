; dat is 2-D data in which nth means MCS ID meeting criteria and 0 not.
; s_min denotes the pixel number in the contiguous region.

undef("Contigous pixels")
function  Contiguous(dat[*][*]:integer, s_min[1]:numeric)
local  q, n, d, t, k, l, r, x, y, vL, Left, Right, js_a, js_b, z_f, z_b, rc_n, rc
begin
dat_c  =  dat

dsizes  =  dimsizes(dat)
rc_n  =  new((/dsizes/), "integer")
rc_n  =  0
rc  =  new((/dsizes/), "integer")
rc  =  0
out_rc  =  rc
q  =  dsizes(0)*dsizes(1)
n  =  0

;-----Calculate the number of conditions that the data meets, and their coordinates
do  i  =  0, dsizes(0)-1
  do  j  =  0, dsizes(1)-1
    z_f  =  new((/q, 2/), "integer")
    z_f  =  99999
    d  =  0
    t  =  0
    if  dat_c(i, j).eq.1  then
      z_f(n, :)  =  (/i, j/);储存满足MCS条件的坐标
      n  =  n+1
      do  c  =  t, q-1
        if  z_f(c, 0).eq.99999  then
          continue
        else
          k  =  z_f(c, 1);j列
          l  =  z_f(c, 0);i行
          z_f(c, :)  =  99999
          r  =  k
          ;从连续区域中的某一像素点所在行开始，由该点从右向左进行判断，符合条件的点设为0，直到到达左边界处
          do while((r.le.k).and.(r.ge.0))  ;Determine the leftmost coordinates；
            if  dat_c(l, r).eq.1  then
              if  r.eq.k  then
                Left  =  r
              else
                d  =  d+1;记录这行像素点左边连续区域像素点的个数
                dat_c(l, r)  =  0
              end if
              Left  =  where(r.eq.0, r, Left)
            else
              Left  =  r+1
              break
            end if
            r  =  r-1
          end do
          do  s  =  k, dsizes(1)-1  ;Determine the rightmost coordinates；从连续区域中的某一像素点所在行开始，由该点从左向右进行判断，符合条件的点设为0，直到到达右边界处
            if  dat_c(l, s).eq.1  then
              d  =  d+1;记录这行像素点右边连续区域像素点的个数
              dat_c(l, s)  =  0
              if  s.eq.(dsizes(1)-1)  then
                Right  =  s
              end if
            else
              Right  =  s-1
              break
            end if
          end do
          js_a  =  0
          do  a  =  Left, Right  ;the previous line
            if  l.gt.0  then
              if  dat_c(l-1, a).eq.1  then;判断像素点上方的点
                js_a  =  js_a+1
                if  a.eq.Right  then;记录边界点
                  z_f(n, 0)  =  l-1
                  z_f(n, 1)  =  a
                  n  =  n+1
                end if
              else
                if  js_a.ne.0  then;当像素点上方的点在区域内但不是边界点，判断该点右边的点，如果右边的点不在区域内，则确定该点为边界点
                  z_f(n, 0)  =  l-1
                  z_f(n, 1)  =  a-1
                  n  =  n+1
                  js_a  =  0
                end if
              end if
            else
              break
            end if
          end do
          js_b  =  0
          do  b  =  Left, Right  ;the next line判断像素点下方的点
            if  (l+1).lt.dsizes(0)  then
              if  dat_c(l+1, b).eq.1  then
                js_b  =  js_b+1
                if  b.eq.Right  then
                  z_f(n, 0)  =  l+1
                  z_f(n, 1)  =  b
                  n  =  n+1
                end if
              else
                if  js_b.ne.0  then
                  z_f(n, 0)  =  l+1
                  z_f(n, 1)  =  b-1
                  n  =  n+1
                  js_b  =  0
                end if
              end if
            else
              break
            end if
          end do
          t  =  0
        end if
      end do
    end if
     rc_n(i, j)  =  d;连续区域点数，每选一次就会有一个格点被填充成d，所以选出来的连续区域不会重复。
  end do
end do
delete(z_f)

m  =  0
do  i  =  0, dsizes(0)-1
  do  j  =  0, dsizes(1)-1
    if  rc_n(i, j).ge.s_min  then;当连续区域面积满足要求时，记录为一个MCS
      m  =  m+1
    end if
  end do
end do

max_rc_n  =  max(rc_n)
mm = ispan(1, m, 1)

;-----reduction
if  .not.(all(ismissing(ind(ndtooned(rc_n).ge.s_min))))  then;如果满足面积要求的区域内不全都是缺失值
  max_ind  =  ind_resolve(ind(ndtooned(rc_n).ge.s_min), dsizes);确定的MCS坐标
  dim_ind  =  dimsizes(max_ind)
  n  =  0
  do  w  =  0, dim_ind(0)-1;一共dim_ind(0)个MCS
    z_b  =  new((/q, 2/), "integer")
    z_b  =  99999
    z_b(n, :)  =  max_ind(w, :);第w个符合条件的像元的坐标
    n  =  n+1
    t  =  0;t一直都是0
    do  c  =  t, q-1
      if  z_b(c, 0).eq.99999  then
        continue
      else
        x  =  z_b(c, 0)
        y  =  z_b(c, 1)
        z_b(c, :)  =  99999
        vL  =  y
        do while((vL.le.y).and.(vL.ge.0))  ;Determine the leftmost coordinates
          if  dat(x, vL).eq.1  then
            if  vL.eq.y  then
              Left  =  y
            else
              rc(x, vL)  =  mm(w)  ;dat(x, vL)
              out_rc(x, vL)  =  rc(x, vL)
              dat(x, vL)  =  0
            end if
            Left  =  where(vL.eq.0, vL, Left)
          else
            Left  =  vL+1
            break
          end if
          vL  =  vL-1
        end do
        do  vR  =  y, dsizes(1)-1  ;Determine the rightmost coordinates
          if  dat(x, vR).eq.1  then
            rc(x, vR)  =  mm(w)   ;dat(x, vR)；编号
            out_rc(x, vR)  =  rc(x, vR)
            dat(x, vR)  =  0
            if  vR.eq.(dsizes(1)-1)  then
              Right  =  vR
            end if
          else
            Right  =  vR-1
            break
          end if
        end do
        js_a  =  0
        do  a  =  Left, Right  ;the previous line
          if  x.gt.0  then
            if  dat(x-1, a).eq.1  then
              js_a  =  js_a+1
              if  a.eq.Right  then
                z_b(n, 0)  =  x-1
                z_b(n, 1)  =  a
                n  =  n+1
              end if
            else
              if  js_a.ne.0  then
                z_b(n, 0)  =  x-1
                z_b(n, 1)  =  a-1
                n  =  n+1
                js_a  =  0
              end if
            end if
          else
            break
          end if
        end do
        js_b  =  0
        do  b  =  Left, Right  ;the next line
          if  (x+1).lt.dsizes(0)  then
            if  dat(x+1, b).eq.1  then
              js_b  =  js_b+1
              if  b.eq.Right  then
                z_b(n, 0)  =  x+1
                z_b(n, 1)  =  b
                n  =  n+1
              end if
            else
              if  js_b.ne.0  then
                z_b(n, 0)  =  x+1
                z_b(n, 1)  =  b-1
                n  =  n+1
                js_b  =  0
              end if
            end if
          else
            break
          end if
        end do
        t  =  0
      end if
    end do
  end do
  rc@number  = m
  return(rc)
else
  rc@number  = 0
  return(rc)
end if
end