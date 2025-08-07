;************** whether D<1000 km **********************
;-3h的形心与0h的冷云团点之间的距离若小于1000km，则认为是
undef("sustain_3h")
function sustain_3h(cdata1,data1,e1,m1,e2,a2,tb2,g2,m2)

begin
if (m1.ne.0 .and. m2.ne.0) then
x = ind(e1.ge.0.4 .and. e1.le.1)+1
data1d      = ndtooned(data1)                                    ; 2-d data into 1-d
dsizes_data = dimsizes(data1)

do i=0,m1-1
xy   := ind_resolve( ind(data1d.eq.x(i)), dsizes_data )          ; 返回的xy是一个二维数组kk*2,kk代表满足逻辑运算的个数,第0列表示y,第1列表示x
if (num(ismissing(xy)).eq.0) then
min_y = min(xy(:,0))                                             ; 边界点（划定minor axis的寻找区域）
max_y = max(xy(:,0))
min_x = min(xy(:,1))  
max_x = max(xy(:,1))  

;=========================================
geo_y = avg(xy(:,0))                                       
geo_x = avg(xy(:,1))

area     = dimsizes(xy(:,0))*(0.07*111)^2
tbmin    = min(ndtooned(cdata1(xy(:,0),xy(:,1))))

;============= ����D ============================
D = new((/m2/),float)

do j=0,m2-1
D(j) = sqrt((geo_y-g2(j,0))^2+(geo_x-g2(j,1))^2)*111
end do
if (num(D.le.1000).eq.0) then
data1(xy(:,0),xy(:,1)) = 0
end if

else
data1 = 0
end if
end do

else
data1 = 0
end if

return(data1)

end