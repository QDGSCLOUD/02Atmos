;********* eccentricity = minor axis/major axis **********
;********* geometric center = avg(i) ave(j) **************
;********* major axis：距离最远的两个点连成的直线 ********
;********* minor axis：经过形心，与长轴垂直的直线 ********

undef("eccentricity")
function eccentricity(cdata,data,length,n)

begin
;================== select boundary points ======================
e     = new((/n/),float)
geo   = new((/n,2/),float)
area  = new((/n/),"float")
tbmin = new((/n/),float)
tbavg = new((/n/),float)
m = 0
data1d      = ndtooned(data)                                  ; 2-d data into 1-d
dsizes_data = dimsizes(data)

do t=1,n
xy         := ind_resolve( ind(data1d.eq.t), dsizes_data )    ; 返回的xy是一个二维数组kk*2,kk代表满足逻辑运算的个数,第0列表示y,第1列表示x
min_y = min(xy(:,0))                                          ; 边界点（划定minor axis的寻找区域）
max_y = max(xy(:,0))
min_x = min(xy(:,1))  
max_x = max(xy(:,1))  

bor_xy := new((/(max_y-min_y+1)*2,2/),integer)                 ; 储存图形的边界点,第0列表示y,第1列表示x
k = 0
do b=min_y,max_y
bor_xy(k,0) = b
bor_xy(k,1) = min(xy(ind(xy(:,0).eq.b),1))
k = k+1
bor_xy(k,0) = b
bor_xy(k,1) = max(xy(ind(xy(:,0).eq.b),1))
k = k+1
end do
kk = k

;================== geometric center =======================
geo_y = avg(xy(:,0))                                       ; geometric center = avg(i) ave(j)
geo_x = avg(xy(:,1))

;================== major axis =============================
major = 0.0
do a=0,kk-2
do b=a+1,kk-1
dist = sqrt((bor_xy(a,0)-bor_xy(b,0))^2+(bor_xy(a,1)-bor_xy(b,1))^2)
if (dist.gt.major) then
major = dist                                                   ; major axis
major_ys = bor_xy(a,0)
major_ye = bor_xy(b,0)
major_xs = bor_xy(a,1)
major_xe = bor_xy(b,1)
end if
end do
end do

;================== minor axis =============================
;在min_x,max_x,min_y,max_y框出的矩形区域内，寻找与major axis垂直，且过geometric center的直线与图形边界的交点
;由于交点不一定在格点上，因此用与交点最近的格点代替

mm = 9999
if (major_ye.ne.major_ys .and. major_xe.ne.major_xs) then
A = (-1.0)/((major_ye-major_ys)/tofloat((major_xe-major_xs)))
C = geo_y-A*geo_x

do i=min_x,max_x
minor_y1 = toint(A*i+C)
if (minor_y1.ge.min_y .and. minor_y1.le.max_y .and. data(minor_y1,i).eq.t) then       ; 寻找左边的minor axis point
break
end if
end do
minor_x1 = i

do j=max_x,min_x,1
minor_y2= toint(A*j+C)
if (minor_y2.ge.min_y .and. minor_y2.le.max_y .and. data(minor_y2,j).eq.t) then       ; 寻找右边的minor axis point
break
end if
end do
minor_x2 = j

else if (major_ye.eq.major_ys) then
do i=min_x,max_x
yy := ind(bor_xy(:,1).eq.i)
if (num(ismissing(yy)).eq.0 .and. (bor_xy(max(yy),0)-bor_xy(min(yy),0).lt.mm)) then
mm = bor_xy(max(yy),0)-bor_xy(min(yy),0)
minor_y2 = bor_xy(max(yy),0)
minor_y1 = bor_xy(min(yy),0)
minor_x2 = i
minor_x1 = i
end if
end do

else if (major_xe.eq.major_xs) then
do i=min_y,max_y
xx := ind(bor_xy(:,0).eq.i)
if (num(ismissing(xx)).eq.0 .and. (bor_xy(max(xx),1)-bor_xy(min(xx),1).lt.mm)) then
mm = bor_xy(max(xx),1)-bor_xy(min(xx),1)
minor_x2 = bor_xy(max(xx),1)
minor_x1 = bor_xy(min(xx),1)
minor_y2 = i
minor_y1 = i
end if
end do

end if
end if
end if
minor = sqrt((minor_y2-minor_y1)^2+(minor_x2-minor_x1)^2)

;================== eccentricity =============================
e(t-1) = minor/major                                             ; eccentricity

if (e(t-1).ge.0.4 .and. e(t-1).le.1) then
area(m)     = dimsizes(xy(:,0))*(0.07*111)^2
tbmin(m)    = min(ndtooned(cdata(xy(:,0),xy(:,1))))
tbavg(m)    = avg(ndtooned(cdata(xy(:,0),xy(:,1))))
geo(m,0)    = geo_y
geo(m,1)    = geo_x
m = m+1
else
data(xy(:,0),xy(:,1)) = 0
end if

end do

return([/data,e,area,tbmin,geo,m/])
;data:离心率合格的保留，否则设为0,e:离心率,area:通过筛选的区域面积,tbmin:亮温最低值,geo:MCS中心,m:通过离心率筛选的MCS个数

end
