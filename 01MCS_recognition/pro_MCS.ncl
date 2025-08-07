load"/home/hhb/tc_cr/MCS/sub_contiguous.ncl"
load"/home/hhb/tc_cr/MCS/sub_eccentricity.ncl"
load"/home/hhb/tc_cr/MCS/sub_before3h.ncl"; 
load"/home/hhb/tc_cr/MCS/sub_sustain_3h.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/time_axis_labels.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin

filew = "/home/hhb/tc_cr/E-P/class/wnp.txt"
files = "/home/hhb/tc_cr/E-P/class/so.txt"
filen = "/home/hhb/tc_cr/E-P/class/other.txt"
fileo = "/home/hhb/tc_cr/E-P/class/o2.txt"
tcw=asciiread(filew,-1,"string")
tcs=asciiread(files,-1,"string")
tco=asciiread(fileo,-1,"string")
tcn=asciiread(filen,-1,"string")

wnp_y="20"+str_get_cols(tcw, 0, 1)
wnp_m=tostring(sprinti("%0.2i",toint(str_get_cols(tcw, 2, 3))))
wnp_d=tostring(sprinti("%0.2i",toint(str_get_cols(tcw, 4, 5))))
times_wnp = wnp_y + wnp_m + wnp_d
delete([/wnp_y,wnp_m,wnp_d/])

so_y="20"+str_get_cols(tcs, 0, 1)
so_m=tostring(sprinti("%0.2i",toint(str_get_cols(tcs, 2, 3))))
so_d=tostring(sprinti("%0.2i",toint(str_get_cols(tcs, 4, 5))))
times_so = so_y + so_m + so_d
delete([/so_y,so_m,so_d/])

o2_y="20"+str_get_cols(tco, 0, 1)
o2_m=tostring(sprinti("%0.2i",toint(str_get_cols(tco, 2, 3))))
o2_d=tostring(sprinti("%0.2i",toint(str_get_cols(tco, 4, 5))))
times_o2 = o2_y + o2_m + o2_d
delete([/o2_y,o2_m,o2_d/])

ot_y="20"+str_get_cols(tcn, 0, 1)
ot_m=tostring(sprinti("%0.2i",toint(str_get_cols(tcn, 2, 3))))
ot_d=tostring(sprinti("%0.2i",toint(str_get_cols(tcn, 4, 5))))
times_ot = ot_y + ot_m + ot_d
delete([/ot_y,ot_m,ot_d,filew,files,filen,fileo,tcw,tcs,tco,tcn/])

;===============================================
file_se = "/home/hhb/tc_cr/TD0_cr_newlist.txt"

tc_tt = asciiread(file_se, -1, "string")

yys = toint(str_get_field(tc_tt,2," "))
mms = toint(str_get_field(tc_tt,3," "))
dds = toint(str_get_field(tc_tt,4," "))
hhs = toint(str_get_field(tc_tt,5," "))
times = yys + tostring(sprinti("%0.2i",mms)) + tostring(sprinti("%0.2i",dds)) ;+ sprinti("%0.2i",hhs)
delete([/yys,mms,dds/])

yye = toint(str_get_field(tc_tt,8," "))
mme = toint(str_get_field(tc_tt,9," "))
dde = toint(str_get_field(tc_tt,10," "))
hhe = toint(str_get_field(tc_tt,11," "))
timee = yye + tostring(sprinti("%0.2i",mme)) + tostring(sprinti("%0.2i",dde)) ;+ sprinti("%0.2i",hhe)
delete([/yye,mme,dde,file_se,tc_tt/])

;===============================================
file_all = "/home/hhb/tc_cr/TD_cr_newlist.txt"

tc_ta = asciiread(file_all, -1, "string")
ya = str_get_field(tc_ta,1," ")
ma = sprinti("%0.2i",toint(str_get_field(tc_ta,2," ")))
da = sprinti("%0.2i",toint(str_get_field(tc_ta,3," ")))
ha = sprinti("%0.2i",toint(str_get_field(tc_ta,4," ")))
lata = sprinti("%0.2i",toint(str_get_field(tc_ta,5," ")))
lona = sprinti("%0.2i",toint(str_get_field(tc_ta,6," ")))
time = ya + ma + da + ha
delete([/tc_ta,file_all/])

;=================================================
fname1   = "/home/hhb/data/GRIDSAT/GRIDSAT-B1."
fname2   = ".v02r01.nc"

mlat = 173
mlon = 287 ; 北纬18-30，东经110-130
;len = 131 ; 研究范围:以TC为中心r=500 km
con = 83  ; MCS筛选条件1: s>5000 km2
;con = 663  ; MCS筛选条件1: s>40000 km2

filein = "/home/hhb/data/GRIDSAT/GRIDSAT-B1.2001.09.05.18.v02r01.nc"
data1 = addfile(filein, "r")
lat = data1->lat
lon = data1->lon

nlat = dimsizes(lat)
nlon = dimsizes(lon)
lon1 = new((/dimsizes(lon)/),float)
lon1(0:2570) = lon(2572:5142)           ; -180~180转换为0~360
lon1(2571:5142) = lon(0:2571) + 360

lat_b = dim_minind(abs(18-lat), 0)
lat_t = dim_minind(abs(30-lat), 0)
lon_l = dim_minind(abs(110-lon1), 0)
lon_r = dim_minind(abs(130-lon1), 0)

OutFile1 = "/home/hhb/tc_cr/MCS/tbbavg_n.txt"
;=================================================
tbb_tcsum = new((/3,mlat,mlon/),float)
do cat = 0,2

if cat.eq.0 then
    dsize := times_wnp
    char := "wnp"
else if cat.eq.1 then
    dsize := times_so
    char := "so"
else if cat.eq.2 then
    dsize := times_o2
    char := "o2"
end if
end if
end if

;=================================================

tbb_sum := new((/dimsizes(dsize),mlat,mlon/),float)
tbb_sum = 0

do j = 0,dimsizes(dsize)-1

time_ind := ind(toint(times).eq.toint(dsize(j)))
ts := times(time_ind) + tostring(sprinti("%0.2i",hhs(time_ind)))
te := timee(time_ind) + tostring(sprinti("%0.2i",hhe(time_ind)))
lat_all := lata(ind(time.ge.ts .and. time.le.te))
lon_all := lona(ind(time.ge.ts .and. time.le.te))
time_all := time(ind(time.ge.ts .and. time.le.te))
timef  := stringtochar(time_all)

if dimsizes(dimsizes(timef)).ne.1 then
year   := chartostring(timef(:,0:3))
mon    := chartostring(timef(:,4:5))
day    := chartostring(timef(:,6:7))
hour   := chartostring(timef(:,8:9))
else
year   := chartostring(timef(0:3))
mon    := chartostring(timef(4:5))
day    := chartostring(timef(6:7))
hour   := chartostring(timef(8:9))
end if

do i = 0,dimsizes(time_all)-1
filein = fname1 + year(i) + "." + mon(i) + "." + day(i) + "." + hour(i) + fname2
;print(i+" "+filein)

data = addfile(filein, "r")
scale_factor = 0.01
add_offset   = 200

tbb1 = data->irwin_cdr(0,:,:)*scale_factor+add_offset
tbb  := new((/nlat,nlon/),float)
tbb(:,0:2570) = tbb1(:,2572:5142)           ; -180~180转换为0~360
tbb(:,2571:5142) = tbb1(:,0:2571)
tbb!0  = "lat"
tbb&lat = lat
tbb!1  = "lon"
tbb&lon = lon1

cut_tbb1 = new((/mlat,mlon/),float)
cut_tbb1 = tbb(lat_b:lat_t,lon_l:lon_r)
poisson_grid_fill(cut_tbb1, True, 1, 100, 1e-1, 0.6, 0)                 ; 插值

;======================== main =========================
pass_data   = where((cut_tbb1.le.214), 1, 0)                           ; MCS筛选条件2: tbb<214 K(-59.15℃)
copy_VarMeta(cut_tbb1,pass_data)
;ii   := ind_resolve(ind(ndtooned(pass_data).eq.1), dimsizes(pass_data))
;new1 := cut_tbb1(ii(:,0),ii(:,1))

return_data = Contiguous(pass_data, con);筛选连续区域设每个格点为m
if return_data@number.gt.0 then
number1   := toint(return_data@number)
comp1     := eccentricity(cut_tbb1,return_data,nlat,number1)
new_data1 := comp1[0]
ec1       := comp1[1];离心率
m1        = comp1[5];通过离心率筛选的MCS个数

;====================================================
if (m1.gt.0 .and. num(new_data1.ne.0).ne.0) then
timeb3 := tostring(before(toint(year(i)),toint(mon(i)),toint(day(i)),toint(hour(i))))
monb3  := str_get_cols(timeb3,4,5)
dayb3  := str_get_cols(timeb3,6,7)
hourb3 := str_get_cols(timeb3,8,9)

fin = fname1 + year(i) + "." + monb3 + "." + dayb3 + "." + hourb3 + fname2 ; before 3h GRIDSAT数据
datab = addfile(fin,"r")
tbb1 = datab->irwin_cdr(0,:,:)*scale_factor+add_offset
tbb  := new((/nlat,nlon/),float)
tbb(:,0:2570) = tbb1(:,2572:5142)                         ; -180~180转换为0~360
tbb(:,2571:5142) = tbb1(:,0:2571)
tbb!0  = "lat"
tbb&lat = lat
tbb!1  = "lon"
tbb&lon = lon1

cut_tbb2 = new((/mlat,mlon/),float)
cut_tbb2 = tbb(lat_b:lat_t,lon_l:lon_r)
poisson_grid_fill(cut_tbb2, True, 1, 100, 1e-1, 0.6, 0)   ; 插值,将网格中所有缺失（_FillValue）的值替换为通过松弛求解泊松方程得出的值

;====================================================
pass_data   := where((cut_tbb2.le.214), 1, 0)              ; MCS筛选条件2: tbb<214 K(-59.15℃)
return_data := Contiguous(pass_data, con)

if return_data@number.gt.0 then
number2   := toint(return_data@number)
comp2     := eccentricity(cut_tbb2,return_data,nlat,number2)
new_data2 := comp2[0]
ec2       := comp2[1]
area2     := comp2[2]
tbmin2    := comp2[3]
geo2      := comp2[4]
m2        := comp2[5]
if (m2.eq.0) then
number2   = 0
new_data2 = 0
end if
else
number2   = 0
new_data2 = 0
ec2       = 0
area2     = 0
tbmin2    = 0
geo2      = 0
m2        = 0
end if

else
number1   = 0
new_data1 = 0
number2   = 0
new_data2 = 0
ec2       = 0
area2     = 0
tbmin2    = 0
geo2      = 0
m2        = 0
end if

else
number1   = 0
new_data1 = 0
ec1       = 0
m1        = 0
number2   = 0
new_data2 = 0
ec2       = 0
area2     = 0
tbmin2    = 0
geo2      = 0
m2        = 0
end if

new_data := sustain_3h(cut_tbb1,new_data1,ec1,m1,ec2,area2,tbmin2,geo2,m2)
tbb_2d := where(new_data.gt.0, 1, 0)

if .not.all(ismissing(ind(ndtooned(tbb_2d).gt.0))) then
tbavg_2d := ind_resolve(ind(ndtooned(tbb_2d).gt.0), dimsizes(tbb_2d))
npts = dimsizes(tbavg_2d(:,0))
tb := new(npts,float)
do m = 0,npts-1
tb(m) = cut_tbb1(tbavg_2d(i,0),tbavg_2d(i,1));用循环获取满足条件的数据，否则可能由于数据网格的问题导致最终数据超出条件
end do
tbavg = avg(tb)
else
tbavg = 0
end if
write_table(OutFile1,"a",[/cat,tbavg/],"%0.1i %4.1f")

tbb_sum(j,:,:) = tbb_sum(j,:,:)+tbb_2d
;tbb_sum(j,:,:) = where((tbb_sum(j,:,:).ne.0),tbb_sum(j,:,:),tbb_sum@_FillValue)
copy_VarMeta(cut_tbb1, tbb_sum(j,:,:))

end do;i

end do;j
printVarSummary(tbb_sum)
tbb_tcsum(cat,:,:) = dim_sum_n(tbb_sum,0)

delete([/filein,data,tbb1,pass_data,return_data,fin,datab,tbb_2d/])
end do;cat

;===========================================================

fout0 =  "/home/hhb/tc_cr/MCS/tbb.nc"
system("rm -rf "+fout0)
fout = addfile(fout0,"c")

fout->tbb_tcsum = tbb_tcsum


end