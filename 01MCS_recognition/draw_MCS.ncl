begin

filein = "/home/hhb/data/GRIDSAT/GRIDSAT-B1.2001.09.05.18.v02r01.nc"
data1 = addfile(filein, "r")
lat = data1->lat
lon = data1->lon

lon1 = new((/dimsizes(lon)/),float)
lon1(0:2570) = lon(2572:5142)           ; -180~180转换为0~360
lon1(2571:5142) = lon(0:2571) + 360

lat_b = dim_minind(abs(18-lat), 0)
lat_t = dim_minind(abs(30-lat), 0)
lon_l = dim_minind(abs(110-lon1), 0)
lon_r = dim_minind(abs(130-lon1), 0)

f = "/home/hhb/tc_cr/MCS/tbb.nc"
fin = addfile(f, "r")
tbb_tcsum = fin->tbb_tcsum

tbb_tcsum!1 = "lat"
tbb_tcsum&lat = lat(lat_b:lat_t)
tbb_tcsum!2 = "lon"
tbb_tcsum&lon = lon1(lon_l:lon_r)

tbb_tcsum(0,:,:) = tbb_tcsum(0,:,:)/62
tbb_tcsum(1,:,:) = tbb_tcsum(1,:,:)/38
tbb_tcsum(2,:,:) = tbb_tcsum(2,:,:)/19
printMinMax(tbb_tcsum, 0)
;==============================================================================
; creat plot
ff = "/home/hhb/tc_cr/MCS/GRIDSAT_sum_t"
wk = "png"
wk@wkWidth  = 2000
wk@wkHeight = 2000
plotmap = new(3, "graphic")
wks=gsn_open_wks(wk,ff)

; set resource
mpres                               = True
mpres@gsnDraw                       = False              ; don't draw
mpres@gsnFrame                      = False              ; don't advance frame
;mpres@gsnRightString                = "700hPa Temp(~F35~J~F~C) Height(dagpm)"
mpres@gsnMaximize                   = True

mpres@mpDataSetName                 = "Earth..4"   ; This new database contains
mpres@mpDataBaseVersion             = "MediumRes"  ; Medium resolution database
mpres@mpOutlineOn                   = True         ; Turn on map outlines
mpres@mpOutlineSpecifiers           = (/"China:states","Taiwan"/)       ;China:states

;mpres@mpProjection                  = "LambertConformal"
;mpres@mpLambertParallel1F           = 30.
;mpres@mpLambertParallel2F           = 60.
;mpres@mpLambertMeridianF            = 105.
mpres@mpLimitMode                   = "LatLon"
mpres@mpMinLatF                     = 18
mpres@mpMaxLatF                     = 30
mpres@mpMinLonF                     = 110
mpres@mpMaxLonF                     = 130
mpres@mpCenterLonF                  = 120
mpres@tmXBLabelFontHeightF          = 0.02     ;set the size of x-Axis words
mpres@tmYLLabelFontHeightF          = 0.02    ;set the size of y-Axis words

mpres@mpGeophysicalLineThicknessF   = 2.
mpres@mpNationalLineThicknessF      = 2.
mpres@mpProvincialLineThicknessF    = 2.
mpres@mpGeophysicalLineColor        = "gray20"
mpres@mpNationalLineColor           = "gray20"
mpres@mpProvincialLineColor         = "gray20"

mpres@mpLandFillColor               = "lightyellow"
mpres@mpOceanFillColor              = "lightcyan1"
mpres@mpInlandWaterFillColor        = "lightcyan1"
/;
mpres@mpGridAndLimbOn               = True
mpres@mpGridLonSpacingF             = 10
mpres@mpGridLatSpacingF             = 10
mpres@mpGridLineThicknessF          = 1
mpres@mpGridLineDashPattern         = 1
mpres@mpGridLineColor               = "grey30"
;/
mpres@pmTickMarkDisplayMode         = "Always"           ; turn on tickmarks

mpres@tmXBMinorOn                   = False
mpres@tmXTMinorOn                   = False
mpres@tmYLMinorOn                   = False
mpres@tmYRMinorOn                   = False


sres=True
sres@gsnFrame = False
sres@gsnDraw  = False

sres@cnFillOn = True
sres@cnFillPalette = "WhiteBlueGreenYellowRed"
sres@cnLineLabelsOn = False
;sres@cnLevelSelectionMode = "AutomaticLevels"
;sres@cnLevelSpacingF = 1
;sres@lbLabelStride = 1
sres@cnInfoLabelOn                 = False
sres@lbLabelBarOn                  = False

sres@cnFillPalette                 = "WhiteBlueGreenYellowRed"
sres@cnLinesOn                     = False
sres@cnLineLabelsOn                = False
sres@cnInfoLabelOn                 = False
sres@lbLabelsOn                    = True

sres@cnLevelSelectionMode = "ExplicitLevels"
sres@cnLevels             = fspan(0.05,0.60,12)
sres@cnFillColors         = (/0,30,55,70,100,120,135,146,155,162,180,200,222/)

x=(/115,125,125,115,115/)
y=(/27,27,21,21,27/)
lnres                             = True
lnres@gsLineColor                 = "orange"
lnres@gsLineThicknessF            = 10.0

;panel
pres                              = True
pres@gsnMaximize                  = True
pres@gsnPanelFigureStrings        = (/"(a) WNP","(b) SO","(c) WNP+SO"/) ; add strings to panel
pres@amJust                       = "TopLeft"
pres@gsnPanelLabelBar             = True
pres@lbLabelFontHeightF           = 0.015    ;set the size of label bar words
pres@vpWidthF                     = 0.40
pres@vpHeightF                    = 0.06
;pres@lbTitleOn                     =  True                ; turn on title
;pres@lbTitleString                 = "m/s"                ; title string
;pres@lbTitlePosition               = "Right"              ; title position
;pres@lbTitleFontHeightF            = 0.013                ; make title smaller
;pres@lbTitleDirection              = "Across"             ; title direction

pres@gsnPanelFigureStringsPerimOn                = True          ;去掉方框的边框
;pres@gsnPanelFigureStringsBackgroundFillColor    =-1              ;方框背景设置为透明
pres@amOrthogonalPosF                            =-0.48
pres@gsnPanelFigureStringsFontHeightF            = 0.013
;pres@gsnPanelYWhiteSpacePercent                  = 8
;pres@gsnPanelXWhiteSpacePercent                  = 5

plotmap(0) = gsn_csm_map(wks,mpres)
plott = gsn_csm_contour(wks,tbb_tcsum(0,:,:),sres)
dum1 = gsn_add_polyline(wks,plotmap(0),x,y,lnres)
overlay(plotmap(0),plott)

plotmap(1) = gsn_csm_map(wks,mpres)
plott = gsn_csm_contour(wks,tbb_tcsum(1,:,:),sres)
dum2 = gsn_add_polyline(wks,plotmap(1),x,y,lnres)
overlay(plotmap(1),plott)

plotmap(2) = gsn_csm_map(wks,mpres)
plott = gsn_csm_contour(wks,tbb_tcsum(2,:,:),sres)
dum3 = gsn_add_polyline(wks,plotmap(2),x,y,lnres)
overlay(plotmap(2),plott)

gsn_panel(wks, plotmap, (/1,3/), pres)
end