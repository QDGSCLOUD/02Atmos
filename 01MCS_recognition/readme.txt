pro_MCS.ncl 主程序，读取数据
sub_contiguous.ncl 识别连续区域
sub_eccentricity.ncl 识别长短轴，长轴为连续区域中相隔最远两点距离，短轴垂直于长轴
sub_before3h.ncl 读取三小时前亮温数据
sub_sustain_3h.ncl 要求MCS维持三小时以上，三小时内相隔一定距离之内认为是同一MCS，这个标准有不同的判别方法，比如光流法。


