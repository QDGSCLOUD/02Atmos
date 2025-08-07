undef("before")
function before(year,mon,day,hour)

begin

if (hour.eq.0) then        ; ����ǰ3h��Ӧ��ʱ��
  hourb3 = 21
  if (day.eq.1) then

    if (mon.eq.2 .or. mon.eq.4 .or. mon.eq.6 .or. mon.eq.8 .or. mon.eq.9 .or. mon.eq.11) then      ; �о���Χ���������£��������������
      dayb3  = 31                                                                            
      monb3  = mon-1                                                                                 ; if��mon����������ǵ�mon-1Ϊ31dayʱ��mon
    else if (mon.eq.5 .or. mon.eq.7 .or. mon.eq.10 .or. mon.eq.12) then
      dayb3  = 30
      monb3  = mon-1
    end if
    end if
  else
  dayb3  = day-1
  monb3  = mon
  end if 
else
  hourb3 = hour-3
  dayb3  = day
  monb3  = mon
end if

timeb3 = year*1000000+monb3*10000+dayb3*100+hourb3
return(timeb3)

end

