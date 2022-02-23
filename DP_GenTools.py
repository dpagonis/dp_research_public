import math

EARTHRADIUS = 6.371e6 #m
SUTHERLAND = 110 #K

def DP_DistFromLL(Coord1,Coord2):
    #return distance in m given tuples with LAT,LON in degrees


    Lat1 = Coord1[0] * math.pi / 180 #in radians
    Lon1 = Coord1[1] * math.pi / 180
    Lat2 = Coord2[0] * math.pi / 180
    Lon2 = Coord2[1] * math.pi / 180

    dist_rad = 2 * math.asin(math.sqrt((math.sin((Lat1-Lat2)/2))**2 + math.cos(Lat1)*math.cos(Lat2)*(math.sin((Lon1-Lon2)/2))**2))
    dist_m = dist_rad * EARTHRADIUS

    return dist_m

def DP_MeanFP(Temp,Pres_mbar):
    #input T (K) and P (mbar)
    #return mean free path of air in m
    return 6.65e-8 * (1013.25/Pres_mbar) * (Temp / 273.15) * (1+SUTHERLAND/273.15) / (1+SUTHERLAND/Temp)

def DP_Kn(Temp,Pres_mbar,diam_m):
    #input T (K), P (mbar), diameter (m)
    #return Knudsen number
    return 2 * DP_MeanFP(Temp,Pres_mbar) / diam_m

#test comment
