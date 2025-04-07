import datetime
import math


def calculate_julian_day(year, month, day, hour=12, minute=0, second=0):
    dt = datetime.datetime(year, month, day, hour, minute, second)
    return dt.toordinal() + 1721425.5


def calculate_sun_position(julian_day):
    # 计算世纪数
    centuries = (julian_day - 2451545.0) / 36525.0

    # 计算太阳的平黄经
    mean_longitude = (280.46646 + centuries * (36000.76983 + centuries * 0.0003032)) % 360

    # 计算太阳的平近点角
    mean_anomaly = 357.52911 + centuries * (35999.05029 - 0.0001537 * centuries)

    # 计算黄道倾斜角
    ecliptic_inclination = 23.439292

    # 计算太阳的方程时
    equation_of_time = 1 / 15 * (mean_longitude - 180 + (
                math.sin(math.radians(mean_anomaly)) * (1.914602 - centuries * (0.004817 + 0.000014 * centuries))) + (
                                             math.sin(math.radians(2 * mean_anomaly)) * (
                                                 0.019993 - 0.000101 * centuries)) + (
                                             math.sin(math.radians(3 * mean_anomaly)) * 0.000289))

    # 计算真太阳时
    true_solar_time = datetime.datetime.utcfromtimestamp((julian_day - 2440587.5) * 86400) + datetime.timedelta(
        minutes=equation_of_time)

    # 计算太阳时角
    hour_angle = ((true_solar_time.hour - 12) * 15) + (true_solar_time.minute / 4) + (true_solar_time.second / 240)

    # 计算太阳高度
    solar_declination = math.degrees(
        math.asin(math.sin(math.radians(ecliptic_inclination)) * math.sin(math.radians(mean_anomaly))))
    solar_elevation = math.degrees(math.asin(
        math.sin(math.radians(solar_declination)) * math.sin(math.radians(23.45)) + math.cos(
            math.radians(solar_declination)) * math.cos(math.radians(23.45)) * math.cos(math.radians(hour_angle))))

    return solar_elevation


# 示例
year, month, day = 2023, 1, 1
julian_day = calculate_julian_day(year, month, day)
sun_elevation = calculate_sun_position(julian_day)

print(f"On {year}-{month:02d}-{day:02d}, the solar elevation is {sun_elevation:.2f} degrees.")
