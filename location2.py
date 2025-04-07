import datetime
import math


def calculate_julian_day(year, month, day, hour=12, minute=0, second=0):
    dt = datetime.datetime(year, month, day, hour, minute, second)
    return dt.toordinal() + 1721425.5


def calculate_solar_declination(julian_day):
    centuries = (julian_day - 2451545.0) / 36525.0

    mean_anomaly = 357.52911 + centuries * (35999.05029 - 0.0001537 * centuries)

    ecliptic_inclination = 23.439292

    solar_declination = math.degrees(
        math.asin(math.sin(math.radians(ecliptic_inclination)) * math.sin(math.radians(mean_anomaly))))

    return solar_declination


def calculate_solar_azimuth(latitude, solar_declination, hour_angle):
    solar_altitude = math.radians(90) - math.radians(latitude) + math.radians(solar_declination)

    cos_azimuth = (math.sin(solar_altitude) * math.sin(math.radians(hour_angle))) / math.cos(solar_altitude)
    sin_azimuth = math.cos(math.radians(solar_declination)) * math.sin(math.radians(hour_angle))

    solar_azimuth = math.degrees(math.atan2(sin_azimuth, cos_azimuth))

    return solar_azimuth


def calculate_solar_coordinates(year, month, day, latitude, longitude, hour=12, minute=0, second=0):
    julian_day = calculate_julian_day(year, month, day, hour, minute, second)
    solar_declination = calculate_solar_declination(julian_day)

    solar_time = hour + minute / 60 + second / 3600 + 4 * (longitude / 60)
    hour_angle = 15 * (solar_time - 12)

    solar_azimuth = calculate_solar_azimuth(latitude, solar_declination, hour_angle)

    return solar_declination, solar_azimuth


# 示例
year, month, day = 2023, 1, 30
latitude, longitude = 0, 0  # 假设观察点在赤道上
solar_declination, solar_azimuth = calculate_solar_coordinates(year, month, day, latitude, longitude)

print(f"On {year}-{month:02d}-{day:02d}, at {latitude}°N, {longitude}°E:")
print(f"Solar Declination: {solar_declination:.2f}°")
print(f"Solar Azimuth: {solar_azimuth:.2f}°")