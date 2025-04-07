import ephem
import datetime
import math

def calculate_solar_angles(latitude, longitude, date_time):
    # 创建Observer对象，表示观测点
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = date_time

    # 创建Sun对象，表示太阳
    sun = ephem.Sun(observer)

    # 计算太阳位置
    solar_altitude = math.degrees(sun.alt)  # 太阳高度角
    solar_azimuth = math.degrees(sun.az)   # 太阳方位角

    # 天顶角的计算
    solar_zenith = 90 - solar_altitude

    return solar_altitude, solar_azimuth, solar_zenith

# 输入观测点的经纬度
# latitude = 37.7749  # 举例：旧金山的纬度
# longitude = -122.4194  # 举例：旧金山的经度
latitude = 32.04  # 举例：南京的纬度
longitude = 118.78  # 举例：南京的经度

# 输入日期和时间
current_time = datetime.datetime.now()

# 计算太阳位置
solar_altitude, solar_azimuth, solar_zenith = calculate_solar_angles(latitude, longitude, current_time)

# 打印结果
print(f"At {current_time}, Solar Elevation Angle: {solar_altitude:.2f} degrees")
print(f"At {current_time}, Solar Azimuth Angle: {solar_azimuth:.2f} degrees")
print(f"At {current_time}, Solar Zenith Angle: {solar_zenith:.2f} degrees")