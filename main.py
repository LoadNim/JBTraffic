import tkinter as tk
from tkinter import messagebox
import cv2
import pickle
import cvzone
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Timer
from plyer import notification as plyer_notification

# 버스 관련
def getBusInfo(busStation, busType):
    """
    인자 값을 바탕으로 해당하는 버스 시간표를 반환하는 함수
    :param busStation: 선택한 노선 (1: 삼송, 2: 백석)
    :param busType: 선택한 등학교 구분 (1: 등교, 2: 하교)
    :return: (pd.DataFrame) 조건에 맞는 정보를 DataFrame 형으로 반환
    """
    stationName = '삼송' if busStation == 1 else '백석'
    typeName = '등교' if busType == 1 else '하교'
    busData = pd.read_excel('./Data/BusData.xlsx')
    busData['시간'] = busData['시간'].astype(str)
    timeTable = busData[(busData['노선'] == stationName) & (busData['구분'] == typeName)]
    curTime = datetime.now().strftime('%H:%M:%S')
    timeTable = timeTable[timeTable['시간'] > curTime]
    return timeTable

# 알림 관련
notification = None  # 현재 설정된 알림 정보
notification_timer = None  # 예약된 알림 타이머

def send_windows_notification(bus_name, bus_type, bus_time):
    """
    윈도우 알림을 표시하는 함수
    :param bus_name: 버스 이름
    :param bus_type: 등교 또는 하교 정보
    :param bus_time: 버스 출발 시간
    """
    plyer_notification.notify(
        title="셔틀버스 알림",
        message=f"{bus_name} ({bus_type}) 출발 10분 전입니다.",
        app_name="JBTraffic",
        timeout=10
    )

def schedule_notification(bus_name, bus_type, bus_time):
    # 특정 시간에 알림을 예약하는 함수
    global notification_timer

    # 기존 알림이 있으면 취소
    if notification_timer is not None:
        notification_timer.cancel()

    now = datetime.now()
    bus_time_obj = datetime.strptime(bus_time, "%H:%M:%S").replace(year=now.year, month=now.month, day=now.day)
    notification_time = bus_time_obj - timedelta(minutes=10)    # 출발 10분 전
    delay = (notification_time - now).total_seconds()

    # 타이머 설정
    if delay > 0:
        notification_timer = Timer(delay, send_windows_notification, args=(bus_name, bus_type, bus_time))
        notification_timer.start()

def set_notification(bus_name, bus_type, bus_time):
    # 알림을 설정하는 함수
    global notification

    # 동일한 알림이면 취소
    if notification == (bus_name, bus_type, bus_time):
        cancel_notification()
        messagebox.showinfo("알림 취소", "같은 알림이 이미 설정되어 있어 취소되었습니다.")
    else:
        if notification is not None:
            messagebox.showinfo("알림 변경", "기존 알림을 새로운 알림으로 대체합니다.")
        notification = (bus_name, bus_type, bus_time)
        schedule_notification(bus_name, bus_type, bus_time)
        messagebox.showinfo("알림 설정", f"{bus_name} ({bus_type}) {bus_time} 알림이 설정되었습니다.")

def cancel_notification():
    # 현재 설정된 알림을 취소하는 함수
    global notification, notification_timer

    if notification_timer is not None:
        notification_timer.cancel()
        notification_timer = None
    notification = None
    messagebox.showinfo("알림 취소", "알림이 취소되었습니다.")

# 주차장 관련
def analyzeParkingLot(width, height, lotNumber):
    # 주차장 영상 데이터 불러오기
    cap = cv2.VideoCapture(f'./Data/ParkingLot/ParkingLot0{lotNumber}/carPark.mp4')

    # 주차공간 데이터 불러오기 (각 주차 공간의 시작 좌표 값)
    with open(f'./Data/ParkingLot/ParkingLot0{lotNumber}/CarParkPos', 'rb') as f:
        posList = pickle.load(f)

    # 기준 픽셀 수 설정 (주차 공간이 비어있다고 판단하기 위한 값)
    thresholdPixelCount = 900 if lotNumber == 1 else 400

    def countFreeSpaces(imgProcessed):
        """
        기준 픽셀 수에 따라 빈 주차 공간을 계산하고 시각화하여 제공하는 함수
        각 주차 공간별 픽셀 수를 분석하고, 빈 공간이면 초록색, 그렇지 않으면 빨간색으로 표시한다.
        :param imgProcessed: (numpy.ndarray) 전처리된 이진 이미지.
        :return: (int) 빈 주차 공간
        """
        spaceCounter = 0
        for pos in posList:
            x, y = pos
            imgCrop = imgProcessed[y:y + height, x:x + width]
            count = cv2.countNonZero(imgCrop)

            # 공간의 상태에 따라 시각적 표시
            if count < thresholdPixelCount:
                color = (0, 255, 0) # 빈 공간 (초록색)
                spaceCounter += 1
            else:
                color = (0, 0, 255) # 주차된 공간 (빨간색)

            # 주차 공간에 사각형과 픽셀 수 표시
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

        # 상단에 전체 빈 공간 수 표시
        cvzone.putTextRect(img, f'Free: {spaceCounter} / {len(posList)}', (100, 50), scale=3, thickness=5, offset=20,
                           colorR=(0, 255, 0))
        return spaceCounter

    while True:
        success, img = cap.read()
        if not success:
            break

        # 영상의 끝에 도달하면 다시 처음으로
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 이미지 처리
        imgProcessed = preprocessImage(img)
        freeSpaces = countFreeSpaces(imgProcessed)

        # 결과 출력
        cv2.imshow("Parking Status", img)
        key = cv2.waitKey(10)
        if key == 27:  # ESC 키
            break

    cap.release()
    cv2.destroyAllWindows()
    return f'{freeSpaces} / {len(posList)}'

def preprocessImage(img):
    """
    이미지 전처리 함수
    주차 공간을 탐지하기 위해 영상에서 캡처한 프레임을 전처리하는 함수. 전치리 과정은 다음과 같다.
    1. 그레이스케일 변환: 컬러 이미지를 흑백으로 변환한다. (효율과 처리속도를 위함)
    2. 블러링: 가우시안 블러를 적용하여 이미지의 노이즈를 줄임
    3. 임계처리: 이미지를 이진화하여, 주차 공간을 잘 구분할 수 있도록 한다.
    4. 미디언 블러: 임계처리 후 발생한 작은 노이즈를 제거한다.
    5. 팽창: 객체의 경계를 두껍게 만들어 주차 공간을 더 명확하게 구분할 수 있도록 한다.
    :param img: (numpy.ndarray) 영상에서 캡처한 한 프레임, 컬러 이미지이다.
    :return: (numpy.ndarray) 전처리된 이진 이미지.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 그레이스케일 변환
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)  # 가우시안 블러 적용
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16) # 임계처리
    imgMedian = cv2.medianBlur(imgThreshold, 5) # 미디언 블러 적용
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1) # 팽창 적용
    return imgDilate

# Tkinter GUI 함수들
def open_shuttle_window():
    shuttle_window = tk.Toplevel(app)
    shuttle_window.title("셔틀버스 시간표")
    shuttle_window.geometry("300x300")
    tk.Label(shuttle_window, text="셔틀버스 시간표 확인", font=("Arial", 14)).pack(pady=10)
    tk.Button(shuttle_window, text="삼송", command=lambda: show_shuttle_info(1)).pack(pady=5)
    tk.Button(shuttle_window, text="백석", command=lambda: show_shuttle_info(2)).pack(pady=5)
    tk.Button(shuttle_window, text="알림 확인", command=show_notifications).pack(pady=5)
    tk.Button(shuttle_window, text="뒤로가기", command=shuttle_window.destroy).pack(pady=10, side=tk.BOTTOM)

def show_shuttle_info(busStation):
    bus_window = tk.Toplevel(app)
    bus_window.title("버스 정보")
    bus_window.geometry("300x300")
    tk.Label(bus_window, text="등/하교 선택", font=("Arial", 14)).pack(pady=10)
    tk.Button(bus_window, text="등교", command=lambda: display_bus_table(busStation, 1)).pack(pady=5)
    tk.Button(bus_window, text="하교", command=lambda: display_bus_table(busStation, 2)).pack(pady=5)

def show_notifications():
    notification_window = tk.Toplevel(app)
    notification_window.title("설정된 알림")
    notification_window.geometry("300x300")
    tk.Label(notification_window, text="설정된 알림", font=("Arial", 14)).pack(pady=10)

    if notification:
        bus_name, bus_type, bus_time = notification
        notification_info = f"{bus_name} ({bus_type}) - {bus_time}"
        tk.Label(notification_window, text=notification_info).pack(pady=5)
        tk.Button(notification_window, text="알림 취소",
                  command=lambda: [cancel_notification(), notification_window.destroy()]).pack(pady=10)
    else:
        tk.Label(notification_window, text="설정된 알림이 없습니다.").pack()

def display_bus_table(busStation, busType):
    table_window = tk.Toplevel(app)
    table_window.title("시간표")
    table_window.geometry("400x400")
    timeTable = getBusInfo(busStation, busType)

    if not timeTable.empty:
        tk.Label(table_window, text="버스 시간표", font=("Arial", 14)).pack(pady=10)
        for _, row in timeTable.iterrows():
            bus_info = f"노선: {row['노선']} | 구분: {row['구분']} | 시간: {row['시간']}"
            frame = tk.Frame(table_window)
            frame.pack(pady=5, anchor="w")

            tk.Label(frame, text=bus_info).pack(side="left")
            tk.Button(frame, text="알림 설정", command=lambda r=row: set_notification(r['노선'], r['구분'], r['시간'])).pack(side="right")
    else:
        tk.Label(table_window, text="현재 이후의 버스 정보가 없습니다.").pack()

def open_parking_window():
    parking_window = tk.Toplevel(app)
    parking_window.title("주차장 정보")
    parking_window.geometry("300x300")
    tk.Label(parking_window, text="주차장 상태", font=("Arial", 14)).pack(pady=10)
    resultParking1 = analyzeParkingLot(107, 48, 1)
    resultParking2 = analyzeParkingLot(70, 30, 2)
    tk.Label(parking_window, text=f'주차장 1: {resultParking1}').pack(pady=5)
    tk.Label(parking_window, text=f'주차장 2: {resultParking2}').pack(pady=5)

# 메인 GUI
app = tk.Tk()
app.title("JBTraffic")
app.geometry("300x300")
tk.Label(app, text="JBTraffic", font=("Arial", 16)).pack(pady=20)
tk.Button(app, text="셔틀버스 시간 확인", command=open_shuttle_window).pack(pady=10)
tk.Button(app, text="주차장 남은 자리 확인", command=open_parking_window).pack(pady=10)
tk.Button(app, text="종료", command=app.quit).pack(pady=20)
app.mainloop()