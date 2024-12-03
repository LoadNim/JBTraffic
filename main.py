import cv2
import pickle
import cvzone
import numpy as np
import pandas as pd
from datetime import datetime

def displayBusInfo():
    # 원하는 노선과 등하교를 입력받아 해당하는 정보를 출력하는 함수
    while True:
        busStation = int(input('1. 삼송, 2. 백석 '))
        busType = int(input('1. 등교, 2. 하교 '))
        if busStation not in [1, 2] or busType not in [1, 2]:
            print('잘못된 입력 값 입니다.')
            break
        getBusInfo(busStation, busType)

def getBusInfo(busStation, busType):
    """
    인자 값을 바탕으로 해당하는 버스 시간표를 반환하는 함수

    :param busStation: 선택한 노선 (1: 삼송, 2: 백석)
    :param busType: 선택한 등하교 구분 (1: 등교, 2: 하교)
    :return: (pd.DataFrame) 조건에 맞는 DataFrame을 반환
    """

    stationName = '삼송' if busStation == 1 else '백석'
    typeName = '등교' if busType == 1 else '하교'
    busData = pd.read_excel('./Data/BusData.xlsx')
    busData['시간'] = busData['시간'].astype(str)
    timeTable = busData[(busData['노선'] == stationName) & (busData['구분'] == typeName)]
    curTime = datetime.now().strftime('%H:%M:%S')
    timeTable = timeTable[timeTable['시간'] > curTime]
    print(f'현재 시간 {curTime} 이후의 {stationName}역 {typeName}버스 정보')
    print(timeTable)


def displayParkingStatus():
    # 각 주차장의 주차 공간을 분석하고 출력하는 함수
    resultParking2 = analyzeParkingLot(70, 30, 2)
    resultParking1 = analyzeParkingLot(107, 48, 1)
    print(f'주차장 1 : {resultParking1}')
    print(f'주차장 2 : {resultParking2}')


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


while True:
    selMenu = int(input('메뉴 입력 (1: 셔틀 버스 정보, 2: 주차장 정보) '))
    if selMenu == 1:
        displayBusInfo()
    elif selMenu == 2:
        displayParkingStatus()
    else:
        break