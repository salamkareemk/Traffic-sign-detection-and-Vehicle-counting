from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse
from django.http import HttpResponse
from django.contrib import messages
from vehiclecount.vehiclecount import VehicleCapture

import numpy as np
import cv2
from time import sleep

vehiclecapture = VehicleCapture()


def homepage(request):
    return render(request, "homepage.html")


def countpage(request):
    return render(request, "countpage.html")


def countpage2(request):
    return render(request, "countpage2.html")


def gen(camera):
    while True:
        frame = camera.vehicleDetect()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video(request):
    cars = vehiclecapture.vehicleDetect()
    print(cars)

    '''if(cars!=0):
        return render(request,'homepage.html',{'q':cars})'''
    # return redirect(countpage2)

    '''return HttpResponse("<h1> Total number of vehicle is {} </h1>".format(cars))'''
    '''return StreamingHttpResponse(gen(VehicleCapture()),
					content_type='multipart/x-mixed-replace; boundary=frame')'''


def countpagecheck(request):
    length_min = 80  # largura_min
    height_min = 80  # altura_min
    offset = 6
    position = 550  # pos_linha
    delay = 60
    detect = []  # detec
    carros = 0

    def pega_centro(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    # cap = cv2.VideoCapture('video.mp4')
    cap = cv2.VideoCapture(0)
    subtract = cv2.createBackgroundSubtractorMOG2()
    global no_of_cars
    while True:
        ret, frame1 = cap.read()
        tempo = float(1 / delay)
        sleep(tempo)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = subtract.apply(blur)
        dilate = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contorno, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, position), (1200, position), (255, 127, 0), 3)
        for (i, c) in enumerate(contorno):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_contorno = (w >= length_min) and (h >= height_min)  # validar_contorno
            if not validate_contorno:
                continue

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = pega_centro(x, y, w, h)  # centro
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                if y < (position + offset) and y > (position - offset):
                    carros += 1
                    cv2.line(frame1, (25, position), (1200, position), (0, 127, 255), 3)
                    detect.remove((x, y))
                    # print("car is detected : "+str(carros))
                    no_of_cars = str(carros)

        cv2.putText(frame1, "VEHICLE COUNT : " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow("Video Original", frame1)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    return render(request, 'countpage2.html', {'q': no_of_cars})