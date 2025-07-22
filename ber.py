import argparse
import hashlib
import os
import numpy as np
import cv2

import tensorflow as tf
from mtcnn import MTCNN
detector = MTCNN()
def read_last_line(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            return ""
def ber_new(frame_path,out_path,l,s,e):

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    detal = 300
    nMax = 18
    wx = 172
    wy = 1
    im_list = os.listdir(frame_path)
    if os.path.exists(frame_path + 'psnr.txt'):
        im_list.remove('psnr.txt')
    if os.path.exists(frame_path + 'file_list.txt'):
        im_list.remove('file_list.txt')
    if os.path.exists(frame_path + 'w.txt'):
        im_list.remove('w.txt')
    if os.path.exists(frame_path + 'ber.txt'):
        im_list.remove('ber.txt')
    if os.path.exists(frame_path + 'embedw.txt'):
        im_list.remove('embedw.txt')
    if os.path.exists(frame_path + 'exactw.txt'):
        im_list.remove('exactw.txt')
    if os.path.exists(frame_path + 'hash.txt'):
        im_list.remove('hash.txt')
    num = 0
    for im_name in im_list:
        if len(im_name.split('.'))>1 and len(im_name.split('.tx'))==1:
            num = num+1
            if num<s or num >e:
                continue
            image = cv2.imread(frame_path+im_name,cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            md5sum = '0'
            detect_res = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if len(detect_res) == 0:
                continue
            bbox = (detect_res[0]['box'][0], detect_res[0]['box'][1], detect_res[0]['box'][2], detect_res[0]['box'][3])
            with open(out_path + '/loc.txt', 'a', encoding='utf-8') as f:
                f.write(im_name + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n")

            x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
            image_face = image[y:y + height, x:x + width, :]
            fr = 256
            fc = 128
            M = 16
            if not os.path.exists(out_path + 'crop_face/'):
                os.makedirs(out_path + 'crop_face/')
            cv2.imwrite(out_path+'crop_face/faceRGBExact.png', image_face,[cv2.IMWRITE_PNG_COMPRESSION,100])
            pp = './run_qingchuan.sh' + ' ./MCR/v93/ ' + ' '+out_path+' '+'crop_face/faceRGBExact.png'+' '+out_path+' '+str(fr)+' '+str(fc)+' '+str(M)
            os.system(pp)
            w = read_last_line(out_path + 'hash.txt')
            print('exacthash:{}'.format(w))

            w1 = bin(int(str(x), 10))[2:].zfill(11)[11-5:]
            w2 = bin(int(str(y), 10))[2:].zfill(11)[11-5:]
            w3 = bin(int(str(width), 10))[2:].zfill(11)[11-5:]
            w4 = bin(int(str(height), 10))[2:].zfill(11)[11-5:]
            xf = bin(int(str(x), 10))[2:].zfill(11)[:11-5]
            yf = bin(int(str(y), 10))[2:].zfill(11)[:11-5]
            widthf = bin(int(str(width), 10))[2:].zfill(11)[:11-5]
            heightf = bin(int(str(height), 10))[2:].zfill(11)[:11-5]

            fber = 0
            ww = w1 + w2 + w3 + w4 + w
            pp = './run_main_exact.sh'+' ./MCR/v93/ ' + str(y) + ' ' + str(x) + ' ' + str(height) + ' ' + str(width) + ' ' + str(nMax) + ' ' + str(wx) + ' ' + str(wy) + ' ' + frame_path + ' ' + out_path + ' ' + str(im_name) + ' ' + str(detal) + ' ' + (ww)+' '+str(l)
            os.system(pp)
            with open(out_path+'exactw.txt', "r") as ffww:
                last_line = ffww.readlines()[-1]
                info = last_line.strip("\n")

                x = int(xf+info[0:5],2)
                y = int(yf+info[5:10],2)
                width = int(widthf+info[10:15],2)
                height = int(heightf+info[15:20],2)

                www = info[20:]
                wl=list(www)
                image_face = image[y:y + height, x:x + width, :]

                cv2.imwrite(out_path+'crop_face/faceRelocal.png', image_face,[cv2.IMWRITE_PNG_COMPRESSION,100])
                pp = './run_qingchuan.sh' + ' ./MCR/v93/ ' + out_path + ' ' + 'crop_face/faceRelocal.png' + ' ' + out_path + ' ' + str(fr) + ' ' + str(fc) + ' ' + str(M)
                os.system(pp)
                facewww = read_last_line(out_path + 'hash.txt')
                print('relochash:{}'.format(facewww))
                be = 0
                for wi in range(0,len(facewww)):
                    if( facewww[ wi ] != www[ wi ]):
                        be+=1
                fber = be/len(facewww)
                berfile = open(out_path+'attack_ber.txt', 'a')
                berfile.write(im_name+" "+str(fber)+"\n")
                berfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default='')
    parser.add_argument('--l', type=int, default=32)
    parser.add_argument('--outp', type=str, default='')
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--e', type=int, default=1)

    args = parser.parse_args()

    l =  args.l
    frame_path = str(args.inp)
    embed_path = str(args.outp)
    s = args.s
    e = args.e
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)
    ber_new(frame_path, embed_path, l,s,e)