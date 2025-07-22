import argparse
import os
import cv2
import hashlib
import tensorflow as tf
from mtcnn import MTCNN

import ber

detector = MTCNN()

def read_last_line(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            return ""
def embed_new( frame_path, embed_path, l,start,end):
    detal = 300
    nMax = 18
    wx = 172
    wy = 1
    im_list = os.listdir(frame_path)
    im_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    n = 0
    for im_name in im_list:
        n= n+ 1
        if n <start or n > end:
            continue
        f = cv2.imread(frame_path + im_name, cv2.IMREAD_COLOR)
        image = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
        detect_res = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(detect_res) == 0:
            continue
        bbox = (detect_res[0]['box'][0], detect_res[0]['box'][1], detect_res[0]['box'][2], detect_res[0]['box'][3])
        with open(embed_path + '/loc.txt', 'a', encoding='utf-8') as f:
            f.write(im_name + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n")

        x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        image_face = image[y:y + height, x:x + width, :]
        fr = 256
        fc = 128
        M = 16
        if not os.path.exists(embed_path + 'crop_face/'):
            os.makedirs(embed_path + 'crop_face/')

        cv2.imwrite(embed_path+'crop_face/faceRGB.png', image_face,[cv2.IMWRITE_PNG_COMPRESSION,100])
        pp = './run_qingchuan.sh' + ' ./MCR/v93/ ' +embed_path+' '+'crop_face/faceRGB.png'+' '+embed_path+' '+str(fr)+' '+str(fc)+' '+str(M)
        os.system(pp)

        w = read_last_line(embed_path + 'hash.txt')
        print('hash:{}'.format(w))
        w1 = bin(x)[2:].zfill(11)[11 - 5:]
        w2 = bin(y)[2:].zfill(11)[11 - 5:]
        w3 = bin(width)[2:].zfill(11)[11 - 5:]
        w4 = bin(height)[2:].zfill(11)[11 - 5:]

        ww = w1 + w2 + w3 + w4 + w
        www = list(ww)
        print('crop_{}.png  '.format(im_name.split('\\')[-1].split('.')[0]))
        pp = './run_main_final.sh' + ' ./MCR/v93/ ' + str(y) + ' ' + str(x) + ' ' + str(height) + ' ' + str(
            width) + ' ' + str(nMax) + ' ' + str(wx) + ' ' + str(
            wy) + ' ' + frame_path + ' ' + embed_path + ' ' + str(im_name) + ' ' + str(detal) + ' ' + (
                 ww) + ' ' + str(l)
        os.system(pp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default='')

    parser.add_argument('--outp', type=str, default='')
    parser.add_argument('--l', type=int, default=32)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--e', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()

    l = args.l
    frame_path = str(args.inp)
    embed_path = str(args.outp)
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)
    embed_new(frame_path, embed_path, l, args.s, args.e)