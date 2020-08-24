import os
import argparse
import lm_process
import img_process
import age_transfer
from torch.autograd import Variable
import torch
import numpy as np
import cv2

def main(args):
    LMP = lm_process.LandmarkProcessing()
    IMP = img_process.ImageProcessing()
    model = age_transfer.Model(args)

    cap = cv2.VideoCapture(0)
    age, gender = 0, 0
    while (cap.isOpened()):
        ret, frame =cap.read()
        if ret == True:
            ages = [ord(str(x)) for x in range(args.Na)]
            cv2.imshow("Frame" , frame)
            flag = cv2.waitKey(1)
            if flag in ages:
                age = flag - 48
                print('Age group{}'.format(age))
            if flag == ord('m'):
                gender = 0
                print('Male')
            elif flag == ord('f'):
                gender = 4
                print('Female')

            try:
                capture_img = frame
                lm = LMP.detector(capture_img)
                cropped = IMP.crop(capture_img, lm, (128,128))
                cropped = cropped.reshape(1, *cropped.shape)

                # # Predict age and gender group
                # age, gender = model.classifier(image)

                generated = model.generate_image(cropped, age, gender)
                # Generated Image
                IMP.show_video(generated)
                IMP.render(generated, frame)
            except:
                pass

    else:
        print('Check your camera!')
        exit()  
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=128, help='cropping size')
    parser.add_argument('--Angle-Loss', action='store_true', default=True, help='Use Angle Loss')
    parser.add_argument('--Na', type=int, default=4, help='initial Number of ID [default: 188]')
    parser.add_argument('--Ng', type=int, default=2, help='initial Number of Pose [default: 13 (Pose from Multi-PIE Database)]')
    parser.add_argument('--snapshot', type=str, default='./snapshot/aaaumm_4gp10_p15_ag5_do0_4G1D_gender_GN_128_AL_loss_10l1-0.1/epoch50', help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')

    args = parser.parse_args()
    main(args)
