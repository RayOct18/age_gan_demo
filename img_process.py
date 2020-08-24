import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2
from skimage import transform as trans
import math
import sys

class ImageProcessing():
    def __init__(self):
        plt.ion()   # interactive mode
        self.DIAMETER = 180 / math.pi
        self.crop_setting = {'scale_size': 60,
                'image_size': 132,
                'upper': 65,
                'lower': 67}
        self.cut = 0
        self.raw_lm = None
        self.new_lm = None


    def crop(self, img, lm, output):
        lm = lm.astype('float').reshape(-1, 2)
        self.raw_lm = lm
        angle, eye_chin_distance = self.calculate_eye(lm)
        processed_image, processed_lm = self.process_image(img, angle, lm, eye_chin_distance)
        cropped, self.new_lm = self.crop_image(processed_image, processed_lm)
        cropped = self.center_crop(cropped, output)
        cropped = cropped.astype(np.float32)
        cropped = (cropped * 2)/255 - 1

        return cropped

    def center_crop(self, image, output_size):
        image = image.transpose((2, 0, 1))
        h, w = image.shape[1:]
        new_h, new_w = output_size
        cut = (h-new_h)//2
        self.cut = cut
        cropped_image = image[:, cut:cut+new_h, cut:cut+new_w]

        return cropped_image

    def calculate_eye(self, lm):
        eye_x1 = lm[39][0] + lm[36][0]
        eye_y1 = lm[39][1] + lm[36][1]
        eye_x2 = lm[45][0] + lm[42][0]
        eye_y2 = lm[45][1] + lm[42][1]
        center_eyes = [[eye_x1 // 2, eye_y1 // 2],
                    [eye_x2 // 2, eye_y2 // 2]]
        chin_1 = lm[8]
        angle = math.atan((center_eyes[1][1] - center_eyes[0][1]) / ((center_eyes[1][0] - center_eyes[0][0]) + sys.float_info.epsilon))
        x_d = (chin_1[0] - (center_eyes[0][0] + center_eyes[1][0]) // 2) ** 2
        y_d = (chin_1[1] - (center_eyes[0][1] + center_eyes[1][1]) // 2) ** 2
        eye_chin_distance = math.sqrt(x_d + y_d)

        return angle, eye_chin_distance

    def get_center_point(self, lm):
        # eye_center_y
        eye_y1 = lm[39][1] + lm[36][1]
        eye_y2 = lm[45][1] + lm[42][1]
        center_eyes_y = (eye_y1 / 2 + eye_y2 / 2) // 2
        # cheek_center_x
        cheek_center_x = (lm[0][0] + lm[16][0]) // 2
        return center_eyes_y, cheek_center_x

    def process_image(self, image, angle, lm, eye_chin_distance):
        processed_lm = []
        x = lm[:, 0] - 0.5 * image.shape[1]
        y = lm[:, 1] - 0.5 * image.shape[0]
        x_r = x * math.cos(-angle) - y * math.sin(-angle)
        y_r = x * math.sin(-angle) + y * math.cos(-angle)
        rotated_image = self.rotateImage(image, angle)
        scale = self.crop_setting['scale_size'] / eye_chin_distance
        width = int(round(rotated_image.shape[1] * scale))
        height = int(round(rotated_image.shape[0] * scale))
        resize_imag = cv2.resize(rotated_image, (width,height), interpolation = cv2.INTER_AREA)
        x_r = (x_r + 0.5 * rotated_image.shape[1]) * scale
        y_r = (y_r + 0.5 * rotated_image.shape[0]) * scale
        processed_lm = np.round(np.append([x_r],[y_r], axis=0).T)

        return resize_imag, np.array(processed_lm)

    def crop_image(self, image, lm):
        center_eyes_y, cheek_center_x = self.get_center_point(lm)
        horizontal = self.crop_setting['image_size'] // 2
        upper_bound = center_eyes_y - self.crop_setting['upper']
        lower_bound = center_eyes_y + self.crop_setting['lower']
        left_bound = cheek_center_x - horizontal
        right_bound = cheek_center_x + horizontal
        crop_image = image[int(upper_bound):int(lower_bound), int(left_bound):int(right_bound), :]
        lm[:, 0] = lm[:, 0] - left_bound
        lm[:, 1] = lm[:, 1] - upper_bound
        return crop_image, lm


    def rotateImage(self, image, angle):
        angle = angle * 180 / math.pi
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


    def render(self, image, frame):
        whole_size = frame.shape
        frame = frame.astype(np.float32)
        frame = (frame * 2)/255 - 1

        M = self.estimate_norm(self.new_lm, self.raw_lm)
        image = image.cpu().numpy().transpose(1, 2, 0)

        # Paste directly
        image = cv2.copyMakeBorder(image, self.cut, self.cut, self.cut, self.cut, cv2.BORDER_CONSTANT, value=0)
        rotated_image_d = cv2.warpAffine(image,M, (whole_size[1], whole_size[0]))

        # # Show Paste directly
        # mask = self.make_mask(rotated_image_d)
        # border = 40
        # int_mask = np.zeros(mask.shape)
        # height, weight = mask.shape
        # mask = cv2.resize(mask.astype(np.uint8), dsize=(weight-border, height-border), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # int_mask[border//2:-(border//2), border//2:-(border//2)] = mask
        # mg_image_d = self.merge_image(rotated_image_d, frame, int_mask)
        # cv2.imshow('Paste directly', mg_image_d)
        # cv2.waitKey(1)

        # Paste face only
        only_face = self.remove_background_LM(image)
        image = cv2.copyMakeBorder(only_face, self.cut, self.cut, self.cut, self.cut, cv2.BORDER_CONSTANT, value=0)
        rotated_image_f = cv2.warpAffine(image,M, (whole_size[1], whole_size[0]))
        mask = self.make_mask(rotated_image_f)
        mg_image_f = self.merge_image(rotated_image_d, frame, mask)
        cv2.imshow('Paste face only', mg_image_f)
        cv2.waitKey(1)

    def remove_background_LM(self, img):
        def getEquidistantPoints(p1, p2, parts):
            return np.linspace(p1[0], p2[0], parts+1), np.linspace(p1[1], p2[1], parts+1)
        max_x = int(max(self.new_lm[:,0]))
        min_x = int(min(self.new_lm[:,0]))
        min_y = int(min(self.new_lm[:,1])) - 25
        min_y = min_y if min_y > 0 else 0

        img[:, :min_x, :] = 0
        img[:, max_x:, :] = 0
        img[:min_y, :, :] = 0

        for i in range(16):
            parts = self.new_lm[i+1][0] - self.new_lm[i][0]
            x, y = getEquidistantPoints(self.new_lm[i], self.new_lm[i+1], parts)
            for y_i, x_i in enumerate(x):
                img[int(round(y[y_i])):, int(round(x_i)), :] = 0
        return img

    def make_mask(self, img):
        mean_img = np.mean(img, axis=2)
        std_img = np.std(img, axis=2)
        mask = np.where(mean_img==0,0,1) + np.where(std_img==0,0,1)
        mask = np.where(mask==2,1,0)
        return mask


    def merge_image(self, img, bg, mask):
        img = img * mask[:,:,np.newaxis]
        inv_mask =  np.where(mask==1,0,1)
        cut_img = bg * inv_mask[:,:,np.newaxis]
        m_img = img + cut_img
        m_img = (m_img+1) * 255 / 2.0
        m_img = m_img.astype(np.uint8)
        return m_img

    def estimate_norm(self, new_lm, raw_lm):
        src = raw_lm
        tform = trans.SimilarityTransform()
        src_map = src
        tform.estimate(new_lm, src_map)
        M = tform.params[0:2,:]
        return M


    def show_video(self, generated):
        generated_image = generated.cpu().data.numpy().transpose(1, 2, 0)
        # generated_image = np.squeeze(generated_image)
        generated_image = (generated_image+1) * 255 / 2.0
        # generated_image = generated_image[:, :, [2, 1, 0]]
        generated_image = generated_image.astype(np.uint8)
        cv2.imshow('image', generated_image)
        cv2.waitKey(1)