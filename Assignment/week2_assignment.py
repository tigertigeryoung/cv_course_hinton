

# Assignment Description:
# Code a function to do median blur operations by your self. Can it be completed in a shorter time complexity ?

import cv2
import numpy as np
import time


class MedianBlur:

    def __init__(self, img_path, filter_radius, filter_method='', padding_method=''):
        """
         :param img_path: storage path of the image to be processed
         :param filter_radius: the radius of the sliding window, here it should be a positive odd number
         :param filter_method: 'classic' or 'huang' can be selected
         :param padding_method: 'nothing', 'zeros' or 'edges' can be used
        """
        self.img = cv2.imread(img_path, 0)
        self.r = filter_radius
        self.p = padding_method
        if filter_method == 'huang':
            print('Huang method is running...')
            self.huang_median_blur()
        elif filter_method == 'classic':
            print('Classic method is running...')
            self.classic_median_blur()

    def classic_median_blur(self):
        image = self.img
        r = self.r
        row, column = image.shape
        image_new = np.zeros((row + 2 * r, column + 2 * r), dtype=np.uint8)
        image_new += 255
        if self.p == 'nothing':
            pass
        # padding with zeros
        elif self.p == 'zeros':
            image_new[r:row + r, r:column + r] = image
            image = image_new
        # padding with the edge's pixels values
        elif self.p == 'edges':
            image_new[r:row + r, r:column + r] = image
            image_new[0:r, r:column + r] = image[0, 0:column]
            image_new[row + r:row + 2 * r, r:column + r] = image[row - 1, 0:column]
            image_new[0:r, 0:r] = image[0][0]
            image_new[0:r, column:column + 2 * r] = image[0][column - 1]
            for i in range(row):
                image_new[i + r:row + 2 * r, 0:r] = image[i][0]
                image_new[i + r:row + 2 * r, column + r:column + 2 * r] = image[i][column - 1]
            image = image_new
        start = time.perf_counter()
        for i in range(r, row - r):
            for j in range(r, column - r):
                temp = image[i - r:i + r + 1, j - r:j + r + 1]
                buffer = []
                for k in range(0, 2*r+1):
                    for l in range(0, 2*r+1):
                        buffer.append(temp[k][l])
                for p1 in range(len(buffer)):
                    for p2 in range(len(buffer) - p1 - 1):
                        if buffer[p2] > buffer[p2 + 1]:
                            buffer[p2], buffer[p2 + 1] = buffer[p2 + 1], buffer[p2]
                median = buffer[int(((2*r+1) * (2*r+1) + 1) / 2 - 1)]
                temp[r][r] = median
                image[i - r:i + r + 1, j - r:j + r + 1] = temp
        end = time.perf_counter()
        if self.p != 'nothing':
            image = image[r:row-1-r, r:column-1-r]
        print('Running time is', end-start, 's', ', padding method is', self.p, '.')
        cv2.imshow("UI", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image

    def huang_median_blur(self):
        image = self.img
        r = self.r
        row, column = image.shape
        # padding method
        image_new = np.zeros((row + 2 * r, column + 2 * r), dtype=np.uint8)
        image_new += 255
        if self.p == 'nothing':
            pass
        elif self.p == 'zeros':
            image_new[r:row + r, r:column + r] = image
            image = image_new
        # 关于padding希望助教给点建议，感觉这方法太笨了。。
        elif self.p == 'edges':
            image_new[r:row + r, r:column + r] = image
            # 上边界
            image_new[0:r, r:column + r] = image[0, 0:column]
            # 下边界
            image_new[row + r:row + 2 * r, r:column + r] = image[row - 1, 0:column]
            # 左上角
            image_new[0:r, 0:r] = image[0][0]
            # 右上角
            image_new[0:r, column:column + 2 * r] = image[0][column - 1]
            # 两侧边
            for i in range(row):
                image_new[i + r:row + 2 * r, 0:r] = image[i][0]
                image_new[i + r:row + 2 * r, column + r:column + 2 * r] = image[i][column - 1]
            image = image_new
        # timer setting
        start = time.perf_counter()
        m_num = ((2 * r + 1) ** 2 + 1) / 2
        # start with a new row
        for i in range(r, row - r):
            # histogram initialization.
            # Input the pixels in the window into the histogram.
            # The index of histogram is the value of the pixel.
            median = 0
            h = [0] * 256
            for w_i in range(-r, r + 1):
                for w_j in range(-r, r + 1):
                    h[image[w_i + i][w_j + r]] += 1
            s = 0
            # calculate the median in the histogram
            for init in range(len(h)):
                s += h[init]
                if s >= m_num:
                    median = init
                    break
            h[image[i][r]] -= 1
            h[median] += 1
            image[i][r] = median
            # move to a new column
            for j in range(r, column - r - 1):
                # update the histogram:
                # move out the leftmost column, move in the new column in the right
                for k in range(-r, r + 1):
                    h[image[i + k][j - r]] -= 1
                    h[image[i + k][j + r + 1]] += 1
                s = 0
                for l in range(len(h)):
                    s += h[l]
                    if s >= m_num:
                        median = l
                        break
                h[image[i][j + 1]] = h[image[i][j + 1]] - 1
                h[median] = h[median] + 1
                image[i][j + 1] = median
        end = time.perf_counter()
        # remove the padding
        if self.p != 'nothing':
            image = image[r:row-1-r, r:column-1-r]
        print('Running time is', end-start, 's', ', padding method is', self.p, '.')
        cv2.imshow("UI", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image


MedianBlur('C:/Users/tiger/my_pyfile/images/1.jpg', 2, 'huang', 'edges')
