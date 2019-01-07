import pandas as pd
import numpy as np
import cv2

anotations_path = 'data/processed_anotations.csv'
images_dir = 'data/processed_images/'
num_of_images = 200

class data:
    def __init__(self, batchsize = 64):
        self.batchsize = batchsize
        self.count = 0
        self.num_train_images = int( num_of_images * 0.8 )
        self.num_test_images = num_of_images - self.num_train_images

        test = pd.read_csv(anotations_path, index_col=0)
        self.anotations = test.to_dict()

        for i in range(num_of_images):
            str = self.anotations['is_visible'][i].replace('[','').replace(']','')
            array = np.array(str.split(' '), dtype=np.float32)
            self.anotations['is_visible'][i] = array

            list = []
            lines = self.anotations['joints_pos'][i].replace('[','').replace(']','').split('\n')
            for line in lines:
                elements = line.split(" ")
                for ele in elements:
                    if ele != '':
                        list.append(ele)
            array = np.array(list, dtype=np.float32)
            self.anotations['joints_pos'][i] = array


        print('datasets inited')

    def NextBatch(self):
        x = np.zeros((self.batchsize, 448, 448, 3), dtype=np.float32)
        y = np.zeros((self.batchsize, 32), dtype=np.float32)
        c = np.zeros((self.batchsize, 1), dtype=np.float32)

        if self.count + self.batchsize < self.num_train_images:
            temp = self.count
            self.count += self.batchsize
            for i in range(temp, self.count):
                img = cv2.imread( images_dir + "{}.jpg".format(self.anotations['id'][i]) )
                img = cv2.resize(img, (448,448) )
                x[i-temp] = img
                for j in range(32):
                    y[i - temp][j] = float(self.anotations['joints_pos'][i][j] )

                c[i - temp] = float(self.anotations['normalizer'][i] )
        else:
            temp = self.count
            self.count = ( self.count + self.batchsize ) % self.num_train_images
            for i in range(temp, self.num_train_images):
                img = cv2.imread( images_dir + "{}.jpg".format(self.anotations['id'][i]) )
                img = cv2.resize(img, (448,448))

                x[i-temp] = img

                y[i - temp] = self.anotations['joints_pos'][i]

                c[i - temp] = self.anotations['normalizer'][i]

            bias = self.num_train_images - temp
            for i in range(0, self.count):
                img = cv2.imread( images_dir + "{}.jpg".format(self.anotations['id'][i]) )
                img = cv2.resize(img, (448,448))

                x[i+bias] = img

                y[i+bias] = self.anotations['joints_pos'][i]

                c[i+bias] = self.anotations['normalizer'][i]

        return x, y, c

    def TestSets(self):
        x = np.zeros((self.num_test_images, 299, 299, 3), dtype=np.float32)
        y = np.zeros((self.num_test_images, 33), dtype=np.float32)

        bias = self.num_train_images
        for i in range(self.num_train_images, num_of_images):
            img = cv2.imread(images_dir + "{}.jpg".format(self.anotations['id'][i]))
            x[i - bias] = img

            y[i - bias][0 : 32] = self.anotations['joints_pos'][i]

            y[i - bias][32] = self.anotations['normalizer'][i]

        return x, y



if __name__ == '__main__':
    a = data()
    counter = 0
    while True:
        n = a.NextBatch()
        print(counter)
        counter += 1