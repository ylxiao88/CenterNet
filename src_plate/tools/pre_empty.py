import os
#import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import shutil
from tqdm import tqdm


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class process_empty:
    def __init__(self, data_path):
        self.classes = ('__background__',
                        'plate'
                        )

        self.num_classes = len(self.classes)
        self.img = []
        for dir_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, dir_name)):
                continue
            img_path = os.path.join(data_path, dir_name, 'image')
            self.img.extend([os.path.join(img_path, i) for i in os.listdir(img_path)])
        self.anno = [i.replace('image','xml').replace('jpg','xml').replace('JPG','xml').replace('png','xml') for i in self.img]


    def _load_annotation(self, index):

        xml_file = self.anno[index]
        tree = ET.parse(xml_file)
        #size = tree.find('size')
        objs = tree.findall('object')
        #width = size.find('width').text
        #height = size.find('height').text
        if objs == []:
            return self._load_annotation(index+1), index+1
        for obj in objs:
            bndbox = obj.find('bndbox')
            [xmin, xmax, ymin, ymax] \
                = [int(bndbox.find('xmin').text) - 1, int(bndbox.find('xmax').text),
                   int(bndbox.find('ymin').text) - 1, int(bndbox.find('ymax').text)]
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            bbox = [xmin, xmax, ymin, ymax]
        return bbox, index

    def makedir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def copy_to_empty(self, src_dir, tar_dir):
        out_img = os.path.join(tar_dir, 'image')
        out_xml = os.path.join(tar_dir, 'xml')

        self.makedir(tar_dir)
        self.makedir(out_img)
        self.makedir(out_xml)

        for img_name in tqdm(os.listdir(src_dir)):
            src = cv2.imread(os.path.join(src_dir,img_name))
            ih,iw,_ = src.shape

            # random pick an image
            ind = np.random.randint(len(self.img), size=1)[0]
            box, ind = self._load_annotation(ind)
            tar = cv2.imread(self.img[ind])
            #plate = tar[box[1]:box[3],box[0]:box[2],:]
            #ph, pw = box[3]-box[1], box[2]-box[0]

            # paste
            x1,x2,y1,y2 = box
            bh,bw = y2-y1, x2-x1
            #x1 = int(max(0, x1-bw))
            #x2 = int(min(iw, x2+bw))
            #y1 = int(max(0, y1-bh))
            #y2 = int(min(ih, y2+bh))
            x1 = int(max(0, x1-50))
            x2 = int(min(iw, x2+50))
            y1 = int(max(0, y1-30))
            y2 = int(min(ih, y2+30))
            src[y1:y2,x1:x2,:] = tar[y1:y2,x1:x2,:]

            # out
            cv2.imwrite(os.path.join(out_img, img_name), src)
            xml_name = img_name.split('.')[0]+'.xml'
            #os.system('cp %s %s'%(self.anno[ind], os.path.join(out_xml, xml_name) ))
            shutil.copy(self.anno[ind], os.path.join(out_xml, xml_name))


if __name__ == "__main__":
    pro = process_empty('/home/ubuntu/data/plate/')
    pro.copy_to_empty('/home/ubuntu/data/empty_small/image/', '/home/ubuntu/data/empty_processed')
