import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from collections import defaultdict
import itertools
from app.utils.text_transformation_utils import TextTransformations


class EDA:
    def __init__(self):
        self.directory = r'E:\Courses\datasets\baymax_xray_dataset\ecgen-radiology'

    def get_img_findings_from_xml(self) -> dict:
        img = []
        img_finding = []
        for filename in tqdm(os.listdir(self.directory)):
            if filename.endswith(".xml"):
                f = self.directory + '/' + filename
                tree = ET.parse(f)
                root = tree.getroot()
                for child in root:
                    if child.tag == 'MedlineCitation':
                        for attr in child:
                            if attr.tag == 'Article':
                                for i in attr:
                                    if i.tag == 'Abstract':
                                        for name in i:
                                            if name.get('Label') == 'FINDINGS':
                                                finding = name.text

                for p_image in root.findall('parentImage'):
                    img.append(p_image.get('id'))
                    img_finding.append(finding)
        return {"image": img, "finding": img_finding}

    def create_dataset(self):
        dataset = pd.DataFrame()
        dataset['Image_path'] = self.get_img_findings_from_xml().get("img")
        dataset['Finding'] = self.get_img_findings_from_xml().get("finding")
        dataset['Image_path'] = dataset['Image_path'].apply(lambda x: self.absolute_path(x))
        return dataset.dropna(axis=0)

    def absolute_path(self, x):
        """Makes the path absolute"""
        x = r'E:\Courses\datasets\baymax_xray_dataset\NLMCXR_png' + r'/' + x + '.png'
        return x

    def find_images_per_patient(self):
        images = {}
        findings = {}
        dataset = self.create_dataset()
        for img, fin in dataset.values:
            a = img.split('-')
            a.pop(len(a) - 1)
            a = '-'.join(a)
            if a in images:
                images[a] += 1
            else:
                images[a] = 1
            findings[a] = fin
        plt.figure(figsize=(17, 5))
        plt.bar(range(len(images.keys())), images.values())
        plt.ylabel('Total Images per individual')
        plt.xlabel('Number of Individuals in the Data')
        plt.show()
        return {"images": images, "findings": findings}

    def train_test_split(self, data):
        persons = list(data.keys())
        persons_train = persons[:2500]
        persons_cv = persons[2500:3000]
        persons_test = persons[3000:3350]
        return persons_train, persons_cv, persons_test

    def combining_images(self, image_set):
        image_per_person = defaultdict(list)
        dataset = self.create_dataset()
        for pid in image_set:
            for img in dataset['Image_path'].values:
                if pid in img:
                    image_per_person[pid].append(img)
                else:
                    continue
        return image_per_person

    def split_dataset(self):
        images = self.find_images_per_patient().get("images")
        images_train, images_cv, images_test = self.train_test_split(images)
        return {"train": images_train, "validation": images_cv, "test": images_test}

    def load_image(self, file):
        img = tf.io.read_file(file)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def create_data(self, image_per_person):
        person_id, image1, image2, report = [], [], [], []
        findings = self.find_images_per_patient().get("findings")
        for pid, imgs in image_per_person.items():
            if len(imgs) == 1:
                image1.append(imgs[0])
                image2.append(imgs[0])
                person_id.append(pid)
                report.append(findings[pid])
            else:
                a = itertools.combinations(imgs, 2)
                for num, i in enumerate(a):
                    image1.append(i[0])
                    image2.append(i[1])
                    person_id.append(pid + '_' + str(num))
                    report.append(findings[pid])
        data = pd.DataFrame()
        data['Person_id'] = person_id
        data['Image1'] = image1
        data['Image2'] = image2
        data['Report'] = report
        return data

    def text_preprocessing(self, text):
        """Combines all the preprocess functions"""
        new_text = TextTransformations(text=text).lowercase()
        new_text = TextTransformations(text=new_text).decontractions()
        new_text = TextTransformations(text=new_text).rem_punctuations()
        new_text = TextTransformations(text=new_text).rem_numbers()
        new_text = TextTransformations(text=new_text).words_filter()
        new_text = TextTransformations(text=new_text).multiple_fullstops()
        new_text = TextTransformations(text=new_text).fullstops()
        new_text = TextTransformations(text=new_text).multiple_spaces()
        new_text = TextTransformations(text=new_text).separting_startg_words()
        new_text = TextTransformations(text=new_text).rem_apostrophes()
        return new_text

    def remodelling(self, x):
        """adds start and end tokens to a sentence """
        return 'startseq' + ' ' + x + ' ' + 'endseq'

    def export_csvs(self):
        img_per_person_train = self.combining_images(self.split_dataset().get("train"))
        img_per_person_cv = self.combining_images(self.split_dataset().get("validation"))
        img_per_person_test = self.combining_images(self.split_dataset().get("test"))
        train = self.create_data(img_per_person_train)
        test = self.create_data(img_per_person_test)
        cv = self.create_data(img_per_person_cv)
        train['Report'] = self.text_preprocessing(train['Report'])
        test['Report'] = self.text_preprocessing(test['Report'])
        cv['Report'] = self.text_preprocessing(cv['Report'])
        train['Report'] = train['Report'].apply(lambda x: self.remodelling(x))
        test['Report'] = test['Report'].apply(lambda x: self.remodelling(x))
        cv['Report'] = cv['Report'].apply(lambda x: self.remodelling(x))
        train.to_csv('Train_Data.csv', index=False)
        test.to_csv('Test_Data.csv', index=False)
        cv.to_csv('CV_Data.csv', index=False)
