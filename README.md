# Локализация и классификация

В этом проекте будут рассмотрены способы решения задачи классификации и локализации на примере набора данных с [котиками и собачками](https://drive.google.com/file/d/1d0Ycm9YnDJf_FBi6dr92fit-aw3VDkm0/view?usp=sharing). Каждому изображению в архиве соответствует файл разметки RoI 
(region of interest) изображения. Файл разметки включает в себя следующие данные: класс, xmin, ymin, xmax, ymax. 
Поставленную задачу можно решить тремя способами в порядке увеличения сложности:
1) Взять готовую нейронную сеть и обучить ее на своих данных
2) Применить Transfer Learning
3) Сформировать собственную свёрточную нейронную сеть.

# Подготовка данных

Для начаал предлагается привести данные к одному формату и аугментировать. Изменный формат предполагает данные в виде: класс, x_c, y_c, w, h. Те от координат углов перейти к координатам центра, ширине и высоте интересующей области.
Скрипт для аугментации данных ()
```Python
import glob
import os
import cv2
data_dir = 
new_dir = 
for image in glob.glob(os.path.join(data_dir,'*.jpg')):
    title, ext = os.path.splitext(os.path.basename(image))
    txt_file_name = title + '.txt'
    cv_image = cv2.imread(image)
    H, W, _ = cv_image.shape
    with open(os.path.join(new_dir,txt_file_name),'r') as old_txt:
        labelList = old_txt.readlines()
        for label in labelList:
            label = label.strip().split()
            class_ = int(label[0])
            x_min = float(label[1])
            x_max = float(label[3])
            y_min = float(label[2])
            y_max = float(label[4])
        angle = 180
        rotated_txt = title + '_' + str(angle) + '.txt'
        with open(os.path.join(new_dir,rotated_txt),'w+') as new_txt:
            new_txt.write(str(int(label[0])) + '\t' + str(1-x_min) 
            + '\t' + str(1-y_min) + '\t' + str() + '\t' + str(W))
        image2 = cv_image.copy()
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(cv_image, M, (W, H))
        cv2.imwrite(title + '_' + str(angle) + '.jpg', rotated)
```
Скрипт для перехода к новому формату данных:
```Python
data_dir = 
new_dir = 
for image in glob.glob(os.path.join(data_dir,'*.jpg')):
    title, ext = os.path.splitext(os.path.basename(image))
    txt_file_name = title + '.txt'
    cv_image = cv2.imread(image)
    H, W, _ = cv_image.shape
    with open(os.path.join(data_dir,txt_file_name),'r') as old_txt:
        with open(os.path.join(new_dir,txt_file_name),'w+') as new_txt:
            labelList = old_txt.readlines()
            for label in labelList:
                label = label.strip().split()
                x_min = float(label[1])/W
                x_max = float(label[3])/W
                y_min = float(label[2])/H
                y_max = float(label[4])/H
                w     = x_max - x_min
                h     = y_max - y_min
            new_txt.write(str(int(label[0])-1) + '\t' + str(x_min+w/2) 
            + '\t' + str(y_min+h/2) + '\t' + str(w) + '\t' + str(h))
    cv2.imwrite(new_dir +'/' + title + '.jpg', cv_image)
    old_txt.close()
    new_txt.close()
```
Далее к решению задачи

# Использование готовой нейронной сети на примере [YOLO](https://github.com/ultralytics/yolov5)
Для применения сети YOLO к решению данной задачи необходимо перевести датасет к определенному формату:
```Python
with open(os.path.join(data_dir,'train.txt'),'w+') as train:
    for file in glob.glob(os.path.join(aug_dir,'*.txt')):
        title, ext = os.path.splitext(os.path.basename(file))
        train.write(data_dir + '/' + title + '.jpg' + '\n')
train.close()
```
А так же создать файл с раширением .yaml следующего содержания:
```
train: #здесь указать путь к тренировой части данных
val: #здесь указать путь к валидационной части данных
# number of classes
nc: 2
# class names
names: ['cat', 'dog']
```
# Результаты YOLO 
- [mIoU](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) = 90.85
- Accuracy = 98.7
- Среднее время обработки одного файла 0.041s
- Размер валидационного датасета 398

![Abyssinian_116](https://user-images.githubusercontent.com/73283847/131541946-98706636-199c-4220-b2c0-e49ef03b3ecb.jpg)  
![american_bulldog_13](https://user-images.githubusercontent.com/73283847/131541954-54829ea5-d28b-4d48-bd49-8c00eb78f2d7.jpg)  
![Persian_194](https://user-images.githubusercontent.com/73283847/131542037-9f6f415b-27c6-4667-ac29-fe15f64bb71b.jpg)  

# Применение transfer learning
На примере MobileNetV2 с весами imagenet. Для решения этой задачи достаточно будет сформаировать выходы соответствующие интересующим нас данным. В данном случае хорошо подойдет следующая конструкция:
![image](https://user-images.githubusercontent.com/73283847/131543217-2f172558-883e-4f55-af95-dea48ae0c21e.png)

Для удобства обучения нейронных сетей стоит реализовать парсер данных для датасета:
```Python
def parse_dataset(dataset_path, ext='jpg'):
    """
    Сбор информации по всем файлам в директории.
    Возвращает df с данными
    """
    def parse_info_from_file(path):
        """
        Сбор данных с файла
        """
        try:
            title, ext = os.path.splitext(os.path.basename(path))
            txt_file_name = title + '.txt'
            with open(os.path.join(os.path.dirname(os.path.abspath(path)),txt_file_name),'r') as old_txt:
                labelList = old_txt.readlines()
                for label in labelList:
                    label = label.strip().split()
                    class_= int(label[1])
                    x1    = float(label[2])
                    y1    = float(label[3])
                    x2    = float(label[4])
                    y2    = float(label[5])
                    
                    w = x2 - x1
                    h = y2 - y1
                    x1 = x1+w/2
                    y1 = y1+h/2
            return class_, x1, y1, w, h
        except Exception as ex:
            return class_, None, None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['class', 'x1', 'y1', 'w', 'h', 'file']
    df = df.dropna()
    
    return df
```
И генератор данных
```Python
class DataGenerator():
    """
    Генератор данных для обучения модели. Используется для оптимазации работы памяти при обучении.  Позволяет не загружать в оперативную память сразу весь датасет. 
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_idx = p[:]
        
        return train_idx
    
    def generate_test_split_indexes(self,length):
        p = np.random.randint(0,len(self.df),length)
        train_idx = p[:]
        
        return train_idx
    
    def preprocess_image(self, img_path):
        """
        Предобработка: приведение к размеру входа нейронной сети, 
        """
        
        im = Image.open(img_path)
        im = im.resize((IMG_WIDTH, IMG_HEIGHT))
        im = np.array(im) / 255.0
        return im
        
    def generate_images(self, image_idx, is_training, batch_size):
        """
        Используется для генерации пачки(Batch), которая пойдет на обучение модели при следующей итерации.
        """
        
        # arrays to store our batched data
        images, classes, x1, y1, x2, y2 = [], [], [], [], [], []
        while True:
            for idx in image_idx:
                imag = self.df.iloc[idx]
                
                classes_ = imag['class']
                x1_ = imag['x1']
                y1_ = imag['y1']
                x2_ = imag['x2']
                y2_ = imag['y2']
                file = imag['file']
                im = self.preprocess_image(file)
                
                classes.append(classes_)
                x1.append(x1_)
                y1.append(y1_)
                x2.append(x2_)
                y2.append(y2_)
                images.append(im)
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(to_categorical(classes,2)),np.array(x1),np.array(y1), np.array(x2), np.array(y2)]
                    images, classes, x1, y1, x2, y2 = [], [], [], [], [], []
                    
            if not is_training:
                break
```
Метрики для обучкения:
```Python
def log_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.log1p(tf.math.squared_difference(y_pred, y_true)), axis=-1)

def focal_loss(alpha=0.9, gamma=2):
  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

  def loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

  return loss
```
Компиляция модели:
```Python
init_lr = 1e-2
epochs = 50
model2.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),#SGD(lr = 1e-2),#Adam(lr=init_lr,beta_1=0.9),
     loss={
                  'Box_output1': log_mse,
                  'Box_output2': log_mse,
                  'Box_output3': log_mse,
                  'Box_output4': log_mse,
                  'class_output': focal_loss()},
     loss_weights={
                  'Box_output1': 1,
                  'Box_output2': 1,
                  'Box_output3': 1,
                  'Box_output4': 1,
                  'class_output': 1},
     metrics={
                  "Box_output1":  'mse',
                  "Box_output2":  'mse',
                  "Box_output3":  'mse',
                  "Box_output4":  'mse',
                  "class_output": 'accuracy',
    },
)
```
Список Callbck:

```Python
def scheduler(epoch, lr):
    if epoch>20:
        return lr*0.9
    else:
        return lr
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001)
callbacks = [
    #ModelCheckpoint(r"C:\Users\densh\Desktop\catdogs\keras_try\Checkpoint", monitor='val_loss', save_best_only=True), 
    reduce_lr, callback,
    keras.callbacks.TensorBoard(
    log_dir=r"C:\Users\densh\Desktop\catdogs\keras_try\log",write_images=True,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch")  # How often to write logs (default: once per epoch)
]
```
Начало обучения
```Python
model2.fit(train_gen,steps_per_epoch=len(train_idx)//(1*batch_size),  epochs=100, callbacks=callbacks, shuffle=True, validation_data=valid_gen, validation_steps=len(valid_idx)//valid_batch_size)
```
# Результаты transfer learning

- mIoU = 58.90
- Accuracy = 97.9
- Среднее время обработки одного файла 0.061s
- Размер валидационного датасета 398

![Imagesbasset_hound_124](https://user-images.githubusercontent.com/73283847/131545670-97b31ba5-07bf-4402-8315-25d181787026.jpg)  

![Imagesbeagle_120](https://user-images.githubusercontent.com/73283847/131545721-4e429bff-cb46-4f98-869c-9bdc10d14833.jpg)

![Imagesamerican_pit_bull_terrier_169](https://user-images.githubusercontent.com/73283847/131545738-0bc83481-d8a9-4bd5-a434-c5a796f2dd4a.jpg)

# Собственная свёрточная нейронная сеть.

В качестве основы для нейронной сети был выбран блок из последовательно соеденных сверточного слоя, слоя максимальное объединения инормализующего слоя. (Convolutional, MaxPooling, batchNormalization). Последовательно соедененные 7 таких блоков были присоеденены к трем сверточным слоям. Для этой нейронной сети было составлено два выхода: классификатор, состоящий из слоя глобального среднего объеденения, двух полностью связанных слоев и выхода с функкцией активации "softmax"; регроссор состоящий из двух описанных выше блоков, двух наборов из сверточного слоя и слоя максимального объеденения, выход - слой с функцией активации "relu" размерности 4. 

![image](https://user-images.githubusercontent.com/73283847/131546092-00ca11c5-68f7-4d74-b51b-db3ce015be4f.png)

- mIoU = 32.50
- Accuracy = 88.0
- Среднее время обработки одного файла 0.051s
- Размер валидационного датасета 398

![american_bulldog_11](https://user-images.githubusercontent.com/73283847/131546220-82baacb4-5bb4-46fa-8022-435a9e141c5f.jpg)

![Abyssinian_154](https://user-images.githubusercontent.com/73283847/131546229-e40d54aa-be0d-4c05-8fc5-b98f4c8f4114.jpg)

![Bengal_174](https://user-images.githubusercontent.com/73283847/131546276-e62e5364-3c96-4fcd-b270-251f43529385.jpg)

# Сводная таблица по результатм работы

![image](https://user-images.githubusercontent.com/73283847/131546407-b60e7cba-64b3-4ad8-a8f1-6faa9cd87f36.png)

Скрипт для оценки качества моделей
```Python
counter = 0
overall_time = 0
mIoU = 0
prediction_count = 0
for image in glob.glob(os.path.join(validation_data_dir,'*.jpg')):
    '''
    Оценка качества работы модели: mIoU, Acc, среднее время работы на 1 фотографию
    '''
    title, ext = os.path.splitext(os.path.basename(image))
    txt_file_name = title + '.txt'
    imag = cv2.imread(image)
    H, W, _  = imag.shape
    img = cv2.resize(imag,(IMGsize1,IMGsize2))    
    img = img.reshape(1,IMGsize1,IMGsize2,num_channels)
    img = np.array(img)/255.0
    tic = time.perf_counter()
    results = model.predict(img)
    toc = time.perf_counter()
    timer = toc - tic
    overall_time = overall_time + timer

    bbox1 = []
    bbox2 = []

    label_names = ['cat','dog']
    class_label = label_names[np.argmax(results[0])]

    x1_, y1_, x2_, y2_ = results[1][:][0]

    x1 = int(W * (x1_ - x2_/2))
    x2 = int(W * (x1_ + x2_/2))
    y1 = int(H * (y1_ - y2_/2))
    y2 = int(H * (y1_ + y2_/2))
    bbox1 = [x1,y1,x2,y2]
    top_left_pred = (x1,y1)
    right_bottom_pred = (x2, y2)

    with open(os.path.join(Label_Save_dir,title + '_pred' + '.txt'),'w+') as new_txt:
        new_txt.write(str(class_label) + '\t' + str(x1) 
        + '\t' + str(y1) + '\t' + str(x2) + '\t' + str(y2) + '\t' + str(timer))
    
    truth_txt = open(os.path.join(Ground_Truth_label_dir,txt_file_name),'r')
    labelList = truth_txt.readlines()
    for label in labelList:
        label = label.strip().split()
        class_=int(label[0])
        x1 = int(label[1])
        x2 = int(label[3])
        y1 = int(label[2])
        y2 = int(label[4])
    
    top_left_true = (x1, y1)
    right_bottom_true = (x2,y2)
    bbox2 = [x1,y1,(x2),y2]

    IoU = float(IOUab(bbox1, bbox2)*100)

    if class_label == label_names[class_-1]:
        prediction_count =  prediction_count + 1
    counter = counter + 1

    cv2.rectangle(imag, top_left_true, right_bottom_true,[0,0,0],2)
    cv2.rectangle(imag, top_left_pred, right_bottom_pred,[255,0,0],2)
    cv2.putText( imag,'IoU = '+f"{IoU:.{2}f}" +' '+  label_names[class_-1], (int(x1), int(y1) + int(28)),  cv2.FONT_HERSHEY_TRIPLEX, 0.8, [0, 0, 0], 2)
    cv2.imwrite(Image_save_dir + title+'.jpg',imag)
    mIoU = mIoU + IoU
    #break
mIoU = mIoU/counter
mean_time = overall_time/counter
Acc = prediction_count/counter
print(mIoU, mean_time, Acc)
```
