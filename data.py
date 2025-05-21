import os
import numpy as np
from skimage import io, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

COLOR_DICT = {
    0: [255, 0, 0],   # Background → kırmızı
    1: [0, 255, 0],   # Adipose/fibrogland → yeşil
    2: [0, 0, 255]    # Pectoral → mavi
} 

def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        
        img = np.concatenate((img,) * 3, axis=-1)  
        img = img / 255.0
        
        
        mask = mask[:, :, :, 0] if len(mask.shape) == 4 else mask[:, :, 0]
        mask = np.array(mask)
        
        
        new_mask = np.zeros(mask.shape + (num_class,))
        # Background: mask değeri 255 ise → index 0
        new_mask[mask == 255., 0] = 1
        # Adipose: mask değeri 128 veya 192 ise → index 1 (fibrogland ile birlikte)
        new_mask[(mask == 128.) | (mask == 192.), 1] = 1
        # Pectoral: mask değeri 64 ise → index 2
        new_mask[mask == 64., 2] = 1
        
        mask = new_mask

    elif np.max(img) > 1:
        img = np.concatenate((img,) * 3, axis=-1)
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                   image_color_mode="grayscale", mask_color_mode="grayscale",
                   image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=True, num_class=3, save_to_dir=None,
                   target_size=(480, 240), seed=42):
    '''
    Aynı anda görüntü ve mask üretebilen generator.
    image_datagen ve mask_datagen için aynı seed kullanılarak dönüşümlerin uyumlu olması sağlanır.
    Eğer generator sonuçlarını görselleştirmek isterseniz, save_to_dir parametresini belirtebilirsiniz.
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    
    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield img, mask

def testGenerator2(test_path, num_image=6, target_size=(256, 256, 3),
                  flag_multi_class=True, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255.0
        img = transform.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img

def testGenerator(test_path, num_image=6, target_size=(480, 240, 3),
                  flag_multi_class=True, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255.0
        img = transform.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    img_out = np.zeros(img[:, :, 0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i, j])
            img_out[i, j] = color_dict[index_of_class]
    return img_out


def saveResult(save_path, npyfile, vid, flag_multi_class=True, num_class=3):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img = img.astype(np.uint8)
        io.imsave(os.path.join(save_path, "%d_predict_%s.png" % (i, vid)), img)