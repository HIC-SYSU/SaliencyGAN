# -*- coding=utf-8 -*-
from util.image_preprocessing_util import *



"""
load labled data from one dir
"""

def load_labeled_data_one(path, images_path, masks_path):
    files_img = os.listdir(path + "/" + images_path)
    files_mask = os.listdir(path + "/" + masks_path)
    print("img number = %i" % len(files_img))
    print("mask number = %i " % len(files_mask))
    images = []
    masks = []
    shapes = []
    i = 0
    for file in files_mask:

        print('==============================================')

        file_path_mask = path + "/" + masks_path + "/" + file
        file = file.replace(".png", ".jpg")
        file_path_img = path + "/" + images_path + "/" + file
        print(file_path_img, " ", file_path_mask)
        image = cv2.imread(file_path_img)
        mask = cv2.imread(file_path_mask)
        mask = cv2.cvtColor(mask, code=cv2.COLOR_BGR2GRAY)

        mask_r, src_shape, resized_shape = image_resize(mask, fix_size=cg.fix_size)
        image_r,src_shape,resized_shape = image_resize(image,fix_size=cg.fix_size)
        th, mask_r = cv2.threshold(mask_r, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_r = np.reshape(mask_r,newshape=[cg.fix_size,cg.fix_size,1])

        shapes.append( [resized_shape,src_shape])
        images.append(image_r)
        masks.append(mask_r)

        # print("image dtype=", image.dtype, "\n", "mask dtype=", mask.dtype)
        # print("image shape=", image.shape, "\n", "mask shape=", mask.shape)
        # cv2.imshow("img", image)
        # cv2.imshow("mask", mask)
        #
        # print("image_r dtype=", image_r.dtype, "\n", "mask_r dtype=", mask_r.dtype)
        # print("image_r shape=", image_r.shape, "\n", "mask_r shape=", mask_r.shape)
        # cv2.imshow("image_r", image_r)
        # cv2.imshow("mask_r", mask_r)
        #
        # cv2.waitKey(0)
        i = i + 1
        print(i)
        #if i > 1:
        #     break
    images = np.array(images,dtype=np.uint8)
    masks = np.array(masks,dtype=np.uint8)
    shapes = np.array(shapes,dtype=np.int32)


    print("images shape=", images.shape, "masks shape=", masks.shape,'shapes shape=',shapes.shape)
    print("images dtype=", images.dtype, "masks dtype=", masks.dtype,'shapes dtype=',shapes.dtype)

    return images,masks,shapes


"""
load unlabled data from one dir
"""

def load_unlabeled_data_one(path, images_path):
    files_img = os.listdir(path + "/" + images_path)

    print("img number = %i" % len(files_img))
    images = []
    shapes = []
    i = 0
    for file in files_img:

        print('==============================================')


        file_path_img = path + "/" + images_path + "/" + file
        print(file_path_img)
        image = cv2.imread(file_path_img)

        image_r,src_shape,resized_shape = image_resize(image,fix_size=cg.fix_size)


        shapes.append( [resized_shape,src_shape])
        images.append(image_r)

        # print("image dtype=", image.dtype)
        # print("image shape=", image.shape)
        # cv2.imshow("img", image)
        #
        # print("image_r dtype=", image_r.dtype)
        # print("image_r shape=", image_r.shape)
        # cv2.imshow("image_r", image_r)
        #
        # cv2.waitKey(0)
        i = i + 1
        print(i)
        # if i > 1:
        #     break
    images = np.array(images,dtype=np.uint8)
    shapes = np.array(shapes, dtype=np.int32)

    print("i=",i)

    print("images shape=", images.shape,'shapes shape=',shapes.shape)
    print("images dtype=", images.dtype,'shapes dtype=',shapes.dtype)

    return images,shapes




if __name__ == '__main__':

    data_root = 'msra10k_0.3/'
    imgs_path = 'imgs/'
    gt_path = 'gt/'
    other_imgs_path = 'other_imgs/'

    print("----------------------")
    print("load train labeled data from %s" % data_root)
    train_images, train_masks,train_shapes = load_labeled_data_one(path=data_root,
                                                      images_path=imgs_path,
                                                      masks_path=gt_path)

    np.save(data_root + '/train_images',train_images)
    np.save(data_root + '/train_masks',train_masks)
    np.save(data_root + '/train_shapes',train_shapes)


    print("----------------------")
    print("load train unlabeled data from %s" % data_root)
    other_images, other_shapes = load_unlabeled_data_one(path=data_root,
                                                                    images_path=other_imgs_path)

    np.save(data_root + '/other_images', other_images)
    np.save(data_root + '/other_shapes', other_shapes)



    print("----------------------")
    print("load train labeled data from %s" % cg.test_path)
    test_images, test_masks, test_shapes = load_labeled_data_one(path=cg.test_path,
                                                                    images_path=cg.test_images_path,
                                                                    masks_path=cg.test_masks_path)

    np.save(data_root + '/test_images', test_images)
    np.save(data_root + '/test_masks', test_masks)
    np.save(data_root + '/test_shapes', test_shapes)





















