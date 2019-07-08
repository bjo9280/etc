from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import vis


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def compute_iou(y_pred, y_true):
#
    #ytrue, ypred is a flatten vector
    #y_pred = y_pred.flatten()
    #y_true = y_true.flatten()#

    true_list = []
    for i in y_true:
        #if i != 0:
        true_list.append(i)
    #remove overlap
    label = list(set(true_list))
    print(label)
    current = confusion_matrix(y_true, y_pred)
    print(current)
    # compute mean iounumpy to tuple
    intersection = np.diag(current)
    print(intersection)
    ground_truth_set = current.sum(axis=1)
    print(ground_truth_set)
    predicted_set = current.sum(axis=0)
    print(predicted_set)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    print(np.mean(IoU))

def make_image_list(image_name):
    img_color = cv2.imread('{}'.format(image_name), cv2.IMREAD_COLOR)
    height, width, chanel = img_color.shape
    image_list = []

    for y in range(0, height):
        for x in range(0, width):
            b = img_color.item(y, x, 0)
            g = img_color.item(y, x, 1)
            r = img_color.item(y, x, 2)
            image_list.append(palettes_dic.get((r,g,b)))
    return image_list

if __name__ == "__main__":

    num_classes = 21
    palettes = vis.make_palette(num_classes)
    palettes_dic = {tuple(palettes[i]): i for i in range(num_classes)}

    image_name = 'image'
    pred = '{}_output.png'.format(image_name)
    true = '{}.png'.format(image_name)
    compute_iou(make_image_list(pred), make_image_list(true))