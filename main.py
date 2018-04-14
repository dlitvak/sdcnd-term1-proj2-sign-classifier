# Load pickled data
import os
import matplotlib.image as mpimg
from helper_functions import *
from traffic_sign_classifier import TrafficSignClassifier

# Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]

image_shape = X_train[0].shape
n_classes = np.max(y_train) + (1 if np.min(y_train) == 0 else 0)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#display_img_types(X_train, y_train, n_classes)

#display_random_img(X_train, y_train)
#visualize_data_distribution(y_train, y_valid, y_test, n_classes)

# display_images_by_class(X_train, y_train, 6)

with open("traffic-signs-data/syn_train_rotate15.pickle", mode='rb') as f:
    syn_train = pickle.load(f)
X_train_syn1, y_train_syn1 = syn_train['syn_features'], syn_train['syn_labels']

X_train = np.vstack((X_train, X_train_syn1))
y_train = np.hstack((y_train, y_train_syn1))

with open("traffic-signs-data/syn_train_zoom_rotate10.pickle", mode='rb') as f:
    syn_train = pickle.load(f)
X_train_syn2, y_train_syn2 = syn_train['syn_features'], syn_train['syn_labels']

X_train = np.vstack((X_train, X_train_syn2))
y_train = np.hstack((y_train, y_train_syn2))


#display_images_by_class(X_train_syn1, y_train_syn1, 12)

# synthesize_data(X_train, y_train, n_classes)

#visualize_data_distribution(y_train, y_valid, y_test, n_classes)


tsClassifier = TrafficSignClassifier(X_train, y_train, X_valid, y_valid)
tsClassifier.train()
test_accuracy = tsClassifier.test(X_test, y_test)
print("Test Accuracy = {:.3f}".format(test_accuracy))
print()

# Test  online images
dir = "traffic-signs-data/online_test_imgs/"
if os.path.exists(dir) and os.path.isdir(dir):
    img_files = os.listdir(dir)
    images = []
    for img_name in img_files:
        if img_name.endswith('.jpg'):
            images.append(img_name)

    cols = 4
    rows = int(np.ceil(len(images)/cols))
    plt.figure(0, (cols, rows))
    k = 0
    for i in range(rows):
        for j in range(cols):
            if k < len(images):
                img_name = images[k]
                k += 1
                img = mpimg.imread(dir + img_name)
                plt.subplot(rows, cols, (i * cols) + (j + 1))
                plt.imshow(img)
                plt.axis('off')

    total_correct, total_imgs = 0, 0
    for img_name in images:
        img = mpimg.imread(dir + img_name)
        f, e = os.path.splitext(img_name)
        n, cls = f.split('-')

        pred_cls = tsClassifier.classify(img)
        total_correct += 1 if pred_cls[0] == int(cls) else 0
        total_imgs += 1
        print("Predicted: ", pred_cls, ", Real: ", cls)

        top5 = tsClassifier.top_softmax_probs(img, top_num=5)
        print("Softmax top5: ", str(top5))
        print()
    print("Online Image Accuracy: {:.3f}".format(total_correct/total_imgs))

img_read = mpimg.imread(dir + "00000-16.jpg")
tsClassifier.outputFeatureMap(img_read)
