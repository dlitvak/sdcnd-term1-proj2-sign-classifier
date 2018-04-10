# Load pickled data
import pickle
from helper_functions import *


# TODO: Fill this in based on where you saved the training and testing data

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


# plt.figure(1, figsize=(2, 2))
# plt.subplot(131), plt.imshow(X_train[0])
# plt.axis('off')
# mod = ndi.rotate(X_train[0], 15.0, reshape=False)
# plt.subplot(132), plt.imshow(mod)
# plt.axis('off')
# zm = ndi.zoom(X_train[0], (0.75, 0.75, 1))
# zm = np.pad(zm, ((4, 4), (4, 4), (0, 0)), 'constant')
# plt.subplot(133), plt.imshow(zm)
# plt.axis('off')
# plt.waitforbuttonpress()

X_train = quick_normalize_img_data(X_train)  # normalize data
X_valid = quick_normalize_img_data(X_valid)
X_test = quick_normalize_img_data(X_test)

# X_train = convert_to_grayscale(X_train)
# X_valid = convert_to_grayscale(X_valid)
# X_test = convert_to_grayscale(X_test)

train_and_test(X_train, y_train, X_valid, y_valid, X_test, y_test)

