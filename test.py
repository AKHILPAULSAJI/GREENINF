import numpy as np
from keras.models import load_model

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model = load_model('model.hdf5')
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
    directory = 'test',
    target_size = (200,150),
    batch_size = 1,
    class_mode = None,

)

test_generator.reset()

pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
print(pred)
predicted_class_indices = np.argmax(pred, axis = 1)
print(predicted_class_indices)
label = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight',
'Potato___healthy','Potato___Late_blight','Tomato__Target_Spot','Tomato__Tomato_mosaic_virus',
'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy',
'Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']

remedies=["1. Use balanced amounts of plant nutrients, especially nitrogen.",
    "Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Prune and remove heavily affected leaves",
	"Use of disease-free seeds that are selected from healthy crop."]


print(label[predicted_class_indices[0]])


print("Remedies:")
print(remedies[predicted_class_indices[0]])

print("\nAccuracy:",max(pred[0]))

"""
file= open('out.txt','w')
file.write(label[predicted_class_indices[0]],"\nRemedies for the disease: \n",remedies[predicted_class_indices[0]])
file.close()"""
file = open("out.txt","w")
file.write(label[predicted_class_indices[0]])
file.close()

file = open("remedies.txt","w")
file.write(remedies[predicted_class_indices[0]])
file.close()